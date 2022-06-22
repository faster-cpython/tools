import json
import logging
import types

from . import _utils, JobError, JobsFS, RequestID


logger = logging.getLogger(__name__)


##################################
# the job queue

class JobQueueError(Exception):
    MSG = 'some problem with the job queue'

    def __init__(self, msg=None):
        super().__init__(msg or self.MSG)


class JobQueuePausedError(JobQueueError):
    MSG = 'job queue paused'


class JobQueueNotPausedError(JobQueueError):
    MSG = 'job queue not paused'


class JobQueueEmptyError(JobQueueError):
    MSG = 'job queue is empty'


class QueuedJobError(JobError, JobQueueError):
    MSG = 'some problem with job {reqid}'


class JobNotQueuedError(QueuedJobError):
    MSG = 'job {reqid} is not in the queue'


class JobAlreadyQueuedError(QueuedJobError):
    MSG = 'job {reqid} is already in the queue'


class JobQueueData(types.SimpleNamespace):

    def __init__(self, jobs, paused):
        super().__init__(
            jobs=jobs,
            paused=paused,
        )

    def __iter__(self):
        yield from self.jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        return self.jobs[idx]

    def __contains__(self, reqid):
        return reqid in self.jobs


class JobQueueSnapshot(JobQueueData):

    def __init__(self, jobs, paused, locked, datafile, lockfile, logfile):
        super().__init__(tuple(jobs), paused)
        self.locked = locked
        self.datafile = datafile
        self.lockfile = lockfile
        self.logfile = logfile

    def read_log(self):
        try:
            logfile = open(self.logfile)
        except FileNotFoundError:
            return
        with logfile:
            yield from _utils.LogSection.read_logfile(logfile)


class JobQueue:

    # XXX Add maxsize.

    @classmethod
    def from_config(cls, cfg):
        fs = JobsFS(self.cfg.data_dir)
        return cls.from_jobsfs(fs)

    @classmethod
    def from_fstree(cls, fs):
        if isinstance(fs, str):
            fs = JobsFS(fs)
        elif not isinstance(fs, JobsFS):
            raise TypeError(f'expected JobsFS, got {fs!r}')
        self = cls(
            datafile=fs.queue.data,
            lockfile=fs.queue.lock,
            logfile=fs.queue.log,
        )
        return self

    def __init__(self, datafile, lockfile, logfile):
        _utils.validate_str(datafile, 'datafile')
        _utils.validate_str(lockfile, 'lockfile')
        _utils.validate_str(logfile, 'logfile')

        self._datafile = datafile
        self._lock = _utils.LockFile(lockfile)
        self._logfile = logfile
        self._data = None

    def __iter__(self):
        with self._lock:
            data = self._load()
        yield from data.jobs

    def __len__(self):
        with self._lock:
            data = self._load()
        return len(data.jobs)

    def __getitem__(self, idx):
        with self._lock:
            data = self._load()
        return data.jobs[idx]

    def _load(self):
        text = _utils.read_file(self._datafile, fail=False)
        data = (json.loads(text) if text else None) or {}
        # Normalize.
        fixed = False
        if 'paused' not in data:
            data['paused'] = False
            fixed = True
        elif data['paused'] not in (True, False):
            data['paused'] = bool(data['paused'])
            fixed = True
        if 'jobs' not in data:
            data['jobs'] = []
            fixed = True
        else:
            jobs = [RequestID.from_raw(v) for v in data['jobs']]
            if any(not j for j in jobs):
                logger.warn('job queue at %s has bad entries', self._datafile)
                fixed = True
            data['jobs'] = [r for r in jobs if r]
        # Save and return the data.
        if fixed:
            with open(self._datafile, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        data = self._data = JobQueueData(**data)
        return data

    def _save(self, data=None):
        if data is None:
            data = self._data
        elif isinstance(data, types.SimpleNamespace):
            if data is not self._data:
                raise NotImplementedError
            data = dict(vars(data))
        self._data = None
        if not data:
            # Nothing to save.
            return
        # Validate.
        if 'paused' not in data or data['paused'] not in (True, False):
            raise NotImplementedError
        if 'jobs' not in data:
            raise NotImplementedError
        elif any(not isinstance(v, RequestID) for v in data['jobs']):
            raise NotImplementedError
        else:
            data['jobs'] = [str(req) for req in data['jobs']]
        # Write to the queue file.
        with open(self._datafile, 'w') as outfile:
            _utils.write_json(data, outfile)

    @property
    def snapshot(self):
        data = self._load()
        try:
            pid = self._lock.read()
        except _utils.OrphanedPIDFileError as exc:
            locked = (exc.pid, False)
        except _utils.InvalidPIDFileError as exc:
            locked = (exc.text, None)
        else:
            locked = (pid, bool(pid))
        return JobQueueSnapshot(
            datafile=self._datafile,
            lockfile=self._lock.filename,
            logfile=self._logfile,
            locked=locked,
            **vars(data),
        )

    @property
    def paused(self):
        with self._lock:
            data = self._load()
            return data.paused

    def pause(self):
        with self._lock:
            data = self._load()
            if data.paused:
                raise JobQueuePausedError()
            data.paused = True
            self._save(data)

    def unpause(self):
        with self._lock:
            data = self._load()
            if not data.paused:
                raise JobQueueNotPausedError()
            data.paused = False
            self._save(data)

    def push(self, reqid):
        with self._lock:
            data = self._load()
            if reqid in data.jobs:
                raise JobAlreadyQueuedError(reqid)

            data.jobs.append(reqid)
            self._save(data)
        return len(data.jobs)

    def pop(self, *, forceifpaused=False):
        with self._lock:
            data = self._load()
            if data.paused:
                if not forceifpaused:
                    raise JobQueuePausedError()
            if not data.jobs:
                raise JobQueueEmptyError()

            reqid = data.jobs.pop(0)
            self._save(data)
        return reqid

    def unpop(self, reqid):
        with self._lock:
            data = self._load()
            if data.jobs and data.jobs[0] == reqid:
                # XXX warn?
                return
            data.jobs.insert(0, reqid)
            self._save(data)

    def move(self, reqid, position, relative=None):
        with self._lock:
            data = self._load()
            if reqid not in data.jobs:
                raise JobNotQueuedError(reqid)

            old = data.jobs.index(reqid)
            if relative == '+':
                idx = min(0, old - position)
            elif relative == '-':
                idx = max(len(data.jobs), old + position)
            else:
                idx = position - 1
            data.jobs.insert(idx, reqid)
            if idx < old:
                old += 1
            del data.jobs[old]

            self._save(data)
        return idx + 1

    def remove(self, reqid):
        with self._lock:
            data = self._load()

            if reqid not in data.jobs:
                raise JobNotQueuedError(reqid)

            data.jobs.remove(reqid)
            self._save(data)

    def read_log(self):
        try:
            logfile = open(self._logfile)
        except FileNotFoundError:
            return
        with logfile:
            yield from _utils.LogSection.read_logfile(logfile)
