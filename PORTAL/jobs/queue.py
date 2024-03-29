import json
import logging
import os.path
import sys
import types
from typing import (
    Iterator, Optional, Sequence, Tuple, Union, TYPE_CHECKING
)

from . import _utils, _common, _job
from .requests import RequestID


if TYPE_CHECKING:
    from . import JobsConfig


logger = logging.getLogger(__name__)


if sys.version_info >= (3, 8):
    from typing import Literal
    LockInfoType = Union[
        Tuple[int, bool],
        Tuple[int, Literal[False]],
        Tuple[str, None]
    ]
else:
    LockInfoType = Union[
        Tuple[int, bool],
        Tuple[None, bool],
        Tuple[str, None]
    ]

##################################
# the job queue


class JobQueueError(Exception):
    MSG = 'some problem with job queue {id!r}'

    def __init__(self, queueid: str, msg: Optional[str] = None):
        msg = (msg or self.MSG).format(id=id)
        super().__init__(msg)
        self.queueid = queueid


class JobQueuePausedError(JobQueueError):
    MSG = 'job queue{id} paused'


class JobQueueNotPausedError(JobQueueError):
    MSG = 'job queue{id} not paused'


class JobQueueEmptyError(JobQueueError):
    MSG = 'job queue{id} is empty'


class QueuedJobError(_job.JobError, JobQueueError):
    MSG = 'some problem with job {reqid} (queue {id})'

    def __init__(
        self,
        queueid: str,
        reqid: RequestID,
        msg: Optional[str] = None
    ):
        _job.JobError.__init__(self, reqid, msg, id=queueid)
        self.queueid = queueid


class JobNotQueuedError(QueuedJobError):
    MSG = 'job {reqid} is not in queue {id}'


class JobAlreadyQueuedError(QueuedJobError):
    MSG = 'job {reqid} is already in queue {id}'


class JobQueueFS(_utils.FSTree):
    """The file structure of the job queue data for a single worker."""

    def __init__(
            self,
            datadir: Union[str, _utils.FSTree]  # _common.JobsFS.requests
    ):
        super().__init__(str(datadir))
        self.data = f'{self.root}/queue.json'
        self.lock = f'{self.root}/queue.lock'
        self.log = f'{self.root}/queue.log'

    def __str__(self):
        return self.data

    def __fspath__(self):
        return self.data


class JobQueuesFS(_utils.FSTree):
    """The file structure of all of the job queues."""
    def resolve_queue(self, queueid: str) -> JobQueueFS:
        return JobQueueFS(os.path.join(self.root, queueid))

    def __str__(self) -> str:
        return self.root

    def __fspath__(self) -> str:
        return self.root


class JobQueueData(types.SimpleNamespace):

    def __init__(self, jobs: Sequence[RequestID], paused: bool):
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
    def __init__(
            self,
            id: str,
            jobs: Sequence[RequestID],
            paused: bool,
            locked: LockInfoType,
            datafile: str,
            lockfile: str,
            logfile: str
    ):
        super().__init__(tuple(jobs), paused)
        self._id = id
        self.locked = locked
        self.datafile = datafile
        self.lockfile = lockfile
        self.logfile = logfile

    def read_log(self) -> Iterator[_utils.LogSection]:
        try:
            logfile = open(self.logfile)
        except FileNotFoundError:
            return
        with logfile:
            yield from _utils.LogSection.read_logfile(logfile)

    @property
    def id(self) -> str:
        return self._id


class JobQueue:

    # XXX Add maxsize.

    @classmethod
    def from_config(cls, cfg: "JobsConfig") -> "JobQueue":
        jobsfs = _common.JobsFS(cfg.data_dir)
        return cls.from_fstree(jobsfs)

    @classmethod
    def from_fstree(
            cls,
            fs: Union[str, JobQueueFS, _common.JobsFS],
            id: Optional[str] = None,
    ) -> "JobQueue":
        queuefs: _utils.FSTree
        if isinstance(fs, str):
            queuefs = JobQueueFS(fs)
        elif isinstance(fs, JobQueueFS):
            queuefs = fs
        elif isinstance(fs, _common.JobsFS):
            queuefs = JobQueueFS(fs.requests)
        else:
            raise TypeError(f'expected JobQueueFS, got {fs!r}')
        self = cls(
            id=id or os.path.basename(queuefs.root),
            datafile=queuefs.data,
            lockfile=queuefs.lock,
            logfile=queuefs.log,
        )
        return self

    def __init__(self, id: str, datafile: str, lockfile: str, logfile: str):
        _utils.validate_str(id, 'id')
        _utils.validate_str(datafile, 'datafile')
        _utils.validate_str(lockfile, 'lockfile')
        _utils.validate_str(logfile, 'logfile')

        self._id = id
        self._datafile = datafile
        self._lock = _utils.LockFile(lockfile)
        self._logfile = logfile
        self._data: Optional[JobQueueData] = None

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

    def _load(self) -> JobQueueData:
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
                logger.warning('job queue at %s has bad entries %s', self._datafile, data['jobs'])
                fixed = True
            data['jobs'] = [r for r in jobs if r]
        # Save and return the data.
        if fixed:
            with open(self._datafile, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        self._data = JobQueueData(**data)
        return self._data

    def _save(self, queuedata: JobQueueData) -> None:
        assert queuedata is self._data, (queuedata, self._data)
        data = dict(vars(queuedata))
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
    def id(self):
        return self._id

    @property
    def snapshot(self) -> JobQueueSnapshot:
        data = self._load()
        locked: LockInfoType
        try:
            pid = self._lock.read()
        except _utils.OrphanedPIDFileError as exc:
            locked = (exc.pid, False)
        except _utils.InvalidPIDFileError as exc:
            locked = (exc.text, None)
        else:
            locked = (pid, bool(pid))
        return JobQueueSnapshot(
            id=self._id,
            datafile=self._datafile,
            lockfile=self._lock.filename,
            logfile=self._logfile,
            locked=locked,
            **vars(data),
        )

    @property
    def paused(self) -> bool:
        with self._lock:
            data = self._load()
            return data.paused

    def pause(self) -> None:
        with self._lock:
            data = self._load()
            if data.paused:
                raise JobQueuePausedError(self._id)
            data.paused = True
            self._save(data)

    def unpause(self) -> None:
        with self._lock:
            data = self._load()
            if not data.paused:
                raise JobQueueNotPausedError(self._id)
            data.paused = False
            self._save(data)

    def push(self, reqid: RequestID) -> int:
        with self._lock:
            data = self._load()
            if reqid in data.jobs:
                raise JobAlreadyQueuedError(self._id, reqid)

            data.jobs.append(reqid)
            self._save(data)
        return len(data.jobs)

    def pop(self, *, forceifpaused: bool = False) -> RequestID:
        with self._lock:
            data = self._load()
            if data.paused:
                if not forceifpaused:
                    raise JobQueuePausedError(self._id)
            if not data.jobs:
                raise JobQueueEmptyError(self._id)

            reqid = data.jobs.pop(0)
            self._save(data)
        return reqid

    def unpop(self, reqid: RequestID) -> None:
        with self._lock:
            data = self._load()
            if data.jobs and data.jobs[0] == reqid:
                # XXX warn?
                return
            data.jobs.insert(0, reqid)
            self._save(data)

    def move(self, reqid: RequestID, position: int, relative: Optional[str]) -> int:
        with self._lock:
            data = self._load()
            if reqid not in data.jobs:
                raise JobNotQueuedError(self._id, reqid)

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

    def remove(self, reqid: RequestID) -> None:
        with self._lock:
            data = self._load()

            if reqid not in data.jobs:
                raise JobNotQueuedError(self._id, reqid)

            data.jobs.remove(reqid)
            self._save(data)

    def read_log(self) -> Iterator[_utils.LogSection]:
        try:
            logfile = open(self._logfile)
        except FileNotFoundError:
            return
        with logfile:
            yield from _utils.LogSection.read_logfile(logfile)


class JobQueues:
    @classmethod
    def from_config(cls, cfg: "JobsConfig", fs: JobQueuesFS):
        return cls(cfg, fs)

    def __init__(self, cfg: "JobsConfig", fs: JobQueuesFS):
        self._cfg = cfg
        self._fs = fs

    def __len__(self):
        return len(self._cfg.workers)

    def __iter__(self) -> Iterator[JobQueue]:
        for queueid in self._cfg.workers.keys():
            yield JobQueue.from_fstree(self._fs.resolve_queue(queueid), queueid)

    def __getitem__(self, queueid: str) -> JobQueue:
        assert queueid in self._cfg.workers, queueid
        return JobQueue.from_fstree(self._fs.resolve_queue(queueid), queueid)
