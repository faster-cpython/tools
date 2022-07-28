import logging
import os
import os.path
import sys

from . import _utils, _pyperformance, _common, _workers, _job, _current
from . import queue as _queue
from .requests import RequestID, Request

# top-level exports
from ._common import JobsError, PKG_ROOT


logger = logging.getLogger(__name__)


class NoRunningJobError(JobsError):
    MSG = 'no job is currently running'


class JobsConfig(_utils.TopConfig):
    """The jobs-related configuration used on the portal host."""

    FIELDS = ['local_user', 'worker', 'data_dir']
    OPTIONAL = ['data_dir']

    FILE = 'jobs.json'
    CONFIG_DIRS = [
        f'{_utils.HOME}/BENCH',
    ]

    def __init__(self,
                 local_user,
                 worker,
                 data_dir=None,
                 **ignored
                 ):
        if not local_user:
            raise ValueError('missing local_user')
        if not worker:
            raise ValueError('missing worker')
        elif not isinstance(worker, _workers.WorkerConfig):
            worker = _workers.WorkerConfig.from_jsonable(worker)
        if data_dir:
            data_dir = os.path.abspath(os.path.expanduser(data_dir))
        else:
            data_dir = f'/home/{local_user}/BENCH'  # This matches DATA_ROOT.
        super().__init__(
            local_user=local_user,
            worker=worker,
            data_dir=data_dir or None,
        )

    @property
    def ssh(self):
        return self.worker.ssh


class PortalFS(_utils.FSTree):

    def __init__(self, root='~/BENCH'):
        super().__init__(root)

        self.jobs = _common.JobsFS(self.root)
        self.jobs.context = 'portal'
        self.jobs.JOBFS = _job.JobFS

        self.jobs.requests.current = _current.symlink_from_jobsfs(self.jobs)

        self.queue = _queue.JobQueueFS(self.jobs.requests)

    @property
    def currentjob(self):
        return self.jobs.requests.current


class Jobs:

    FS = PortalFS

    def __init__(self, cfg, *, devmode=False):
        self._cfg = cfg
        self._devmode = devmode
        self._fs = self.FS(cfg.data_dir)
        self._workers = _workers.Workers.from_config(cfg)
        self._store = _pyperformance.FasterCPythonResults.from_remote()

    def __str__(self):
        return self._fs.root

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def cfg(self):
        return self._cfg

    @property
    def devmode(self):
        return self._devmode

    @property
    def fs(self):
        """Files on the portal host."""
        return self._fs.jobs.copy()

    @property
    def queue(self):
        try:
            return self._queue
        except AttributeError:
            self._queue = _queue.JobQueue.from_fstree(self._fs.queue)
            return self._queue

    def iter_all(self):
        for name in os.listdir(str(self._fs.jobs.requests)):
            reqid = RequestID.parse(name)
            if not reqid:
                continue
            yield self._get(reqid)

    def _get(self, reqid):
        return _job.Job(
            reqid,
            self._fs.jobs.resolve_request(reqid),
            self._workers.resolve_job(reqid),
            self._cfg,
            self._store,
        )

    def get_current(self):
        reqid = _current.get_staged_request(self._fs.jobs)
        if not reqid:
            return None
        return self._get(reqid)

    def get(self, reqid=None):
        if not reqid:
            return self.get_current()
        orig = reqid
        reqid = RequestID.from_raw(orig)
        if not reqid:
            if isinstance(orig, str):
                reqid = self._parse_reqdir(orig)
            if not reqid:
                return None
        return self._get(reqid)

    def _parse_reqdir(self, filename):
        dirname, basename = os.path.split(filename)
        reqid = RequestID.parse(basename)
        if not reqid:
            # It must be a file in the request dir.
            reqid = RequestID.parse(os.path.basename(dirname))
        return reqid

    def match_results(self, specifier, suites=None):
        matched = list(self._store.match(specifier, suites=suites))
        if matched:
            yield from matched
        else:
            yield from self._match_job_results(specifier, suites)

    def _match_job_results(self, specifier, suites):
        if isinstance(specifier, str):
            job = self.get(specifier)
            if job:
                filename = job.fs.result.pyperformance_results
                if suites:
                    # XXX Handle this?
                    pass
                yield _pyperformance.PyperfResultsFile(
                    filename,
                    resultsroot=self._fs.jobs.results.root,
                )

    def create(self, reqid, kind_kwargs=None, pushfsattrs=None, pullfsattrs=None):
        if kind_kwargs is None:
            kind_kwargs = {}
        job = self._get(reqid)
        job._create(kind_kwargs, pushfsattrs, pullfsattrs, self._fs.queue.log)
        return job

    def activate(self, reqid):
        logger.debug('# staging request')
        _current.stage_request(reqid, self._fs.jobs)
        logger.debug('# done staging request')
        job = self._get(reqid)
        job.set_status('activated')
        return job

    def wait_until_job_started(self, job=None, *, timeout=True):
        current = _current.get_staged_request(self._fs.jobs)
        if isinstance(job, _job.Job):
            reqid = job.reqid
        else:
            reqid = job
            if not reqid:
                reqid = current
                if not reqid:
                    raise NoRunningJobError
            job = self._get(reqid)
        if timeout is True:
            # Calculate the timeout.
            if current:
                if reqid == current:
                    timeout = 0
                else:
                    try:
                        jobkind = _common.resolve_job_kind(reqid.kind)
                    except KeyError:
                        raise NotImplementedError(reqid)
                    expected = jobkind.TYPICAL_DURATION_SECS
                    # We could subtract the elapsed time, but it isn't worth it.
                    timeout = expected
            if timeout:
                # Add the expected time for everything in the queue before the job.
                if timeout is True:
                    timeout = 0
                for i, queued in enumerate(self.queue.snapshot):
                    if queued == reqid:
                        # Play it safe by doubling the timeout.
                        timeout *= 2
                        break
                    try:
                        jobkind = _common.resolve_job_kind(queued.kind)
                    except KeyError:
                        raise NotImplementedError(queued)
                    expected = jobkind.TYPICAL_DURATION_SECS
                    timeout += expected
                else:
                    # Either it hasn't been queued or it already finished.
                    timeout = 0
        # Wait!
        pid = job.wait_until_started(timeout)
        return job, pid

    def ensure_next(self):
        logger.debug('Making sure a job is running, if possible')
        # XXX Return (queued job, already running job).
        job = self.get_current()
        if job is not None:
            logger.debug('A job is already running (and will kick off the next one from the queue)')
            # XXX Check the pidfile.
            return
        queue = self.queue.snapshot
        if queue.paused:
            logger.debug('No job is running but the queue is paused')
            return
        if not queue:
            logger.debug('No job is running and none are queued')
            return
        # Run in the background.
        cfgfile = self._cfg.filename
        if not cfgfile:
            raise NotImplementedError
        logger.debug('No job is running so we will run the next one from the queue')
        _utils.run_bg(
            [
                sys.executable, '-u', '-m', 'jobs', '-v',
                'internal-run-next',
                '--config', cfgfile,
                #'--logfile', self._fs.queue.log,
            ],
            logfile=self._fs.queue.log,
            cwd=_common.SYS_PATH_ENTRY,
        )

    def cancel_current(self, reqid=None, *, ifstatus=None):
        job = self.get(reqid)
        if job is None:
            raise NoRunningJobError
        job.cancel(ifstatus=ifstatus)

        logger.info('# unstaging request %s', reqid)
        try:
            _current.unstage_request(job.reqid, self._fs.jobs)
        except _current.RequestNotStagedError:
            pass
        logger.info('# done unstaging request')
        return job

    def finish_successful(self, reqid):
        logger.info('# unstaging request %s', reqid)
        try:
            _current.unstage_request(reqid, self._fs.jobs)
        except _current.RequestNotStagedError:
            pass
        logger.info('# done unstaging request')

        job = self._get(reqid)
        job.close()
        return job


_SORT = {
    'reqid': (lambda j: j.reqid),
}


def sort_jobs(jobs, sortby=None, *, ascending=False):
    if isinstance(jobs, Jobs):
        jobs = list(jobs.iter_all())
    if not sortby:
        sortby = ['reqid']
    elif isinstance(sortby, str):
        sortby = sortby.split(',')
    done = set()
    for kind in sortby:
        if not kind:
            raise NotImplementedError(repr(kind))
        if kind in done:
            raise NotImplementedError(kind)
        done.add(kind)
        try:
            key = _SORT[kind]
        except KeyError:
            raise ValueError(f'unsupported sort kind {kind!r}')
        jobs = sorted(jobs, key=key, reverse=ascending)
    return jobs


def select_job(jobs, criteria=None):
    raise NotImplementedError


def select_jobs(jobs, criteria=None):
    # CSV
    # ranges (i.e. slice)
    if isinstance(jobs, Jobs):
        jobs = list(jobs.iter_all())
    if not criteria:
        yield from jobs
        return
    if isinstance(criteria, str):
        criteria = [criteria]
    else:
        try:
            criteria = list(criteria)
        except TypeError:
            criteria = [criteria]
    if len(criteria) > 1:
        raise NotImplementedError(criteria)
    selection = _utils.get_slice(criteria[0])
    if not isinstance(jobs, (list, tuple)):
        jobs = list(jobs)
    yield from jobs[selection]
