import logging
import os
import os.path
import sys

from . import _utils, _pyperformance, _common, _workers, _job
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


class JobsFS(_common.JobsFS):
    """The file structure of the jobs data."""

    context = 'portal'

    JOBFS = _job.JobFS

    def __init__(self, root='~/BENCH'):
        super().__init__(root)

        self.requests.current = f'{self.requests}/CURRENT'

        self.queue = _utils.FSTree(f'{self.requests}/queue.json')
        self.queue.data = f'{self.requests}/queue.json'
        self.queue.lock = f'{self.requests}/queue.lock'
        self.queue.log = f'{self.requests}/queue.log'


class RequestDirError(Exception):
    def __init__(self, reqid, reqdir, reason, msg):
        super().__init__(f'{reason} ({msg} - {reqdir})')
        self.reqid = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def _check_reqdir(reqdir, pfiles, cls=RequestDirError):
    requests, reqidstr = os.path.split(reqdir)
    if requests != pfiles.requests.root:
        raise cls(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    reqid = RequestID.parse(reqidstr)
    if not reqid:
        raise cls(None, reqdir, 'invalid', f'{reqidstr!r} not a request ID')
    if not os.path.exists(reqdir):
        raise cls(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        raise cls(reqid, reqdir, 'malformed', 'target is not a directory')
    return reqid


##################################
# jobs

class Jobs:

    FS = JobsFS

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
        return self._fs.copy()

    @property
    def queue(self):
        try:
            return self._queue
        except AttributeError:
            self._queue = _queue.JobQueue.from_fstree(self._fs)
            return self._queue

    def iter_all(self):
        for name in os.listdir(str(self._fs.requests)):
            reqid = RequestID.parse(name)
            if not reqid:
                continue
            yield self._get(reqid)

    def _get(self, reqid):
        return _job.Job(
            reqid,
            self._fs.resolve_request(reqid),
            self._workers.resolve_job(reqid),
            self._cfg,
            self._store,
        )

    def get_current(self):
        reqid = _get_staged_request(self._fs)
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
                    resultsroot=self._fs.results.root,
                )

    def create(self, reqid, kind_kwargs=None, pushfsattrs=None, pullfsattrs=None):
        if kind_kwargs is None:
            kind_kwargs = {}
        job = self._get(reqid)
        job._create(kind_kwargs, pushfsattrs, pullfsattrs, self._fs.queue.log)
        return job

    def activate(self, reqid):
        logger.debug('# staging request')
        _stage_request(reqid, self._fs)
        logger.debug('# done staging request')
        job = self._get(reqid)
        job.set_status('activated')
        return job

    def wait_until_job_started(self, job=None, *, timeout=True):
        current = _get_staged_request(self._fs)
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
            _unstage_request(job.reqid, self._fs)
        except RequestNotStagedError:
            pass
        logger.info('# done unstaging request')
        return job

    def finish_successful(self, reqid):
        logger.info('# unstaging request %s', reqid)
        try:
            _unstage_request(reqid, self._fs)
        except RequestNotStagedError:
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


##################################
# the current job

class StagedRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class StagedRequestDirError(StagedRequestError, RequestDirError):

    def __init__(self, reqid, reqdir, reason, msg):
        RequestDirError.__init__(self, reqid, reqdir, reason, msg)


class StagedRequestStatusError(StagedRequestError):

    reason = None

    def __init__(self, reqid, status):
        assert self.reason
        super().__init__(reqid, self.reason)
        self.status = status


class OutdatedStagedRequestError(StagedRequestStatusError):
    pass


class StagedRequestNotRunningError(OutdatedStagedRequestError):
    reason = 'is no longer running'


class StagedRequestAlreadyFinishedError(OutdatedStagedRequestError):
    reason = 'is still "current" even though it finished'


class StagedRequestUnexpectedStatusError(StagedRequestStatusError):
    reason = 'was set as the current job incorrectly'


class StagedRequestInvalidMetadataError(StagedRequestStatusError):
    reason = 'has invalid metadata'


class StagedRequestMissingStatusError(StagedRequestInvalidMetadataError):
    reason = 'is missing status metadata'


class StagingRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class RequestNotPendingError(StagingRequestError):

    def __init__(self, reqid, status=None):
        super().__init__(reqid, f'could not stage {reqid} (expected pending, got {status or "???"} status)')
        self.status = status


class RequestAlreadyStagedError(StagingRequestError):

    def __init__(self, reqid, curid):
        super().__init__(reqid, f'could not stage {reqid} ({curid} already staged)')
        self.curid = curid


class UnstagingRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class RequestNotStagedError(UnstagingRequestError):

    def __init__(self, reqid, curid=None):
        msg = f'{reqid} is not currently staged'
        if curid:
            msg = f'{msg} ({curid} is)'
        super().__init__(reqid, msg)
        self.curid = curid


def _read_staged(pfiles):
    link = pfiles.requests.current
    try:
        reqdir = os.readlink(link)
    except FileNotFoundError:
        return None
    except OSError:
        if os.path.islink(link):
            raise  # re-raise
        exc = RequestDirError(None, link, 'malformed', 'target is not a link')
        try:
            exc.reqid = _check_reqdir(link, pfiles, StagedRequestDirError)
        except StagedRequestDirError:
            raise exc
        raise exc
    else:
        return _check_reqdir(reqdir, pfiles, StagedRequestDirError)


def _check_staged_request(reqid, pfiles):
    # Check the request status.
    reqfs = pfiles.resolve_request(reqid)
    try:
        status = Result.read_status(str(reqfs.result.metadata))
    except _utils.MissingMetadataError:
        raise StagedRequestMissingStatusError(reqid, None)
    except _utils.InvalidMetadataError:
        raise StagedRequestInvalidMetadataError(reqid, None)
    else:
        if status in Result.ACTIVE and status != 'pending':
            if not _utils.PIDFile(str(reqfs.pidfile)).read(orphaned='ignore'):
                raise StagedRequestNotRunningError(reqid, status)
        elif status in Result.FINISHED:
            raise StagedRequestAlreadyFinishedError(reqid, status)
        else:  # created, pending
            raise StagedRequestUnexpectedStatusError(reqid, status)
    # XXX Do other checks?


def _set_staged(reqid, reqdir, pfiles):
    try:
        os.symlink(reqdir, pfiles.requests.current)
    except FileExistsError:
        try:
            curid = _read_staged(pfiles)
        except StagedRequestError as exc:
            # One was already set but the link is invalid.
            _clear_staged(pfiles, exc)
        else:
            if curid == reqid:
                # XXX Fail?
                logger.warn(f'{reqid} is already set as the current job')
                return
            elif curid:
                # Clear it if it is no longer valid.
                try:
                    _check_staged_request(curid, pfiles)
                except StagedRequestError as exc:
                    _clear_staged(pfiles, exc)
                else:
                    raise RequestAlreadyStagedError(reqid, curid)
        logger.info('trying again')
        # XXX Guard against infinite recursion?
        return _set_staged(reqid, reqdir, pfiles)


def _clear_staged(pfiles, exc=None):
    if exc is not None:
        if isinstance(exc, StagedRequestInvalidMetadataError):
            log = logger.error
        else:
            log = logger.warn
        reqid = getattr(exc, 'reqid', None)
        reason = getattr(exc, 'reason', None) or 'broken'
        if not reason.startswith(('is', 'was', 'has')):
            reason = f'is {reason}'
        log(f'request {reqid or "???"} {reason} ({exc}); unsetting as the current job...')
    os.unlink(pfiles.requests.current)


# These are the higher-level helpers:

def _get_staged_request(pfiles):
    try:
        curid = _read_staged(pfiles)
    except StagedRequestError as exc:
        _clear_staged(pfiles, exc)
        return None
    if curid:
        try:
            _check_staged_request(curid, pfiles)
        except StagedRequestError as exc:
            _clear_staged(pfiles, exc)
            curid = None
    return curid


def _stage_request(reqid, pfiles):
    jobfs = pfiles.resolve_request(reqid)
    status = Result.read_status(jobfs.result.metadata, fail=False)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
    _set_staged(reqid, jobfs.request, pfiles)


def _unstage_request(reqid, pfiles):
    reqid = RequestID.from_raw(reqid)
    try:
        curid = _read_staged(pfiles)
    except StagedRequestError as exc:
        # One was already set but the link is invalid.
        _clear_staged(pfiles, exc)
        raise RequestNotStagedError(reqid)
    else:
        if curid == reqid:
            # It's a match!
            _clear_staged(pfiles)
        else:
            if curid:
                # Clear it if it is no longer valid.
                try:
                    _check_staged_request(curid, pfiles)
                except StagedRequestError as exc:
                    _clear_staged(pfiles, exc)
            raise RequestNotStagedError(reqid)


##################################
# avoid circular imports

from . import queue as _queue
