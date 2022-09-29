import logging
import os
import os.path
import sys
from typing import (
    Any, Callable, Iterator, Mapping, MutableMapping, Optional,
    Sequence, Tuple, Type, Union
)

from . import _utils, _pyperformance, _common, _workers, _job, _current
from . import queue as queue_mod
from .requests import RequestID, ToRequestIDType

# top-level exports
from ._common import JobsError


logger = logging.getLogger(__name__)


class NoRunningJobError(JobsError):
    MSG = 'no job is currently running'


class JobsConfig(_utils.TopConfig):
    """The jobs-related configuration used on the portal host."""

    FIELDS = ['local_user', 'workers', 'data_dir', 'results_repo_root']
    OPTIONAL = ['data_dir', 'results_repo_root']

    FILE = 'jobs.json'
    CONFIG_DIRS = [
        f'{_utils.HOME}/BENCH',
    ]

    def __init__(
        self,
        local_user: str,
        workers: Optional[
            MutableMapping[str, Union[Mapping[str, Any], _workers.WorkerConfig]]
        ],
        data_dir: Optional[str] = None,
        results_repo_root: Optional[str] = None,
        **ignored
    ):
        if not local_user:
            raise ValueError('missing local_user')
        if not workers:
            raise ValueError('missing workers')

        for worker_name, worker in workers.items():
            if not isinstance(worker, _workers.WorkerConfig):
                workers[worker_name] = _workers.WorkerConfig.from_jsonable(worker)

        if not data_dir:
            data_dir = f'~{local_user}/BENCH'  # This matches DATA_ROOT.
        data_dir = os.path.abspath(os.path.expanduser(data_dir))

        super().__init__(
            local_user=local_user,
            workers=workers,
            data_dir=data_dir or None,
            results_repo_root=results_repo_root
        )

    @property
    def queues(self):
        try:
            return self._queues
        except AttributeError:
            self._queues = {w: {} for w in self.workers}
            return self._queues


class PortalJobsFS(_common.JobsFS):
    JOBFS = _job.JobFS

    def __init__(self, root: str):
        super().__init__(root)
        self.requests.current = _current.symlink_from_jobsfs(self)
        #self.current = _current.symlink_from_jobsfs(self)


class PortalFS(_utils.FSTree):

    def __init__(self, root: Optional[str] = None):
        if root is None:
            root = "~/BENCH"

        super().__init__(root)

        self.jobs = PortalJobsFS(self.root)

    @property
    def currentjob(self) -> str:
        return self.jobs.requests.current

    @property
    def queues(self) -> queue_mod.JobQueuesFS:
        return queue_mod.JobQueuesFS(self.jobs.queues)


class Jobs:

    FS: Type[PortalFS] = PortalFS
    _queues: queue_mod.JobQueues

    def __init__(self, cfg: JobsConfig, *, devmode: bool = False):
        self._cfg = cfg
        self._devmode = devmode
        self._fs = self.FS(cfg.data_dir)
        self._workers = _workers.Workers.from_config(cfg)
        self._store = _pyperformance.FasterCPythonResults.from_remote(
            root=getattr(cfg, "results_repo_root", None)
        )

    def __str__(self):
        return self._fs.root

    def __eq__(self, _other):
        raise NotImplementedError

    @property
    def cfg(self) -> JobsConfig:
        return self._cfg

    @property
    def devmode(self) -> bool:
        return self._devmode

    @property
    def fs(self) -> _utils.FSTree:
        """Files on the portal host."""
        return self._fs.jobs.copy()

    @property
    def queues(self) -> queue_mod.JobQueues:
        try:
            return self._queues
        except AttributeError:
            self._queues = queue_mod.JobQueues.from_config(
                self._cfg, self._fs.queues
            )
            return self._queues

    def iter_all(self) -> Iterator[_job.Job]:
        for name in os.listdir(str(self._fs.jobs.requests)):
            reqid = RequestID.parse(name)
            if not reqid:
                continue
            yield self._get(reqid)

    def _get(self, reqid: RequestID) -> _job.Job:
        return _job.Job(
            reqid,
            self._fs.jobs.resolve_request(reqid),
            self._workers.resolve_job(reqid),
            self._cfg,
            self._store,
        )

    def get_current(self) -> Optional[_job.Job]:
        reqid = _current.get_staged_request(self._fs.jobs)
        if not reqid:
            return None
        return self._get(reqid)

    def get(self, reqid: Optional[ToRequestIDType] = None) -> Optional[_job.Job]:
        if not reqid:
            return self.get_current()
        reqid_resolved = RequestID.from_raw(reqid)
        if not reqid_resolved:
            if isinstance(reqid, str):
                reqid_resolved = self._parse_reqdir(reqid)
            if not reqid_resolved:
                return None
        return self._get(reqid_resolved)

    def _parse_reqdir(self, filename: str) -> RequestID:
        dirname, basename = os.path.split(filename)
        reqid = RequestID.parse(basename)
        if not reqid:
            # It must be a file in the request dir.
            reqid = RequestID.parse(os.path.basename(dirname))
        return reqid

    def match_results(
            self,
            specifier: Any,
            suites: _pyperformance.SuitesType = None
    ) -> Iterator[_pyperformance.PyperfResultsFile]:
        matched = list(self._store.match(specifier, suites=suites))
        if matched:
            yield from matched
        else:
            yield from self._match_job_results(specifier, suites)

    def _match_job_results(
            self,
            specifier: Union[str, RequestID],
            suites: Optional[Any]
    ) -> Iterator[_pyperformance.PyperfResultsFile]:
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

    def create(
            self,
            reqid: RequestID,
            kind_kwargs: Optional[Mapping[str, Any]] = None,
            pushfsattrs: Optional[_common.FsAttrsType] = None,
            pullfsattrs: Optional[_common.FsAttrsType] = None
    ) -> _job.Job:
        if kind_kwargs is None:
            kind_kwargs = {}
        job = self._get(reqid)
        job._create(
            kind_kwargs,
            pushfsattrs,
            pullfsattrs,
            self._fs.queues.resolve_queue(reqid.workerid).log
        )
        return job

    def activate(self, reqid: RequestID) -> _job.Job:
        logger.debug('# staging request')
        _current.stage_request(reqid, self._fs.jobs)
        logger.debug('# done staging request')
        job = self._get(reqid)
        job.set_status('activated')
        return job

    def wait_until_job_started(
            self,
            job: Optional[Union[_job.Job, RequestID]] = None,
            *,
            timeout: Union[int, bool] = True
    ) -> Tuple[_job.Job, Optional[int]]:
        current = _current.get_staged_request(self._fs.jobs)
        if isinstance(job, _job.Job):
            reqid = job.reqid
        else:
            reqid_source = job or current
            if not reqid_source:
                raise NoRunningJobError
            job = self._get(reqid_source)
            reqid = job.reqid
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
                    expected = jobkind.TYPICAL_DURATION_SECS or 0
                    # We could subtract the elapsed time, but it isn't worth it.
                    timeout = expected
            if timeout:
                if timeout is True:
                    timeout = 0
                # Add the expected time for everything in the queue before the job.
                for queued in self.queues[reqid.workerid].snapshot:
                    if queued == reqid:
                        # Play it safe by doubling the timeout.
                        timeout *= 2
                        break
                    try:
                        jobkind = _common.resolve_job_kind(queued.kind)
                    except KeyError:
                        raise NotImplementedError(queued)
                    expected = jobkind.TYPICAL_DURATION_SECS or 0
                    timeout += expected
                else:
                    # Either it hasn't been queued or it already finished.
                    timeout = 0
        # Wait!
        pid = job.wait_until_started(timeout)
        return job, pid

    def ensure_next(self) -> None:
        logger.debug('Making sure a job is running, if possible')
        # XXX Return (queued job, already running job).
        job = self.get_current()
        if job is not None:
            logger.debug('A job is already running (and will kick off the next one from the queue)')
            # XXX Check the pidfile.
            return
        for _queue in self.queues:
            queue = _queue.snapshot
            if queue.paused:
                logger.debug('No job is running but the queue is paused')
                continue
            if not queue:
                logger.debug('No job is running and none are queued')
                continue
            # Run in the background.
            cfgfile = self._cfg.filename
            if not cfgfile:
                raise NotImplementedError
            logger.debug('No job is running so we will run the next one from the queue')
            _utils.run_bg(
                [
                    sys.executable, '-u', '-m', 'jobs', '-v',
                    'internal-run-next',
                    queue.id,
                    '--config', cfgfile,
                    #'--logfile', self._fs.queue.log,
                ],
                logfile=self._fs.queues.resolve_queue(queue.id).log,
                cwd=_common.SYS_PATH_ENTRY,
            )

    def cancel_current(
            self,
            reqid: Optional[RequestID] = None,
            *,
            ifstatus: Optional[str] = None
    ) -> _job.Job:
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

    def finish_successful(self, reqid: RequestID) -> _job.Job:
        logger.info('# unstaging request %s', reqid)
        try:
            _current.unstage_request(reqid, self._fs.jobs)
        except _current.RequestNotStagedError:
            pass
        logger.info('# done unstaging request')

        job = self._get(reqid)
        job.close()
        return job


_SORT: Mapping[str, Callable[[_job.Job], RequestID]] = {
    'reqid': (lambda j: j.reqid),
}


ToJobsType = Union[Jobs, Sequence[_job.Job]]


def sort_jobs(
        jobs: ToJobsType,
        sortby: Optional[Union[str, Sequence[str]]] = None,
        # sortby must be one of the keys in _SORT
        *,
        ascending: bool = False
) -> Sequence[_job.Job]:
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


def select_job(
        jobs: ToJobsType,
        criteria: Optional[Union[str, Sequence[str]]] = None
) -> Iterator[_job.Job]:
    raise NotImplementedError


def select_jobs(
        jobs: ToJobsType,
        criteria: Optional[Union[str, Sequence[str]]] = None
) -> Iterator[_job.Job]:
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
        criteria = list(criteria)
    if len(criteria) > 1:
        raise NotImplementedError(criteria)
    selection = _utils.get_slice(criteria[0])
    if not isinstance(jobs, (list, tuple)):
        jobs = list(jobs)
    yield from jobs[selection]
