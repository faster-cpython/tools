import logging
from typing import Any, Mapping, Type, TYPE_CHECKING

from . import _utils, _common


if TYPE_CHECKING:
    from . import JobsConfig, requests


logger = logging.getLogger(__name__)


class WorkerConfig(_utils.Config):

    FIELDS = ['user', 'ssh_host', 'ssh_port']

    def __init__(self,
                 user: str,
                 ssh_host: str,
                 ssh_port: int,
                 **ignored
                 ):
        if not user:
            raise ValueError('missing user')
        ssh = _utils.SSHConnectionConfig(user, ssh_host, ssh_port)
        super().__init__(
            user=user,
            ssh=ssh,
        )

    def as_jsonable(self) -> Any:  # type: ignore
        return {
            'user': self.user,
            'ssh_host': self.ssh.host,
            'ssh_port': self.ssh.port,
        }


class WorkerJobFS(_common.JobFS):
    """The file structure of a job's data on a worker."""

    CONTEXT = 'job-worker'


class WorkerJobsFS(_common.JobsFS):
    """The file structure of the jobs data on a worker."""

    JOBFS = WorkerJobFS

    def __init__(self, root=None):
        super().__init__(root)

        # the local git repositories used by the job
        self.repos = _utils.FSTree(f'{self}/repositories')
        self.repos.cpython = f'{self.repos}/cpython'
        self.repos.pyperformance = f'{self.repos}/pyperformance'
        self.repos.pyston_benchmarks = f'{self.repos}/pyston-benchmarks'


class JobWorker:
    """A worker assigned to run a requested job."""

    def __init__(self, worker: "Worker", fs: WorkerJobFS):
        self._worker = worker
        self._fs = fs

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'worker fs'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def fs(self) -> WorkerJobFS:
        return self._fs

    @property
    def topfs(self) -> WorkerJobsFS:
        return self._worker.fs

    @property
    def ssh(self) -> _utils.SSHClient:
        return self._worker.ssh

    @property
    def worker(self) -> "Worker":
        return self._worker

    @property
    def id(self) -> str:
        return self._worker.id


class Worker:
    """A single configured worker."""

    @classmethod
    def from_config(
        cls,
        workerid: str,
        cfg: WorkerConfig,
        JobsFS: Type[WorkerJobsFS] = WorkerJobsFS
    ) -> "Worker":
        fs: WorkerJobsFS = JobsFS.from_user(cfg.user)  # type: ignore
        ssh = _utils.SSHClient.from_config(cfg.ssh)
        return cls(workerid, fs, ssh)

    def __init__(
        self,
        workerid: str,
        fs: WorkerJobsFS,
        ssh: _utils.SSHClient
    ):
        self._id = workerid
        self._fs = fs
        self._ssh = ssh

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'fs ssh'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def id(self) -> str:
        return self._id

    @property
    def fs(self) -> WorkerJobsFS:
        return self._fs

    @property
    def ssh(self) -> _utils.SSHClient:
        return self._ssh

    def resolve_job(
            self,
            reqid: "requests.ToRequestIDType"
    ) -> JobWorker:
        fs = self._fs.resolve_request(reqid)
        return JobWorker(self, fs)  # type: ignore


class Workers:
    """The set of configured workers."""

    @classmethod
    def from_config(
            cls,
            cfg: "JobsConfig",
            JobsFS: Type[WorkerJobsFS] = WorkerJobsFS
    ) -> "Workers":
        workers = {}
        for workerid, worker in cfg.workers.items():
            workers[workerid] = Worker.from_config(workerid, worker, JobsFS)
        return cls(workers)

    def __init__(self, workers: Mapping[str, Worker]):
        self._workers = workers

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'workers'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    def resolve_job(
        self,
        reqid: "requests.ToRequestIDType"
    ) -> JobWorker:
        from . import requests

        reqid_resolved = requests.RequestID.from_raw(reqid)
        workerid = reqid_resolved.workerid
        if workerid not in self._workers:
            raise KeyError(f"Unknown worker {workerid}")
        return self._workers[workerid].resolve_job(reqid_resolved)
