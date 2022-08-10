import logging

from . import _utils, _common


logger = logging.getLogger(__name__)


class WorkerConfig(_utils.Config):

    FIELDS = ['user', 'ssh_host', 'ssh_port']

    def __init__(self,
                 user,
                 ssh_host,
                 ssh_port,
                 **ignored
                 ):
        if not user:
            raise ValueError('missing user')
        ssh = _utils.SSHConnectionConfig(user, ssh_host, ssh_port)
        super().__init__(
            user=user,
            ssh=ssh,
        )

    def as_jsonable(self):
        return {
            'user': self.user,
            'ssh_host': self.ssh.host,
            'ssh_port': self.ssh.port,
        }


class WorkerJobFS(_common.JobFS):
    """The file structure of a job's data on a worker."""

    context = 'job-worker'


class WorkerJobsFS(_common.JobsFS):
    """The file structure of the jobs data on a worker."""

    context = 'job-worker'

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

    def __init__(self, worker, fs):
        self._worker = worker
        self._fs = fs

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'worker fs'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def fs(self):
        return self._fs

    @property
    def topfs(self):
        return self._worker.fs

    @property
    def ssh(self):
        return self._worker.ssh


class Worker:
    """A single configured worker."""

    @classmethod
    def from_config(cls, name, cfg, JobsFS=WorkerJobsFS):
        fs = JobsFS.from_user(cfg.user)
        ssh = _utils.SSHClient.from_config(cfg.ssh)
        return cls(name, fs, ssh)

    def __init__(self, name, fs, ssh):
        self._name = name
        self._fs = fs
        self._ssh = ssh

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'fs ssh'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def fs(self):
        return self._fs

    @property
    def ssh(self):
        return self._ssh

    def resolve_job(self, reqid):
        fs = self._fs.resolve_request(reqid)
        return JobWorker(self, fs)


class Workers:
    """The set of configured workers."""

    @classmethod
    def from_config(cls, cfg, JobsFS=WorkerJobsFS):
        workers = {}
        for worker_name, worker in cfg.items():
            workers[worker_name] = Worker.from_config(worker_name, worker, JobsFS)
        return cls(workers)

    def __init__(self, workers):
        self._workers = workers

    def __eq__(self, other):
        raise NotImplementedError

    def __getitem__(self, worker_name):
        if worker_name not in self._workers:
            raise ValueError(
                f"Unknown worker '{worker_name}', "
                f"must be one of {self._workers.keys()}"
            )
        return self._workers[worker_name]
