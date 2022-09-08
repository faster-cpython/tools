import logging
import os.path
import types
from typing import Any, List, Optional, Tuple, Type, Union, TYPE_CHECKING

from . import _utils
from . import requests


if TYPE_CHECKING:
    from . import _job
    from . import _workers


PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PATH_ENTRY = os.path.dirname(PKG_ROOT)


FsAttrsType = List[Union[Tuple[str, str], str]]


logger = logging.getLogger(__name__)


class JobsError(RuntimeError):
    MSG = 'a jobs-related problem'

    def __init__(self, msg: Optional[str] = None):
        super().__init__(msg or self.MSG)


##################################
# request kinds

class JobKind:

    NAME: Optional[str] = None

    TYPICAL_DURATION_SECS: Optional[int] = None

    REQFS_FIELDS = [
        'job_script',
        'portal_script',  # from _job.JobFS
        'ssh_okay',  # from _job.JobFS
    ]
    RESFS_FIELDS = [
        'pidfile',
        'logfile',
    ]

    Request: Type[_utils.Metadata] = requests.Request
    Result: Type[_utils.Metadata] = requests.Result

    #def __new__(cls, *args, **kwargs):
    #    raise TypeError(f'{cls.__name__} instances not supported')

    def set_request_fs(
            self,
            fs: "JobRequestFS",
            context: str
    ) -> None:
        raise NotImplementedError

    def set_work_fs(
            self,
            fs: "JobWorkFS",
            context: Optional[str]
    ) -> None:
        raise NotImplementedError

    def set_result_fs(
            self,
            fs: "JobResultFS",
            context: Optional[str]
    ) -> None:
        raise NotImplementedError

    def create(
            self,
            reqid: requests.ToRequestIDType,
            jobfs: "_job.JobFS",
            workerfs: "_workers.WorkerJobsFS",
            **req_kwargs
    ):
        raise NotImplementedError

    def as_row(self, req: requests.Request):
        raise NotImplementedError


def resolve_job_kind(kind: str) -> JobKind:
    if kind == 'compile-bench':
        from ._job_compile_bench import CompileBenchKind
        return CompileBenchKind()
    else:
        raise KeyError(f'unsupported job kind {kind}')


##################################
# job files

class RequestDirError(Exception):
    def __init__(
            self,
            reqid: Optional[requests.RequestID],
            reqdir: Optional[str],
            reason: str,
            msg: str
    ):
        super().__init__(f'{reason} ({msg} - {reqdir})')
        self.reqid: Optional[requests.RequestID] = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def check_reqdir(
        reqdir: str,
        pfiles: "JobsFS",
        cls=RequestDirError
) -> requests.RequestID:
    requests_str, reqid_str = os.path.split(reqdir)
    if requests_str != pfiles.requests.root:
        raise cls(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    reqid = requests.RequestID.parse(reqid_str)
    if not reqid:
        raise cls(None, reqdir, 'invalid', f'{reqid_str!r} not a request ID')
    if not os.path.exists(reqdir):
        raise cls(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        raise cls(reqid, reqdir, 'malformed', 'target is not a directory')
    return reqid


class JobRequestFS(_utils.FSTree):
    def __init__(self, root: str):
        super().__init__(root)
        self.requestroot = os.path.dirname(root)
        self.metadata = f'{self.root}/request.json'


class JobResultFS(_utils.FSTree):
    def __init__(self, root: str):
        super().__init__(root)
        self.resultroot = os.path.dirname(root)
        self.metadata = f'{self.root}/results.json'


class JobWorkFS(_utils.FSTree):
    def __init__(self, root: str):
        super().__init__(root)
        self.job_script = f'{self.root}/run.sh'
        self.pidfile = f'{self.root}/job.pid'
        self.logfile = f'{self.root}/job.log'


class JobFS(types.SimpleNamespace):
    """File locations for a single requested job, which apply in all contexts.

    This serves as the base class for context-specific subclasses.
    """

    context: Optional[str] = None  # required in subclasses

    @classmethod
    def from_jobsfs(
            cls,
            jobsfs: "JobsFS",
            reqid: requests.ToRequestIDType
    ) -> "JobFS":
        requestfs = JobRequestFS.from_raw(f"{jobsfs.requests}/{reqid}")
        resultfs = JobResultFS.from_raw(f"{jobsfs.results}/{reqid}")
        workfs = JobWorkFS.from_raw(f"{jobsfs.work}/{reqid}")
        self = cls(
            request=requestfs,
            result=resultfs,
            work=workfs,
            reqid=reqid,
        )
        self._jobs = jobsfs
        return self

    def __init__(
            self,
            request: Union[str, JobRequestFS],
            result: Union[str, JobResultFS],
            work: Union[str, JobWorkFS],
            reqid: Optional[requests.ToRequestIDType] = None
    ):
        request_fs = JobRequestFS.from_raw(request)
        result_fs = JobResultFS.from_raw(result)
        work_fs = JobWorkFS.from_raw(work)
        
        if not reqid:
            reqid = os.path.basename(request)
            reqid_obj = requests.RequestID.from_raw(reqid)
            if not reqid_obj:
                raise ValueError('missing reqid')
        else:
            orig = reqid
            reqid_obj = requests.RequestID.from_raw(reqid)
            if not reqid_obj:
                raise ValueError(f'unsupported reqid {orig!r}')

        super().__init__(
            reqid=reqid,
            request=request_fs,
            work=work_fs,
            result=result_fs,
        )

        self._custom_init()

        if self.context is None:
            raise ValueError(f"No context set on {self}")

        jobkind = resolve_job_kind(reqid_obj.kind)
        jobkind.set_request_fs(request_fs, self.context)
        jobkind.set_work_fs(work_fs, self.context)
        jobkind.set_result_fs(result_fs, self.context)

    def _custom_init(self) -> None:
        pass

    def __str__(self):
        return str(self.request)

    def __fspath__(self):
        return str(self.request)

    @property
    def requestsroot(self) -> str:
        try:
            return self.request.requestsroot
        except AttributeError:
            #return self.jobs.requests.root
            return os.path.dirname(self.request.root)

    @property
    def resultsroot(self) -> str:
        try:
            return self.result.resultsroot
        except AttributeError:
            #return self.jobs.results.root
            return os.path.dirname(self.result.root)

    @property
    def job_script(self) -> str:
        return self.work.job_script

    @property
    def pidfile(self) -> str:
        return self.work.pidfile

    @property
    def logfile(self) -> str:
        return self.work.logfile

    def look_up(self, name: str, subname: Optional[str] = None) -> Any:
        value = getattr(self, name)
        if subname:
            value = getattr(value, subname)
        return value

    def copy(self) -> "JobFS":
        return type(self)(
            str(self.request),
            str(self.result),
            str(self.work),
            self.reqid,
        )


class JobsFS(_utils.FSTree):
    """File locations for a set of jobs, which apply in all contexts.

    This serves as the base class for context-specific subclasses.
    """

    context: Optional[str] = None  # required in subclasses

    JOBFS = JobFS

    @classmethod
    def from_user(cls, user: str) -> "JobsFS":
        return cls(f'/home/{user}/BENCH')

    @classmethod
    def from_raw(
            cls,
            raw: Optional[Union["JobsFS", str]],
            *,
            name: Any = None
    ) -> "JobsFS":
        if not raw:
            raise ValueError('missing jobsfs')
        elif isinstance(raw, JobsFS):
            return raw
        elif isinstance(raw, str):
            if os.path.basename(raw) == 'REQUESTS':
                raw = os.path.dirname(raw)
            return cls(raw)
        else:
            raise TypeError(raw)

    def __init__(self, root: Optional[str] = None):
        if not root:
            root = '~/BENCH'
        super().__init__(root)

        self.requests = _utils.FSTree(f'{root}/REQUESTS')
        self.work = _utils.FSTree(self.requests.root)
        self.results = _utils.FSTree(self.requests.root)

    def __str__(self):
        return self.root

    def resolve_request(self, reqid: requests.ToRequestIDType) -> JobFS:
        return self.JOBFS.from_jobsfs(self, reqid)

    def copy(self) -> "JobsFS":
        return type(self)(self.root)
