import logging
import os.path
import types
from typing import Any, Optional, Type, Union

from . import _utils
from .requests import RequestID, Request, Result, ToRequestIDType
from . import requests
from . import _job
from . import _workers


PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PATH_ENTRY = os.path.dirname(PKG_ROOT)

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

    Request: Type[_utils.Metadata] = Request
    Result: Type[_utils.Metadata] = Result

    #def __new__(cls, *args, **kwargs):
    #    raise TypeError(f'{cls.__name__} instances not supported')

    def set_request_fs(
            self,
            fs: _utils.FSTree,
            context: Optional[str]
    ) -> None:
        raise NotImplementedError

    def set_work_fs(self, fs: _utils.FSTree, context: Optional[str]) -> None:
        raise NotImplementedError

    def set_result_fs(self, fs: _utils.FSTree, context: Optional[str]) -> None:
        raise NotImplementedError

    def create(
            self,
            reqid: str,
            jobfs: _job.JobFS,
            workerfs: _utils.FSTree,
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
            reqid: Optional[RequestID],
            reqdir: Optional[str],
            reason: str,
            msg: str
    ):
        super().__init__(f'{reason} ({msg} - {reqdir})')
        self.reqid: Optional[RequestID] = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def check_reqdir(reqdir: str, pfiles: _utils.FSTree, cls=RequestDirError) -> RequestID:
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


class JobFS(types.SimpleNamespace):
    """File locations for a single requested job, which apply in all contexts.

    This serves as the base class for context-specific subclasses.
    """

    context: Optional[str] = None

    @classmethod
    def from_jobsfs(cls, jobsfs: "JobsFS", reqid: ToRequestIDType) -> "JobFS":
        requestfs = _utils.FSTree(f'{jobsfs.requests}/{reqid}')
        requestfs.requestsroot = jobsfs.requests.root
        resultfs = _utils.FSTree(f'{jobsfs.results}/{reqid}')
        resultfs.resultsroot = jobsfs.results.root
        workfs = _utils.FSTree(f'{jobsfs.work}/{reqid}')
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
            request: _utils.FSTree,
            result: _utils.FSTree,
            work: _utils.FSTree,
            reqid: Optional[ToRequestIDType] = None
    ):
        request = _utils.FSTree.from_raw(request, name='request')
        work = _utils.FSTree.from_raw(work, name='work')
        result = _utils.FSTree.from_raw(result, name='result')
        if not reqid:
            reqid = os.path.basename(request)
            reqid_obj = RequestID.from_raw(reqid)
            if not reqid_obj:
                raise ValueError('missing reqid')
        else:
            orig = reqid
            reqid_obj = RequestID.from_raw(reqid)
            if not reqid_obj:
                raise ValueError(f'unsupported reqid {orig!r}')

        # the request
        request.metadata = f'{request}/request.json'
        # the job
        work.job_script = f'{work}/run.sh'
        work.pidfile = f'{work}/job.pid'
        work.logfile = f'{work}/job.log'
        # the results
        result.metadata = f'{result}/results.json'

        super().__init__(
            reqid=reqid,
            request=request,
            work=work,
            result=result,
        )

        self._custom_init()

        jobkind = resolve_job_kind(reqid_obj.kind)
        jobkind.set_request_fs(request, self.context)
        jobkind.set_work_fs(work, self.context)
        jobkind.set_result_fs(result, self.context)

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
        # XXX This may not work, since it doesn't do a deep copy of the first
        # three arguments
        return type(self)(
            _utils.FSTree(str(self.request)),
            _utils.FSTree(str(self.work)),
            _utils.FSTree(str(self.result)),
            self.reqid,
        )


class JobsFS(_utils.FSTree):
    """File locations for a set of jobs, which apply in all contexts.

    This serves as the base class for context-specific subclasses.
    """

    context: Optional[str] = None

    JOBFS = JobFS

    @classmethod
    def from_user(cls, user: str) -> "JobsFS":
        return cls(f'/home/{user}/BENCH')

    @classmethod
    def from_raw(
            cls,
            raw: Optional[Union["JobsFS", str]],
            *,
            name: Any=None
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

    def resolve_request(self, reqid: ToRequestIDType) -> JobFS:
        return self.JOBFS.from_jobsfs(self, reqid)

    def copy(self) -> "JobsFS":
        return type(self)(self.root)
