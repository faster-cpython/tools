import logging
import os.path
import types

from . import _utils
from .requests import RequestID, Request, Result


PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PATH_ENTRY = os.path.dirname(PKG_ROOT)

logger = logging.getLogger(__name__)


class JobsError(RuntimeError):
    MSG = 'a jobs-related problem'

    def __init__(self, msg=None):
        super().__init__(msg or self.MSG)


##################################
# request kinds

class JobKind:

    NAME = None

    TYPICAL_DURATION_SECS = None

    REQFS_FIELDS = [
        'job_script',
        'portal_script',
        'ssh_okay',
    ]
    RESFS_FIELDS = [
        'pidfile',
        'logfile',
    ]

    Request = Request
    Result = Result

    #def __new__(cls, *args, **kwargs):
    #    raise TypeError(f'{cls.__name__} instances not supported')

    def set_request_fs(self, fs, context):
        raise NotImplementedError

    def set_work_fs(self, fs, context):
        raise NotImplementedError

    def set_result_fs(self, fs, context):
        raise NotImplementedError

    def create(self, reqid, jobfs, workerfs, **req_kwargs):
        raise NotImplementedError

    def as_row(self, req):
        raise NotImplementedError


def resolve_job_kind(kind):
    if kind == 'compile-bench':
        from ._job_compile_bench import CompileBenchKind
        return CompileBenchKind()
    else:
        raise KeyError(f'unsupported job kind {kind}')


##################################
# job files

class JobFS(types.SimpleNamespace):
    """File locations for a single requested job, which apply in all contexts.

    This serves as the base class for context-specific subclasses.
    """

    context = None

    @classmethod
    def from_jobsfs(cls, jobsfs, reqid):
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

    @classmethod
    def _jobsroot_from_reqdir(cls):
        dirname, reqid = os.path.split(self.request)
        if str(self.reqid) != reqid:
            raise NotImplementedError
        root, requests = os.path.split(dirname)
        if requests != 'REQUESTS':
            raise NotImplementedError
        return root

    @classmethod
    def _get_jobsfs(cls, root):
        return JobsFS(root)

    def __init__(self, request, result, work, reqid=None):
        request = _utils.FSTree.from_raw(request, name='request')
        work = _utils.FSTree.from_raw(work, name='work')
        result = _utils.FSTree.from_raw(result, name='result')
        if not reqid:
            reqid = os.path.basename(request)
            reqid = RequestID.from_raw(reqid)
            if not reqid:
                raise ValueError('missing reqid')
        else:
            orig = reqid
            reqid = RequestID.from_raw(reqid)
            if not reqid:
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

        jobkind = resolve_job_kind(reqid.kind)
        jobkind.set_request_fs(request, self.context)
        jobkind.set_work_fs(work, self.context)
        jobkind.set_result_fs(result, self.context)

    def _custom_init(self):
        pass

    def __str__(self):
        return str(self.request)

    def __fspath__(self):
        return str(self.request)

    @property
    def jobs(self):
        try:
            return self._jobs
        except AttributeError:
            root = self._jobsroot_from_reqdir(self.request)
            self._jobs = self._get_jobsfs(root)
            return self._jobs

    @property
    def requestsroot(self):
        try:
            return self.request.requestsroot
        except AttributeError:
            #return self.jobs.requests.root
            return os.path.dirname(self.request.root)

    @property
    def resultsroot(self):
        try:
            return self.result.resultsroot
        except AttributeError:
            #return self.jobs.results.root
            return os.path.dirname(self.result.root)

    @property
    def job_script(self):
        return self.work.job_script

    @property
    def pidfile(self):
        return self.work.pidfile

    @property
    def logfile(self):
        return self.work.logfile

    def look_up(self, name, subname=None):
        value = getattr(self, name)
        if subname:
            value = getattr(value, subname)
        return value

    def copy(self):
        return type(self)(
            str(self.request),
            str(self.work),
            str(self.result),
            self.reqid,
        )


class JobsFS(_utils.FSTree):
    """File locations for a set of jobs, which apply in all contexts.

    This serves as the base class for context-specific subclasses.
    """

    context = None

    JOBFS = JobFS

    @classmethod
    def from_user(cls, user):
        return cls(f'/home/{user}/BENCH')

    def __init__(self, root='~/BENCH'):
        if not root:
            root = '~/BENCH'
        super().__init__(root)

        self.requests = _utils.FSTree(f'{root}/REQUESTS')
        self.work = _utils.FSTree(self.requests.root)
        self.results = _utils.FSTree(self.requests.root)

    def __str__(self):
        return self.root

    def resolve_request(self, reqid):
        return self.JOBFS.from_jobsfs(self, reqid)

    def copy(self):
        return type(self)(self.root)
