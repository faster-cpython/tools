from collections import namedtuple
import configparser
import datetime
import json
import os
import os.path
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import types

'''
sudo adduser --gecos '' --disabled-password <username>
sudo --login --user <username> ssh-import-id gh:<username>
'''


USER = os.environ.get('USER', '').strip()
SUDO_USER = os.environ.get('SUDO_USER', '').strip()

HOME = os.path.expanduser('~')


##################################
# requests

class RequestID(namedtuple('RequestID', 'kind timestamp user')):

    KIND = types.SimpleNamespace(
        BENCHMARKS='compile-bench',
    )
    _KIND_BY_VALUE = {v: v for _, v in vars(KIND).items()}

    @classmethod
    def from_raw(cls, raw):
        if isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        else:
            raise NotImplementedError(raw)

    @classmethod
    def parse(cls, idstr):
        kinds = '|'.join(cls._KIND_BY_VALUE)
        m = re.match(rf'^req-(?:({kinds})-)?(\d{{10}})-(\w+)$', idstr)
        if not m:
            return None
        kind, timestamp, user = m.groups()
        return cls(kind, int(timestamp), user)

    @classmethod
    def generate(cls, cfg, user=None, kind=KIND.BENCHMARKS):
        user = _resolve_user(cfg, user)
        timestamp = int(_utcnow())
        return cls(kind, timestamp, user)

    def __new__(cls, kind, timestamp, user):
        if not kind:
            kind = cls.KIND.BENCHMARKS
        else:
            try:
                kind = cls._KIND_BY_VALUE[kind]
            except KeyError:
                raise ValueError(f'unsupported kind {kind!r}')

        if not timestamp:
            raise ValueError('missing timestamp')
        elif isinstance(timestamp, str):
            timestamp, _, _ = timestamp.partition('.')
            timestamp = int(timestamp)
        elif not isinstance(timestamp, int):
            try:
                timestamp = int(timestamp)
            except TypeError:
                raise TypeError(f'expected int timestamp, got {timestamp!r}')

        if not user:
            raise ValueError('missing user')
        elif not isinstance(user, str):
            raise TypeError(f'expected str for user, got {user!r}')
        else:
            _check_name(user)

        self = super().__new__(
            cls,
            kind=kind,
            timestamp=timestamp,
            user=user,
        )
        return self

    def __str__(self):
        return f'req-{self.kind}-{self.timestamp}-{self.user}'

    @property
    def date(self):
        return datetime.datetime.fromtimestamp(
            self.timestamp,
            datetime.timezone.utc,
        )


class Metadata(types.SimpleNamespace):

    FIELDS = None
    OPTIONAL = None

    _extra = None

    @classmethod
    def load(cls, resfile):
        if isinstance(resfile, str):
            filename = resfile
            with open(filename) as resfile:
                return cls.load(resfile)
        data = json.load(resfile)
        return cls.from_jsonable(data)

    @classmethod
    def from_jsonable(cls, data):
        kwargs = {}
        extra = {}
        unused = set(cls.FIELDS or ())
        for field in data:
            if field in unused:
                kwargs[field] = data[field]
                unused.remove(field)
            elif not field.startswith('_'):
                extra[field] = data[field]
        unused -= set(cls.OPTIONAL or ())
        if unused:
            missing = ', '.join(sorted(unused))
            raise ValueError(f'missing required data (fields: {missing})')
        if kwargs:
            self = cls(**kwargs)
            if extra:
                self._extra = extra
        else:
            self = cls(**extra)
        return self

    def refresh(self, resfile):
        """Reload from the given file."""
        fresh = self.load(resfile)
        # This isn't the best way to go, but it's simple.
        vars(self).clear()
        self.__init__(**vars(fresh))
        return fresh

    def as_jsonable(self):
        fields = self.FIELDS
        if not fields:
            fields = (f for f in vars(self) if not f.startswith('_'))
        optional = set(self.OPTIONAL or ())
        data = {}
        for field in fields:
            try:
                value = getattr(self, field)
            except AttributeError:
                # XXX Fail?  Warn?  Add a default?
                continue
            if hasattr(value, 'as_jsonable'):
                value = value.as_jsonable()
            data[field] = value
        return data

    def save(self, resfile):
        if isinstance(resfile, str):
            filename = resfile
            with open(filename, 'w') as resfile:
                return self.save(resfile)
        data = self.as_jsonable()
        json.dump(data, resfile, indent=4)
        print(file=resfile)


class Request(Metadata):

    FIELDS = [
        'kind',
        'id',
        'datadir',
        'user',
        'date',
    ]

    def __init__(self, id, datadir, *,
                 # These are ignored (duplicated by id):
                 kind=None, user=None, date=None,
                 ):
        if not id:
            raise ValueError('missing id')
        id = RequestID.from_raw(id)

        if not datadir:
            raise ValueError('missing datadir')
        if not isinstance(datadir, str):
            raise TypeError(f'expected dirname for datadir, got {datadir!r}')

        super().__init__(
            id=id,
            datadir=datadir,
        )

    def __str__(self):
        return self.id

    @property
    def reqid(self):
        return self.id

    @property
    def reqdir(self):
        return self.datadir

    @property
    def kind(self):
        return self.id.kind

    @property
    def user(self):
        return self.id.user

    @property
    def date(self):
        return self.id.date

    def as_jsonable(self):
        data = super().as_jsonable()
        data['id'] = str(data['id'])
        data['date'] = self.date.isoformat()
        return data


class Result(Metadata):

    FIELDS = [
        'reqid',
        'reqdir',
        'status',
        'history',
    ]

    STATUS = types.SimpleNamespace(
        CREATED='created',
        PENDING='pending',
        ACTIVE='active',
        RUNNING='running',
        SUCCESS='success',
        FAILED='failed',
        CANCELED='cancelled',
    )
    _STATUS_BY_VALUE = {v: v for _, v in vars(STATUS).items()}
    CLOSED = 'closed'

    def __init__(self, reqid, reqdir, status=STATUS.CREATED, history=None):
        if not reqid:
            raise ValueError('missing reqid')
        reqid = RequestID.from_raw(reqid)

        if not reqdir:
            raise ValueError('missing reqdir')
        if not isinstance(reqdir, str):
            raise TypeError(f'expected dirname for reqdir, got {reqdir!r}')

        if status == self.STATUS.CREATED:
            status = None
        elif status:
            try:
                status = self._STATUS_BY_VALUE[status]
            except KeyError:
                raise ValueError(f'unsupported status {status!r}')
        else:
            status = None

        if history:
            h = []
            for st, date in history:
                try:
                    st = self._STATUS_BY_VALUE[st]
                except KeyError:
                    if st == self.CLOSED:
                        st = self.CLOSED
                    else:
                        raise ValueError(f'unsupported history status {st!r}')
                if not date:
                    date = None
                elif isinstance(date, str):
                    date = datetime.datetime.fromisoformat(date)
                elif isinstance(date, int):
                    date = datetime.datetime.utcfromtimestamp(date)
                elif not isinstance(date, datetime.datetime):
                    raise TypeError(f'unsupported history date {date!r}')
                h.append((st, date))
            history = h
        else:
            history = [
                (self.STATUS.CREATED, reqid.date),
            ]
            if status is not None:
                for st in self._STATUS_BY_VALUE:
                    history.append((st, None))
                    if status == st:
                        break
                    if status == self.STATUS.RUNNING:
                        history.append((status, None))

        super().__init__(
            reqid=reqid,
            reqdir=reqdir,
            status=status,
            history=history,
        )

    def __str__(self):
        return str(self.reqid)

    @property
    def short(self):
        if not self.status:
            return f'<{self.reqid}: (created)>'
        return f'<{self.reqid}: {self.status}>'

    @property
    def request(self):
        try:
            return self._request
        except AttributeError:
            self._request = Request(self.reqid, self.reqdir)
            return self._request

    def set_status(self, status):
        if not status:
            raise ValueError('missing status')
        try:
            status = self._STATUS_BY_VALUE[status]
        except KeyError:
            raise ValueError(f'unsupported status {status!r}')
        if self.history[-1][0] is self.CLOSED:
            raise Exception(f'req {self.reqid} is already closed')
        # XXX Make sure it is the next possible status?
        self.history.append(
            (status, datetime.datetime.now(datetime.timezone.utc)),
        )
        self.status = status

    def close(self):
        if self.history[-1][0] is self.CLOSED:
            # XXX Fail?
            return
        self.history.append(
            (self.CLOSED, datetime.datetime.now(datetime.timezone.utc)),
        )

    def as_jsonable(self):
        data = super().as_jsonable()
        data['reqid'] = str(data['reqid'])
        data['history'] = [(st, d.isoformat() if d else None)
                           for st, d in data['history']]
        return data


##################################
# minor utils

def _utcnow():
    if time.tzname[0] == 'UTC':
        return time.time()
    return time.mktime(time.gmtime())


def _resolve_user(cfg, user=None):
    if not user:
        user = USER
        if not user or user == 'benchmarking':
            user = SUDO_USER
            if not user:
                raise Exception('could not determine user')
    if not user.isidentifier():
        raise ValueError(f'invalid user {user!r}')
    return user


def _check_name(name, *, loose=False):
    if not name:
        raise ValueError(name)
    orig = name
    if loose:
        name = '_' + name.replace('-', '_')
    if not name.isidentifier():
        raise ValueError(orig)


class Version(namedtuple('Version', 'major minor micro level serial')):

    prefix = None

    @classmethod
    def parse(cls, verstr):
        m = re.match(r'^(v)?(\d+)\.(\d+)(?:\.(\d+))?(?:(a|b|c|rc|f)(\d+))?$',
                     verstr)
        if not m:
            return None
        prefix, major, minor, micro, level, serial = m.groups()
        if level == 'a':
            level = 'alpha'
        elif level == 'b':
            level = 'beta'
        elif level in ('c', 'rc'):
            level = 'candidate'
        elif level == 'f':
            level = 'final'
        elif level:
            raise NotImplementedError(repr(verstr))
        self = cls(
            int(major),
            int(minor),
            int(micro) if micro else 0,
            level or 'final',
            int(serial) if serial else 0,
        )
        if prefix:
            self.prefix = prefix
        return self

    def as_tag(self):
        micro = f'.{self.micro}' if self.micro else ''
        if self.level == 'alpha':
            release = f'a{self.serial}'
        elif self.level == 'beta':
            release = f'b{self.serial}'
        elif self.level == 'candidate':
            release = f'rc{self.serial}'
        elif self.level == 'final':
            release = ''
        else:
            raise NotImplementedError(self.level)
        return f'v{self.major}.{self.minor}{micro}{release}'


def _read_file(filename):
    with open(filename) as infile:
        return infile.read()


##################################
# git utils

def git(*args, GIT=shutil.which('git')):
    print(f'# running: {" ".join(args)}')
    proc = subprocess.run(
        [GIT, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
    )
    return proc.returncode, proc.stdout


class GitHubTarget(types.SimpleNamespace):

    @classmethod
    def origin(cls, org, project):
        return cls(org, project, remote_name='origin')

    def __init__(self, org, project, ref=None, remote_name=None, upstream=None):
        _check_name(org, loose=True)
        _check_name(project, loose=True)
        if not ref:
            ref = None
        elif not _looks_like_git_branch(ref):
            if not _looks_like_git_revision(ref):
                raise ValueError(ref)
        if not remote_name:
            remote_name = None
        else:
            _check_name(remote_name, loose=True)
        if upstream is not None and not isinstance(upstream, GitHubTarget):
            raise TypeError(upstream)

        kwargs = dict(locals())
        del kwargs['self']
        del kwargs['__class__']
        super().__init__(**kwargs)

    @property
    def remote(self):
        if self.remote_name:
            return self.remote_name
        return self.org if self.upstream else 'upstream'

    @property
    def fullref(self):
        if self.ref:
            if _looks_like_git_revision(self.ref):
                return self.ref
            branch = self.ref
        else:
            branch = 'main'
        return f'{self.remote}/{branch}' if self.remote else branch

    @property
    def url(self):
        return f'https://github.com/{self.org}/{self.project}'

    @property
    def archive_url(self):
        ref = self.ref or 'main'
        return f'{self.url}/archive/{self.ref or "main"}.tar.gz'

    def copy(self, ref=None):
        return type(self)(
            org=self.org,
            project=self.project,
            ref=ref or self.ref,
            remote_name=self.remote_name,
            upstream=self.upstream,
        )

    def fork(self, org, project=None, ref=None, remote_name=None):
        return type(self)(
            org=org,
            project=project or self.project,
            ref=ref or self.ref,
            remote_name=remote_name,
            upstream=self,
        )

    def as_jsonable(self):
        return dict(vars(self))


def _git_remote_from_ref(ref):
    ...


def _resolve_git_remote(remote, user=None, revision=None, branch=None):
    if remote:
        # XXX Parse "NAME|URL" and "gh:NAME"?
        # XXX Validate it?
        return remote

    if revision:
        remote = _git_remote_from_ref(revision)
        if remote:
            return remote



def _resolve_git_remote(remote, user=None, revision=None, branch=None):
    if remote:
        # XXX Validate it?
        return remote

    # XXX Try $GITHUB_USER or something?
    # XXX Fall back to "user"?
    raise ValueError('missing remote')


def _looks_like_git_branch(value):
    return bool(re.match(r'^[\w][\w.-]*$', value))


def _looks_like_git_revision(value):
    return bool(re.match(r'^[a-fA-F0-9]{4,40}$', value))


#def _resolve_git_revision(revision, branch=None):
#    if not revision:
#        if not branch:
#            raise ValueError('missing revision')
#        if _looks_like_git_revision(branch):
#            return branch
#        return None
#
#    if _looks_like_git_revision(revision):
#        return revision
#
#    if not branch:
#        if not _looks_like_git_branch(revision):
#            raise ValueError(f'invalid revision {revision!r}')
#        # _resolve_git_branch() should use the old revision value.
#        return None
#    if revision != branch:
#        raise ValueError(f'invalid revision {revision!r}')
#    return None
#
#
#def _resolve_git_branch(revision, branch=None):
#    #if not revision:
#    #    if not branch:
#    #        raise ValueError('missing revision')
#    #    if _looks_like_git_revision(branch):
#    #        return branch
#    #    return None
#
#    #if _looks_like_git_revision(revision):
#    #    return revision
#
#    #if not branch:
#    #    if not _looks_like_git_branch(revision):
#    #        raise ValueError(f'invalid revision {revision!r}')
#    #    # _resolve_git_branch() should use the old revision value.
#    #    return None
#    #if revision != branch:
#    #    raise ValueError(f'invalid revision {revision!r}')
#    #return None
#
#
#def _resolve_git_revision_and_branch(revision, branch):
#    if branch:
#        revision = _resolve_git_revision(revision, branch)
#        branch = _resolve_git_branch(branch, revision)
#    else:
#        branch = _resolve_git_branch(branch, revision)
#        revision = _resolve_git_revision(revision, branch)
#    return revision, branch


def _find_git_ref(remote, ref, latest=False):
    version = Version.parse(ref)
    if version:
        if not latest and ref != f'{version.major}.{version.minor}':
            ref = version.as_tag()
    elif latest:
        raise ValueError(f'expected version, got {ref!r}')
    # Get the full list of refs for the remote.
    if remote == 'origin' or not remote:
        url = 'https://github.com/python/cpython'
    elif remote == 'upstream':
        url = 'https://github.com/faster-cpython/cpython'
    else:
        url = f'https://github.com/{remote}/cpython'
    ec, text = git('ls-remote', '--refs', '--tags', '--heads', url)
    if ec != 0:
        return None, None, None
    branches = {}
    tags = {}
    for line in text.splitlines():
        m = re.match(r'^([a-zA-Z0-9]+)\s+refs/(heads|tags)/(\S.*)$', line)
        if not m:
            continue
        commit, kind, name = m.groups()
        if kind == 'heads':
            group = branches
        elif kind == 'tags':
            group = tags
        else:
            raise NotImplementedError
        group[name] = commit
    # Find the matching ref.
    if latest:
        branch = f'{version.major}.{version.minor}'
        matched = {}
        # Find the latest tag that matches the branch.
        for tag in tags:
            tagver = Version.parse(tag)
            if tagver and f'{tagver.major}.{tagver.minor}' == branch:
                matched[tagver] = tags[tag]
        if matched:
            key = sorted(matched)[-1]
            commit = matched[key]
            return branch, key.as_tag(), commit
        # Fall back to the branch.
        for name in branches:
            if name != branch:
                continue
            commit = branches[branch]
            return branch, None, commit
        else:
            return None, None, None
    else:
        # Match branches first.
        for branch in branches:
            if branch != ref:
                continue
            commit = branches[branch]
            return branch, None, commit
        # Then try tags.
        if version:
            for tag in tags:
                tagver = Version.parse(tag)
                if tagver != version:
                    continue
                commit = tags[tag]
                branch = f'{version.major}.{version.minor}'
                if branch not in branches:
                    branch = None
                return branch, version.as_tag(), commit
        else:
            for tag in tags:
                if name != tag:
                    continue
                branch = None
                commit = tags[tag]
                return branch, version.as_tag(), commit
        return None, None, None


def _resolve_git_revision_and_branch(revision, branch, remote):
    if not branch:
        branch = _branch = None
    elif not _looks_like_git_branch(branch):
        raise ValueError(f'bad branch {branch!r}')

    if not revision:
        raise ValueError('missing revision')
    if revision == 'latest':
        if not branch:
            raise ValueError('missing branch')
        if not re.match(r'^\d+\.\d+$', branch):
            raise ValueError(f'expected version branch, got {branch!r}')
        _, tag, revision = _find_git_ref(remote, branch, latest=True)
        if not revision:
            raise ValueError(f'branch {branch!r} not found')
    elif not _looks_like_git_revision(revision):
        # It must be a branch or tag.
        _branch, tag, _revision = _find_git_ref(remote, revision)
        if not revision:
            raise ValueError(f'bad revision {revision!r}')
        revision = _revision
    elif _looks_like_git_branch(revision):
        # It might be a branch or tag.
        _branch, tag, _revision = _find_git_ref(remote, revision)
        if revision:
            revision = _revision
    else:
        tag = None
    return revision, branch or _branch, tag


##################################
# files

DATA_ROOT = os.path.expanduser(f'{HOME}/BENCH')


class BaseRequestFS(types.SimpleNamespace):
    """The file structure that the portal and bench hosts have in common."""

    @classmethod
    def from_user(cls, user, reqid):
        topdata = f'/home/{user}/BENCH'
        return cls(reqid, topdata)

    def __init__(self, reqid, topdata=DATA_ROOT):
        reqid = RequestID.from_raw(reqid)
        if topdata:
            # XXX Leaving this as-is may be more user-friendly.
            topdata = os.path.abspath(os.path.expanduser(topdata))
        else:
            topdata = DATA_ROOT
        reqdir = f'{topdata}/REQUESTS/{reqid}'
        super().__init__(
            reqid=reqid,
            topdata=topdata,
            requests=f'{topdata}/REQUESTS',
            reqdir=reqdir,
            resdir=reqdir,
        )

    # the request

    @property
    def request_meta(self):
        """The request metadata file."""
        return f'{self.reqdir}/request.json'

    @property
    def manifest(self):
        """The manifest file pyperformance will use."""
        return f'{self.reqdir}/benchmarks.manifest'

    @property
    def pyperformance_config(self):
        """The config file pyperformance will use."""
        #return f'{self.reqdir}/compile.ini'
        return f'{self.reqdir}/pyperformance.ini'

    # the job

    @property
    def bench_script(self):
        """The script that runs the job on the bench host."""
        return f'{self.reqdir}/run.sh'

    # the results

    @property
    def results_meta(self):
        """The results metadata file."""
        return f'{self.resdir}/results.json'

    @property
    def pyperformance_log(self):
        """The log file for output from pyperformance."""
        #return f'{self.resdir}/run.log'
        return f'{self.resdir}/pyperformance.log'

    @property
    def pyperformance_results(self):
        """The benchmarks results file generated by pyperformance."""
        #return f'{self.reqdir}/results-data.json.gz'
        return f'{self.reqdir}/pyperformance-results.json.gz'


class PortalRequestFS(BaseRequestFS):
    """Files on the portal host."""

    def __init__(self, reqid, portaldata=DATA_ROOT):
        super().__init__(reqid, portaldata)
        self.portaldata = self.topdata

    @property
    def current_request(self):
        """The directory of the currently running request, if any."""
        return f'{self.requests}/CURRENT'

    # the job

    @property
    def portal_script(self):
        """The script that preps the job, runs it, and cleans up.

        This is run on the portal host and exclusively targets
        the bench host.  The prep entails sending request files
        and cleanup involves retreiving the results.
        """
        return f'{self.reqdir}/send.sh'

    @property
    def logfile(self):
        """Where the job output is written."""
        return f'{self.reqdir}/job.log'


class BenchRequestFS(BaseRequestFS):
    """Files on the bench host."""

    def __init__(self, reqid, benchdata=DATA_ROOT):
        super().__init__(reqid, benchdata)
        self.benchdata = self.topdata
        self.repos = f'{self.benchdata}/repositories'

    # the local git repositories used by the job

    @property
    def cpython_repo(self):
        return f'{self.repos}/cpython'

    @property
    def pyperformance_repo(self):
        return f'{self.repos}/pyperformance'

    @property
    def pyston_benchmarks_repo(self):
        return f'{self.repos}/pyston-benchmarks'

    # other directories needed by the job

    @property
    def venv(self):
        """The venv where pyperformance should be installed."""
        return f'{self.reqdir}/pyperformance-venv'

    @property
    def scratch_dir(self):
        """Where pyperformance will keep intermediate files.

        For example, CPython is built and installed here.
        """
        return f'{self.reqdir}/pyperformance-scratch'

    # the results

    @property
    def pyperformance_results_glob(self):
        """Finds the benchmarks results file generated by pyperformance."""
        return f'{self.resdir}/*.json.gz'


##################################
# config

class PortalConfig(types.SimpleNamespace):

    CONFIG = f'{DATA_ROOT}/portal.json'

    @classmethod
    def load(cls, filename=None):
        if not filename:
            filename = cls.CONFIG
        with open(filename) as infile:
            data = json.load(infile)
        self = cls(**data)
        self._filename = os.path.abspath(os.path.expanduser(filename))
        return self

    def __init__(self,
                 bench_user,
                 send_user,
                 send_host,
                 send_port,
                 data_dir=None,
                 ):
        if not bench_user:
            raise ValueError('missing bench_user')
        if not send_user:
            send_user = bench_user
        if not send_host:
            raise ValueError('missing send_host')
        if not send_port:
            raise ValueError('missing send_port')
        if data_dir:
            datadir = os.path.abspath(os.path.expanduser(data_dir))
        super().__init__(
            bench_user=bench_user,
            send_user=send_user,
            send_host=send_host,
            send_port=send_port,
            data_dir=data_dir or None,
        )

    def __str__(self):
        return self.filename

    @property
    def filename(self):
        try:
            return self._filename
        except AttributeError:
            return None


class BenchConfig(types.SimpleNamespace):

    CONFIG = f'{DATA_ROOT}/bench.json'

    @classmethod
    def load(cls, filename=None):
        with open(filename or cls.CONFIG) as infile:
            data = json.load(infile)
        return cls(**data)

    def __init__(self,
                 portal,
                 ):
        super().__init__(
            portal=portal,
        )


##################################
# staging requests

class StagedRequestError(Exception):
    pass


class RequestAlreadyStagedError(StagedRequestError):

    def __init__(self, reqid, curid):
        super().__init__(f'could not stage {reqid} ({curid} already staged)')
        self.reqid = reqid
        self.curid = curid


class RequestNotStagedError(StagedRequestError):

    def __init__(self, reqid, curid=None):
        msg = f'{reqid} is not currently staged'
        if curid:
            msg = f'{msg} ({curid} is)'
        super().__init__(msg)
        self.reqid = reqid
        self.curid = curid


class StagedRequestResolveError(Exception):
    def __init__(self, reqid, reqdir, reason, msg):
        super().__init__(f'{reason} ({msg} - {reqdir})')
        self.reqid = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def _get_staged_request(pfiles):
    try:
        reqdir = os.readlink(pfiles.current_request)
    except FileNotFoundError:
        return None
    requests, reqidstr = os.path.split(reqdir)
    reqid = RequestID.parse(reqidstr)
    if not reqid:
        return StagedRequestResolveError(None, reqdir, 'invalid', f'{reqidstr!r} not a request ID')
    if requests != pfiles.requests:
        return StagedRequestResolveError(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    if not os.path.exists(reqdir):
        return StagedRequestResolveError(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        return StagedRequestResolveError(reqid, reqdir, 'malformed', 'target is not a directory')
    # XXX Do other checks?
    return reqid


def stage_request(reqid, pfiles):
    try:
        os.symlink(pfiles.reqdir, pfiles.current_request)
    except FileExistsError:
        # XXX Delete the existing one if bogus?
        curid = _get_staged_request(pfiles) or '???'
        if isinstance(curid, Exception):
            raise RequestAlreadyStagedError(reqid, '???') from curid
        else:
            raise RequestAlreadyStagedError(reqid, curid)


def unstage_request(reqid, pfiles):
    reqid = RequestID.from_raw(reqid)
    curid = _get_staged_request(pfiles)
    if not curid or not isinstance(curid, (str, RequestID)):
        raise RequestNotStagedError(reqid)
    elif str(curid) != str(reqid):
        raise RequestNotStagedError(reqid, curid)
    os.unlink(pfiles.current_request)


##################################
# "compile"

class BenchCompileRequest(Request):

    FIELDS = Request.FIELDS + [
        'ref',
        'remote',
        'branch',
        'benchmarks',
        'optimize',
        'debug',
    ]
    OPTIONAL = ['remote', 'branch', 'benchmarks', 'optimize', 'debug']

    CPYTHON = GitHubTarget.origin('python', 'cpython')
    PYPERFORMANCE = GitHubTarget.origin('python', 'pyperformance')
    PYSTON_BENCHMARKS = GitHubTarget.origin('pyston', 'python-macrobenchmarks')

    #pyperformance = PYPERFORMANCE.copy('034f58b')  # 1.0.4 release (2022-01-26)
    pyperformance = PYPERFORMANCE.copy('5b6142e')  # will be 1.0.5 release
    pyston_benchmarks = PYSTON_BENCHMARKS.copy('96e7bb3')  # main from 2022-01-21
    #pyperformance = PYPERFORMANCE.fork('ericsnowcurrently', 'python-performance', 'benchmark-management')
    #pyston_benchmarks = PYSTON_BENCHMARKS.fork('ericsnowcurrently', 'pyston-macrobenchmarks', 'pyperformance')

    def __init__(self,
                 id,
                 reqdir,
                 ref,
                 remote=None,
                 branch=None,
                 benchmarks=None,
                 optimize=True,
                 debug=False,
                 ):
        if branch and not _looks_like_git_branch(branch):
            raise ValueError(branch)
        if not _looks_like_git_branch(ref):
            if not _looks_like_git_revision(ref):
                raise ValueError(ref)

        super().__init__(id, reqdir)
        self.ref = ref
        self.remote = remote
        self.branch = branch
        self.benchmarks = benchmarks
        self.optimize = optimize
        self.debug = debug

    @property
    def cpython(self):
        if self.remote:
            return self.CPYTHON.fork(self.remote, ref=self.ref)
        else:
            return self.CPYTHON.copy(ref=self.ref)

    @property
    def result(self):
        return BenchCompileResult(self.id, self.reqdir)


class BenchCompileResult(Result):

    FIELDS = Result.FIELDS + [
        'pyperformance_results',
        'pyperformance_results_orig',
    ]
    OPTIONAL = [
        'pyperformance_results',
        'pyperformance_results_orig',
    ]

    def __init__(self, reqid, reqdir, *,
                 status=None,
                 pyperformance_results=None,
                 pyperformance_results_orig=None,
                 **kwargs
                 ):
        super().__init__(reqid, reqdir, status, **kwargs)
        self.pyperformance_results = pyperformance_results
        self.pyperformance_results_orig = pyperformance_results_orig

    def as_jsonable(self):
        data = super().as_jsonable()
        for field in ['pyperformance_results', 'pyperformance_results_orig']:
            if not data[field]:
                del data[field]
        data['reqid'] = str(data['reqid'])
        return data


def _resolve_bench_compile_request(reqid, reqdir, remote, revision, branch,
                                   benchmarks,
                                   *,
                                   optimize,
                                   debug,
                                   ):
    commit, branch, tag = _resolve_git_revision_and_branch(revision, branch, remote)
    remote = _resolve_git_remote(remote, reqid.user, branch, commit)

    if isinstance(benchmarks, str):
        benchmarks = benchmarks.replace(',', ' ').split()
    if benchmarks:
        benchmarks = (b.strip() for b in benchmarks)
        benchmarks = [b for b in benchmarks if b]

    meta = BenchCompileRequest(
        id=reqid,
        reqdir=reqdir,
        # XXX Add a "commit" field and use "tag or branch" for ref.
        ref=commit,
        remote=remote,
        branch=branch,
        benchmarks=benchmarks or None,
        optimize=bool(optimize),
        debug=bool(debug),
    )
    return meta


def _build_manifest(req, bfiles):
    return textwrap.dedent(f'''
        [includes]
        <default>
        {bfiles.pyston_benchmarks_repo}/benchmarks/MANIFEST
    '''[1:-1])


def _build_pyperformance_config(req, pfiles, bfiles):
    cfg = configparser.ConfigParser()

    cfg['config'] = {}
    cfg['config']['json_dir'] = bfiles.resdir
    cfg['config']['debug'] = str(req.debug)
    # XXX pyperformance should be looking in [scm] for this.
    cfg['config']['git_remote'] = req.remote

    cfg['scm'] = {}
    cfg['scm']['repo_dir'] = bfiles.cpython_repo
    cfg['scm']['git_remote'] = req.remote
    cfg['scm']['update'] = 'True'

    cfg['compile'] = {}
    cfg['compile']['bench_dir'] = bfiles.scratch_dir
    cfg['compile']['pgo'] = str(req.optimize)
    cfg['compile']['lto'] = str(req.optimize)
    cfg['compile']['install'] = 'True'

    cfg['run_benchmark'] = {}
    cfg['run_benchmark']['manifest'] = pfiles.manifest
    cfg['run_benchmark']['benchmarks'] = ','.join(req.benchmarks or ())
    cfg['run_benchmark']['system_tune'] = 'True'
    cfg['run_benchmark']['upload'] = 'False'

    return cfg


def _check_shell_str(value, *, required=True, allowspaces=False):
    if not value and required:
        raise ValueError(f'missing required value')
    if not isinstance(value, str):
        raise TypeError(f'expected str, got {value!r}')
    if not allowspaces and ' ' in value:
        raise ValueError(f'unexpected space in {value!r}')
    return value


def _quote_shell_str(value, *, required=True):
    _check_shell_str(value, required=required, allowspaces=True)
    return shlex.quote(value)


def _build_compile_script(req, bfiles):
    python = 'python3.9'  # On the bench host:
    numjobs = 20

    _check_shell_str(str(req.id) if req.id else '')
    _check_shell_str(req.cpython.url)
    _check_shell_str(req.cpython.remote)
    _check_shell_str(req.pyperformance.url)
    _check_shell_str(req.pyperformance.remote)
    _check_shell_str(req.pyston_benchmarks.url)
    _check_shell_str(req.pyston_benchmarks.remote)
    _check_shell_str(req.branch, required=False)
    maybe_branch = req.branch or ''
    _check_shell_str(req.ref)

    _check_shell_str(bfiles.pyperformance_config)
    _check_shell_str(bfiles.cpython_repo)
    _check_shell_str(bfiles.pyperformance_repo)
    _check_shell_str(bfiles.pyston_benchmarks_repo)
    _check_shell_str(bfiles.pyperformance_log)
    _check_shell_str(bfiles.results_meta)
    _check_shell_str(bfiles.pyperformance_results)
    _check_shell_str(bfiles.pyperformance_results_glob)

    _check_shell_str(python)

    # Set this to a number to skip actually running pyperformance.
    exitcode = ''

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # This script runs only on the bench host.

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        #####################
        # Mark the result as running.

        status=$(jq -r '.status' {bfiles.results_meta})
        if [ "$status" != 'active' ]; then
            2>&1 echo "ERROR: expected active status, got $status"
            2>&1 echo "       (see {bfiles.results_meta})"
            exit 1
        fi

        ( set -x
        jq --arg date $(date -u -Iseconds) '.history += [["running", $date]]' {bfiles.results_meta} > {bfiles.results_meta}.tmp
        mv {bfiles.results_meta}.tmp {bfiles.results_meta}
        )

        #####################
        # Ensure the dependencies.

        if [ ! -e {bfiles.cpython_repo} ]; then
            ( set -x
            git clone https://github.com/python/cpython {bfiles.cpython_repo}
            )
        fi
        if [ ! -e {bfiles.pyperformance_repo} ]; then
            ( set -x
            git clone https://github.com/python/pyperformance {bfiles.pyperformance_repo}
            )
        fi
        if [ ! -e {bfiles.pyston_benchmarks_repo} ]; then
            ( set -x
            git clone https://github.com/pyston/python-macrobenchmarks {bfiles.pyston_benchmarks_repo}
            )
        fi

        #####################
        # Get the repos are ready for the requested remotes and revisions.

        remote='{req.cpython.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {bfiles.cpython_repo} remote add {req.cpython.remote} {req.cpython.url}
            git -C {bfiles.cpython_repo} fetch --tags {req.cpython.remote}
            )
        fi
        # Get the upstream tags, just in case.
        ( set -x
        git -C {bfiles.cpython_repo} fetch --tags origin
        )
        branch='{maybe_branch}'
        if [ -n "$branch" ]; then
            if ! ( set -x
                git -C {bfiles.cpython_repo} checkout -b {req.branch or '$branch'} --track {req.cpython.remote}/{req.branch or '$branch'}
            ); then
                echo "It already exists; resetting to the right target."
                ( set -x
                git -C {bfiles.cpython_repo} checkout {req.branch or '$branch'}
                git -C {bfiles.cpython_repo} reset --hard {req.cpython.remote}/{req.branch or '$branch'}
                )
            fi
        fi

        remote='{req.pyperformance.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {bfiles.pyperformance_repo} remote add {req.pyperformance.remote} {req.pyperformance.url}
            )
        fi
        ( set -x
        git -C {bfiles.pyperformance_repo} fetch --tags {req.pyperformance.remote}
        git -C {bfiles.pyperformance_repo} checkout {req.pyperformance.fullref}
        )

        remote='{req.pyston_benchmarks.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {bfiles.pyston_benchmarks_repo} remote add {req.pyston_benchmarks.remote} {req.pyston_benchmarks.url}
            )
        fi
        ( set -x
        git -C {bfiles.pyston_benchmarks_repo} fetch --tags {req.pyston_benchmarks.remote}
        git -C {bfiles.pyston_benchmarks_repo} checkout {req.pyston_benchmarks.fullref}
        )

        #####################
        # Run the benchmarks.

        echo "running the benchmarks..."
        echo "(logging to {bfiles.pyperformance_log})"
        exitcode='{exitcode}'
        echo "$exitcode"
        if [ -n "$exitcode" ]; then
            ( set -x
            touch {bfiles.pyperformance_log}
            touch {bfiles.reqdir}//pyperformance-dummy-results.json.gz
            )
        else
            ( set -x
            MAKEFLAGS='-j{numjobs}' \\
                {python} {bfiles.pyperformance_repo}/dev.py compile \\
                {bfiles.pyperformance_config} \\
                {req.ref} {maybe_branch} \\
                2>&1 | tee {bfiles.pyperformance_log}
            )
            exitcode=$?
        fi

        #####################
        # Record the results metadata.

        results=$(2>/dev/null ls {bfiles.pyperformance_results_glob})
        results_name=$(2>/dev/null basename $results)

        echo "saving results..."
        if [ $exitcode -eq 0 -a -n "$results" ]; then
            ( set -x
            jq '.status = "success"' {bfiles.results_meta} > {bfiles.results_meta}.tmp
            mv {bfiles.results_meta}.tmp {bfiles.results_meta}

            jq --arg results "$results" '.pyperformance_data_orig = $results' {bfiles.results_meta} > {bfiles.results_meta}.tmp
            mv {bfiles.results_meta}.tmp {bfiles.results_meta}

            jq --arg date $(date -u -Iseconds) '.history += [["success", $date]]' {bfiles.results_meta} > {bfiles.results_meta}.tmp
            mv {bfiles.results_meta}.tmp {bfiles.results_meta}
            )
        else
            ( set -x
            jq '.status = "failed"' {bfiles.results_meta} > {bfiles.results_meta}.tmp
            mv {bfiles.results_meta}.tmp {bfiles.results_meta}

            jq --arg date $(date -u -Iseconds) '.history += [["failed", $date]]' {bfiles.results_meta} > {bfiles.results_meta}.tmp
            mv {bfiles.results_meta}.tmp {bfiles.results_meta}
            )
        fi

        if [ -n "$results" -a -e "$results" ]; then
            ( set -x
            ln -s $results {bfiles.pyperformance_results}
            )
        fi

        echo "...done!"
    '''[1:-1])


def _build_send_script(cfg, req, pfiles, bfiles, *, hidecfg=False):
    cfgfile = _quote_shell_str(cfg.filename or cfg.CONFIG)
    if hidecfg:
        benchuser = '$benchuser'
        user = '$user'
        host = _host = '$host'
        port = '$port'
    else:
        benchuser = _check_shell_str(cfg.bench_user)
        user = _check_shell_str(cfg.send_user)
        host = _check_shell_str(cfg.send_host)
        port = cfg.send_port
    conn = f'{user}@{host}'

    #reqdir = _quote_shell_str(pfiles.current_request)
    reqdir = _quote_shell_str(pfiles.reqdir)
    results_meta = _quote_shell_str(pfiles.results_meta)
    pyperformance_results = _quote_shell_str(pfiles.pyperformance_results)
    pyperformance_log = _quote_shell_str(pfiles.pyperformance_log)

    _check_shell_str(bfiles.reqdir)
    _check_shell_str(bfiles.requests)
    _check_shell_str(bfiles.bench_script)
    _check_shell_str(bfiles.scratch_dir)
    _check_shell_str(bfiles.resdir)
    _check_shell_str(bfiles.results_meta)
    _check_shell_str(bfiles.pyperformance_results)
    _check_shell_str(bfiles.pyperformance_log)

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # This script only runs on the portal host.
        # It does 4 things:
        #   1. switch to the {benchuser} user, if necessary
        #   2. prepare the bench host, including sending all
        #      the request files to the bench host (over SSH)
        #   3. run the job (e.g. run the benchmarks)
        #   4. pull the results-related files from the bench host (over SSH)

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        cfgfile='{cfgfile}'

        benchuser=$(jq -r '.bench_user' {cfgfile})
        if [ "$USER" != '{benchuser}' ]; then
            setfacl -m {benchuser}:x $(dirname "$SSH_AUTH_SOCK")
            setfacl -m {benchuser}:rwx "$SSH_AUTH_SOCK"
            # Stop running and re-run this script as the {benchuser} user.
            exec sudo --login --user {benchuser} --preserve-env='SSH_AUTH_SOCK' "$0" "$@"
        fi

        user=$(jq -r '.send_user' {cfgfile})
        host=$(jq -r '.send_host' {cfgfile})
        port=$(jq -r '.send_port' {cfgfile})

        if ssh -p {port} {conn} test -e {bfiles.reqdir}; then
            >&2 echo "request {req.id} was already sent"
            exit 1
        fi

        set -x

        # Set up before running.
        ssh -p {port} {conn} mkdir -p {bfiles.requests}
        scp -rp -P {port} {reqdir} {conn}:{bfiles.reqdir}
        ssh -p {port} {conn} mkdir -p {bfiles.scratch_dir}
        ssh -p {port} {conn} mkdir -p {bfiles.resdir}

        # Run the request.
        ssh -p {port} {conn} {bfiles.bench_script}

        # Finish up.
        scp -p -P {port} {conn}:{bfiles.results_meta} {results_meta}
        scp -rp -P {port} {conn}:{bfiles.pyperformance_results} {pyperformance_results}
        scp -rp -P {port} {conn}:{bfiles.pyperformance_log} {pyperformance_log}
    '''[1:-1])


def render_request(reqid, pfiles):
    yield f'(from {pfiles.request_meta}):'
    yield ''
    # XXX Show something better?
    text = _read_file(pfiles.request_meta)
    yield from text.splitlines()


def render_results(reqid, pfiles):
    yield f'(from {pfiles.results_meta}):'
    yield ''
    # XXX Show something better?
    text = _read_file(pfiles.results_meta)
    yield from text.splitlines()


##################################
# commands

def cmd_list(cfg):
    raise NotImplementedError


def cmd_show(cfg, reqid=None, *, lines=None, follow=False):
    raise NotImplementedError


def cmd_request_compile_bench(cfg, reqid, revision, *,
                              remote=None,
                              branch=None,
                              benchmarks=None,
                              optimize=False,
                              debug=False,
                              ):
    pfiles = PortalRequestFS(reqid, cfg.data_dir)
    bfiles = BenchRequestFS.from_user(cfg.bench_user, reqid)

    print(f'generating request files in {pfiles.reqdir}...')

    req = _resolve_bench_compile_request(
        reqid, pfiles.reqdir, remote, revision, branch, benchmarks,
        optimize=optimize,
        debug=debug,
    )
    result = req.result
    resfile = pfiles.results_meta

    os.makedirs(pfiles.reqdir, exist_ok=True)

    # Write metadata.
    req.save(pfiles.request_meta)
    result.save(resfile)

    # Write the benchmarks manifest.
    manifest = _build_manifest(req, bfiles)
    with open(pfiles.manifest, 'w') as outfile:
        outfile.write(manifest)

    # Write the config.
    ini = _build_pyperformance_config(req, pfiles, bfiles)
    with open(pfiles.pyperformance_config, 'w') as outfile:
        ini.write(outfile)

    # Write the commands to execute remotely.
    script = _build_compile_script(req, bfiles)
    with open(pfiles.bench_script, 'w') as outfile:
        outfile.write(script)
    os.chmod(pfiles.bench_script, 0o755)

    # Write the commands to execute locally.
    script = _build_send_script(cfg, req, pfiles, bfiles)
    with open(pfiles.portal_script, 'w') as outfile:
        outfile.write(script)
    os.chmod(pfiles.portal_script, 0o755)

    print('...done (generating request files)')
    print()
    for line in render_request(reqid, pfiles):
        print(f'  {line}')


def cmd_copy(cfg, reqid=None):
    raise NotImplementedError


def cmd_remove(cfg, reqid):
    raise NotImplementedError


def cmd_run(cfg, reqid, *, attach=False, copy=False, force=False):
    if copy:
        raise NotImplementedError
    if force:
        raise NotImplementedError

    print('# sending request')
    print()
    pfiles = PortalRequestFS(reqid, cfg.data_dir)

    resfile = pfiles.results_meta
    result = BenchCompileResult.load(resfile)

    result.set_status(result.STATUS.PENDING)
    result.save(resfile)

    print('staging...')
    try:
        stage_request(reqid, pfiles)

        result.set_status(result.STATUS.ACTIVE)
        result.save(resfile)
    except RequestAlreadyStagedError as exc:
        # XXX Offer to clear CURRENT?
        sys.exit(f'ERROR: {exc}')
    except Exception:
        result.set_status(result.STATUS.FAILED)
        raise  # re-raise

    try:
        print('...running....')
        subprocess.run(pfiles.portal_script)
    except KeyboardInterrupt:
        # XXX Try to download the results file directly?
        result.set_status(result.STATUS.CANCELED)
        result.save(resfile)
        raise  # re-raise
    else:
        result.refresh(resfile)
    finally:
        print('...unstaging...')
        unstage_request(reqid, pfiles)
        print('...done!')

        result.close()
        result.save(resfile)

        print()
        print('Results:')
        for line in render_results(reqid, pfiles):
            print(line)


def cmd_attach(cfg, reqid=None, *, lines=None):
    return cmd_show(cfg, reqid, lines=lines, follow=True)


def cmd_cancel(cfg, reqid=None):
    raise NotImplementedError


COMMANDS = {
    'list': cmd_list,
    'show': cmd_show,
    'copy': cmd_copy,
    'remove': cmd_remove,
    'run': cmd_run,
    'attach': cmd_attach,
    'cancel': cmd_cancel,
    # Specific jobs
    'request-compile-bench': cmd_request_compile_bench,
}


##################################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse

    ##########
    # First, pull out the common args.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', dest='cfgfile')
    args, argv = common.parse_known_args(argv)
    cfgfile = args.cfgfile

    ##########
    # Create the top-level parser.
    parser = argparse.ArgumentParser(
        prog=prog,
        parents=[common],
    )
    subs = parser.add_subparsers(dest='cmd', metavar='CMD')

    ##########
    # Add the subcommands for managing jobs.

    def add_cmd(name, **kwargs):
        return subs.add_parser(name, parents=[common], **kwargs)

#    sub = add_cmd('list', help='Print a table of all known jobs')

    sub = add_cmd('show', help='Print a summary of the given (or current) job')
    sub.add_argument('-n', '--lines', type=int, default=-1,
                     help='Show the last n lines of the job\'s output')
    sub.add_argument('--follow', action='store_true',
                     help='Attach stdout to the job\'s output')
    sub.add_argument('reqid', nargs='?')

    sub = add_cmd('request', aliases=['add'], help='Create a new job request')
    jobs = sub.add_subparsers(dest='job')
    # Subcommands for different jobs are added below.

#    sub = add_cmd('copy', help='Create a new copy of an existing job request')
#    sub.add_argument('reqid', nargs='?')

#    sub = add_cmd('remove', help='Delete a job request')
#    sub.add_argument('reqid')

    sub = add_cmd('run', help='Run a previously created job request')
    sub.add_argument('--attach', action='store_true')
    sub.add_argument('--no-attach', dest='attach',
                     action='store_const', const=False)
#    sub.add_argument('--copy', action='store_true',
#                     help='Run a new copy of the given job request')
#    sub.add_argument('--force', action='store_true',
#                     help='Run the job even if another is already running')
    sub.add_argument('reqid')

    sub = add_cmd('attach', help='Equivalent to show --follow')
    sub.add_argument('-n', '--lines', type=int, default=-1,
                     help='Show the last n lines of the job\'s output')
    sub.add_argument('reqid', nargs='?')

    sub = add_cmd('cancel', help='Stop the current job (or prevent a pending one)')
    sub.add_argument('reqid', nargs='?')

    # XXX Also add export and import?

    ##########
    # Add the "add" subcommands for the different jobs.

    _common = argparse.ArgumentParser(add_help=False)
    _common.add_argument('--run', dest='after',
                         action='store_const', const=('run', 'attach'))
    _common.add_argument('--run-attached', dest='after',
                         action='store_const', const=('run', 'attach'))
    _common.add_argument('--run-detached', dest='after',
                         action='store_const', const=('run',))
    _common.add_argument('--no-run', dest='after',
                         action='store_const', const=())

    # This is the default (and the only one, for now).
    sub = jobs.add_parser('compile-bench', parents=[common, _common],
                          help='Request a compile-and-run-benchmarks job')
    sub.add_argument('--optimize', dest='optimize',
                     action='store_true')
    sub.add_argument('--no-optimize', dest='optimize',
                     action='store_const', const=False)
    sub.add_argument('--debug', action='store_true')
    sub.add_argument('--benchmarks')
    sub.add_argument('--branch')
    sub.add_argument('--remote', required=True)
    sub.add_argument('revision')

    ##########
    # Finally, parse the args.

    args = parser.parse_args(argv)
    ns = vars(args)

    ns.pop('cfgfile')  # We already got it earlier.

    cmd = ns.pop('cmd')

    if cmd in ('add', 'request'):
        cmd = 'request-' + ns.pop('job')

    return cmd, ns, cfgfile


def main(cmd, cmd_kwargs, cfgfile=None):
    try:
        run_cmd = COMMANDS[cmd]
    except KeyError:
        sys.exit(f'unsupported cmd {cmd!r}')

    after = []
    for _cmd in cmd_kwargs.pop('after', None) or ():
        try:
            _run_cmd = COMMANDS[_cmd]
        except KeyError:
            sys.exit(f'unsupported "after" cmd {_cmd!r}')
        after.append((_cmd, _run_cmd))

    # Load the config.
    if not cfgfile:
        cfgfile = PortalConfig.CONFIG
    print()
    print(f'# loading config from {cfgfile}')
    cfg = PortalConfig.load(cfgfile)

    # Resolve the request ID, if any.
    if 'reqid' in cmd_kwargs:
        cmd_kwargs['reqid'] = RequestID.parse(cmd_kwargs['reqid'])
    elif cmd.startswith('request-'):
        cmd_kwargs['reqid'] = RequestID.generate(cfg, kind=cmd[8:])
    reqid = cmd_kwargs.get('reqid')

    # Run the command.
    print()
    print('#'*40)
    if reqid:
        print(f'# Running {cmd!r} command for request {reqid}')
    else:
        print(f'# Running {cmd!r} command')
    print()
    run_cmd(cfg, **cmd_kwargs)

    # Run "after" commands, if any
    for cmd, run_cmd in after:
        print()
        print('#'*40)
        if reqid:
            print(f'# Running {cmd!r} command for request {reqid}')
        else:
            print(f'# Running {cmd!r} command')
        print()
        run_cmd(cfg, reqid=reqid)


if __name__ == '__main__':
    cmd, cmd_kwargs, cfgfile = parse_args()
    main(cmd, cmd_kwargs, cfgfile)
