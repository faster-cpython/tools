from collections import namedtuple
import configparser
import datetime
import json
import os
import os.path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
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
# config

class Config(types.SimpleNamespace):
    """The base config for the benchmarking machinery."""

    CONFIG_DIRS = [
        f'{HOME}/.config',
        HOME,
        f'{HOME}/BENCH',
    ]
    CONFIG = 'benchmarking.json'
    ALT_CONFIG = None

    # XXX Get FIELDS from the __init__() signature?
    FIELDS = ()
    OPTIONAL = ()

    @classmethod
    def find_config(cls):
        for dirname in cls.CONFIG_DIRS:
            filename = f'{dirname}/{cls.CONFIG}'
            if os.path.exists(filename):
                return filename
        else:
            if cls.ALT_CONFIG:
                filename = f'{HOME}/BENCH/{cls.ALT_CONFIG}'
                if os.path.exists(filename):
                    return filename
            raise FileNotFoundError('could not find config file')

    @classmethod
    def load(cls, filename=None, *, preserveorig=True):
        if not filename:
            filename = cls.find_config()

        with open(filename) as infile:
            data = json.load(infile)
        if preserveorig:
            loaded = dict(data, _filename=filename)

        includes = data.pop('include', None) or ()
        if includes:
            includes = list(cls._load_includes(includes, set()))
            for field in cls.FIELDS:
                if data.get(field):
                    continue
                if field in cls.OPTIONAL:
                    continue
                for included in includes:
                    value = included.get(field)
                    if value:
                        data[field] = value
                        break

        self = cls(**data)
        self._filename = os.path.abspath(os.path.expanduser(filename))
        if preserveorig:
            self._loaded = loaded
            self._includes = includes
        return self

    @classmethod
    def _load_includes(cls, includes, seen):
        if isinstance(includes, str):
            includes = [includes]
        for i, filename in enumerate(includes):
            if not filename:
                continue
            filename = os.path.abspath(os.path.expanduser(filename))
            if filename in seen:
                continue
            seen.add(filename)
            text = _read_file(filename, fail=False)
            if not text:
                continue
            included = json.loads(text)
            included['_filename'] = filename
            yield included

            subincludes = included.get('include')
            if subincludes:
                yield from cls._load_includes(subincludes, seen)

    def __init__(self, **kwargs):
        for name in list(kwargs):
            value = kwargs[name]
            if not value:
                if name in self.OPTIONAL:
                    del kwargs[name]
        super().__init__(**kwargs)

    def __str__(self):
        if not self.filename:
            return super().__str__()
        return self.filename or super().__str__()

    @property
    def filename(self):
        try:
            return self._filename
        except AttributeError:
            return None

    def as_jsonable(self, *, withmissingoptional=True):
        # XXX Hide sensitive data?
        data = {k: v
                for k, v in vars(self).items()
                if not k.startswith('_')}
        if withmissingoptional:
            for name in self.OPTIONAL:
                if name not in data:
                    data[name] = None
        return data

    def render(self):
        data = self.as_jsonable()
        text = json.dumps(data, indent=4)
        yield from text.splitlines()


class PortalConfig(Config):

    CONFIG = 'benchmarking-portal.json'
    ALT_CONFIG = 'portal.json'

    FIELDS = ['bench_user', 'send_user', 'send_host', 'send_port', 'data_dir']
    OPTIONAL = ['data_dir']

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
            data_dir = os.path.abspath(os.path.expanduser(data_dir))
        else:
            data_dir = f'/home/{send_user}/BENCH'  # This matches DATA_ROOT.
        super().__init__(
            bench_user=bench_user,
            send_user=send_user,
            send_host=send_host,
            send_port=send_port,
            data_dir=data_dir or None,
        )


#class BenchConfig(Config):
#
#    CONFIG = f'benchmarking-bench.json'
#    ALT_CONFIG = f'bench.json'
#
#    FIELDS = ['portal']
#
#    def __init__(self,
#                 portal,
#                 ):
#        super().__init__(
#            portal=portal,
#        )


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
        return get_utc_datetime(self.timestamp)


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

    @classmethod
    def read_kind(cls, metafile):
        text = _read_file(metafile, fail=False)
        if not text:
            return None
        data = json.loads(text)
        if not data:
            return None
        kind = data.get('kind')
        if not kind:
            return None
        try:
            return RequestID._KIND_BY_VALUE[kind]
        except KeyError:
            raise ValueError(f'unsupported kind {kind!r}')

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
        return str(self.id)

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
        CANCELED='canceled',
    )
    _STATUS_BY_VALUE = {v: v for _, v in vars(STATUS).items()}
    _STATUS_BY_VALUE['cancelled'] = STATUS.CANCELED
    FINISHED = frozenset([
        STATUS.SUCCESS,
        STATUS.FAILED,
        STATUS.CANCELED,
    ])
    CLOSED = 'closed'

    @classmethod
    def read_status(cls, metafile):
       text = _read_file(metafile, fail=False)
       if not text:
           return None
       data = json.loads(text)
       return cls._STATUS_BY_VALUE.get(data.get('status'))

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
                    date = get_utc_datetime(date)
                elif isinstance(date, int):
                    date = get_utc_datetime(date)
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


def get_utc_datetime(timestamp):
    if isinstance(timestamp, datetime.datetime):
        pass
    elif isinstance(timestamp, int):
        return datetime.datetime.fromtimestamp(
            timestamp,
            datetime.timezone.utc,
        )
    elif isinstance(timestamp, str):
        if hasattr(datetime.datetime, 'fromisoformat'):  # 3.7+
            timestamp = datetime.datetime.fromisoformat(timestamp)
        else:
            m = re.match(r'(\d{4}-\d\d-\d\d(.)\d\d:\d\d:\d\d)(\.\d{3}(?:\d{3})?)?([+-]\d\d:?\d\d.*)?', timestamp)
            if not m:
                raise NotImplementedError
            body, sep, subzero, tz = m.groups()
            timestamp = body
            fmt = f'%Y-%m-%d{sep}%H:%M:%S'
            if subzero:
                if len(subzero) == 4:
                    subzero += '000'
                timestamp += subzero
                fmt += '.%f'
            if tz:
                timestamp += tz.replace(':', '')
                fmt += '%z'
            timestamp = datetime.datetime.strptime(timestamp, fmt)
    else:
        raise TypeError(f'unsupported timestamp {timestamp!r}')
    # XXX Treat naive as UTC?
    return timestamp.astimezone(datetime.timezone.utc)


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


def _read_file(filename, *, fail=True):
    try:
        with open(filename) as infile:
            return infile.read()
    except OSError as exc:
        if fail:
            raise  # re-raise
        if os.path.exists(filename):
            # XXX Use a logger.
            print(f'WARNING: could not load PID file {filename!r}')
        return None


def read_pidfile(pidfile):
    if isinstance(pidfile, str):
        filename = pidfile
        text = _read_file(pidfile, fail=False) or ''
    else:
        text = pidfile.read()
    text = text.strip()
    if not text:
        return None
    return int(text)


def tail_file(filename, nlines, *, follow=None):
    tail_args = []
    if nlines:
        tail_args.extend(['-n', f'{lines}' if nlines > 0 else '+0'])
    if follow:
        tail_args.append('--follow')
        if follow is not True:
            pid = follow
            tail_args.extend(['--pid', f'{pid}'])
    subprocess.run([shutil.which('tail'), *tail_args, filename])


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
        reqid = RequestID.from_raw(reqid) if reqid else None
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
    def pidfile(self):
        """The PID of the "send" script is written here."""
        return f'{self.reqdir}/send.pid'

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
# staging requests

class StagedRequestError(Exception):
    pass


class RequestNotPendingError(StagedRequestError):

    def __init__(self, reqid, status=None):
        super().__init__(f'could not stage {reqid} (expected pending, got {status or "???"} status)')
        self.reqid = reqid
        self.status = status


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
    status = Result.read_status(pfiles.results_meta)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
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
                 datadir,
                 ref,
                 remote=None,
                 branch=None,
                 benchmarks=None,
                 optimize=True,
                 debug=False,
                 **kwargs
                 ):
        if branch and not _looks_like_git_branch(branch):
            raise ValueError(branch)
        if not _looks_like_git_branch(ref):
            if not _looks_like_git_revision(ref):
                raise ValueError(ref)

        super().__init__(id, datadir, **kwargs)
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
        datadir=reqdir,
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
    if not cfg.filename:
        raise NotImplementedError(cfg)
    cfgfile = _quote_shell_str(cfg.filename)
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
    conn = f'{benchuser}@{host}'

    #reqdir = _quote_shell_str(pfiles.current_request)
    reqdir = _quote_shell_str(pfiles.reqdir)
    results_meta = _quote_shell_str(pfiles.results_meta)
    pidfile = _quote_shell_str(pfiles.pidfile)
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

    jobs_script = _quote_shell_str(os.path.abspath(__file__))

    if cfg.send_host == 'localhost':
        ssh = 'ssh -o StrictHostKeyChecking=no'
        scp = 'scp -o StrictHostKeyChecking=no'
    else:
        ssh = 'ssh'
        scp = 'scp'

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # This script only runs on the portal host.
        # It does 4 things:
        #   1. switch to the {user} user, if necessary
        #   2. prepare the bench host, including sending all
        #      the request files to the bench host (over SSH)
        #   3. run the job (e.g. run the benchmarks)
        #   4. pull the results-related files from the bench host (over SSH)

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        cfgfile='{cfgfile}'

        # Mark the script as running.
        echo "$$" > {pidfile}
        echo "(the "'"'"{req.kind}"'"'" job, {req.id}, has started)"
        echo

        user=$(jq -r '.send_user' {cfgfile})
        if [ "$USER" != '{user}' ]; then
            echo "(switching users from $USER to {user})"
            echo
            setfacl -m {user}:x $(dirname "$SSH_AUTH_SOCK")
            setfacl -m {user}:rwx "$SSH_AUTH_SOCK"
            # Stop running and re-run this script as the {user} user.
            exec sudo --login --user {user} --preserve-env='SSH_AUTH_SOCK' "$0" "$@"
        fi
        host=$(jq -r '.send_host' {cfgfile})
        port=$(jq -r '.send_port' {cfgfile})

        exitcode=0
        if ssh -p {port} {conn} test -e {bfiles.reqdir}; then
            >&2 echo "request {req.id} was already sent"
            exitcode=1
        else
            ( set -x

            # Set up before running.
            {ssh} -p {port} {conn} mkdir -p {bfiles.requests}
            {scp} -rp -P {port} {reqdir} {conn}:{bfiles.reqdir}
            {ssh} -p {port} {conn} mkdir -p {bfiles.scratch_dir}
            {ssh} -p {port} {conn} mkdir -p {bfiles.resdir}

            # Run the request.
            {ssh} -p {port} {conn} {bfiles.bench_script}
            exitcode=$?

            # Finish up.
            {scp} -p -P {port} {conn}:{bfiles.results_meta} {results_meta}
            {scp} -rp -P {port} {conn}:{bfiles.pyperformance_results} {pyperformance_results}
            {scp} -rp -P {port} {conn}:{bfiles.pyperformance_log} {pyperformance_log}
            )
        fi

        # Unstage the request.
        {sys.executable} {jobs_script} internal-finish-run --config {cfgfile} {req.id}

        # Mark the script as complete.
        echo
        echo "(the "'"'"{req.kind}"'"'" job, {req.id} has finished)"
        #rm -f {pidfile}

        exit $exitcode
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


def cmd_show(cfg, reqid=None, fmt=None, *, lines=None):
    if not fmt:
        fmt = 'summary'

    pfiles = PortalRequestFS(reqid, cfg.data_dir)
    if not reqid:
        reqid = _get_staged_request(pfiles)
        if not reqid:
            # XXX Use the last finished?
            raise NotImplementedError
        pfiles = PortalRequestFS(reqid, cfg.data_dir)
    reqfs_fields = [
        'bench_script',
        'portal_script',
    ]
    resfs_fields = [
        'pidfile',
        'logfile',
    ]
    kind = Request.read_kind(pfiles.request_meta)
    if kind is RequestID.KIND.BENCHMARKS:
        req_cls = BenchCompileRequest
        res_cls = BenchCompileResult
        reqfs_fields.extend([
            'manifest',
            'pyperformance_config',
        ])
        resfs_fields.extend([
            'pyperformance_log',
            'pyperformance_results',
        ])
    else:
        raise NotImplementedError(kind)
    req = req_cls.load(pfiles.request_meta)
    res = res_cls.load(pfiles.results_meta)
    pid = read_pidfile(pfiles.pidfile)

    if fmt == 'summary':
        print(f'Request {reqid}:')
        print(f'  {"kind:":20} {req.kind}')
        print(f'  {"user:":20} {req.user}')
        if pid:
            print(f'  {"PID:":20} {pid}')
        print(f'  {"status:":20} {res.status}')
        print()
        print('Details:')
        for field in req_cls.FIELDS:
            if field in ('id', 'reqid', 'kind', 'user', 'date', 'datadir'):
                continue
            value = getattr(req, field)
            if isinstance(value, str) and value.strip() != value:
                value = repr(value)
            print(f'  {field + ":":20} {value}')
        print()
        print('History:')
        for st, ts in res.history:
            print(f'  {st + ":":20} {ts:%Y-%m-%d %H:%M:%S}')
        print()
        print('Request files:')
        print(f'  {"data root:":20} {req.reqdir}')
        print(f'  {"metadata:":20} {pfiles.request_meta}')
        for field in reqfs_fields:
            value = getattr(pfiles, field, None)
            if value and not os.path.exists(value):
                value = None
            print(f'  {field + ":":20} {value or "---"}')
        print()
        print('Result files:')
        print(f'  {"data root:":20} {pfiles.resdir}')
        print(f'  {"metadata:":20} {pfiles.results_meta}')
        for field in resfs_fields:
            value = getattr(pfiles, field, None)
            if value and not os.path.exists(value):
                value = None
            print(f'  {field + ":":20} {value or "---"}')
    else:
        raise ValueError(f'unsupported fmt {fmt!r}')

    if lines:
        tail_file(pfiles.logfile, lines, follow=False)


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

    print('# staging request')
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

    # Run the send.sh script in the background.
    try:
        script = textwrap.dedent(f"""
            #!/usr/bin/env bash
            "{pfiles.portal_script}" > "{pfiles.logfile}" 2>&1 &
        """).lstrip()
        scriptfile = tempfile.NamedTemporaryFile(mode='w', delete=False)
        try:
            with scriptfile:
                scriptfile.write(script)
            os.chmod(scriptfile.name, 0o755)
            subprocess.run(scriptfile.name)
        finally:
            os.unlink(scriptfile.name)
    except Exception as exc:
        result.set_status(result.STATUS.CANCELED)
        result.save(resfile)

        result.close()
        result.save(resfile)

        raise  # re-raise
    except KeyboardInterrupt:
        cmd_cancel(cfg, reqid)
        raise  # re-raise


def cmd_attach(cfg, reqid=None, *, lines=None):
    pfiles = PortalRequestFS(reqid, cfg.data_dir)
    if not reqid:
        reqid = _get_staged_request(pfiles)
        if not reqid:
            sys.exit('ERROR: no current request to attach')
        pfiles = PortalRequestFS(reqid, cfg.data_dir)

    # Wait for the request to start.
    pid = read_pidfile(pfiles.pidfile)
    while pid is None:
        status = Result.read_status(pfiles.results_meta)
        if status is Result.FINISHED:
            # XXX Use a logger.
            print(f'WARNING: job not started')
            return
        time.sleep(0.01)
        pid = read_pidfile(pfiles.pidfile)

    # XXX Cancel the job for KeyboardInterrupt?
    if pid:
        tail_file(pfiles.logfile, lines, follow=pid)
    elif lines:
        tail_file(pfiles.logfile, lines, follow=False)


def cmd_cancel(cfg, reqid=None):
    pfiles = PortalRequestFS(reqid, cfg.data_dir)
    current = _get_staged_request(pfiles)
    if not reqid:
        if not current:
            sys.exit('ERROR: no current request to cancel')
        reqid = current
        pfiles = PortalRequestFS(reqid, cfg.data_dir)
    # XXX Use the right result type.
    resfile = pfiles.results_meta
    result = BenchCompileResult.load(resfile)

    result.set_status(result.STATUS.CANCELED)
    result.close()
    result.save(resfile)

    if reqid == current:
        print(f'# unstaging request {reqid}')
        try:
            unstage_request(reqid, pfiles)
        except RequestNotStagedError:
            pass
        print('# done unstaging request')

        # Kill the process.
        pid = read_pidfile(pfiles.pidfile)
        if pid:
            print(f'# killing PID {pid}')
            os.kill(pid, signal.SIGKILL)

    # XXX Try to download the results directly?

    print()
    print('Results:')
    for line in render_results(reqid, pfiles):
        print(line)


def cmd_finish_run(cfg, reqid):
    pfiles = PortalRequestFS(reqid, cfg.data_dir)

    print(f'# unstaging request {reqid}')
    try:
        unstage_request(reqid, pfiles)
    except RequestNotStagedError:
        pass
    print('# done unstaging request')

    resfile = pfiles.results_meta
    # XXX Use the correct type for the request.
    result = BenchCompileResult.load(resfile)

    result.close()
    result.save(resfile)

    print()
    print('Results:')
    for line in render_results(reqid, pfiles):
        print(line)


def cmd_config_show(cfg):
    for line in cfg.render():
        print(line)


COMMANDS = {
    # job management
    'list': cmd_list,
    'show': cmd_show,
    'copy': cmd_copy,
    'remove': cmd_remove,
    'run': cmd_run,
    'attach': cmd_attach,
    'cancel': cmd_cancel,
    # specific jobs
    'request-compile-bench': cmd_request_compile_bench,
    # other public commands
    'config-show': cmd_config_show,
    # internal-only
    'internal-finish-run': cmd_finish_run,
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
    sub.add_argument('-n', '--lines', type=int, default=0,
                     help='Show the last n lines of the job\'s output')
    sub.add_argument('reqid', nargs='?')

    sub = add_cmd('request', aliases=['add'], help='Create a new job request')
    jobs = sub.add_subparsers(dest='job')
    # Subcommands for different jobs are added below.

#    sub = add_cmd('copy', help='Create a new copy of an existing job request')
#    sub.add_argument('reqid', nargs='?')

#    sub = add_cmd('remove', help='Delete a job request')
#    sub.add_argument('reqid')

    sub = add_cmd('run', help='Run a previously created job request')
    sub.add_argument('--attach', dest='after',
                     action='store_const', const=('attach',))
    sub.add_argument('--no-attach', dest='after',
                     action='store_const', const=())
#    sub.add_argument('--copy', action='store_true',
#                     help='Run a new copy of the given job request')
#    sub.add_argument('--force', action='store_true',
#                     help='Run the job even if another is already running')
    sub.add_argument('reqid')

    sub = add_cmd('attach', help='Tail the job log file')
    sub.add_argument('-n', '--lines', type=int, default=0,
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
    # Add other public commands.

    sub = add_cmd('config', help='show the config')

    ##########
    # Add internal commands.

    sub = add_cmd('internal-finish-run', help='(internal-only; do not use)')
    sub.add_argument('reqid')

    ##########
    # Finally, parse the args.

    args = parser.parse_args(argv)
    ns = vars(args)

    ns.pop('cfgfile')  # We already got it earlier.

    cmd = ns.pop('cmd')

    if cmd in ('add', 'request'):
        cmd = 'request-' + ns.pop('job')
    elif cmd == 'config':
        cmd = 'config-show'

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
        cfgfile = PortalConfig.find_config()
    print()
    print(f'# loading config from {cfgfile}')
    cfg = PortalConfig.load(cfgfile)

    # Resolve the request ID, if any.
    if 'reqid' in cmd_kwargs:
        if cmd_kwargs['reqid']:
            cmd_kwargs['reqid'] = RequestID.parse(cmd_kwargs['reqid'])
        else:
            cmd_kwargs['reqid'] = None
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
        # XXX Add --lines='-1' for attach.
        run_cmd(cfg, reqid=reqid)


if __name__ == '__main__':
    cmd, cmd_kwargs, cfgfile = parse_args()
    main(cmd, cmd_kwargs, cfgfile)
