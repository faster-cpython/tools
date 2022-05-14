from collections import namedtuple
import configparser
import datetime
import json
import logging
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
import traceback
import types

'''
sudo adduser --gecos '' --disabled-password <username>
sudo --login --user <username> ssh-import-id gh:<username>
'''


USER = os.environ.get('USER', '').strip()
SUDO_USER = os.environ.get('SUDO_USER', '').strip()
HOME = os.path.expanduser('~')
CWD = os.getcwd()
PID = os.getpid()

JOBS_SCRIPT = os.path.abspath(__file__)

FAKE_DELAY = 3


# XXX "portal" -> "control"?
# XXX "bench" -> "worker"?


logger = logging.getLogger(__name__)


##################################
# string utils

def _check_name(name, *, loose=False):
    if not name:
        raise ValueError(name)
    orig = name
    if loose:
        name = '_' + name.replace('-', '_')
    if not name.isidentifier():
        raise ValueError(orig)


def _validate_string(value, argname=None, *, required=True):
    if not value and required:
        raise ValueError(f'missing {argname or "required value"}')
    if not isinstance(value, str):
        label = f' for {argname}' if argname else ''
        raise TypeError(f'expected str{label}, got {value!r}')


##################################
# date/time utils

SECOND = datetime.timedelta(seconds=1)
DAY = datetime.timedelta(days=1)


def _utcnow():
    if time.tzname[0] == 'UTC':
        return time.time()
    return time.mktime(time.gmtime())


def get_utc_datetime(timestamp=None, *, fail=True):
    tzinfo = datetime.timezone.utc
    if timestamp is None:
        timestamp = int(_utcnow())
    if isinstance(timestamp, int):
        timestamp = datetime.datetime.fromtimestamp(timestamp, tzinfo)
    elif isinstance(timestamp, str):
        if re.match(r'^\d{4}-\d\d-\d\d$', timestamp):
            timestamp = datetime.date(*(int(v) for v in timestamp.split('-')))
        elif hasattr(datetime.datetime, 'fromisoformat'):  # 3.7+
            timestamp = datetime.datetime.fromisoformat(timestamp)
            timestamp = timestamp.astimezone(tzinfo)
        else:
            m = re.match(r'(\d{4}-\d\d-\d\d(.)\d\d:\d\d:\d\d)(\.\d{3}(?:\d{3})?)?([+-]\d\d:?\d\d.*)?', timestamp)
            if not m:
                if fail:
                    raise NotImplementedError(repr(timestamp))
                return None, None
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
            timestamp = timestamp.astimezone(tzinfo)
    elif isinstance(timestamp, datetime.datetime):
        # XXX Treat naive as UTC?
        timestamp = timestamp.astimezone(tzinfo)
    elif not isinstance(timestamp, datetime.date):
        raise TypeError(f'unsupported timestamp {timestamp!r}')
    hastime = True
    if type(timestamp) is datetime.date:
        d = timestamp
        timestamp = datetime.datetime(d.year, d.month, d.day, tzinfo=tzinfo)
        #timestamp = datetime.datetime.combine(timestamp, None, datetime.timezone.utc)
        hastime = False
    return timestamp, hastime


##################################
# file utils

def _check_shell_str(value, *, required=True, allowspaces=False):
    _validate_string(value, required=required)
    if not allowspaces and ' ' in value:
        raise ValueError(f'unexpected space in {value!r}')
    return value


def _quote_shell_str(value, *, required=True):
    _check_shell_str(value, required=required, allowspaces=True)
    return shlex.quote(value)


def _write_json(data, outfile, *, _print=print):
    json.dump(data, outfile, indent=4)
    _print(file=outfile)


def _wait_for_file(filename, *, timeout=None):
    if timeout is not None and timeout > 0:
        if not isinstance(timeout, (int, float)):
            raise TypeError(f'timeout must be an float or int, got {timeout!r}')
        end = time.time() + int(timeout)
        while not os.path.exists(filename):
            time.sleep(0.01)
            if time.time() >= end:
                raise TimeoutError
    else:
        while not os.path.exists(filename):
            time.sleep(0.01)


def _read_file(filename, *, fail=True):
    try:
        with open(filename) as infile:
            return infile.read()
    except OSError as exc:
        if fail:
            raise  # re-raise
        if os.path.exists(filename):
            logger.warn('could not load file %r', filename)
        return None


def tail_file(filename, nlines, *, follow=None):
    tail_args = []
    if nlines:
        tail_args.extend(['-n', f'{nlines}' if nlines > 0 else '+0'])
    if follow:
        tail_args.append('--follow')
        if follow is not True:
            pid = follow
            tail_args.extend(['--pid', f'{pid}'])
    subprocess.run([shutil.which('tail'), *tail_args, filename])


def _render_file(filename):
    if not filename:
        return '---'
    elif isinstance(filename, FSTree):
        filename = filename.root
    if not os.path.exists(filename):
        return f'({filename})'
    elif filename[0].isspace() or filename[-1].isspace():
        return repr(filename)
    else:
        return filename


class FSTree(types.SimpleNamespace):

    @classmethod
    def from_raw(cls, raw, *, name=None):
        if isinstance(raw, cls):
            return raw
        elif not raw:
            raise ValueError('missing {name or "raw"}')
        elif isinstance(raw, str):
            return cls(raw)
        else:
            raise TypeError(f'expected FSTree, got {raw!r}')

    def __init__(self, root):
        if not root or root == '.':
            root = CWD
        else:
            root = os.path.abspath(os.path.expanduser(root))
        super().__init__(root=root)

    def __str__(self):
        return self.root

    def __fspath__(self):
        return self.root


class InvalidPIDFileError(RuntimeError):

    def __init__(self, filename, text, reason=None):
        msg = f'PID file {filename!r} is not valid'
        if reason:
            msg = f'{msg} ({reason})'
        super().__init__(msg)
        self.filename = filename
        self.text = text
        self.reason = reason


class OrphanedPIDFileError(InvalidPIDFileError):

    def __init__(self, filename, pid):
        super().__init__(filename, str(pid), f'proc {pid} not running')
        self.pid = pid


class PIDFile:

    def __init__(self, filename):
        self._filename = filename

    def __repr__(self):
        return f'{type(self).__name__}({self._filename!r})'

    @property
    def filename(self):
        return self._filename

    def read(self, *, invalid='fail', orphaned=None):
        """Return the PID recorded in the file."""
        if invalid is None:
            def handle_invalid(text):
                return text
        if invalid == 'fail':
            def handle_invalid(text):
                raise InvalidPIDFileError(self._filename, text)
        elif invalid == 'remove':
            def handle_invalid(text):
                logger.warn('removing invalid PID file (%s)', self._filename)
                self.remove()
                return None
        else:
            raise ValueError(f'unsupported invalid handler {invalid!r}')

        #text = _read_file(self._filename, fail=False) or ''
        try:
            with open(self._filename) as pidfile:
                text = pidfile.read()
        except FileNotFoundError:
            return None

        text = text.strip()
        if not text or not text.isdigit():
            return handle_invalid(text)
        pid = int(text)
        if pid <= 0:
            return handle_invalid(text)

        if orphaned is not None and not _is_proc_running(pid):
            if orphaned == 'fail':
                raise OrphanedPIDFileError(self._filename, pid)
            elif orphaned == 'remove':
                logger.warn('removing orphaned PID file (%s)', self._filename)
                self.remove()
                return None
            elif orphaned == 'ignore':
                return None
            else:
                raise ValueError(f'unsupported orphaned handler {orphaned!r}')
        return pid

    def write(self, pid=PID, *, exclusive=True, **read_kwargs):
        """Return True for success after trying to create the file."""
        pid = int(pid) if pid else PID
        assert pid > 0, pid
        try:
            if exclusive:
                try:
                    pidfile = open(self._filename, 'x')
                except FileExistsError:
                    _pid = self.read(**read_kwargs)
                    if _pid == pid:
                        return pid
                    elif _pid is not None:
                        return None
                    # Looks like there was a race or invalid files.
                    #  Try one more time.
                    pidfile = open(self._filename, 'x')
            else:
                pidfile = open(self._filename, 'w')
            with pidfile:
                pidfile.write(f'{pid}')
            return pid
        except OSError as exc:
            logger.warn('failed to create PID file (%s): %s', self._filename, exc)
            return None

    def remove(self):
        try:
            os.unlink(self._filename)
        except FileNotFoundError:
            logger.warn('lock file not found (%s)', self._filename)


class LockFile:
    """A multi-process equivalent to threading.RLock."""

    def __init__(self, filename):
        self._pidfile = PIDFile(filename)
        self._count = 0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    @property
    def filename(self):
        return self._pidfile.filename

    def read(self):
        return self._pidfile.read(invalid='fail', orphaned='fail')

    def owned(self):
        """Return True if the current process holds the lock."""
        owner = self.owner()
        if owner is None:
            return False
        return owner == PID

    def owner(self):
        """Return the PID of the process that is holding the lock."""
        if self._count > 0:
            return PID
        pid = self._pidfile.read(invalid='remove', orphaned='remove')
        if pid == PID:
            assert self._count == 0, self._count
            raise NotImplementedError
        return pid

    ###################
    # threading.Lock API

    def locked(self):
        return self.owner() is not None

    def acquire(self, blocking=True, timeout=-1):
        if self._count == 0:
            if timeout is not None and timeout >= 0:
                raise NotImplementedError
            while True:
                pid = self._pidfile.write(
                    PID,
                    invalid='remove',
                    orphaned='remove',
                )
                if pid is not None:
                    break
                if not blocking:
                    return False
        self._count += 1
        return True

    def release(self):
        if self._count == 0:
            # XXX double-check the file?
            raise RuntimeError('lock not held')
        self._count -= 1
        if self._count == 0:
            self._pidfile.remove()


##################################
# logging utils

class LogSection(types.SimpleNamespace):
    """A titled, grouped sequence of log entries."""

    @classmethod
    def read_logfile(cls, logfile):
        # Currently only a "simple" format is supported.
        if isinstance(logfile, str):
            filename = logfile
            with open(filename) as logfile:
                yield from cls.read_logfile(logfile)
                return

        parsed = cls._iter_lines_and_headers(logfile)
        # Ignore everything up to the first header.
        for value in parsed:
            if not isinstance(value, str):
                _, title, _, timestamp = value
                section = cls.from_title(title[2:], timestamp[2:].strip())
                break
        else:
            return
        # Yield a LogSection for each header found.
        for value in parsed:
            if isinstance(value, str):
                section.add_lines(value)
            else:
                yield section
                _, title, _, timestamp = value
                section = cls.from_title(title[2:], timestamp[2:].strip())
        yield section

    @classmethod
    def _iter_lines_and_headers(cls, lines):
        header = None
        for line in lines:
            if line.endswith('\n'):
                # XXX Windows?
                line = line[:-1]
            if header:
                matched = False
                if len(header) == 1:
                    if line.startswith('# ') and line[2:].strip():
                        header.append(line)
                        matched = True
                elif len(header) == 2:
                    if not line:
                        header.append(line)
                        matched = True
                elif re.match(r'^# \d{4}-\d\d-\d\d \d\d:\d\d:\d\d$', line):
                    header.append(line)
                    yield header
                    header = None
                    matched = True
                if not matched:
                    yield from header
                    header = None
            elif line == ('#'*40):
                header = [line]
            else:
                yield line

    @classmethod
    def from_title(cls, title, timestamp=None, **logrecord_kwargs):
        if not title or not title.strip():
            raise ValueError('missing title')
        timestamp, _ = get_utc_datetime(timestamp or None)

        logrecord_kwargs.setdefault('name', None)
        logrecord_kwargs.setdefault('level', None)
        # These could be extrapolated:
        logrecord_kwargs.setdefault('pathname', None)
        logrecord_kwargs.setdefault('lineno', None)
        logrecord_kwargs.setdefault('exc_info', None)
        logrecord_kwargs.setdefault('func', None)

        header = logging.LogRecord(
            msg=title.strip(),
            args=None,
            **logrecord_kwargs,
        )
        header.created = timestamp.timestamp()
        header.msecs = 0
        self = cls(header)
        self._timestamp = timestamp
        return self

    def __init__(self, header):
        if not header:
            raise ValueError('missing header')
        elif not isinstance(header, logging.LogRecord):
            raise TypeError(f'expected logging.LogRecord, got {header!r}')
        super().__init__(
            header=header,
            body=[],
        )

    @property
    def title(self):
        return self.header.getMessage()

    @property
    def timestamp(self):
        try:
            return self._timestamp
        except AttributeError:
            self._timestamp = get_utc_timestamp(self.header.created)
            return self._timestamp

    def add_record(self, record):
        if isinstance(record, str):
            msg = record
            record = logging.LogRecord(
                msg=msg,
                args=None,
                name=self.header.name,
                level=self.header.levelname,
                # XXX Conditionally extrapolate the rest?
                pathname=self.header.pathname,
                lineno=self.header.lineno,
                exc_info=self.header.exc_info,
                func=self.header.funcName,
            )
            record.created = self.header.created
            record.msecs = 0
        elif not isinstance(header, logging.LogRecord):
            raise TypeError(f'expected logging.LogRecord, got {record!r}')
        self.body.append(record)

    def add_lines(self, lines):
        if isinstance(lines, str):
            lines = lines.splitlines()
        for line in lines:
            self.add_record(line)

    def render(self):
        # Currently only a "simple" format is supported.
        yield '#' * 40
        yield f'# {self.title}'
        yield ''
        yield f'# {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
        for rec in self.body:
            yield rec.getMessage()


##################################
# git utils

def git(*args, GIT=shutil.which('git')):
    logger.debug('# running: %s', ' '.join(args))
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
# other utils

def _ensure_int(raw, min=None):
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        value = int(raw)
    else:
        raise TypeError(raw)
    if value < min:
        raise ValueError(raw)
    return value


def _get_slice(raw):
    if isinstance(raw, int):
        start = stop = None
        if raw < 0:
            start = raw
        elif criteria > 0:
            stop = raw
        return slice(start, stop)
    elif isinstance(raw, str):
        if raw.isdigit():
            return _get_slice(int(raw))
        elif raw.startswith('-') and raw[1:].isdigit():
            return _get_slice(int(raw))
        else:
            raise NotImplementedError(repr(raw))
    else:
        raise TypeError(f'expected str, got {criteria!r}')


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


def _is_proc_running(pid):
    if pid == PID:
        return True
    try:
        if os.name == 'nt':
            os.waitpid(pid, os.WNOHANG)
        else:
            os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # XXX Does this *always* mean there's a proc?
        return True
    else:
        return True


def _run_fg(cmd, *args):
    argv = [cmd, *args]
    logger.debug('# running: %s', ' '.join(shlex.quote(a) for a in argv))
    return subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
    )


def _run_bg(argv, logfile=None):
    if not argv:
        raise ValueError('missing argv')
    elif isinstance(argv, str):
        if not argv.strip():
            raise ValueError('missing argv')
        cmd = argv
    else:
        cmd = ' '.join(shlex.quote(a) for a in argv)

    if logfile:
        logfile = _quote_shell_str(logfile)
        cmd = f'{cmd} >> {logfile}'
    cmd = f'{cmd} 2>&1'

    logger.debug('# running (background): %s', cmd)
    #subprocess.run(cmd, shell=True)
    subprocess.Popen(
        cmd,
        #creationflags=subprocess.DETACHED_PROCESS,
        #creationflags=subprocess.CREATE_NEW_CONSOLE,
        #creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        shell=True,
    )


class MetadataError(ValueError):
    def __init__(self, msg=None):
        super().__init__(msg or 'metadata-related error')


class MissingMetadataError(MetadataError):
    def __init__(self, msg=None, source=None):
        if not msg:
            msg = 'missing metadata'
            if source:
                msg = f'{msg} (in {{source}})'
        super().__init__(msg.format(source=source))


class InvalidMetadataError(MetadataError):
    def __init__(self, msg=None, source=None):
        if not msg:
            msg = 'invalid metadata'
            if source:
                msg = f'{msg} (in {{source}})'
        super().__init__(msg.format(source=source))


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

    def as_jsonable(self, *, withextra=False):
        fields = self.FIELDS
        if not fields:
            fields = [f for f in vars(self) if not f.startswith('_')]
        elif withextra:
            fields.extend((getattr(self, '_extra', None) or {}).keys())
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

    def save(self, resfile, *, withextra=False):
        if isinstance(resfile, str):
            filename = resfile
            with open(filename, 'w') as resfile:
                return self.save(resfile, withextra=withextra)
        data = self.as_jsonable(withextra=withextra)
        _write_json(data, resfile)


class SSHCommands:

    SSH = shutil.which('ssh')
    SCP = shutil.which('scp')

    def __init__(self, host, port, user, *, ssh=None, scp=None):
        self.host = _check_shell_str(host)
        self.port = int(port)
        if self.port < 1:
            raise ValueError(f'invalid port {self.port}')
        self.user = _check_shell_str(user)

        opts = []
        if self.host == 'localhost':
            opts.extend(['-o', 'StrictHostKeyChecking=no'])
        self._ssh = ssh or self.SSH
        self._ssh_opts = [*opts, '-p', str(self.port)]
        self._scp = scp or self.SCP
        self._scp_opts = [*opts, '-P', str(self.port)]

    def __repr__(self):
        args = (f'{n}={getattr(self, n)!r}'
                for n in 'host port user'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def run(self, cmd, *args):
        conn = f'{self.user}@{self.host}'
        if not os.path.isabs(cmd):
            raise ValueError(f'expected absolute path for cmd, got {cmd!r}')
        return [self._ssh, *self._ssh_opts, conn, cmd, *args]

    def run_shell(self, cmd):
        conn = f'{self.user}@{self.host}'
        return [self._ssh, *self._ssh_opts, conn, *shlex.split(cmd)]

    def push(self, source, target):
        conn = f'{self.user}@{self.host}'
        return [self._scp, *self._scp_opts, '-rp', source, f'{conn}:{target}']

    def pull(self, source, target):
        conn = f'{self.user}@{self.host}'
        return [self._scp, *self._scp_opts, '-rp', f'{conn}:{source}', target]

    def ensure_user_with_agent(self, user):
        raise NotImplementedError


class SSHShellCommands(SSHCommands):

    SSH = 'ssh'
    SCP = 'scp'

    def run(self, cmd, *args):
        return ' '.join(shlex.quote(a) for a in super().run(cmd, *args))

    def run_shell(self, cmd):
        return ' '.join(super().run_shell(cmd))

    def push(self, source, target):
        return ' '.join(super().push(source, target))

    def pull(self, source, target):
        return ' '.join(super().pull(source, target))

    def ensure_user_with_agent(self, user):
        return [
            f'setfacl -m {user}:x $(dirname "$SSH_AUTH_SOCK")',
            f'setfacl -m {user}:rwx "$SSH_AUTH_SOCK"',
            f'# Stop running and re-run this script as the {user} user.',
            f'''exec sudo --login --user {user} --preserve-env='SSH_AUTH_SOCK' "$0" "$@"''',
        ]


class SSHClient(SSHCommands):

    @property
    def commands(self):
        return SSHCommands(self.host, self.port, self.user)

    @property
    def shell_commands(self):
        return SSHShellCommands(self.host, self.port, self.user)

    def check(self):
        return (self.run_shell('true').returncode == 0)

    def run(self, cmd, *args):
        argv = super().run(cmd, *args)
        return _run_fg(*argv)

    def run_shell(self, cmd, *args):
        argv = super().run_shell(cmd, *args)
        return _run_fg(*argv)

    def push(self, source, target):
        argv = super().push(*args)
        return _run_fg(*argv)

    def pull(self, source, target):
        argv = super().push(*args)
        return _run_fg(*argv)

    def read(self, filename):
        if not filename:
            raise ValueError(f'missing filename')
        proc = self.run_shell(f'cat {filename}')
        if proc.returncode != 0:
            return None
        return proc.stdout


##################################
# jobs config

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
# job files

class JobFS(types.SimpleNamespace):
    """The file structure of a job's data."""

    @classmethod
    def from_jobsfs(cls, jobsfs, reqid):
        self = cls(
            f'{jobsfs.requests}/{reqid}',
            f'{jobsfs.requests}/{reqid}',
            f'{jobsfs.work}/{reqid}',
            reqid,
            jobsfs.context,
        )
        return self

    def __init__(self, request, work, result, reqid=None, context='portal'):
        request = FSTree.from_raw(request, name='request')
        work = FSTree.from_raw(work, name='work')
        result = FSTree.from_raw(result, name='result')
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
        if not context:
            context = 'portal'
        elif context not in ('portal', 'bench'):
            raise ValueError(f'unsupported context {context!r}')

        # the request
        request.metadata = f'{request}/request.json'
        # the job
        work.bench_script = f'{work}/run.sh'
        if context == 'portal':
            work.portal_script = f'{work}/send.sh'
            work.pidfile = f'{work}/send.pid'
            work.logfile = f'{work}/job.log'
            work.ssh_okay = f'{work}/ssh.ok'
        elif context == 'bench':
            work.pidfile = f'{work}/job.pid'
            work.logfile = f'{work}/job.log'
        # the results
        result.metadata = f'{result}/results.json'

        super().__init__(
            reqid=reqid,
            context=context,
            request=request,
            work=work,
            result=result,
        )

        # XXX Move these to a subclass?
        if reqid.kind == 'compile-bench':
            request.manifest = f'{request}/benchmarks.manifest'
            #request.pyperformance_config = f'{request}/compile.ini'
            request.pyperformance_config = f'{request}/pyperformance.ini'
            #result.pyperformance_log = f'{result}/run.log'
            result.pyperformance_log = f'{result}/pyperformance.log'
            #result.pyperformance_results = f'{result}/results-data.json.gz'
            result.pyperformance_results = f'{result}/pyperformance-results.json.gz'
            if self.context == 'bench':
                # other directories needed by the job
                work.venv = f'{work}/pyperformance-venv'
                work.scratch_dir = f'{work}/pyperformance-scratch'
                # the results
                # XXX Is this right?
                work.pyperformance_results_glob = f'{work}/*.json.gz'
        else:
            raise ValueError(f'unsupported job kind for {reqid}')

    def __str__(self):
        return str(self.request)

    def __fspath__(self):
        return str(self.request)

    @property
    def jobs(self):
        dirname, reqid = os.path.split(self.request)
        if str(self.reqid) != reqid:
            raise NotImplementedError
        root, requests = os.path.split(dirname)
        if requests != 'REQUESTS':
            raise NotImplementedError
        return JobsFS(root)

    @property
    def bench_script(self):
        return self.work.bench_script

    @property
    def portal_script(self):
        return self.work.portal_script

    @property
    def pidfile(self):
        return self.work.pidfile

    @property
    def logfile(self):
        return self.work.logfile

    @property
    def ssh_okay(self):
        return self.work.ssh_okay

    def copy(self):
        return type(self)(
            str(self.request),
            str(self.work),
            str(self.result),
            self.reqid,
            self.context,
        )


class JobsFS(FSTree):
    """The file structure of the jobs data."""

    JOBFS = JobFS

    @classmethod
    def from_user(cls, user, context='portal'):
        return cls(f'/home/{user}/BENCH', context)

    def __init__(self, root='~/BENCH', context='portal'):
        if not root:
            root = '~/BENCH'
        super().__init__(root)
        self.context = context or 'portal'

        self.requests = FSTree(f'{root}/REQUESTS')
        if context == 'portal':
            self.requests.current = f'{self.requests}/CURRENT'

        self.work = FSTree(self.requests.root)
        self.results = FSTree(self.requests.root)

        if not context:
            context = 'portal'
        elif context not in ('portal', 'bench'):
            raise ValueError(f'unsupported context {context!r}')

        if context == 'portal':
            self.queue = FSTree(f'{self.requests}/queue.json')
            self.queue.data = f'{self.requests}/queue.json'
            self.queue.lock = f'{self.requests}/queue.lock'
            self.queue.log = f'{self.requests}/queue.log'
        elif context == 'bench':
            # the local git repositories used by the job
            self.repos = FSTree(f'{self}/repositories')
            self.repos.cpython = f'{self.repos}/cpython'
            self.repos.pyperformance = f'{self.repos}/pyperformance'
            self.repos.pyston_benchmarks = f'{self.repos}/pyston-benchmarks'
        else:
            raise ValueError(f'unsupported context {context!r}')

    def __str__(self):
        return self.root

    def resolve_request(self, reqid):
        return self.JOBFS.from_jobsfs(self, reqid)

    def copy(self):
        return type(self)(self.root, self.context)


##################################
# workers

class Worker:

    @classmethod
    def from_config(cls, cfg, JobsFS=JobsFS):
        fs = JobsFS.from_user(cfg.bench_user, 'bench')
        ssh = SSHClient(cfg.send_host, cfg.send_port, cfg.bench_user)
        return cls(fs, ssh)

    def __init__(self, fs, ssh):
        self._fs = fs
        self._ssh = ssh

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'fs ssh'.split())
        return f'{type(self).__name__}({"".join(args)})'

    @property
    def fs(self):
        return self._fs

    @property
    def ssh(self):
        return self._ssh

    def resolve(self, reqid):
        fs = self._fs.resolve_request(reqid)
        return JobWorker(self, fs)


class JobWorker:

    def __init__(self, worker, fs):
        self._worker = worker
        self._fs = fs

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'worker fs'.split())
        return f'{type(self).__name__}({"".join(args)})'

    @property
    def worker(self):
        return self._worker

    @property
    def fs(self):
        return self._fs

    @property
    def topfs(self):
        return self._worker.fs

    @property
    def ssh(self):
        return self._worker.ssh


##################################
# jobs

class JobsError(RuntimeError):
    MSG = 'a jobs-related problem'

    def __init__(self, msg=None):
        super().__init__(msg or self.MSG)


class NoRunningJobError(JobsError):
    MSG = 'no job is currently running'


class JobError(JobsError):
    MSG = 'job {reqid} has a problem'

    def __init__(self, reqid, msg=None):
        msg = (msg or self.MSG).format(reqid=str(reqid))
        super().__init__(msg)
        self.reqid = reqid


class JobNotRunningError(JobError):
    MSG = 'job {reqid} is not running'


class JobNeverStartedError(JobNotRunningError):
    MSG = 'job {reqid} was never started'


class Job:

    def __init__(self, reqid, fs, worker, cfg):
        if not reqid:
            raise ValueError('missing reqid')
        if not fs:
            raise ValueError('missing fs')
        elif not isinstance(fs, JobFS):
            raise TypeError(f'expected JobFS for fs, got {fs!r}')
        if not worker:
            raise ValueError('missing worker')
        elif not isinstance(worker, JobWorker):
            raise TypeError(f'expected JobWorker for worker, got {worker!r}')
        self._reqid = RequestID.from_raw(reqid)
        self._fs = fs
        self._worker = worker
        self._cfg = cfg
        self._pidfile = PIDFile(fs.pidfile)

    def __repr__(self):
        args = (f'{n}={str(getattr(self, "_"+n))!r}'
                for n in 'reqid fs worker cfg'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __str__(self):
        return str(self._reqid)

    @property
    def reqid(self):
        return self._reqid

    @property
    def fs(self):
        return self._fs

    @property
    def worker(self):
        return self._worker

    @property
    def cfg(self):
        return self._cfg

    @property
    def kind(self):
        return self.reqid.kind

    @property
    def request(self):
        return Request(self._reqid, str(self.fs))

    def load_result(self):
        return Result.load(self._fs.result.metadata)

    def get_status(self, *, fail=True):
        try:
            return Result.read_status(self._fs.result.metadata)
        except FileNotFoundError:
            # XXX Create it?
            if fail:
                raise
            return None
        except MissingMetadataError:
            # XXX Re-create it?
            if fail:
                raise
            return None
        except InvalidMetadataError:
            # XXX Fix it?
            if fail:
                raise
            return None

    def set_status(self, status):
        status = Result.resolve_status(status)
        result = self.load_result()
        result.set_status(status)
        result.save(self._fs.result.metadata, withextra=True)

    def _create(self, kind_kwargs, reqfsattrs, queue_log):
        os.makedirs(self._fs.request.root, exist_ok=True)
        os.makedirs(self._fs.work.root, exist_ok=True)
        os.makedirs(self._fs.result.root, exist_ok=True)

        if self._reqid.kind == 'compile-bench':
            fake = kind_kwargs.pop('_fake', None)
            req = _resolve_bench_compile_request(
                self._reqid,
                self._fs.work.root,
                **kind_kwargs,
            )

            # Write the benchmarks manifest.
            manifest = _build_pyperformance_manifest(req, self._worker.topfs)
            with open(self._fs.request.manifest, 'w') as outfile:
                outfile.write(manifest)

            # Write the config.
            ini = _build_pyperformance_config(req, self._worker.topfs)
            with open(self._fs.request.pyperformance_config, 'w') as outfile:
                ini.write(outfile)

            # Build the script for the commands to execute remotely.
            script = _build_compile_script(req, self._worker.topfs, fake)
        else:
            raise ValueError(f'unsupported job kind in {self._reqid}')

        # Write metadata.
        req.save(self._fs.request.metadata)
        req.result.save(self._fs.result.metadata)

        # Write the commands to execute remotely.
        with open(self._fs.bench_script, 'w') as outfile:
            outfile.write(script)
        os.chmod(self._fs.bench_script, 0o755)

        # Write the commands to execute locally.
        script = self._build_send_script(queue_log, reqfsattrs)
        with open(self._fs.portal_script, 'w') as outfile:
            outfile.write(script)
        os.chmod(self._fs.portal_script, 0o755)

    def _build_send_script(self, queue_log, resfsfields=None, *,
                           hidecfg=False,
                           ):
        reqid = self.reqid
        pfiles = self._fs
        bfiles = self._worker.fs

        jobs_script = _quote_shell_str(JOBS_SCRIPT)

        if not self._cfg.filename:
            raise NotImplementedError(self._cfg)
        cfgfile = _quote_shell_str(self._cfg.filename)
        if hidecfg:
            ssh = SSHShellCommands('$host', '$port', '$benchuser')
            user = '$user'
        else:
            user = _check_shell_str(self._cfg.send_user)
            _check_shell_str(self._cfg.send_host)
            _check_shell_str(self._cfg.bench_user)
            ssh = self._worker.ssh.shell_commands

        queue_log = _quote_shell_str(queue_log)

        reqdir = _quote_shell_str(pfiles.request.root)
        results_meta = _quote_shell_str(pfiles.result.metadata)
        pidfile = _quote_shell_str(pfiles.pidfile)

        _check_shell_str(bfiles.request.root)
        _check_shell_str(bfiles.bench_script)
        _check_shell_str(bfiles.result.root)
        _check_shell_str(bfiles.result.metadata)

        ensure_user = ssh.ensure_user_with_agent(user)
        ensure_user = '\n                '.join(ensure_user)

        resfiles = []
        for attr in resfsfields or ():
            pvalue = getattr(pfiles.result, attr)
            pvalue = _quote_shell_str(pvalue)
            bvalue = getattr(bfiles.result, attr)
            _check_shell_str(bvalue)
            resfiles.append((bvalue, pvalue))
        resfiles = (ssh.pull(s, t) for s, t in resfiles)
        resfiles = '\n                '.join(resfiles)

        push = ssh.push
        pull = ssh.pull
        ssh = ssh.run_shell

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
            echo "(the "'"'"{reqid.kind}"'"'" job, {reqid}, has started)"
            echo

            user=$(jq -r '.send_user' {cfgfile})
            if [ "$USER" != '{user}' ]; then
                echo "(switching users from $USER to {user})"
                echo
                {ensure_user}
            fi
            host=$(jq -r '.send_host' {cfgfile})
            port=$(jq -r '.send_port' {cfgfile})

            exitcode=0
            if {ssh(f'test -e {bfiles.request}')}; then
                >&2 echo "request {reqid} was already sent"
                exitcode=1
            else
                ( set -x

                # Set up before running.
                {ssh(f'mkdir -p {bfiles.request}')}
                {push(f'{reqdir}/*', bfiles.request)}
                {ssh(f'mkdir -p {bfiles.result}')}

                # Run the request.
                {ssh(bfiles.bench_script)}
                exitcode=$?

                # Finish up.
                # XXX Push from the bench host in run.sh instead of pulling here?
                {pull(bfiles.result.metadata, results_meta)}
                {resfiles}
                )
            fi

            # Unstage the request.
            {sys.executable} {jobs_script} internal-finish-run -v --config {cfgfile} {reqid}

            # Mark the script as complete.
            echo
            echo "(the "'"'"{reqid.kind}"'"'" job, {reqid} has finished)"
            #rm -f {pidfile}

            # Trigger the next job.
            {sys.executable} {jobs_script} internal-run-next -v --config {cfgfile} --logfile {queue_log} &

            exit $exitcode
        '''[1:-1])

    def check_ssh(self, *, onunknown=None, fail=True, _print=print):
        filename = self._fs.ssh_okay
        if onunknown is None:
            save = False
        elif isinstance(onunknown, str):
            if onunknown == 'save':
                save = True
            elif onunknown == 'wait':
                _wait_for_file(self._fs.ssh_okay)
                save = False
            elif onunknown.startswith('wait:'):
                _, _, timeout = onunknown.partition(':')
                if not timeout or not timeout.isdigit():
                    raise ValueError(f'invalid onunknown timeout ({onunknown})')
                _wait_for_file(filename, timeout=int(timeout))
                save = False
            else:
                raise NotImplementedError(repr(onunknown))
        else:
            raise TypeError(f'expected str, got {onunknown!r}')
        text = _read_file(filename, fail=False)
        if text is None:
            okay = self._worker.ssh.check()
            if save:
                with open(filename, 'w') as outfile:
                    outfile.write('0' if okay else '1')
                    _print(file=outfile)
        else:
            text = text.strip()
            if text == '0':
                okay = True
            elif text == '1':
                okay = False
            else:
                if fail:
                    raise Exception(f'invalid ssh_okay {text!r} in {filename}')
                okay = False
        if fail and not okay:
            raise ConnectionRefusedError('SSH failed (is your SSH agent running and up-to-date?)')
        return okay

    def run(self, *, background=False):
        self.check_ssh(onunknown='save')
        if background:
            script = _quote_shell_str(self._fs.portal_script)
            logfile = _quote_shell_str(self._fs.logfile)
            _run_bg(f'{script} > {logfile}')
            return 0
        else:
            proc = subprocess.run([self.fs.portal_script])
            return proc.returncode

    def get_pid(self):
        return self._pidfile.read()

    def kill(self):
        pid = self.get_pid()
        if pid:
            logger.info('# killing PID %s', pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.warn(f'job {self._reqid} no longer (PID: {pid})')
        # Kill the worker process, if running.
        text = self._worker.ssh.read(self._worker.fs.pidfile)
        if text and text.isdigit():
            self._worker.ssh.run_shell(f'kill {text}')

    def wait_until_started(self, *, checkssh=False):
        # XXX Add a timeout?
        pid = self.get_pid()
        while pid is None:
            status = self.get_status()
            if status in Result.FINISHED:
                raise JobNeverStartedError(reqid)
            if checkssh and os.path.exists(self.fs.ssh_okay):
                break
            time.sleep(0.01)
            pid = self.get_pid()
        return pid

    def attach(self, lines=None):
        pid = self.wait_until_started()
        try:
            if pid:
                tail_file(self.fs.logfile, lines, follow=pid)
            elif lines:
                tail_file(self.fs.logfile, lines, follow=False)
        except KeyboardInterrupt:
            # XXX Prompt to cancel the job?
            return

    def cancel(self, *, ifstatus=None):
        if ifstatus is not None:
            if job.get_status() not in (Result.STATUS.CREATED, ifstatus):
                return
        self.kill()
        # XXX Try to download the results directly?
        self.set_status('canceled')

    def close(self):
        result = self.load_result()
        result.close()
        result.save(self._fs.result.metadata, withextra=True)

    def render(self, fmt=None):
        reqfile = self._fs.request.metadata
        resfile = self._fs.result.metadata
        if not fmt:
            fmt = 'summary'
        elif fmt in ('reqfile', 'resfile'):
            filename = reqfile if fmt == 'reqfile' else resfile
            yield f'(from {filename})'
            yield ''
            text = _read_file(filename)
            for line in text.splitlines():
                yield f'  {line}'
            return

        reqfs_fields = [
            'bench_script',
            'portal_script',
            'ssh_okay',
        ]
        resfs_fields = [
            'pidfile',
            'logfile',
        ]
        if self.kind is RequestID.KIND.BENCHMARKS:
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
        req = req_cls.load(reqfile)
        res = res_cls.load(resfile)
        pid = PIDFile(self._fs.pidfile).read()
        try:
            staged = _get_staged_request(self._fs.jobs)
        except StagedRequestError:
            isstaged = False
        else:
            isstaged = (self.reqid == staged)
        if self.check_ssh(fail=False):
            ssh_okay = 'yes'
        elif os.path.exists(self._fs.ssh_okay):
            ssh_okay = 'no'
        else:
            ssh_okay = '???'

        if fmt == 'summary':
            yield f'Request {self.reqid}:'
            yield f'  {"kind:":22} {req.kind}'
            yield f'  {"user:":22} {req.user}'
            if pid:
                yield f'  {"PID:":22} {pid}'
            yield f'  {"status:":22} {res.status or "(created)"}'
            yield f'  {"is staged:":22} {isstaged}'
            yield ''
            yield 'Details:'
            for field in req_cls.FIELDS:
                if field in ('id', 'reqid', 'kind', 'user', 'date', 'datadir'):
                    continue
                value = getattr(req, field)
                if isinstance(value, str) and value.strip() != value:
                    value = repr(value)
                yield f'  {field + ":":22} {value}'
            yield f'  {"ssh okay:":22} {ssh_okay}'
            yield ''
            yield 'History:'
            for st, ts in res.history:
                yield f'  {st + ":":22} {ts:%Y-%m-%d %H:%M:%S}'
            yield ''
            yield 'Request files:'
            yield f'  {"data root:":22} {_render_file(req.reqdir)}'
            yield f'  {"metadata:":22} {_render_file(self._fs.request.metadata)}'
            for field in reqfs_fields:
                value = getattr(self._fs.request, field, None)
                if value is None:
                    value = getattr(self._fs.work, field, None)
                yield f'  {field + ":":22} {_render_file(value)}'
            yield ''
            yield 'Result files:'
            yield f'  {"data root:":22} {_render_file(self._fs.result)}'
            yield f'  {"metadata:":22} {_render_file(self._fs.result.metadata)}'
            for field in resfs_fields:
                value = getattr(self._fs.result, field, None)
                if value is None:
                    value = getattr(self._fs.work, field, None)
                yield f'  {field + ":":22} {_render_file(value)}'
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')


class Jobs:

    FS = JobsFS

    def __init__(self, cfg):
        self._cfg = cfg
        self._fs = self.FS(cfg.data_dir)
        self._worker = Worker.from_config(cfg, self.FS)

    def __str__(self):
        return self.fs.root

    @property
    def cfg(self):
        return self._cfg

    @property
    def fs(self):
        """Files on the portal host."""
        return self._fs.copy()

    @property
    def queue(self):
        try:
            return self._queue
        except AttributeError:
            self._queue = JobQueue.from_fstree(self.fs)
            return self._queue

    def iter_all(self):
        for name in os.listdir(str(self._fs.requests)):
            reqid = RequestID.parse(name)
            if not reqid:
                continue
            yield self._get(reqid)

    def _get(self, reqid):
        return Job(
            reqid,
            self._fs.resolve_request(reqid),
            self._worker.resolve(reqid),
            self._cfg,
        )

    def get_current(self):
        reqid = _get_staged_request(self._fs)
        if not reqid:
            return None
        return self.get(reqid)

    def get(self, reqid):
        return self._get(reqid)

    def create(self, reqid, kind_kwargs=None, reqfsattrs=None):
        if kind_kwargs is None:
            kind_kwargs = {}

        job = self._get(reqid)
        job._create(kind_kwargs, reqfsattrs, self._fs.queue.log)
        return job

    def activate(self, reqid):
        logger.debug('# staging request')
        stage_request(reqid, self.fs)
        logger.debug('# done staging request')
        job = self._get(reqid)
        job.set_status('active')
        return job

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
        _run_bg(
            [
                sys.executable, '-u', JOBS_SCRIPT, '-v',
                'internal-run-next',
                '--config', cfgfile,
                #'--logfile', self._fs.queue.log,
            ],
            logfile=self._fs.queue.log,
        )

    def cancel_current(self, reqid=None, *, ifstatus=None):
        if not reqid:
            job = self.get_current()
            if job is None:
                raise NoRunningJobError()
        else:
            job = self._get(reqid)
        job.cancel(ifstatus=ifstatus)

        logger.info('# unstaging request %s', reqid)
        try:
            unstage_request(job.reqid, self._fs)
        except RequestNotStagedError:
            pass
        logger.info('# done unstaging request')
        return job

    def finish_successful(self, reqid):
        logger.info('# unstaging request %s', reqid)
        try:
            unstage_request(reqid, self._fs)
        except RequestNotStagedError:
            pass
        logger.info('# done unstaging request')

        job = self._get(reqid)
        job.close()
        return job


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
    jobs = sorted(jobs, key=(lambda j: j.reqid))
    selection = _get_slice(criteria[0])
    yield from jobs[selection]


##################################
# the current job

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
        reqdir = os.readlink(pfiles.requests.current)
    except FileNotFoundError:
        return None
    requests, reqidstr = os.path.split(reqdir)
    if requests != pfiles.requests.root:
        return StagedRequestResolveError(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    reqid = RequestID.parse(reqidstr)
    if not reqid:
        return StagedRequestResolveError(None, reqdir, 'invalid', f'{reqidstr!r} not a request ID')
    if not os.path.exists(reqdir):
        return StagedRequestResolveError(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        return StagedRequestResolveError(reqid, reqdir, 'malformed', 'target is not a directory')
    reqfs = pfiles.resolve_request(reqid)
    # Check if the request is still running.
    status = Result.read_status(str(reqfs.result.metadata), fail=False)
    if not status or status in ('created', 'pending'):
        logger.error(f'request {reqid} was set as the current job incorrectly; unsetting...')
        os.unlink(pfiles.requests.current)
        reqid = None
    elif status not in ('active', 'running'):
        logger.warn(f'request {reqid} is still "current" even though it finished; unsetting...')
        os.unlink(pfiles.requests.current)
        reqid = None
    elif not PIDFile(str(reqfs.pidfile)).read(orphaned='ignore'):
        logger.warn(f'request {reqid} is no longer running; unsetting as the current job...')
        os.unlink(pfiles.requests.current)
        reqid = None
    # XXX Do other checks?
    return reqid


def stage_request(reqid, pfiles):
    jobfs = pfiles.resolve_request(reqid)
    status = Result.read_status(jobfs.result.metadata, fail=False)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
    try:
        os.symlink(jobfs.request, pfiles.requests.current)
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
    os.unlink(pfiles.requests.current)


##################################
# the job queue

class JobQueueError(Exception):
    MSG = 'some problem with the job queue'

    def __init__(self, msg=None):
        super().__init__(msg or self.MSG)


class JobQueuePausedError(JobQueueError):
    MSG = 'job queue paused'


class JobQueueNotPausedError(JobQueueError):
    MSG = 'job queue not paused'


class JobQueueEmptyError(JobQueueError):
    MSG = 'job queue is empty'


class QueuedJobError(JobError, JobQueueError):
    MSG = 'some problem with job {reqid}'


class JobNotQueuedError(QueuedJobError):
    MSG = 'job {reqid} is not in the queue'


class JobAlreadyQueuedError(QueuedJobError):
    MSG = 'job {reqid} is already in the queue'


class JobQueueData(types.SimpleNamespace):

    def __init__(self, jobs, paused):
        super().__init__(
            jobs=jobs,
            paused=paused,
        )

    def __iter__(self):
        yield from self.jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        return self.jobs[idx]


class JobQueueSnapshot(JobQueueData):

    def __init__(self, jobs, paused, locked, datafile, lockfile, logfile):
        super().__init__(tuple(jobs), paused)
        self.locked = locked
        self.datafile = datafile
        self.lockfile = lockfile
        self.logfile = logfile

    def read_log(self):
        try:
            logfile = open(self.logfile)
        except FileNotFoundError:
            return
        with logfile:
            yield from LogSection.read_logfile(logfile)


class JobQueue:

    # XXX Add maxsize.

    @classmethod
    def from_config(cls, cfg):
        fs = JobsFS(self.cfg.data_dir)
        return cls.from_jobsfs(fs)

    @classmethod
    def from_fstree(cls, fs):
        if isinstance(fs, str):
            fs = JobsFS(fs)
        elif not isinstance(fs, JobsFS):
            raise TypeError(f'expected JobsFS, got {fs!r}')
        self = cls(
            datafile=fs.queue.data,
            lockfile=fs.queue.lock,
            logfile=fs.queue.log,
        )
        return self

    def __init__(self, datafile, lockfile, logfile):
        _validate_string(datafile, 'datafile')
        _validate_string(lockfile, 'lockfile')
        _validate_string(logfile, 'logfile')

        self._datafile = datafile
        self._lock = LockFile(lockfile)
        self._logfile = logfile
        self._data = None

    def __iter__(self):
        with self._lock:
            data = self._load()
        yield from data.jobs

    def __len__(self):
        with self._lock:
            data = self._load()
        return len(data.jobs)

    def __getitem__(self, idx):
        with self._lock:
            data = self._load()
        return data.jobs[idx]

    def _load(self):
        text = _read_file(self._datafile, fail=False)
        data = (json.loads(text) if text else None) or {}
        # Normalize.
        fixed = False
        if 'paused' not in data:
            data['paused'] = False
            fixed = True
        elif data['paused'] not in (True, False):
            data['paused'] = bool(data['paused'])
            fixed = True
        if 'jobs' not in data:
            data['jobs'] = []
            fixed = True
        else:
            jobs = [RequestID.from_raw(v) for v in data['jobs']]
            if any(not j for j in jobs):
                logger.warn('job queue at %s has bad entries', self._datafile)
                fixed = True
            data['jobs'] = [r for r in jobs if r]
        # Save and return the data.
        if fixed:
            with open(self._datafile, 'w') as outfile:
                json.dump(data, outfile, indent=4)
        data = self._data = JobQueueData(**data)
        return data

    def _save(self, data=None):
        if data is None:
            data = self._data
        elif isinstance(data, types.SimpleNamespace):
            if data is not self._data:
                raise NotImplementedError
            data = dict(vars(data))
        self._data = None
        if not data:
            # Nothing to save.
            return
        # Validate.
        if 'paused' not in data or data['paused'] not in (True, False):
            raise NotImplementedError
        if 'jobs' not in data:
            raise NotImplementedError
        elif any(not isinstance(v, RequestID) for v in data['jobs']):
            raise NotImplementedError
        else:
            data['jobs'] = [str(req) for req in data['jobs']]
        # Write to the queue file.
        with open(self._datafile, 'w') as outfile:
            _write_json(data, outfile)

    @property
    def snapshot(self):
        data = self._load()
        try:
            pid = self._lock.read()
        except OrphanedPIDFileError as exc:
            locked = (exc.pid, False)
        except InvalidPIDFileError as exc:
            locked = (exc.text, None)
        else:
            locked = (pid, bool(pid))
        return JobQueueSnapshot(
            datafile=self._datafile,
            lockfile=self._lock.filename,
            logfile=self._logfile,
            locked=locked,
            **vars(data),
        )

    @property
    def paused(self):
        with self._lock:
            data = self._load()
            return data.paused

    def pause(self):
        with self._lock:
            data = self._load()
            if data.paused:
                raise JobQueuePausedError()
            data.paused = True
            self._save(data)

    def unpause(self):
        with self._lock:
            data = self._load()
            if not data.paused:
                raise JobQueueNotPausedError()
            data.paused = False
            self._save(data)

    def push(self, reqid):
        with self._lock:
            data = self._load()
            if reqid in data.jobs:
                raise JobAlreadyQueuedError(reqid)

            data.jobs.append(reqid)
            self._save(data)
        return len(data.jobs)

    def pop(self, *, forceifpaused=False):
        with self._lock:
            data = self._load()
            if data.paused:
                if not forceifpaused:
                    raise JobQueuePausedError()
            if not data.jobs:
                raise JobQueueEmptyError()

            reqid = data.jobs.pop(0)
            self._save(data)
        return reqid

    def unpop(self, reqid):
        with self._lock:
            data = self._load()
            if data.jobs and data.jobs[0] == reqid:
                # XXX warn?
                return
            data.jobs.insert(0, reqid)
            self._save(data)

    def move(self, reqid, position, relative=None):
        with self._lock:
            data = self._load()
            if reqid not in data.jobs:
                raise JobNotQueuedError(reqid)

            old = data.jobs.index(reqid)
            if relative == '+':
                idx = min(0, old - position)
            elif relative == '-':
                idx = max(len(data.jobs), old + position)
            else:
                idx = position - 1
            data.jobs.insert(idx, reqid)
            if idx < old:
                old += 1
            del data.jobs[old]

            self._save(data)
        return idx + 1

    def remove(self, reqid):
        with self._lock:
            data = self._load()

            if reqid not in data.jobs:
                raise JobNotQueuedError(reqid)

            data.jobs.remove(reqid)
            self._save(data)

    def read_log(self):
        try:
            logfile = open(self._logfile)
        except FileNotFoundError:
            return
        with logfile:
            yield from LogSection.read_logfile(logfile)


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
        elif not raw:
            raise NotImplementedError(raw)
        else:
            try:
                args = tuple(raw)
            except TypeError:
                raise NotImplementedError(repr(raw))
            return cls(*args)

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
        dt, _ = get_utc_datetime(self.timestamp)
        return dt


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

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
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
    def resolve_status(cls, status):
        try:
            return cls._STATUS_BY_VALUE[status]
        except KeyError:
            raise ValueError(f'unsupported status {status!r}')

    @classmethod
    def read_status(cls, metafile, *, fail=True):
        missing = None
        text = _read_file(metafile, fail=fail)
        if text is None:
            return None
        elif not text:
            missing = True
        elif text:
            try:
                data = json.loads(text)
            except json.decoder.JSONDecodeError:
                missing = False
            else:
                if 'status' not in data:
                    missing = True
                else:
                    status = data['status']
                    try:
                        return cls._STATUS_BY_VALUE[status]
                    except KeyError:
                        missing = False
        if not fail:
            return None
        elif missing:
            raise MissingMetadataError(source=metafile)
        else:
            raise InvalidMetadataError(source=metafile)

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
            status = self.resolve_status(status)
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
                    date, _ = get_utc_datetime(date)
                elif isinstance(date, int):
                    date, _ = get_utc_datetime(date)
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
        status = self.resolve_status(status)
        if self.history[-1][0] is self.CLOSED:
            raise Exception(f'req {self.reqid} is already closed')
        # XXX Make sure it is the next possible status?
        self.history.append(
            (status, datetime.datetime.now(datetime.timezone.utc)),
        )
        self.status = None if status is self.STATUS.CREATED else status
        if status in self.FINISHED:
            self.close()

    def close(self):
        if self.history[-1][0] is self.CLOSED:
            # XXX Fail?
            return
        self.history.append(
            (self.CLOSED, datetime.datetime.now(datetime.timezone.utc)),
        )

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
        if self.status is None:
            data['status'] = self.STATUS.CREATED
        data['reqid'] = str(data['reqid'])
        data['history'] = [(st, d.isoformat() if d else None)
                           for st, d in data['history']]
        return data


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

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
        for field in ['pyperformance_results', 'pyperformance_results_orig']:
            if not data[field]:
                del data[field]
        data['reqid'] = str(data['reqid'])
        return data


def _resolve_bench_compile_request(reqid, workdir, remote, revision, branch,
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
        datadir=workdir,
        # XXX Add a "commit" field and use "tag or branch" for ref.
        ref=commit,
        remote=remote,
        branch=branch,
        benchmarks=benchmarks or None,
        optimize=bool(optimize),
        debug=bool(debug),
    )
    return meta


def _build_pyperformance_manifest(req, bfiles):
    return textwrap.dedent(f'''
        [includes]
        <default>
        {bfiles.repos.pyston_benchmarks}/benchmarks/MANIFEST
    '''[1:-1])


def _build_pyperformance_config(req, bfiles):
    cpython = bfiles.repos.cpython
    bfiles = bfiles.resolve_request(req.id)
    cfg = configparser.ConfigParser()

    cfg['config'] = {}
    cfg['config']['json_dir'] = bfiles.result.root
    cfg['config']['debug'] = str(req.debug)
    # XXX pyperformance should be looking in [scm] for this.
    cfg['config']['git_remote'] = req.remote

    cfg['scm'] = {}
    cfg['scm']['repo_dir'] = cpython
    cfg['scm']['git_remote'] = req.remote
    cfg['scm']['update'] = 'True'

    cfg['compile'] = {}
    cfg['compile']['bench_dir'] = bfiles.work.scratch_dir
    cfg['compile']['pgo'] = str(req.optimize)
    cfg['compile']['lto'] = str(req.optimize)
    cfg['compile']['install'] = 'True'

    cfg['run_benchmark'] = {}
    cfg['run_benchmark']['manifest'] = bfiles.request.manifest
    cfg['run_benchmark']['benchmarks'] = ','.join(req.benchmarks or ())
    cfg['run_benchmark']['system_tune'] = 'True'
    cfg['run_benchmark']['upload'] = 'False'

    return cfg


def _build_compile_script(req, bfiles, fake=None):
    fakedelay = FAKE_DELAY
    if fake is False or fake is None:
        fake = (None, None)
    elif fake is True:
        fake = (0, None)
    elif isinstance(fake, (int, str)):
        fake = (fake, None)
    exitcode, fakedelay = fake
    if fakedelay is None:
        fakedelay = FAKE_DELAY
    else:
        fakedelay = _ensure_int(fakedelay, min=0)
        if exitcode is None:
            exitcode = 0
        elif exitcode == '':
            logger.warn(f'fakedelay ({fakedelay}) will not be used')
    if exitcode is None:
        exitcode = ''
    elif exitcode != '':
        exitcode = _ensure_int(exitcode, min=0)
        logger.warn('we will pretend pyperformance will run with exitcode %s', exitcode)
    python = 'python3.9'  # On the bench host.
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

    cpython_repo = _quote_shell_str(bfiles.repos.cpython)
    pyperformance_repo = _quote_shell_str(bfiles.repos.pyperformance)
    pyston_benchmarks_repo = _quote_shell_str(bfiles.repos.pyston_benchmarks)

    bfiles = bfiles.resolve_request(req.id)
    _check_shell_str(bfiles.work.pidfile)
    _check_shell_str(bfiles.work.logfile)
    _check_shell_str(bfiles.work.scratch_dir)
    _check_shell_str(bfiles.request.pyperformance_config)
    _check_shell_str(bfiles.result.pyperformance_log)
    _check_shell_str(bfiles.result.metadata)
    _check_shell_str(bfiles.result.pyperformance_results)
    _check_shell_str(bfiles.work.pyperformance_results_glob)

    _check_shell_str(python)

    # XXX Kill any zombie job processes?

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # This script runs only on the bench host.

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        #####################
        # Mark the result as running.

        echo "$$" > {bfiles.work.pidfile}

        status=$(jq -r '.status' {bfiles.result.metadata})
        if [ "$status" != 'active' ]; then
            2>&1 echo "ERROR: expected active status, got $status"
            2>&1 echo "       (see {bfiles.result.metadata})"
            exit 1
        fi

        ( set -x
        jq --arg date $(date -u -Iseconds) '.history += [["running", $date]]' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
        mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}
        )

        #####################
        # Ensure the dependencies.

        if [ ! -e {cpython_repo} ]; then
            ( set -x
            git clone https://github.com/python/cpython {cpython_repo}
            )
        fi
        if [ ! -e {pyperformance_repo} ]; then
            ( set -x
            git clone https://github.com/python/pyperformance {pyperformance_repo}
            )
        fi
        if [ ! -e {pyston_benchmarks_repo} ]; then
            ( set -x
            git clone https://github.com/pyston/python-macrobenchmarks {pyston_benchmarks_repo}
            )
        fi

        #####################
        # Get the repos are ready for the requested remotes and revisions.

        remote='{req.cpython.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {cpython_repo} remote add {req.cpython.remote} {req.cpython.url}
            git -C {cpython_repo} fetch --tags {req.cpython.remote}
            )
        fi
        # Get the upstream tags, just in case.
        ( set -x
        git -C {cpython_repo} fetch --tags origin
        )
        branch='{maybe_branch}'
        if [ -n "$branch" ]; then
            if ! ( set -x
                git -C {cpython_repo} checkout -b {req.branch or '$branch'} --track {req.cpython.remote}/{req.branch or '$branch'}
            ); then
                echo "It already exists; resetting to the right target."
                ( set -x
                git -C {cpython_repo} checkout {req.branch or '$branch'}
                git -C {cpython_repo} reset --hard {req.cpython.remote}/{req.branch or '$branch'}
                )
            fi
        fi

        remote='{req.pyperformance.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {pyperformance_repo} remote add {req.pyperformance.remote} {req.pyperformance.url}
            )
        fi
        ( set -x
        git -C {pyperformance_repo} fetch --tags {req.pyperformance.remote}
        git -C {pyperformance_repo} checkout {req.pyperformance.fullref}
        )

        remote='{req.pyston_benchmarks.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {pyston_benchmarks_repo} remote add {req.pyston_benchmarks.remote} {req.pyston_benchmarks.url}
            )
        fi
        ( set -x
        git -C {pyston_benchmarks_repo} fetch --tags {req.pyston_benchmarks.remote}
        git -C {pyston_benchmarks_repo} checkout {req.pyston_benchmarks.fullref}
        )

        #####################
        # Run the benchmarks.

        ( set -x
        mkdir -p {bfiles.work.scratch_dir}
        )

        echo "running the benchmarks..."
        echo "(logging to {bfiles.work.logfile})"
        exitcode='{exitcode}'
        if [ -n "$exitcode" ]; then
            ( set -x
            sleep {fakedelay}
            touch {bfiles.work.logfile}
            touch {bfiles.request}/pyperformance-dummy-results.json.gz
            )
        else
            ( set -x
            MAKEFLAGS='-j{numjobs}' \\
                {python} {pyperformance_repo}/dev.py compile \\
                {bfiles.request.pyperformance_config} \\
                {req.ref} {maybe_branch} \\
                2>&1 | tee {bfiles.work.logfile}
            )
            exitcode=$?
        fi

        #####################
        # Record the results.

        if [ -e {bfiles.work.logfile} ]; then
            ln -s {bfiles.work.logfile} {bfiles.result.pyperformance_log}
        fi

        results=$(2>/dev/null ls {bfiles.work.pyperformance_results_glob})
        results_name=$(2>/dev/null basename $results)

        echo "saving results..."
        if [ $exitcode -eq 0 -a -n "$results" ]; then
            ( set -x
            jq '.status = "success"' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}

            jq --arg results "$results" '.pyperformance_data_orig = $results' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}

            jq --arg date $(date -u -Iseconds) '.history += [["success", $date]]' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}
            )
        else
            ( set -x
            jq '.status = "failed"' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}

            jq --arg date $(date -u -Iseconds) '.history += [["failed", $date]]' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}
            )
        fi

        if [ -n "$results" -a -e "$results" ]; then
            ( set -x
            ln -s $results {bfiles.result.pyperformance_results}
            )
        fi

        echo "...done!"

        #rm -f {bfiles.work.pidfile}
    '''[1:-1])


##################################
# commands

def cmd_list(jobs, selections=None):
#    requests = (RequestID.parse(n) for n in os.listdir(jobs.fs.requests.root))
    alljobs = list(jobs.iter_all())
    total = len(alljobs)
    selected = list(select_jobs(alljobs, selections))
    print(f'{"request ID".center(48)} {"status".center(10)} {"created".center(19)}')
    print(f'{"-"*48} {"-"*10} {"-"*19}')
    for job in selected:
        reqid = job.reqid
        status = job.get_status(fail=False)
        print(f'{reqid!s:48} {status or "???":10} {reqid.date:%Y-%m-%d %H:%M:%S}')
        #for line in job.render(fmt='row'):
        #    print(line)
    logger.info('')
    if len(selected) == total:
        logger.info('(total: %s)', total)
    else:
        logger.info('(matched: %s)', len(selected))
        logger.info('(total:   %s)', total)


def cmd_show(jobs, reqid=None, fmt=None, *, lines=None):
    if reqid:
        job = jobs.get(reqid)
    else:
        job = jobs.get_current()
        if not job:
            # XXX Use the last finished?
            logger.error('no job currently running')
            sys.exit(1)

    for line in job.render(fmt=fmt):
        print(line)

    if lines:
        tail_file(job.fs.logfile, lines, follow=False)


def cmd_request_compile_bench(jobs, reqid, revision, *,
                              remote=None,
                              branch=None,
                              benchmarks=None,
                              optimize=False,
                              debug=False,
                              _fake=None,
                              ):
    if not reqid:
        raise NotImplementedError
    assert reqid.kind == 'compile-bench', reqid
    reqroot = jobs.fs.resolve_request(reqid).request.root
    logger.info('generating request files in %s...', reqroot)
    job = jobs.create(
        reqid,
        kind_kwargs=dict(
            revision=revision,
            remote=remote,
            branch=branch,
            benchmarks=benchmarks,
            optimize=optimize,
            debug=debug,
            _fake=_fake,
        ),
        reqfsattrs=['pyperformance_results', 'pyperformance_log'],
    )
    logger.info('...done (generating request files)')
    return job


def cmd_copy(jobs, reqid=None):
    raise NotImplementedError


def cmd_remove(jobs, reqid):
    raise NotImplementedError


def cmd_run(jobs, reqid, *, copy=False, force=False):
    if copy:
        raise NotImplementedError
    if force:
        raise NotImplementedError

    if not reqid:
        raise NotImplementedError

    if not jobs.queue.paused:
        cmd_queue_push(jobs, reqid)
    else:
        job = _cmd_run(jobs, reqid)
        job.check_ssh(onunknown='wait:3')


def _cmd_run(jobs, reqid):
    # Try staging it directly.
    try:
        job = jobs.activate(reqid)
    except RequestAlreadyStagedError as exc:
        # XXX Offer to clear CURRENT?
        logger.error('%s', exc)
        sys.exit(1)
    except Exception:
        logger.error('could not stage request')
        logger.info('')
        job = jobs.get(reqid)
        job.set_status('failed')
        raise  # re-raise
    else:
        job.run(background=True)
        return job


def cmd_attach(jobs, reqid=None, *, lines=None):
    if not reqid:
        job = jobs.get_current()
        if not job:
            logger.error('no current request to attach')
            sys.exit(1)
    else:
        job = jobs.get(reqid)
    job.wait_until_started(checkssh=True)
    job.check_ssh()
    try:
        job.attach(lines)
    except JobNeverStartedError:
        logger.warn('job not started')


def cmd_cancel(jobs, reqid=None, *, _status=None):
    if not reqid:
        try:
            job = current = jobs.cancel_current(ifstatus=_status)
        except NoRunningJobError:
            logger.error('no current request to cancel')
            sys.exit(1)
    else:
        current = jobs.get_current()
        if current and reqid == current.reqid:
            try:
                job = jobs.cancel_current(current.reqid, ifstatus=_status)
            except NoRunningJobError:
                logger.warn('job just finished')
        else:
            cmd_queue_remove(jobs, reqid)
            job = jobs.get(reqid)
            job.cancel(ifstatus=_status)

    logger.info('')
    logger.info('Results:')
    # XXX Show something better?
    for line in job.render(fmt='resfile'):
        logger.info(line)

    if current:
        jobs.ensure_next()


# internal
def cmd_finish_run(jobs, reqid):
    job = jobs.finish_successful(reqid)

    logger.info('')
    logger.info('Results:')
    # XXX Show something better?
    for line in job.render(fmt='resfile'):
        logger.info(line)


# internal
def cmd_run_next(jobs):
    logentry = LogSection.from_title('Running next queued job')
    print()
    for line in logentry.render():
        print(line)
    print()

    try:
        reqid = jobs.queue.pop()
    except JobQueuePausedError:
        logger.info('done (job queue is paused)')
    except JobQueueEmptyError:
        logger.info('done (job queue is empty)')
        return

    try:
        try:
            job = jobs.get(reqid)
            status = job.get_status()
        except Exception:
            logger.error('could not load results metadata')
            logger.warning('%s status could not be updated (to "failed")', reqid)
            logger.error('')
            traceback.print_exc()
            logger.info('')
            logger.info('trying next job...')
            cmd_run_next(jobs)
            return

        if not status:
            logger.warn('queued request (%s) not found', reqid)
            logger.info('trying next job...')
            cmd_run_next(jobs)
            return
        elif status is not Result.STATUS.PENDING:
            logger.warn('expected "pending" status for queued request %s, got %r', reqid, status)
            # XXX Give the option to force the status to "active"?
            logger.info('trying next job...')
            cmd_run_next(jobs)
            return

        # We're okay to run the job.
        logger.info('Running next job from queue (%s)', reqid)
        logger.info('')
        try:
            _cmd_run(jobs, reqid)
        except RequestAlreadyStagedError:
            if reqid == exc.curid:
                logger.warn('%s is already running', reqid)
                # XXX Check the pidfile?
            else:
                logger.warn('another job is already running, adding %s back to the queue', reqid)
                jobs.queue.unpop(reqid)
    except KeyboardInterrupt:
        cmd_cancel(jobs, reqid, _status=Result.STATUS.PENDING)
        raise  # re-raise


def cmd_queue_info(jobs, *, withlog=True):
    _queue = jobs.queue.snapshot
    queued = _queue.jobs
    paused = _queue.paused
    pid, pid_running = _queue.locked
    if withlog:
        log = list(_queue.read_log())

    print('Job Queue:')
    print(f'  size:     {len(queued)}')
    #print(f'  max size: {maxsize}')
    print(f'  paused:   {paused}')
    if isinstance(pid, str):
        assert pid_running is None, repr(pid_running)
        print(f'  lock:     bad PID file (content: {pid!r})')
    elif pid:
        running = '' if pid_running else ' (not running)'
        print(f'  lock:     held by process {pid}{running}')
    else:
        print('  lock:     (not locked)')
    print()
    print('Files:')
    print(f'  data:      {_render_file(_queue.datafile)}')
    print(f'  lock:      {_render_file(_queue.lockfile)}')
    print(f'  log:       {_render_file(_queue.logfile)}')
    print()
    print('Top 5:')
    if queued:
        for i in range(min(5, len(queued))):
            print(f'  {i+1} {queued[i]}')
    else:
        print('  (queue is empty)')
    if withlog:
        print()
        print(f'Log size:    {len(log)}')
        print('Last log entry:')
        if log:
            print('-'*30)
            print()
            for line in log[-1].render():
                print(line)
            print()
            print('-'*30)
        else:
            print('  (log is empty)')


def cmd_queue_list(jobs):
    if jobs.queue.paused:
        logger.warn('job queue is paused')

    if not jobs.queue:
        print('no jobs queued')
        return

    print('Queued jobs:')
    for i, reqid in enumerate(jobs.queue, 1):
        print(f'{i:>3} {reqid}')
    print()
    print(f'(total: {i})')


def cmd_queue_pause(jobs):
    try:
       jobs.queue.pause()
    except JobQueuePausedError:
        logger.warn('job queue was already paused')
    else:
        logger.info('job queue paused')


def cmd_queue_unpause(jobs):
    try:
       jobs.queue.unpause()
    except JobQueueNotPausedError:
        logger.warn('job queue was not paused')
    else:
        logger.info('job queue unpaused')
        jobs.ensure_next()


def cmd_queue_push(jobs, reqid):
    reqid = RequestID.from_raw(reqid)
    logger.info(f'Adding job {reqid} to the queue')
    job = jobs.get(reqid)

    status = job.get_status()
    if not status:
        logger.error('request %s not found', reqid)
        sys.exit(1)
    elif status is not Result.STATUS.CREATED:
        logger.error('request %s has already been used', reqid)
        sys.exit(1)

    if jobs.queue.paused:
        logger.warn('job queue is paused')

    try:
        pos = jobs.queue.push(reqid)
    except JobAlreadyQueuedError:
        for pos, queued in enumerate(jobs.queue, 1):
            if queued == reqid:
                logger.warn('%s was already queued', reqid)
                break
        else:
            raise NotImplementedError

    job.set_status('pending')

    logger.info('%s added to the job queue at position %s', reqid, pos)

    jobs.ensure_next()


def cmd_queue_pop(jobs):
    logger.info(f'Popping the next job from the queue...')
    try:
        reqid = jobs.queue.pop()
    except JobQueuePausedError:
        logger.warn('job queue is paused')
        return
    except JobQueueEmptyError:
        logger.error('job queue is empty')
        sys.exit(1)
    job = jobs.get(reqid)

    status = job.get_status()
    if not status:
        logger.warn('queued request (%s) not found', reqid)
    elif status is not Result.STATUS.PENDING:
        logger.warn(f'expected "pending" status for queued request %s, got %r', reqid, status)
        # XXX Give the option to force the status to "active"?
    else:
        # XXX Set the status to "active"?
        pass

    print(reqid)


def cmd_queue_move(jobs, reqid, position, relative=None):
    position = int(position)
    if position <= 0:
        raise ValueError(f'expected positive position, got {position}')
    if relative and relative not in '+-':
        raise ValueError(f'expected relative of + or -, got {relative}')

    reqid = RequestID.from_raw(reqid)
    if relative:
        logger.info('Moving job %s %s%s in the queue...', reqid, relative, position)
    else:
        logger.info('Moving job %s to position %s in the queue...', reqid, position)
    job = jobs.get(reqid)

    if jobs.queue.paused:
        logger.warn('job queue is paused')

    status = job.get_status()
    if not status:
        logger.error('request %s not found', reqid)
        sys.exit(1)
    elif status is not Result.STATUS.PENDING:
        logger.warn('request %s has been updated since queued', reqid)

    pos = jobs.queue.move(reqid, position, relative)
    logger.info('...moved to position %s', pos)


def cmd_queue_remove(jobs, reqid):
    reqid = RequestID.from_raw(reqid)
    logger.info('Removing job %s from the queue...', reqid)
    job = jobs.get(reqid)

    if jobs.queue.paused:
        logger.warn('job queue is paused')

    status = job.get_status()
    if not status:
        logger.warn('request %s not found', reqid)
    elif status is not Result.STATUS.PENDING:
        logger.warn('request %s has been updated since queued', reqid)

    try:
        jobs.queue.remove(reqid)
    except JobNotQueuedError:
        logger.warn('%s was not queued', reqid)

    if status is Result.STATUS.PENDING:
        job.set_status('created')

    logger.info('...done!')


def cmd_config_show(jobs):
    for line in jobs.cfg.render():
        print(line)


def cmd_bench_host_clean(jobs):
    raise NotImplementedError


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
    # queue management
    'queue-info': cmd_queue_info,
    'queue-pause': cmd_queue_pause,
    'queue-unpause': cmd_queue_unpause,
    'queue-list': cmd_queue_list,
    'queue-push': cmd_queue_push,
    'queue-pop': cmd_queue_pop,
    'queue-move': cmd_queue_move,
    'queue-remove': cmd_queue_remove,
    # other public commands
    'config-show': cmd_config_show,
    'bench-host-clean': cmd_bench_host_clean,
    # internal-only
    'internal-finish-run': cmd_finish_run,
    'internal-run-next': cmd_run_next,
}


##################################
# the script

VERBOSITY = 3


def configure_logger(logger, verbosity=VERBOSITY, logfile=None, *,
                     maxlevel=logging.CRITICAL,
                     ):
    level = max(1,  # 0 disables it, so we use the next lowest.
                min(maxlevel,
                    maxlevel - verbosity * 10))
    logger.setLevel(level)
    #logger.propagate = False

    assert not logger.handlers, logger.handlers
    if logfile:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    #formatter = logging.Formatter()
    class Formatter(logging.Formatter):
        def format(self, record):
            text = super().format(record)
            if record.levelname not in ('DEBUG', 'INFO'):
                text = f'{record.levelname}: {text}'
            return text
    formatter = Formatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse

    add_hidden = ('-h' not in argv and '--help' not in argv)

    ##########
    # First, pull out the common args.

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', dest='cfgfile')
    common.add_argument('-v', '--verbose', action='count', default=0)
    common.add_argument('-q', '--quiet', action='count', default=0)
    common.add_argument('--logfile')
    args, argv = common.parse_known_args(argv)
    cfgfile = args.cfgfile
    verbosity = max(0, VERBOSITY + args.verbose - args.quiet)
    logfile = args.logfile

    ##########
    # Create the top-level parser.

    parser = argparse.ArgumentParser(
        prog=prog,
        parents=[common],
    )
    subs = parser.add_subparsers(dest='cmd', metavar='CMD')

    ##########
    # Add the subcommands for managing jobs.

    def add_cmd(name, subs=subs, *, parents=(), **kwargs):
        return subs.add_parser(name, parents=[common, *parents], **kwargs)

    sub = add_cmd('list', help='Print a table of all known jobs')
    sub.add_argument('selections', nargs='*')

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
#    sub.add_argument('reqid', nargs='+', help='the requests to delete (globs okay)')

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

    sub = add_cmd('queue', help='Manage the job queue')
    queue = sub.add_subparsers(dest='action')
    # Subcommands for different actions are added below.

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
                         action='store_const', const=(),
                         help='(the default)')
    add_job = (lambda job, **kw: add_cmd(job, jobs, parents=[_common], **kw))

    # This is the default (and the only one, for now).
    sub = add_job('compile-bench',
                  help='Request a compile-and-run-benchmarks job')
    sub.add_argument('--optimize', dest='optimize',
                     action='store_const', const=True,
                     help='(the default)')
    sub.add_argument('--no-optimize', dest='optimize', action='store_false')
    sub.add_argument('--debug', action='store_true')
    sub.add_argument('--benchmarks')
    sub.add_argument('--branch')
    sub.add_argument('--remote', required=True)
    sub.add_argument('revision')
    if add_hidden:
        # Use these flags to skip actually running pyperformance.
        sub.add_argument('--fake-success', dest='exitcode',
                         action='store_const', const=0)
        sub.add_argument('--fake-failure', dest='exitcode',
                         action='store_const', const=1)
        sub.add_argument('--fake-delay', dest='fakedelay')

    ##########
    # Add the "queue" subcomamnds.

    sub = add_cmd('info', queue, help='Print a summary of the state of the jobs queue')
    sub.add_argument('--without-log', dest='withlog', action='store_false')
    sub.add_argument('--with-log', dest='withlog',
                     action='store_const', const=True)

    sub = add_cmd('pause', queue, help='Do not let queued jobs run')

    sub = add_cmd('unpause', queue, help='Let queued jobs run')

    sub = add_cmd('list', queue, help='List the queued jobs')

    sub = add_cmd('push', queue, help='Add a job to the queue')
    sub.add_argument('reqid')

    sub = add_cmd('pop', queue, help='Get the next job from the queue')

    sub = add_cmd('move', queue, help='Move a job up or down in the queue')
    sub.add_argument('reqid')
    sub.add_argument('position')

    sub = add_cmd('remove', queue, help='Remove a job from the queue')
    sub.add_argument('reqid')

    ##########
    # Add other public commands.

    sub = add_cmd('config', help='show the config')

#    sub = add_cmd('bench-host', help='manage the host where benchmarks run')
#    benchhost = sub.add_subparsers(dest='action')
#
#    sub = add_cmd('clean', benchhost, help='clean up old files')

    ##########
    # Add internal commands.

    if add_hidden:
        sub = add_cmd('internal-finish-run')
        sub.add_argument('reqid')

        sub = add_cmd('internal-run-next')

    ##########
    # Finally, parse the args.

    args = parser.parse_args(argv)
    ns = vars(args)

    # Drop args we already handled earlier.
    ns.pop('cfgfile')
    ns.pop('verbose')
    ns.pop('quiet')
    ns.pop('logfile')

    # Process commands and command-specific args.
    cmd = ns.pop('cmd')
    if cmd in ('add', 'request'):
        job = ns.pop('job')
        cmd = f'request-{job}'
        if job == 'compile-bench':
            # Process hidden args.
            fake = (ns.pop('exitcode'), ns.pop('fakedelay'))
            if any(v is not None for v in fake):
                args._fake = fake
    elif cmd == 'config':
        cmd = 'config-show'
    elif cmd == 'queue':
        action = ns.pop('action')
        cmd = f'queue-{action}'
        if action == 'move':
            pos = args.position
            if pos == '+':
                pos = '1'
                relative = '+'
            elif pos == '-':
                pos = '1'
                relative = '-'
            elif pos.startswith('+'):
                pos = pos[1:]
                relative = '+'
            elif pos.startswith('-'):
                pos = pos[1:]
                relative = '-'
            else:
                # an absolute move
                relative = None
            if not pos.isdigit():
                parser.error('position must be positive int')
            pos = int(pos)
            if pos == 0:
                parser.error('position must be positive int')
            args.position = pos
            args.relative = relative
    elif cmd == 'bench-host':
        action = ns.pop('action')
        cmd = f'bench-host-{action}'

    return cmd, ns, cfgfile, verbosity, logfile


def main(cmd, cmd_kwargs, cfgfile=None):
    try:
        run_cmd = COMMANDS[cmd]
    except KeyError:
        logger.error('unsupported cmd %r', cmd)
        sys.exit(1)

    after = []
    for _cmd in cmd_kwargs.pop('after', None) or ():
        try:
            run_after = COMMANDS[_cmd]
        except KeyError:
            logger.error('unsupported "after" cmd %r', _cmd)
            sys.exit(1)
        after.append((_cmd, run_after))

    logger.debug('')
    logger.debug('# PID: %s', PID)

    # Load the config.
    if not cfgfile:
        cfgfile = PortalConfig.find_config()
    logger.debug('')
    logger.debug('# loading config from %s', cfgfile)
    cfg = PortalConfig.load(cfgfile)

    jobs = Jobs(cfg)

    if cmd != 'queue-info' and not cmd.startswith('internal-'):
        # In some cases the mechanism to run jobs from the queue may
        # get interrupted, so we re-start it manually here if necessary.
        jobs.ensure_next()

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
    logger.info('')
    logger.info('#'*40)
    if reqid:
        logger.info('# Running %r command for request %s', cmd, reqid)
    else:
        logger.info('# Running %r command', cmd)
    logger.info('')
    job = run_cmd(jobs, **cmd_kwargs)

    if cmd.startswith('request-'):
        _fmt = 'reqfile' if after else 'summary'
        logger.info('')
        # XXX Show something better?
        for line in job.render(fmt=_fmt):
            logger.info(line)

    # Run "after" commands, if any
    for cmd, run_cmd in after:
        logger.info('')
        logger.info('#'*40)
        if reqid:
            logger.info('# Running %r command for request %s', cmd, reqid)
        else:
            logger.info('# Running %r command', cmd)
        logger.info('')
        # XXX Add --lines='-1' for attach.
        run_cmd(jobs, reqid=reqid)


if __name__ == '__main__':
    cmd, cmd_kwargs, cfgfile, verbosity, logfile = parse_args()
    configure_logger(logger, verbosity, logfile)
    if not os.isatty(sys.stdout.fileno()):
        print = (lambda m='': logger.info(m))
    main(cmd, cmd_kwargs, cfgfile)
