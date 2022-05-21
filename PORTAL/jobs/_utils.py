from collections import namedtuple
import datetime
import json
import logging
import os
import os.path
import re
import shlex
import shutil
import subprocess
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
CWD = os.getcwd()
PID = os.getpid()

logger = logging.getLogger(__name__)


##################################
# string utils

def check_name(name, *, loose=False):
    if not name:
        raise ValueError(name)
    orig = name
    if loose:
        name = '_' + name.replace('-', '_')
    if not name.isidentifier():
        raise ValueError(orig)


def validate_string(value, argname=None, *, required=True):
    if not value and required:
        raise ValueError(f'missing {argname or "required value"}')
    if not isinstance(value, str):
        label = f' for {argname}' if argname else ''
        raise TypeError(f'expected str{label}, got {value!r}')


##################################
# date/time utils

SECOND = datetime.timedelta(seconds=1)
DAY = datetime.timedelta(days=1)


def utcnow():
    if time.tzname[0] == 'UTC':
        return time.time()
    return time.mktime(time.gmtime())


def get_utc_datetime(timestamp=None, *, fail=True):
    tzinfo = datetime.timezone.utc
    if timestamp is None:
        timestamp = int(utcnow())
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

def check_shell_str(value, *, required=True, allowspaces=False):
    validate_string(value, required=required)
    if not allowspaces and ' ' in value:
        raise ValueError(f'unexpected space in {value!r}')
    return value


def quote_shell_str(value, *, required=True):
    check_shell_str(value, required=required, allowspaces=True)
    return shlex.quote(value)


def write_json(data, outfile):
    json.dump(data, outfile, indent=4)
    print(file=outfile)


def wait_for_file(filename, *, timeout=None):
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


def read_file(filename, *, fail=True):
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


def render_file(filename):
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

        #text = read_file(self._filename, fail=False) or ''
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

def looks_like_git_commit(value):
    return bool(re.match(r'^[a-fA-F0-9]{4,40}$', value))


def looks_like_git_name(value):
    # check_name() is too strict, even with loose=True.
    return bool(re.match(r'^[\w][\w.-]*$', value))


def looks_like_git_tag(value):
    return looks_like_git_name(value)


def looks_like_git_branch(value):
    return looks_like_git_name(value)


def looks_like_git_remote(value):
    return looks_like_git_name(value)


def looks_like_git_ref(value):
    if not value:
        return False
    elif value == 'latest':
        return True
    elif value == 'HEAD':
        return True
    elif looks_like_git_commit(value):
        return True
    elif looks_like_git_tag(value):
        return True
    elif looks_like_git_branch(value):
        return True
    else:
        return False


def git(*args, cwd=HOME, GIT=shutil.which('git')):
    logger.debug('# running: %s', ' '.join(args))
    proc = subprocess.run(
        [GIT, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        cwd=cwd,
    )
    return proc.returncode, proc.stdout


class GitHubTarget(types.SimpleNamespace):

    @classmethod
    def origin(cls, org, project):
        return cls(org, project, remote='origin')

    def __init__(self, org, project, ref=None, remote=None, upstream=None):
        check_name(org, loose=True)
        check_name(project, loose=True)
        if not ref:
            ref = None
        elif not isinstance(ref, str):
            raise NotImplementedError(ref)
        elif not looks_like_git_ref(ref):
            raise ValueError(ref)
        if not remote:
            remote = None
        elif not isinstance(remote, str):
            raise NotImplementedError(remote)
        elif not looks_like_git_remote(remote):
            raise ValueError(remote)
        if upstream is not None and not isinstance(upstream, GitHubTarget):
            raise TypeError(upstream)

        kwargs = dict(locals())
        del kwargs['self']
        del kwargs['__class__']
        super().__init__(**kwargs)

    @property
    def remote(self):
        remote = vars(self)['remote']
        if remote:
            return remote
        return self.org if self.upstream else 'upstream'

    @property
    def fullref(self):
        if self.ref:
            if looks_like_git_commit(self.ref):
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
            remote=vars(self)['remote'],
            upstream=self.upstream,
        )

    def fork(self, org, project=None, ref=None, remote=None):
        return type(self)(
            org=org,
            project=project or self.project,
            ref=ref or self.ref,
            remote=remote,
            upstream=self,
        )

    def as_jsonable(self):
        return dict(vars(self))


class GitRefCandidates:

    @classmethod
    def from_revision(cls, revision, branch, remote):
        if not revision:
            revision = None
        elif revision.upper() == 'HEAD':
            revision = 'HEAD'
        elif not looks_like_git_ref(revision):
            raise ValueError(f'unsupported revision {revision!r}')
        if not branch:
            branch = None
        elif not looks_like_git_branch(branch):
            raise ValueError(f'unsupported branch {branch!r}')
        if not remote:
            remote = None
        elif not looks_like_git_remote(remote):
            raise ValueError(f'unsupported remote {remote!r}')

        if branch == 'main':
            refs = cls._from_main(revision, remote)
        elif remote == 'origin':
            refs = cls._from_origin(revision, branch)
        elif remote:
            refs = cls._from_non_origin(revision, branch, remote)
        elif branch:  # no remote, maybe a revision
            # We know all remotes have release (version) branches.
            refs = cls._from_version(revision, branch, 'origin', required=False)
            if not refs:
                # We don't bother trying to guess the remote at this point.
                raise ValueError('missing remote for branch {branch!r}')
        elif revision:  # no remote or branch
            refs = cls._from_origin(revision, branch, required=False)
            if not refs:
                raise ValueError(f'missing remote for revision {revision!r}')
        else:
            refs = cls._from_main(revision, remote)
        return cls(refs)

    @classmethod
    def _from_main(cls, revision, remote):
        # The main branch defaults to origin.
        # We do not support tags for the main branch.
        if not remote:
            remote = 'origin'
        if not revision:
            return [(remote, 'main', 'HEAD')]
        elif revision in ('latest', 'HEAD'):
            return [(remote, 'main', 'HEAD')]
        elif looks_like_git_commit(revision):
            return [(remote, 'main', revision)]
        else:
            raise ValueError(f'unexpected revision {revision!r}')

    @classmethod
    def _from_origin(cls, revision, branch, required=True):
        if branch == 'main':
            return cls._from_main(revision, 'origin')
        elif branch:
            return cls._from_version(revision, branch, 'origin')
        elif revision == 'main':
            return cls._from_main(None, 'origin')
        elif revision in ('latest', 'HEAD'):
            return cls._from_main(revision, 'origin')
        elif revision:
            if looks_like_git_commit(revision):
                return [('origin', None, revision)]
            else:
                # The only remaining possibility for origin
                # is a release branch or tag.
                return cls._from_version(revision, branch, 'origin', required)
        else:
            return cls._from_main(revision, 'origin')

    @classmethod
    def _from_non_origin(cls, revision, branch, remote):
        if branch:
            if not revision:
                # For non-origin, we don't bother with "latest" for versions.
                return [(remote, branch, 'HEAD')]
            elif Version.parse(branch, match=revision):
                return [(remote, branch, revision)]
            else:
                # For non-origin, revision can be any tag or commit.
                return [(remote, branch, revision)]
        else:
            if not revision:
                # Unlike for origin, here we don't assume "main".
                raise ValueError('missing revision')
            elif revision in ('latest', 'HEAD'):
                raise ValueError('missing branch')
            elif revision == 'main':
                return cls._from_main(None, remote)
            elif looks_like_git_commit(revision):
                return [(remote, None, revision)]
            else:
                refs = cls._from_version(revision, None, remote, required=False)
                if refs:
                    return refs
                if looks_like_git_branch(revision):
                    return [
                        (remote, None, revision),
                        (remote, revision, 'latest'),
                    ]
                else:
                    raise ValueError(f'unexpected revision {revision!r}')

    @classmethod
    def _from_version(cls, revision, branch, remote, required=True):
        if not remote:
            remote = 'origin'
        if branch:
            version = Version.parse(branch)
            if not version:
                if required:
                    raise ValueError(f'unexpected branch {branch!r}')
                return None
            verstr = f'{version.major}.{version.minor}'
            if verstr != branch:
                if required:
                    raise ValueError(f'unexpected branch {branch!r}')
                return None
            if not revision or revision == 'latest':
                return [(remote, branch, 'latest')]
            elif revision == 'HEAD':
                return [(remote, branch, revision)]
            elif looks_like_git_commit(revision):
                return [(remote, branch, revision)]
            else:
                tagver = Version.parse(revision)
                if not tagver:
                    raise ValueError(f'unexpected revision {revision!r}')
                if tagver[:2] != version[:2]:
                    raise ValueError(f'tag {revision!r} does not match branch {branch!r}')
                return [(remote, branch, revision)]
        else:
            tagver = Version.parse(revision)
            if not tagver:
                if required:
                    raise ValueError(f'unexpected revision {revision!r}')
                return None
            verstr = f'{tagver.major}.{tagver.minor}'
            return [
                (remote, None, revision),
                (remote, verstr, 'latest' if revision == verstr else revision),
            ]

    def __init__(self, refs):
        self._refs = refs

    def __repr__(self):
        return f'{type(self).__name__}({self._refs})'

    def __len__(self):
        return len(self._refs)

    def __iter__(self):
        yield from self._refs

    def __getitem__(self, index):
        return self._refs[index]

    def find_ref(self):
        by_remote = {}
        for remote, branch, revision in self._refs:
            if remote not in by_remote:
                by_remote[remote] = GitRefs.from_remote(remote)
            repo_refs = by_remote[remote]

            _branch = tag = commit = ref = kind = None
            if revision == 'HEAD':
                assert branch, self
                matched = repo_refs.match_branch(branch)
                if matched:
                    _branch, tag, commit = matched
                    kind = 'branch'
                    ref = revision
                else:
                    logger.warning(f'branch {branch} not found')
            elif revision == 'latest':
                assert branch, self
                assert Version.parse(branch), (branch, self)
                matched = repo_refs.match_latest_version(branch)
                if matched:
                    _branch, tag, commit = matched
                    kind = 'tag'
                    ref = tag or branch
                    branch = _branch
                elif repo_refs.match_branch(branch):
                    logger.warning(f'latest tag for branch {branch} not found')
                else:
                    logger.warning(f'branch {branch} not found')
            elif looks_like_git_commit(revision):
                matched = repo_refs.match_commit(revision)
                if matched:
                    _branch, tag, commit = matched
                    assert not branch or _branch == branch, (branch, _branch, self)
                else:
                    if branch:
                        if not repo_refs.match_branch(branch):
                            logger.warning(f'branch {branch} not found')
                    commit = revision
                if commit:
                    kind = 'commit'
                    ref = commit
            else:
                assert looks_like_git_tag(revision), (revision, self)
                matched = repo_refs.match_tag(revision)
                if matched:
                    _branch, tag, commit = matched
                    kind = 'tag'
                    ref = tag

            if commit:
                if branch and _branch != branch:
                    logger.warning(f'branch mismatch (wanted {branch}, found {_branch})')
                else:
                    assert ref, (commit, branch or _branch, remote)
                    return GitRef(ref, None, commit, branch or _branch, remote)
        else:
            return None


class GitRef(namedtuple('GitRef', 'ref kind commit branch remote')):

    KINDS = {
        'commit',
        'tag',
        'branch',
        'other',
    }

    def __new__(cls, ref, kind, commit, branch, remote):
        if not ref:
            raise ValueError('missing ref')

        if not kind:
            if ref == commit:
                kind = 'commit'
            elif ref == branch:
                kind = 'branch'
            elif ref == 'HEAD':
                kind = 'other'
            elif looks_like_git_commit(ref):
                kind = 'commit'
                if not commit:
                    commit = ref
                elif ref != commit:
                    raise ValueError(f'commit ref {ref!r} does not match commit {commit!r}')
            elif looks_like_git_branch(ref):
                kind = 'tag'
            else:
                raise ValueError('missing kind')
        elif kind == 'commit':
            if not commit:
                commit = ref
            elif ref != commit:
                raise ValueError(f'commit ref {ref!r} does not match commit {commit!r}')
        elif kind == 'tag':
            if not looks_like_git_branch(ref):
                raise ValueError(f'invalid tag {ref!r}')
        elif kind == 'branch':
            if not branch:
                branch = ref
            elif ref != branch:
                raise ValueError(f'branch ref {ref!r} does not match branch {branch!r}')
        elif kind == 'other':
            if ref != 'HEAD':
                raise NotImplementedError(ref)
        else:
            raise ValueError(f'unsupported kind {kind!r}')

        if not commit:
            raise ValueError('missing commit')
        elif not looks_like_git_commit(commit):
            raise ValueError(f'invalid commit {commit!r}')
        if branch and not looks_like_git_branch(branch):
            raise ValueError(f'invalid branch {branch!r}')
        if remote and not looks_like_git_branch(remote):
            raise ValueError(f'invalid remote {remote!r}')

        self = super().__new__(
            cls,
            ref=ref,
            kind=kind,
            commit=commit,
            branch=branch or None,
            remote=remote or None,
        )
        return self

    def __str__(self):
        return self.commit


class GitRefs(types.SimpleNamespace):

    @classmethod
    def from_remote(cls, remote):
        if remote == 'origin' or not remote:
            url = 'https://github.com/python/cpython'
        elif remote == 'upstream':
            url = 'https://github.com/faster-cpython/cpython'
        else:
            url = f'https://github.com/{remote}/cpython'
        return cls.from_url(url)

    @classmethod
    def from_url(cls, url):
        ec, text = git('ls-remote', '--refs', '--tags', '--heads', url)
        if ec != 0:
            return None, None, None
        return cls._parse_ls_remote(text.splitlines())

    @classmethod
    def _parse_ls_remote(cls, lines):
        branches = {}
        tags = {}
        for line in lines:
            m = re.match(r'^([a-zA-Z0-9]+)\s+refs/(heads|tags)/(\S.*)$', line)
            if not m:
                continue
            commit, kind, name = m.groups()
            if kind == 'heads':
                group = branches
            elif kind == 'tags':
                group = tags
            else:
                raise NotImplementedError(kind)
            group[name] = commit
        return cls(branches, tags)

    def __init__(self, branches, tags):
        super().__init__(
            branches=branches,
            tags=tags,
        )

    def _release_branches(self):
        by_version = {}
        for branch in self.branches:
            version = Version.parse(branch)
            if version and branch == f'{version.major}.{version.minor}':
                by_version[version] = branch
        return by_version

    def _next_versions(self):
        releases = self._release_branches()
        if not releases:
            return ()
        latest, _ = sorted(releases.items())[-1]
        return [
            Version(latest.major, latest.minor + 1),
            Version(latest.major + 1, 0),
        ]

    def match_ref(self, ref):
        assert ref
        if looks_like_git_commit(ref):
            for tag, commit in self.tags.items():
                assert commit
                if ref == commit:
                    return self.match_tag(tag)
            for branch, commit in self.branches.items():
                assert commit
                if ref == commit:
                    return branch, None, commit
            return None
        else:
            matched = self.match_tag(ref)
            if matched:
                return matched
            return self.match_branch(ref)

    def match_branch(self, ref):
        if not ref or not looks_like_git_branch(ref):
            return None
        if ref in self.branches:
            branch = ref
        else:
            # Treat it like main if one higher than the latest.
            branch = None
            version = Version.parse(ref)
            if version and ref == f'{version.major}.{version.minor}':
                if version in self._next_versions():
                    branch = 'main'
        if branch:
            commit = self.branches.get(branch)
            if commit:
                return branch, None, commit
        return None

    def match_tag(self, ref): 
        if not ref or not looks_like_git_branch(ref):
            return None
        version = Version.parse(ref)
        if version:
            # Find a tag that matches the version.
            for tag in self.tags:
                tagver = Version.parse(tag)
                if tagver:
                    if tagver == version:
                        commit = self.tags[tag]
                        branch = f'{version.major}.{version.minor}'
                        if branch not in self.branches:
                            branch = None
                        #tag = version.as_tag()
                        assert commit
                        return branch, tag, commit
        else:
            # Find a tag that matches exactly.
            if ref in self.tags:
                commit = self.tags[ref]
                assert commit
                return None, ref, commit
        # No tags matched!
        return None

    def match_latest_version(self, branch):
        if not branch or not looks_like_git_branch(branch):
            return None
        version = Version.parse(branch)
        if version:
            # Find the latest tag that matches the branch.
            matched = {}
            for tag in self.tags:
                tagver = Version.parse(tag)
                if version.match(tagver):
                    matched[(tagver.full, tag)] = (self.tags[tag], tagver)
            if matched:
                key = sorted(matched)[-1]
                commit, tagver = matched[key]
                _, tag = key
                #tag = tagver.as_tag()
                assert commit
                return branch, tag, commit
        # Fall back to the branch.
        return self.match_branch(branch)


def resolve_git_revision_and_branch(revision, branch, remote):
    candidates = GitRefCandidates.from_revision(revision, branch, remote)
    return candidates.find_ref()


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
            text = read_file(filename, fail=False)
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


##################################
# other utils

def ensure_int(raw, min=None):
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        value = int(raw)
    else:
        raise TypeError(raw)
    if value < min:
        raise ValueError(raw)
    return value


def coerce_int(value, *, fail=True):
    if isinstance(value, int):
        return value
    elif not value:
        if fail:
            raise ValueError('missing')
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            if fail:
                raise  # re-raise
    else:
        if fail:
            raise TypeError(f'unsupported value {value!r}')
    return None


def validate_int(value, name=None, *, range=None, required=True):
    if isinstance(value, int):
        if not range:
            return value
        elif range == 'non-negative':
            if value >= 0:
                return value
        else:
            raise NotImplementedError(f'unsupported range {range!r}')
        Error = ValueError
    elif not value:
        if not required:
            return None
        raise ValueError(f'missing {name}' if name else 'missing')
    else:
        Error = TypeError
    # Failed!
    qualifier = f'a {range}' if range else 'an'
    namepart = f' for {name}' if name else ''
    raise Error(f'expected {qualifier} int{namepart}, got {value}')


def normalize_int(value, name=None, *,
                  range=None,
                  coerce=False,
                  required=True,
                  ):
    if coerce:
        value = coerce_int(value)
    return validate_int(value, name, range=range, required=required)


def get_slice(raw):
    if isinstance(raw, int):
        start = stop = None
        if raw < 0:
            start = raw
        elif criteria > 0:
            stop = raw
        return slice(start, stop)
    elif isinstance(raw, str):
        if raw.isdigit():
            return get_slice(int(raw))
        elif raw.startswith('-') and raw[1:].isdigit():
            return get_slice(int(raw))
        else:
            raise NotImplementedError(repr(raw))
    else:
        raise TypeError(f'expected str, got {criteria!r}')


def resolve_user(cfg, user=None):
    if not user:
        user = USER
        if not user or user == 'benchmarking':
            user = SUDO_USER
            if not user:
                raise Exception('could not determine user')
    if not user.isidentifier():
        raise ValueError(f'invalid user {user!r}')
    return user


class VersionRelease(namedtuple('VersionRelease', 'level serial')):

    LEVELS = {
        'alpha': 'a',
        'beta': 'b',
        'candidate': 'rc',
        'final': 'f',
    }
    LEVEL_SYMBOLS = {s: l for l, s in LEVELS.items()}
    LEVEL_SYMBOLS['c'] = 'candidate'

    PAT = textwrap.dedent(rf'''(?:
        ( {'|'.join(LEVEL_SYMBOLS)} )  # <level>
        ( \d+ )  # <serial>
    )''')

    @classmethod
    def from_values(cls, level=None, serial=None, *, usedefault=True):
        if isinstance(serial, int):
            return cls(level, serial)
        else:
            if not serial:
                if level == 'final':
                    serial = 0
                elif not level:
                    if not usedefault:
                        return None
                    level = 'final'
                    serial = 0
            elif isinstance(serial, str):
                serial = coerce_int(serial, fail=False)
            try:
                level = cls.LEVEL_SYMBOLS[level]
            except (KeyError, TypeError):
                pass
            return cls(level, serial)

    @classmethod
    def validate(cls, release):
        if not isinstance(release, cls):
            raise TypeError(f'expected a {cls.__name__}, got {release!r}')
        release._validate()

    def __init__(self, *args, **kwargs):
        self._validate()

    def __str__(self):
        level = self.LEVELS[self.level]
        return f'{self.LEVELS[self.level]}{self.serial}'

    def _validate(self):
        if not self.level:
            raise ValueError('missing level')
        elif self.level not in self.LEVELS:
            raise ValueError(f'unsupported level {self.level}')
        elif self.level == 'final':
            if self.serial != 0:
                raise ValueError(f'final releases always have a serial of 0, got {self.serial}')
        validate_int(self.serial, 'serial', range='non-negative', required=True)


class Version(namedtuple('Version', 'major minor micro release')):

    PAT = textwrap.dedent(rf'''(?:
        (\d+)  # <major>
        \.
        (\d+)  # <minor>
        (?:
            (?:
                \.
                (\d+)  # <micro>
             )?
            (?:
                {VersionRelease.PAT}  # <level> <serial>
            )?
         )?
    )''')
    REGEX = re.compile(f'^v?{PAT}$', re.VERBOSE)

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        elif isinstance(raw, (tuple, list)):
            return cls(*raw)
        else:
            return cls(**raw)

    @classmethod
    def parse(cls, verstr, match=None):
        m = cls.REGEX.match(verstr)
        if not m:
            return None
        major, minor, micro, level, serial = m.groups()
        release = VersionRelease.from_values(level, serial, usedefault=False)
        self = cls.__new__(
            cls,
            int(major),
            int(minor),
            int(micro) if micro else 0 if release else None,
            release,
        )
        self._raw = verstr
        if match is not None and not self.match(match):
            return None
        return self

    def __new__(cls, major, minor, micro=None, release=None):
        return super().__new__(cls, major, minor, micro, release or None)

    def __init__(self, *args, **kwargs):
        self._validate()

    def __str__(self):
        return self.render()

    def _validate(self):
        def _validate_int(name, *, required=False):
            val = getattr(self, name)
            validate_int(val, name, range='non-negative', required=required)
        _validate_int('major', required=True)
        _validate_int('minor', required=True)
        _validate_int('micro')
        if self.release is not None:
            VersionRelease.validate(self.release)
            if self.micro is None:
                raise ValueError('missing micro')

    @property
    def branch(self):
        return self[:2]

    @property
    def full(self):
        if self.release:
            return self
        major, minor, micro = self[:3]
        release = VersionRelease.from_values()
        cls = type(self)
        full = cls.__new__(cls, major, minor, micro or 0, release)
        full._raw = self._raw
        return full

    @property
    def plain(self):
        major, minor, micro = self[:3]
        if micro and not self.release:
            return self
        cls = type(self)
        plain = cls.__new__(cls, major, minor, micro or 0)
        plain._raw = self.raw
        return plain

    @property
    def flat(self):
        major, minor, micro, release = self
        level, serial = release if release else (None, None)
        return major, minor, micro, level, serial

    @property
    def raw(self):
        try:
            return self._raw
        except AttributeError:
            self._raw = self.render()
            return self._raw

#    def compare(self, other):
#        raise NotImplementedError

    def match(self, other, *, subversiononly=False):
        """Return True if other is a subversion."""
        if not other:
            return None
        else:
            other = Version.from_raw(other)
            if not other:
                return None
        if not subversiononly and self == other:
            return True
        if not self.release:
            if self.micro is not None:
                if other.release:
                    return self[:3] == other[:3]
            else:
                if other.micro is not None:
                    return self[:2] == other[:2]
        return False

    def render(self):
        if self.release:
            return f'{self.major}.{self.minor}.{self.micro}{self.release}'
        elif self.micro:
            return f'{self.major}.{self.minor}.{self.micro}'
        else:
            return f'{self.major}.{self.minor}'

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


def parse_bool_env_var(valstr, *, failunknown=False):
    m = re.match(r'^\s*(?:(1|y(?:es)?|t(?:rue)?)|(0|no?|f(?:alse)?))\s*$',
                 valstr.lower())
    if not m:
        if failunknown:
            raise ValueError(f'unsupported env var bool value {valstr!r}')
        return None
    yes, no = m.groups()
    return True if yes else False


def get_bool_env_var(name, default=None, *, failunknown=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return parse_bool_env_var(value, failunknown=failunknown)


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


def run_fg(cmd, *args):
    argv = [cmd, *args]
    logger.debug('# running: %s', ' '.join(shlex.quote(a) for a in argv))
    return subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
    )


def run_bg(argv, logfile=None, cwd=None):
    if not argv:
        raise ValueError('missing argv')
    elif isinstance(argv, str):
        if not argv.strip():
            raise ValueError('missing argv')
        cmd = argv
    else:
        cmd = ' '.join(shlex.quote(a) for a in argv)

    if logfile:
        logfile = quote_shell_str(logfile)
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
        cwd=cwd,
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
        write_json(data, resfile)


class SSHCommands:

    SSH = shutil.which('ssh')
    SCP = shutil.which('scp')

    def __init__(self, host, port, user, *, ssh=None, scp=None):
        self.host = check_shell_str(host)
        self.port = int(port)
        if self.port < 1:
            raise ValueError(f'invalid port {self.port}')
        self.user = check_shell_str(user)

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
        return run_fg(*argv)

    def run_shell(self, cmd, *args):
        argv = super().run_shell(cmd, *args)
        return run_fg(*argv)

    def push(self, source, target):
        argv = super().push(*args)
        return run_fg(*argv)

    def pull(self, source, target):
        argv = super().push(*args)
        return run_fg(*argv)

    def read(self, filename):
        if not filename:
            raise ValueError(f'missing filename')
        proc = self.run_shell(f'cat {filename}')
        if proc.returncode != 0:
            return None
        return proc.stdout
