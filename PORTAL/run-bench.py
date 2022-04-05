from collections import namedtuple
import configparser
import json
import os
import os.path
import re
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


def next_req_id(user=None, cfg=None):
    if not cfg:
        cfg = PortalConfig.load()

    user = _resolve_user(cfg, user)
    timestamp = int(_utcnow())
    return f'req-{timestamp}-{user}'


def parse_req_id(reqid):
    m = re.match(r'^req-(\d{10})-(\w+)$', reqid)
    if not m:
        return None, None
    timestamp, user = m.groups()
    return user, timestamp


##################################
# minor utils

def _utcnow():
    if time.tzname[0] == 'UTC':
        return time.time()
    timestamp = time.mktime(time.gmtime())


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
REQUESTS = f'{DATA_ROOT}/REQUESTS'
STAGING = f'{REQUESTS}/CURRENT'


def ensure_dirs():
    for dirname in (DATA_ROOT, REQUESTS):
        os.makedirs(DATA_ROOT, exist_ok=True)


class PortalRequestFS(types.SimpleNamespace):
    # On the portal host.

    def __init__(self, req):
        if isinstance(req, str):
            reqid = req
        else:
            raise NotImplementedError(req)
        super().__init__(reqid=reqid)

    @property
    def data_root(self):
        return f'{REQUESTS}/{self.reqid}'
    reqdir = data_root

    @property
    def request(self):
        return f'{self.data_root}/request.json'

    @property
    def manifest(self):
        return f'{self.data_root}/benchmarks.manifest'

    @property
    def compile_config(self):
        return f'{self.data_root}/compile.ini'

    @property
    def portal_script(self):
        return f'{self.data_root}/send.sh'

    @property
    def bench_script(self):
        return f'{self.data_root}/run.sh'

    @property
    def results_meta(self):
        return f'{self.data_root}/results.json'

    @property
    def results_data(self):
        return f'{self.data_root}/results-data.json.gz'

    @property
    def results_log(self):
        return f'{self.data_root}/run.log'


class BenchRequestFS(types.SimpleNamespace):
    # On the bench host.

    REPOS = f'{DATA_ROOT}/repositories'

    def __init__(self, req):
        if isinstance(req, str):
            reqid = req
        else:
            raise NotImplementedError(req)
        super().__init__(reqid=reqid)

    @property
    def cpython(self):
        return f'{self.REPOS}/cpython'

    @property
    def pyperformance(self):
        return f'{self.REPOS}/pyperformance'

    @property
    def pyston_benchmarks(self):
        return f'{self.REPOS}/pyston-benchmarks'

    @property
    def data_root(self):
        return f'{REQUESTS}/{self.reqid}'
    reqdir = data_root

    @property
    def venv(self):
        return f'{self.data_root}/pyperformance-venv'

    @property
    def scratch_dir(self):
        return f'{self.data_root}/pyperformance-scratch'

    @property
    def results_dir(self):
        return self.data_root

    @property
    def results_data(self):
        return f'{self.results_dir}/*.json.gz'


##################################
# config

class PortalConfig(types.SimpleNamespace):

    CONFIG = f'{DATA_ROOT}/portal.json'

    @classmethod
    def load(cls, filename=None):
        with open(filename or cls.CONFIG) as infile:
            data = json.load(infile)
        return cls(**data)

    def __init__(self,
                 bench_user,
                 send_user,
                 send_host,
                 send_port,
                 ):
        if not bench_user:
            raise ValueError('missing bench_user')
        if not send_user:
            send_user = bench_user
        if not send_host:
            raise ValueError('missing send_host')
        if not send_port:
            raise ValueError('missing send_port')
        super().__init__(
            bench_user=bench_user,
            send_user=send_user,
            send_host=send_host,
            send_port=send_port,
        )


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
        super().__init__(f'{raeson} ({msg} - {reqdir})')
        self.reqid = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def _get_staged_request():
    try:
        reqdir = os.readlink(STAGING)
    except FileNotFoundError:
        return None
    reqid = os.path.basename(reqdir)
    if not parse_req_id(reqid)[0]:
        return StagedRequsetResolveError(None, reqdir, 'invalid', 'target not a request ID')
    if os.path.dirname(reqdir) != REQUESTS:
        return StagedRequsetResolveError(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    if not os.path.exists(reqdir):
        return StagedRequsetResolveError(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        return StagedRequsetResolveError(reqid, reqdir, 'malformed', 'target is not a directory')
    # XXX Do other checks?
    return reqid


def stage_request(reqid):
    reqdir = PortalRequestFS(reqid).reqdir
    try:
        os.symlink(reqdir, STAGING)
    except FileExistsError:
        curid = _get_staged_request() or '???'
        # XXX Delete the existing one if bogus?
        raise RequestAlreadyStagedError(reqid, curid)


def unstage_request(reqid):
    curid = _get_staged_request()
    if not curid or not isinstance(curid, str):
        raise RequestStagedRequestError(reqid)
    elif curid != reqid:
        raise RequestStagedRequestError(reqid, curid)
    os.unlink(STAGING)


##################################
# "compile"

class BenchCompileRequest(types.SimpleNamespace):

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

        kwargs = dict(locals())
        del kwargs['self']
        del kwargs['__class__']
        super().__init__(
            kind='bench-compile',
            **kwargs
        )

    @property
    def cpython(self):
        if self.remote:
            return self.CPYTHON.fork(self.remote, ref=self.ref)
        else:
            return self.CPYTHON.copy(ref=self.ref)

    def as_jsonable(self):
        return dict(vars(self))


def _resolve_bench_compile_request(cfg, remote, revision, branch, benchmarks, *,
                                   optimize,
                                   debug,
                                   ):
    user = _resolve_user(cfg)
    reqid = next_req_id(user, cfg=cfg)
    commit, branch, tag = _resolve_git_revision_and_branch(revision, branch, remote)
    if isinstance(benchmarks, str):
        benchmarks = benchmarks.replace(',', ' ').split()
    if benchmarks:
        benchmarks = (b.strip() for b in benchmarks)
        benchmarks = [b for b in benchmarks if b]

    meta = BenchCompileRequest(
        id=reqid,
        # XXX Add a "commit" field and use "tag or branch" for ref.
        ref=commit,
        remote=_resolve_git_remote(remote, user, branch, commit),
        branch=branch,
        benchmarks=benchmarks or None,
        optimize=bool(optimize),
        debug=bool(debug),
    )
    return reqid, meta


def _build_manifest(req):
    bfiles = BenchRequestFS(req.id)
    return textwrap.dedent(f'''
        [includes]
        <default>
        {bfiles.pyston_benchmarks}/benchmarks/MANIFEST
    '''[1:-1])


def _build_compile_config(cfg, req):
    pfiles = PortalRequestFS(req.id)
    bfiles = BenchRequestFS(req.id)

    cfg = configparser.ConfigParser()

    cfg['config'] = {}
    cfg['config']['json_dir'] = bfiles.results_dir
    cfg['config']['debug'] = str(req.debug)
    # XXX pyperformance should be looking in [scm] for this.
    cfg['config']['git_remote'] = req.remote

    cfg['scm'] = {}
    cfg['scm']['repo_dir'] = bfiles.cpython
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


def _build_compile_script(cfg, req):
    pfiles = PortalRequestFS(req.id)
    bfiles = BenchRequestFS(req.id)

    # On the bench host:
    python = 'python3.9'
    numjobs = 20

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        #####################
        # Ensure the dependencies.

        if [ ! -e {bfiles.cpython} ]; then
            ( set -x
            git clone https://github.com/python/cpython "{bfiles.cpython}"
            )
        fi
        if [ ! -e {bfiles.pyperformance} ]; then
            ( set -x
            git clone https://github.com/python/pyperformance "{bfiles.pyperformance}"
            )
        fi
        if [ ! -e {bfiles.pyston_benchmarks} ]; then
            ( set -x
            git clone https://github.com/pyston/python-macrobenchmarks "{bfiles.pyston_benchmarks}"
            )
        fi

        #####################
        # Get the repos are ready for the requested remotes and revisions.

        remote="{req.cpython.remote}"
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C "{bfiles.cpython}" remote add "{req.cpython.remote}" "{req.cpython.url}"
            git -C "{bfiles.cpython}" fetch --tags "{req.cpython.remote}"
            )
        fi
        # Get the upstream tags, just in case.
        ( set -x
        git -C "{bfiles.cpython}" fetch --tags origin
        )
        branch="{req.branch or ''}"
        if [ -n "$branch" ]; then
            if ! ( set -x
                git -C "{bfiles.cpython}" checkout -b "{req.branch or '$branch'}" --track "{req.cpython.remote or 'origin'}/{req.branch or '$branch'}"
            ); then
                echo "It already exists; resetting to the right target."
                ( set -x
                git -C "{bfiles.cpython}" checkout "{req.branch or '$branch'}"
                git -C "{bfiles.cpython}" reset --hard "{req.cpython.remote}/{req.branch or '$branch'}"
                )
            fi
        fi

        remote="{req.pyperformance.remote}"
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C "{bfiles.pyperformance}" remote add "{req.pyperformance.remote}" "{req.pyperformance.url}"
            )
        fi
        ( set -x
        git -C "{bfiles.pyperformance}" fetch --tags "{req.pyperformance.remote}"
        git -C "{bfiles.pyperformance}" checkout "{req.pyperformance.fullref}"
        )

        remote="{req.pyston_benchmarks.remote}"
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C "{bfiles.pyston_benchmarks}" remote add "{req.pyston_benchmarks.remote}" "{req.pyston_benchmarks.url}"
            )
        fi
        ( set -x
        git -C "{bfiles.pyston_benchmarks}" fetch --tags "{req.pyston_benchmarks.remote}"
        git -C "{bfiles.pyston_benchmarks}" checkout "{req.pyston_benchmarks.fullref}"
        )

        #####################
        # Run the benchmarks.

        ( set -x
        MAKEFLAGS="-j{numjobs}" \\
            "{python}" {bfiles.pyperformance}/dev.py compile \\
            "{pfiles.compile_config}" \\
            "{req.ref}" {('"' + req.branch + '"') if req.branch else ''} \\
            2>&1 | tee {pfiles.results_log}
        )
        exitcode=$?

        #####################
        # Record the results metadata.

        results=$(2>/dev/null ls {bfiles.results_data})
        results_name=$(2>/dev/null basename $results)

        echo "saving results..."
        if [ $exitcode -eq 0 -a -n "$results" ]; then
            cat > {pfiles.results_meta} << EOF
        {{
            "reqid": "{req.id}",
            "status": "success",
            "orig_data_file": "$results_name"
        }}
        EOF
        else
            cat > {pfiles.results_meta} << EOF
        {{
            "reqid": "{req.id}",
            "status": "failed"
        }}
        EOF
        fi

        if [ -n "$results" -a -e $results ]; then
            (
            set -x
            ln -s $results {pfiles.results_data}
            )
        fi

        echo "...done!"
    '''[1:-1])


def _build_send_script(cfg, req, *, hidecfg=False):
    pfiles = PortalRequestFS(req.id)
    bfiles = BenchRequestFS(req.id)

    if hidecfg:
        benchuser = '$benchuser'
        user = '$user'
        host = '$host'
        port = '$port'
    else:
        benchuser = cfg.bench_user
        user = cfg.send_user
        host = cfg.send_host
        port = cfg.send_port
    conn = f'{user}@{host}'

    #reqdir = STAGING
    reqdir = pfiles.reqdir

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        cfgfile='{cfg.CONFIG}'

        benchuser="$(jq -r '.bench_user' $cfgfile)"
        if [ "$USER" != "{benchuser}" ]; then
            setfacl -m {benchuser}:x $(dirname "$SSH_AUTH_SOCK")
            setfacl -m {benchuser}:rwx "$SSH_AUTH_SOCK"
            exec sudo --login --user --preserve-env SSH_AUTH_SOCK {benchuser} $@
        fi

        user="$(jq -r '.send_user' $cfgfile)"
        host="$(jq -r '.send_host' $cfgfile)"
        port="$(jq -r '.send_port' $cfgfile)"

        if ssh -p {port} "{conn}" test -e {reqdir}; then
            >&2 echo "{req.id} was already sent"
            exit 1
        fi

        set -x

        # Set up before running.
        ssh -p {port} "{conn}" mkdir -p {REQUESTS}
        scp -rp -P {port} {reqdir} "{conn}":{reqdir}
        ssh -p {port} "{conn}" mkdir -p {bfiles.scratch_dir}
        ssh -p {port} "{conn}" mkdir -p {bfiles.results_dir}

        # Run the request.
        ssh -p {port} "{conn}" {pfiles.bench_script}
        
        # Finish up.
        scp -p -P {port} "{conn}":{pfiles.results_meta} {reqdir}
        scp -rp -P {port} "{conn}":{pfiles.results_data} {reqdir}
        scp -rp -P {port} "{conn}":{pfiles.results_log} {reqdir}
    '''[1:-1])


def create_bench_compile_request(remote, revision, branch=None, *,
                                 benchmarks=None,
                                 optimize=False,
                                 debug=False,
                                 cfg=None,
                                 ):
    if not cfg:
        cfg = PortalConfig.load()

    ensure_dirs()

    reqid, req = _resolve_bench_compile_request(
        cfg, remote, revision, branch, benchmarks,
        optimize=optimize,
        debug=debug,
    )
    pfiles = PortalRequestFS(reqid)

    os.mkdir(pfiles.reqdir)

    # Write metadata.
    with open(pfiles.request, 'w') as outfile:
        json.dump(req.as_jsonable(), outfile, indent=4)
        print(file=outfile)

    # Write the benchmarks manifest.
    manifest = _build_manifest(req)
    with open(pfiles.manifest, 'w') as outfile:
        outfile.write(manifest)

    # Write the config.
    ini = _build_compile_config(cfg, req)
    with open(pfiles.compile_config, 'w') as outfile:
        ini.write(outfile)

    # Write the commands to execute remotely.
    script = _build_compile_script(cfg, req)
    with open(pfiles.bench_script, 'w') as outfile:
        outfile.write(script)
    os.chmod(pfiles.bench_script, 0o755)

    # Write the commands to execute locally.
    script = _build_send_script(cfg, req)
    with open(pfiles.portal_script, 'w') as outfile:
        outfile.write(script)
    os.chmod(pfiles.portal_script, 0o755)

    return reqid


def send_bench_compile_request(remote, revision, branch=None, *,
                               benchmarks=None,
                               optimize=False,
                               debug=False,
                               cfg=None,
                               ):
    if not cfg:
        cfg = PortalConfig.load()

    reqid = create_bench_compile_request(
        remote=remote,
        revision=revision,
        branch=branch,
        benchmarks=benchmarks,
        optimize=optimize,
        debug=debug,
        cfg=cfg,
    )
    pfiles = PortalRequestFS(reqid)

    print('staging...')
    try:
        stage_request(reqid)
    except RequestAlreadyStagedError as exc:
        # XXX Offer to clear CURRENT?
        sys.exit(f'ERROR: {exc}')
    except Exception:
        shutil.rmtree(pfiles.reqdir)
        raise  # re-raise
    try:
        print('...running....')
        #print(_read_file(pfiles.portal_script))
        subprocess.run(pfiles.portal_script)
    except KeyboardInterrupt:
        # XXX Mark it as canceled.
        raise  # re-raise
    finally:    
        print('...unstaging...')
        unstage_request(reqid)
        print('...done!')

        print()
        print('Results:')
        for line in render_results(reqid, pfiles):
            print(line)


def render_request(reqid, pfiles=None):
    if not pfiles:
        pfiles = PortalRequestFS(reqid)

    yield f'(in {pfiles.reqdir})'
    yield ''
    # XXX Show something better?
    text = _read_file(pfiles.request)
    yield from text.splitlines()


def render_results(reqid, pfiles=None):
    if not pfiles:
        pfiles = PortalRequestFS(reqid)

    yield f'(in {pfiles.reqdir})'
    yield ''
    # XXX Show something better?
    text = _read_file(pfiles.results_meta)
    yield from text.splitlines()


##################################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(prog=prog)

    parser.add_argument('--create-only', dest='createonly',
                        action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--benchmarks')
    parser.add_argument('--branch')
    parser.add_argument('--remote', required=True)
    parser.add_argument('revision')

    args = parser.parse_args()

    return vars(args)


def main(*, createonly=False, **kwargs):
    cfg = PortalConfig.load()

    #if USER != cfg.bench_user:
    #    os.execl('sudo', '--login', '--user', cfg.bench_user, *sys.argv[1:])

    if createonly:
        reqid = create_bench_compile_request(cfg=cfg, **kwargs)
        print(f'Created request {reqid}:')
        print()
        for line in render_request(reqid):
            print(line)
        return

    # XXX
    send_bench_compile_request(cfg=cfg, **kwargs)
    #send_bench_compile_request('origin', 'master', debug=True)
    #send_bench_compile_request('origin', 'deadbeef', 'master', debug=True)


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
