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
    return bool(re.match(r'^[\w-]+$', value))


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


def _resolve_git_revision_and_branch(revision, branch):
    if not revision:
        if not branch:
            raise ValueError('missing revision')
        if re.match(r'^[a-fA-F0-9]{4,40}$', branch):
            # XXX
            ...
    elif not branch:
        if re.match(r'^[\w-]+$', revision):
            # XXX
            ...
        else:
            branch = None

    return revision, branch


def _read_file(filename):
    with open(filename) as infile:
        return infile.read()


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

    def __init__(self,
                 id,
                 remote,
                 revision,
                 branch,
                 benchmarks,
                 optimize,
                 debug,
                 ):
        kwargs = dict(locals())
        del kwargs['self']
        del kwargs['__class__']
        super().__init__(
            kind='bench-compile',
            **kwargs
        )

    @property
    def remote_name(self):
        return self.remote

    @property
    def remote_url(self):
        return f'https://github.com/{self.remote}/cpython'

    def as_jsonable(self):
        return dict(vars(self))


def _resolve_bench_compile_request(cfg, remote, revision, branch, benchmarks, *,
                                   optimize,
                                   debug,
                                   ):
    user = _resolve_user(cfg)
    reqid = next_req_id(user, cfg=cfg)
    revision, branch = _resolve_git_revision_and_branch(revision, branch)
    if isinstance(benchmarks, str):
        benchmarks = benchmarks.replace(',', ' ').split()
    if benchmarks:
        benchmarks = (b.strip() for b in benchmarks)
        benchmarks = [b for b in benchmarks if b]

    meta = BenchCompileRequest(
        id=reqid,
        remote=_resolve_git_remote(remote, user, branch, revision),
        revision=revision,
        branch=branch,
        benchmarks=benchmarks or None,
        optimize=bool(optimize),
        debug=bool(debug),
    )
    return reqid, meta


def _build_compile_config(cfg, req):
    benchreq = BenchRequestFS(req.id)

    cfg = configparser.ConfigParser()

    cfg['config'] = {}
    cfg['config']['json_dir'] = benchreq.results_dir
    cfg['config']['debug'] = str(req.debug)
    # XXX pyperformance should be looking in [scm] for this.
    cfg['config']['git_remote'] = req.remote

    cfg['scm'] = {}
    cfg['scm']['repo_dir'] = benchreq.cpython
    cfg['scm']['git_remote'] = req.remote
    cfg['scm']['update'] = 'True'

    cfg['compile'] = {}
    cfg['compile']['bench_dir'] = benchreq.scratch_dir
    cfg['compile']['pgo'] = str(req.optimize)
    cfg['compile']['lto'] = str(req.optimize)
    cfg['compile']['install'] = 'True'

    cfg['run_benchmark'] = {}
    cfg['run_benchmark']['benchmarks'] = ','.join(req.benchmarks or ())
    cfg['run_benchmark']['system_tune'] = 'False'
    cfg['run_benchmark']['upload'] = 'False'

    return cfg


def _build_compile_script(cfg, req):
    pfiles = PortalRequestFS(req.id)
    bfiles = BenchRequestFS(req.id)

    remote = req.remote_name or ""
    revision = req.revision or ""
    branch = req.branch or ""

    # On the bench host:
    python = 'python3.9'
    numjobs = 20

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        pushd {bfiles.cpython}
        ( set -x
        2>/dev/null git remote add $remote {req.remote_url}
        git fetch --tags {remote};
        # Get the upstream tags, just in case.
        git fetch --tags origin;
        )
        branch="{branch}"
        if [ -n "$branch" ]; then
            if ! ( set -x
                git checkout -b {branch} --track {remote}/{branch}
            ); then
                echo "It already exists; resetting to the right target."
                ( set -x
                git checout {branch}
                git reset --hard {remote}/{branch}
                )
            fi
        fi
        popd

        ( set -x
        PYTHONPATH={bfiles.pyperformance} MAKEFLAGS="-j{numjobs}" \\
            {python} -m pyperformance compile \\
            --venv {bfiles.venv} \\
            {pfiles.compile_config} \\
            {revision} \\
            {branch} \\
            2>&1 | tee {pfiles.results_log}
        )
        exitcode=$?

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

    # write metadata.
    with open(pfiles.request, 'w') as outfile:
        json.dump(req.as_jsonable(), outfile, indent=4)
        print(file=outfile)

    # write the config.
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
    parser.add_argument('--optimize', action='store_true')
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
