import hashlib
import json
import os
import os.path
import re
import shlex
import subprocess
import sys
import sysconfig


try:
    PLATLIBDIR = sys.platlibdir
except AttributeError:
    PLATLIBDIR = 'lib'


def is_python(filename):
    m = re.match(r'^python[\d\.]*(\.exe)?$',
                 os.path.basename(filename).lower())
    return m is not None


def _hash_file(filename, hashobj=None, gitinfo=None, *, userev=None):
    if hashobj is None:
        hashobj = hashlib.sha256()

    revision = None
    if gitinfo:
        if isinstance(gitinfo, str):
            reporoot = gitinfo
        else:
            reporoot, revision = gitinfo
    else:
        reporoot = None

    relpath = None
    if reporoot and not os.path.isabs(filename):
        relpath = filename
        filename = os.path.join(reporoot, filename)

    if userev and relpath and revision:
        proc = subprocess.run(f'git show {revision}:{relpath}',
                              shell=True,
                              capture_output=True,
                              text=False,
                              cwd=reporoot,
                              )
        proc.check_returncode()
        hashobj.update(proc.stdout)
    else:
        with open(filename, 'rb') as infile:
            while True:
                data = infile.read(1024)
                if not data:
                    break
                hashobj.update(data)

    return hashobj


def _parse_version(version):
    version, _, extra = sys.version.partition(' ')

    if extra.endswith(']'):
        extra, _, compiler = extra[:-1].rpartition('[')
    else:
        raise NotImplementedError
    extra = extra.strip()

    if extra.startswith('(') and extra.endswith(')'):
        build = extra[1:-1]
    else:
        raise NotImplementedError
    git, _, builddate = build.partition(',')
    gitref, _, gitrev = git.partition(':')
    git = (None, gitref, gitrev)

    return version, git, builddate.strip(), compiler


def _parse_git_tag(text):
    tag = text
    branch = None
    m = re.match(r'^(?:(?:(\w[^-]*)-\d+(?:-\d+)?)|(?:v(\d+\.\d+)(?:[.abr].*)?)|(\d+\.\d+))$', text)
    if m:
        outdated, release, oldbranch = m.groups()
        branch = outdated or release or oldbranch or None
    return branch, tag


def _resolve_git(info, isdev):
    reporoot = None
    if isdev:
        reporoot = os.path.dirname(os.path.dirname(os.__file__))

    _, ref, rev = info
    if not rev:
        if isdev:
            # XXX Run f'git rev-parse --short {ref}'?
            raise RuntimeError('a dev Python should have git info')
        if not ref:
            _, ref, rev = sys._git

    branch = None
    tag = None
    if ref and ref != 'default':
        kind, sep, name = ref.partition('/')
        if not sep:
            raise NotImplementedError(ref)
        if kind == 'tags':
            branch, tag = _parse_git_tag(name)
        elif kind == 'heads':
            branch = name
        else:
            raise NotImplementedError(ref)

    date = None
    if rev and reporoot:
        proc = subprocess.run(f"git show --format='%aI' --no-patch {rev}",
                              shell=True,
                              capture_output=True,
                              text=True,
                              cwd=reporoot,
                              )
        if proc.returncode == 0:
            date = proc.stdout.strip()
            #date = datetime.datetime.fromisoformat(proc.stdout.strip())

    return {
        'root': reporoot,
        'branch': branch,
        'tag': tag,
        'revision': rev or None,
        'date': date,
    }


def _get_runtime_info(python=None, full=False):
    if python and python != sys.executable:
        argv = [python, __file__, '_dump']
        if full:
            argv.append('--full')
        proc = subprocess.run(argv,
                              text=True,
                              capture_output=True,
                              )
        try:
            proc.check_returncode()
        except subprocess.CalledProcessError as exc:
            print(exc.stderr)
            raise  # re-raise
        return json.loads(proc.stdout)

    version, _git, builddate, compiler = _parse_version(sys.version)

    stdlib = os.path.dirname(os.__file__)
    if os.path.basename(stdlib) == 'Lib':
        base_executable = os.path.join(os.path.dirname(stdlib), 'python')
        if not os.path.exists(base_executable):
            raise NotImplementedError(base_executable)
        isdev = True
    else:
        major, minor, *_ = sys.version_info
        if stdlib == os.path.join(sys.prefix, PLATLIBDIR, f'python{major}.{minor}'):
            base_executable = sys.executable
            isdev = False
        else:
            raise NotImplementedError(stdlib)
    isvenv = sys.prefix != sys.base_prefix

#    isdev = False
#    isvenv = False
#    stdlib = os.path.dirname(os.__file__)
#    if stdlib == os.path.join(os.path.dirname(sys.executable), 'Lib'):
#        isdev = True
#    else:
#        isvenv = sys.prefix != sys.base_prefix
#        if isvenv:
#            if os.__file__ == os.path.join(stdlib, 'os.py'):
#                isdev = True
#    if not isdev:
#        major, minor, *_ = sys.version_info
#        stdlib = os.path.join(sys.prefix, PLATLIBDIR, f'python{major}.{minor}')

    git = _resolve_git(_git, isdev)

    info = {
        'version': version,
        'version_str': sys.version,
        'hexversion': sys.hexversion,
        'apiversion': sys.api_version,
        'implementation': sys.implementation.name,
        'platform': {
            'name': sys.platform,
            'byteorder': sys.byteorder,
        },
        'build': {
            'date': builddate,
            'compiler': compiler,
            'isdev': isdev,
            'git': git,
            'configure_args': None,  # set if full
        },
        'install': {
            'executable': sys.executable,
            'prefix': sys.prefix,
            'exec_prefix': sys.exec_prefix,
            'base_executable': base_executable,
            'base_prefix': sys.base_prefix,
            'base_exec_prefix': sys.base_exec_prefix,
            'stdlib': stdlib,
            'isvenv': isvenv,
        },
        #'distro': 'cpython',
        #'distro': 'system',
    }

    if full:
        configvars = sysconfig.get_config_vars()
        # XXX Pull in everything from sysconfig.get_config_vars()?
        configargs = configvars.get('CONFIG_ARGS')
        if configargs is not None:
            info['build']['configure_args'] = shlex.split(configargs)

    return info


def _get_python_id(python, gitinfo=None):
    m = hashlib.sha256()
    if gitinfo:
        _hash_file('pyconfig.h', m, gitinfo)
        _hash_file(os.path.join('Misc', 'python-config.sh'), m, gitinfo)
        _, rev = gitinfo
        m.update(rev.encode('utf-8'))
    else:
        _hash_file(python, m)
    return m.hexdigest()


def _get_python_config(python, prefix, full=False):
    # XXX
    return None
    ...
    if info['build']['isdev']:
        ...
    proc = subprocess.run([],
                          text=True,
                          capture_output=True,
                          )
    proc.check_returncode()
    lines = proc.stdout.splitlines()



def get_python_info(python, *, full=False):
    info = _get_runtime_info(python, full)
    python = info['install']['executable']

    git = info['build']['git']
    gitinfo = (git['root'], git['revision']) if git['root'] else None
    assert 'id' not in info
    info['id'] = _get_python_id(python, gitinfo)

    assert 'config' not in info
    info['config'] = _get_python_config(python,
                                        info['install']['prefix'],
                                        full,
                                        )

    return info


#############################
# commands

def cmd_show(python, *, fmt='table', full=False):
    info = get_python_info(python, full=full)

    if fmt == 'table':
        # XXX
        import pprint
        pprint.pprint(info)
    elif fmt == 'json':
        json.dump(info, sys.stdout, indent=4)
        print()
    else:
        raise ValueError(f'unsupported fmt {fmt!r}')


def cmd_dump(*, full=False):
    info = _get_runtime_info(full=full)
    json.dump(info, sys.stdout, indent='  ')


COMMANDS = {
    'show': cmd_show,
    '_dump': cmd_dump,
}


#############################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(
            prog=prog,
            )

    subs = parser.add_subparsers(dest='cmd')

    sub = subs.add_parser('show')
    sub.add_argument('--format', dest='fmt', default='table')
    sub.add_argument('--full', action='store_true')
    sub.add_argument('python', nargs='?', default=sys.executable)

    sub = subs.add_parser('_dump')
    sub.add_argument('--full', action='store_true')

    args = parser.parse_args(argv)
    ns = vars(args)

    cmd = ns.pop('cmd')

    return cmd, ns


def main(cmd, cmd_kwargs):
    try:
        run = COMMANDS[cmd]
    except KeyError:
        raise ValueError(f'unsupported cmd {cmd!r}')
    run(**cmd_kwargs)


if __name__ == '__main__':
    cmd, cmd_kwargs = parse_args()
    main(cmd, cmd_kwargs)
