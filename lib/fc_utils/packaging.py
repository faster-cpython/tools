import os.path
import subprocess
import sys


def get_requirements(projectroot):
    # XXX Try other locations/sources too.
    lockfile = os.path.join(projectroot, 'requirements.txt')
    try:
        return read_requirements(lockfile)
    except FileNotFoundError:
        return []


def read_requirements(lockfile):
    if isinstance(lockfile, str):
        filename = lockfile
        with open(filename) as lockfile:
            return read_requirements(lockfile)

    reqs = []
    for line in lockfile:
        line = line.partition('#')[0].strip()
        if not line:
            continue
        reqs.append(line)
    return reqs


def ensure_requirements(reqs, python=sys.executable, *,
                        showdeps=False,
                        dryrun=False,
                        ):
    if os.path.isdir(python):
        raise NotImplementedError(python)

    if not reqs:
        logger.info('no requirements to install')
        return

    logger.info(f'installing {len(reqs)} requirements: {" ".join(sorted(reqs))}')
    if not dryrun:
        _ensure_reqs(reqs, python)


def _ensure_reqs(reqs, python):
    subprocess.run(
        [python, '-m', 'pip', 'install', '-U', *reqs],
        check=True,
    )


def get_dependencies(reqs, python=None):
    if not python:
        python = sys.executable
    deps = []
    for req in reqs:
        for c in '<=>;':
            req = req.partition(c)[0]
        text = _get_dependencies(python, req, stdout=subprocess.PIPE)
        # XXX Parse it.
        raise NotImplementedError
    return deps



def show_dependencies(reqs, python=None, *, dryrun=False):
    if not python:
        python = sys.executable

    print('dependencies:')
#    python = _venv.resolve_venv_file(venvroot, 'bin', 'python')
    for req in reqs:
        for c in '<=>;':
            req = req.partition(c)[0]
        if dryrun:
            print(' ', req)
        else:
            print()
            _get_dependencies(python, req)


def _get_dependencies(python, req, **kwargs):
    proc = subprocess.run(
        [sys.executable, '-m', 'pipdeptree', '--python', python, '-p', req],
        check=True,
        text=True,
        **kwargs
    )
    return proc.stdout
