import os.path
import subprocess
import sys


SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
CPYTHON_SCRIPT = os.path.join(SCRIPTS_DIR, 'cpython.sh')
PROFILER_SCRIPT = os.path.join(SCRIPTS_DIR, 'run-profiler.sh')

DATA_DIR = os.path.join(os.path.expanduser('~'), 'perf-data')
PYTHON = os.path.join(DATA_DIR, 'cpython', 'python')  # See cpython.sh.

VERBOSITY = 3


def build_python_command(build, *, nosite=False, **_ignored):
    if nosite:
        return f'{PYTHON} -S -c pass'
    else:
        return f'{PYTHON} -c pass'


def get_python_tags(build, *, nosite=False, **_ignored):
    tags = ''

    if build and build != 'release':
        tags = f'{tags}-{build}'

    if nosite:
        tags = f'{tags}-nosite'

    return tags


def needs_python(cmd, cmd_kwargs):
    if cmd == 'flamegraph':
        if cmd_kwargs.get('datafileonly'):
            return False
        return True
    else:
        raise NotImplementedError('cmd')


def build_python(build, *,
                 datadir=None,
                 optlevel=0,  # unoptimized
                 prefix=None,
                 force=False,
                 verbose=False,
                 ):
    #verbose = True
    argv = [
        CPYTHON_SCRIPT,
        #'--prep',
        '--datadir', datadir or '-',
        '--build', build or '-',
        '--optlevel', str(optlevel) if optlevel is not None else '-',
        '--prefix', prefix or '-',
    ]
    if force:
        argv.append('--force')
    if verbose:
        argv.append('--verbose')

    #print(f'# {" ".join(argv)}')
    proc = subprocess.run(argv)
    return proc.returncode == 0


#######################################
# commands

def cmd_flamegraph(tool, pycmd, pytags, *,
                   datadir=None,
                   stamp=False,
                   datafileonly=False,
                   upload=False,
                   **tool_kwargs):
    argv = [
        PROFILER_SCRIPT,
        tool,
        '--datadir', datadir or '-',
        '--stamp' if stamp else '--no-stamp',
        '--tags', pytags or '-',
        '--upload' if upload else '--no-upload',
    ]

    if tool == 'perf':
        frequency = tool_kwargs.pop('frequency')
        if frequency:
            argv.extend([
                '--frequency', str(frequency),
            ])
    if tool_kwargs:
        raise RuntimeError(f'unused tool kwargs: {tool_kwargs}')

    if datafileonly:
        argv.extend([
            '--datafile-only',
            *'/dev/null this will not get run'.split(),
        ])
    else:
        # We could call shlex.split() and then argv.extend() but this is fine.
        argv.append(pycmd)

        print()
        print(f'*** profiling {pycmd} (using {tool}) ***')

    #print(f'# {" ".join(argv)}')
    proc = subprocess.run(argv)
    if proc.returncode != 0:
        exit('ERROR: profiler command failed')


COMMANDS = {
    'flamegraph': cmd_flamegraph,
}


#######################################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-v', '--verbose',
                        action='store_const', const=1, default=0)
    common.add_argument('-q', '--quiet',
                        action='store_const', const=1, default=0)

    datacommon = argparse.ArgumentParser(add_help=False)
    datacommon.add_argument('--datadir')
    datacommon.add_argument('--stamp', action='store_true')
    datacommon.add_argument('--no-stamp', dest='stamp', action='store_false')
    datacommon.set_defaults(stamp=False)

    pycommon = argparse.ArgumentParser(add_help=False)
    #pycommon = argparse.ArgumentParser(parents=[datacommon], add_help=False)
    pycommon.add_argument('--rebuild', dest='pyrebuild',
                          action='store_true')
    pycommon.add_argument('--no-rebuild', dest='pyrebuild',
                          action='store_false')
    pycommon.set_defaults(pyrebuild=None)
    pycommon.add_argument('--release', dest='pybuild',
                          action='store_const', const='release')
    pycommon.add_argument('--debug', dest='pybuild',
                          action='store_const', const='debug')
    pycommon.set_defaults(pybuild='release')
    pycommon.add_argument('--site', dest='nosite', action='store_false')
    pycommon.add_argument('--no-site', dest='nosite', action='store_true')
    pycommon.set_defaults(nosite=False)

    profcommon = argparse.ArgumentParser(add_help=False)
    #profcommon = argparse.ArgumentParser(parents=[datacommon], add_help=False)
    profcommon.add_argument('--datafile-only', dest='datafileonly',
                            action='store_true')

    uploadcommon = argparse.ArgumentParser(add_help=False)
    #uploadcommon = argparse.ArgumentParser(parents=[datacommon], add_help=False)
    uploadcommon.add_argument('--upload', action='store_true')
    uploadcommon.add_argument('--no-upload', dest='upload', action='store_false')
    uploadcommon.set_defaults(upload=False)

    parser = argparse.ArgumentParser(
        prog=prog,
        parents=[common],
    )

    subs = parser.add_subparsers(dest='cmd')

    sub = subs.add_parser(
        'flamegraph',
        parents=[common, datacommon, pycommon, profcommon, uploadcommon],
    )
    fgsubs = sub.add_subparsers(dest='tool')

    fgsub = fgsubs.add_parser(
        'perf',
        parents=[common, datacommon, pycommon, profcommon, uploadcommon],
    )
    fgsub.add_argument('--frequency', type=int)

    fgsub = fgsubs.add_parser(
        'uftrace',
        parents=[common, datacommon, pycommon, profcommon, uploadcommon],
    )

    args = parser.parse_args(argv)
    ns = vars(args)

    verbosity = max(0,
                    VERBOSITY + ns.pop('verbose') - ns.pop('quiet'))

    cmd = ns.pop('cmd')

    needspy = needs_python(cmd, ns)
    if needspy or 'pybuild' in ns:
        pybuild = ns.pop('pybuild')
        nosite = ns.pop('nosite')
        args.pycmd = build_python_command(pybuild, nosite=nosite)
        args.pytags = get_python_tags(pybuild, nosite=nosite)

        pyrebuild = ns.pop('pyrebuild')
        if pyrebuild is False:
            pybuild = None
    else:
        assert 'pyrebuild' not in ns
    if not needspy:
        pybuild = pyrebuild = None

    if 'tool' in ns:
        if not args.tool:
            parser.error('missing tool arg')

    return cmd, ns, pybuild, pyrebuild, verbosity


def main(cmd, cmd_kwargs, *, pybuild=None, pyrebuild=False):
    try:
        run_cmd = COMMANDS[cmd]
    except KeyError:
        raise ValueError(f'unsupported cmd {cmd!r}')

    if pybuild:
        datadir = cmd_kwargs.get('datadir')
        if not build_python(pybuild, datadir=datadir, force=pyrebuild):
            sys.exit('ERROR: failed to build Python')

    run_cmd(**cmd_kwargs)


if __name__ == '__main__':
    cmd, cmd_kwargs, pybuild, pyrebuild, verbosity = parse_args()
    main(cmd, cmd_kwargs, pybuild=pybuild, pyrebuild=pyrebuild)
