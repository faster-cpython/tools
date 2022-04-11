import os.path
import re
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert os.path.isdir(os.path.join(ROOT, '.git'))
SCRIPT = os.path.join(ROOT, 'scripts', 'add-benchmark-results.py')

BENCH_USER = os.path.expanduser('~benchmarking')
RESULTS_DIR = os.path.join(BENCH_USER, 'BENCH', 'REQUESTS')
RESULTS_DIR_RE = re.compile(rf'^.*({RESULTS_DIR}/req-\d+-\w+)')

RESULTS_REPO = os.path.join(os.path.expanduser('~'), 'faster-cpython-ideas')


def resolve_results_files(results):
    """Return the absolute paths for the given values."""
    filenames = []
    for name in results:
        if name.endswith('.log'):
            with open(name) as infile:
                for line in infile:
                    m = RESULTS_DIR_RE.match(line)
                    if m:
                        dirname, = m.groups()
                        filename = os.path.join(dirname, 'results-data.json.gz')
                        break
                else:
                    raise NotImplementedError(name)
        elif os.path.isabs(name):
            filename = name
        else:
            if name.endswith('.json') or '.json' in os.path.basename(name):
                dirname, basename = os.path.split(name)
            elif os.path.basename(name).startswith('req-'):
                dirname = name
                basename = 'results-data.json.gz'
            else:
                raise NotImplementedError(name)
            # Deal with when it matches RESULTS_DIR.
            if os.path.basename(dirname).startswith('req-'):
                parent, req = os.path.split(dirname)
                if not parent:
                    dirname = os.path.join(RESULTS_DIR, req)
                elif os.path.basename(parent) == 'REQUESTS':
                    parent = os.path.dirname(parent)
                    if not parent:
                        dirname = os.path.join(RESULTS_DIR, req)
                    elif os.path.basename(parent) == 'BENCH':
                        parent = os.path.dirname(parent)
                        if not parent:
                            dirname = os.path.join(RESULTS_DIR, req)
                        # Otherwise leave dirname alone.
            if not dirname:
                dirname = os.path.abspath('.')
            elif not os.path.isabs(dirname):
                dirname = os.path.abspath(dirname)
            filename = os.path.join(dirname, basename)
        filenames.append(filename)
    return filenames


def _run_script(filenames, repo, branch, release, host):
    releasearg = ('--release', release) if release else ()
    hostarg = ('--host', host) if host else ()
    brancharg = ('--branch', branch) if branch else ()
    argv = [
        sys.executable, SCRIPT,
        *releasearg,
        *hostarg,
        '--repo', repo or RESULTS_REPO,
        *brancharg,
        '--upload',
        *filenames,
    ]
    subprocess.run(argv, check=True)


#######################################
# the script

import sys


def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument('--release')
    parser.add_argument('--host')
    parser.add_argument('--repo', default=RESULTS_REPO)
    parser.add_argument('--branch')
    parser.add_argument('results', nargs='+')

    args = parser.parse_args(argv)
    ns = vars(args)

    return ns


def main(results, release=None, host=None, repo=RESULTS_REPO, branch=None):
    filenames = resolve_results_files(results)
    _run_script(filenames, repo, branch, release, host)


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
