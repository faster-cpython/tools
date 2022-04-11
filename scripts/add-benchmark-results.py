import contextlib
import gzip
import hashlib
import json
import os
import os.path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.parse


GIT = shutil.which('git')

REPO = 'https://github.com/faster-cpython/ideas'
DIRNAME = 'benchmark-results'
BRANCH = 'add-benchmark-results'

HOME = os.path.expanduser('~')
REPOS_DIR = os.path.join(HOME, 'repos')
BENCHMARKS = {
    'pyperformance': {
        'url': 'https://github.com/python/pyperformance',
        'reldir': 'pyperformance/data-files/benchmarks',
    },
    'pyston': {
        'url': 'https://github.com/pyston/python-macrobenchmarks',
        'reldir': 'benchmarks',
    },
}


def get_os_name(metadata=None):
    if metadata:
        platform = metadata['platform'].lower()
        if 'linux' in platform:
            return 'linux'
        elif 'darwin' in platform or 'macos' in platform or 'osx' in platform:
            return 'mac'
        elif 'win' in platform:
            return 'windows'
        else:
            raise NotImplementedError(platform)
    else:
        if sys.platform == 'win32':
            return 'windows'
        elif sys.paltform == 'linux':
            return 'linux'
        elif sys.platform == 'darwin':
            return 'mac'
        else:
            raise NotImplementedError(sys.platform)


def get_arch(metadata=None):
    if metadata:
        platform = metadata['platform'].lower()
        if 'x86_64' in platform:
            return 'x86_64'
        elif 'amd64' in platform:
            return 'amd64'

        procinfo = metadata['cpu_model_name'].lower()
        if 'aarch64' in procinfo:
            return 'arm64'
        elif 'arm' in procinfo:
            if '64' in procinfo:
                return 'arm64'
            else:
                return 'arm32'
        elif 'intel' in procinfo:
            return 'x86_64'
        else:
            raise NotImplementedError((platform, procinfo))
    else:
        uname = platform.uname()
        machine = uname.machine.lower()
        if machine in ('amd64', 'x86_64'):
            return machine
        elif machine == 'aarch64':
            return 'arm64'
        elif 'arm' in machine:
            return 'arm'
        else:
            raise NotImplementedError(machine)


def resolve_host(host, metadata=None):
    """Return the best string to use as a label for the host."""
    if host:
        return host
    # We could use metadata['hostname'] but that doesn't
    # make a great label in the default case.
    host = get_os_name(metadata)
    arch = get_arch(metadata)
    if arch in ('arm32', 'arm64'):
        host += '-arm'
    # Ignore everything else.
    return host


def git(cmd, *args, root, capture=False, quiet=False):
    """Run git with the given command."""
    argv = [GIT, cmd, *args]
    if not quiet:
        print(f'# {" ".join(shlex.quote(a) for a in argv)} (cwd: {root})')
    kwargs = dict(
        cwd=root,
        check=True,
    )
    if capture:
        kwargs.update(dict(
            stdout=subprocess.PIPE,
            encoding='utf-8',
        ))
        proc = subprocess.run(argv, **kwargs)
        return (proc.stdout or '').strip()
    else:
        subprocess.run(argv, **kwargs)
        return ''


def ensure_git_branch(root, branch):
    """Switch to the given branch, creating it if necessary."""
    actual = git(
        'rev-parse', '--abbrev-ref', 'HEAD',
        root=root,
        capture=True,
        quiet=True,
    )
    if actual != branch:
        text = git('branch', '--list', root=root, capture=True)
        if branch in text.split():
            # It alrady exists.
            git('checkout', branch, root=root)
        else:
            git('checkout', 'main', root=root)
            git('checkout', '-b', branch, root=root)
    # else we're already there so do nothing.


def resolve_repo(repo):
    """Return (actual, isremote) for the given raw value."""
    if not repo:
        return REPO, True
    # This is a best-effort.
    parsed = urllib.parse.urlparse(repo)
    if parsed.scheme:
        return repo, True
    elif parsed.path != repo:
        raise NotImplementedError(repr(repo))
    elif not os.path.exists(repo):
        git('clone', REPO, repo)
    elif not os.path.isdir(repo):
        raise NotADirectoryError(repo)
    else:
        return repo, False


def read_results(filename):
    """Read the benchmark results from the given file."""
    if filename.endswith('.json'):
        _open = open
    elif filename.endswith('.json.gz'):
        _open = gzip.open
    else:
        raise NotImplementedError(filename)
    with _open(filename) as infile:
        results = json.load(infile)
    if results['version'] == '1.0':
        return results
    else:
        raise NotImplementedError(results['version'])


def get_compat_id(metadata, *, short=True):
    """Return a unique ID corresponding to the given results metadata."""
    data = [
        metadata['hostname'],
        metadata['platform'],
        metadata.get('perf_version'),
        metadata['performance_version'],
        metadata['cpu_model_name'],
        metadata.get('cpu_freq'),
        metadata['cpu_config'],
        metadata.get('cpu_affinity'),
    ]

    h = hashlib.sha256()
    for value in data:
        if not value:
            continue
        h.update(value.encode('utf-8'))
    compat = h.hexdigest()
    if short:
        compat = compat[:12]
    return compat


def get_benchmarks(*, _cache={}):
    """Return the per-suite lists of benchmarks."""
    benchmarks = {}
    for suite, info in BENCHMARKS.items():
        if suite in _cache:
            benchmarks[suite] = list(_cache[suite])
            continue
        url = info['url']
        reldir = info['reldir']
        reporoot = os.path.join(REPOS_DIR, os.path.basename(url))
        if not os.path.exists(reporoot):
            if not os.path.exists(REPOS_DIR):
                os.makedirs(REPOS_DIR)
            git('clone', url, reporoot, root=None)
        names = _get_benchmark_names(os.path.join(reporoot, reldir))
        benchmarks[suite] = _cache[suite] = names
    return benchmarks


def _get_benchmark_names(benchmarksdir):
    manifest = os.path.join(benchmarksdir, 'MANIFEST')
    if os.path.isfile(manifest):
        with open(manifest) as infile:
            for line in infile:
                if line.strip() == '[benchmarks]':
                    for line in infile:
                        if line.strip() == 'name\tmetafile':
                            break
                    else:
                        raise NotImplementedError(manifest)
                    break
            else:
                raise NotImplementedError(manifest)
            for line in infile:
                if line.startswith('['):
                    break
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                name, _ = line.split('\t')
                yield name
    else:
        for name in os.listdir(benchmarksdir):
            if name.startswith('bm_'):
                yield name[3:]


def split_benchmarks(results):
    """Return results collated by suite."""
    split = {}
    benchmarks = get_benchmarks()
    by_name = {}
    for suite, names in benchmarks.items():
        for name in names:
            if name in by_name:
                raise NotImplementedError((suite, name))
            by_name[name] = suite
    for data in results['benchmarks']:
        name = data['metadata']['name']
        try:
            suite = by_name[name]
        except KeyError:
            # Some benchmarks actually produce results for
            # sub-benchmarks (e.g. logging -> logging_simple).
            _name = name
            while '_' in _name:
                _name, _, _ = _name.rpartition('_')
                if _name in by_name:
                    suite = by_name[_name]
                    break
            else:
                suite = 'unknown'
        if suite not in split:
            split[suite] = {k: v
                            for k, v in results.items()
                            if k != 'benchmarks'}
            split[suite]['benchmarks'] = []
        split[suite]['benchmarks'].append(data)
    return split


def get_uploaded_name(metadata, release=None, host=None, suite=None):
    """Return the base filename to use for the given results metadata.

    See https://github.com/faster-cpython/ideas/tree/main/benchmark-results/README.md
    for details on this filename format.
    """
    implname = 'cpython'
    if not release:
        release = 'main'
    commit = metadata.get('commit_id')
    if not commit:
        raise NotImplementedError
    if not host:
        host = metadata['hostname']
    compat = get_compat_id(metadata)
    suite = f'-{suite}' if suite and suite != 'pyperformance' else ''
    return f'{implname}-{release}-{commit[:10]}-{host}-{compat}{suite}.json'


def prepare_repo(repo, branch=BRANCH):
    """Get the repo ready before adding results."""
    repo, isremote = resolve_repo(repo)

    if isremote:
        raise NotImplementedError
    else:
        # Note that we do not switch the branch back when we are done.
        ensure_git_branch(repo, branch)

    return repo, isremote


def add_results_to_local(results, reporoot, name, localfile, *, branch=BRANCH):
    """Add the file to a local repo using the given name."""
    reltarget = os.path.join(DIRNAME, name)
    target = os.path.join(reporoot, reltarget)
    if os.path.exists(target):
        # XXX ignore if the same?
        raise Exception(f'{target} already exists')
    with open(target, 'w') as outfile:
        json.dump(results, outfile, indent=2)
    git('add', reltarget, root=reporoot)
    git('commit', '-m', f'Add Benchmark Results ({name})', root=reporoot)
    return textwrap.dedent(f'''
        DONE: added benchmark results to local repo
        from: {localfile}
        to:   repo at {reporoot}
        as:   {DIRNAME}/{name}
    ''').strip()


def add_results_to_remote(url, name, localfile, *, branch=BRANCH):
    # XXX Possible solutions:
    # * directly add the file using the GH API
    #   (see https://gist.github.com/sumit-s03/64f7e160db5195bc242368120d247317)
    # * create a pull request using the GH API using the given repo root
    # * create a pull request using the GH API using a temporary local clone
    # * create a pull request using just the GH API (is this possible?)
    raise NotImplementedError


#######################################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument('--release')
    parser.add_argument('--host')
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--branch', default=BRANCH)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('filenames', nargs='+')

    args = parser.parse_args(argv)
    ns = vars(args)

    return ns


def main(filenames, *,
         release=None,
         host=None,
         repo=REPO,
         branch=BRANCH,
         upload=False,
         ):
    repo, isremote = prepare_repo(repo, branch)
    add_results = add_results_to_remote if isremote else add_results_to_local
    assert filenames
    for filename in filenames:
        print()
        print('#' * 40)
        print(f'adding {filename} to repo at {repo}...')
        print()
        results = read_results(filename)
        metadata = results['metadata']
        _host = resolve_host(host, metadata)
        split = split_benchmarks(results)
        if 'other' in split:
            raise NotImplementedError(sorted(split))
        for suite in split:
            name = get_uploaded_name(metadata, release, _host, suite)
            msg = add_results(split[suite], repo, name, filename, branch=branch)
            print()
            print(msg)
            print()

    if not isremote:
        # XXX Optionally create the pull request?
        if upload:
            git('push', root=repo)
            print()
            print('(Now you may make a pull request.)')
        else:
            print()
            print('(Now you may push to your GitHub clone and make a pull request.)')
        ghuser = '<your GH user>'
        print('({REPO}/compare/main...{ghuser}:{branch}?expand=1)')


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
