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


@contextlib.contextmanager
def ensure_json(filename):
    """Return the filename of the corresponding plain JSON file."""
    # We trust the file suffix.
    tmpdir = None
    if filename.endswith('.json'):
        jsonfile = filename
    elif filename.endswith('.json.gz'):
        jsonfile = filename[:-3]
        if not os.path.exists(jsonfile):
            tmpdir = tempfile.TemporaryDirectory()
            jsonfile = os.path.join(tmpdir.name, os.path.basename(jsonfile))
            with gzip.open(filename, 'rb') as infile:
                with open(jsonfile, 'wb') as outfile:
                    shutil.copyfileobj(infile, outfile)
        # XXX Otherwise, make sure it matches?
    else:
        raise NotImplementedError(repr(filename))
    try:
        yield jsonfile
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def parse_metadata(data):
    """Return the metadata corresponding to the given results."""
    if isinstance(data, str):
        data = json.loads(data)
    if data['version'] == '1.0':
        return data['metadata']
    else:
        raise NotImplementedError(data['version'])


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


def get_uploaded_name(metadata, release=None, host=None):
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
    return f'{implname}-{release}-{commit[:10]}-{host}-{compat}.json'


def prepare_repo(repo, branch=BRANCH):
    """Get the repo ready before adding results."""
    repo, isremote = resolve_repo(repo)

    if isremote:
        raise NotImplementedError
    else:
        # Note that we do not switch the branch back when we are done.
        ensure_git_branch(repo, branch)

    return repo, isremote


def add_results_to_local(reporoot, name, localfile, *, branch=BRANCH):
    """Add the file to a local repo using the given name."""
    reltarget = os.path.join(DIRNAME, name)
    target = os.path.join(reporoot, reltarget)
    if os.path.exists(target):
        # XXX ignore if the same?
        raise Exception(f'{target} already exists')
    shutil.copyfile(localfile, target)
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
        with ensure_json(filename) as localfile:
            with open(localfile) as infile:
                text = infile.read()
            metadata = parse_metadata(text)
            _host = resolve_host(host, metadata)
            name = get_uploaded_name(metadata, release, _host)
            msg = add_results(repo, name, localfile, branch=branch)
        print()
        print(msg)
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
