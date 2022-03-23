import contextlib
import hashlib
import json
import os
import os.path
import shlex
import shutil
import subprocess
import textwrap
import urllib.parse


GIT = shutil.which('git')

REPO = 'https://github.com/faster-cpython/ideas'
DIRNAME = 'benchmark-results'
BRANCH = 'add-benchmark-results'


def git(cmd, *args, reporoot, capture=False, quiet=False):
    """Run git with the given command."""
    argv = [GIT, cmd, *args]
    if not quiet:
        print(f'# {" ".join(shlex.quote(a) for a in argv)} (cwd: {reporoot})')
    kwargs = dict(
        cwd=reporoot,
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


def ensure_git_branch(reporoot, branch):
    """Switch to the given branch, creating it if necessary."""
    actual = git(
        'rev-parse', '--abbrev-ref', 'HEAD',
        reporoot=reporoot,
        capture=True,
        quiet=True,
    )
    if actual != branch:
        text = git('branch', '--list', reporoot=reporoot, capture=True)
        if branch in text.split():
            # It alrady exists.
            git('checkout', branch)
        else:
            git('checkout', '-b', branch)
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
    cleanup = True
    if filename.endswith('.json'):
        jsonfile = filename
        cleanup = False
    elif filename.endswith('.json.gz'):
        jsonfile = filename[:-3]
        if os.path.exists(jsonfile):
            # XXX Make sure it matches?
            cleanup = False
        else:
            with gzip.open(filename) as infile:
                with open(jsonfile, 'w') as outfile:
                    shutil.copyfileobj(infile, outfile)
    else:
        raise NotImplementedError(repr(filename))
    try:
        yield jsonfile
    finally:
        if cleanup:
            os.unlink(jsonfile)


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
        metadata['perf_version'],
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


def add_results_to_local(reporoot, name, localfile, *, branch=BRANCH):
    """Add the file to a local repo using the given name."""
    target = os.path.join(reporoot, DIRNAME, name)
    if os.path.exists(target):
        # XXX ignore if the same?
        raise Exception(f'{target} already exists')
    # Note that we do not switch the branch back when we are done.
    ensure_git_branch(reporoot, branch)
    shutil.copyfile(localfile, target)
    git('add', target, reporoot=reporoot)
    git('commit', '-m', f'Add Benchmark Results ({name})', reporoot=reporoot)
    return textwrap.dedent(f'''
        DONE: added benchmark results to local repo
        from: {localfile}
        to:   repo at {reporoot}
        as:   {DIRNAME}/{name}

        (Now you may push to your GitHub clone and make a pull request.)
    ''').strip()


def add_results_to_remote(url, name, localfile, *, branch=BRANCH):
    # XXX Possible solutions:
    # * directly add the file using the GH API
    #   (see https://gist.github.com/sumit-s03/64f7e160db5195bc242368120d247317)
    # * create a pull request using the GH API using the given repo root
    # * create a pull request using the GH API using a temporary local clone
    # * create a pull request using just the GH API (is this possible?)
    raise NotImplementedError


def add_results(localfile, remotename, *, repo=REPO, branch=BRANCH):
    repo, isremote = resolve_repo(repo)
    if isremote:
        return add_results_to_remote(repo, remotename, localfile, branch=branch)
    else:
        return add_results_to_local(repo, remotename, localfile, branch=branch)


#######################################
# the script

import sys


def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument('--release')
    parser.add_argument('--host')
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--branch', default=BRANCH)
    parser.add_argument('filename')

    args = parser.parse_args(argv)
    ns = vars(args)

    return ns


def main(filename, release=None, host=None, repo=REPO, branch=BRANCH):
    with ensure_json(filename) as localfile:
        with open(localfile) as infile:
            text = infile.read()
        metadata = parse_metadata(text)
        name = get_uploaded_name(metadata, release, host)
        msg = add_results(
            localfile,
            name,
            repo=repo,
            branch=branch,
        )
    print()
    print(msg)


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
