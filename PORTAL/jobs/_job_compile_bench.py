import configparser
import logging
import os.path
import textwrap

from . import JobKind
from . import _utils, _pyperformance
from .requests import Request, Result


FAKE_DELAY = 3

logger = logging.getLogger(__name__)


class CompileBenchRequest(Request):

    FIELDS = Request.FIELDS + [
        'ref',
        'pyperformance_ref',  # XXX Should be required instead of ref.
        'remote',
        'revision',
        'branch',
        'benchmarks',
        'optimize',
        'debug',
    ]
    OPTIONAL = [
        'pyperformance_ref',
        'remote',
        'revision',
        'branch',
        'benchmarks',
        'optimize',
        'debug',
    ]

    CPYTHON = _utils.GitHubTarget.from_origin('python', 'cpython')
    PYPERFORMANCE = _utils.GitHubTarget.from_origin('python', 'pyperformance')
    PYSTON_BENCHMARKS = _utils.GitHubTarget.from_origin('pyston', 'python-macrobenchmarks')

    #pyperformance = PYPERFORMANCE.copy('034f58b')  # 1.0.4 release (2022-01-26)
    pyperformance = PYPERFORMANCE.copy('5b6142e')  # will be 1.0.5 release
    pyston_benchmarks = PYSTON_BENCHMARKS.copy('96e7bb3')  # main from 2022-01-21
    #pyperformance = PYPERFORMANCE.fork('ericsnowcurrently', 'python-performance', 'benchmark-management')
    #pyston_benchmarks = PYSTON_BENCHMARKS.fork('ericsnowcurrently', 'pyston-macrobenchmarks', 'pyperformance')

    @classmethod
    def _extract_kwargs(cls, data, optional, filename):
        # This is a backward-compatibility shim.
        try:
            return super()._extract_kwargs(data, optional, filename)
        except ValueError:
            optional = [*optional, 'datadir', 'date', 'ref', 'user']
            kwargs, extra = super()._extract_kwargs(data, optional, filename)
            reqid = RequestID.from_raw(kwargs['id'])
            kwargs.setdefault('datadir', os.path.dirname(filename))
            kwargs.setdefault('date', reqid.date.isoformat())
            kwargs.setdefault('user', reqid.user)
            kwargs.setdefault('ref', 'deadbeef')
            return kwargs, extra

    def __init__(self,
                 id,
                 datadir,
                 ref,
                 pyperformance_ref=None,
                 remote=None,
                 revision=None,
                 branch=None,
                 benchmarks=None,
                 optimize=True,
                 debug=False,
                 **kwargs
                 ):
        if remote and not _utils.looks_like_git_remote(remote):
            raise ValueError(remote)
        if branch and not _utils.looks_like_git_branch(branch):
            raise ValueError(branch)

        super().__init__(id, datadir, **kwargs)

        if isinstance(ref, str):
            fast = True
            if fast:
                tag = commit = None
                if ref and _utils.looks_like_git_commit(ref):
                    commit = ref
                try:
                    ref = _utils.GitRef.from_values(remote, branch, tag, commit, ref)
                except ValueError:
                    # backward compatibility
                    GR = _utils.GitRef
                    ref = GR.__new__(GR, remote, branch, tag, commit, ref, None)
            else:
                refstr = ref
                ref = _utils.GitRef.resolve(revision, branch, remote)
                if refstr not in (ref.commit, ref.branch, ref.tag, None):
                    raise ValueError(f'unexpected ref {refstr!r}')
        else:
            ref = _utils.GitRef.from_raw(ref)

        self.ref = ref
        self.pyperformance_ref = pyperformance_ref or str(ref)
        self.remote = ref.remote
        self.revision = revision
        self.branch = ref.branch
        self.benchmarks = benchmarks
        self.optimize = True if optimize is None else optimize
        self.debug = debug
        self._impl = _utils.CPython()

    @property
    def cpython(self):
        # XXX Pass self.ref directly?
        ref = str(self.ref)
        if self.remote and self.remote != 'origin':
            return self.CPYTHON.fork(self.remote, ref=ref)
        else:
            return self.CPYTHON.copy(ref=ref)

    @property
    def release(self):
        if self.remote == 'origin':
            if not self.branch:
                release = 'main'
                #raise NotImplementedError
            elif self.branch == 'main':
                release = 'main'
            elif self._impl.parse_version(self.branch):
                tag = self.ref.tag
                if tag:
                    ver = self._impl.parse_version(tag)
                    if not ver:
                        raise NotImplementedError(tag)
                    release = str(ver)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError(self.branch)
        else:
            raise NotImplementedError(self.remote)
        return release

    @property
    def versionstr(self):
        return _utils.CPythonVersion.render_extended(
            version=self.release,
            bits=64,
            commit=self.ref.commit[:10],
        )
        return f'{version} ({bits}-bit) revision {commit}'

    @property
    def result(self):
        return CompileBenchResult(self.id, self.reqdir)


class CompileBenchResult(Result):

    FIELDS = Result.FIELDS + [
        'pyperformance_results',
        'pyperformance_results_orig',
    ]
    OPTIONAL = [
        'pyperformance_results',
        'pyperformance_results_orig',
    ]

    @classmethod
    def _extract_kwargs(cls, data, optional, filename):
        # This is a backward-compatibility shim.
        try:
            return super()._extract_kwargs(data, optional, filename)
        except ValueError:
            optional = [*optional, 'reqdir', 'history']
            kwargs, extra = super()._extract_kwargs(data, optional, filename)
            kwargs.setdefault('reqdir', os.path.dirname(filename))
            kwargs.setdefault('history', None)
            return kwargs, extra

    def __init__(self, reqid, reqdir, *,
                 status=None,
                 #pyperformance_results=None,
                 #pyperformance_results_orig=None,
                 **kwargs
                 ):
        super().__init__(reqid, reqdir, status, **kwargs)
        #self.pyperformance_results = pyperformance_results
        #self.pyperformance_results_orig = pyperformance_results_orig

    @property
    def pyperf(self):
        try:
            return self._pyperf
        except AttributeError:
            filename = self.fs.pyperformance_results
            resfile = _pyperformance.PyperfResultsFile(
                filename,
                resultsroot=self.fs.resultsroot,
            )
#            self._pyperf = resfile.read(self.host, self.request.release)
            results = resfile.read()

            modified = False
            PMD = _pyperformance.PyperfResultsMetadata
            if PMD.overwrite_raw(results.data, 'hostid', self.host):
                modified = True
            pyversion = self.request.versionstr
            if PMD.overwrite_raw_all(results.data, 'python_version', pyversion):
                modified = True
            if modified:
                resfile.write(results)

            self._pyperf = results
            return self._pyperf


def resolve_compile_bench_request(reqid, workdir, remote, revision, branch,
                                  benchmarks,
                                  *,
                                  optimize,
                                  debug,
                                  ):
    if isinstance(benchmarks, str):
        benchmarks = benchmarks.replace(',', ' ').split()
    if benchmarks:
        benchmarks = (b.strip() for b in benchmarks)
        benchmarks = [b for b in benchmarks if b]

    ref = _utils.GitRef.resolve(revision, branch, remote)
    if not ref:
        raise Exception(f'could not find ref for {(remote, branch, revision)}')
    assert ref.commit, repr(ref)

#    if not branch and ref.branch == revision:
#        revision = 'latest'

    meta = CompileBenchRequest(
        id=reqid,
        datadir=workdir,
        ref=ref,
        pyperformance_ref=ref.commit,
        remote=remote or None,
        revision=revision or None,
        branch=branch or None,
        benchmarks=benchmarks or None,
        optimize=bool(optimize),
        debug=bool(debug),
    )
    return meta


def build_pyperformance_manifest(req, bfiles):
    return textwrap.dedent(f'''
        [includes]
        <default>
        {bfiles.repos.pyston_benchmarks}/benchmarks/MANIFEST
    '''[1:-1])


def build_pyperformance_config(req, bfiles):
    cpython = bfiles.repos.cpython
    bfiles = bfiles.resolve_request(req.id)
    cfg = configparser.ConfigParser()

    cfg['config'] = {}
    cfg['config']['json_dir'] = bfiles.result.root
    cfg['config']['debug'] = str(req.debug)
    # XXX pyperformance should be looking in [scm] for this.
    cfg['config']['git_remote'] = req.remote

    cfg['scm'] = {}
    cfg['scm']['repo_dir'] = cpython
    cfg['scm']['git_remote'] = req.remote
    cfg['scm']['update'] = 'True'

    cfg['compile'] = {}
    cfg['compile']['bench_dir'] = bfiles.work.scratch_dir
    cfg['compile']['pgo'] = str(req.optimize)
    cfg['compile']['lto'] = str(req.optimize)
    cfg['compile']['install'] = 'True'

    cfg['run_benchmark'] = {}
    cfg['run_benchmark']['manifest'] = bfiles.request.pyperformance_manifest
    cfg['run_benchmark']['benchmarks'] = ','.join(req.benchmarks or ())
    cfg['run_benchmark']['system_tune'] = 'True'
    cfg['run_benchmark']['upload'] = 'False'

    return cfg


def build_compile_script(req, bfiles, fake=None):
    fakedelay = FAKE_DELAY
    if fake is False or fake is None:
        fake = (None, None)
    elif fake is True:
        fake = (0, None)
    elif isinstance(fake, (int, str)):
        fake = (fake, None)
    exitcode, fakedelay = fake
    if fakedelay is None:
        fakedelay = FAKE_DELAY
    else:
        fakedelay = _utils.ensure_int(fakedelay, min=0)
        if exitcode is None:
            exitcode = 0
        elif exitcode == '':
            logger.warn(f'fakedelay ({fakedelay}) will not be used')
    if exitcode is None:
        exitcode = ''
    elif exitcode != '':
        exitcode = _utils.ensure_int(exitcode, min=0)
        logger.warn('we will pretend pyperformance will run with exitcode %s', exitcode)
    python = 'python3.9'  # On the bench host.
    numjobs = 20

    _utils.check_shell_str(str(req.id) if req.id else '')
    _utils.check_shell_str(req.cpython.url)
    _utils.check_shell_str(req.cpython.remote)
    _utils.check_shell_str(req.pyperformance.url)
    _utils.check_shell_str(req.pyperformance.remote)
    _utils.check_shell_str(req.pyston_benchmarks.url)
    _utils.check_shell_str(req.pyston_benchmarks.remote)
    branch = req.branch
    _utils.check_shell_str(branch, required=False)
    maybe_branch = branch or ''
    ref = _utils.check_shell_str(req.pyperformance_ref)

    cpython_repo = _utils.quote_shell_str(bfiles.repos.cpython)
    pyperformance_repo = _utils.quote_shell_str(bfiles.repos.pyperformance)
    pyston_benchmarks_repo = _utils.quote_shell_str(bfiles.repos.pyston_benchmarks)

    bfiles = bfiles.resolve_request(req.id)
    _utils.check_shell_str(bfiles.work.pidfile)
    _utils.check_shell_str(bfiles.work.logfile)
    _utils.check_shell_str(bfiles.work.scratch_dir)
    _utils.check_shell_str(bfiles.request.pyperformance_config)
    _utils.check_shell_str(bfiles.result.pyperformance_log)
    _utils.check_shell_str(bfiles.result.metadata)
    _utils.check_shell_str(bfiles.result.pyperformance_results)
    _utils.check_shell_str(bfiles.work.pyperformance_results_glob)

    _utils.check_shell_str(python)

    # XXX Kill any zombie job processes?

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # This script runs only on the bench host.

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        #####################
        # Mark the result as running.

        echo "$$" > {bfiles.work.pidfile}

        status=$(jq -r '.status' {bfiles.result.metadata})
        if [ "$status" != 'activated' ]; then
            2>&1 echo "ERROR: expected activated status, got $status"
            2>&1 echo "       (see {bfiles.result.metadata})"
            exit 1
        fi

        ( set -x
        jq --arg date $(date -u -Iseconds) '.history += [["running", $date]]' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
        mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}
        )

        #####################
        # Ensure the dependencies.

        if [ ! -e {cpython_repo} ]; then
            ( set -x
            git clone https://github.com/python/cpython {cpython_repo}
            )
        fi
        if [ ! -e {pyperformance_repo} ]; then
            ( set -x
            git clone https://github.com/python/pyperformance {pyperformance_repo}
            )
        fi
        if [ ! -e {pyston_benchmarks_repo} ]; then
            ( set -x
            git clone https://github.com/pyston/python-macrobenchmarks {pyston_benchmarks_repo}
            )
        fi

        #####################
        # Get the repos are ready for the requested remotes and revisions.

        remote='{req.cpython.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {cpython_repo} remote add {req.cpython.remote} {req.cpython.url}
            git -C {cpython_repo} fetch --tags {req.cpython.remote}
            )
        fi
        # Get the upstream tags, just in case.
        ( set -x
        git -C {cpython_repo} fetch --tags origin
        )
        branch='{maybe_branch}'
        if [ -n "$branch" ]; then
            if ! ( set -x
                git -C {cpython_repo} checkout -b {branch or '$branch'} --track {req.cpython.remote}/{branch or '$branch'}
            ); then
                echo "It already exists; resetting to the right target."
                ( set -x
                git -C {cpython_repo} checkout {branch or '$branch'}
                git -C {cpython_repo} reset --hard {req.cpython.remote}/{branch or '$branch'}
                )
            fi
        fi

        remote='{req.pyperformance.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {pyperformance_repo} remote add {req.pyperformance.remote} {req.pyperformance.url}
            )
        fi
        ( set -x
        git -C {pyperformance_repo} fetch --tags {req.pyperformance.remote}
        git -C {pyperformance_repo} checkout {req.pyperformance.fullref}
        )

        remote='{req.pyston_benchmarks.remote}'
        if [ "$remote" != 'origin' ]; then
            ( set -x
            2>/dev/null git -C {pyston_benchmarks_repo} remote add {req.pyston_benchmarks.remote} {req.pyston_benchmarks.url}
            )
        fi
        ( set -x
        git -C {pyston_benchmarks_repo} fetch --tags {req.pyston_benchmarks.remote}
        git -C {pyston_benchmarks_repo} checkout {req.pyston_benchmarks.fullref}
        )

        #####################
        # Run the benchmarks.

        ( set -x
        mkdir -p {bfiles.work.scratch_dir}
        )

        echo "running the benchmarks..."
        echo "(logging to {bfiles.work.logfile})"
        exitcode='{exitcode}'
        if [ -n "$exitcode" ]; then
            ( set -x
            sleep {fakedelay}
            touch {bfiles.work.logfile}
            touch {bfiles.request}/pyperformance-dummy-results.json.gz
            )
        else
            ( set -x
            MAKEFLAGS='-j{numjobs}' \\
                {python} {pyperformance_repo}/dev.py compile \\
                {bfiles.request.pyperformance_config} \\
                {ref} {maybe_branch} \\
                2>&1 | tee {bfiles.work.logfile}
            )
            exitcode=$?
        fi

        #####################
        # Record the results.

        if [ -e {bfiles.work.logfile} ]; then
            ln -s {bfiles.work.logfile} {bfiles.result.pyperformance_log}
        fi

        results=$(2>/dev/null ls {bfiles.work.pyperformance_results_glob})
        results_name=$(2>/dev/null basename $results)

        echo "saving results..."
        if [ $exitcode -eq 0 -a -n "$results" ]; then
            ( set -x
            jq '.status = "success"' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}

            jq --arg results "$results" '.pyperformance_data_orig = $results' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}

            jq --arg date $(date -u -Iseconds) '.history += [["success", $date]]' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}
            )
        else
            ( set -x
            jq '.status = "failed"' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}

            jq --arg date $(date -u -Iseconds) '.history += [["failed", $date]]' {bfiles.result.metadata} > {bfiles.result.metadata}.tmp
            mv {bfiles.result.metadata}.tmp {bfiles.result.metadata}
            )
        fi

        if [ -n "$results" -a -e "$results" ]; then
            ( set -x
            ln -s $results {bfiles.result.pyperformance_results}
            )
        fi

        echo "...done!"

        #rm -f {bfiles.work.pidfile}
    '''[1:-1])


class CompileBenchKind(JobKind):

    NAME = 'compile-bench'

    TYPICAL_DURATION_SECS = 40 * 60

    REQFS_FIELDS = JobKind.REQFS_FIELDS + [
        'pyperformance_manifest',
        'pyperformance_config',
    ]
    RESFS_FIELDS = JobKind.RESFS_FIELDS + [
        'pyperformance_log',
        'pyperformance_results',
    ]

    Request = CompileBenchRequest
    Result = CompileBenchResult

    def set_request_fs(self, fs, context):
        fs.pyperformance_manifest = f'{fs}/benchmarks.manifest'
        #fs.pyperformance_config = f'{fs}/compile.ini'
        fs.pyperformance_config = f'{fs}/pyperformance.ini'

    def set_work_fs(self, fs, context):
        if context == 'bench':
            # other directories needed by the job
            fs.venv = f'{fs}/pyperformance-venv'
            fs.scratch_dir = f'{fs}/pyperformance-scratch'
            # the results
            # XXX Is this right?
            fs.pyperformance_results_glob = f'{fs}/*.json.gz'

    def set_result_fs(self, fs, context):
        #fs.pyperformance_log = f'{fs}/run.log'
        fs.pyperformance_log = f'{fs}/pyperformance.log'
        #fs.pyperformance_results = f'{fs}/results-data.json.gz'
        fs.pyperformance_results = f'{fs}/pyperformance-results.json.gz'

    def create(self, reqid, jobfs, workerfs, *, _fake=None, **req_kwargs):
        req = resolve_compile_bench_request(
            reqid,
            jobfs.work.root,
            **req_kwargs,
        )

        # Write the benchmarks manifest.
        manifest = build_pyperformance_manifest(req, workerfs)
        with open(jobfs.request.pyperformance_manifest, 'w') as outfile:
            outfile.write(manifest)

        # Write the config.
        ini = build_pyperformance_config(req, workerfs)
        with open(jobfs.request.pyperformance_config, 'w') as outfile:
            ini.write(outfile)

        # Build the script for the commands to execute remotely.
        script = build_compile_script(req, workerfs, _fake)
        return req, script

    def as_row(self, req):
        ref = req.ref
        assert not isinstance(ref, str), repr(ref)
        return ref.full, ref, ref.remote, ref.branch, ref.tag, ref.commit
