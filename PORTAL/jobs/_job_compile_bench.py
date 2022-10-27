import configparser
import logging
import os.path
import textwrap
from typing import Any, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

from . import _utils, _pyperformance, _common
from .requests import Request, RequestID, Result, ToRequestIDType
from . import requests

if TYPE_CHECKING:
    from . import _job, _workers


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
    pyperformance = PYPERFORMANCE.copy('4622a0b')  # will be 1.0.6 release
    pyston_benchmarks = PYSTON_BENCHMARKS.copy('797dfd')  # main from 2022-08-24

    @classmethod
    def _extract_kwargs(
            cls,
            data: Any,
            optional: List[str],
            filename
    ) -> Tuple[dict, Optional[dict]]:
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

    def __init__(
            self,
            id: ToRequestIDType,
            datadir: str,
            ref: _utils.ToGitRefType,
            pyperformance_ref: Optional[_utils.ToGitRefType] = None,
            remote: Optional[str] = None,
            revision: Optional[str] = None,
            branch: Optional[str] = None,
            benchmarks: Optional[List[str]] = None,
            optimize: bool = True,
            debug: bool = False,
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
                    gitref = _utils.GitRef.from_values(remote, branch, tag, commit, ref)
                    if gitref is None:
                        raise ValueError
                except ValueError:
                    # backward compatibility
                    GR = _utils.GitRef
                    gitref = GR.__new__(GR, remote, branch, tag, commit, ref, None)
            else:
                refstr = ref
                gitref = _utils.GitRef.resolve(revision, branch, remote)
                if gitref is None:
                    raise ValueError("Could not resolve ref {ref}")
                if refstr not in (gitref.commit, gitref.branch, gitref.tag, None):
                    raise ValueError(f'unexpected ref {refstr!r}')
        else:
            gitref = _utils.GitRef.from_raw(ref)

        self.ref = gitref
        self.pyperformance_ref = pyperformance_ref or str(ref)
        self.remote = gitref.remote
        self.revision = revision
        self.branch = gitref.branch
        self.benchmarks = benchmarks
        self.optimize = True if optimize is None else optimize
        self.debug = debug
        self._impl = _utils.CPython()

    @property
    def cpython(self) -> _utils.GitHubTarget:
        # XXX Pass self.ref directly?
        ref = str(self.ref)
        if self.remote and self.remote != 'origin':
            return self.CPYTHON.fork(self.remote, ref=ref)
        else:
            return self.CPYTHON.copy(ref=ref)

    @property
    def release(self) -> str:
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
    def versionstr(self) -> str:
        return _utils.CPythonVersion.render_extended(
            version=self.release,
            bits=64,
            commit=self.ref.commit[:10],
        )

    @property
    def result(self) -> "CompileBenchResult":
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

    _pyperf: _pyperformance.PyperfResults

    @classmethod
    def _extract_kwargs(
            cls,
            data: Any,
            optional: List[str],
            filename: str
    ) -> Tuple[dict, Optional[dict]]:
        # This is a backward-compatibility shim.
        try:
            return super()._extract_kwargs(data, optional, filename)
        except ValueError:
            optional = [*optional, 'reqdir', 'history']
            kwargs, extra = super()._extract_kwargs(data, optional, filename)
            kwargs.setdefault('reqdir', os.path.dirname(filename))
            kwargs.setdefault('history', None)
            return kwargs, extra

    def __init__(
            self,
            reqid: ToRequestIDType,
            reqdir: str,
            *,
            status: Optional[str] = None,
            #pyperformance_results=None,
            #pyperformance_results_orig=None,
            **kwargs
    ):
        super().__init__(reqid, reqdir, status, **kwargs)
        #self.pyperformance_results = pyperformance_results
        #self.pyperformance_results_orig = pyperformance_results_orig

    @property
    def pyperf(self) -> _pyperformance.PyperfResults:
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


def resolve_compile_bench_request(
        reqid: ToRequestIDType,
        workdir: str,
        remote: str,
        revision: str,
        branch: str,
        benchmarks: Optional[Union[str, Iterable[str]]],
        *,
        optimize: bool,
        debug: bool,
) -> CompileBenchRequest:
    if isinstance(benchmarks, str):
        benchmarks_seq = benchmarks.replace(',', ' ').split()
    elif benchmarks is None:
        benchmarks_seq = []
    else:
        benchmarks_seq = list(benchmarks)
    benchmarks_list = [b.strip() for b in benchmarks_seq if b]

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
        benchmarks=benchmarks_list or None,
        optimize=bool(optimize),
        debug=bool(debug),
    )
    return meta


def build_pyperformance_manifest(
        req: CompileBenchRequest,
        bfiles: "_workers.WorkerJobsFS",
) -> str:
    return textwrap.dedent(f'''
        [includes]
        <default>
        {bfiles.repos.pyston_benchmarks}/benchmarks/MANIFEST
    '''[1:-1])


def build_pyperformance_config(
        req: CompileBenchRequest,
        bfiles: "_workers.WorkerJobsFS"
) -> configparser.ConfigParser:
    cpython = bfiles.repos.cpython
    bfiles_jobfs = bfiles.resolve_request(req.id)
    cfg = configparser.ConfigParser()

    cfg['config'] = {}
    cfg['config']['json_dir'] = bfiles_jobfs.result.root
    cfg['config']['debug'] = str(req.debug)
    # XXX pyperformance should be looking in [scm] for this.
    cfg['config']['git_remote'] = req.remote

    cfg['scm'] = {}
    cfg['scm']['repo_dir'] = cpython
    cfg['scm']['git_remote'] = req.remote
    cfg['scm']['update'] = 'True'

    cfg['compile'] = {}
    cfg['compile']['bench_dir'] = bfiles_jobfs.work.scratch_dir
    cfg['compile']['pgo'] = str(req.optimize)
    cfg['compile']['lto'] = str(req.optimize)
    cfg['compile']['install'] = 'True'

    cfg['run_benchmark'] = {}
    cfg['run_benchmark']['manifest'] = bfiles_jobfs.request.pyperformance_manifest
    cfg['run_benchmark']['benchmarks'] = ','.join(req.benchmarks or ())
    cfg['run_benchmark']['system_tune'] = 'True'
    cfg['run_benchmark']['upload'] = 'False'

    return cfg


def build_compile_script(
        req: CompileBenchRequest,
        bfiles: "_workers.WorkerJobsFS",
        fake: Optional[Any] = None
) -> str:
    exitcode: Optional[Union[str, int]]
    fakedelay: Optional[int] = FAKE_DELAY
    if fake is False or fake is None:
        exitcode, fakedelay = (None, None)
    elif fake is True:
        exitcode, fakedelay = (0, None)
    elif isinstance(fake, (int, str)):
        exitcode, fakedelay = (fake, None)
    else:
        exitcode, fakedelay = (None, None)
    if fakedelay is None:
        fakedelay = FAKE_DELAY
    else:
        fakedelay = _utils.ensure_int(fakedelay, min=0)
        if exitcode is None:
            exitcode = 0
        elif exitcode == '':
            logger.warning(f'fakedelay ({fakedelay}) will not be used')
    if exitcode is None:
        exitcode = ''
    elif exitcode != '':
        exitcode = _utils.ensure_int(exitcode, min=0)
        logger.warning('we will pretend pyperformance will run with exitcode %s', exitcode)
    python = 'python3.9'  # On the job worker.
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

    cpython_repo = bfiles.repos.cpython
    pyperformance_repo = bfiles.repos.pyperformance
    pyston_benchmarks_repo = bfiles.repos.pyston_benchmarks
    _utils.check_shell_str(cpython_repo)
    _utils.check_shell_str(pyperformance_repo)
    _utils.check_shell_str(pyston_benchmarks_repo)

    _bfiles = bfiles.resolve_request(req.id)
    _utils.check_shell_str(_bfiles.work.pidfile)
    _utils.check_shell_str(_bfiles.work.logfile)
    _utils.check_shell_str(_bfiles.work.scratch_dir)
    _utils.check_shell_str(_bfiles.request.pyperformance_config)
    _utils.check_shell_str(_bfiles.result.pyperformance_log)
    _utils.check_shell_str(_bfiles.result.metadata)
    _utils.check_shell_str(_bfiles.result.pyperformance_results)
    _utils.check_shell_str(_bfiles.work.pyperformance_results_glob)

    _utils.check_shell_str(python)

    # XXX Kill any zombie job processes?

    return textwrap.dedent(f'''
        #!/usr/bin/env bash

        # This script runs only on the job worker.

        # The commands in this script are deliberately explicit
        # so you can copy-and-paste them selectively.

        #####################
        # Mark the result as running.

        echo "$$" > {_bfiles.work.pidfile}

        status=$(jq -r '.status' {_bfiles.result.metadata})
        if [ "$status" != 'activated' ]; then
            2>&1 echo "ERROR: expected activated status, got $status"
            2>&1 echo "       (see {_bfiles.result.metadata})"
            exit 1
        fi

        ( set -x
        jq --arg date $(date -u -Iseconds) '.history += [["running", $date]]' {_bfiles.result.metadata} > {_bfiles.result.metadata}.tmp
        mv {_bfiles.result.metadata}.tmp {_bfiles.result.metadata}
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
        mkdir -p {_bfiles.work.scratch_dir}
        )

        echo "running the benchmarks..."
        echo "(logging to {_bfiles.work.logfile})"
        exitcode='{exitcode}'
        if [ -n "$exitcode" ]; then
            ( set -x
            sleep {fakedelay}
            touch {_bfiles.work.logfile}
            touch {_bfiles.request}/pyperformance-dummy-results.json.gz
            )
        else
            # Remove pyperformance's .venvs directory to force an upgrade of its
            # dependencies
            rm -rf {pyperformance_repo}/.venvs
            ( set -x
            MAKEFLAGS='-j{numjobs}' \\
                {python} {pyperformance_repo}/dev.py compile \\
                {_bfiles.request.pyperformance_config} \\
                {ref} {maybe_branch} \\
                2>&1 | tee {_bfiles.work.logfile}
            )
            exitcode=$?
        fi

        #####################
        # Record the results.

        if [ -e {_bfiles.work.logfile} ]; then
            ln -s {_bfiles.work.logfile} {_bfiles.result.pyperformance_log}
        fi

        results=$(2>/dev/null ls {_bfiles.work.pyperformance_results_glob})
        results_name=$(2>/dev/null basename $results)

        echo "saving results..."
        if [ $exitcode -eq 0 -a -n "$results" ]; then
            ( set -x
            jq '.status = "success"' {_bfiles.result.metadata} > {_bfiles.result.metadata}.tmp
            mv {_bfiles.result.metadata}.tmp {_bfiles.result.metadata}

            jq --arg results "$results" '.pyperformance_data_orig = $results' {_bfiles.result.metadata} > {_bfiles.result.metadata}.tmp
            mv {_bfiles.result.metadata}.tmp {_bfiles.result.metadata}

            jq --arg date $(date -u -Iseconds) '.history += [["success", $date]]' {_bfiles.result.metadata} > {_bfiles.result.metadata}.tmp
            mv {_bfiles.result.metadata}.tmp {_bfiles.result.metadata}
            )
        else
            ( set -x
            jq '.status = "failed"' {_bfiles.result.metadata} > {_bfiles.result.metadata}.tmp
            mv {_bfiles.result.metadata}.tmp {_bfiles.result.metadata}

            jq --arg date $(date -u -Iseconds) '.history += [["failed", $date]]' {_bfiles.result.metadata} > {_bfiles.result.metadata}.tmp
            mv {_bfiles.result.metadata}.tmp {_bfiles.result.metadata}
            )
        fi

        if [ -n "$results" -a -e "$results" ]; then
            ( set -x
            ln -s $results {_bfiles.result.pyperformance_results}
            )
        fi

        echo "...done!"

        #rm -f {_bfiles.work.pidfile}
    '''[1:-1])


class CompileBenchKind(_common.JobKind):

    NAME = 'compile-bench'

    TYPICAL_DURATION_SECS = 40 * 60

    REQFS_FIELDS = _common.JobKind.REQFS_FIELDS + [
        'pyperformance_manifest',
        'pyperformance_config',
    ]
    RESFS_FIELDS = _common.JobKind.RESFS_FIELDS + [
        'pyperformance_log',
        'pyperformance_results',
    ]

    Request = CompileBenchRequest
    Result = CompileBenchResult

    def set_request_fs(
            self,
            fs: _common.JobRequestFS,
            context: Optional[str]
    ) -> None:
        fs.pyperformance_manifest = f'{fs}/benchmarks.manifest'
        #fs.pyperformance_config = f'{fs}/compile.ini'
        fs.pyperformance_config = f'{fs}/pyperformance.ini'

    def set_work_fs(
            self,
            fs: _common.JobWorkFS,
            context: Optional[str]
    ) -> None:
        if context == 'job-worker':
            # other directories needed by the job
            fs.venv = f'{fs}/pyperformance-venv'
            fs.scratch_dir = f'{fs}/pyperformance-scratch'
            # the results
            # XXX Is this right?
            fs.pyperformance_results_glob = f'{fs}/*.json.gz'

    def set_result_fs(
            self,
            fs: _common.JobResultFS,
            context: Optional[str]
    ) -> None:
        #fs.pyperformance_log = f'{fs}/run.log'
        fs.pyperformance_log = f'{fs}/pyperformance.log'
        #fs.pyperformance_results = f'{fs}/results-data.json.gz'
        fs.pyperformance_results = f'{fs}/pyperformance-results.json.gz'

    def create(
            self,
            reqid: ToRequestIDType,
            jobfs: "_job.JobFS",
            workerfs: "_workers.WorkerJobsFS",
            *,
            _fake: Optional[Any] = None,
            **req_kwargs
    ) -> Tuple[CompileBenchRequest, str]:
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

    def as_row(self, req: requests.Request) -> Tuple[str, str, str, str, str, str]:
        ref = req.ref
        assert not isinstance(ref, str), repr(ref)
        return ref.full, ref, ref.remote, ref.branch, ref.tag, ref.commit
