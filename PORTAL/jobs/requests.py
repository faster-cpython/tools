from collections import namedtuple
import configparser
import datetime
import gzip
import hashlib
import json
import logging
import os.path
import platform
import re
import textwrap
import types

from . import _utils


FAKE_DELAY = 3

logger = logging.getLogger(__name__)


class RequestID(namedtuple('RequestID', 'kind timestamp user')):

    KIND = types.SimpleNamespace(
        BENCHMARKS='compile-bench',
    )
    _KIND_BY_VALUE = {v: v for _, v in vars(KIND).items()}

    @classmethod
    def from_raw(cls, raw):
        if isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        elif not raw:
            raise NotImplementedError(raw)
        else:
            try:
                args = tuple(raw)
            except TypeError:
                raise NotImplementedError(repr(raw))
            return cls(*args)

    @classmethod
    def parse(cls, idstr):
        kinds = '|'.join(cls._KIND_BY_VALUE)
        m = re.match(rf'^req-(?:({kinds})-)?(\d{{10}})-(\w+)$', idstr)
        if not m:
            return None
        kind, timestamp, user = m.groups()
        return cls(kind, int(timestamp), user)

    @classmethod
    def generate(cls, cfg, user=None, kind=KIND.BENCHMARKS):
        user = _utils.resolve_user(cfg, user)
        timestamp = int(_utils.utcnow())
        return cls(kind, timestamp, user)

    def __new__(cls, kind, timestamp, user):
        if not kind:
            kind = cls.KIND.BENCHMARKS
        else:
            try:
                kind = cls._KIND_BY_VALUE[kind]
            except KeyError:
                raise ValueError(f'unsupported kind {kind!r}')

        if not timestamp:
            raise ValueError('missing timestamp')
        elif isinstance(timestamp, str):
            timestamp, _, _ = timestamp.partition('.')
            timestamp = int(timestamp)
        elif not isinstance(timestamp, int):
            try:
                timestamp = int(timestamp)
            except TypeError:
                raise TypeError(f'expected int timestamp, got {timestamp!r}')

        if not user:
            raise ValueError('missing user')
        elif not isinstance(user, str):
            raise TypeError(f'expected str for user, got {user!r}')
        else:
            _utils.check_name(user)

        self = super().__new__(
            cls,
            kind=kind,
            timestamp=timestamp,
            user=user,
        )
        return self

    def __str__(self):
        return f'req-{self.kind}-{self.timestamp}-{self.user}'

    @property
    def date(self):
        dt, _ = _utils.get_utc_datetime(self.timestamp)
        return dt


class Request(_utils.Metadata):

    FIELDS = [
        'kind',
        'id',
        'datadir',
        'user',
        'date',
    ]

    @classmethod
    def read_kind(cls, metafile):
        text = _utils.read_file(metafile, fail=False)
        if not text:
            return None
        data = json.loads(text)
        if not data:
            return None
        kind = data.get('kind')
        if not kind:
            return None
        try:
            return RequestID._KIND_BY_VALUE[kind]
        except KeyError:
            raise ValueError(f'unsupported kind {kind!r}')

    @classmethod
    def load(cls, reqfile, *, fs=None, **kwargs):
        self = super().load(reqfile, **kwargs)
        if fs:
            self._fs = fs
        return self

    def __init__(self, id, datadir, *,
                 # These are ignored (duplicated by id):
                 kind=None, user=None, date=None,
                 ):
        if not id:
            raise ValueError('missing id')
        id = RequestID.from_raw(id)

        if not datadir:
            raise ValueError('missing datadir')
        if not isinstance(datadir, str):
            raise TypeError(f'expected dirname for datadir, got {datadir!r}')

        super().__init__(
            id=id,
            datadir=datadir,
        )

    def __str__(self):
        return str(self.id)

    @property
    def reqid(self):
        return self.id

    @property
    def reqdir(self):
        return self.datadir

    @property
    def kind(self):
        return self.id.kind

    @property
    def user(self):
        return self.id.user

    @property
    def date(self):
        return self.id.date

    @property
    def fs(self):
        try:
            return self._fs
        except AttributeError:
            raise NotImplementedError

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
        data['id'] = str(data['id'])
        data['date'] = self.date.isoformat()
        return data


class Result(_utils.Metadata):

    FIELDS = [
        'reqid',
        'reqdir',
        'status',
        'history',
    ]

    STATUS = types.SimpleNamespace(
        CREATED='created',
        PENDING='pending',
        ACTIVE='active',
        RUNNING='running',
        SUCCESS='success',
        FAILED='failed',
        CANCELED='canceled',
    )
    _STATUS_BY_VALUE = {v: v for _, v in vars(STATUS).items()}
    _STATUS_BY_VALUE['cancelled'] = STATUS.CANCELED
    FINISHED = frozenset([
        STATUS.SUCCESS,
        STATUS.FAILED,
        STATUS.CANCELED,
    ])
    CLOSED = 'closed'

    @classmethod
    def resolve_status(cls, status):
        try:
            return cls._STATUS_BY_VALUE[status]
        except KeyError:
            raise ValueError(f'unsupported status {status!r}')

    @classmethod
    def read_status(cls, metafile, *, fail=True):
        missing = None
        text = _utils.read_file(metafile, fail=fail)
        if text is None:
            return None
        elif not text:
            missing = True
        elif text:
            try:
                data = json.loads(text)
            except json.decoder.JSONDecodeError:
                missing = False
            else:
                if 'status' not in data:
                    missing = True
                else:
                    status = data['status']
                    try:
                        return cls._STATUS_BY_VALUE[status]
                    except KeyError:
                        missing = False
        if not fail:
            return None
        elif missing:
            raise _utils.MissingMetadataError(source=metafile)
        else:
            raise _utils.InvalidMetadataError(source=metafile)

    @classmethod
    def load(cls, resfile, *, fs=None, request=None, **kwargs):
        self = super().load(resfile, **kwargs)
        if fs:
            self._fs = fs
        if callable(request):
            self._get_request = request
        elif request:
            self._request = request
        return self

    def __init__(self, reqid, reqdir, status=STATUS.CREATED, history=None):
        if not reqid:
            raise ValueError('missing reqid')
        reqid = RequestID.from_raw(reqid)

        if not reqdir:
            raise ValueError('missing reqdir')
        if not isinstance(reqdir, str):
            raise TypeError(f'expected dirname for reqdir, got {reqdir!r}')

        if status == self.STATUS.CREATED:
            status = None
        elif status:
            status = self.resolve_status(status)
        else:
            status = None

        if history:
            h = []
            for st, date in history:
                try:
                    st = self._STATUS_BY_VALUE[st]
                except KeyError:
                    if st == self.CLOSED:
                        st = self.CLOSED
                    else:
                        raise ValueError(f'unsupported history status {st!r}')
                if not date:
                    date = None
                elif isinstance(date, str):
                    date, _ = _utils.get_utc_datetime(date)
                elif isinstance(date, int):
                    date, _ = _utils.get_utc_datetime(date)
                elif not isinstance(date, datetime.datetime):
                    raise TypeError(f'unsupported history date {date!r}')
                h.append((st, date))
            history = h
        else:
            history = [
                (self.STATUS.CREATED, reqid.date),
            ]
            if status is not None:
                for st in self._STATUS_BY_VALUE:
                    history.append((st, None))
                    if status == st:
                        break
                    if status == self.STATUS.RUNNING:
                        history.append((status, None))

        super().__init__(
            reqid=reqid,
            reqdir=reqdir,
            status=status,
            history=history,
        )

    def __str__(self):
        return str(self.reqid)

    @property
    def short(self):
        if not self.status:
            return f'<{self.reqid}: (created)>'
        return f'<{self.reqid}: {self.status}>'

    @property
    def request(self):
        try:
            return self._request
        except AttributeError:
            get_request = getattr(self, '_get_request', Request)
            self._request = get_request(self.reqid, self.reqdir)
            return self._request

    @property
    def fs(self):
        try:
            return self._fs
        except AttributeError:
            raise NotImplementedError

    @property
    def host(self):
        # XXX This will need to support other hosts.
        return 'fc_linux'

    @property
    def started(self):
        history = list(self.history)
        if not history:
            return None
        last_st, last_date = history[-1]
        if last_st == Result.STATUS.ACTIVE:
            return last_date, last_st
        for st, date in reversed(history):
            if st == Result.STATUS.RUNNING:
                return date, st
        else:
            return None, None

    @property
    def finished(self):
        history = list(self.history)
        if not history:
            return None
        for st, date in reversed(history):
            if st in Result.FINISHED:
                return date, st
        else:
            return None, None

    def set_status(self, status):
        if not status:
            raise ValueError('missing status')
        status = self.resolve_status(status)
        if self.history[-1][0] is self.CLOSED:
            raise Exception(f'req {self.reqid} is already closed')
        # XXX Make sure it is the next possible status?
        self.history.append(
            (status, datetime.datetime.now(datetime.timezone.utc)),
        )
        self.status = None if status is self.STATUS.CREATED else status
        if status in self.FINISHED:
            self.close()

    def close(self):
        if self.history[-1][0] is self.CLOSED:
            # XXX Fail?
            return
        self.history.append(
            (self.CLOSED, datetime.datetime.now(datetime.timezone.utc)),
        )

    def upload(self, req):
        raise NotImplementedError

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
        if self.status is None:
            data['status'] = self.STATUS.CREATED
        data['reqid'] = str(data['reqid'])
        data['history'] = [(st, d.isoformat() if d else None)
                           for st, d in data['history']]
        return data


##################################
# "compile"

class BenchCompileRequest(Request):

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
            elif _utils.Version.parse(self.branch):
                tag = self.ref.tag
                if tag:
                    ver = _utils.Version.parse(tag)
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
    def result(self):
        return BenchCompileResult(self.id, self.reqdir)


class BenchCompileResult(Result):

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
            self._pyperf = PyperfResults.from_file(filename,
                                                   host=self.host,
                                                   source=self.request.release,
                                                   )
            return self._pyperf


def resolve_bench_compile_request(reqid, workdir, remote, revision, branch,
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

    meta = BenchCompileRequest(
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
    cfg['run_benchmark']['manifest'] = bfiles.request.manifest
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
        if [ "$status" != 'active' ]; then
            2>&1 echo "ERROR: expected active status, got $status"
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


##################################
# pyperformance helpers

class Benchmarks:

    REPOS = os.path.join(_utils.HOME, 'repos')
    SUITES = {
        'pyperformance': {
            'url': 'https://github.com/python/pyperformance',
            'reldir': 'pyperformance/data-files/benchmarks',
        },
        'pyston': {
            'url': 'https://github.com/pyston/python-macrobenchmarks',
            'reldir': 'benchmarks',
        },
    }

    def __init__(self):
        self._cache = {}

    def load(self):
        """Return the per-suite lists of benchmarks."""
        benchmarks = {}
        for suite, info in self.SUITES.items():
            if suite in self._cache:
                benchmarks[suite] = list(self._cache[suite])
                continue
            url = info['url']
            reldir = info['reldir']
            reporoot = os.path.join(self.REPOS,
                                    os.path.basename(url))
            if not os.path.exists(reporoot):
                if not os.path.exists(self.REPOS):
                    os.makedirs(self.REPOS)
                _utils.git('clone', url, reporoot, cwd=None)
            names = self._get_names(os.path.join(reporoot, reldir))
            benchmarks[suite] = self._cache[suite] = names
        return benchmarks

    def _get_names(self, benchmarksdir):
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


class PyperfResults:

    @classmethod
    def from_file(cls, filename, host=None, source=None):
        if not filename:
            raise ValueError('missing filename')
        data = cls._read(filename)
        self = cls(data, filename, host, source)
        return self

    @classmethod
    def _read(cls, filename):
        if filename.endswith('.json.gz'):
            _open = gzip.open
        else:
            raise NotImplementedError(filename)
        with _open(filename) as infile:
            return json.load(infile)

    @classmethod
    def _get_os_name(cls, metadata=None):
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

    @classmethod
    def _get_arch(cls, metadata=None):
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
            uname = _platform.uname()
            machine = uname.machine.lower()
            if machine in ('amd64', 'x86_64'):
                return machine
            elif machine == 'aarch64':
                return 'arm64'
            elif 'arm' in machine:
                return 'arm'
            else:
                raise NotImplementedError(machine)

    def __init__(self, data, filename=None, host=None, source=None, suite=None):
        if data:
            self._set_data(data)
        elif not filename:
            raise ValueError('missing filename')
        self.filename = filename or None
        if host:
            self._host = host
        self.source = source
        self.suite = suite or None

    def _set_data(self, data):
        if hasattr(self, '_data'):
            raise Exception('already set')
        # XXX Validate?
        if data['version'] == '1.0':
            self._data = data
        else:
            raise NotImplementedError(data['version'])

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            data = self._read(self.filename)
            self._set_data(data)
            return self._data

    @property
    def metadata(self):
        return self.data['metadata']

    @property
    def host(self):
        try:
            return self._host
        except AttributeError:
            metadata = self.metadata
            # We could use metadata['hostname'] but that doesn't
            # make a great label in the default case.
            host = self._get_os_name(metadata)
            arch = self._get_arch(metadata)
            if arch in ('arm32', 'arm64'):
                host += '-arm'
            # Ignore everything else.
            return host
            return self._host

    @property
    def implementation(self):
        return 'cpython'

    @property
    def compat_id(self):
        return self._get_compat_id()

    def _get_compat_id(self, *, short=True):
        metadata = self.metadata
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

    def get_upload_name(self):
        # See https://github.com/faster-cpython/ideas/tree/main/benchmark-results/README.md
        # for details on this filename format.
        metadata = self.metadata
        commit = metadata.get('commit_id')
        if not commit:
            raise NotImplementedError
        compat = self._get_compat_id()
        source = self.source or 'main'
        implname = self.implementation
        host = self.host
        name = f'{implname}-{source}-{commit[:10]}-{host}-{compat}'
        if self.suite and self.suite != 'pyperformance':
            name = f'{name}-{self.suite}'
        return name

    def split_benchmarks(self):
        """Return results collated by suite."""
        if self.suite:
            raise Exception(f'already split ({self.suite})')
        by_suite = {}
        benchmarks = Benchmarks().load()
        by_name = {}
        for suite, names in benchmarks.items():
            for name in names:
                if name in by_name:
                    raise NotImplementedError((suite, name))
                by_name[name] = suite
        results = self.data
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
            if suite not in by_suite:
                by_suite[suite] = {k: v
                                   for k, v in results.items()
                                   if k != 'benchmarks'}
                by_suite[suite]['benchmarks'] = []
            by_suite[suite]['benchmarks'].append(data)
        cls = type(self)
        for suite, data in by_suite.items():
            host = getattr(self, '_host', None)
            results = cls(None, self.filename, host, self.source, suite)
            results._data = data
            by_suite[suite] = results
        return by_suite


class PyperfResultsStorage:

    def add(self, results, *, unzipped=True):
        raise NotImplementedError


class PyperfResultsRepo(PyperfResultsStorage):

    BRANCH = 'add-benchmark-results'

    def __init__(self, root, remote=None, datadir=None):
        if root:
            root = os.path.abspath(root)
        if remote:
            if isinstance(remote, str):
                remote = _utils.GitHubTarget.resolve(remote, root)
            elif not isinstance(remote, _utils.GitHubTarget):
                raise TypeError(f'unsupported remote {remote!r}')
            root = remote.ensure_local(root)
        else:
            if root:
                if not os.path.exists(root):
                    raise FileNotFoundError(root)
                #_utils.verify_git_repo(root)
            else:
                raise ValueError('missing root')
            remote = None
        self.root = root
        self.remote = remote
        self.datadir = datadir or None

    def git(self, *args, cfg=None):
        ec, text = _utils.git(*args, cwd=self.root, cfg=cfg)
        if ec:
            raise NotImplementedError((ec, text))
        return text

    def add(self, results, *,
            branch=None,
            author=None,
            unzipped=True,
            split=True,
            push=True,
            ):
        if not branch:
            branch = self.BRANCH

        if not isinstance(results, PyperfResults):
            raise NotImplementedError(results)
        source = results.filename
        if not os.path.exists(source):
            logger.error(f'results not found at {source}')
            return
        if split:
            by_suite = results.split_benchmarks()
            if 'other' in by_suite:
                raise NotImplementedError(sorted(by_suite))
        else:
            by_suite = {None: results}

        if self.remote:
            self.remote.ensure_local(self.root)

        authorargs = ()
        cfg = {}
        if not author:
            pass
        elif isinstance(author, str):
            parsed = _utils.parse_email_address(author)
            if not parsed:
                raise ValueError(f'invalid author {author!r}')
            name, email = parsed
            if not name:
                name = '???'
                author = f'??? <{author}>'
            cfg['user.name'] = name
            cfg['user.email'] = email
            #authorargs = ('--author', author)
        else:
            raise NotImplementedError(author)

        logger.info(f'adding results {source}...')
        for suite in by_suite:
            suite_results = by_suite[suite]
            name = suite_results.get_upload_name()
            reltarget = f'{name}.json'
            if self.datadir:
                reltarget = f'{self.datadir}/{reltarget}'
            if not unzipped:
                reltarget += '.gz'
            target = os.path.join(self.root, reltarget)

            logger.info(f'...as {target}...')
            self.git('checkout', '-B', branch)
            if unzipped:
                data = suite_results.data
                print(target)
                with open(target, 'w') as outfile:
                    json.dump(data, outfile, indent=2)
            else:
                shutil.copyfile(source, target)
            self.git('add', reltarget)
            msg = f'Add Benchmark Results ({name})'
            self.git('commit', *authorargs, '-m', msg, cfg=cfg)
        logger.info('...done adding')

        if push:
            self._upload(reltarget)

    def _upload(self, reltarget):
        if not self.remote:
            raise Exception('missing remote')
        url = f'{self.remote.url}/tree/main/{reltarget}'
        logger.info(f'uploading results to {url}...')
        self.git('push', self.remote.push_url)
        logger.info('...done uploading')


class FasterCPythonResults(PyperfResultsRepo):

    REMOTE = _utils.GitHubTarget.from_origin('faster-cpython', 'ideas', ssh=True)
    DATADIR = 'benchmark-results'

    def __init__(self, root=None, remote=None):
        if not remote:
            remote = self.REMOTE
        super().__init__(root, remote, self.DATADIR)
