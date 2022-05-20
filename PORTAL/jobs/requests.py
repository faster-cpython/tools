from collections import namedtuple
import configparser
import datetime
import json
import logging
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
            self._request = Request(self.reqid, self.reqdir)
            return self._request

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
        'remote',
        'branch',
        'benchmarks',
        'optimize',
        'debug',
    ]
    OPTIONAL = ['remote', 'branch', 'benchmarks', 'optimize', 'debug']

    CPYTHON = _utils.GitHubTarget.origin('python', 'cpython')
    PYPERFORMANCE = _utils.GitHubTarget.origin('python', 'pyperformance')
    PYSTON_BENCHMARKS = _utils.GitHubTarget.origin('pyston', 'python-macrobenchmarks')

    #pyperformance = PYPERFORMANCE.copy('034f58b')  # 1.0.4 release (2022-01-26)
    pyperformance = PYPERFORMANCE.copy('5b6142e')  # will be 1.0.5 release
    pyston_benchmarks = PYSTON_BENCHMARKS.copy('96e7bb3')  # main from 2022-01-21
    #pyperformance = PYPERFORMANCE.fork('ericsnowcurrently', 'python-performance', 'benchmark-management')
    #pyston_benchmarks = PYSTON_BENCHMARKS.fork('ericsnowcurrently', 'pyston-macrobenchmarks', 'pyperformance')

    def __init__(self,
                 id,
                 datadir,
                 ref,
                 remote=None,
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
        if not _utils.looks_like_git_ref(ref):
            raise ValueError(ref)

        super().__init__(id, datadir, **kwargs)
        self.ref = ref
        self.remote = remote
        self.branch = branch
        self.benchmarks = benchmarks
        self.optimize = True if optimize is None else optimize
        self.debug = debug

    @property
    def cpython(self):
        if self.remote:
            return self.CPYTHON.fork(self.remote, ref=self.ref)
        else:
            return self.CPYTHON.copy(ref=self.ref)

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

    def __init__(self, reqid, reqdir, *,
                 status=None,
                 pyperformance_results=None,
                 pyperformance_results_orig=None,
                 **kwargs
                 ):
        super().__init__(reqid, reqdir, status, **kwargs)
        self.pyperformance_results = pyperformance_results
        self.pyperformance_results_orig = pyperformance_results_orig

    def as_jsonable(self, *, withextra=False):
        data = super().as_jsonable(withextra=withextra)
        for field in ['pyperformance_results', 'pyperformance_results_orig']:
            if not data[field]:
                del data[field]
        data['reqid'] = str(data['reqid'])
        return data


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

    ref = _utils.resolve_git_revision_and_branch(revision, branch, remote)
    if not ref:
        raise Exception(f'could not find ref for {(remote, branch, revision)}')

    meta = BenchCompileRequest(
        id=reqid,
        datadir=workdir,
        # XXX Add a "commit" field and use "ref" for the actual ref.
        ref=ref.commit,
        #commit=ref.commit,
        #ref=ref.ref or ref.commit,
        remote=ref.remote,
        branch=ref.branch,
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
    _utils.check_shell_str(req.branch, required=False)
    maybe_branch = req.branch or ''
    _utils.check_shell_str(req.ref)

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
                git -C {cpython_repo} checkout -b {req.branch or '$branch'} --track {req.cpython.remote}/{req.branch or '$branch'}
            ); then
                echo "It already exists; resetting to the right target."
                ( set -x
                git -C {cpython_repo} checkout {req.branch or '$branch'}
                git -C {cpython_repo} reset --hard {req.cpython.remote}/{req.branch or '$branch'}
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
                {req.ref} {maybe_branch} \\
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
