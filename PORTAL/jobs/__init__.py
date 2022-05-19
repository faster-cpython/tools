import logging
import os
import os.path
import signal
import subprocess
import sys
import textwrap
import time
import types

from . import _utils, requests as _requests
from .requests import RequestID, Request, Result


PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PATH_ENTRY = os.path.dirname(PKG_ROOT)

logger = logging.getLogger(__name__)


##################################
# jobs config

class PortalConfig(_utils.Config):

    CONFIG = 'benchmarking-portal.json'
    ALT_CONFIG = 'portal.json'

    FIELDS = ['bench_user', 'send_user', 'send_host', 'send_port', 'data_dir']
    OPTIONAL = ['data_dir']

    def __init__(self,
                 bench_user,
                 send_user,
                 send_host,
                 send_port,
                 data_dir=None,
                 ):
        if not bench_user:
            raise ValueError('missing bench_user')
        if not send_user:
            send_user = bench_user
        if not send_host:
            raise ValueError('missing send_host')
        if not send_port:
            raise ValueError('missing send_port')
        if data_dir:
            data_dir = os.path.abspath(os.path.expanduser(data_dir))
        else:
            data_dir = f'/home/{send_user}/BENCH'  # This matches DATA_ROOT.
        super().__init__(
            bench_user=bench_user,
            send_user=send_user,
            send_host=send_host,
            send_port=send_port,
            data_dir=data_dir or None,
        )


#class BenchConfig(_utils.Config):
#
#    CONFIG = f'benchmarking-bench.json'
#    ALT_CONFIG = f'bench.json'
#
#    FIELDS = ['portal']
#
#    def __init__(self,
#                 portal,
#                 ):
#        super().__init__(
#            portal=portal,
#        )


##################################
# job files

class JobFS(types.SimpleNamespace):
    """The file structure of a job's data."""

    @classmethod
    def from_jobsfs(cls, jobsfs, reqid):
        self = cls(
            request=f'{jobsfs.requests}/{reqid}',
            result=f'{jobsfs.requests}/{reqid}',
            work=f'{jobsfs.work}/{reqid}',
            reqid=reqid,
            context=jobsfs.context,
        )
        return self

    def __init__(self, request, result, work, reqid=None, context='portal'):
        request = _utils.FSTree.from_raw(request, name='request')
        work = _utils.FSTree.from_raw(work, name='work')
        result = _utils.FSTree.from_raw(result, name='result')
        if not reqid:
            reqid = os.path.basename(request)
            reqid = RequestID.from_raw(reqid)
            if not reqid:
                raise ValueError('missing reqid')
        else:
            orig = reqid
            reqid = RequestID.from_raw(reqid)
            if not reqid:
                raise ValueError(f'unsupported reqid {orig!r}')

        if not context:
            context = 'portal'
        elif context not in ('portal', 'bench'):
            raise ValueError(f'unsupported context {context!r}')

        # the request
        request.metadata = f'{request}/request.json'
        # the job
        work.bench_script = f'{work}/run.sh'
        if context == 'portal':
            work.portal_script = f'{work}/send.sh'
            work.pidfile = f'{work}/send.pid'
            work.logfile = f'{work}/job.log'
            work.ssh_okay = f'{work}/ssh.ok'
        elif context == 'bench':
            work.pidfile = f'{work}/job.pid'
            work.logfile = f'{work}/job.log'
        # the results
        result.metadata = f'{result}/results.json'

        super().__init__(
            reqid=reqid,
            context=context,
            request=request,
            work=work,
            result=result,
        )

        # XXX Move these to a subclass?
        if reqid.kind == 'compile-bench':
            request.manifest = f'{request}/benchmarks.manifest'
            #request.pyperformance_config = f'{request}/compile.ini'
            request.pyperformance_config = f'{request}/pyperformance.ini'
            #result.pyperformance_log = f'{result}/run.log'
            result.pyperformance_log = f'{result}/pyperformance.log'
            #result.pyperformance_results = f'{result}/results-data.json.gz'
            result.pyperformance_results = f'{result}/pyperformance-results.json.gz'
            if self.context == 'bench':
                # other directories needed by the job
                work.venv = f'{work}/pyperformance-venv'
                work.scratch_dir = f'{work}/pyperformance-scratch'
                # the results
                # XXX Is this right?
                work.pyperformance_results_glob = f'{work}/*.json.gz'
        else:
            raise ValueError(f'unsupported job kind for {reqid}')

    def __str__(self):
        return str(self.request)

    def __fspath__(self):
        return str(self.request)

    @property
    def jobs(self):
        dirname, reqid = os.path.split(self.request)
        if str(self.reqid) != reqid:
            raise NotImplementedError
        root, requests = os.path.split(dirname)
        if requests != 'REQUESTS':
            raise NotImplementedError
        return JobsFS(root)

    @property
    def bench_script(self):
        return self.work.bench_script

    @property
    def portal_script(self):
        return self.work.portal_script

    @property
    def pidfile(self):
        return self.work.pidfile

    @property
    def logfile(self):
        return self.work.logfile

    @property
    def ssh_okay(self):
        return self.work.ssh_okay

    def copy(self):
        return type(self)(
            str(self.request),
            str(self.work),
            str(self.result),
            self.reqid,
            self.context,
        )


class JobsFS(_utils.FSTree):
    """The file structure of the jobs data."""

    JOBFS = JobFS

    @classmethod
    def from_user(cls, user, context='portal'):
        return cls(f'/home/{user}/BENCH', context)

    def __init__(self, root='~/BENCH', context='portal'):
        if not root:
            root = '~/BENCH'
        super().__init__(root)

        if not context:
            context = 'portal'
        elif context not in ('portal', 'bench'):
            raise ValueError(f'unsupported context {context!r}')
        self.context = context

        self.requests = _utils.FSTree(f'{root}/REQUESTS')
        if context == 'portal':
            self.requests.current = f'{self.requests}/CURRENT'

        self.work = _utils.FSTree(self.requests.root)
        self.results = _utils.FSTree(self.requests.root)

        if context == 'portal':
            self.queue = _utils.FSTree(f'{self.requests}/queue.json')
            self.queue.data = f'{self.requests}/queue.json'
            self.queue.lock = f'{self.requests}/queue.lock'
            self.queue.log = f'{self.requests}/queue.log'
        elif context == 'bench':
            # the local git repositories used by the job
            self.repos = _utils.FSTree(f'{self}/repositories')
            self.repos.cpython = f'{self.repos}/cpython'
            self.repos.pyperformance = f'{self.repos}/pyperformance'
            self.repos.pyston_benchmarks = f'{self.repos}/pyston-benchmarks'
        else:
            raise ValueError(f'unsupported context {context!r}')

    def __str__(self):
        return self.root

    def resolve_request(self, reqid):
        return self.JOBFS.from_jobsfs(self, reqid)

    def copy(self):
        return type(self)(self.root, self.context)


##################################
# workers

class Worker:

    @classmethod
    def from_config(cls, cfg, JobsFS=JobsFS):
        fs = JobsFS.from_user(cfg.bench_user, 'bench')
        ssh = _utils.SSHClient(cfg.send_host, cfg.send_port, cfg.bench_user)
        return cls(fs, ssh)

    def __init__(self, fs, ssh):
        self._fs = fs
        self._ssh = ssh

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'fs ssh'.split())
        return f'{type(self).__name__}({"".join(args)})'

    @property
    def fs(self):
        return self._fs

    @property
    def ssh(self):
        return self._ssh

    def resolve(self, reqid):
        fs = self._fs.resolve_request(reqid)
        return JobWorker(self, fs)


class JobWorker:

    def __init__(self, worker, fs):
        self._worker = worker
        self._fs = fs

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'worker fs'.split())
        return f'{type(self).__name__}({"".join(args)})'

    @property
    def worker(self):
        return self._worker

    @property
    def fs(self):
        return self._fs

    @property
    def topfs(self):
        return self._worker.fs

    @property
    def ssh(self):
        return self._worker.ssh


##################################
# jobs

class JobsError(RuntimeError):
    MSG = 'a jobs-related problem'

    def __init__(self, msg=None):
        super().__init__(msg or self.MSG)


class NoRunningJobError(JobsError):
    MSG = 'no job is currently running'


class JobError(JobsError):
    MSG = 'job {reqid} has a problem'

    def __init__(self, reqid, msg=None):
        msg = (msg or self.MSG).format(reqid=str(reqid))
        super().__init__(msg)
        self.reqid = reqid


class JobNotRunningError(JobError):
    MSG = 'job {reqid} is not running'


class JobNeverStartedError(JobNotRunningError):
    MSG = 'job {reqid} was never started'


class Job:

    def __init__(self, reqid, fs, worker, cfg):
        if not reqid:
            raise ValueError('missing reqid')
        if not fs:
            raise ValueError('missing fs')
        elif not isinstance(fs, JobFS):
            raise TypeError(f'expected JobFS for fs, got {fs!r}')
        if not worker:
            raise ValueError('missing worker')
        elif not isinstance(worker, JobWorker):
            raise TypeError(f'expected JobWorker for worker, got {worker!r}')
        self._reqid = RequestID.from_raw(reqid)
        self._fs = fs
        self._worker = worker
        self._cfg = cfg
        self._pidfile = _utils.PIDFile(fs.pidfile)

    def __repr__(self):
        args = (f'{n}={str(getattr(self, "_"+n))!r}'
                for n in 'reqid fs worker cfg'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __str__(self):
        return str(self._reqid)

    @property
    def reqid(self):
        return self._reqid

    @property
    def fs(self):
        return self._fs

    @property
    def worker(self):
        return self._worker

    @property
    def cfg(self):
        return self._cfg

    @property
    def kind(self):
        return self.reqid.kind

    @property
    def request(self):
        return Request(self._reqid, str(self.fs))

    def load_result(self):
        return Result.load(self._fs.result.metadata)

    def get_status(self, *, fail=True):
        try:
            return Result.read_status(self._fs.result.metadata)
        except FileNotFoundError:
            # XXX Create it?
            if fail:
                raise
            return None
        except _utils.MissingMetadataError:
            # XXX Re-create it?
            if fail:
                raise
            return None
        except _utils.InvalidMetadataError:
            # XXX Fix it?
            if fail:
                raise
            return None

    def set_status(self, status):
        status = Result.resolve_status(status)
        result = self.load_result()
        result.set_status(status)
        result.save(self._fs.result.metadata, withextra=True)

    def _create(self, kind_kwargs, reqfsattrs, queue_log):
        os.makedirs(self._fs.request.root, exist_ok=True)
        os.makedirs(self._fs.work.root, exist_ok=True)
        os.makedirs(self._fs.result.root, exist_ok=True)

        if self._reqid.kind == 'compile-bench':
            fake = kind_kwargs.pop('_fake', None)
            req = _requests.resolve_bench_compile_request(
                self._reqid,
                self._fs.work.root,
                **kind_kwargs,
            )

            # Write the benchmarks manifest.
            manifest = _requests.build_pyperformance_manifest(req, self._worker.topfs)
            with open(self._fs.request.manifest, 'w') as outfile:
                outfile.write(manifest)

            # Write the config.
            ini = _requests.build_pyperformance_config(req, self._worker.topfs)
            with open(self._fs.request.pyperformance_config, 'w') as outfile:
                ini.write(outfile)

            # Build the script for the commands to execute remotely.
            script = _requests.build_compile_script(req, self._worker.topfs, fake)
        else:
            raise ValueError(f'unsupported job kind in {self._reqid}')

        # Write metadata.
        req.save(self._fs.request.metadata)
        req.result.save(self._fs.result.metadata)

        # Write the commands to execute remotely.
        with open(self._fs.bench_script, 'w') as outfile:
            outfile.write(script)
        os.chmod(self._fs.bench_script, 0o755)

        # Write the commands to execute locally.
        script = self._build_send_script(queue_log, reqfsattrs)
        with open(self._fs.portal_script, 'w') as outfile:
            outfile.write(script)
        os.chmod(self._fs.portal_script, 0o755)

    def _build_send_script(self, queue_log, resfsfields=None, *,
                           hidecfg=False,
                           ):
        reqid = self.reqid
        pfiles = self._fs
        bfiles = self._worker.fs

        if not self._cfg.filename:
            raise NotImplementedError(self._cfg)
        cfgfile = _utils.quote_shell_str(self._cfg.filename)
        if hidecfg:
            ssh = _utils.SSHShellCommands('$host', '$port', '$benchuser')
            user = '$user'
        else:
            user = _utils.check_shell_str(self._cfg.send_user)
            _utils.check_shell_str(self._cfg.send_host)
            _utils.check_shell_str(self._cfg.bench_user)
            ssh = self._worker.ssh.shell_commands

        queue_log = _utils.quote_shell_str(queue_log)

        reqdir = _utils.quote_shell_str(pfiles.request.root)
        results_meta = _utils.quote_shell_str(pfiles.result.metadata)
        pidfile = _utils.quote_shell_str(pfiles.pidfile)

        _utils.check_shell_str(bfiles.request.root)
        _utils.check_shell_str(bfiles.bench_script)
        _utils.check_shell_str(bfiles.result.root)
        _utils.check_shell_str(bfiles.result.metadata)

        ensure_user = ssh.ensure_user_with_agent(user)
        ensure_user = '\n                '.join(ensure_user)

        resfiles = []
        for attr in resfsfields or ():
            pvalue = getattr(pfiles.result, attr)
            pvalue = _utils.quote_shell_str(pvalue)
            bvalue = getattr(bfiles.result, attr)
            _utils.check_shell_str(bvalue)
            resfiles.append((bvalue, pvalue))
        resfiles = (ssh.pull(s, t) for s, t in resfiles)
        resfiles = '\n                '.join(resfiles)

        push = ssh.push
        pull = ssh.pull
        ssh = ssh.run_shell

        return textwrap.dedent(f'''
            #!/usr/bin/env bash

            # This script only runs on the portal host.
            # It does 4 things:
            #   1. switch to the {user} user, if necessary
            #   2. prepare the bench host, including sending all
            #      the request files to the bench host (over SSH)
            #   3. run the job (e.g. run the benchmarks)
            #   4. pull the results-related files from the bench host (over SSH)

            # The commands in this script are deliberately explicit
            # so you can copy-and-paste them selectively.

            cfgfile='{cfgfile}'

            # Mark the script as running.
            echo "$$" > {pidfile}
            echo "(the "'"'"{reqid.kind}"'"'" job, {reqid}, has started)"
            echo

            user=$(jq -r '.send_user' {cfgfile})
            if [ "$USER" != '{user}' ]; then
                echo "(switching users from $USER to {user})"
                echo
                {ensure_user}
            fi
            host=$(jq -r '.send_host' {cfgfile})
            port=$(jq -r '.send_port' {cfgfile})

            exitcode=0
            if {ssh(f'test -e {bfiles.request}')}; then
                >&2 echo "request {reqid} was already sent"
                exitcode=1
            else
                ( set -x

                # Set up before running.
                {ssh(f'mkdir -p {bfiles.request}')}
                {push(f'{reqdir}/*', bfiles.request)}
                {ssh(f'mkdir -p {bfiles.result}')}

                # Run the request.
                {ssh(bfiles.bench_script)}
                exitcode=$?

                # Finish up.
                # XXX Push from the bench host in run.sh instead of pulling here?
                {pull(bfiles.result.metadata, results_meta)}
                {resfiles}
                )
            fi

            # Unstage the request.
            pushd {SYS_PATH_ENTRY}
            {sys.executable} -u -m jobs internal-finish-run -v --config {cfgfile} {reqid}
            popd

            # Mark the script as complete.
            echo
            echo "(the "'"'"{reqid.kind}"'"'" job, {reqid} has finished)"
            #rm -f {pidfile}

            # Trigger the next job.
            pushd {SYS_PATH_ENTRY}
            {sys.executable} -u -m jobs internal-run-next -v --config {cfgfile} --logfile {queue_log} &

            exit $exitcode
        '''[1:-1])

    def check_ssh(self, *, onunknown=None, fail=True):
        filename = self._fs.ssh_okay
        if onunknown is None:
            save = False
        elif isinstance(onunknown, str):
            if onunknown == 'save':
                save = True
            elif onunknown == 'wait':
                _utils.wait_for_file(self._fs.ssh_okay)
                save = False
            elif onunknown.startswith('wait:'):
                _, _, timeout = onunknown.partition(':')
                if not timeout or not timeout.isdigit():
                    raise ValueError(f'invalid onunknown timeout ({onunknown})')
                _utils.wait_for_file(filename, timeout=int(timeout))
                save = False
            else:
                raise NotImplementedError(repr(onunknown))
        else:
            raise TypeError(f'expected str, got {onunknown!r}')
        text = _utils.read_file(filename, fail=False)
        if text is None:
            okay = self._worker.ssh.check()
            if save:
                with open(filename, 'w') as outfile:
                    outfile.write('0' if okay else '1')
                    print(file=outfile)
        else:
            text = text.strip()
            if text == '0':
                okay = True
            elif text == '1':
                okay = False
            else:
                if fail:
                    raise Exception(f'invalid ssh_okay {text!r} in {filename}')
                okay = False
        if fail and not okay:
            raise ConnectionRefusedError('SSH failed (is your SSH agent running and up-to-date?)')
        return okay

    def run(self, *, background=False):
        self.check_ssh(onunknown='save')
        if background:
            script = _utils.quote_shell_str(self._fs.portal_script)
            logfile = _utils.quote_shell_str(self._fs.logfile)
            _utils.run_bg(f'{script} > {logfile}')
            return 0
        else:
            proc = subprocess.run([self.fs.portal_script])
            return proc.returncode

    def get_pid(self):
        return self._pidfile.read()

    def kill(self):
        pid = self.get_pid()
        if pid:
            logger.info('# killing PID %s', pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.warn(f'job {self._reqid} no longer (PID: {pid})')
        # Kill the worker process, if running.
        text = self._worker.ssh.read(self._worker.fs.pidfile)
        if text and text.isdigit():
            self._worker.ssh.run_shell(f'kill {text}')

    def wait_until_started(self, *, checkssh=False):
        # XXX Add a timeout?
        pid = self.get_pid()
        while pid is None:
            status = self.get_status()
            if status in Result.FINISHED:
                raise JobNeverStartedError(reqid)
            if checkssh and os.path.exists(self.fs.ssh_okay):
                break
            time.sleep(0.01)
            pid = self.get_pid()
        return pid

    def attach(self, lines=None):
        pid = self.wait_until_started()
        try:
            if pid:
                _utils.tail_file(self.fs.logfile, lines, follow=pid)
            elif lines:
                _utils.tail_file(self.fs.logfile, lines, follow=False)
        except KeyboardInterrupt:
            # XXX Prompt to cancel the job?
            return

    def cancel(self, *, ifstatus=None):
        if ifstatus is not None:
            if job.get_status() not in (Result.STATUS.CREATED, ifstatus):
                return
        self.kill()
        # XXX Try to download the results directly?
        self.set_status('canceled')

    def close(self):
        result = self.load_result()
        result.close()
        result.save(self._fs.result.metadata, withextra=True)

    def render(self, fmt=None):
        if not fmt:
            fmt = 'summary'
        reqfile = self._fs.request.metadata
        resfile = self._fs.result.metadata
        if fmt in ('reqfile', 'resfile'):
            filename = reqfile if fmt == 'reqfile' else resfile
            yield f'(from {filename})'
            yield ''
            text = _utils.read_file(filename)
            for line in text.splitlines():
                yield f'  {line}'
        elif fmt == 'summary':
            yield from self._render_summary('verbose')
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')

    def render_for_row(self, attrs):
        if isinstance(attrs, str):
            raise NotImplementedError(attrs)
        try:
            res = self.load_result()
        except FileNotFoundError:
            status = started = finished = None
        else:
            status = res.status or 'created'
            started, _ = res.started
            finished, _ = res.finished

        def render_attr(name):
            if name == 'reqid':
                return str(self.reqid)
            elif name == 'status':
                return str(status) if status else '???'
            elif name == 'created':
                return f'{self.reqid.date:%Y-%m-%d %H:%M:%S}'
            elif name == 'started':
                return f'{started:%Y-%m-%d %H:%M:%S}' if started else '---'
            elif name == 'finished':
                return f'{finished:%Y-%m-%d %H:%M:%S}' if finished else '---'
            elif name == 'duration':
                if not started:
                    return '---'
                elif not finished:
                    return '...'
                else:
                    duration = finished - started
                    # The following is mostly borrowed from Timedelta.__str__().
                    mm, ss = divmod(duration.seconds, 60)
                    hh, mm = divmod(mm, 60)
                    hh += 24 * duration.days
                    return "%d:%02d:%02d" % (hh, mm, ss)
            else:
                raise NotImplementedError(name)

        for name in attrs:
            if ',' in name:
                primary, _, secondary = name.partition(',')
                primary = render_attr(primary)
                if primary in ('---', '...', '???'):
                    secondary = render_attr(secondary)
                    if secondary not in ('---', '...', '???'):
                        yield f'({secondary})'
                        continue
                yield f' {primary} '
            else:
                yield render_attr(name)

    def _render_summary(self, fmt=None):
        if not fmt:
            fmt = 'verbose'

        reqfs_fields = [
            'bench_script',
            'portal_script',
            'ssh_okay',
        ]
        resfs_fields = [
            'pidfile',
            'logfile',
        ]
        if self.kind is RequestID.KIND.BENCHMARKS:
            req_cls = _requests.BenchCompileRequest
            res_cls = _requests.BenchCompileResult
            reqfs_fields.extend([
                'manifest',
                'pyperformance_config',
            ])
            resfs_fields.extend([
                'pyperformance_log',
                'pyperformance_results',
            ])
        else:
            raise NotImplementedError(self.kind)

        fs = self._fs
        req = req_cls.load(fs.request.metadata)
        res = res_cls.load(fs.result.metadata)
        pid = _utils.PIDFile(fs.pidfile).read()

        try:
            staged = _get_staged_request(fs.jobs)
        except StagedRequestError:
            isstaged = False
        else:
            isstaged = (self.reqid == staged)

        if self.check_ssh(fail=False):
            ssh_okay = 'yes'
        elif os.path.exists(fs.ssh_okay):
            ssh_okay = 'no'
        else:
            ssh_okay = '???'

        if fmt == 'verbose':
            yield f'Request {self.reqid}:'
            yield f'  {"kind:":22} {req.kind}'
            yield f'  {"user:":22} {req.user}'
            if pid:
                yield f'  {"PID:":22} {pid}'
            yield f'  {"status:":22} {res.status or "(created)"}'
            yield f'  {"is staged:":22} {isstaged}'
            yield ''
            yield 'Details:'
            for field in req_cls.FIELDS:
                if field in ('id', 'reqid', 'kind', 'user', 'date', 'datadir'):
                    continue
                value = getattr(req, field)
                if isinstance(value, str) and value.strip() != value:
                    value = repr(value)
                yield f'  {field + ":":22} {value}'
            yield f'  {"ssh okay:":22} {ssh_okay}'
            yield ''
            yield 'History:'
            for st, ts in res.history:
                yield f'  {st + ":":22} {ts:%Y-%m-%d %H:%M:%S}'
            yield ''
            yield 'Request files:'
            yield f'  {"data root:":22} {_utils.render_file(req.reqdir)}'
            yield f'  {"metadata:":22} {_utils.render_file(fs.request.metadata)}'
            for field in reqfs_fields:
                value = getattr(fs.request, field, None)
                if value is None:
                    value = getattr(fs.work, field, None)
                yield f'  {field + ":":22} {_utils.render_file(value)}'
            yield ''
            yield 'Result files:'
            yield f'  {"data root:":22} {_utils.render_file(fs.result)}'
            yield f'  {"metadata:":22} {_utils.render_file(fs.result.metadata)}'
            for field in resfs_fields:
                value = getattr(fs.result, field, None)
                if value is None:
                    value = getattr(fs.work, field, None)
                yield f'  {field + ":":22} {_utils.render_file(value)}'
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')


class Jobs:

    FS = JobsFS

    def __init__(self, cfg, *, devmode=False):
        self._cfg = cfg
        self._devmode = devmode
        self._fs = self.FS(cfg.data_dir)
        self._worker = Worker.from_config(cfg, self.FS)

    def __str__(self):
        return self.fs.root

    @property
    def cfg(self):
        return self._cfg

    @property
    def devmode(self):
        return self._devmode

    @property
    def fs(self):
        """Files on the portal host."""
        return self._fs.copy()

    @property
    def queue(self):
        try:
            return self._queue
        except AttributeError:
            self._queue = _queue.JobQueue.from_fstree(self.fs)
            return self._queue

    def iter_all(self):
        for name in os.listdir(str(self._fs.requests)):
            reqid = RequestID.parse(name)
            if not reqid:
                continue
            yield self._get(reqid)

    def _get(self, reqid):
        return Job(
            reqid,
            self._fs.resolve_request(reqid),
            self._worker.resolve(reqid),
            self._cfg,
        )

    def get_current(self):
        reqid = _get_staged_request(self._fs)
        if not reqid:
            return None
        return self.get(reqid)

    def get(self, reqid):
        return self._get(reqid)

    def create(self, reqid, kind_kwargs=None, reqfsattrs=None):
        if kind_kwargs is None:
            kind_kwargs = {}

        job = self._get(reqid)
        job._create(kind_kwargs, reqfsattrs, self._fs.queue.log)
        return job

    def activate(self, reqid):
        logger.debug('# staging request')
        _stage_request(reqid, self.fs)
        logger.debug('# done staging request')
        job = self._get(reqid)
        job.set_status('active')
        return job

    def ensure_next(self):
        logger.debug('Making sure a job is running, if possible')
        # XXX Return (queued job, already running job).
        job = self.get_current()
        if job is not None:
            logger.debug('A job is already running (and will kick off the next one from the queue)')
            # XXX Check the pidfile.
            return
        queue = self.queue.snapshot
        if queue.paused:
            logger.debug('No job is running but the queue is paused')
            return
        if not queue:
            logger.debug('No job is running and none are queued')
            return
        # Run in the background.
        cfgfile = self._cfg.filename
        if not cfgfile:
            raise NotImplementedError
        logger.debug('No job is running so we will run the next one from the queue')
        _utils.run_bg(
            [
                sys.executable, '-u', '-m', 'jobs', '-v',
                'internal-run-next',
                '--config', cfgfile,
                #'--logfile', self._fs.queue.log,
            ],
            logfile=self._fs.queue.log,
            cwd=SYS_PATH_ENTRY,
        )

    def cancel_current(self, reqid=None, *, ifstatus=None):
        if not reqid:
            job = self.get_current()
            if job is None:
                raise NoRunningJobError()
        else:
            job = self._get(reqid)
        job.cancel(ifstatus=ifstatus)

        logger.info('# unstaging request %s', reqid)
        try:
            _unstage_request(job.reqid, self._fs)
        except RequestNotStagedError:
            pass
        logger.info('# done unstaging request')
        return job

    def finish_successful(self, reqid):
        logger.info('# unstaging request %s', reqid)
        try:
            _unstage_request(reqid, self._fs)
        except RequestNotStagedError:
            pass
        logger.info('# done unstaging request')

        job = self._get(reqid)
        job.close()
        return job


_SORT = {
    'reqid': (lambda j: j.reqid),
}


def sort_jobs(jobs, sortby=None, *, ascending=False):
    if isinstance(jobs, Jobs):
        jobs = list(jobs.iter_all())
    if not sortby:
        sortby = ['reqid']
    elif isinstance(sortby, str):
        sortby = sortby.split(',')
    done = set()
    for kind in sortby:
        if not kind:
            raise NotImplementedError(repr(kind))
        if kind in done:
            raise NotImplementedError(kind)
        done.add(kind)
        try:
            key = _SORT[kind]
        except KeyError:
            raise ValueError(f'unsupported sort kind {kind!r}')
        jobs = sorted(jobs, key=key, reverse=ascending)
    return jobs


def select_job(jobs, criteria=None):
    raise NotImplementedError


def select_jobs(jobs, criteria=None):
    # CSV
    # ranges (i.e. slice)
    if isinstance(jobs, Jobs):
        jobs = list(jobs.iter_all())
    if not criteria:
        yield from jobs
        return
    if isinstance(criteria, str):
        criteria = [criteria]
    else:
        try:
            criteria = list(criteria)
        except TypeError:
            criteria = [criteria]
    if len(criteria) > 1:
        raise NotImplementedError(criteria)
    selection = _utils.get_slice(criteria[0])
    if not isinstance(jobs, (list, tuple)):
        jobs = list(jobs)
    yield from jobs[selection]


##################################
# the current job

class StagedRequestError(Exception):
    pass


class RequestNotPendingError(StagedRequestError):

    def __init__(self, reqid, status=None):
        super().__init__(f'could not stage {reqid} (expected pending, got {status or "???"} status)')
        self.reqid = reqid
        self.status = status


class RequestAlreadyStagedError(StagedRequestError):

    def __init__(self, reqid, curid):
        super().__init__(f'could not stage {reqid} ({curid} already staged)')
        self.reqid = reqid
        self.curid = curid


class RequestNotStagedError(StagedRequestError):

    def __init__(self, reqid, curid=None):
        msg = f'{reqid} is not currently staged'
        if curid:
            msg = f'{msg} ({curid} is)'
        super().__init__(msg)
        self.reqid = reqid
        self.curid = curid


class StagedRequestResolveError(Exception):
    def __init__(self, reqid, reqdir, reason, msg):
        super().__init__(f'{reason} ({msg} - {reqdir})')
        self.reqid = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def _get_staged_request(pfiles):
    try:
        reqdir = os.readlink(pfiles.requests.current)
    except FileNotFoundError:
        return None
    requests, reqidstr = os.path.split(reqdir)
    if requests != pfiles.requests.root:
        return StagedRequestResolveError(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    reqid = RequestID.parse(reqidstr)
    if not reqid:
        return StagedRequestResolveError(None, reqdir, 'invalid', f'{reqidstr!r} not a request ID')
    if not os.path.exists(reqdir):
        return StagedRequestResolveError(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        return StagedRequestResolveError(reqid, reqdir, 'malformed', 'target is not a directory')
    reqfs = pfiles.resolve_request(reqid)
    # Check if the request is still running.
    status = Result.read_status(str(reqfs.result.metadata), fail=False)
    if not status or status in ('created', 'pending'):
        logger.error(f'request {reqid} was set as the current job incorrectly; unsetting...')
        os.unlink(pfiles.requests.current)
        reqid = None
    elif status not in ('active', 'running'):
        logger.warn(f'request {reqid} is still "current" even though it finished; unsetting...')
        os.unlink(pfiles.requests.current)
        reqid = None
    elif not _utils.PIDFile(str(reqfs.pidfile)).read(orphaned='ignore'):
        logger.warn(f'request {reqid} is no longer running; unsetting as the current job...')
        os.unlink(pfiles.requests.current)
        reqid = None
    # XXX Do other checks?
    return reqid


def _stage_request(reqid, pfiles):
    jobfs = pfiles.resolve_request(reqid)
    status = Result.read_status(jobfs.result.metadata, fail=False)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
    try:
        os.symlink(jobfs.request, pfiles.requests.current)
    except FileExistsError:
        # XXX Delete the existing one if bogus?
        curid = _get_staged_request(pfiles) or '???'
        if isinstance(curid, Exception):
            raise RequestAlreadyStagedError(reqid, '???') from curid
        else:
            raise RequestAlreadyStagedError(reqid, curid)


def _unstage_request(reqid, pfiles):
    reqid = RequestID.from_raw(reqid)
    curid = _get_staged_request(pfiles)
    if not curid or not isinstance(curid, (str, RequestID)):
        raise RequestNotStagedError(reqid)
    elif str(curid) != str(reqid):
        raise RequestNotStagedError(reqid, curid)
    os.unlink(pfiles.requests.current)


##################################
# avoid circular imports

from . import queue as _queue
