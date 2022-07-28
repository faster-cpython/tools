import json
import logging
import os
import os.path
import signal
import subprocess
import sys
import textwrap
import time
import types

from . import _utils, _pyperformance, _common
from .requests import RequestID, Request

# top-level exports
from ._common import JobsError


PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PATH_ENTRY = os.path.dirname(PKG_ROOT)

logger = logging.getLogger(__name__)


##################################
# jobs config

class JobsConfig(_utils.TopConfig):
    """The jobs-related configuration used on the portal host."""

    FIELDS = ['local_user', 'worker', 'data_dir']
    OPTIONAL = ['data_dir']

    FILE = 'jobs.json'
    CONFIG_DIRS = [
        f'{_utils.HOME}/BENCH',
    ]

    def __init__(self,
                 local_user,
                 worker,
                 data_dir=None,
                 **ignored
                 ):
        if not local_user:
            raise ValueError('missing local_user')
        if not worker:
            raise ValueError('missing worker')
        elif not isinstance(worker, WorkerConfig):
            worker = WorkerConfig.from_jsonable(worker)
        if data_dir:
            data_dir = os.path.abspath(os.path.expanduser(data_dir))
        else:
            data_dir = f'/home/{local_user}/BENCH'  # This matches DATA_ROOT.
        super().__init__(
            local_user=local_user,
            worker=worker,
            data_dir=data_dir or None,
        )

    @property
    def ssh(self):
        return self.worker.ssh


##################################
# job files

class JobFS(_common.JobFS):
    """The file structure of a job's data."""

    context = 'portal'

    @classmethod
    def _get_jobsfs(cls, root):
        return JobsFS(root)

    def _custom_init(self):
        work = self.work
        work.pidfile = f'{work}/send.pid'  # overrides base
        work.portal_script = f'{work}/send.sh'
        work.ssh_okay = f'{work}/ssh.ok'

    @property
    def portal_script(self):
        return self.work.portal_script

    @property
    def ssh_okay(self):
        return self.work.ssh_okay


class JobsFS(_common.JobsFS):
    """The file structure of the jobs data."""

    context = 'portal'

    JOBFS = JobFS

    def __init__(self, root='~/BENCH'):
        super().__init__(root)

        self.requests.current = f'{self.requests}/CURRENT'

        self.queue = _utils.FSTree(f'{self.requests}/queue.json')
        self.queue.data = f'{self.requests}/queue.json'
        self.queue.lock = f'{self.requests}/queue.lock'
        self.queue.log = f'{self.requests}/queue.log'


class RequestDirError(Exception):
    def __init__(self, reqid, reqdir, reason, msg):
        super().__init__(f'{reason} ({msg} - {reqdir})')
        self.reqid = reqid
        self.reqdir = reqdir
        self.reason = reason
        self.msg = msg


def _check_reqdir(reqdir, pfiles, cls=RequestDirError):
    requests, reqidstr = os.path.split(reqdir)
    if requests != pfiles.requests.root:
        raise cls(None, reqdir, 'invalid', 'target not in ~/BENCH/REQUESTS/')
    reqid = RequestID.parse(reqidstr)
    if not reqid:
        raise cls(None, reqdir, 'invalid', f'{reqidstr!r} not a request ID')
    if not os.path.exists(reqdir):
        raise cls(reqid, reqdir, 'missing', 'target request dir missing')
    if not os.path.isdir(reqdir):
        raise cls(reqid, reqdir, 'malformed', 'target is not a directory')
    return reqid


##################################
# workers

class WorkerConfig(_utils.Config):

    FIELDS = ['user', 'ssh_host', 'ssh_port']

    def __init__(self,
                 user,
                 ssh_host,
                 ssh_port,
                 **ignored
                 ):
        if not user:
            raise ValueError('missing user')
        ssh = _utils.SSHConnectionConfig(user, ssh_host, ssh_port)
        super().__init__(
            user=user,
            ssh=ssh,
        )

    def as_jsonable(self):
        return {
            'user': self.user,
            'ssh_host': self.ssh.host,
            'ssh_port': self.ssh.port,
        }


class WorkerJobFS(_common.JobFS):
    """The file structure of a job's data on a worker."""

    context = 'job-worker'

    @classmethod
    def _get_jobsfs(cls, root):
        return WorkerJobsFS(root)


class WorkerJobsFS(_common.JobsFS):
    """The file structure of the jobs data on a worker."""

    context = 'job-worker'

    JOBFS = WorkerJobFS

    def __init__(self, root='~/BENCH'):
        super().__init__(root)

        # the local git repositories used by the job
        self.repos = _utils.FSTree(f'{self}/repositories')
        self.repos.cpython = f'{self.repos}/cpython'
        self.repos.pyperformance = f'{self.repos}/pyperformance'
        self.repos.pyston_benchmarks = f'{self.repos}/pyston-benchmarks'


class JobWorker:
    """A worker assigned to run a requested job."""

    def __init__(self, worker, fs):
        self._worker = worker
        self._fs = fs

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'worker fs'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def fs(self):
        return self._fs

    @property
    def topfs(self):
        return self._worker.fs

    @property
    def ssh(self):
        return self._worker.ssh


class Worker:
    """A single configured worker."""

    @classmethod
    def from_config(cls, cfg, JobsFS=WorkerJobsFS):
        fs = JobsFS.from_user(cfg.user)
        ssh = _utils.SSHClient.from_config(cfg.ssh)
        return cls(fs, ssh)

    def __init__(self, fs, ssh):
        self._fs = fs
        self._ssh = ssh

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'fs ssh'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def fs(self):
        return self._fs

    @property
    def ssh(self):
        return self._ssh

    def resolve_job(self, reqid):
        fs = self._fs.resolve_request(reqid)
        return JobWorker(self, fs)


class Workers:
    """The set of configured workers."""

    @classmethod
    def from_config(cls, cfg, JobsFS=WorkerJobsFS):
        worker = Worker.from_config(cfg.worker, JobsFS)
        return cls(worker)

    def __init__(self, worker):
        self._worker = worker

    def __repr__(self):
        args = (f'{n}={getattr(self, "_"+n)!r}'
                for n in 'worker'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    def resolve_job(self, reqid):
        return self._worker.resolve_job(reqid)


##################################
# jobs

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


class JobFinishedError(JobNotRunningError):
    MSG = 'job {reqid} is done'


class JobAlreadyFinishedError(JobFinishedError):
    MSG = 'job {reqid} is already done'


class Job:

    def __init__(self, reqid, fs, worker, cfg, store):
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
        self._store = store
        self._pidfile = _utils.PIDFile(fs.pidfile)
        self._kind = _common.resolve_job_kind(self._reqid.kind)

    def __repr__(self):
        args = (f'{n}={str(getattr(self, "_"+n))!r}'
                for n in 'reqid fs worker cfg store'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __str__(self):
        return str(self._reqid)

    def __eq__(self, other):
        raise NotImplementedError

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
        return Request(self._reqid, str(self._fs))

    def load_request(self, *, fail=True):
        req = self._kind.Request.load(
            self._fs.request.metadata,
            fs=self._fs.request,
            fail=fail,
        )
        return req

    def load_result(self, *, fail=True):
        res = self._kind.Result.load(
            self._fs.result.metadata,
            fs=self._fs.result,
            request=(lambda *a, **k: self.load_request()),
            fail=fail,
        )
        return res

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

    def _create(self, kind_kwargs, pushfsattrs, pullfsattrs, queue_log):
        os.makedirs(self._fs.request.root, exist_ok=True)
        os.makedirs(self._fs.work.root, exist_ok=True)
        os.makedirs(self._fs.result.root, exist_ok=True)

        req, worker_script = self._kind.create(
            self._reqid,
            self._fs,
            self._worker.topfs,
            **kind_kwargs,
        )

        # Write metadata.
        req.save(self._fs.request.metadata)
        req.result.save(self._fs.result.metadata)

        # Write the commands to execute remotely.
        with open(self._fs.job_script, 'w') as outfile:
            outfile.write(worker_script)
        os.chmod(self._fs.job_script, 0o755)

        # Write the commands to execute locally.
        script = self._build_send_script(queue_log, pushfsattrs, pullfsattrs)
        with open(self._fs.portal_script, 'w') as outfile:
            outfile.write(script)
        os.chmod(self._fs.portal_script, 0o755)

    def _build_send_script(self, queue_log, pushfsfields=None, pullfsfields=None, *,
                           hidecfg=False,
                           ):
        reqid = self.reqid
        pfiles = self._fs
        bfiles = self._worker.fs

        if not self._cfg.filename:
            raise NotImplementedError(self._cfg)
        cfgfile = _utils.quote_shell_str(self._cfg.filename)
        if hidecfg:
            ssh = _utils.SSHShellCommands('$ssh_user', '$ssh_host', '$ssh_port')
            user = '$user'
        else:
            user = _utils.check_shell_str(self._cfg.local_user)
            _utils.check_shell_str(self._cfg.ssh.user)
            _utils.check_shell_str(self._cfg.ssh.host)
            ssh = self._worker.ssh.shell_commands

        queue_log = _utils.quote_shell_str(queue_log)

        reqdir = _utils.quote_shell_str(pfiles.request.root)
        results_meta = _utils.quote_shell_str(pfiles.result.metadata)
        pidfile = _utils.quote_shell_str(pfiles.pidfile)

        _utils.check_shell_str(bfiles.request.root)
        _utils.check_shell_str(bfiles.job_script)
        _utils.check_shell_str(bfiles.work.root)
        _utils.check_shell_str(bfiles.result.root)
        _utils.check_shell_str(bfiles.result.metadata)

        ensure_user = ssh.ensure_user(user, agent=False)
        ensure_user = '\n                '.join(ensure_user)

        pushfsfields = [
            # Technically we don't need request.json on the worker,
            # but it can help with debugging.
            ('request', 'metadata'),
            ('work', 'job_script'),
            ('result', 'metadata'),
            *pushfsfields,
        ]
        pushfiles = []
        for field in pushfsfields or ():
            if isinstance(field, str):
                field = ('request', field)
            area, name = field
            pvalue = pfiles.look_up(area, name)
            pvalue = _utils.quote_shell_str(pvalue)
            bvalue = bfiles.look_up(area, name)
            _utils.check_shell_str(bvalue)
            pushfiles.append((bvalue, pvalue))
        pushfiles = (ssh.push(s, t) for s, t in pushfiles)
        pushfiles = '\n                '.join(pushfiles)

        pullfsfields = [
            ('result', 'metadata'),
            *pullfsfields,
        ]
        pullfiles = []
        for field in pullfsfields or ():
            if isinstance(field, str):
                field = ('result', field)
            area, name = field
            pvalue = pfiles.look_up(area, name)
            pvalue = _utils.quote_shell_str(pvalue)
            bvalue = bfiles.look_up(area, name)
            _utils.check_shell_str(bvalue)
            pullfiles.append((bvalue, pvalue))
        pullfiles = (ssh.pull(s, t) for s, t in pullfiles)
        pullfiles = '\n                '.join(pullfiles)

        #push = ssh.push
        #pull = ssh.pull
        ssh = ssh.run_shell

        return textwrap.dedent(f'''
            #!/usr/bin/env bash

            # This script only runs on the portal host.
            # It does 4 things:
            #   1. switch to the {user} user, if necessary
            #   2. prepare the job worker, including sending all
            #      the request files to the worker (over SSH)
            #   3. run the job (e.g. run the benchmarks)
            #   4. pull the results-related files from the worker (over SSH)

            # The commands in this script are deliberately explicit
            # so you can copy-and-paste them selectively.

            cfgfile='{cfgfile}'

            # Mark the script as running.
            echo "$$" > {pidfile}
            echo "(the "'"'"{reqid.kind}"'"'" job, {reqid}, has started)"
            echo

            user=$(jq -r '.local_user' {cfgfile})
            if [ "$USER" != '{user}' ]; then
                echo "(switching users from $USER to {user})"
                echo
                {ensure_user}
            fi
            ssh_user=$(jq -r '.worker.user' {cfgfile})
            ssh_host=$(jq -r '.worker.ssh_host' {cfgfile})
            ssh_port=$(jq -r '.worker.ssh_port' {cfgfile})

            exitcode=0
            if {ssh(f'test -e {bfiles.request}')}; then
                >&2 echo "request {reqid} was already sent"
                exitcode=1
            else
                ( set -x

                # Set up before running.
                {ssh(f'mkdir -p {bfiles.request}')}
                {ssh(f'mkdir -p {bfiles.work}')}
                {ssh(f'mkdir -p {bfiles.result}')}
                {pushfiles}

                # Run the request.
                {ssh(bfiles.job_script)}
                exitcode=$?

                # Finish up.
                # XXX Push from the worker in run.sh instead of pulling here?
                {pullfiles}

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

    def _get_ssh_agent(self):
        agent = self._cfg.worker.ssh.agent
        if not agent or not agent.check():
            agent = _utils.SSHAgentInfo.find_latest()
            if not agent:
                agent = SSHAgentInfo.from_env_vars()
        if agent:
            logger.debug(f'(using SSH agent at {agent.auth_sock})')
        else:
            logger.debug('(no SSH agent running)')
        return agent

    def check_ssh(self, *, onunknown=None, agent=None, fail=True):
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
            if not agent:
                agent = self._get_ssh_agent()
            okay = self._worker.ssh.check(agent=agent)
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
        agent = self._get_ssh_agent()
        env = agent.apply_env() if agent else None
        self.check_ssh(onunknown='save', agent=agent)
        if background:
            script = _utils.quote_shell_str(self._fs.portal_script)
            logfile = _utils.quote_shell_str(self._fs.logfile)
            _utils.run_bg(f'{script} > {logfile}', env=env)
            return 0
        else:
            proc = subprocess.run([self._fs.portal_script], env=env)
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
        agent = self._get_ssh_agent()
        text = self._worker.ssh.read(self._worker.fs.pidfile, agent=agent)
        if text and text.isdigit():
            self._worker.ssh.run_shell(f'kill {text}', agent=agent)

    def wait_until_started(self, timeout=None, *, checkssh=False):
        if not timeout or timeout < 0:
            timeout = None
        else:
            end = time.time() + timeout
        # First make sure it is queued or running.
        status = self.get_status()
        if status in Result.FINISHED:
            raise JobAlreadyFinishedError(self.reqid)
        elif status not in Result.ACTIVE:
            raise JobNeverStartedError(self.reqid)
        # Go until it has a PID or it finiishes.
        pid = self.get_pid()
        while pid is None:
            if checkssh and os.path.exists(self._fs.ssh_okay):
                break
            if timeout and time.time() > end:
                raise TimeoutError(f'timed out after {timeout} seconds')
            time.sleep(0.01)
            status = self.get_status()
            if status in Result.FINISHED:
                # It finished but a PID file wasn't left behind.
                raise JobFinishedError(self.reqid)
            pid = self.get_pid()
        return pid

    def attach(self, lines=None):
        pid = self.wait_until_started()
        try:
            if pid:
                _utils.tail_file(self._fs.logfile, lines, follow=pid)
            elif lines:
                _utils.tail_file(self._fs.logfile, lines, follow=False)
        except KeyboardInterrupt:
            # XXX Prompt to cancel the job?
            return

    def wait_until_finished(self, pid=None, *, timeout=True):
        if timeout is True:
            # Default to double the typical.
            timeout = self._kind.TYPICAL_DURATION_SECS or None
            if timeout:
                timeout *= 2
        elif not timeout or timeout < 0:
            timeout = None
        if timeout:
            end = time.time() + timeout
        if not pid:
            try:
                pid = self.wait_until_started(timeout)
            except JobFinishedError:
                return
        while _utils.is_proc_running(pid):
            if timeout and time.time() > end:
                raise TimeoutError(f'timed out after {timeout} seconds')
            time.sleep(0.1)
        # Make sure it finished.
        status = self.get_status()
        if status not in Result.FINISHED:
            assert status not in Result.ACTIVE, (self, status)
            raise JobNeverStartedError(self.reqid)

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

    def upload_result(self, author=None, *, clean=True, push=True):
        res = self.load_result()
        # We upload directly.
        self._store.add(
            res.pyperf,
            branch='main',
            author=author,
            compressed=False,
            split=True,
            clean=clean,
            push=push,
        )

    def as_row(self):  # XXX Move to JobSummary.
        try:
            res = self.load_result()
        except FileNotFoundError:
            status = started = finished = None
        else:
            status = res.status or 'created'
            started, _ = res.started
            finished, _ = res.finished
        if not started:
            elapsed = None
        else:
            end = finished
            if not end:
                end, _ = _utils.get_utc_datetime()
            elapsed = end - started
        date = (started, self.reqid.date)
        if not any(date):
            date = None
        fullref = ref = remote = branch = tag = commit = None
        req = self.load_request(fail=False)
        if req:
            fullref, ref, remote, branch, tag, commit = self._kind.as_row(req)
        data = {
            'reqid': self.reqid,
            'status': status,
            'date': date,
            'created': self.reqid.date,
            'started': started,
            'finished': finished,
            'elapsed': elapsed,
            'ref': ref,
            'fullref': fullref,
            'remote': remote,
            'branch': branch,
            'tag': tag,
            'commit': commit,
        }

        def render_value(colname):
            raw = data[colname]
            if raw is None:
                if colname == 'status':
                    rendered = '???'
                else:
                    rendered = '---'
            elif colname == 'reqid':
                rendered = str(raw)
            elif colname == 'status':
                rendered = str(raw)
            elif colname in ('created', 'started', 'finished'):
                rendered = f'{raw:%Y-%m-%d %H:%M:%S}'
            elif colname == 'date':
                started, created = raw
                if started:
                    rendered = f' {started:%Y-%m-%d %H:%M:%S} '
                else:
                    rendered = f'({created:%Y-%m-%d %H:%M:%S})'
            elif colname == 'elapsed':
                fmt = "%d:%02d:%02d"
                fmt = f' {fmt} ' if finished else f'({fmt})'
                # The following is mostly borrowed from Timedelta.__str__().
                mm, ss = divmod(raw.seconds, 60)
                hh, mm = divmod(mm, 60)
                hh += 24 * raw.days
                rendered = fmt % (hh, mm, ss)
            elif colname in 'ref':
                rendered = str(raw)
            elif isinstance(raw, str):
                rendered = raw
            else:
                raise NotImplementedError(colname)
            return rendered

        return _utils.TableRow(data, render_value)

    # XXX Add as_summary().

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

    def _render_summary(self, fmt=None):
        if not fmt:
            fmt = 'verbose'

        reqfs_fields = self._kind.REQFS_FIELDS
        resfs_fields = self._kind.RESFS_FIELDS

        fs = self._fs
        req = self.load_request()
        res = self.load_result()
        pid = _utils.PIDFile(fs.pidfile).read()

        staged = _get_staged_request(fs.jobs)
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
            for field in self._kind.Request.FIELDS:
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
        self._workers = Workers.from_config(cfg)
        self._store = _pyperformance.FasterCPythonResults.from_remote()

    def __str__(self):
        return self._fs.root

    def __eq__(self, other):
        raise NotImplementedError

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
            self._queue = _queue.JobQueue.from_fstree(self._fs)
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
            self._workers.resolve_job(reqid),
            self._cfg,
            self._store,
        )

    def get_current(self):
        reqid = _get_staged_request(self._fs)
        if not reqid:
            return None
        return self._get(reqid)

    def get(self, reqid=None):
        if not reqid:
            return self.get_current()
        orig = reqid
        reqid = RequestID.from_raw(orig)
        if not reqid:
            if isinstance(orig, str):
                reqid = self._parse_reqdir(orig)
            if not reqid:
                return None
        return self._get(reqid)

    def _parse_reqdir(self, filename):
        dirname, basename = os.path.split(filename)
        reqid = RequestID.parse(basename)
        if not reqid:
            # It must be a file in the request dir.
            reqid = RequestID.parse(os.path.basename(dirname))
        return reqid

    def match_results(self, specifier, suites=None):
        matched = list(self._store.match(specifier, suites=suites))
        if matched:
            yield from matched
        else:
            yield from self._match_job_results(specifier, suites)

    def _match_job_results(self, specifier, suites):
        if isinstance(specifier, str):
            job = self.get(specifier)
            if job:
                filename = job.fs.result.pyperformance_results
                if suites:
                    # XXX Handle this?
                    pass
                yield _pyperformance.PyperfResultsFile(
                    filename,
                    resultsroot=self._fs.results.root,
                )

    def create(self, reqid, kind_kwargs=None, pushfsattrs=None, pullfsattrs=None):
        if kind_kwargs is None:
            kind_kwargs = {}
        job = self._get(reqid)
        job._create(kind_kwargs, pushfsattrs, pullfsattrs, self._fs.queue.log)
        return job

    def activate(self, reqid):
        logger.debug('# staging request')
        _stage_request(reqid, self._fs)
        logger.debug('# done staging request')
        job = self._get(reqid)
        job.set_status('activated')
        return job

    def wait_until_job_started(self, job=None, *, timeout=True):
        current = _get_staged_request(self._fs)
        if isinstance(job, Job):
            reqid = job.reqid
        else:
            reqid = job
            if not reqid:
                reqid = current
                if not reqid:
                    raise NoRunningJobError
            job = self._get(reqid)
        if timeout is True:
            # Calculate the timeout.
            if current:
                if reqid == current:
                    timeout = 0
                else:
                    try:
                        jobkind = _common.resolve_job_kind(reqid.kind)
                    except KeyError:
                        raise NotImplementedError(reqid)
                    expected = jobkind.TYPICAL_DURATION_SECS
                    # We could subtract the elapsed time, but it isn't worth it.
                    timeout = expected
            if timeout:
                # Add the expected time for everything in the queue before the job.
                if timeout is True:
                    timeout = 0
                for i, queued in enumerate(self.queue.snapshot):
                    if queued == reqid:
                        # Play it safe by doubling the timeout.
                        timeout *= 2
                        break
                    try:
                        jobkind = _common.resolve_job_kind(queued.kind)
                    except KeyError:
                        raise NotImplementedError(queued)
                    expected = jobkind.TYPICAL_DURATION_SECS
                    timeout += expected
                else:
                    # Either it hasn't been queued or it already finished.
                    timeout = 0
        # Wait!
        pid = job.wait_until_started(timeout)
        return job, pid

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
        job = self.get(reqid)
        if job is None:
            raise NoRunningJobError
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

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class StagedRequestDirError(StagedRequestError, RequestDirError):

    def __init__(self, reqid, reqdir, reason, msg):
        RequestDirError.__init__(self, reqid, reqdir, reason, msg)


class StagedRequestStatusError(StagedRequestError):

    reason = None

    def __init__(self, reqid, status):
        assert self.reason
        super().__init__(reqid, self.reason)
        self.status = status


class OutdatedStagedRequestError(StagedRequestStatusError):
    pass


class StagedRequestNotRunningError(OutdatedStagedRequestError):
    reason = 'is no longer running'


class StagedRequestAlreadyFinishedError(OutdatedStagedRequestError):
    reason = 'is still "current" even though it finished'


class StagedRequestUnexpectedStatusError(StagedRequestStatusError):
    reason = 'was set as the current job incorrectly'


class StagedRequestInvalidMetadataError(StagedRequestStatusError):
    reason = 'has invalid metadata'


class StagedRequestMissingStatusError(StagedRequestInvalidMetadataError):
    reason = 'is missing status metadata'


class StagingRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class RequestNotPendingError(StagingRequestError):

    def __init__(self, reqid, status=None):
        super().__init__(reqid, f'could not stage {reqid} (expected pending, got {status or "???"} status)')
        self.status = status


class RequestAlreadyStagedError(StagingRequestError):

    def __init__(self, reqid, curid):
        super().__init__(reqid, f'could not stage {reqid} ({curid} already staged)')
        self.curid = curid


class UnstagingRequestError(Exception):

    def __init__(self, reqid, msg):
        super().__init__(msg)
        self.reqid = reqid


class RequestNotStagedError(UnstagingRequestError):

    def __init__(self, reqid, curid=None):
        msg = f'{reqid} is not currently staged'
        if curid:
            msg = f'{msg} ({curid} is)'
        super().__init__(reqid, msg)
        self.curid = curid


def _read_staged(pfiles):
    link = pfiles.requests.current
    try:
        reqdir = os.readlink(link)
    except FileNotFoundError:
        return None
    except OSError:
        if os.path.islink(link):
            raise  # re-raise
        exc = RequestDirError(None, link, 'malformed', 'target is not a link')
        try:
            exc.reqid = _check_reqdir(link, pfiles, StagedRequestDirError)
        except StagedRequestDirError:
            raise exc
        raise exc
    else:
        return _check_reqdir(reqdir, pfiles, StagedRequestDirError)


def _check_staged_request(reqid, pfiles):
    # Check the request status.
    reqfs = pfiles.resolve_request(reqid)
    try:
        status = Result.read_status(str(reqfs.result.metadata))
    except _utils.MissingMetadataError:
        raise StagedRequestMissingStatusError(reqid, None)
    except _utils.InvalidMetadataError:
        raise StagedRequestInvalidMetadataError(reqid, None)
    else:
        if status in Result.ACTIVE and status != 'pending':
            if not _utils.PIDFile(str(reqfs.pidfile)).read(orphaned='ignore'):
                raise StagedRequestNotRunningError(reqid, status)
        elif status in Result.FINISHED:
            raise StagedRequestAlreadyFinishedError(reqid, status)
        else:  # created, pending
            raise StagedRequestUnexpectedStatusError(reqid, status)
    # XXX Do other checks?


def _set_staged(reqid, reqdir, pfiles):
    try:
        os.symlink(reqdir, pfiles.requests.current)
    except FileExistsError:
        try:
            curid = _read_staged(pfiles)
        except StagedRequestError as exc:
            # One was already set but the link is invalid.
            _clear_staged(pfiles, exc)
        else:
            if curid == reqid:
                # XXX Fail?
                logger.warn(f'{reqid} is already set as the current job')
                return
            elif curid:
                # Clear it if it is no longer valid.
                try:
                    _check_staged_request(curid, pfiles)
                except StagedRequestError as exc:
                    _clear_staged(pfiles, exc)
                else:
                    raise RequestAlreadyStagedError(reqid, curid)
        logger.info('trying again')
        # XXX Guard against infinite recursion?
        return _set_staged(reqid, reqdir, pfiles)


def _clear_staged(pfiles, exc=None):
    if exc is not None:
        if isinstance(exc, StagedRequestInvalidMetadataError):
            log = logger.error
        else:
            log = logger.warn
        reqid = getattr(exc, 'reqid', None)
        reason = getattr(exc, 'reason', None) or 'broken'
        if not reason.startswith(('is', 'was', 'has')):
            reason = f'is {reason}'
        log(f'request {reqid or "???"} {reason} ({exc}); unsetting as the current job...')
    os.unlink(pfiles.requests.current)


# These are the higher-level helpers:

def _get_staged_request(pfiles):
    try:
        curid = _read_staged(pfiles)
    except StagedRequestError as exc:
        _clear_staged(pfiles, exc)
        return None
    if curid:
        try:
            _check_staged_request(curid, pfiles)
        except StagedRequestError as exc:
            _clear_staged(pfiles, exc)
            curid = None
    return curid


def _stage_request(reqid, pfiles):
    jobfs = pfiles.resolve_request(reqid)
    status = Result.read_status(jobfs.result.metadata, fail=False)
    if status is not Result.STATUS.PENDING:
        raise RequestNotPendingError(reqid, status)
    _set_staged(reqid, jobfs.request, pfiles)


def _unstage_request(reqid, pfiles):
    reqid = RequestID.from_raw(reqid)
    try:
        curid = _read_staged(pfiles)
    except StagedRequestError as exc:
        # One was already set but the link is invalid.
        _clear_staged(pfiles, exc)
        raise RequestNotStagedError(reqid)
    else:
        if curid == reqid:
            # It's a match!
            _clear_staged(pfiles)
        else:
            if curid:
                # Clear it if it is no longer valid.
                try:
                    _check_staged_request(curid, pfiles)
                except StagedRequestError as exc:
                    _clear_staged(pfiles, exc)
            raise RequestNotStagedError(reqid)


##################################
# avoid circular imports

from . import queue as _queue
