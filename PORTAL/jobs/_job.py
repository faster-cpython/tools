import logging
import os
import os.path
import signal
import subprocess
import sys
import textwrap
import time
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from . import _utils, _common, _workers, _current
from .requests import RequestID, Request, Result, ToRequestIDType


if TYPE_CHECKING:
    from . import JobsConfig


logger = logging.getLogger(__name__)


class JobError(_common.JobsError):
    MSG = 'job {reqid} has a problem'

    def __init__(self, reqid: RequestID, msg: Optional[str] = None):
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


class JobFS(_common.JobFS):
    """The file structure of a job's data."""

    context = 'portal'

    def _custom_init(self):
        work = self.work
        work.pidfile = f'{work}/send.pid'  # overrides base
        work.portal_script = f'{work}/send.sh'
        work.ssh_okay = f'{work}/ssh.ok'

    @property
    def portal_script(self) -> str:
        return self.work.portal_script

    @property
    def ssh_okay(self) -> str:
        return self.work.ssh_okay


class Job:
    """A single requested job."""

    def __init__(
            self,
            reqid: ToRequestIDType,
            fs: _common.JobFS,
            worker: _workers.JobWorker,
            cfg: "JobsConfig",
            store: Any = None
    ):
        if not fs:
            raise ValueError('missing fs')
        elif not isinstance(fs, JobFS):
            raise TypeError(f'expected JobFS for fs, got {fs!r}')
        if not worker:
            raise ValueError('missing worker')
        elif not isinstance(worker, _workers.JobWorker):
            raise TypeError(f'expected JobWorker for worker, got {worker!r}')
        resolved_reqid = RequestID.from_raw(reqid)
        if not resolved_reqid:
            raise ValueError('missing reqid')
        self._reqid = resolved_reqid
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
    def reqid(self) -> RequestID:
        return self._reqid

    @property
    def fs(self) -> _common.JobFS:
        return self._fs

    @property
    def worker(self) -> _workers.JobWorker:
        return self._worker

    @property
    def cfg(self) -> "JobsConfig":
        return self._cfg

    @property
    def kind(self) -> str:
        return self.reqid.kind

    @property
    def request(self) -> Request:
        return Request(self._reqid, str(self._fs))

    def load_request(self, *, fail: bool = True) -> Request:
        req = self._kind.Request.load(
            self._fs.request.metadata,
            fs=self._fs.request,
            fail=fail,
        )
        return req

    def load_result(self, *, fail: bool = True) -> Result:
        res = self._kind.Result.load(
            self._fs.result.metadata,
            fs=self._fs.result,
            request=(lambda *a, **k: self.load_request()),
            fail=fail,
        )
        return res

    def get_status(self, *, fail: bool = True) -> Optional[str]:
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

    def set_status(self, status: str) -> None:
        status = Result.resolve_status(status)
        result = self.load_result()
        result.set_status(status)
        result.save(self._fs.result.metadata, withextra=True)

    def _create(
            self,
            kind_kwargs: Mapping[str, Any],
            pushfsattrs: Optional[_common.FsAttrsType],
            pullfsattrs: Optional[_common.FsAttrsType],
            queue_log: str
    ):
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

    def _build_send_script(
            self,
            queue_log: str,
            pushfsfields: Optional[_common.FsAttrsType] = None,
            pullfsfields: Optional[_common.FsAttrsType] = None,
            *,
            hidecfg: bool = False,
    ) -> str:
        reqid = self.reqid
        pfiles = self._fs
        bfiles = self._worker.fs

        if not self._cfg.filename:
            raise NotImplementedError(self._cfg)
        cfgfile = _utils.quote_shell_str(self._cfg.filename)
        if hidecfg:
            sshcmds = _utils.SSHShellCommands('$ssh_user', '$ssh_host', '$ssh_port')
            user = '$user'
        else:
            cfg_user = _utils.check_shell_str(self._cfg.local_user)
            if cfg_user is None:
                raise ValueError("Couldn't find local_user")
            user = cfg_user
            _utils.check_shell_str(self._cfg.ssh.user)
            _utils.check_shell_str(self._cfg.ssh.host)
            sshcmds = self._worker.ssh.shell_commands

        queue_log = _utils.quote_shell_str(queue_log)

        reqdir = _utils.quote_shell_str(pfiles.request.root)
        results_meta = _utils.quote_shell_str(pfiles.result.metadata)
        pidfile = _utils.quote_shell_str(pfiles.pidfile)

        _utils.check_shell_str(bfiles.request.root)
        _utils.check_shell_str(bfiles.job_script)
        _utils.check_shell_str(bfiles.work.root)
        _utils.check_shell_str(bfiles.result.root)
        _utils.check_shell_str(bfiles.result.metadata)

        ensure_user = sshcmds.ensure_user(user, agent=False)
        ensure_user = '\n                '.join(ensure_user)

        pushfsfields = [
            # Technically we don't need request.json on the worker,
            # but it can help with debugging.
            ('request', 'metadata'),
            ('work', 'job_script'),
            ('result', 'metadata'),
            *(pushfsfields or []),
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
        pushfiles_str = '\n                '.join(
            sshcmds.push(s, t) for s, t in pushfiles
        )

        pullfsfields = [
            ('result', 'metadata'),
            *(pullfsfields or []),
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
        pullfiles_str = '\n                '.join(
            sshcmds.pull(s, t) for s, t in pullfiles
        )

        #push = ssh.push
        #pull = ssh.pull
        ssh = sshcmds.run_shell

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
                {pushfiles_str}

                # Run the request.
                {ssh(bfiles.job_script)}
                exitcode=$?

                # Finish up.
                # XXX Push from the worker in run.sh instead of pulling here?
                {pullfiles_str}

                )
            fi

            # Unstage the request.
            pushd {_common.SYS_PATH_ENTRY}
            {sys.executable} -u -m jobs internal-finish-run -v --config {cfgfile} {reqid}
            popd

            # Mark the script as complete.
            echo
            echo "(the "'"'"{reqid.kind}"'"'" job, {reqid} has finished)"
            #rm -f {pidfile}

            # Trigger the next job.
            pushd {_common.SYS_PATH_ENTRY}
            {sys.executable} -u -m jobs internal-run-next -v --config {cfgfile} --logfile {queue_log} &

            exit $exitcode
        '''[1:-1])

    def _get_ssh_agent(self) -> _utils.SSHAgentInfo:
        agent = self._cfg.worker.ssh.agent
        if not agent or not agent.check():
            agent = _utils.SSHAgentInfo.find_latest()
            if not agent:
                agent = _utils.SSHAgentInfo.from_env_vars()
        if agent:
            logger.debug(f'(using SSH agent at {agent.auth_sock})')
        else:
            logger.debug('(no SSH agent running)')
        return agent

    def check_ssh(
            self,
            *,
            onunknown: Optional[str] = None,
            agent: Optional[_utils.SSHAgentInfo] = None,
            fail: bool = True
    ) -> bool:
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

    def run(self, *, background: bool = False) -> int:
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

    def get_pid(self) -> Optional[int]:
        return self._pidfile.read()

    def kill(self) -> None:
        pid = self.get_pid()
        if pid:
            logger.info('# killing PID %s', pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.warning(f'job {self._reqid} no longer (PID: {pid})')
        # Kill the worker process, if running.
        agent = self._get_ssh_agent()
        text = self._worker.ssh.read(self._worker.fs.pidfile, agent=agent)
        if text and text.isdigit():
            self._worker.ssh.run_shell(f'kill {text}', agent=agent)

    def wait_until_started(
            self,
            timeout: Any = None,
            *,
            checkssh: bool = False
    ) -> Optional[int]:
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

    def attach(self, lines: Optional[Iterable[str]] = None) -> None:
        pid = self.wait_until_started()
        try:
            if pid:
                _utils.tail_file(self._fs.logfile, lines, follow=pid)
            elif lines:
                _utils.tail_file(self._fs.logfile, lines, follow=False)
        except KeyboardInterrupt:
            # XXX Prompt to cancel the job?
            return

    def wait_until_finished(
            self,
            pid: Optional[int] = None,
            *,
            timeout: Any = True
    ) -> None:
        if timeout is True:
            # Default to double the typical.
            timeout = self._kind.TYPICAL_DURATION_SECS or None
            if timeout:
                timeout *= 2
        elif not timeout or timeout < 0:
            timeout = None
        if timeout:
            end = time.time() + timeout
        else:
            end = None
        if not pid:
            try:
                pid = self.wait_until_started(timeout)
            except JobFinishedError:
                return
        if pid is not None:
            while _utils.is_proc_running(pid):
                if timeout and (end is not None and time.time() > end):
                    raise TimeoutError(f'timed out after {timeout} seconds')
                time.sleep(0.1)
        # Make sure it finished.
        status = self.get_status()
        if status not in Result.FINISHED:
            assert status not in Result.ACTIVE, (self, status)
            raise JobNeverStartedError(self.reqid)

    def cancel(self, *, ifstatus: Optional[str] = None) -> None:
        if ifstatus is not None:
            if self.get_status() not in (Result.STATUS.CREATED, ifstatus):
                return
        self.kill()
        # XXX Try to download the results directly?
        self.set_status('canceled')

    def close(self) -> None:
        result = self.load_result()
        result.close()
        result.save(self._fs.result.metadata, withextra=True)

    def upload_result(
            self,
            author: Optional[str] = None,
            *,
            clean: bool = True,
            push: bool = True
    ) -> None:
        res = self.load_result()
        # We upload directly.
        if self._store is not None:
            self._store.add(
                res.pyperf,
                branch='main',
                author=author,
                compressed=False,
                split=True,
                clean=clean,
                push=push,
            )
        else:
            raise ValueError("No store")

    def as_row(self) -> _utils.TableRow:  # XXX Move to JobSummary.
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
            if end is not None:
                elapsed = end - started
            else:
                # TODO: Handle this situation better
                elapsed = None
        date_options = (started, self.reqid.date)
        date = any(date_options) and date_options or None
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

    def render(self, fmt: Optional[str] = None) -> Iterable[str]:
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


    def _render_summary(self, fmt: Optional[str] = None) -> Iterable[str]:
        if not fmt:
            fmt = 'verbose'

        reqfs_fields = self._kind.REQFS_FIELDS
        resfs_fields = self._kind.RESFS_FIELDS

        fs = self._fs
        req = self.load_request()
        res = self.load_result()
        pid = _utils.PIDFile(fs.pidfile).read()

        staged = _current.get_staged_request(fs.jobs)
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
