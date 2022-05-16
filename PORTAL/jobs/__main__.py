import logging
import os
import os.path
import sys
import traceback

from . import (
    NoRunningJobError, JobNeverStartedError, RequestAlreadyStagedError,
    PortalConfig, Jobs, Worker, RequestID,
    select_jobs,
)
from .queue import (
    JobQueuePausedError, JobQueueNotPausedError, JobQueueEmptyError,
    JobNotQueuedError, JobAlreadyQueuedError,
)
from ._utils import LogSection, tail_file, render_file


PID = os.getpid()

logger = logging.getLogger(__name__)


##################################
# commands

def cmd_list(jobs, selections=None):
#    requests = (RequestID.parse(n) for n in os.listdir(jobs.fs.requests.root))
    alljobs = list(jobs.iter_all())
    total = len(alljobs)
    selected = list(select_jobs(alljobs, selections))
    print(f'{"request ID".center(48)} {"status".center(10)} {"created".center(19)}')
    print(f'{"-"*48} {"-"*10} {"-"*19}')
    for job in selected:
        reqid = job.reqid
        status = job.get_status(fail=False)
        print(f'{reqid!s:48} {status or "???":10} {reqid.date:%Y-%m-%d %H:%M:%S}')
        #for line in job.render(fmt='row'):
        #    print(line)
    logger.info('')
    if len(selected) == total:
        logger.info('(total: %s)', total)
    else:
        logger.info('(matched: %s)', len(selected))
        logger.info('(total:   %s)', total)


def cmd_show(jobs, reqid=None, fmt=None, *, lines=None):
    if reqid:
        job = jobs.get(reqid)
    else:
        job = jobs.get_current()
        if not job:
            # XXX Use the last finished?
            logger.error('no job currently running')
            sys.exit(1)

    for line in job.render(fmt=fmt):
        print(line)

    if lines:
        tail_file(job.fs.logfile, lines, follow=False)


def cmd_request_compile_bench(jobs, reqid, revision, *,
                              remote=None,
                              branch=None,
                              benchmarks=None,
                              optimize=False,
                              debug=False,
                              _fake=None,
                              ):
    if not reqid:
        raise NotImplementedError
    assert reqid.kind == 'compile-bench', reqid
    reqroot = jobs.fs.resolve_request(reqid).request.root
    logger.info('generating request files in %s...', reqroot)
    job = jobs.create(
        reqid,
        kind_kwargs=dict(
            revision=revision,
            remote=remote,
            branch=branch,
            benchmarks=benchmarks,
            optimize=optimize,
            debug=debug,
            _fake=_fake,
        ),
        reqfsattrs=['pyperformance_results', 'pyperformance_log'],
    )
    logger.info('...done (generating request files)')
    return job


def cmd_copy(jobs, reqid=None):
    raise NotImplementedError


def cmd_remove(jobs, reqid):
    raise NotImplementedError


def cmd_run(jobs, reqid, *, copy=False, force=False):
    if copy:
        raise NotImplementedError
    if force:
        raise NotImplementedError

    if not reqid:
        raise NotImplementedError

    if not jobs.queue.paused:
        cmd_queue_push(jobs, reqid)
    else:
        job = _cmd_run(jobs, reqid)
        job.check_ssh(onunknown='wait:3')


def _cmd_run(jobs, reqid):
    # Try staging it directly.
    try:
        job = jobs.activate(reqid)
    except RequestAlreadyStagedError as exc:
        # XXX Offer to clear CURRENT?
        logger.error('%s', exc)
        sys.exit(1)
    except Exception:
        logger.error('could not stage request')
        logger.info('')
        job = jobs.get(reqid)
        job.set_status('failed')
        raise  # re-raise
    else:
        job.run(background=True)
        return job


def cmd_attach(jobs, reqid=None, *, lines=None):
    if not reqid:
        job = jobs.get_current()
        if not job:
            logger.error('no current request to attach')
            sys.exit(1)
    else:
        job = jobs.get(reqid)
    job.wait_until_started(checkssh=True)
    job.check_ssh()
    try:
        job.attach(lines)
    except JobNeverStartedError:
        logger.warn('job not started')


def cmd_cancel(jobs, reqid=None, *, _status=None):
    if not reqid:
        try:
            job = current = jobs.cancel_current(ifstatus=_status)
        except NoRunningJobError:
            logger.error('no current request to cancel')
            sys.exit(1)
    else:
        current = jobs.get_current()
        if current and reqid == current.reqid:
            try:
                job = jobs.cancel_current(current.reqid, ifstatus=_status)
            except NoRunningJobError:
                logger.warn('job just finished')
        else:
            cmd_queue_remove(jobs, reqid)
            job = jobs.get(reqid)
            job.cancel(ifstatus=_status)

    logger.info('')
    logger.info('Results:')
    # XXX Show something better?
    for line in job.render(fmt='resfile'):
        logger.info(line)

    if current:
        jobs.ensure_next()


# internal
def cmd_finish_run(jobs, reqid):
    job = jobs.finish_successful(reqid)

    logger.info('')
    logger.info('Results:')
    # XXX Show something better?
    for line in job.render(fmt='resfile'):
        logger.info(line)


# internal
def cmd_run_next(jobs):
    logentry = LogSection.from_title('Running next queued job')
    print()
    for line in logentry.render():
        print(line)
    print()

    try:
        reqid = jobs.queue.pop()
    except JobQueuePausedError:
        logger.info('done (job queue is paused)')
    except JobQueueEmptyError:
        logger.info('done (job queue is empty)')
        return

    try:
        try:
            job = jobs.get(reqid)
            status = job.get_status()
        except Exception:
            logger.error('could not load results metadata')
            logger.warning('%s status could not be updated (to "failed")', reqid)
            logger.error('')
            traceback.print_exc()
            logger.info('')
            logger.info('trying next job...')
            cmd_run_next(jobs)
            return

        if not status:
            logger.warn('queued request (%s) not found', reqid)
            logger.info('trying next job...')
            cmd_run_next(jobs)
            return
        elif status is not Result.STATUS.PENDING:
            logger.warn('expected "pending" status for queued request %s, got %r', reqid, status)
            # XXX Give the option to force the status to "active"?
            logger.info('trying next job...')
            cmd_run_next(jobs)
            return

        # We're okay to run the job.
        logger.info('Running next job from queue (%s)', reqid)
        logger.info('')
        try:
            _cmd_run(jobs, reqid)
        except RequestAlreadyStagedError:
            if reqid == exc.curid:
                logger.warn('%s is already running', reqid)
                # XXX Check the pidfile?
            else:
                logger.warn('another job is already running, adding %s back to the queue', reqid)
                jobs.queue.unpop(reqid)
    except KeyboardInterrupt:
        cmd_cancel(jobs, reqid, _status=Result.STATUS.PENDING)
        raise  # re-raise


def cmd_queue_info(jobs, *, withlog=True):
    _queue = jobs.queue.snapshot
    queued = _queue.jobs
    paused = _queue.paused
    pid, pid_running = _queue.locked
    if withlog:
        log = list(_queue.read_log())

    print('Job Queue:')
    print(f'  size:     {len(queued)}')
    #print(f'  max size: {maxsize}')
    print(f'  paused:   {paused}')
    if isinstance(pid, str):
        assert pid_running is None, repr(pid_running)
        print(f'  lock:     bad PID file (content: {pid!r})')
    elif pid:
        running = '' if pid_running else ' (not running)'
        print(f'  lock:     held by process {pid}{running}')
    else:
        print('  lock:     (not locked)')
    print()
    print('Files:')
    print(f'  data:      {render_file(_queue.datafile)}')
    print(f'  lock:      {render_file(_queue.lockfile)}')
    print(f'  log:       {render_file(_queue.logfile)}')
    print()
    print('Top 5:')
    if queued:
        for i in range(min(5, len(queued))):
            print(f'  {i+1} {queued[i]}')
    else:
        print('  (queue is empty)')
    if withlog:
        print()
        print(f'Log size:    {len(log)}')
        print('Last log entry:')
        if log:
            print('-'*30)
            print()
            for line in log[-1].render():
                print(line)
            print()
            print('-'*30)
        else:
            print('  (log is empty)')


def cmd_queue_list(jobs):
    if jobs.queue.paused:
        logger.warn('job queue is paused')

    if not jobs.queue:
        print('no jobs queued')
        return

    print('Queued jobs:')
    for i, reqid in enumerate(jobs.queue, 1):
        print(f'{i:>3} {reqid}')
    print()
    print(f'(total: {i})')


def cmd_queue_pause(jobs):
    try:
       jobs.queue.pause()
    except JobQueuePausedError:
        logger.warn('job queue was already paused')
    else:
        logger.info('job queue paused')


def cmd_queue_unpause(jobs):
    try:
       jobs.queue.unpause()
    except JobQueueNotPausedError:
        logger.warn('job queue was not paused')
    else:
        logger.info('job queue unpaused')
        jobs.ensure_next()


def cmd_queue_push(jobs, reqid):
    reqid = RequestID.from_raw(reqid)
    logger.info(f'Adding job {reqid} to the queue')
    job = jobs.get(reqid)

    status = job.get_status()
    if not status:
        logger.error('request %s not found', reqid)
        sys.exit(1)
    elif status is not Result.STATUS.CREATED:
        logger.error('request %s has already been used', reqid)
        sys.exit(1)

    if jobs.queue.paused:
        logger.warn('job queue is paused')

    try:
        pos = jobs.queue.push(reqid)
    except JobAlreadyQueuedError:
        for pos, queued in enumerate(jobs.queue, 1):
            if queued == reqid:
                logger.warn('%s was already queued', reqid)
                break
        else:
            raise NotImplementedError

    job.set_status('pending')

    logger.info('%s added to the job queue at position %s', reqid, pos)

    jobs.ensure_next()


def cmd_queue_pop(jobs):
    logger.info(f'Popping the next job from the queue...')
    try:
        reqid = jobs.queue.pop()
    except JobQueuePausedError:
        logger.warn('job queue is paused')
        return
    except JobQueueEmptyError:
        logger.error('job queue is empty')
        sys.exit(1)
    job = jobs.get(reqid)

    status = job.get_status()
    if not status:
        logger.warn('queued request (%s) not found', reqid)
    elif status is not Result.STATUS.PENDING:
        logger.warn(f'expected "pending" status for queued request %s, got %r', reqid, status)
        # XXX Give the option to force the status to "active"?
    else:
        # XXX Set the status to "active"?
        pass

    print(reqid)


def cmd_queue_move(jobs, reqid, position, relative=None):
    position = int(position)
    if position <= 0:
        raise ValueError(f'expected positive position, got {position}')
    if relative and relative not in '+-':
        raise ValueError(f'expected relative of + or -, got {relative}')

    reqid = RequestID.from_raw(reqid)
    if relative:
        logger.info('Moving job %s %s%s in the queue...', reqid, relative, position)
    else:
        logger.info('Moving job %s to position %s in the queue...', reqid, position)
    job = jobs.get(reqid)

    if jobs.queue.paused:
        logger.warn('job queue is paused')

    status = job.get_status()
    if not status:
        logger.error('request %s not found', reqid)
        sys.exit(1)
    elif status is not Result.STATUS.PENDING:
        logger.warn('request %s has been updated since queued', reqid)

    pos = jobs.queue.move(reqid, position, relative)
    logger.info('...moved to position %s', pos)


def cmd_queue_remove(jobs, reqid):
    reqid = RequestID.from_raw(reqid)
    logger.info('Removing job %s from the queue...', reqid)
    job = jobs.get(reqid)

    if jobs.queue.paused:
        logger.warn('job queue is paused')

    status = job.get_status()
    if not status:
        logger.warn('request %s not found', reqid)
    elif status is not Result.STATUS.PENDING:
        logger.warn('request %s has been updated since queued', reqid)

    try:
        jobs.queue.remove(reqid)
    except JobNotQueuedError:
        logger.warn('%s was not queued', reqid)

    if status is Result.STATUS.PENDING:
        job.set_status('created')

    logger.info('...done!')


def cmd_config_show(jobs):
    for line in jobs.cfg.render():
        print(line)


def cmd_bench_host_clean(jobs):
    raise NotImplementedError


COMMANDS = {
    # job management
    'list': cmd_list,
    'show': cmd_show,
    'copy': cmd_copy,
    'remove': cmd_remove,
    'run': cmd_run,
    'attach': cmd_attach,
    'cancel': cmd_cancel,
    # specific jobs
    'request-compile-bench': cmd_request_compile_bench,
    # queue management
    'queue-info': cmd_queue_info,
    'queue-pause': cmd_queue_pause,
    'queue-unpause': cmd_queue_unpause,
    'queue-list': cmd_queue_list,
    'queue-push': cmd_queue_push,
    'queue-pop': cmd_queue_pop,
    'queue-move': cmd_queue_move,
    'queue-remove': cmd_queue_remove,
    # other public commands
    'config-show': cmd_config_show,
    'bench-host-clean': cmd_bench_host_clean,
    # internal-only
    'internal-finish-run': cmd_finish_run,
    'internal-run-next': cmd_run_next,
}


##################################
# the script

VERBOSITY = 3


def configure_root_logger(verbosity=VERBOSITY, logfile=None, *,
                          maxlevel=logging.CRITICAL,
                          ):
    logger = logging.getLogger()

    level = max(1,  # 0 disables it, so we use the next lowest.
                min(maxlevel,
                    maxlevel - verbosity * 10))
    logger.setLevel(level)
    #logger.propagate = False

    assert not logger.handlers, logger.handlers
    if logfile:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    #formatter = logging.Formatter()
    class Formatter(logging.Formatter):
        def format(self, record):
            text = super().format(record)
            if record.levelname not in ('DEBUG', 'INFO'):
                text = f'{record.levelname}: {text}'
            return text
    formatter = Formatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not os.isatty(sys.stdout.fileno()):
        global print
        print = (lambda m='': logger.info(m))


def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse

    add_hidden = ('-h' not in argv and '--help' not in argv)

    ##########
    # First, pull out the common args.

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', dest='cfgfile')
    common.add_argument('-v', '--verbose', action='count', default=0)
    common.add_argument('-q', '--quiet', action='count', default=0)
    common.add_argument('--logfile')
    args, argv = common.parse_known_args(argv)
    cfgfile = args.cfgfile
    verbosity = max(0, VERBOSITY + args.verbose - args.quiet)
    logfile = args.logfile

    ##########
    # Create the top-level parser.

    parser = argparse.ArgumentParser(
        prog=prog,
        parents=[common],
    )
    subs = parser.add_subparsers(dest='cmd', metavar='CMD')

    ##########
    # Add the subcommands for managing jobs.

    def add_cmd(name, subs=subs, *, parents=(), **kwargs):
        return subs.add_parser(name, parents=[common, *parents], **kwargs)

    sub = add_cmd('list', help='Print a table of all known jobs')
    sub.add_argument('selections', nargs='*')

    sub = add_cmd('show', help='Print a summary of the given (or current) job')
    sub.add_argument('-n', '--lines', type=int, default=0,
                     help='Show the last n lines of the job\'s output')
    sub.add_argument('reqid', nargs='?')

    sub = add_cmd('request', aliases=['add'], help='Create a new job request')
    jobs = sub.add_subparsers(dest='job')
    # Subcommands for different jobs are added below.

#    sub = add_cmd('copy', help='Create a new copy of an existing job request')
#    sub.add_argument('reqid', nargs='?')

#    sub = add_cmd('remove', help='Delete a job request')
#    sub.add_argument('reqid', nargs='+', help='the requests to delete (globs okay)')

    sub = add_cmd('run', help='Run a previously created job request')
    sub.add_argument('--attach', dest='after',
                     action='store_const', const=('attach',))
    sub.add_argument('--no-attach', dest='after',
                     action='store_const', const=())
#    sub.add_argument('--copy', action='store_true',
#                     help='Run a new copy of the given job request')
#    sub.add_argument('--force', action='store_true',
#                     help='Run the job even if another is already running')
    sub.add_argument('reqid')

    sub = add_cmd('attach', help='Tail the job log file')
    sub.add_argument('-n', '--lines', type=int, default=0,
                     help='Show the last n lines of the job\'s output')
    sub.add_argument('reqid', nargs='?')

    sub = add_cmd('cancel', help='Stop the current job (or prevent a pending one)')
    sub.add_argument('reqid', nargs='?')

    # XXX Also add export and import?

    sub = add_cmd('queue', help='Manage the job queue')
    queue = sub.add_subparsers(dest='action')
    # Subcommands for different actions are added below.

    ##########
    # Add the "add" subcommands for the different jobs.

    _common = argparse.ArgumentParser(add_help=False)
    _common.add_argument('--run', dest='after',
                         action='store_const', const=('run', 'attach'))
    _common.add_argument('--run-attached', dest='after',
                         action='store_const', const=('run', 'attach'))
    _common.add_argument('--run-detached', dest='after',
                         action='store_const', const=('run',))
    _common.add_argument('--no-run', dest='after',
                         action='store_const', const=(),
                         help='(the default)')
    add_job = (lambda job, **kw: add_cmd(job, jobs, parents=[_common], **kw))

    # This is the default (and the only one, for now).
    sub = add_job('compile-bench',
                  help='Request a compile-and-run-benchmarks job')
    sub.add_argument('--optimize', dest='optimize',
                     action='store_const', const=True,
                     help='(the default)')
    sub.add_argument('--no-optimize', dest='optimize', action='store_false')
    sub.add_argument('--debug', action='store_true')
    sub.add_argument('--benchmarks')
    sub.add_argument('--branch')
    sub.add_argument('--remote', required=True)
    sub.add_argument('revision')
    if add_hidden:
        # Use these flags to skip actually running pyperformance.
        sub.add_argument('--fake-success', dest='exitcode',
                         action='store_const', const=0)
        sub.add_argument('--fake-failure', dest='exitcode',
                         action='store_const', const=1)
        sub.add_argument('--fake-delay', dest='fakedelay')

    ##########
    # Add the "queue" subcomamnds.

    sub = add_cmd('info', queue, help='Print a summary of the state of the jobs queue')
    sub.add_argument('--without-log', dest='withlog', action='store_false')
    sub.add_argument('--with-log', dest='withlog',
                     action='store_const', const=True)

    sub = add_cmd('pause', queue, help='Do not let queued jobs run')

    sub = add_cmd('unpause', queue, help='Let queued jobs run')

    sub = add_cmd('list', queue, help='List the queued jobs')

    sub = add_cmd('push', queue, help='Add a job to the queue')
    sub.add_argument('reqid')

    sub = add_cmd('pop', queue, help='Get the next job from the queue')

    sub = add_cmd('move', queue, help='Move a job up or down in the queue')
    sub.add_argument('reqid')
    sub.add_argument('position')

    sub = add_cmd('remove', queue, help='Remove a job from the queue')
    sub.add_argument('reqid')

    ##########
    # Add other public commands.

    sub = add_cmd('config', help='show the config')

#    sub = add_cmd('bench-host', help='manage the host where benchmarks run')
#    benchhost = sub.add_subparsers(dest='action')
#
#    sub = add_cmd('clean', benchhost, help='clean up old files')

    ##########
    # Add internal commands.

    if add_hidden:
        sub = add_cmd('internal-finish-run')
        sub.add_argument('reqid')

        sub = add_cmd('internal-run-next')

    ##########
    # Finally, parse the args.

    args = parser.parse_args(argv)
    ns = vars(args)

    # Drop args we already handled earlier.
    ns.pop('cfgfile')
    ns.pop('verbose')
    ns.pop('quiet')
    ns.pop('logfile')

    # Process commands and command-specific args.
    cmd = ns.pop('cmd')
    if cmd in ('add', 'request'):
        job = ns.pop('job')
        cmd = f'request-{job}'
        if job == 'compile-bench':
            # Process hidden args.
            fake = (ns.pop('exitcode'), ns.pop('fakedelay'))
            if any(v is not None for v in fake):
                args._fake = fake
    elif cmd == 'config':
        cmd = 'config-show'
    elif cmd == 'queue':
        action = ns.pop('action')
        cmd = f'queue-{action}'
        if action == 'move':
            pos = args.position
            if pos == '+':
                pos = '1'
                relative = '+'
            elif pos == '-':
                pos = '1'
                relative = '-'
            elif pos.startswith('+'):
                pos = pos[1:]
                relative = '+'
            elif pos.startswith('-'):
                pos = pos[1:]
                relative = '-'
            else:
                # an absolute move
                relative = None
            if not pos.isdigit():
                parser.error('position must be positive int')
            pos = int(pos)
            if pos == 0:
                parser.error('position must be positive int')
            args.position = pos
            args.relative = relative
    elif cmd == 'bench-host':
        action = ns.pop('action')
        cmd = f'bench-host-{action}'

    return cmd, ns, cfgfile, verbosity, logfile


def main(cmd, cmd_kwargs, cfgfile=None):
    try:
        run_cmd = COMMANDS[cmd]
    except KeyError:
        logger.error('unsupported cmd %r', cmd)
        sys.exit(1)

    after = []
    for _cmd in cmd_kwargs.pop('after', None) or ():
        try:
            run_after = COMMANDS[_cmd]
        except KeyError:
            logger.error('unsupported "after" cmd %r', _cmd)
            sys.exit(1)
        after.append((_cmd, run_after))

    logger.debug('')
    logger.debug('# PID: %s', PID)

    # Load the config.
    if not cfgfile:
        cfgfile = PortalConfig.find_config()
    logger.debug('')
    logger.debug('# loading config from %s', cfgfile)
    cfg = PortalConfig.load(cfgfile)

    jobs = Jobs(cfg)

    if cmd != 'queue-info' and not cmd.startswith('internal-'):
        # In some cases the mechanism to run jobs from the queue may
        # get interrupted, so we re-start it manually here if necessary.
        jobs.ensure_next()

    # Resolve the request ID, if any.
    if 'reqid' in cmd_kwargs:
        if cmd_kwargs['reqid']:
            cmd_kwargs['reqid'] = RequestID.parse(cmd_kwargs['reqid'])
        else:
            cmd_kwargs['reqid'] = None
    elif cmd.startswith('request-'):
        cmd_kwargs['reqid'] = RequestID.generate(cfg, kind=cmd[8:])
    reqid = cmd_kwargs.get('reqid')

    # Run the command.
    logger.info('')
    logger.info('#'*40)
    if reqid:
        logger.info('# Running %r command for request %s', cmd, reqid)
    else:
        logger.info('# Running %r command', cmd)
    logger.info('')
    job = run_cmd(jobs, **cmd_kwargs)

    if cmd.startswith('request-'):
        _fmt = 'reqfile' if after else 'summary'
        logger.info('')
        # XXX Show something better?
        for line in job.render(fmt=_fmt):
            logger.info(line)

    # Run "after" commands, if any
    for cmd, run_cmd in after:
        logger.info('')
        logger.info('#'*40)
        if reqid:
            logger.info('# Running %r command for request %s', cmd, reqid)
        else:
            logger.info('# Running %r command', cmd)
        logger.info('')
        # XXX Add --lines='-1' for attach.
        run_cmd(jobs, reqid=reqid)


if __name__ == '__main__':
    cmd, cmd_kwargs, cfgfile, verbosity, logfile = parse_args()
    configure_root_logger(verbosity, logfile)
    main(cmd, cmd_kwargs, cfgfile)
