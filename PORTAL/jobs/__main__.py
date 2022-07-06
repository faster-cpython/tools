import argparse
import logging
import os
import os.path
import sys
import traceback

from . import (
    NoRunningJobError, JobNeverStartedError,
    JobAlreadyFinishedError, JobFinishedError,
    RequestAlreadyStagedError,
    JobsConfig, Jobs, Worker, RequestID, Result,
    sort_jobs, select_jobs,
)
from .queue import (
    JobQueuePausedError, JobQueueNotPausedError, JobQueueEmptyError,
    JobNotQueuedError, JobAlreadyQueuedError,
)
#from ._pyperformance import PyperfTable
from ._utils import (
    LogSection, tail_file, render_file, get_bool_env_var, TableSpec,
)


PID = os.getpid()

logger = logging.getLogger(__name__)


##################################
# commands

def cmd_list(jobs, selections=None, columns=None):
#    requests = (RequestID.parse(n) for n in os.listdir(jobs.fs.requests.root))
    alljobs = list(jobs.iter_all())
    total = len(alljobs)
    alljobs = sort_jobs(alljobs)
    selected = list(select_jobs(alljobs, selections))

    colspecs = [
        ('reqid', 'request ID', 45, None),
        ('status', None, 10, None),
        ('elapsed', None, 10, '>'),
        ('date', 'started / (created)', 21, None),
        ('created', None, 19, None),
        ('started', None, 19, None),
        ('finished', None, 19, None),
        ('ref', None, 40, None),
        ('fullref', 'full ref', 35, None),
        ('remote', None, 20, None),
        ('branch', None, 25, None),
        ('tag', None, 10, None),
        ('commit', None, 40, None),
    ]
    if not columns:
        columns = 'reqid,status,elapsed,date,fullref'
    table = TableSpec.from_columns(colspecs, columns)

    rows = (j.as_row() for j in selected)
    periodic = [table.div, table.header, table.div]
    rows = table.render_rows(rows, periodic, len(selected))

    logger.info('(all times in UTC)')
    logger.info('')
    logger.info(table.div)
    print(table.header)
    print(table.div)
    for line in rows:
        print(line)
    logger.info(table.div)
    for line in rows.render_count(total):
        logger.info(line)
    logger.info('')

    current = jobs.get_current()
    current = current.reqid if current else 'none'
    logger.info(f'Currently running: {current}')
    logger.info('')

    queue = jobs.queue.snapshot
    logger.info(f'Queued jobs: {len(queue)}')
    if queue:
        logger.info('-' * 40)
        for i, reqid in enumerate(queue, 1):
            logger.info(f'  {i:>3} {reqid}')
    logger.info('')


def cmd_show(jobs, reqid=None, fmt=None, *, lines=None):
    job = jobs.get(reqid)
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
                              optimize=True,
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
        pushfsattrs=['pyperformance_manifest', 'pyperformance_config'],
        pullfsattrs=['pyperformance_results', 'pyperformance_log'],
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
    try:
        try:
            job, pid = jobs.wait_until_job_started(reqid)
        except NoRunningJobError:
            logger.error('no current request to attach')
            sys.exit(1)
        except JobAlreadyFinishedError as exc:
            logger.warning(f'job {exc.reqid} was already done')
        job.check_ssh()
        job.attach(lines)
    except JobNeverStartedError:
        # XXX Optionally wait anyway?
        logger.warn('job not started')
    except JobFinishedError:
        # It already finished.
        pass


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


def cmd_wait(jobs, reqid=None):
    try:
        try:
            job, pid = jobs.wait_until_job_started(reqid)
        except NoRunningJobError:
            logger.error('no current request to wait for')
            sys.exit(1)
        except JobAlreadyFinishedError as exc:
            logger.warning(f'job {exc.reqid} was already done')
        else:
            assert pid, job and job.reqid
            job.wait_until_finished(pid)
    except JobNeverStartedError as exc:
        # XXX Optionally wait anyway?
        logger.warn('job not started')
    except JobFinishedError:
        # It already finished.
        pass


def cmd_upload(jobs, reqid, *, author=None, clean=True, push=True):
    job = jobs.get(reqid)
    job.upload_result(author, clean=clean, push=push)


def cmd_compare(jobs, res1, others, *, meanonly=False, pyston=False):
    suites = ['pyston'] if pyston else ['pyperformance']
    matched = list(jobs.match_results(res1, suites=suites))
    if not matched:
        logger.error(f'no results matched {res1!r}')
        sys.exit(1)
    res1, = matched
    for _ in range(len(others)):
        spec = others.pop(0)
        matched = list(jobs.match_results(spec, suites=suites))
        if not matched:
            logger.error(f'no results matched {spec!r}')
            sys.exit(1)
        others.extend(matched)
    #others = [*jobs.match_results(r) for r in others]
    compared = res1.compare(others)
    table = compared.table
    fmt = 'meanonly' if meanonly else 'raw'
    for line in table.render(fmt):
        print(line)


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
            # XXX Give the option to force the status to "activated"?
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
    queue = jobs.queue.snapshot
    if queue.paused:
        logger.warn('job queue is paused')

    if not queue:
        print('no jobs queued')
        return

    print('Queued jobs:')
    for i, reqid in enumerate(queue, 1):
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
        # XXX Give the option to force the status to "activated"?
    else:
        # XXX Set the status to "activated"?
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
    'wait': cmd_wait,
    'upload': cmd_upload,
    'compare': cmd_compare,
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


def _add_request_cli(add_cmd, add_hidden=True):
    compile_bench = argparse.ArgumentParser(add_help=False)
    compile_bench.add_argument('--optimize', dest='optimize',
                               action='store_const', const=True, default=True,
                               help='do PGO and LTO (default)')
    compile_bench.add_argument('--no-optimize', dest='optimize', action='store_false')
    compile_bench.add_argument('--debug', action='store_true')
    compile_bench.add_argument('--benchmarks')
    compile_bench.add_argument('--remote', help='(default: origin)')
    compile_bench.add_argument('--branch', help='(default: main or None)')
    compile_bench.add_argument('revision',
                               help='CPython tag/commit/branch to benchmark (default: latest)')

    # Add the commands.

    sub = add_cmd('run-bench', parents=[compile_bench],
                  help='Create a benchmarking request and run it')
    sub.add_argument('--attached', dest='after',
                     action='store_const', const=('run', 'attach'),
                     help='attach after the job has been queued (default)')
    sub.add_argument('--detached', dest='after',
                     action='store_const', const=('run',),
                     help='do not attach')
    sub.set_defaults(
        job='compile-bench',
        # Args set for the "request" command:
        uploadargs=[],
        exitcode=None,
        fakedelay=None,
    )

    if add_hidden:
        sub = add_cmd('add', aliases=['request'], help='Create a new job request')
        jobs = sub.add_subparsers(
            dest='job',
            metavar='JOB',
            title='subcommands',
            #required=False,
            #default='compile-bench',
        )

        _common = argparse.ArgumentParser(add_help=False)
        if add_hidden:
            _common.add_argument('--user', help='use the given user')
        _common.add_argument('--run', dest='after',
                             action='store_const', const=('run', 'attach'),
                             help='(alias for --run-attached)')
        _common.add_argument('--run-attached', dest='after',
                             action='store_const', const=('run', 'attach'),
                             help='queue and attach once created (default)')
        _common.add_argument('--run-detached', dest='after',
                             action='store_const', const=('run',),
                             help='queue once created (but do not attach)')
        _common.add_argument('--no-run', dest='after',
                             action='store_const', const=(),
                             help='only create the job')
        _common.add_argument('--wait', dest='after',
                             action='store_const', const=('run', 'wait'),
                             help='wait for the job to finish')
        _common.add_argument('--upload', dest='after',
                             action='store_const', const=('run', 'wait', 'upload'),
                             help='upload after the job finishes')
        if add_hidden:
            _common.add_argument('--upload-arg', dest='uploadargs',
                                 action='append', default=[])
        def add_job(job, p=(), **kw):
            return add_cmd(job, jobs, parents=[_common, *p], **kw)

        # This is the default (and the only one, for now).
        sub = add_job('compile-bench', [compile_bench],
                      help='Request a compile-and-run-benchmarks job')
        if add_hidden:
            # Use these flags to skip actually running pyperformance.
            sub.add_argument('--fake-success', dest='exitcode',
                             action='store_const', const=0)
            sub.add_argument('--fake-failure', dest='exitcode',
                             action='store_const', const=1)
            sub.add_argument('--fake-delay', dest='fakedelay')

    def handle_args(args, parser):
        if args.cmd in ('add', 'request', 'run-bench'):
            ns = vars(args)
            job = ns.pop('job')
            args.cmd = f'request-{job}'
            if job == 'compile-bench':
                # Process hidden args.
                fake = (ns.pop('exitcode'), ns.pop('fakedelay'))
                if any(v is not None for v in fake):
                    args._fake = fake
            else:
                raise NotImplementedError((job, args))
            if args.after is None:
                # Use --run-attached as the default.
                args.after = ('run', 'attach')
            elif type(args.after) is not tuple:
                raise NotImplementedError(args.after)
            # Handle --upload-arg.
            uploadargs = ns.pop('uploadargs')
            if uploadargs:
                if 'upload' not in args.after:
                    parser.error('--upload-arg requires --upload')
                args.upload_kwargs = {}
                for arg in uploadargs:
                    if arg == '<no-push>':
                        args.upload_kwargs['push'] = False
                    #elif arg in ('', '-', '<>', '<default>'):
                    #    args.upload_kwargs['repo'] = 'gh:faster-cpython/ideas'
                    else:
                        raise NotImplementedError(arg)
            elif 'upload' in args.after:
                args.upload_kwargs = {}
    return handle_args


def _add_queue_cli(add_cmd, add_hidden=True):
    if not add_hidden:
        return (lambda *a, **k: None)

    sub = add_cmd('queue', help='Manage the job queue')
    queue = sub.add_subparsers(
        dest='queue_cmd',
        metavar='CMD',
        title='subcommands',
        #required=False,
        #default='list',
    )

    sub = add_cmd('info', queue, help='Print a summary of the state of the jobs queue')
    sub.add_argument('--without-log', dest='withlog', action='store_false',
                     help='do not show any of the job queue log file')
    sub.add_argument('--with-log', dest='withlog',
                     action='store_const', const=True,
                     help='also show last 10 lines of the job queue log file')

    sub = add_cmd('pause', queue, help='Do not let queued jobs run')

    sub = add_cmd('unpause', queue, help='Let queued jobs run')

    sub = add_cmd('list', queue, help='List the queued jobs')

    sub = add_cmd('push', queue, help='Add a job to the back of the queue')
    sub.add_argument('reqid')

    sub = add_cmd('pop', queue, help='Get the next job from the front of the queue')

    sub = add_cmd('move', queue, help='Move a job up or down in the queue')
    sub.add_argument('reqid')
    sub.add_argument('position', help='the new index for the job (1-based or relative)')

    sub = add_cmd('remove', queue, help='Remove a job from the queue')
    sub.add_argument('reqid')

    def handle_args(args, parser):
        if args.cmd != 'queue':
            return
        ns = vars(args)
        action = ns.pop('queue_cmd')
        args.cmd = f'queue-{action}'
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
    return handle_args


def _add_config_cli(add_cmd, add_hidden=True):
    if not add_hidden:
        return (lambda *a, **k: None)
    sub = add_cmd('config', help='Show the config')

    def handle_args(args, parser):
        if args.cmd != 'config':
            return
        args.cmd = 'config-show'
    return handle_args


def _add_bench_host_cli(add_cmd, add_hidden=True):
    if not add_hidden:
        return (lambda *a, **k: None)
    sub = add_cmd('bench-host', help='Manage the host where benchmarks run')
    benchhost = sub.add_subparsers(dest='action')
    raise NotImplementedError

    def handle_args(args, parser):
        if args.cmd != 'bench-host':
            return
        action = ns.pop('action')
        args.cmd = f'bench-host-{action}'
    return handle_args


def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):

    ##########
    # Resolve dev mode.

    dev = argparse.ArgumentParser(add_help=False)
    dev.add_argument('--tool-devel', dest='devmode', action='store_true')
    args, argv = dev.parse_known_args(argv)
    devmode = args.devmode
    #devmode = get_bool_env_var('BENCH_JOBS_DEV_MODE')

    add_hidden = devmode or ('-h' not in argv and '--help' not in argv)

    ##########
    # Pull out the common args.

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-v', '--verbose', action='count', default=0)
    common.add_argument('-q', '--quiet', action='count', default=0)
    if add_hidden:
        common.add_argument('--config', dest='cfgfile', metavar='FILE',
                            help='(default: ~benchmarking/BENCH/portal.json))')
        common.add_argument('--logfile', metavar='FILE')
        args, argv = common.parse_known_args(argv)
        cfgfile = args.cfgfile
        logfile = args.logfile
    else:
        args, argv = common.parse_known_args(argv)
        cfgile = None
        logfile = None
    verbosity = max(0, VERBOSITY + args.verbose - args.quiet)

    ##########
    # Create the top-level parser.

    parser = argparse.ArgumentParser(
        prog=prog,
        parents=[common],
    )
    subs = parser.add_subparsers(
        dest='cmd',
        metavar='CMD',
        title='subcommands',
    )

    ##########
    # Add the subcommands for managing jobs.

    def add_cmd(name, subs=subs, *, parents=(), **kwargs):
        return subs.add_parser(name, parents=[common, *parents], **kwargs)

    sub = add_cmd('list', help='Print a table of all known jobs')
    sub.add_argument('--columns', help='a CSV of column names to show')
    sub.add_argument('selections', nargs='*', metavar='SELECTION',
                     help='a specifier for a subset (e.g. -10 for the last 10)')

    sub = add_cmd('show', help='Print a summary of the given (or current) job')
    sub.add_argument('-n', '--lines', type=int, default=0, metavar='N',
                     help='show the last N lines of the job\'s output (default: 0)')
    sub.add_argument('reqid', nargs='?',
                     help='(default: the currently running job, if any)')

    handle_request_args = _add_request_cli(add_cmd, add_hidden)

    #sub = add_cmd('copy', help='Create a new copy of an existing job request')
    #sub.add_argument('reqid', nargs='?')

    #sub = add_cmd('remove', help='Delete a job request')
    #sub.add_argument('reqid', nargs='+', help='the requests to delete (globs okay)')

    if add_hidden:
        sub = add_cmd('run', help='Run a previously created job request')
        sub.add_argument('--attach', dest='after',
                         action='store_const', const=('attach',))
        sub.add_argument('--no-attach', dest='after',
                         action='store_const', const=())
        #sub.add_argument('--copy', action='store_true',
        #                 help='Run a new copy of the given job request')
        #sub.add_argument('--force', action='store_true',
        #                 help='Run the job even if another is already running')
        sub.add_argument('reqid')

    sub = add_cmd('attach', help='Tail a job\'s log file')
    sub.add_argument('-n', '--lines', type=int, default=0, metavar='N',
                     help='show the last N lines of the job\'s output (default: 0)')
    sub.add_argument('reqid', nargs='?',
                     help='(default: the currently running job, if any)')

    sub = add_cmd('cancel', help='Stop the current job (or prevent a pending one)')
    sub.add_argument('reqid', nargs='?',
                     help='(default: the currently running job, if any)')

    if add_hidden:
        sub = add_cmd('wait', help='wait until the given (or current) job finishes')
        # XXX Add a --timeout arg?
        sub.add_argument('reqid', nargs='?',
                         help='(default: the currently running job, if any)')

    sub = add_cmd('upload', help='Upload benchmark results to the public data store')
    if add_hidden:
        sub.add_argument('--author')
        sub.add_argument('--no-clean', dest='clean', action='store_false')
        sub.add_argument('--clean', dest='clean', action='store_const', const=True)
        sub.add_argument('--no-push', dest='push', action='store_false')
        sub.add_argument('--push', dest='push', action='store_const', const=True)
    sub.add_argument('reqid')

    sub = add_cmd('compare', help='Compare two or more results')
    #sub.add_argument('--fmt', choices=PyperfTable.FORMATS)
    sub.add_argument('--mean-only', dest='meanonly', action='store_true')
    sub.add_argument('--pyston', action='store_true')
    sub.add_argument('res1')
    sub.add_argument('others', nargs='+')

    # XXX Also add export and import?

    handle_queue_args = _add_queue_cli(add_cmd, add_hidden)

    ##########
    # Add other public commands.

    handle_config_args = _add_config_cli(add_cmd, add_hidden)

    #handle_bench_host_args = _add_bench_host_cli(add_cmd, add_hidden)

    #if add_hidden:
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

    # Deal with args we already handled earlier.
    if add_hidden:
        ns.pop('cfgfile')
        ns.pop('logfile')
    ns.pop('verbose')
    ns.pop('quiet')

    # Process commands and command-specific args.
    handle_request_args(args, parser)
    handle_config_args(args, parser)
    handle_queue_args(args, parser)
    cmd = ns.pop('cmd')

    user = ns.pop('user', None)

    return cmd, ns, cfgfile, user, verbosity, logfile, devmode


def main(cmd, cmd_kwargs, cfgfile=None, user=None, devmode=False):
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
        _cmd_kwargs = cmd_kwargs.pop(f'{_cmd}_kwargs', None)
        after.append((_cmd, run_after, _cmd_kwargs))

    logger.debug('')
    logger.debug('# PID: %s', PID)

    # Load the config.
    if not cfgfile:
        cfgfile = JobsConfig.find_config()
    logger.debug('')
    logger.debug('# loading config from %s', cfgfile)
    cfg = JobsConfig.load(cfgfile)

    jobs = Jobs(cfg, devmode=devmode)

    if cmd != 'queue-info' and not cmd.startswith('internal-'):
        # In some cases the mechanism to run jobs from the queue may
        # get interrupted, so we re-start it manually here if necessary.
        jobs.ensure_next()

    # Resolve the request ID, if any.
    if 'reqid' in cmd_kwargs:
        reqid = cmd_kwargs['reqid'] or None
        if reqid:
            parsed = RequestID.parse(reqid)
            if parsed is None:
                logger.error(f'expected a valid reqid, got {reqid!r}')
                sys.exit(1)
            reqid = parsed
        cmd_kwargs['reqid'] = reqid
    elif cmd.startswith('request-'):
        reqid = RequestID.generate(cfg, user, kind=cmd[8:])
        cmd_kwargs['reqid'] = reqid
    else:
        reqid = None

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
    for cmd, run_cmd, cmd_kwargs in after:
        logger.info('')
        logger.info('#'*40)
        if reqid:
            logger.info('# Running %r command for request %s', cmd, reqid)
        else:
            logger.info('# Running %r command', cmd)
        logger.info('')
        # XXX Add --lines='-1' for attach.
        run_cmd(jobs, reqid=reqid, **(cmd_kwargs or {}))


if __name__ == '__main__':
    cmd, cmd_kwargs, cfgfile, user, verbosity, logfile, devmode = parse_args()
    configure_root_logger(verbosity, logfile)
    main(cmd, cmd_kwargs, cfgfile, user, devmode)
