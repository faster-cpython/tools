import datetime
from decimal import Decimal
import logging
import shlex
import shutil
import subprocess
import sys

import fc_utils.scriptutil
_utils = fc_utils
del fc_utils


# The order here is how they are rendered.
KINDS = ['real', 'user', 'sys']

logger = logging.getLogger(__name__)


def parse_duration(duration):
    seconds = Decimal(0)
    if 'm' in duration:
        minutes, duration = duration.split('m')
        assert minutes.isdigit()
        seconds += int(minutes) * 60
    if 's' in duration:
        secs, duration = duration.split('s')
        seconds += Decimal(secs)
        assert minutes.isdigit()
    assert not duration
    return seconds


def render_duration(seconds):
    minutes = int(seconds) // 60
    assert minutes < 60
    seconds -= minutes * 60
    return f'{minutes}m{seconds:.3f}'


def get_command_info(argv):
    executable = argv[0]
    if _utils.is_python(executable):
        kind = 'python'
        data =_utils.get_python_info(executable, full=True)
    else:
        kind = None
        data = None
    return {
        'kind': kind,
        'executable': executable,
        'argv': argv,
        'data': data,
    }


##################################
# low-level

def _get_cmd_runner(time_argv, cmd_argv, _bash=shutil.which('bash')):
    argv = ['time', *time_argv, *cmd_argv]
    argv = [_bash, '-c', f'{shlex.join(argv)}']

    logger.info(f'# {shlex.join(argv)}')

    def run_cmd(*, capture=True):
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            text=capture,
            check=True,
        )
        return proc.stdout

    return run_cmd


#def _build_argv(cmd_argv, *time_argv, _bash=shutil.which('bash')):
#    argv = ['time', *time_argv, *cmd_argv]
#    return [_bash, '-c', f'{shlex.join(argv)}']
#
#
#def _run_cmd(cmd_argv, *time_argv,
#             capture=True,
#             repeating=False,
#             _bash=shutil.which('bash'),
#             ):
#    argv = _build_argv(cmd_argv, *time_argv)
#    if not repeating:
#        logger.info(f'# {shlex.join(argv)}')
#    proc = subprocess.run(
#        argv,
#        stdout=subprocess.PIPE if capture else None,
#        stderr=subprocess.STDOUT if capture else None,
#        text=capture,
#        check=True,
#    )
#    return proc.stdout


def _parse_output(text):
    '''

    real    0m0.025s
    user    0m0.021s
    sys     0m0.001s
    '''
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        kind, duration = line.split()
        if kind not in KINDS:
            raise NotImplementedError
        if kind in result:
            raise NotImplementedError
        result[kind] = parse_duration(duration)
    return result


def _combine_results(*results, average=True):
    if not results:
        return None
    elif len(results) == 1:
        return results[0]

    combined = {k: 0 for k in KINDS}
    for result in results:
        for kind in KINDS:
            combined[kind] += result[kind]
    if average:
        for kind in KINDS:
            combined[kind] /= len(results)
    return combined


def _run_time(cmd_argv, time_argv, repeat, average):
    assert repeat > 1, repeat
    # time bash -c "for i in {1..100}; do ./python -c pass; done"
    run_cmd = _get_cmd_runner(time_argv, cmd_argv)

    header = None
    aggregate = None
    for _ in _utils.scriptutil.iter_with_markers(repeat, verbosity=logger):
        text = run_cmd(capture=True)
        result = _parse_output(text)
        if aggregate:
            aggregate = _combine_results(aggregate, result, average=average)
        else:
            aggregate = result
    return aggregate


##################################
# output

def _render_command_info(info):
    if info is None:
        return

    kind = info['kind']
    data = info['data']
    if not kind:
        return
    elif info['kind'] == 'python':
        version = data['version_str']
        yield f'# Python {version}'

        git = [
            data['build']['git']['revision'][:10],
        ]
        git_extra = []
        branch = data['build']['git']['branch']
        if branch:
            git_extra.append(branch)
        gitdate = data['build']['git']['date']
        if gitdate:
            gitdate = datetime.datetime.fromisoformat(gitdate)
            gitdate = gitdate.strftime('%b |%d %Y, %H:%M:%S %z')
            gitdate = gitdate.replace('|0', '').replace('|', '')
#            gitdate = gitdate.replace('T', ' ').replace('-', ' -').replace('+', ' +')
            git_extra.append(gitdate)
        git.append(f'({", ".join(git_extra)})')
        yield f'# git rev {" ".join(git)}'

        configargs = data['build'].get('configure_args')
        if configargs is not None:
            configargs.insert(0, './configure')
            yield f'# {" ".join(configargs)}'
    else:
        raise NotImplementedError(info)


def _render_result(result):
    yield ''
    for kind in KINDS:
        duration = render_duration(result[kind])
        yield f'{kind + ":":8} {duration}'


##################################
# the script

def parse_args(argv=sys.argv[1:]):
    import argparse
    parser = argparse.ArgumentParser()
    process_verbosity = _utils.scriptutil.add_verbosity_cli(parser)
    parser.add_argument('--repeat', type=int)
    parser.add_argument('--aggregate', dest='average', action='store_false')
    parser.add_argument('--average', action='store_const', const=True)

    args, cmd_argv = parser.parse_known_args(argv)
    # "cmd_argv" should always come after the script's args, so we check:
    parser.parse_args(argv[:-len(cmd_argv)])
    ns = vars(args)

    ns['cmd_argv'] = cmd_argv
    if not cmd_argv:
        parser.error('missing cmd to be traced')

    verbosity = process_verbosity(args)

    return ns, verbosity


def configure_logger(verbosity):
    _utils.scriptutil.configure_logger(logger, verbosity)


def main(cmd_argv, *, repeat=None, average=True):
    time_argv = ()

    if not repeat or repeat <= 1:
        run_cmd = _get_cmd_runner(time_argv, cmd_argv)
        run_cmd(capture=False)
        return

    info = get_command_info(cmd_argv)
    printed = False
    for line in _render_command_info(info):
        print(line)
        printed = True
    if printed:
        print()

    result = _run_time(cmd_argv, time_argv, repeat, average)
    for line in _render_result(result):
        print(line)


if __name__ == '__main__':
    kwargs, verbosity = parse_args()
    configure_logger(verbosity)
    main(**kwargs)
