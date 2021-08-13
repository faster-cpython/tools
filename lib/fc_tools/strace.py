from decimal import Decimal
import itertools
import logging
import shlex
import shutil
import subprocess
import sys

import fc_utils.scriptutil
import fc_utils.tables
_utils = fc_utils
del fc_utils


logger = logging.getLogger(__name__)


def run_strace(argv, *args, repeat=None):
    if not repeat or repeat < 1:
        repeat = 1
    return _run_strace(argv, args, repeat)


##################################
# low-level

def _get_cmd_runner(strace_argv, cmd_argv, *, _strace=shutil.which('strace')):
    argv = [_strace, *strace_argv, *cmd_argv]

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


def _parse_summary(lines):
    '''
    % time     seconds  usecs/call     calls    errors syscall
    ------ ----------- ----------- --------- --------- ----------------
     69.92    0.011653          54       214           rt_sigaction
      7.20    0.001200          39        31        15 stat
    '''
    header, rows = _utils.tables.parse_table(lines)
    _, columns, _ = header

    strace = []
    for row in rows:
        if isinstance(row, str):
            strace.append(row)
            continue
        entry = {}
        emptyvals = set()
        for column, value in zip(columns, row):
            colname = column.strip()
            value = value.strip()
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = Decimal(value)
            elif value == '':
                if colname != 'errors':
                    emptyvals.add(colname)
                if colname != 'syscall':
                    value = 0
            entry[colname] = value
        if 'usecs/call' in emptyvals:
            if entry['syscall'] == 'total':
                emptyvals.remove('usecs/call')
        if emptyvals:
            raise TableParserError(f'unexpectedly empty ({", ".join(sorted(emptyvals))}) in {row}')
        strace.append(entry)
    return strace, header


def _render_strace(strace, header):
    _, columns, _ = header

    rows = []
    for entry in strace:
        if isinstance(entry, str):
            rows.append(entry)
            continue
        row = []
        for column in columns:
            colname = column.strip()
            value = entry[colname]
            if colname == '% time':
                if int(value * 100) == 0:
                    valstr = '0.00'
                else:
                    valstr = f'{value:>4.2f}'
            elif colname == 'seconds':
                valstr = f'{value:>11.6f}'
            elif colname == 'calls':
                valstr = str(int(value))
            elif colname == 'errors':
                valstr = str(int(value))
            elif colname == 'errors' and not value:
                valstr = ''
            #elif entry['syscall'] == 'total' and colname == 'usecs/call':
            #    valstr == ''
            else:
                valstr = str(value)
            row.append(valstr)
        rows.append(row)
    return _utils.tables.render_table(header, rows)


def _combine_straces(*straces, average=True):
    if not straces:
        return None
    elif len(straces) == 1:
        return straces[0]

    total = 0
    combined = {}
    for strace in straces:
        assert strace, straces
        entries = [e for e in strace if not isinstance(e, str)]
        assert entries, straces
        entry0 = entries[0]
        if entry0['% time'] > 0:
            total += entries[0]['seconds'] / (entries[0]['% time'] / 100)
        for entry in entries:
            name = entry['syscall']
            if name in combined:
                for colname in entry:
                    if colname != 'syscall':
                        combined[name][colname] += entry[colname]
            else:
                combined[name] = dict(entry)
    entry_total = combined.pop('total')
    combined = sorted(combined.values(), reverse=True, key=lambda v: v['seconds'])
    for entry in combined:
        entry['% time'] = entry['seconds'] / total * 100
        entry['usecs/call'] = int(entry['seconds'] * 1_000_000) // int(entry['calls'])
    if average:
        for entry in combined:
            entry['seconds'] /= len(straces)
            entry['calls'] /= len(straces)
            entry['errors'] /= len(straces)

    entry_total['seconds'] = sum(e['seconds'] for e in combined)
    entry_total['calls'] = sum(e['calls'] for e in combined)
    entry_total['errors'] = sum(e['errors'] for e in combined)
    entry_total['% time'] = Decimal('100.00')
    #entry_total['usecs/call'] = 0
    entry_total['usecs/call'] = int(entry_total['seconds'] * 1_000_000) // int(entry_total['calls'])

    combined[0:0] = [
        _utils.tables.COLUMNS,
        _utils.tables.DIVIDER,
    ]
    combined.extend([
        _utils.tables.DIVIDER,
        entry_total
    ])
    return combined


def _run_strace(cmd_argv, strace_argv, repeat, average):
    assert repeat > 1, repeat
    # strace -e 'trace=!rt_sigprocmask,rt_sigreturn,wait4,clone' -c bash -c "for i in {1..100}; do ./python -c pass; done"
    run_cmd = _get_cmd_runner(strace_argv, cmd_argv)

    header = None
    aggregate = None
    for _ in _utils.scriptutil.iter_with_markers(repeat, verbosity=logger):
        text = run_cmd(capture=True)
        strace, _header = _parse_summary(text)
        if aggregate:
            aggregate = _combine_straces(aggregate, strace, average=average)
            assert _header == header, (_header, header)
        else:
            aggregate = strace
            header = _header
    return aggregate, header


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
    strace_argv = ['-c']

    if not repeat or repeat <= 1:
        run_cmd = _get_cmd_runner(strace_argv, cmd_argv)
        run_cmd(capture=False)
        return

    results, header = _run_strace(cmd_argv, strace_argv, repeat, average)
    for line in _render_strace(results, header):
        print(line)


if __name__ == '__main__':
    kwargs, verbosity = parse_args()
    configure_logger(verbosity)
    main(**kwargs)
