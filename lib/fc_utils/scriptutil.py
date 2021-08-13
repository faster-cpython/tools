import contextlib
import logging
import os
import sys


TRACEBACK = os.environ.get('SHOW_TRACEBACK', '').strip()
TRACEBACK = bool(TRACEBACK and TRACEBACK.upper() not in ('0', 'FALSE', 'NO'))

VERBOSITY = 3


def add_logging_cli(parser):
    parser.add_argument('--logfile')

    def process_args(args):
        ns = vars(args)
        return ns.pop('logfile')
    return process_args


def add_verbosity_cli(parser, *, default=VERBOSITY):
    parser.add_argument('-q', '--quiet', action='count', default=0)
    parser.add_argument('-v', '--verbose', action='count', default=0)

    def process_args(args):
        ns = vars(args)
        verbosity = max(0, default + ns.pop('verbose') - ns.pop('quiet'))
        return verbosity
    return process_args


def add_traceback_cli(parser, *, default=TRACEBACK):
    parser.add_argument('--traceback', '--tb', action='store_true',
                        default=default)
    parser.add_argument('--no-traceback', '--no-tb', dest='traceback',
                        action='store_const', const=False)

    def process_args(args):
        ns = vars(args)
        showtb = ns.pop('traceback')

        @contextlib.contextmanager
        def traceback_cm():
            try:
                yield
            except BrokenPipeError:
                # It was piped to "head" or something similar.
                pass
            except Exception as exc:
                if not showtb:
                    sys.exit(f'ERROR: {exc}')
                raise  # re-raise
            except KeyboardInterrupt:
                if not showtb:
                    sys.exit('\nINTERRUPTED')
                raise  # re-raise
            except BaseException as exc:
                if not showtb:
                    sys.exit(f'{type(exc).__name__}: {exc}')
                raise  # re-raise
        return traceback_cm()
    return process_args


##################################
# logging

def configure_logger(logger, verbosity=VERBOSITY, *,
                     logfile=None,
                     maxlevel=logging.CRITICAL,
                     ):
    level = verbosity_to_loglevel(verbosity)
    logger.setLevel(level)
    #logger.propagate = False

    if not logger.handlers:
        if logfile:
            handler = logging.FileHandler(logfile)
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        #handler.setFormatter(logging.Formatter())
        logger.addHandler(handler)


def verbosity_to_loglevel(verbosity, *, maxlevel=logging.CRITICAL):
    assert maxlevel is not None
    if isinstance(maxlevel, str):
        _maxlevel = get_valid_loglevel(maxlevel)
        if _maxlevel is None:
            raise ValueError(f'unsupported maxlevel {maxlevel!r}')
        maxlevel = maxlevel
    elif logging.getLevelName(maxlevel).startswith('Level '):
        # We are okay with a level higher that CRITICAL
        # but it must be a multiple of 10.
        if maxlevel % 10:
            # XXX Use the ceiling instead of the floor?
            maxlevel -= maxlevel % 10
    return max(1,  # 0 disables it, so we use the next lowest.
               min(maxlevel,
                   maxlevel - verbosity * 10))


def loglevel_to_verbosity(loglevel):
    if isinstance(loglevel, str):
        _loglevel = get_valid_loglevel(loglevel)
        if _loglevel is None:
            raise ValueError(f'unsupported loglevel {loglevel!r}')
        loglevel = loglevel
    elif not isinstance(loglevel, int):
        # It must be a logger.
        loglevel = loglevel.getEffectiveLevel()
    return _loglevel_to_verbosity(loglevel)


def _loglevel_to_verbosity(loglevel):
    verbosity = loglevel // 10
    if loglevel % 10:
        verbosity += 1
    return verbosity


def get_valid_loglevel(loglevel):
    if isinstance(loglevel, int):
        # It's okay it it's bigger than logging.CRITICAL
        # but it can't be negative.
        return loglevel if loglevel >= 0 else None
    loglevel = logging.getLevelName(loglevel)
    return loglevel if isinstance(loglevel, int) else None


def check_verbosity(verbosity, minverbosity=VERBOSITY):
    if minverbosity is None:
        return True
    if isinstance(verbosity, str):
        loglevel = get_valid_loglevel(verbosity)
        if loglevel is None:
            raise ValueError(f'unsupported verbosity {verbosity!r}')
        verbosity = _loglevel_to_verbosity(loglevel)
    elif isinstance(verbosity, logging.Logger):
        verbosity = loglevel_to_verbosity(verbosity)
    return verbosity < minverbosity


##################################
# tables

def generate_table(cols):
    header = []
    div = []
    fmt = []
    for name, width in cols.items():
        #header.append(f'{:^%d}' % (width,))
        header.append(name.center(width + 2))
        div.append('-' * (width + 2))
        fmt.append(' {:%s} ' % (width,))
    return ' '.join(header), ' '.join(div), ' '.join(fmt)


def render_table(rows, cols, *, show_total=True, fit=True):
    header, div, fmt = generate_table(cols)
    yield header
    yield div
    total = 0
    if fit:
        widths = [w for w in cols.values()][:-1]
        for total, row in enumerate(rows):
            fixed = (str(v)[:w] for v, w in zip(row, widths))
            yield fmt.format(*fixed, *row[-1:])
    else:
        for total, row in enumerate(rows):
            yield fmt.format(*row)
    yield div
    if show_total:
        yield ''
        yield f'total: {total}'


##################################
# main output

def iter_with_markers(items, small=5, smallperline=4, linespergroup=5, *,
                      minverbosity=VERBOSITY,
                      #minverbosity=None,
                      verbosity=None,
                      ):
    if isinstance(items, int):
        items = range(items)
    if verbosity is not None and not check_verbosity(verbosity, minverbosity):
        yield from items
        return

    line = small * smallperline
    group = line * linespergroup
    items = enumerate(items, 1)
    try:
        i = None
        for i, item in items:
            print('.', end='', file=sys.stderr, flush=True)
            if i % small == 0:
                print(' ', end='', file=sys.stderr, flush=True)
            if i % line == 0:
                print(file=sys.stderr, flush=True)
            if i % group == 0:
                print(file=sys.stderr, flush=True)
            yield item
    finally:
        # If there were no items then we don't need an EOL.
        if (i or line) % line:
            print(file=sys.stderr, flush=True)
