from collections import namedtuple
import datetime
import decimal
import glob
import json
import logging
import os
import os.path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import types
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO,
    Tuple, Type, Union
)


'''
sudo adduser --gecos '' --disabled-password <username>
sudo --login --user <username> ssh-import-id gh:<username>
'''


ListOrStrType = Union[str, List[str]]


USER = os.environ.get('USER', '').strip()
SUDO_USER = os.environ.get('SUDO_USER', '').strip()
HOME = os.path.expanduser('~')
CWD = os.getcwd()
PID = os.getpid()

logger = logging.getLogger(__name__)


##################################
# string utils

def check_name(name: str, *, loose: bool = False) -> None:
    if not name:
        raise ValueError(name)
    orig = name
    if loose:
        name = '_' + name.replace('-', '_')
    if not name.isidentifier():
        raise ValueError(orig)


def check_str(
        valuestr: str,
        label: Optional[str] = None,
        *,
        required: bool = False,
        fail: bool = False
) -> bool:
    if not valuestr:
        if required:
            if fail:
                raise ValueError(f'missing {label}' if label else 'missing')
            return False
    elif not isinstance(valuestr, str):
        if fail or fail is None:
            raise TypeError(valuestr)
        return False
    return True


def validate_str(
        value: str,
        argname: Optional[str] = None,
        *,
        required: bool = True
) -> None:
    validate_arg(value, str, argname, required=required)


##################################
# int utils

def ensure_int(raw: Any, min: Optional[int] = None) -> int:
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        value = int(raw)
    else:
        raise TypeError(raw)
    if min is not None and value < min:
        raise ValueError(raw)
    return value


def coerce_int(value: Any, *, fail: bool = False) -> Optional[int]:
    if isinstance(value, int):
        return value
    elif not value:
        if fail:
            raise ValueError('missing')
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            if fail or fail is None:
                raise  # re-raise
    else:
        if fail:
            raise TypeError(f'unsupported value {value!r}')
    return None


def validate_int(
        value: Any,
        name: Optional[str] = None,
        *,
        range: Optional[str] = None,
        required: bool = True
) -> Optional[int]:
    def fail(value=value, name=name, range=range):
        qualifier = 'an'
        if isinstance(value, int):
            if range:
                qualifier = f'a {range}'
            Error = ValueError
        else:
            Error = TypeError
        namepart = f' for {name}' if name else ''
        raise Error(f'expected {qualifier} int{namepart}, got {value!r}')

    if isinstance(value, int) and value is not False:
        if not range:
            return value
        elif range == 'non-negative':
            if value < 0:
                fail()
            return value
        elif range == 'positive':
            if value <= 0:
                fail()
            return value
        else:
            raise NotImplementedError(f'unsupported range {range!r}')
    elif not value:
        if not required:
            return None
        raise ValueError(f'missing {name}' if name else 'missing')
    else:
        fail()
    return None  # unreachable

def normalize_int(
        value: Any,
        name: Optional[str] = None,
        *,
        range: Optional[str] = None,
        coerce: bool = False,
        required: bool = True,
) -> Optional[int]:
    if coerce:
        value = coerce_int(value)
    return validate_int(value, name, range=range, required=required)


##################################
# validation utils

def validate_arg(
        value: Any,
        expected: Type,
        argname: Optional[str] = None,
        *,
        required: bool = True
) -> None:
    if not isinstance(expected, type):
        raise NotImplementedError(expected)
    if not value:
        if not required:
            return
        raise ValueError(f'missing {argname or "required value"}')
    if not isinstance(value, expected):
        label = f' for {argname}' if argname else ''
        raise TypeError(f'expected {expected.__name__}{label}, got {value!r}')


##################################
# tables

class ColumnSpec(namedtuple('ColumnSpec', 'name title width align')):
    @classmethod
    def from_raw(cls, raw: Any):
        if isinstance(raw, cls):
            return raw
        else:
            return cls(*raw)


class TableSpec(namedtuple('TableSpec', 'columns header div rowfmt')):

    DIV = '-'
    SEP = ' '
    MARGIN = 1

    @classmethod
    def from_columns(
            cls,
            specs: Iterable[Any],
            names: Optional[Union[str, Iterable[str]]] = None,
            *,
            maxwidth: Optional[int] = None
    ) -> "TableSpec":
        columns, normalized_names = cls._normalize_columns(specs, names, maxwidth)
        margin = ' ' * cls.MARGIN

        header = div = rowfmt = ''
        for col in columns:
            if header:
                header += cls.SEP
                div += cls.SEP
                rowfmt += cls.SEP
            header += f'{margin}{col.title or col.name:^{col.width}}{margin}'
            div += cls.DIV * (cls.MARGIN + col.width + cls.MARGIN)
            if col.align:
                rowfmt += f'{margin}{{:{col.align}{col.width}}}{margin}'
            else:
                rowfmt += f'{margin}{{:{col.width}}}{margin}'

        self = cls(columns, header, div, rowfmt)
        if names:
            self._colnames = normalized_names  # type: ignore[has-type]
        else:
            self._colnames = None  # type: ignore[has-type]
        return self

    @classmethod
    def _normalize_columns(
            cls,
            specs: Iterable[Any],
            names: Optional[Union[str, Iterable[str]]],
            maxwidth: Optional[int]
    ) -> Tuple[List[Any], Optional[List[str]]]:
        spec_objs = (ColumnSpec.from_raw(s) for s in specs)
        if names:
            if isinstance(names, str):
                names_list = names.replace(',', ' ').split()
            else:
                names_list = list(names)
            specs_by_name = {s.name: s for s in spec_objs}
            columns = [specs_by_name[n] for n in names_list]
        else:
            columns = list(specs)
            names_list = None

        if not maxwidth or maxwidth < 0:
            # XXX Use a minimum set of columns to determine minwidth.
            minwidth = 80
            maxwidth_int = max(minwidth, get_termwidth())
            #print(' '*maxwidth + '|')
        else:
            maxwidth_int = maxwidth

        sep = cls.SEP
        if not sep:
            raise ValueError('sep missing')
        elif not isinstance(sep, str):
            raise TypeError(sep)

        margin = cls.MARGIN
        if margin is None:
            raise ValueError('margin missing')
        elif not isinstance(margin, int):
            raise TypeError(cls.MARGIN)
        elif margin < 0:
            raise ValueError(f'invalid margin {margin}')

        # Drop columns until below maxwidth.
        size = 0
        for i in range(len(columns)):
            if i > 0:
                size += len(sep)
            size += margin + columns[i].width + margin
            if size > maxwidth_int:
                # XXX Maybe drop other columns than just the tail?
                # XXX Maybe combine some columns?
                columns[i:] = []
                if names_list:
                    names_list[i:] = []
                break

        #if termwidth > minwidth + (19 + 3) * 3:
        #    columns.extend([
        #        ('created', None, 19, None),
        #        ('started', None, 19, None),
        #        ('finished', None, 19, None),
        #    ])
        #elif termwidth > minwidth + (21 + 3) + (19 + 3):
        #    columns.extend([
        #        ('started,created', 'started / (created)', 21, None),
        #        ('finished', None, 19, None),
        #    ])
        #elif termwidth > minwidth + (21 + 3):
        #    columns.append(
        #        ('started,created', 'start / (created)', 21, None),
        #    )

        return columns, names_list

    @property
    def colnames(self) -> List[str]:
        try:
            return self._colnames  # type: ignore[has-type]
        except AttributeError:
            self._colnames = [c.name for c in self.columns]
            return self._colnames

    def render(
            self,
            rows: Iterable["TableRow"],
            *,
            numrows: Optional[int] = None,
            total: Optional[int] = None
    ) -> Iterable[str]:
        # We use a default format here.
        header = [self.div, self.header, self.div]
        rendered_rows = self._render_rows(rows)
        rendering_rows = RenderingRows(rendered_rows, header, numrows)
        yield from header
        yield from rendering_rows
        yield self.div
        yield from rendering_rows.render_count(total)

    def render_rows(
            self,
            rows: Iterable["TableRow"],
            periodic: Optional[Union[str, Iterable[str]]] = None,
            numrows: Optional[int] = None
    ) -> "RenderingRows":
        rendered_rows = self._render_rows(rows)
        return RenderingRows(rendered_rows, periodic, numrows)

    def _render_rows(self, rows: Iterable["TableRow"]) -> Iterable[str]:
        fmt = self.rowfmt
        for row in rows:
            values = row.render_values(self.colnames)
            yield fmt.format(*values)

    def _render_row(self, row: "TableRow") -> str:
        values = row.render_values(self.colnames)
        return self.rowfmt.format(*values)


class RenderingRows:
    pending: List[str]

    def __init__(
            self,
            rows: Iterable[str],
            periodic: Optional[Union[str, Iterable[str]]] = None,
            numrows: Optional[int] = None
    ):
        if not periodic:
            periodic = None
        elif isinstance(periodic, str):
            periodic = [periodic]
        if not numrows:
            try:
                numrows = len(list(rows))
            except TypeError:
                numrows = None
        self.rows = iter(rows)
        self.periodic = periodic
        self.numrows = numrows

        self.count = 0
        self.pending = []

    def __eq__(self, other):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self.pending:
            return self.pending.pop(0)
        row = next(self.rows)
        self.count += 1
        if self.periodic:
            if self.count % 25 == 0:
                if not self.numrows or self.numrows - self.count >= 15:
                    self.pending.extend(self.periodic)
                    self.pending.append(row)
                    return self.pending.pop(0)
        return row

    def render_count(
            self,
            total: Optional[int] = None,
            label: str = 'matched'
    ) -> Iterable[str]:
        if total is None:
            yield f'(total: {total})'
        elif self.count == total:
            yield f'(total: {total})'
        else:
            yield f'({label}: {self.count})'
            yield f'(total:   {total})'


class TableRow:

    def __init__(
            self,
            data: Mapping[str, Any],
            render_value: Callable[[str], str]
    ):
        self.data = data
        self.render_value = render_value
        self._colnames = list(data)

    def __repr__(self):
        return f'{type(self).__name__}({self.data!r})'

    def __eq__(self, other):
        raise NotImplementedError

    def render_values(
            self,
            colnames: Optional[Iterable[str]] = None
    ) -> Iterable[str]:
        if not colnames:
            colnames = self._colnames
        for colname in colnames:
            if ',' in colname:
                primary, _, secondary = colname.partition(',')
                if self.data[primary] is None:
                    if self.data[secondary] is not None:
                        value = self.render_value(secondary)
                        yield f'({secondary})'
                        continue
                value = self.render_value(primary)
                yield f' {value} '
            else:
                yield self.render_value(colname)


##################################
# date/time utils

SECOND = datetime.timedelta(seconds=1)
DAY = datetime.timedelta(days=1)


def utcnow() -> float:
    if time.tzname[0] == 'UTC':
        return time.time()
    return time.mktime(time.gmtime())


def get_utc_datetime(
        timestamp: Optional[Any] = None,
        *,
        fail: bool = True
) -> Tuple[Optional[datetime.datetime], Optional[bool]]:
    tzinfo = datetime.timezone.utc
    if timestamp is None:
        timestamp = int(utcnow())
    if isinstance(timestamp, int):
        timestamp = datetime.datetime.fromtimestamp(timestamp, tzinfo)
    elif isinstance(timestamp, str):
        if re.match(r'^\d{4}-\d\d-\d\d$', timestamp):
            timestamp = datetime.date(*(int(v) for v in timestamp.split('-')))
        elif hasattr(datetime.datetime, 'fromisoformat'):  # 3.7+
            timestamp = datetime.datetime.fromisoformat(timestamp)
            timestamp = timestamp.astimezone(tzinfo)
        else:
            m = re.match(r'(\d{4}-\d\d-\d\d(.)\d\d:\d\d:\d\d)(\.\d{3}(?:\d{3})?)?([+-]\d\d:?\d\d.*)?', timestamp)
            if not m:
                if fail:
                    raise NotImplementedError(repr(timestamp))
                return None, None
            body, sep, subzero, tz = m.groups()
            timestamp = body
            fmt = f'%Y-%m-%d{sep}%H:%M:%S'
            if subzero:
                if len(subzero) == 4:
                    subzero += '000'
                timestamp += subzero
                fmt += '.%f'
            if tz:
                timestamp += tz.replace(':', '')
                fmt += '%z'
            timestamp = datetime.datetime.strptime(timestamp, fmt)
            timestamp = timestamp.astimezone(tzinfo)
    elif isinstance(timestamp, datetime.datetime):
        # XXX Treat naive as UTC?
        timestamp = timestamp.astimezone(tzinfo)
    elif not isinstance(timestamp, datetime.date):
        raise TypeError(f'unsupported timestamp {timestamp!r}')
    hastime = True
    if type(timestamp) is datetime.date:
        d = timestamp
        timestamp = datetime.datetime(d.year, d.month, d.day, tzinfo=tzinfo)
        #timestamp = datetime.datetime.combine(timestamp, None, datetime.timezone.utc)
        hastime = False
    return timestamp, hastime


class ElapsedTimeWithUnits:
    """A quanitity of elapsed time in specific units."""
    # We'd subclass decimal.Decimal or datetime.timedelta,
    # but neither is friendly to subclassing (methods drop
    # the subclass).  Plus, timedelta doesn't go smaller
    # than microseconds.

    __slots__ = ['_elapsed', '_units']

    UNITS = ['s', 'ms', 'us', 'ns']  # us == microseconds
    _UNITS = {s: 1000**i for i, s in enumerate(UNITS)}
    PAT = rf'''
        (?:
            \b
            (
                (?: 0 | 0* [1-9] \d* )
                (?: \. \d+ )?
             )  # <elapsed>
            (?:
                \s+
                ( sec | {'|'.join(UNITS)} )  # <units>
             )?
            \b
         )
        '''
    REGEX = re.compile(f'^{PAT}$', re.VERBOSE)
    _units: str

    @classmethod
    def parse(cls, elapsedstr, *, fail=False):
        m = cls.REGEX.match(elapsedstr)
        if not m:
            if fail:
                raise ValueError(f'could not parse {elapsedstr!r}')
            return None
        elapsedstr, units = m.groups()
        if units == 'sec':
            units = 's'
        #elapsed = decimal.Decimal(elapsedstr).normalize()
        elapsed = decimal.Decimal(elapsedstr)
        return cls._from_values(elapsed, units)

    @classmethod
    def from_seconds(cls, elapsed, *, normalize=True):
        elapsed = decimal.Decimal(elapsed)
        cls._validate_elapsed(elapsed)
        return cls._from_values(elapsed, None if normalize else 's')

    @classmethod
    def from_timedelta(cls, elapsed):
        if not isinstance(elapsed, datetime.timedelta):
            raise TypeError(elapsed)
        micro = elapsed.microseconds
        elapsed = decimal.Decimal(elapsed.total_seconds())
        if micro:
            elapsed = (elapsed * 1_000_000 + micro) / 1_000_000
        cls._validate_elapsed(elapsed)
        return cls._from_values(elapsed, units=None)

    @classmethod
    def from_values(cls, elapsed, units=None):
        if isinstance(elapsed, datetime.timedelta):
            if units:
                raise TypeError('datetime.timedelta has fixed units; use cls.from_timedelta()')
            return cls.from_timedelta(elapsed)
        else:
            elapsed = decimal.Decimal(elapsed)
            cls._validate_elapsed(elapsed)
            return cls._from_values(elapsed, units)

    @classmethod
    def _from_values(cls, elapsed, units):
        assert units
        if units:
            self = cls.__new__(cls, elapsed, units)
            self._check_for_abnormal(elapsed, units)
        else:
            self = cls.__new__(cls, elapsed, 's')
            self = self.normalize()
        return self

    @classmethod
    def _convert_to_seconds(
            cls,
            elapsed,
            units: Optional[str]
    ) -> "ElapsedTimeWithUnits":
        if units == 's':
            return elapsed
        return (elapsed / cls._resolve_units(units)).normalize()

    @classmethod
    def _convert_from_seconds(
            cls,
            elapsed,
            units: str
    ) -> "ElapsedTimeWithUnits":
        if units == 's':
            return elapsed
        return (elapsed * cls._resolve_units(units)).normalize()

    @classmethod
    def _resolve_units(cls, units: Optional[str]) -> int:
        if units not in cls._UNITS:
            raise ValueError(f'unsupported units {units!r}')
        return cls._UNITS[units]

    @classmethod
    def _validate_elapsed(cls, elapsed) -> None:
        if elapsed < 0:
            raise ValueError(f'expected non-negative value, got {elapsed}')

    @classmethod
    def _check_for_abnormal(cls, elapsed, units: str) -> None:
        if elapsed < 1 or elapsed >= 1000:
            logger.warning('abnormal elapsed value {elapsed} {units}, consider normalizing')

    def __new__(
            cls,
            elapsed,
            units: str = 's'
    ) -> "ElapsedTimeWithUnits":
        self = super().__new__(cls)
        self._elapsed = decimal.Decimal(elapsed)
        self._units = units
        return self

    def __init__(self, *args, **kwargs):
        self._elapsed = self._elapsed.normalize()
        self._validate()

    def _validate(self):
        self._validate_elapsed(self._elapsed)
        if not self._units:
            raise ValueError('missing units')
        elif self._units not in self._UNITS:
            raise ValueError(f'unsupported units {self._units!r}')
        self._check_for_abnormal(self._elapsed, self._units)

    def __repr__(self):
        return f'{type(self).__name__}({self._elapsed}, {self._units})'

    def __str__(self):
        # XXX Limit to 2 decimal places?
        return f'{self._elapsed} {self_units}'

    def __hash__(self):
        return hash((self._elapsed, self._units))

    def __eq__(self, other):
        if not isinstance(other, ElapsedTimeWithUnits):
            return NotImplemented
        # XXX Normalize units first?
        if self._units != other._units:
            return False
        if self._elapsed != other._elapsed:
            return False
        return True

    @property
    def value(self):
        return self._elapsed

    @property
    def units(self):
        return self._units

    def normalize(self):
        elapsed = self._elapsed
        if elapsed >= 0 and elapsed < 1000:
            return self
        if elapsed >= 1000:
            candidates = iter(reversed(self.UNITS))
            for units in candidates:
                if units == self._units:
                    break
            for units in candidates:
                elapsed /= 1000
                if elapsed < 1000:
                    break
            else:
                # We leave it at the maximum units.
                pass
        else:
            candidates = iter(self.UNITS)
            for units in candidates:
                if units == self._units:
                    break
            for units in candidates:
                elapsed *= 1000
                if elapsed >= 1:
                    break
            else:
                # XXX Stick with the smallest units?
                raise ValueError(f'{self} is smaller than the smallest units ({units})')
        cls = type(self)
        return cls.__new__(cls, elapsed.normalize(), units)

    def convert_to(self, units='s'):
        if not units:
            raise ValueError('missing units')
        if self._units == units:
            return self
        # We always convert to seconds (the largest unit) first.
        converted = self._convert_to_seconds(self._elapsed, self._units)
        # Now we convert to the target.
        converted = self._convert_from_seconds(converted, units)
        cls = type(self)
        return cls.__new__(cls, converted, units)

    def as_seconds(self):
        return self.convert_to()._elapsed

    def as_timedelta(self):
        # XXX Fail if there are any digits smaller than microseconds?
        return datetime.timedelta(seconds=float(self._elapsed))


class ElapsedTimeComparison:
    """The relative time difference between two elapsed time values."""
    # We would subclass Decimal but it isn't subclass-friendly
    # (methods drop the subclass).

    __slots__ = ['_raw']

    PAT = r'''
        (?:
            ( [1-9]\d*(?:\.\d+) )  # <value>
            x
            \s+
            ( faster | slower )  # <direction>
         )
        '''
    REGEX = re.compile(f'^{PAT}$', re.VERBOSE)

    @classmethod
    def parse(cls, comparisonstr, *, fail=False):
        m = cls.REGEX.match(comparisonstr)
        if not m:
            if fail:
                raise ValueError(f'could not parse {comparisonstr!r}')
            return None
        valuestr, direction = m.groups()
        return cls.from_parsed_values(valuestr, direction, comparisonstr)

    @classmethod
    def from_parsed_values(cls, valuestr, direction, raw=None):
        if direction == 'slower':
            valuestr = f'-{valuestr}'
        elif direction != 'faster':
            raise NotImplementedError(raw or direction)
        return cls.__new__(cls, valuestr)

    def __new__(cls, value):
        self = super().__new__(cls)
        self._raw = decimal.Decimal(value)
        return self

    def __init__(self, *args, **kwargs):
        self._validate()

    def __hash__(self):
        return hash(self._raw)

    def _validate(self):
        if not (self._raw % 1):
            raise ValueError(f'expected >= 1 or <= -1, got {self._raw}')

    def __repr__(self):
        return f'{type(self).__name__}({self._raw})'

    def __str__(self):
        if self._raw < 0:
            return f'{-self._raw}x slower'
        else:
            return f'{self._raw}x faster'

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def raw(self):
        return self._raw


##################################
# OS utils

def resolve_user(cfg: Any, user: Optional[str] = None) -> str:
    if not user:
        user = USER
        if not user or user == 'benchmarking':
            user = SUDO_USER
            if not user:
                raise Exception('could not determine user')
    if not user.isidentifier():
        raise ValueError(f'invalid user {user!r}')
    return user


def parse_bool_env_var(
        valstr: str,
        *,
        failunknown: bool = False
) -> Optional[bool]:
    m = re.match(r'^\s*(?:(1|y(?:es)?|t(?:rue)?)|(0|no?|f(?:alse)?))\s*$',
                 valstr.lower())
    if not m:
        if failunknown:
            raise ValueError(f'unsupported env var bool value {valstr!r}')
        return None
    yes, no = m.groups()
    return True if yes else False


def get_bool_env_var(
        name: str,
        default: Optional[bool] = None,
        *,
        failunknown: bool = False
) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return default
    return parse_bool_env_var(value, failunknown=failunknown)


def resolve_os_name() -> str:
    if sys.platform == 'win32':
        return 'windows'
    elif sys.platform == 'linux':
        return 'linux'
    elif sys.platform == 'darwin':
        return 'mac'
    else:
        raise NotImplementedError(sys.platform)


def resolve_cpu_arch() -> str:
    uname = platform.uname()
    machine = uname.machine.lower()
    if machine in ('amd64', 'x86_64'):
        return machine
    elif machine == 'aarch64':
        return 'arm64'
    elif 'arm' in machine:
        return 'arm'
    else:
        raise NotImplementedError(machine)


def get_termwidth(*, notty: int = 1000, unknown: int = 80) -> int:
    if os.isatty(sys.stdout.fileno()):
        try:
            termsize = os.get_terminal_size()
        except OSError:
            return notty or 1000
        else:
            return termsize.columns
    else:
        return unknown or 80


def is_proc_running(pid: int) -> bool:
    if pid == PID:
        return True
    try:
        if os.name == 'nt':
            os.waitpid(pid, os.WNOHANG)
        else:
            os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # XXX Does this *always* mean there's a proc?
        return True
    else:
        return True


def run_fg(
        cmd: str,
        *args,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None
) -> subprocess.CompletedProcess:
    if not cwd:
        cwd = os.getcwd()
    argv = [cmd, *args]
    logger.debug('# running: %s  (CWD: %s)',
                 ' '.join(shlex.quote(a) for a in argv),
                 cwd)
    return subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        cwd=cwd,
        env=env,
    )


def run_bg(
        argv: Union[str, Sequence[str]],
        logfile: Optional[str] = None,
        *,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None
) -> None:
    if not cwd:
        cwd = os.getcwd()

    if not argv:
        raise ValueError('missing argv')
    elif isinstance(argv, str):
        if not argv.strip():
            raise ValueError('missing argv')
        cmd = argv
    else:
        cmd = ' '.join(shlex.quote(a) for a in argv)

    if logfile:
        logfile = quote_shell_str(logfile)
        cmd = f'{cmd} >> {logfile}'
    cmd = f'{cmd} 2>&1'

    logger.debug('# running (background): %s  (CWD: %s)', cmd, cwd)
    #subprocess.run(cmd, shell=True)
    subprocess.Popen(
        cmd,
        #creationflags=subprocess.DETACHED_PROCESS,
        #creationflags=subprocess.CREATE_NEW_CONSOLE,
        #creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        shell=True,
        cwd=cwd,
        env=env,
    )


##################################
# file utils

def check_shell_str(
        value: Any,
        *,
        required: bool = True,
        allowspaces: bool = False
) -> Optional[str]:
    validate_str(value, required=required)
    if not value:
        return None
    if not allowspaces and ' ' in value:
        raise ValueError(f'unexpected space in {value!r}')
    return value


def quote_shell_str(
        value: Any,
        *,
        required: bool = True
) -> str:
    value = check_shell_str(value, required=required, allowspaces=True)
    if value is not None:
        value = shlex.quote(value)
    return value


def get_next_line(
        lines: Iterable[str],
        notfound: Optional[str] = None,
        *,
        skipempty: bool = False
) -> Optional[str]:
    for line in lines:
        if not skipempty or line.rstrip():
            return line
    else:
        return notfound


def strict_relpath(filename: str, rootdir: Optional[str]) -> str:
    relfile = os.path.relpath(filename, rootdir)
    if relfile.startswith('..' + os.path.sep):
        raise ValueError(f'relpath mismatch ({filename!r}, {rootdir!r})')
    return relfile


def write_json(data, outfile):
    json.dump(data, outfile, indent=4)
    print(file=outfile)


def wait_for_file(filename, *, timeout=None):
    if timeout is not None and timeout > 0:
        if not isinstance(timeout, (int, float)):
            raise TypeError(f'timeout must be an float or int, got {timeout!r}')
        end = time.time() + int(timeout)
        while not os.path.exists(filename):
            time.sleep(0.01)
            if time.time() >= end:
                raise TimeoutError
    else:
        while not os.path.exists(filename):
            time.sleep(0.01)


def read_file(filename, *, fail=True):
    try:
        with open(filename) as infile:
            return infile.read()
    except OSError as exc:
        if fail:
            raise  # re-raise
        if os.path.exists(filename):
            logger.warning('could not load file %r', filename)
        return None


def tail_file(filename, nlines, *, follow=None):
    tail_args = []
    if nlines:
        tail_args.extend(['-n', f'{nlines}' if nlines > 0 else '+0'])
    if follow:
        tail_args.append('--follow')
        if follow is not True:
            pid = follow
            tail_args.extend(['--pid', f'{pid}'])
    subprocess.run([shutil.which('tail'), *tail_args, filename])


def render_file(filename):
    if not filename:
        return '---'
    elif isinstance(filename, FSTree):
        filename = filename.root
    if not os.path.exists(filename):
        return f'({filename})'
    elif filename[0].isspace() or filename[-1].isspace():
        return repr(filename)
    else:
        return filename


class FSTree(types.SimpleNamespace):

    @classmethod
    def from_raw(cls, raw, *, name=None):
        if isinstance(raw, cls):
            return raw
        elif not raw:
            raise ValueError('missing {name or "raw"}')
        elif isinstance(raw, str):
            return cls(raw)
        else:
            raise TypeError(f'expected FSTree, got {raw!r}')

    def __init__(self, root):
        if not root or root == '.':
            root = CWD
        else:
            root = os.path.abspath(os.path.expanduser(root))
        super().__init__(root=root)

    def __str__(self):
        return self.root

    def __fspath__(self):
        return self.root


class InvalidPIDFileError(RuntimeError):

    def __init__(self, filename, text, reason=None):
        msg = f'PID file {filename!r} is not valid'
        if reason:
            msg = f'{msg} ({reason})'
        super().__init__(msg)
        self.filename = filename
        self.text = text
        self.reason = reason


class OrphanedPIDFileError(InvalidPIDFileError):

    def __init__(self, filename, pid):
        super().__init__(filename, str(pid), f'proc {pid} not running')
        self.pid = pid


class PIDFile:

    def __init__(self, filename):
        self._filename = filename

    def __repr__(self):
        return f'{type(self).__name__}({self._filename!r})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def filename(self):
        return self._filename

    def read(self, *, invalid='fail', orphaned=None):
        """Return the PID recorded in the file."""
        if invalid is None:
            def handle_invalid(text):
                return text
        if invalid == 'fail':
            def handle_invalid(text):
                raise InvalidPIDFileError(self._filename, text)
        elif invalid == 'remove':
            def handle_invalid(text):
                logger.warning('removing invalid PID file (%s)', self._filename)
                self.remove()
                return None
        else:
            raise ValueError(f'unsupported invalid handler {invalid!r}')

        #text = read_file(self._filename, fail=False) or ''
        try:
            with open(self._filename) as pidfile:
                text = pidfile.read()
        except FileNotFoundError:
            return None

        text = text.strip()
        if not text or not text.isdigit():
            return handle_invalid(text)
        pid = int(text)
        if pid <= 0:
            return handle_invalid(text)

        if orphaned is not None and not is_proc_running(pid):
            if orphaned == 'fail':
                raise OrphanedPIDFileError(self._filename, pid)
            elif orphaned == 'remove':
                logger.warning('removing orphaned PID file (%s)', self._filename)
                self.remove()
                return None
            elif orphaned == 'ignore':
                return None
            else:
                raise ValueError(f'unsupported orphaned handler {orphaned!r}')
        return pid

    def write(self, pid=PID, *, exclusive=True, **read_kwargs):
        """Return True for success after trying to create the file."""
        pid = int(pid) if pid else PID
        assert pid > 0, pid
        try:
            if exclusive:
                try:
                    pidfile = open(self._filename, 'x')
                except FileExistsError:
                    _pid = self.read(**read_kwargs)
                    if _pid == pid:
                        return pid
                    elif _pid is not None:
                        return None
                    # Looks like there was a race or invalid files.
                    #  Try one more time.
                    pidfile = open(self._filename, 'x')
            else:
                pidfile = open(self._filename, 'w')
            with pidfile:
                pidfile.write(f'{pid}')
            return pid
        except OSError as exc:
            logger.warning('failed to create PID file (%s): %s', self._filename, exc)
            return None

    def remove(self):
        try:
            os.unlink(self._filename)
        except FileNotFoundError:
            logger.warning('lock file not found (%s)', self._filename)


class LockFile:
    """A multi-process equivalent to threading.RLock."""

    def __init__(self, filename):
        self._pidfile = PIDFile(filename)
        self._count = 0

    def __eq__(self, other):
        raise NotImplementedError

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    @property
    def filename(self):
        return self._pidfile.filename

    def read(self):
        return self._pidfile.read(invalid='fail', orphaned='fail')

    def owned(self):
        """Return True if the current process holds the lock."""
        owner = self.owner()
        if owner is None:
            return False
        return owner == PID

    def owner(self):
        """Return the PID of the process that is holding the lock."""
        if self._count > 0:
            return PID
        pid = self._pidfile.read(invalid='remove', orphaned='remove')
        if pid == PID:
            assert self._count == 0, self._count
            raise NotImplementedError
        return pid

    ###################
    # threading.Lock API

    def locked(self):
        return self.owner() is not None

    def acquire(self, blocking=True, timeout=-1):
        if self._count == 0:
            if timeout is not None and timeout >= 0:
                raise NotImplementedError
            while True:
                pid = self._pidfile.write(
                    PID,
                    invalid='remove',
                    orphaned='remove',
                )
                if pid is not None:
                    break
                if not blocking:
                    return False
        self._count += 1
        return True

    def release(self):
        if self._count == 0:
            # XXX double-check the file?
            raise RuntimeError('lock not held')
        self._count -= 1
        if self._count == 0:
            self._pidfile.remove()


##################################
# logging utils

class LogSection(types.SimpleNamespace):
    """A titled, grouped sequence of log entries."""

    @classmethod
    def read_logfile(cls, logfile):
        # Currently only a "simple" format is supported.
        if isinstance(logfile, str):
            filename = logfile
            with open(filename) as logfile:
                yield from cls.read_logfile(logfile)
                return

        parsed = cls._iter_lines_and_headers(logfile)
        # Ignore everything up to the first header.
        for value in parsed:
            if not isinstance(value, str):
                _, title, _, timestamp = value
                section = cls.from_title(title[2:], timestamp[2:].strip())
                break
        else:
            return
        # Yield a LogSection for each header found.
        for value in parsed:
            if isinstance(value, str):
                section.add_lines(value)
            else:
                yield section
                _, title, _, timestamp = value
                section = cls.from_title(title[2:], timestamp[2:].strip())
        yield section

    @classmethod
    def _iter_lines_and_headers(cls, lines):
        header = None
        for line in lines:
            if line.endswith('\n'):
                # XXX Windows?
                line = line[:-1]
            if header:
                matched = False
                if len(header) == 1:
                    if line.startswith('# ') and line[2:].strip():
                        header.append(line)
                        matched = True
                elif len(header) == 2:
                    if not line:
                        header.append(line)
                        matched = True
                elif re.match(r'^# \d{4}-\d\d-\d\d \d\d:\d\d:\d\d$', line):
                    header.append(line)
                    yield header
                    header = None
                    matched = True
                if not matched:
                    yield from header
                    header = None
            elif line == ('#'*40):
                header = [line]
            else:
                yield line

    @classmethod
    def from_title(cls, title, timestamp=None, **logrecord_kwargs):
        if not title or not title.strip():
            raise ValueError('missing title')
        timestamp, _ = get_utc_datetime(timestamp or None)

        logrecord_kwargs.setdefault('name', None)
        logrecord_kwargs.setdefault('level', None)
        # These could be extrapolated:
        logrecord_kwargs.setdefault('pathname', None)
        logrecord_kwargs.setdefault('lineno', None)
        logrecord_kwargs.setdefault('exc_info', None)
        logrecord_kwargs.setdefault('func', None)

        header = logging.LogRecord(
            msg=title.strip(),
            args=None,
            **logrecord_kwargs,
        )
        header.created = timestamp.timestamp()
        header.msecs = 0
        self = cls(header)
        self._timestamp = timestamp
        return self

    def __init__(self, header):
        if not header:
            raise ValueError('missing header')
        elif not isinstance(header, logging.LogRecord):
            raise TypeError(f'expected logging.LogRecord, got {header!r}')
        super().__init__(
            header=header,
            body=[],
        )

    @property
    def title(self):
        return self.header.getMessage()

    @property
    def timestamp(self):
        try:
            return self._timestamp
        except AttributeError:
            self._timestamp = get_utc_timestamp(self.header.created)
            return self._timestamp

    def add_record(self, record):
        if isinstance(record, str):
            msg = record
            record = logging.LogRecord(
                msg=msg,
                args=None,
                name=self.header.name,
                level=self.header.levelname,
                # XXX Conditionally extrapolate the rest?
                pathname=self.header.pathname,
                lineno=self.header.lineno,
                exc_info=self.header.exc_info,
                func=self.header.funcName,
            )
            record.created = self.header.created
            record.msecs = 0
        elif not isinstance(header, logging.LogRecord):
            raise TypeError(f'expected logging.LogRecord, got {record!r}')
        self.body.append(record)

    def add_lines(self, lines):
        if isinstance(lines, str):
            lines = lines.splitlines()
        for line in lines:
            self.add_record(line)

    def render(self):
        # Currently only a "simple" format is supported.
        yield '#' * 40
        yield f'# {self.title}'
        yield ''
        yield f'# {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}'
        for rec in self.body:
            yield rec.getMessage()


##################################
# git utils

GITHUB_REMOTE_URL = re.compile(r'''
    ^(
        (
            https://github\.com/
            ( [^/]+ )  # <https_org>
            /
            ( .* )  # <https_repo>
            /?
         )
        |
        (
            git@github\.com:
            ( [^/]+ )  # <ssh_org>
            /
            ( .* )  # <ssh_repo>
            \.git
         )
    )$
''', re.VERBOSE)


def parse_github_url(value):
    m = GITHUB_REMOTE_URL.match(value)
    if not m:
        return None
    https_org, https_repo, ssh_org, ssh_repo = m
    return (https_org or ssh_org, https_repo or ssh_repo)


def looks_like_git_commit(value):
    return bool(re.match(r'^[a-fA-F0-9]{4,40}$', value))


def looks_like_git_name(value):
    # check_name() is too strict, even with loose=True.
    return bool(re.match(r'^[\w][\w.-]*$', value))


def looks_like_git_tag(value):
    return looks_like_git_name(value)


def looks_like_git_branch(value):
    return looks_like_git_name(value)


def looks_like_git_remote_url(value):
    # XXX We could accept more values.
    return GITHUB_REMOTE_URL.match(value)


def looks_like_git_remote(value):
    if looks_like_git_remote_url(value):
        return True
    return looks_like_git_name(value)


def looks_like_git_ref(value):
    if not value:
        return False
    elif value == 'latest':
        return True
    elif value == 'HEAD':
        return True
    elif looks_like_git_commit(value):
        return True
    elif looks_like_git_tag(value):
        return True
    elif looks_like_git_branch(value):
        return True
    else:
        return False


def git(cmd, *args, cwd=HOME, cfg=None):
    return _git(cmd, args, cwd, cfg)


def git_raw(cmd, *args, cwd=HOME, cfg=None):
    return _git_raw(cmd, args, cwd, cfg)


def _git(cmd, args, cwd, cfg):
    ec, text = _git_raw(cmd, args, cwd, cfg)
    if ec != 0:
        print(text)
        raise NotImplementedError(ec)
    return text


def _git_raw(cmd, args, cwd, cfg, *, GIT=shutil.which('git')):
    env = dict(os.environ)
    preargs = []
    if cfg:
        for name, value in cfg.items():
            if value is None:
                raise NotImplementedError
            value = str(value)
            preargs.extend(['-c', f'{name}={value}'])
            if name == 'user.name':
                env['GIT_AUTHOR_NAME'] = value
                env['GIT_COMMITTER_NAME'] = value
            elif name == 'user.email':
                env['GIT_AUTHOR_EMAIL'] = value
                env['GIT_COMMITTER_EMAIL'] = value
    proc = run_fg(GIT, *preargs, cmd, *args, cwd=cwd, env=env)
    return proc.returncode, proc.stdout


class GitHubTarget(types.SimpleNamespace):

    @classmethod
    def from_origin(cls, org, project, *, ssh=False):
        return cls(org, project, remote='origin', ssh=ssh)

    @classmethod
    def from_url(cls, url, remote=None, upstream=None):
        m = GITHUB_REMOTE_URL.match(value)
        if not m:
            return None
        https_org, https_repo, ssh_org, ssh_repo = m
        org = https_org or ssh_org
        project = https_repo or ssh_repo
        ssh = bool(ssh_org)
        ref = None
        return cls(org, project, ref, remote, upstream, ssh=ssh)

    @classmethod
    def resolve(cls, remote, reporoot=None):
        if not remote:
            raise ValueError('missing remote')
        if looks_like_git_name(remote):
            if not reporoot:
                raise ValueError('missing reporoot')
            return cls._from_remote_name(remote, reporoot)
        else:
            if reporoot:
                return cls.find(remote, reporoot)
            else:
                return cls.from_url(remote)

    @classmethod
    def from_remote_name(cls, remote, reporoot):
        if not looks_like_git_name(remote):
            raise ValueError(f'invalid remote {name!r}')
        return cls._from_remote_name(remote, reporoot)

    @classmethod
    def find(cls, remote, reporoot, upstream=None):
        if upstream and isinstance(upstream, str):
            upstream = cls.resolve(upstream, reporoot)
        self = cls.from_url(remote, upstream=upstream)
        self.remote = cls._find(remote, reporoot)
        if not self.remote:
            raise Exception(f'no remote matching {remote} found')
        if not upstream:
            self.upstream = cls.from_remote_name('origin', reporoot)
        return self

    @classmethod
    def _from_remote_name(cls, remote, reporoot):
        ec, url = git_raw('remote', 'get-url', remote, cwd=reporoot)
        if ec:
            return None
        if remote == 'origin':
            upstream = None
        else:
            upstream = cls._from_remote_name('origin', reporoot)
        return cls.from_url(url, remote, upstream)

    @classmethod
    def _find(cls, remote, reporoot):
        text = git('remote', '-v', cwd=reporoot)
        for line in text.splitlines():
            if line.endswith(' (fetch)'):
                name, _url, _ = line.split()
                if _url == remote:
                    return name
            elif line.endswith(' (pull)'):
                pass
            else:
                raise NotImplementedError(line)
        else:
            return None

    def __init__(self, org, project, ref=None, remote=None, upstream=None, *,
                 ssh=False,
                 ):
        check_name(org, loose=True)
        check_name(project, loose=True)
        if not ref:
            ref = None
        elif not isinstance(ref, str):
            raise NotImplementedError(ref)
        elif not looks_like_git_ref(ref):
            raise ValueError(ref)
        if not remote:
            remote = None
        elif not isinstance(remote, str):
            raise NotImplementedError(remote)
        elif not looks_like_git_remote(remote):
            raise ValueError(remote)
        if upstream is not None and not isinstance(upstream, GitHubTarget):
            raise TypeError(upstream)
        ssh = bool(ssh)

        kwargs = dict(locals())
        del kwargs['self']
        del kwargs['__class__']
        super().__init__(**kwargs)

    @property
    def remote(self):
        remote = vars(self)['remote']
        if remote:
            return remote
        return self.org if self.upstream else 'upstream'

    remote_name = remote

    @property
    def url(self):
        return f'https://github.com/{self.org}/{self.project}'

    @property
    def push_url(self):
        if self.ssh:
            return f'git@github.com:{self.org}/{self.project}.git'
        else:
            return self.url

    @property
    def archive_url(self):
        ref = self.ref or 'main'
        return f'{self.url}/archive/{self.ref or "main"}.tar.gz'

    @property
    def fullref(self):
        if self.ref:
            if looks_like_git_commit(self.ref):
                return self.ref
            branch = self.ref
        else:
            branch = 'main'
        return f'{self.remote}/{branch}' if self.remote else branch

    @property
    def origin(self):
        remote = vars(self)['remote']
        if remote and remote == 'origin':
            return self
        elif not self.upstream:
            return None  # unknown
        else:
            return self.upstream.origin

    def as_remote_info(self, name=None):
        if not name:
            _name = vars(self)['remote']
            if _name:
                name = _name
            elif name is not None:
                name = self.org
        if name:
            return GitRemoteInfo.from_name(name, self.url, self.push_url)
        else:
            return GitRemoteInfo.from_url(self.url, self.push_url)

    def _resolve_origin_info(self, remotename):
        if remotename:
            if not looks_like_git_name(remotename):
                raise ValueError(remotename)
            copied = self.copy()
            vars(copied)['remote'] = remotename
            remote = copied.as_remote_info()
        else:
            remote = self.as_remote_info()
        return remote if remote.name else remote.as_origin()

    def ensure_local(self, reporoot=None, remotename=None):
        origin = self._resolve_origin_info(remotename)
        if reporoot:
            reporoot = os.path.abspath(reporoot)
        else:
            reporoot = os.path.join(HOME, f'{self.org}-{self.project}')
        repo = GitLocalRepo.ensure(reporoot, origin)
        if origin.name != 'origin':
            repo.remotes.add(remote)
        return repo

    def copy(self, ref=None):
        return type(self)(
            org=self.org,
            project=self.project,
            ref=ref or self.ref,
            remote=vars(self)['remote'],
            upstream=self.upstream,
            ssh=self.ssh,
        )

    def fork(self, org, project=None, ref=None, remote=None):
        return type(self)(
            org=org,
            project=project or self.project,
            ref=ref or self.ref,
            remote=remote,
            upstream=self,
            ssh=self.ssh,
        )

    def as_jsonable(self):
        return dict(vars(self))


class GitRemoteInfo(namedtuple('GitRemoteInfo', 'url name pushurl')):

    @classmethod
    def from_raw(cls, raw, *, fail=None):
        if not raw:
            if fail:
                raise ValueError('missing remote info')
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str) and looks_like_git_remote_url(raw):
            return cls.from_url(raw)
        else:
            if fail or fail is None:
                raise TypeError(raw)
            return None

    @classmethod
    def from_name(cls, name, url, pushurl=True):
        if not name:
            raise ValueError('missing name')
        return cls._from_values(url, name, pushurl)

    @classmethod
    def from_url(cls, url, pushurl=True):
        return cls._from_values(url, None, pushurl)

    @classmethod
    def _from_values(cls, url, name, pushurl):
        if pushurl is True:
            pushurl = url
        return cls(url, name, pushurl)

    @classmethod
    def _validate_name(cls, name):
        if not name:
            return
        if not looks_like_git_name(name):
            raise ValueError(name)

    def __new__(cls, url, name=None, pushurl=None):
        self = super().__new__(
            cls,
            url=url or None,
            name=name or None,
            pushurl=pushurl or None,
        )
        return self

    def __init__(self, *args, **kwargs):
        self._validate()

    def _validate(self):
        if not self.url:
            raise ValueError('missing url')
        elif not looks_like_git_remote_url(self.url):
            raise ValueError(self.url)

        self._validate_name(self.name)

        if self.pushurl != self.url:
            if not self.pushurl:
                raise ValueError('missing pushurl')
            elif not looks_like_git_remote_url(self.pushurl):
                raise ValueError(self.pushurl)

    def __str__(self):
        return self.url

    def change_name(self, name):
        self._validate_name(name)
        return self._replace(name=name)

    def as_origin(self):
        if self.name == 'origin':
            return self
        return self._replace(name='origin')


class GitRemotes:

    def __init__(self, repo):
        _repo = GitLocalRepo.from_raw(repo)
        if not _repo:
            raise TypeError(_repo)
        self._repo = repo

    def _git(self, cmd, *args):
        return self._repo.git('remote', cmd, *args)

    def resolve(self, remote):
        try:
            remote = GitRemoteInfo.from_raw(remote, fail=True)
        except Exception:
            if not remote:
                raise  # re-raise
            GitRemoteInfo.from_raw(remote, fail=False)
            if not isinstance(remote, str) or not looks_like_git_name(remote):
                raise  # re-raise
            return self.get(remote)
        else:
            # XXX Match an existing remote.
            return remote

    def get(self, name):
        if not name:
            raise ValueError('missing name')
        elif not isinstance(name, str):
            raise TypeError(name)
        elif not looks_like_git_name(name):
            raise ValueError(name)
        ec, url = self._repo.git_raw('remote', 'get-url', name)
        if ec != 0:
            url = None
        ec, pushurl = self._repo.git_raw('remote', 'get-url', '--push', name)
        if ec != 0:
            pushurl = None
        if not url or not pushurl:
            return None
        return GitRemoteInfo.from_name(name, url, pushurl)

    def add(self, remote):
        remote = GitRemoteInfo.from_raw(remote, fail=True)
        if not remote.name:
            raise ValueError('missing remote name')
        self._git('add', remote.name, remote.url)
        if not remote.pushurl:
            # XXX Set to a bogus value?
            raise NotImplementedError
        elif remote.pushurl != remote.url:
            self._git('set-url', '--push', remote.name, remote.pushurl)


class GitBranches:

    def __init__(self, repo):
        _repo = GitLocalRepo.from_raw(repo)
        if not _repo:
            raise TypeError(repo)
        self._repo = _repo

    @property
    def current(self):
        return self._repo.git('rev-parse', '--abbrev-ref', 'HEAD')


class GitRepo:

    @classmethod
    def from_raw(cls, raw, *, fail=None):
        if not raw:
            if fail:
                raise ValueError('missing repo')
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls._parse(raw)
        else:
            raise TypeError(raw)

    @classmethod
    def _parse(cls, repostr):
        raise NotImplementedError(repostr)

    def __init__(self, location, cfg=None):
        if not location:
            raise ValueError('missing location')
        self._loc = location
        self._cfg = cfg or None

    def __repr__(self):
        return f'{type(self).__name__}({self._loc!r})'

    def __str__(self):
        return self._loc

    def __eq__(self, other):
        if not isinstance(other, GitRepo):
            return NotImplemented
        return self._loc == other._loc

    def __hash__(self):
        return hash(self._loc)

    def _verify(self):
        raise NotImplementedError

    @property
    def location(self):
        return self._loc

    @property
    def branches(self):
        try:
            return self._branches
        except AttributeError:
            self._branches = GitBranches(self)
            return self._branches

    @property
    def exists(self):
        raise NotImplementedError

    def git(self, cmd, *args, cfg=None):
        if cfg is None:
            cfg = self._cfg
        return self._git(cmd, args, cfg)

    def git_raw(self, cmd, *args, cfg=None):
        if cfg is None:
            cfg = self._cfg
        return self._git_raw(cmd, args, cfg)

    def _git(self, cmd, args, cfg):
        raise NotImplementedError

    def _git_raw(self, cmd, args, cfg):
        raise NotImplementedError

    def verify(self):
        if not self.exists:
            raise ValueError(f'repo missing ({self._loc})')
        self._verify()

    def resolve(self, *relpath):
        raise NotImplementedError


class GitLocalRepo(GitRepo):

    @classmethod
    def ensure(cls, root, origin=None):
        origin = GitRemoteInfo.from_raw(origin)
        self = cls(root)
        if self.exists:
            self._verify()
            if origin:
                remote = origin
                origin = origin.as_origin()
                if not self.remotes.resolve(origin):
                    raise ValueError(f'missing/mismatched origin {origin!r}')
                self.git('fetch', '--tags')
                if remote.name and remote.name != 'origin':
                    self.git('fetch', '--tags', remote.name)
            else:
                self.git('fetch', '--tags')
        elif not origin:
            # XXX init the repo?
            raise ValueError('missing origin')
        else:
            self._clone(origin.url)
            if origin.pushurl != origin.url:
                self.remotes.add(origin)
            self.git('fetch', '--tags')
        return self

    @classmethod
    def _parse(cls, repostr):
        # XXX Make sure it looks like a dirname.
        return cls(raw)

    def __init__(self, root, cfg=None):
        super().__init__(root, cfg)
        if not os.path.isabs(root):
            raise ValueError(f'expected absolute root, got {root!r}')
        self._root = root

    def _verify(self):
        # XXX Check .git, etc.
        ...

    def _clone(self, origin):
        git('clone', str(origin), self._root)

    @property
    def root(self):
        return self._root

    @property
    def remotes(self):
        try:
            return self._remotes
        except AttributeError:
            self._remotes = GitRemotes(self)
            return self._remotes

    @property
    def branches(self):
        try:
            return self._branches
        except AttributeError:
            self._branches = GitBranches(self)
            return self._branches

    @property
    def exists(self):
        return os.path.exists(self._root)

    def _git(self, cmd, args, cfg):
        text = _git(cmd, args, self._root, cfg)
        return text.strip()

    def _git_raw(self, cmd, args, cfg):
        ec, text = _git_raw(cmd, args, self._root, cfg)
        return ec, text.strip()

    def resolve(self, *relpath):
        return os.path.join(self._root, *relpath)

    def relpath(self, filename):
        relfile = os.path.relpath(filename, self._root)
        if relfile.startswith(f'..{os.path.sep}') or os.path.isabs(relfile):
            raise NotImplementedError(filename)
        return relfile

    def fetch(self, remote=None):
        if remote:
            remote = self.remotes.resolve(remote)
            self.git('fetch', '--tags', str(remote))
        else:
            self.git('fetch', '--tags')

    def pull(self, remote=None):
        if remote:
            remote = self.remotes.resolve(remote)
            self.git('pull', str(remote))
        else:
            self.git('pull')

    def is_clean(self):
        ec, _ = self.git_raw('diff-index', '--quiet', 'HEAD')
        return True if ec == 0 else False

    def clean(self):
        self.git_raw('reset', '--hard', 'HEAD')
        self.git_raw('clean', '-d', '--force')

    def refresh(self, remote='origin'):
        self.git('fetch', '--tags')
        if remote:
            remote = self.remotes.resolve(remote)
            if remote.name != 'origin':
                self.git('fetch', '--tags', str(remote))
            branch = self.branches.current
            self.git('reset', '--hard', f'{remote.name}/{branch}')
        else:
            self.clean()

    def checkout(self, ref='HEAD'):
        if not ref:
            raise ValueError('missing ref')
        elif not isinstance(ref, str):
            raise NotImplementedError(ref)
        if not self.is_clean():
            raise NotImplementedError
        self.git('checkout', ref)

    def switch_branch(self, branch, base='main'):
        if not branch:
            raise ValueError('missing branch')
        elif not isinstance(branch, str):
            raise NotImplementedError(branch)
        self.checkout(base)
        self.git('checkout', '-B', branch)

    def add(self, filename, *others):
        relfile = self.relpath(filename)
        relothers = (self.relpath(f) for f in others)
        self.git('add', relfile, *relothers)

    def commit(self, msg):
        self.git('commit', '-m', msg)

    def push(self, remote=None):
        if not remote:
            self.git('push')
        else:
            if not isinstance(remote, str):
                raise NotImplementedError(remote)
            elif not looks_like_git_remote(remote):
                raise ValueError(remote)
            self.git('push', remote)

    def swap_config(self, cfg):
        cls = type(self)
        return cls(self._root, cfg)

    def using_author(self, author=None):
        cfg = {}
        if not author:
            pass
        elif isinstance(author, str):
            parsed = parse_email_address(author)
            if not parsed:
                raise ValueError(f'invalid author {author!r}')
            name, email = parsed
            if not name:
                name = '???'
                author = f'??? <{author}>'
            cfg['user.name'] = name
            cfg['user.email'] = email
        else:
            raise NotImplementedError(author)
        return self.swap_config(cfg)


class GitRefRequest(namedtuple('GitRefRequest', 'remote branch revision')):

    #__slots__ = ['_orig']

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            raise NotImplementedError(raw)
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            raise NotImplementedError(raw)
        elif hasattr(raw, 'items'):
            orig = raw.pop('orig', None)
            self = cls(**raw)
            if orig:
                self._orig = GitRefRequest.from_raw(orig)
            return self
        else:
            return cls(*raw)

    def __new__(cls, remote, branch, revision):
        if not revision:
            revision = None
        elif revision == 'latest':
            if not branch:
                raise ValueError('missing branch')
        elif revision.upper() == 'HEAD':
            if not branch:
                raise ValueError('missing branch')
            revision = 'HEAD'
        elif not looks_like_git_ref(revision):
            raise ValueError(f'unsupported revision {revision!r}')

        if not branch:
            branch = None
        elif not looks_like_git_branch(branch):
            raise ValueError(f'unsupported branch {branch!r}')

        if not remote:
            remote = None
        elif not looks_like_git_remote(remote):
            raise ValueError(f'unsupported remote {remote!r}')

        return super().__new__(cls, remote, branch, revision)

    @property
    def orig(self):
        return getattr(self, '_orig', None)

    def as_jsonable(self):
        data = self._asdict()
        if self.orig:
            data['orig'] = self.orig.as_jsonable()
        return data


class GitRefCandidates:

    @classmethod
    def from_revision(cls, revision, branch, remote, *, parse_version=None):
        if parse_version is None:
            parse_version = Version.parse
        orig = GitRefRequest(remote, branch, revision)
        reqs = cls._from_revision(orig.revision, orig.branch, orig.remote,
                                  parse_version)
        self = cls(parse_version=parse_version)
        for req in reqs:
            self._add_req(req, orig)
        return self

    @classmethod
    def _from_revision(cls, revision, branch, remote, parse_version):
        if branch == 'main':
            reqs = cls._from_main(revision, remote)
        elif remote == 'origin':
            reqs = cls._from_origin(revision, branch, parse_version)
        elif remote:
            reqs = cls._from_non_origin(revision, branch, remote, parse_version)
        elif branch:  # no remote, maybe a revision
            # We know all remotes have release (version) branches.
            reqs = cls._from_version(revision, branch, 'origin',
                                     parse_version, required=False)
            if not reqs:
                # We don't bother trying to guess the remote at this point.
                raise ValueError('missing remote for branch {branch!r}')
        elif revision:  # no remote or branch
            reqs = cls._from_origin(revision, branch,
                                    parse_version, required=False)
            if not reqs:
                raise ValueError(f'missing remote for revision {revision!r}')
        else:
            reqs = cls._from_main(revision, remote)
        return reqs

    @classmethod
    def _from_main(cls, revision, remote):
        # The main branch defaults to origin.
        # We do not support tags for the main branch.
        if not remote:
            remote = 'origin'
        if not revision:
            return [(remote, 'main', 'HEAD')]
        elif revision in ('latest', 'HEAD'):
            return [(remote, 'main', 'HEAD')]
        elif looks_like_git_commit(revision):
            return [(remote, 'main', revision)]
        else:
            raise ValueError(f'unexpected revision {revision!r}')

    @classmethod
    def _from_origin(cls, revision, branch, parse_version, required=True):
        if branch == 'main':
            return cls._from_main(revision, 'origin')
        elif branch:
            return cls._from_version(revision, branch, 'origin', parse_version)
        elif revision == 'main':
            return cls._from_main(None, 'origin')
        elif revision in ('latest', 'HEAD'):
            return cls._from_main(revision, 'origin')
        elif revision:
            if looks_like_git_commit(revision):
                return [('origin', None, revision)]
            else:
                # The only remaining possibility for origin
                # is a release branch or tag.
                return cls._from_version(revision, branch, 'origin',
                                         parse_version, required)
        else:
            return cls._from_main(revision, 'origin')

    @classmethod
    def _from_non_origin(cls, revision, branch, remote, parse_version):
        if branch:
            if not revision:
                # For non-origin, we don't bother with "latest" for versions.
                return [(remote, branch, 'HEAD')]
            elif parse_version(branch, match=revision):
                return [(remote, branch, revision)]
            else:
                if revision == 'latest':
                    revision = 'HEAD'
                # For non-origin, revision can be any tag or commit.
                return [(remote, branch, revision)]
        else:
            if not revision:
                # Unlike for origin, here we don't assume "main".
                raise ValueError('missing revision')
            elif revision in ('latest', 'HEAD'):
                raise ValueError('missing branch')
            elif revision == 'main':
                return cls._from_main(None, remote)
            elif looks_like_git_commit(revision):
                return [(remote, None, revision)]
            else:
                reqs = cls._from_version(revision, None, remote,
                                         parse_version, required=False)
                if reqs:
                    return reqs
                if looks_like_git_branch(revision):
                    return [
                        (remote, None, revision),
                        (remote, revision, 'HEAD'),
                    ]
                else:
                    raise ValueError(f'unexpected revision {revision!r}')

    @classmethod
    def _from_version(cls, revision, branch, remote,
                      parse_version, required=True):
        if not remote:
            remote = 'origin'
        if branch:
            version = parse_version(branch)
            if not version:
                if required:
                    raise ValueError(f'unexpected branch {branch!r}')
                return None
            verstr = f'{version.major}.{version.minor}'
            if verstr != branch:
                if required:
                    raise ValueError(f'unexpected branch {branch!r}')
                return None
            if not revision or revision == 'latest':
                return [(remote, branch, 'latest')]
            elif revision == 'HEAD':
                return [(remote, branch, revision)]
            elif looks_like_git_commit(revision):
                return [(remote, branch, revision)]
            else:
                tagver = parse_version(revision)
                if not tagver:
                    raise ValueError(f'unexpected revision {revision!r}')
                if tagver[:2] != version[:2]:
                    raise ValueError(f'tag {revision!r} does not match branch {branch!r}')
                return [(remote, branch, revision)]
        else:
            tagver = parse_version(revision)
            if not tagver:
                if required:
                    raise ValueError(f'unexpected revision {revision!r}')
                return None
            verstr = f'{tagver.major}.{tagver.minor}'
            return [
                (remote, None, revision),
                (remote, verstr, 'latest' if revision == verstr else revision),
            ]

    def __init__(self, reqs=None, *, parse_version=None):
        if parse_version is None:
            parse_version = Version.parse
        self._parse_version = parse_version
        self._reqs = []
        for req in reqs or ():
            self._add_req(req)

    def __repr__(self):
        return f'{type(self).__name__}({self._reqs})'

    def __eq__(self, other):
        raise NotImplementedError

    def __len__(self):
        return len(self._reqs)

    def __iter__(self):
        yield from self._reqs

    def __getitem__(self, index):
        return self._reqs[index]

    def _add_req(self, req, orig=None):
        req = GitRefRequest.from_raw(req)
        if orig:
            if req.orig:
                raise NotImplementedError((orig, req.orig))
            req._orig = GitRefRequest.from_raw(orig)
        self._reqs.append(req)

    def find_ref(self, new_ref=None):
        if new_ref is None:
            new_ref = GitRef
        by_remote = {}
        for req in self._reqs:
            remote, branch, revision = req
            if remote not in by_remote:
                by_remote[remote] = CPythonGitRefs.from_remote(remote)
            repo_refs = by_remote[remote]

            match = self._find_ref(branch, revision, repo_refs)
            if match:
                _branch, tag, commit, name = match
                if branch and _branch != branch:
                    logger.warning(f'branch mismatch (wanted {branch}, found {_branch})')
                else:
                    assert commit, (name, commit, branch or _branch, remote)
                    return new_ref(remote, _branch, tag, commit, name, req)
        else:
            return None

    def _find_ref(self, branch, revision, repo_refs):
        _branch = tag = commit = name = None
        if revision == 'HEAD':
            assert branch, self
            matched = repo_refs.match_branch(branch)
            if matched:
                _branch, tag, commit = matched
                assert _branch == branch, (branch, _branch)
                name = _branch
            else:
                logger.warning(f'branch {branch} not found')
        elif revision == 'latest':
            assert branch, self
            assert self._parse_version(branch), (branch, self)
            matched = repo_refs.match_latest_version(branch)
            if matched:
                _branch, tag, commit = matched
                name = tag or _branch
                assert name, (matched, self)
            elif repo_refs.match_branch(branch):
                logger.warning(f'latest tag for branch {branch} not found')
            else:
                logger.warning(f'branch {branch} not found')
        elif looks_like_git_commit(revision):
            matched = repo_refs.match_commit(revision)
            if matched:
                _branch, tag, commit = matched
                assert not branch or _branch == branch, (branch, _branch, self)
            else:
                if branch:
                    if not repo_refs.match_branch(branch):
                        logger.warning(f'branch {branch} not found')
                commit = revision
            name = None
        else:
            # The other cases cover branches, so we only consider tags here.
            assert looks_like_git_tag(revision), (revision, self)
            matched = repo_refs.match_tag(revision)
            if matched:
                _branch, tag, commit = matched
                assert tag, (matched, self)
                name = tag
        if not commit:
            return None
        return _branch, tag, commit, name


ToGitRefType = Union[
    "GitRef",
    str,
    Dict[str, str],
    Tuple[str, str, str, str, str, str]
]


class GitRef(namedtuple('GitRef', 'remote branch tag commit name requested')):

    # XXX Use requested.orig.revision instead of requested.revision?

    KINDS = {
        'commit',
        'tag',
        'branch',
    }

    @classmethod
    def from_raw(cls, raw: ToGitRefType):
        if isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            raise TypeError(raw)
        elif isinstance(raw, dict):
            if raw.get('requested'):
                raw['requested'] = GitRefRequest.from_raw(raw['requested'])
            return cls(**raw)
        else:
            remote, branch, tag, commit, name, requested = raw
            requested = GitRefRequest.from_raw(requested) if requested else None
            return cls(remote, branch, tag, commit, name, requested)

    @classmethod
    def resolve(cls, revision, branch, remote, *, parse_version=None):
        if parse_version is None:
            parse_version = Version.parse
        candidates = GitRefCandidates.from_revision(revision, branch, remote,
                                                    parse_version=parse_version)
        return candidates.find_ref(cls)

    @classmethod
    def from_values(cls, remote, branch, tag, commit,
                    name=None, requested=None, kind=None):
        if not remote:
            # XXX Infer "origin" if branch is main or looks like a version?
            raise ValueError('missing remote')
        elif not looks_like_git_remote(remote):
            raise ValueError(f'invalid remote {remote!r}')
        if not branch:
            branch = None
        elif not looks_like_git_branch(branch):
            raise ValueError(f'invalid branch {branch!r}')
        if not tag:
            tag = None
        elif not looks_like_git_tag(tag):
            raise ValueError(f'invalid tag {tag!r}')
        if not commit:
            if looks_like_git_commit(name):
                commit = name
                name = None
            else:
                raise ValueError('missing commit')
        elif not looks_like_git_commit(commit):
            raise ValueError(f'invalid commit {commit!r}')
        if not name:
            name = None
        # We don't both validating name.
        requested = GitRefRequest.from_raw(requested) if requested else None

        if kind == 'branch':
            if not branch:
                if not name or not looks_like_git_branch(name):
                    raise ValueError('missing branch')
                branch = name
            elif not name:
                name = branch
            elif name != branch:
                raise ValueError(f'branch mismatch ({name!r} != {branch!r})')
        elif kind == 'tag':
            if not tag:
                if not name or not looks_like_git_tag(name):
                    raise ValueError('missing tag')
                tag = name
            elif not name:
                name = tag
            elif name != tag:
                raise ValueError(f'tag mismatch ({name!r} != {tag!r})')
        elif kind == 'commit':
            assert commit
            if name:
                if name != commit:
                    raise ValueError(f'commit mismatch ({name} != {commit})')
                name = None
        elif kind:
            raise NotImplementedError(kind)
        else:
            if not name:
                if tag:
                    name = tag
                elif branch:
                    name = branch
                elif not commit:
                    raise ValueError('missing commit')
            elif name == 'HEAD':
                # Tags and HEAD don't mix.
                if branch:
                    name = branch
                else:
                    raise ValueError('missing branch')
            elif looks_like_git_commit(name):
                if not commit:
                    commit = name
                elif name != commit:
                    raise ValueError(f'commit mismatch ({name} != {commit})')
                name = None
            elif name == branch:
                pass
            elif name == tag:
                pass
            elif branch and not tag and looks_like_git_tag(name):
                tag = name
            else:
                if not branch and not tag:
                    raise ValueError('missing branch')
                raise ValueError(f'unsupported name {name!r}')

        return cls.__new__(cls, remote, branch, tag, commit, name, requested)

    def __init__(self, *args, **kwargs):
        self._validate()

    def __str__(self):
        return self.name or self.commit

    def _validate(self):
        if not self.remote:
            raise ValueError('missing remote')
        elif not looks_like_git_branch(self.remote):
            raise ValueError(f'invalid remote {self.remote!r}')
        if self.branch and not looks_like_git_branch(self.branch):
            raise ValueError(f'invalid branch {self.branch!r}')
        if self.tag and not looks_like_git_tag(self.tag):
            raise ValueError(f'invalid tag {self.tag!r}')
        if not self.commit:
            raise ValueError('missing commit')
        elif not looks_like_git_commit(self.commit):
            raise ValueError(f'invalid commit {self.commit!r}')

        if self.name:
            if self.name not in (self.branch, self.tag):
                raise ValueError(f'unsupported name {self.name!r}')
        elif not self.commit:
            raise ValueError('missing name')

        if self.requested:
            if not isinstance(self.requested, GitRefRequest):
                raise TypeError(self.requested)
            # XXX Compare self.requested to the other fields?

    @property
    def kind(self):
        if not self.name:
            return 'commit'
        elif self.name == self.branch:
            return 'branch'
        elif self.name == self.tag:
            return 'tag'
        else:
            raise NotImplementedError(self.name)

    @property
    def full(self):
        ref = self.name
        if ref:
            if ref == self.branch:
                if self.commit:
                    ref = f'{ref} ({self.commit[:8]})'
        elif self.commit:
            ref = self.commit
        else:
            ref = '???'
        if ref == self.commit:
            branch = self.branch if self.branch != 'main' else None
            # XXX Is this an okay shortening?
            ref = f'{branch} ({ref[:8]})' if branch else ref[:12]
        if self.remote and self.remote != 'origin':
            return f'{self.remote}:{ref}'
        else:
            return ref

    def as_jsonable(self):
        data = self._asdict()
        if self.requested:
            data['requested'] = self.requested.as_jsonable()
        return data


class CPythonGitRefs(types.SimpleNamespace):

    @classmethod
    def from_remote(cls, remote):
        if remote == 'origin' or not remote:
            url = 'https://github.com/python/cpython'
        elif remote == 'upstream':
            url = 'https://github.com/faster-cpython/cpython'
        else:
            url = f'https://github.com/{remote}/cpython'
        return cls.from_url(url)

    @classmethod
    def from_url(cls, url):
        ec, text = git_raw('ls-remote', '--refs', '--tags', '--heads', url)
        if ec != 0:
            if text.strip():
                for line in text.splitlines():
                    logger.debug(line)
            else:
                logger.debug('(no output)')
            return None, None, None
        return cls._parse_ls_remote(text.splitlines())

    @classmethod
    def _parse_ls_remote(cls, lines):
        branches = {}
        tags = {}
        for line in lines:
            m = re.match(r'^([a-zA-Z0-9]+)\s+refs/(heads|tags)/(\S.*)$', line)
            if not m:
                continue
            commit, kind, name = m.groups()
            if kind == 'heads':
                group = branches
            elif kind == 'tags':
                group = tags
            else:
                raise NotImplementedError(kind)
            group[name] = commit
        return cls(branches, tags)

    def __init__(self, branches, tags):
        super().__init__(
            branches=branches,
            tags=tags,
        )

    @property
    def _impl(self):
        return CPython()

    def _release_branches(self):
        by_version = {}
        for branch in self.branches:
            version = self._impl.parse_version(branch)
            if version and branch == f'{version.major}.{version.minor}':
                by_version[version] = branch
        return by_version

    def _next_versions(self):
        releases = self._release_branches()
        if not releases:
            return ()
        latest, _ = sorted(releases.items())[-1]
        return [
            latest.next_minor(),
            latest.next_major(),
        ]

    def match_ref(self, ref):
        assert ref
        if looks_like_git_commit(ref):
            return self.match_commit(ref)
        else:
            matched = self.match_tag(ref)
            if matched:
                return matched
            return self.match_branch(ref)

    def match_branch(self, ref):
        if not ref or not looks_like_git_branch(ref):
            return None
        if ref in self.branches:
            branch = ref
        else:
            # Treat it like main if one higher than the latest.
            branch = None
            version = self._impl.parse_version(ref)
            if version and ref == f'{version.major}.{version.minor}':
                if version in self._next_versions():
                    branch = 'main'
        if branch:
            commit = self.branches.get(branch)
            if commit:
                # It might match a tag too.
                for tag, actual in self.tags.items():
                    assert actual
                    if actual == commit:
                        return branch, tag, commit
                else:
                    return branch, None, commit
        return None

    def match_tag(self, ref):
        if not ref or not looks_like_git_branch(ref):
            return None
        version = self._impl.parse_version(ref)
        if version:
            # Find a tag that matches the version.
            for tag in self.tags:
                tagver = self._impl.parse_version(tag)
                if tagver:
                    if tagver == version:
                        commit = self.tags[tag]
                        branch = f'{version.major}.{version.minor}'
                        if branch not in self.branches:
                            branch = None
                        #tag = version.as_tag()
                        assert commit
                        return branch, tag, commit
        else:
            # Find a tag that matches exactly.
            if ref in self.tags:
                commit = self.tags[ref]
                assert commit
                # It might also match a branch.
                for branch, actual in self.branches.items():
                    assert actual
                    if actual == commit:
                        return branch, ref, actual
                else:
                    return None, ref, commit
        # No tags matched!
        return None

    def match_commit(self, commit):
        for tag, actual in self.tags.items():
            assert actual
            if actual == commit:
                return self.match_tag(tag)
        for branch, actual in self.branches.items():
            assert actual
            if actual == commit:
                return branch, None, actual
        return None

    def match_latest_version(self, branch):
        if not branch or not looks_like_git_branch(branch):
            return None
        version = CPython.version.parse(branch)
        if version:
            # Find the latest tag that matches the branch.
            matched = {}
            for tag in self.tags:
                tagver = self._impl.parse_version(tag)
                if version.match(tagver):
                    matched[(tagver.full, tag)] = (self.tags[tag], tagver)
            if matched:
                key = sorted(matched)[-1]
                commit, tagver = matched[key]
                _, tag = key
                #tag = tagver.as_tag()
                assert commit
                return branch, tag, commit
        # Fall back to the branch.
        return self.match_branch(branch)


##################################
# config

class Config(types.SimpleNamespace):
    """The base config for the benchmarking machinery."""

    # XXX Get FIELDS from the __init__() signature?
    FIELDS: List[str] = []
    OPTIONAL: List[str] = []

    FILE = 'benchmarking.json'

    @classmethod
    def load(cls, filename, *, preserveorig=True):
        if os.path.isdir(filename):
            if not cls.FILE:
                raise AttributeError(f'missing {cls.__name__}.FILE')
            filename = os.path.join(filename, cls.FILE)

        with open(filename) as infile:
            data = json.load(infile)
        if preserveorig:
            loaded = dict(data, _filename=filename)

        includes = data.pop('include', None) or ()
        if includes:
            includes = list(cls._load_includes(includes, set()))
            for field in cls.FIELDS:
                if data.get(field):
                    continue
                if field in cls.OPTIONAL:
                    continue
                for included in includes:
                    value = included.get(field)
                    if value:
                        data[field] = value
                        break

        self = cls(**data)
        self._filename = os.path.abspath(os.path.expanduser(filename))
        if preserveorig:
            self._loaded = loaded
            self._includes = includes
        return self

    @classmethod
    def _load_includes(cls, includes, seen):
        if isinstance(includes, str):
            includes = [includes]
        for i, filename in enumerate(includes):
            if not filename:
                continue
            filename = os.path.abspath(os.path.expanduser(filename))
            if filename in seen:
                continue
            logger.debug('# including config from %s', filename)
            seen.add(filename)
            text = read_file(filename, fail=False)
            if not text:
                if not os.path.exists(filename):
                    logger.debug('# (not found)')
                else:
                    logger.debug('# (empty or could not be read)')
                continue
            included = json.loads(text)
            included['_filename'] = filename
            yield included

            subincludes = included.pop('include', ())
            if subincludes:
                yield from cls._load_includes(subincludes, seen)

    @classmethod
    def from_jsonable(cls, data):
        return cls(**data)

    def __init__(self, **kwargs):
        for name in list(kwargs):
            value = kwargs[name]
            if not value:
                if name in self.OPTIONAL:
                    del kwargs[name]
        super().__init__(**kwargs)

    def __str__(self):
        if not self.filename:
            return super().__str__()
        return self.filename or super().__str__()

    @property
    def filename(self):
        try:
            return self._filename
        except AttributeError:
            return None

    def as_jsonable(self, *, withmissingoptional=True):
        # XXX Hide sensitive data?
        data = {k: as_jsonable(v)
                for k, v in vars(self).items()
                if not k.startswith('_')}
        if withmissingoptional:
            for name in self.OPTIONAL:
                if name not in data:
                    data[name] = None
        return data

    def render(self):
        data = self.as_jsonable()
        text = json.dumps(data, indent=4)
        yield from text.splitlines()


class TopConfig(Config):

    CONFIG_DIRS = [
        f'{HOME}/.config',
        HOME,
    ]

    @classmethod
    def find_config(cls, cfgdirs=None):
        if not cfgdirs:
            cfgdirs = cls.CONFIG_DIRS
            if not cfgdirs:
                raise ValueError('missing cfgdirs')
        elif isinstance(cfgdirs, str):
            cfgdirs = [cfgdirs]
        if not cls.FILE:
            raise AttributeError(f'missing {cls.__name__}.FILE')
        for dirname in cfgdirs:
            filename = f'{dirname}/{cls.FILE}'
            if os.path.exists(filename):
                if os.path.isdir(filename):
                    logger.warning(f'expected file, found {filename}')
                else:
                    return filename
        else:
            raise FileNotFoundError('could not find config file')

    @classmethod
    def load(cls, filename=None, **kwargs):
        if not filename:
            filename = cls.find_config()
        return super().load(filename, **kwargs)


##################################
# host info

class HostInfo(namedtuple('HostInfo', 'id name dnsname cpu platform')):

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            raise ValueError('missing host info')
        elif isinstance(raw, cls):
            return raw
        else:
            raise TypeError(raw)

    @classmethod
    def from_metadata(cls, hostid, hostname, dnsname, platform,
                      cpu_model_name, cpu_config,
                      cpu_frequency=None, cpu_count=None, cpu_affinity=False):
        platform = PlatformInfo.parse(platform)
        cpu = CPUInfo.from_metadata(cpu_model_name, platform, cpu_config,
                                    cpu_frequency, cpu_count, cpu_affinity)
        (hostid, hostname, dnsname,
         ) = cls._normalize_ids(hostid, hostname, dnsname, platform, cpu)
        return cls.__new__(cls, hostid, hostname, dnsname, cpu, platform)

    @classmethod
    def _normalize_ids(cls, hostid, hostname, dnsname, platform, cpu):
        validate_str(hostname, 'hostname')

        if hostid:
            validate_str(hostid, 'hostid')
        else:
            for hostid in [hostname, dnsname]:
                if hostid and cls._is_good_hostid(hostid):
                    break
            else:
                # We fall back to a single unique-enough ID.
                # We could use additional identifying information,
                # but it doesn't buy us much (and makes the ID longer).
                hostid = platform.os_name
                if not hostid:
                    raise NotImplementedError
                if cpu.arch in ('arm32', 'arm64'):
                    hostid += '-arm'
                # XXX
                #if not cls._is_good_hostid(hostid):
                #    raise ValueError('missing hostid')

        if dnsname:
            validate_str(dnsname, 'dnsname')
        elif re.match(rf'^{DOMAIN_NAME}$', hostname):
            dnsname = hostname

        return hostid, hostname, dnsname

    @classmethod
    def _is_good_hostid(cls, hostid):
        # XXX What makes a good one?
        return False

    def __new__(cls, id, name, dnsname, cpu, platform):
        self = super().__new__(
            cls,
            id=id or None,
            name=name or None,
            dnsname=dnsname or None,
            cpu=cpu or None,
            platform=platform or None,
        )
        return self

    def __init__(self, *args, **kwargs):
        self._validate(self)

    def _validate(self):
        validate_str(self.id, 'id')
        validate_str(self.name, 'name')
        validate_str(self.dnsname, 'dnsname', required=False)
        if self.dnsname and not re.match(rf'^{DOMAIN_NAME}$', self.dnsname):
            raise ValueError(f'invalid dnsname {self.dnsname!r}')
        validate_arg(self.cpu, CPUInfo, 'cpu')
        validate_arg(self.platform, PlatformInfo, 'platform')

    def __str__(self):
        return self.name

    @property
    def os_name(self):
        try:
            return self._os_name
        except AttributeError:
            os_name = self.platform.os_name
            if not os_name:
                raise NotImplementedError
            self._os_name = name
            return name

    def as_metadata(self):
        metadata = {
            'hostid': self.id,
            'hostname': self.name,
            'platform': str(self.platform),
            **self.cpu.as_metadata(),
        }
        if self.dnsname:
            metadata['dnsname'] = self.dnsname
        return metadata


class PlatformInfo(namedtuple('PlatformInfo', 'kernel')):

    # "Linux-5.4.0-91-generic-x86_64-with-glibc2.31"
    KERNEL = r'''
        (?:
            (?:
                ( Linux )  # <name>
                -
                ( \d+\.\d+\.\d+ )  # <version>
                -
                ( \d+ -
                    (?:
                        generic
                     )
                    -
                    (
                        # 64-bit
                        x86_64 | amd64 | aarch64 | arm64
                        |
                        # 32-bit
                        x86 | arm32
                     )  # <arch>
                    -
                    (?:
                        .*
                     )  # opts
                 )  # <build>
             )
         )
        '''
    REGEX = re.compile(rf'^({KERNEL})$', re.VERBOSE)

    @classmethod
    def from_raw(cls, raw, *, fail=None):
        if not raw:
            if fail:
                raise ValueError('missing platform info')
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        else:
            if fail or fail is None:
                raise TypeError(raw)
        return None

    @classmethod
    def parse(cls, platstr):
        m = cls.REGEX.match(platstr)
        if not m:
            return None
        (kernel,
         *kernel_parts
         ) = m.groups()
        self = cls.__new__(cls, kernel)
        self._handle_parsed_kernel(
            ((p.lower() if p else p) for p in kernel_parts)
        )
        self._raw = platstr
        return self

    def __init__(self, *args, **kwargs):
        self._parse_kernel()
        self._validate()

    def _validate(self):
        pass

    def __str__(self):
        try:
            return self._raw
        except AttributeError:
            self._raw = f'{self.kernel}'
            return self._raw

    def _parse_kernel(self):
        kernelstr = self.kernel.lower()
        m = re.match(rf'^({KERNEL})$', kernelstr, re.VERBOSE)
        if not m:
            raise ValueError(f'unsupported kernel str{self.kernel!r}')
        self._handle_parsed_kernel(m.groups())

    def _handle_parsed_kernel(self, parts):
        (linux_name, linux_version, linux_build, linux_arch,
         ) = parts
        if linux_name:
            (name, version, build, arch,
             ) = linux_name, linux_version, linux_build, linux_arch,
        else:
            raise NotImplementedError(self.kernel)
        version = Version.parse(version)
        self._kernel_name = name
        self._kernel_version = version
        self._kernel_build = build
        self._kernel_arch = arch

    @property
    def kernel_name(self):
        try:
            return self._kernel_name
        except AttributeError:
            self._parse_kernel()
            return self._kernel_name

    @property
    def kernel_version(self):
        try:
            return self._kernel_version
        except AttributeError:
            self._parse_kernel()
            return self._kernel_version

    @property
    def kernel_build(self):
        try:
            return self._kernel_build
        except AttributeError:
            self._parse_kernel()
            return self._kernel_build

    @property
    def kernel_arch(self):
        try:
            return self._kernel_arch
        except AttributeError:
            self._parse_kernel()
            return self._kernel_arch

    @property
    def os_name(self):
        platform = str(self).lower()
        if 'linux' in platform:
            return 'linux'
        elif 'darwin' in platform or 'macos' in platform or 'osx' in platform:
            return 'mac'
        elif 'win' in platform:
            return 'windows'
        else:
            return None


class CPUInfo(namedtuple('CPUInfo', 'model arch cores')):
    # model: "Intel(R) Xeon(R) W-2255 CPU @ 3.70GHz"

    @classmethod
    def from_metadata(cls, model, platform, config,
                      frequency=None, count=None, affinity=None):
        validate_str(model, 'model')
        arch = cls._arch_from_metadata(platform, model)
        cores = cls._cores_from_metadata(count, config, frequency, affinity)
        return cls.__new__(cls, model, arch, cores)

    @classmethod
    def _arch_from_metadata(cls, platform, model):
        platform = PlatformInfo.from_raw(platform)
        arch = platform.kernel_arch if platform else None
        if not arch:
            model = model.lower()
            if 'aarch64' in model:
                arch = 'arm64'
            elif 'arm' in model:
                if '64' in model:
                    arch = 'arm64'
                else:
                    arch = 'arm32'
            elif 'intel' in model:
                arch = 'x86_64'
            else:
                raise NotImplementedError((platform, model))
        return arch

    @classmethod
    def _cores_from_metadata(cls, count, configs, frequencies, affinities):
        count = max(0, int(count)) if count else 0

        cores_by_id = {}
        def add_core(coreid):
            core = cores_by_id[coreid] = {
                'id': coreid,
                'config': [],
                'frequency': None,
                'affinity': None,
            }
            return core
        if count:
            for i in range(count):
                add_core(i)
            def ensure_core(coreid):
                if coreid >= count:
                    raise ValueError(f'expected {count} cores, got {coreid+1}')
                return cores_by_id[coreid]
        else:
            def ensure_core(coreid):
                if coreid < count:
                    return cores_by_id[coreid]
                count = coreid + 1
                for i in range(count, coreid):
                    add_core(i)
                return add_core(coreid)

        parsed = cls._parse_grouped_core_data(configs)
        for coreid, configstr in parsed.items():
            core = ensure_core(coreid)
            core['config'] = CPUConfig.parse(configstr)

        if frequencies:
            parsed = cls._parse_grouped_core_data(frequencies)
            for coreid, freq in parsed.items():
                core = ensure_core(coreid)
                core['frequency'] = cls._normalize_frequency(freq)

        if affinities:
            for coreid in cls._parse_core_ids(affinities):
                core = ensure_core(coreid)
                core['affinity'] = True

        if not count:
            raise ValueError('missing count')
        return (CPUCoreInfo(**cores_by_id[i]) for i in range(count))

    @classmethod
    def _parse_grouped_core_data(cls, datastr):
        # "0=...; 1,11,14=..."
        data_per_core = {}
        for corepart in datastr.split(';'):
            m = re.match(r'^\s*(?:(\d+(?:,\d+(?:-\d+)?)*)=)?(.*?)\s*$', corepart)
            if not m:
                raise ValueError(f'invalid core part {corepart!r} ({datastr})')
            coreids, coredatastr = m.groups()
            if not coreids:
                continue  # XXX
            for coreid in cls._parse_core_ids(coreids):
                if coreid in data_per_core:
                    raise ValueError(f'duplicate core ID {coreid}')
                data_per_core[coreid] = coredatastr
        return data_per_core

    @classmethod
    def _render_core_data_grouped(cls, data_per_core):
        coreparts = {}
        for core, data in sorted(data_per_core.items(),
                                 key=lambda v: (v[1], v[0])):
            if isinstance(core, int):
                coreid = core
            else:
                assert isinstance(core, CPUCoreInfo), core
                coreid = core.id
            if data not in coreparts:
                coreparts[data] = [coreid]
            else:
                coreparts[data].append(coreid)
        for data in list(coreparts):
            coreids = cls._render_core_ids(coreparts.pop(data))
            coreparts[coreids] = data
        return '; '.join(f'{i}={c}' for i, c in sorted(coreparts.items()))

    @classmethod
    def _parse_core_ids(cls, rawstr):
        # "0-1,11,14"
        parsed = []
        for coreid in rawstr.split(','):
            if '-' in coreid:
                start, stop = coreid.split('-')
                parsed.extend(range(int(start), int(stop) + 1))
            else:
                parsed.append(int(coreid))
        return parsed

    @classmethod
    def _render_core_ids(cls, coreids):
        ranges = []
        start = end = None
        for coreid in sorted(coreids):
            if start is None:
                start = coreid
                end = coreid
            else:
                if coreid == start + 1:
                    end += 1
                elif coreid == end:
                    raise ValueError(f'duplicate CPU ID {coreid}')
                else:
                    ranges.append(
                        str(start) if start is end else f'{start}-{end}'
                    )
        return ','.join(ranges)

    @classmethod
    def _normalize_frequency(cls, freq):
        raise NotImplementedError  # XXX

    def __new__(cls, model, arch, cores):
        self = super().__new__(
            cls,
            model=model,
            arch = arch or None,
            cores=tuple(CPUCoreInfo.from_raw(c) for c in cores or ()),
        )
        return self

    def __init__(self, *args, **kwargs):
        self._validate()

    def _validate(self):
        validate_str(self.model, 'model')
        validate_str(self.arch, 'arch')
        validate_arg(self.cores, tuple, 'cores')
        for i, core in enumerate(self.cores):
            validate_arg(core, CPUCoreInfo, f'core {i}')
            if core.id != i:
                raise ValueError(f'cores[{i}].id is {core.id}')

    @property
    def num_cores(self):
        return len(self.cores)

    def as_metadata(self):
        metadata = {
            'cpu_model_name': self.model,
            'cpu_count': self.num_cores,
            'cpu_arch': self.arch,
        }

        frequencies = {}
        configs = {}
        affinity = []
        for core in self.cores:
            core_meta = core.as_metadata()
            configs[core] = core_meta['cpu_config']
            freq = core_meta.get('cpu_freq')
            if freq:
                frequencies[core] = self._normalize_frequency(freq)
            if core_meta.get('cpu_affinity'):
                affinity.append(core.id)
        if configs:
            metadata['cpu_config'] = self._render_core_data_grouped(configs)
        if frequencies:
            metadata['cpu_freq'] = self._render_core_data_grouped(frequencies)
        if affinity:
            metadata['cpu_affinity'] = self._render_core_ids(frequencies)

        return metadata


class CPUCoreInfo(namedtuple('CPUCoreInfo', 'id config frequency affinity')):

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            raise ValueError('missing CPU core info')
        elif isinstance(raw, cls):
            return raw
        else:
            raise TypeError(raw)

    def __new__(cls, id, config, frequency=None, affinity=False):
        self = super().__new__(
            cls,
            id=coerce_int(id),
            config=CPUConfig.from_raw(config),
            frequency=frequency or None,
            affinity=True if affinity else False,
        )
        return self

    def __init__(self, *args, **kwargs):
        self._validate()

    def _validate(self):
        #validate_arg(self.config, CPUConfig, 'config')
        validate_int(self.frequency, 'frequency', required=False)
        #if not isinstance(self.affinity, bool):
        #    raise TypeError(f'expected True/False for affinity, got {self.affinity!r}')

    def as_metadata(self):
        metadata = {
           'cpu_config': self.config,
        }
        if self.frequency:
           metadata['cpu_freq'] = self.frequency
        if self.affinity:
           metadata['cpu_affinity'] = True
        return metadata


class CPUConfig:

    @classmethod
    def from_raw(cls, raw):
        if raw == []:
            return cls([])
        elif not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        else:
            raise TypeError(raw)

    @classmethod
    def parse(cls, configstr):
        # "driver:intel_pstate, intel_pstate:no turbo, governor:performance, isolated"
        if not configstr:
            raise ValueError('missing configstr')
        configs = configstr.replace(',', ' ').split()
        self = cls(configs)
        self._raw = configstr
        return self

    def __init__(self, data):
        if not all(data):
            raise ValueError(f'empty value(s) in {data!r}')
        self._data = tuple(data)

    def __repr__(self):
        return f'{type(self).__name__}({self._data!r})'

    def __str__(self):
        try:
            return self._raw
        except AttributeError:
            self._raw = ', '.join(self._data)
            return self._raw

    def __hash__(self):
        return hash(self._data)

    def __eq__(self, other):
        try:
            return self._data == other._data
        except AttributeError:
            return NotImplemented

    def __lt__(self, other):
        try:
            return self._data < other._data
        except AttributeError:
            return NotImplemented

    @property
    def values(self):
        return self._data


##################################
# network utils

# We don't bother going full RFC 5322.
# See http://emailregex.com/.
DOMAIN_PART = r'(?:\b[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\b)'
DOMAIN_NAME = rf'(?:\b{DOMAIN_PART}(?:\.{DOMAIN_PART})+\b)'
EMAIL_USER = r'(?:\b[\w+-]+(?:\.[\w+-]+)*\b)'
EMAIL_ADDR = rf'(?:\b{EMAIL_USER}@{DOMAIN_NAME}\b)'
EMAIL = rf'(?:.* <{EMAIL_ADDR}>|{EMAIL_ADDR})'


def parse_email_address(addr):
    if not addr:
        raise ValueError('missing addr')
    elif isinstance(addr, str):
        m = re.match(f'^({EMAIL_ADDR})|(\S.*) <({EMAIL_ADDR})>$', addr)
        if not m:
            return None
        addr1, name2, addr2 = m.groups()
        if addr1:
            return (None, addr1)
        else:
            return (name2.strip(), addr2)
    else:
        raise NotImplementedError(addr)


class SSHAgentInfo(namedtuple('SSHAgentInfo', 'auth_sock pid')):

    SCRIPT = os.path.join(HOME, '.ssh', 'agent.sh')

    @classmethod
    def from_env_vars(cls, *, requirepid=False):
        sock = os.environ.get('SSH_AUTH_SOCK')
        if sock:
            if not os.path.exists(sock):
                logger.warning(f'auth sock {sock} missing')
        else:
            return None

        pid = os.environ.get('SSH_AGENT_PID')
        if pid:
            pid = int(pid)
        elif requirepid:
            raise ValueError('SSH_AGENT_PID not found')
        else:
            pid = None

        return cls.__new__(cls, sock, pid)

    @classmethod
    def parse_script(cls, script=None, *, requirepid=False):
        """Return the info parsed from the given lines.

        The expected text is the output of running the ssh-agent command.
        For example:

        SSH_AUTH_SOCK=/tmp/ssh-7yRJ1mCaatzW/agent.8926; export SSH_AUTH_SOCK;
        SSH_AGENT_PID=8927; export SSH_AGENT_PID;
        echo Agent pid 8927;
        """
        if not script:
            if not os.path.exists(cls.SCRIPT):
                return None
            with open(cls.SCRIPT) as infile:
                text = infile.read()
        elif isinstance(script, str):
            text = script
        elif hasattr(script, 'read'):
            text = script.read()
        else:
            text = '\n'.join(script)

        m = re.match(r'^SSH_AUTH_SOCK=(/tmp/ssh-.+/agent\.\d+)(?:;|$)', text)
        if m:
            sock, = m.groups()
            if not os.path.exists(sock):
                logger.warning(f'auth sock {sock} missing')
        else:
            raise ValueError('SSH_AUTH_SOCK not found')

        m = re.match(r'^SSH_AGENT_PID=(\d+)(?:;|$)', text)
        if m:
            pid, = m.groups()
            pid = int(pid)
        elif requirepid:
            raise ValueError('SSH_AGENT_PID not found')
        else:
            pid = None

        return cls.__new__(cls, sock, pid)

    @classmethod
    def find_latest(cls):
        latest = None
        created = None
        # This will only match for the current user.
        for filename in glob.iglob('/tmp/ssh-*/agent.*'):
            if not latest:
                latest = filename
            else:
                _created = os.stat(filename).st_ctime
                if _created > created:
                    latest = filename
                    created = _created
        if not latest:
            return None
        return cls.__new__(cls, latest, None)

    @classmethod
    def from_jsonable(cls, data):
        return cls(**data)

    def __init__(self, *args, **kwargs):
        self._validate()

    def _validate(self):
        if not self.auth_sock:
            raise ValueError('missing auth_sock')
        elif not os.path.exists(self.auth_sock):
            logger.warning(f'auth sock {self.auth_sock} missing')
        if not self.pid:
            logger.warning(f'missing pid')
        else:
            validate_int(self.pid, name='pid')

    @property
    def env(self):
        return {
            'SSH_AUTH_SOCK': self.auth_sock,
            **({'SSH_AGENT_PID': str(self.pid)} if self.pid else {}),
        }

    def apply_env(self, env=None):
        if env is None:
            env = os.environ
        return dict(env, **self.env)

    def check(self):
        """Return True if the info is valid."""
        return os.path.exists(self.auth_sock)

    def as_jsonable(self):
        return self._asdict()


class SSHConnectionConfig(Config):

    FIELDS = ['user', 'host', 'port', 'agent']
    OPTIONAL = ['agent']

    CONFIG = 'ssh-conn.json'

    @classmethod
    def from_raw(cls, raw):
        if isinstance(raw, cls):
            return raw
        elif not raw:
            raise ValueError('missing data')
        else:
            return cls(**raw)

    def __init__(self, user, host, port, agent=None):
        if not user:
            raise ValueError('missing user')
        if not host:
            raise ValueError('missing host')
        if not port:
            raise ValueError('missing port')
        if not agent:
            agent = SSHAgentInfo.parse_script()
            #if not agent:
            #    agent = SSHAgentInfo.from_env_vars()
        else:
            agent = SSHAgentInfo.from_jsonable(agent)
        super().__init__(
            user=user,
            host=host,
            port=port,
            agent=agent,
        )


class SSHCommands:

    SSH = shutil.which('ssh')
    SCP = shutil.which('scp')

    @classmethod
    def from_config(cls, cfg, **kwargs):
        return cls(cfg.user, cfg.host, cfg.port, **kwargs)

    def __init__(self, user, host, port, *, ssh=None, scp=None):
        self.user = check_shell_str(user)
        self.host = check_shell_str(host)
        self.port = int(port)
        if self.port < 1:
            raise ValueError(f'invalid port {self.port}')

        opts = []
        if self.host == 'localhost':
            opts.extend(['-o', 'StrictHostKeyChecking=no'])
        self._ssh = ssh or self.SSH
        self._ssh_opts = [*opts, '-p', str(self.port)]
        self._scp = scp or self.SCP
        self._scp_opts = [*opts, '-P', str(self.port)]

    def __repr__(self):
        args = (f'{n}={getattr(self, n)!r}'
                for n in 'host port user'.split())
        return f'{type(self).__name__}({"".join(args)})'

    def __eq__(self, other):
        raise NotImplementedError

    def run(self, cmd, *args, agent=None):
        conn = f'{self.user}@{self.host}'
        if not os.path.isabs(cmd):
            raise ValueError(f'expected absolute path for cmd, got {cmd!r}')
        return [self._ssh, *self._ssh_opts, conn, cmd, *args]

    def run_shell(self, cmd, *, agent=None):
        conn = f'{self.user}@{self.host}'
        return [self._ssh, *self._ssh_opts, conn, *shlex.split(cmd)]

    def push(self, source, target, *, agent=None):
        conn = f'{self.user}@{self.host}'
        return [self._scp, *self._scp_opts, '-rp', source, f'{conn}:{target}']

    def pull(self, source, target, *, agent=None):
        conn = f'{self.user}@{self.host}'
        return [self._scp, *self._scp_opts, '-rp', f'{conn}:{source}', target]

    def ensure_user(self, user, *, agent=False):
        raise NotImplementedError


class SSHShellCommands(SSHCommands):

    SSH = 'ssh'
    SCP = 'scp'

    def run(self, cmd, *args, agent=None):
        return ' '.join(shlex.quote(a) for a in super().run(cmd, *args))

    def run_shell(self, cmd, *, agent=None):
        return ' '.join(super().run_shell(cmd))

    def push(self, source, target, *, agent=None):
        return ' '.join(super().push(source, target))

    def pull(self, source, target, *, agent=None):
        return ' '.join(super().pull(source, target))

    def ensure_user(self, user, *, agent=False):
        commands = []
        if agent:
            commands.extend([
                f'setfacl -m {user}:x $(dirname "$SSH_AUTH_SOCK")',
                f'setfacl -m {user}:rwx "$SSH_AUTH_SOCK"',
            ])
        commands.extend([
            f'# Stop running and re-run this script as the {user} user.',
            f'''exec sudo --login --user {user} --preserve-env='SSH_AUTH_SOCK' "$0" "$@"''',
        ])
        return commands


class SSHClient(SSHCommands):

    @property
    def commands(self):
        return SSHCommands(self.user, self.host, self.port)

    @property
    def shell_commands(self):
        return SSHShellCommands(self.user, self.host, self.port)

    def check(self, *, agent=None):
        return (self.run_shell('true', agent=agent).returncode == 0)

    def run(self, cmd, *args, agent=None):
        argv = super().run(cmd, *args)
        env = agent.apply_env() if agent else None
        return run_fg(*argv, env=env)

    def run_shell(self, cmd, *args, agent=None):
        argv = super().run_shell(cmd, *args)
        env = agent.apply_env() if agent else None
        return run_fg(*argv, env=env)

    def push(self, source, target, *, agent=None):
        argv = super().push(*args)
        env = agent.apply_env() if agent else None
        return run_fg(*argv, env=env)

    def pull(self, source, target, *, agent=None):
        argv = super().push(*args)
        env = agent.apply_env() if agent else None
        return run_fg(*argv, env=env)

    def read(self, filename, *, agent=None):
        if not filename:
            raise ValueError(f'missing filename')
        proc = self.run_shell(f'cat {filename}', agent=agent)
        if proc.returncode != 0:
            return None
        return proc.stdout


##################################
# versions

class VersionRelease(namedtuple('VersionRelease', 'level serial')):

    LEVELS = {
        'alpha': 'a',
        'beta': 'b',
        'candidate': 'rc',
        'final': 'f',
    }
    LEVEL_SYMBOLS = {s: l for l, s in LEVELS.items()}
    LEVEL_SYMBOLS['c'] = 'candidate'

    PAT = textwrap.dedent(rf'''(?:
        ( {'|'.join(LEVEL_SYMBOLS)} )  # <level>
        ( \d+ )  # <serial>
    )''')

    @classmethod
    def unreleased(cls):
        return cls.__new__(cls, 'alpha', 0)

    @classmethod
    def initial(cls, level=None):
        return cls.__new__(cls, level or 'alpha', 1)

    @classmethod
    def final(cls):
        return cls.__new__(cls, 'final', 0)

    @classmethod
    def from_values(cls, level=None, serial=None):
        # self._validate() will catch any bogus/missing level or serial.
        try:
            level = cls.LEVEL_SYMBOLS[level]
        except (KeyError, TypeError):
            pass

        if isinstance(serial, int):
            pass
        elif not serial:
            if level == 'final':
                serial = 0
            elif not level:
                level = 'final'
                serial = 0
        elif isinstance(serial, str):
            serial = coerce_int(serial, fail=False)
        return cls(level, serial)

    @classmethod
    def handle_parsed(cls, level, serial):
        level, serial = cls._handle_parsed(level, serial)
        if not level:
            return None
        return cls.__new__(cls, level, serial)

    @classmethod
    def _handle_parsed(cls, level, serial):
        if level:
            assert serial, (level, serial)
            return (
                cls.LEVEL_SYMBOLS[level],
                coerce_int(serial, fail=False),
            )
        else:
            assert not serial, (level, serial)
            return None, None

    @classmethod
    def validate(cls, release):
        if not isinstance(release, cls):
            raise TypeError(f'expected a {cls.__name__}, got {release!r}')
        release._validate()

    def __init__(self, *args, **kwargs):
        self._validate()

    def __str__(self):
        return self.render()

    def _validate(self):
        if not self.level:
            raise ValueError('missing level')
        elif self.level not in self.LEVELS:
            raise ValueError(f'unsupported level {self.level}')
        elif self.level == 'final':
            if self.serial != 0:
                raise ValueError(f'final releases always have a serial of 0, got {self.serial}')
        if self.level == 'alpha':
            validate_int(self.serial, 'serial', range='non-negative', required=True)
        else:
            validate_int(self.serial, 'serial', range='positive', required=True)

    @property
    def is_unreleased(self):
        return self.level == 'alpha' and self.serial == 0

    def next_level(self):
        cls = type(self)
        if self.level == 'alpha':
            return cls.__new__(cls, 'beta', 1)
        elif self.level == 'beta':
            return cls.__new__(cls, 'candidate', 1)
        elif self.level == 'candidate':
            return cls.__new__(cls, 'final', 0)
        elif self.level == 'final':
            raise ValueError('a final release has no next level')
        else:
            raise NotImplementedError(self.level)

    def next_serial(self, max=None):
        if self.level == 'final':
            raise ValueError('a final release has no next serial')
        elif max and self.serial >= max:
            return self.next_level()
        else:
            cls = type(self)
            return cls.__new__(cls, self.level, self.serial + 1)

    def next(self, plan=None):
        if self.level == 'alpha':
            maxserial = plan.nalphas if plan else None
        elif self.level == 'beta':
            maxserial = plan.nbetas if plan else None
        elif self.level == 'candidate':
            maxserial = plan.ncandidates if plan else None
        elif self.level == 'final':
            maxserial = None
        else:
            raise NotImplementedError(self.level)
        return self.next_serial(maxserial)

    def render(self):
        level = self.LEVELS[self.level]
        return f'{self.LEVELS[self.level]}{self.serial}'


class ReleaseInfo(namedtuple('ReleaseInfo', 'date rm')):

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, (datetime.date, datetime.datetime)):
            return cls(date=raw)
        else:
            raise TypeError(f'unsupported raw value {raw!r}')

    def __new__(cls, date=None, rm=None):
        return super().__new__(cls, date, rm)

    def __init__(self, *args, **kwargs):
        self._validate()

    def _validate(self):
        ...


class ReleasePlan:

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        else:
            return cls(raw)

    @classmethod
    def _normalize_data(cls, data):
        levels = sorted(VersionRelease.LEVELS)

        schedule = []
        bylevel = {l: [] for l in levels}
        bylevel['final'] = None
        if hasattr(data, 'items'):
            if set(data) - bylevel:
                raise ValueError(f'got unexpected release levels in {data}')
            for level in levels:
                if level == 'final':
                    continue
                for info in data.get(level) or ():
                    info = ReleaseInfo.from_raw(info)
                    schedule.append((level, info))
                    bylevel[level].append(info)
            final = ReleaseInfo.from_raw(data.get('final'))
            if final:
                schedule.append(('final', final))
                bylevel['final'] = final
        elif isinstance(data, str):
            raise NotImplementedError(repr(schedule))
        else:
            for entry in data:
                if isinstance(entry, (tuple, list)):
                    level, info= entry
                    if level not in bylevel:
                        raise ValueError(f'unsupported release level {level!r}')
                    info = ReleaseInfo.from_raw(info)
                    schedule.append((level, info))
                    bylevel[level].append(info)
                else:
                    raise NotImplementedError(entry)
        return schedule, bylevel

    def __init__(self, data):
        if not data:
            raise ValueError('missing data')
        self._schedule, self._bylevel = self._normalize_data(data)

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def nalphas(self):
        return len(self._bylevel['alpha'])

    @property
    def nbetas(self):
        return len(self._bylevel['beta'])

    @property
    def ncandidates(self):
        return len(self._bylevel['candidate'])


class ReleasePlans:

    @classmethod
    def _normalize_data(cls, data):
        plans = {}
        for ver, plan in data.items():
            version = Version.from_raw(ver)
            if not version:
                raise ValueError(f'bad version {ver!r}')
            version = version.plain
            if version in plans:
                raise KeyError(f'duplicate release plan for {version}')
            plan = ReleasePlan.from_raw(plan)
            plans[version.plain] = plan
        return plans

    def __init__(self, plans):
        if not plans:
            raise ValueError('missing plans')
        self._plans = self._normalize_data(plans)

    def __eq__(self, other):
        raise NotImplementedError

    def get(self, version):
        return self._plans.get(version.plain)


class Version(namedtuple('Version', 'major minor micro release')):

    PAT = textwrap.dedent(rf'''(?:
        (\d+)  # <major>
        \.
        (\d+)  # <minor>
        (?:
            \.
            (\d+)  # <micro>
         )?
        (?:
            {VersionRelease.PAT}  # <level> <serial>
         )?
        ( \+ )?  # <extra>
    )''')
    REGEX = re.compile(f'^v?{PAT}$', re.VERBOSE)

    _extra = None

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls.parse(raw)
        elif isinstance(raw, (tuple, list)):
            return cls(*raw)
        else:
            return cls(**raw)

    @classmethod
    def resolve_main(cls):
        raise NotImplementedError

    @classmethod
    def parse(cls, verstr, match=None):
        m = cls.REGEX.match(verstr)
        if not m:
            return None
        self = cls._from_parsed(verstr, m.groups())
        if match is not None and not self.match(match):
            return None
        return self

    @classmethod
    def _from_parsed(cls, verstr, parts):
        (major, minor, micro, release, extra,
         ) = cls._handle_parsed(verstr, *parts)
        cls._validate_values(verstr, major, minor, micro, release)
        cls._validate_extra(verstr, extra, major, minor, micro, release)
        self = cls.__new__(cls, major, minor, micro, release)
        self._raw = verstr
        self._extra = extra
        return self

    @classmethod
    def _handle_parsed(cls, verstr, major, minor, micro, level, serial, extra):
        release = VersionRelease.handle_parsed(level, serial)
        return (
            int(major),
            int(minor),
            int(micro) if micro else 0 if release else None,
            release,
            extra or None,
        )

    @classmethod
    def _validate_values(cls, verstr, major, minor, micro, release):
        return

    @classmethod
    def _validate_extra(cls, verstr, extra, major, minor, micro, release):
        return

    def __new__(cls, major, minor, micro=None, release=None):
        return super().__new__(cls, major, minor, micro, release or None)

    def __init__(self, *args, **kwargs):
        self._validate()

    def __str__(self):
        return self.render()

    def _validate(self):
        def _validate_int(name, *, required=False):
            val = getattr(self, name)
            validate_int(val, name, range='non-negative', required=required)
        _validate_int('major', required=True)
        _validate_int('minor', required=True)
        _validate_int('micro')
        if self.release is not None:
            VersionRelease.validate(self.release)
            if self.micro is None:
                raise ValueError('missing micro')
        self._validate_values(self, *self)
        self._validate_extra(self, self._extra, *self)

    @property
    def branch(self):
        return self[:2]

    @property
    def full(self):
        if self.release:
            return self
        major, minor, micro, _ = self
        release = VersionRelease.final()
        cls = type(self)
        full = cls.__new__(cls, major, minor, micro or 0, release)
        full._raw = self._raw
        return full

    @property
    def plain(self):
        major, minor, micro, _ = self
        if micro and not self.release:
            return self
        cls = type(self)
        plain = cls.__new__(cls, major, minor, micro or 0)
        plain._raw = self.raw
        return plain

    @property
    def flat(self):
        major, minor, micro, release = self
        level, serial = release if release else (None, None)
        return major, minor, micro, level, serial
        #return major, minor, micro, level, serial, self._extra

    @property
    def raw(self):
        try:
            return self._raw
        except AttributeError:
            self._raw = self.render()
            return self._raw

    @property
    def unreleased(self):
        cls = type(self)
        major, minor, micro, _ = self
        release = VersionRelease.unreleased()
        return cls.__new__(cls, major, minor, micro or 0, release)

    @property
    def initial(self):
        cls = type(self)
        major, minor, micro, _ = self
        release = VersionRelease.initial()
        return cls.__new__(cls, major, minor, micro or 0, release)

    @property
    def final(self):
        cls = type(self)
        major, minor, micro, _ = self
        release = VersionRelease.final()
        return cls.__new__(cls, major, minor, micro or 0, release)

    def next_major(self):
        cls = type(self)
        return cls.__new__(cls, self.major + 1, 0)

    def next_minor(self):
        cls = type(self)
        return cls.__new__(cls, self.major, self.minor + 1)

    def next_micro(self):
        major, minor, micro, _ = self
        cls = type(self)
        return cls.__new__(cls, major, minor, (micro or 0) + 1)

    def next(self):
        if self.release:
            return self.next_release()
        elif micro:
            return self.next_micro()
        else:
            return self.next_minor()

    def next_release(self, plans=None):
        cls = type(self)
        major, minor, micro, release = self
        if not release or release.level == 'final':
            # It's a final release.
            return self.next_micro().initial
        plan = plans.get(self) if plans else None
        release = release.next(plan)
        return cls.__new__(cls, major, minor, micro, release)

#    def compare(self, other):
#        raise NotImplementedError

    def match(self, other, *, subversiononly=False):
        """Return True if other is a subversion."""
        if not other:
            return None
        else:
            other = Version.from_raw(other)
            if not other:
                return None
        if not subversiononly and self == other:
            return True
        if not self.release:
            if self.micro is not None:
                if other.release:
                    return self[:3] == other[:3]
            else:
                if other.micro is not None:
                    return self[:2] == other[:2]
        return False

    def render(self, *, withextra=True):
        if self.release:
            verstr = f'{self.major}.{self.minor}.{self.micro}{self.release}'
        elif self.micro:
            verstr = f'{self.major}.{self.minor}.{self.micro}'
        else:
            verstr = f'{self.major}.{self.minor}'
        return f'{verstr}{self._extra}' if self._extra else verstr

    def as_tag(self):
        micro = f'.{self.micro}' if self.micro else ''
        if self.level == 'alpha':
            release = f'a{self.serial}'
        elif self.level == 'beta':
            release = f'b{self.serial}'
        elif self.level == 'candidate':
            release = f'rc{self.serial}'
        elif self.level == 'final':
            release = ''
        else:
            raise NotImplementedError(self.level)
        return f'v{self.major}.{self.minor}{micro}{release}'


class CPythonVersion(Version):

    @classmethod
    def parse_extended(cls, verstr):
        # "3.11.0a7 (64-bit) revision 45772541f6"
        m = re.match(
            rf'^({Version.PAT}) \((\d+)-bit\) revision ([a-fA-F0-9]{{4,40}})$',
            verstr,
            re.VERBOSE,
        )
        if not m:
            return None
        verstr, *verparts, bits, commit = m.groups()
        version = cls._from_parsed(verstr, verparts)
        return version, int(bits), commit

    @classmethod
    def render_extended(cls, version, bits, commit):
        if isinstance(version, str):
            if not CPythonVersion.parse(version):
                if version == 'main':
                    version = cls.resolve_main()
                else:
                    raise ValueError(version)
        elif isinstance(version, Version):
            pass
#            version = version.full
        else:
            raise TypeError(version)
        return f'{version} ({bits}-bit) revision {commit}'

    @classmethod
    def resolve_main(cls):
        # XXX Use CPythonGitRefs to get the right one.
        return cls(3, 12, 0).unreleased

    #@classmethod
    #def _validate_values(cls, verstr, major, minor, micro, release):
    #    if micro:
    #        if release and release.level != 'final':
    #            raise ValueError(f'bugfix versions can only be final, got {verstr})')

    @classmethod
    def _validate_extra(cls, verstr, extra, major, minor, micro, release):
        if extra == '+':
            if not self.release:
                raise ValueError('missing release (for dev version)')
            elif self.release.is_dev:
                raise ValueError('dev release not supported for dev version')
        elif extra:
            raise ValueError(f'unexpected data {extra!r}')

    @property
    def is_dev(self):
        if self._extra:
            return True
        return self.release and self.release.is_unreleased

    @property
    def is_bugfix(self):
        return bool(self.micro)

    def next_release(self, plans=None):
        if self.micro:
            return self.next_micro().final
        else:
            return super().next_release(plans)


##################################
# Python

class PythonImplementation:

    VERSION = Version

    def __init__(self, name):
        if not name:
            raise ValueError('missing name')
        self._name = name

    def __repr__(self):
        return f'{type(self).__name__}({self._name!r})'

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if not isinstance(other, PythonImplementation):
            return NotImplemented
        return self._name == other._name

    # XXX Support comparison with str.

    @property
    def name(self):
        return self._name

    def parse_version(self, version, *, requirestr=True) -> Version:
        try:
            return self.VERSION.parse(version)
        except TypeError:
            if requirestr:
                raise  # re-rase
            return self.VERSION.from_raw(version)


class CPython(PythonImplementation):

    VERSION = CPythonVersion

    def __init__(self):
        super().__init__('cpython')


def resolve_python_implementation(impl) -> PythonImplementation:
    if isinstance(impl, str):
        if impl == 'cpython':
            return CPython()
        else:
            return PythonImplementation(impl)
    elif isinstance(impl, PythonImplementation):
        return impl
    else:
        raise NotImplementedError(repr(impl))


##################################
# other utils

def hashable(value):
    try:
        hash(value)
    except TypeError as exc:
        if 'unhashable type' not in str(exc):
            raise  # re-raise
        return False
    else:
        return True


def iterable(value):
    try:
        iter(value)
    except TypeError as exc:
        if 'is not iterable' not in str(exc):
            raise  # re-raise
        return False
    else:
        return True


class Sentinel:

    def __init__(self, label):
        if not label:
            raise ValueError('missing label')
        elif not isinstance(label, str):
            raise TypeError(label)
        self._label = label

    def __repr__(self):
        return f'{type(self).__name__}({self._label!r})'

    def __str__(self):
        return self._label


def get_slice(raw):
    if isinstance(raw, int):
        start = stop = None
        if raw < 0:
            start = raw
        elif criteria > 0:
            stop = raw
        return slice(start, stop)
    elif isinstance(raw, str):
        if raw.isdigit():
            return get_slice(int(raw))
        elif raw.startswith('-') and raw[1:].isdigit():
            return get_slice(int(raw))
        else:
            raise NotImplementedError(repr(raw))
    else:
        raise TypeError(f'expected str, got {criteria!r}')


def as_jsonable(data):
    if hasattr(data, 'as_jsonable'):
        return data.as_jsonable()
    elif data in (True, False, None):
        return data
    elif type(data) in (int, float, str):
        return data
    else:
        # Recurse into containers.
        if hasattr(data, 'items'):
            return {k: as_jsonable(v) for k, v in data.items()}
        elif hasattr(data, '__getitem__'):
            return [as_jsonable(v) for v in data]
        else:
            raise TypeError(f'unsupported data {data!r}')


class MetadataError(ValueError):
    def __init__(self, msg=None):
        super().__init__(msg or 'metadata-related error')


class MissingMetadataError(MetadataError):
    def __init__(self, msg=None, source=None):
        if not msg:
            msg = 'missing metadata'
            if source:
                msg = f'{msg} (in {{source}})'
        super().__init__(msg.format(source=source))


class InvalidMetadataError(MetadataError):
    def __init__(self, msg=None, source=None):
        if not msg:
            msg = 'invalid metadata'
            if source:
                msg = f'{msg} (in {{source}})'
        super().__init__(msg.format(source=source))


class Metadata(types.SimpleNamespace):

    FIELDS: List[str] = []
    OPTIONAL: List[str] = []

    _extra = None

    @classmethod
    def load(
            cls,
            metafile: Union[str, TextIO],
            *,
            optional: Optional[List[str]] = None,
            fail: bool = True,
            **kwargs
    ):
        if optional is None:
            optional = cls.OPTIONAL or []
        filename = getattr(metafile, 'name', metafile)
        try:
            data = cls._load_data(metafile)
            kwargs, extra = cls._extract_kwargs(data, optional, filename)
            self = cls(**kwargs)
        except Exception:
            logger.debug(f'failed to load metadata from {filename!r}')
            if not fail:
                return None
            raise  # re-raise
        if extra:
            self._extra = extra
        return self

    @classmethod
    def from_jsonable(
            cls,
            data: Any,
            *,
            optional: Optional[List[str]] = None
    ):
        if optional is None:
            optional = cls.OPTIONAL or []
        kwargs, extra = cls._extract_kwargs(data, optional, None)
        self = cls(**kwargs)
        if extra:
            self._extra = extra
        return self

    @classmethod
    def _load_data(
            cls,
            metafile: Union[str, TextIO],
            fail: bool = False
    ) -> Any:
        if isinstance(metafile, str):
            filename = metafile
            with open(filename) as metafile:
                return cls._load_data(metafile)
        try:
            return json.load(metafile)
        except json.decoder.JSONDecodeError as exc:
            if fail:
                raise  # re-raise
            logger.error(f'failed to decode metadata in {metafile.name}: {exc}')
            return None

    @classmethod
    def _extract_kwargs(
            cls,
            data: dict,
            optional: List[str],
            filename
    ) -> Tuple[dict, Optional[dict]]:
        kwargs = {}
        extra = {}
        unused = set(cls.FIELDS or [])
        for field in data:
            if field in unused:
                kwargs[field] = data[field]
                unused.remove(field)
            elif not field.startswith('_'):
                extra[field] = data[field]
        unused -= set(optional)
        if unused:
            missing = ', '.join(sorted(unused))
            raise ValueError(f'missing required data (fields: {missing})')
        return (kwargs, extra) if kwargs else (extra, None)

    def as_jsonable(self, *, withextra: bool = False) -> dict:
        fields = self.FIELDS
        if not fields:
            fields = [f for f in vars(self) if not f.startswith('_')]
        elif withextra:
            fields.extend((getattr(self, '_extra', None) or {}).keys())
        # TODO: "optional" should be used in the loop to be smarter
        # about how to handle AttributeError
        optional = set(self.OPTIONAL or ())
        data = {}
        for field in fields:
            try:
                value = getattr(self, field)
            except AttributeError:
                # XXX Fail?  Warn?  Add a default?
                continue
            if hasattr(value, 'as_jsonable'):
                value = value.as_jsonable()
            data[field] = value
        return data

    def save(
            self,
            resfile: Union[str, TextIO],
            *,
            withextra: bool = False
    ) -> None:
        if isinstance(resfile, str):
            filename = resfile
            with open(filename, 'w') as resfile:
                return self.save(resfile, withextra=withextra)
        data = self.as_jsonable(withextra=withextra)
        write_json(data, resfile)
