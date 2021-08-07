import datetime
from decimal import Decimal
import re
import shlex
import subprocess
import sys


_ONE = Decimal(1)

# https://en.wikipedia.org/wiki/Orders_of_magnitude_(time)
TIME_UNITS = [
    # These are ordered by magnitude.
    ('w', 7),
    ('d', 24),
    ('h', 60),
    ('m', 60),
    ('s', 1000),
    ('ms', 1000),
    ('us', 1000),
    ('ns', 1000),
    ('ps', 1000),
    ('fs', 1000),
    ('as', 1000),
    ('zs', 1000),
    ('ys', None)
    # Smaller units aren't defined.
]
TIME_UNITS_DESCENDING = {}
TIME_UNITS_ASCENDING = {}
for i, v in enumerate(TIME_UNITS):
    s, m = v
    if i == 0:
        TIME_UNITS_ASCENDING[s] = (None, None)
    else:
        v = TIME_UNITS[i-1]
        TIME_UNITS_ASCENDING[s] = (v[0], _ONE / v[1])
    if i + 1 == len(TIME_UNITS):
        TIME_UNITS_DESCENDING[s] = (None, None)
    else:
        v = TIME_UNITS[i+1]
        TIME_UNITS_DESCENDING[s] = (v[0], m)
del i, s, m, v
TIME_UNITS = [s for s, _ in TIME_UNITS]


def _get_whole_time_units(value, units='s', *, allowsmall=False):
    assert isinstance(value, Decimal)
    if units not in TIME_UNITS:
        raise ValueError(f'unsupported units {units!r}')
    orig = f'{value} {units}'
    while int(value) != value:
        units, multiplier = TIME_UNITS_DESCENDING[units]
        if not units:
            if allowsmall:
                units = prev
            raise ValueError(f'{orig} too small for supported units')
        value *= multiplier
    return value, units


def _ensure_time_units(value, units, target):
    value = Decimal(value)
    try:
        u_index = TIME_UNITS.index(units)
    except ValueError:
        raise ValueError(f'unsupported units {units!r}')
    try:
        t_index = TIME_UNITS.index(target)
    except ValueError:
        raise ValueError(f'unsupported target {target!r}')
    candidates = TIME_UNITS_DESCENDING if u_index < t_index else TIME_UNITS_ASCENDING
    symbol = units
    while symbol != target:
        _symbol = symbol
        symbol, multiplier = candidates[symbol]
        assert multiplier is not None, (_symbol, units, target)
        value *= multiplier
    return value


def _add_raw_timedeltas(td1, td2):
    val1, units1 = td1
    val2, units2 = td2
    if units1 == units2:
        return (val1 + val2, units1)
    if TIME_UNITS.index(units1) > TIME_UNITS.index(units2):
        units = units1
        val2 = _ensure_time_units(val2, units2, units1)
    else:
        units = units2
        val1 = _ensure_time_units(val1, units1, units2)
    return val1 + val2, units


class TinyTimeDelta(datetime.timedelta):
    __slots__ = ['_tiny']

    @classmethod
    def parse(cls, raw):
        value, _, units = raw.strip().partition(' ')
        tiny = _get_whole_time_units(Decimal(value), units)
        self = cls(tiny=tiny)
        return self

    @classmethod
    def _timedelta_as_decimal(cls, td, units='us'):
        value = _ensure_time_units(td.days, 'd', units)
        value += _ensure_time_units(td.seconds, 's', units)
        value += _ensure_time_units(td.microseconds, 'us', units)
        return value

    def __new__(cls, microseconds=0, tiny=None):
        dec_us = Decimal(microseconds)
        if tiny:
            tinyval, tinyunits = tiny
            if tinyval is None:
                tiny = None
            elif tinyunits not in TIME_UNITS:
                raise ValueError(f'bad tiny units in {tiny}')
            else:
                dec_us +=  _ensure_time_units(tinyval, tinyunits, 'us')
        microseconds = int(dec_us)
        tiny = _get_whole_time_units(dec_us - microseconds, 'us')
        if tiny[0] == 0:
            tiny = None
        self = super().__new__(cls, microseconds=microseconds)
        self._tiny = tiny
        return self

    def __repr__(self):
        kwargs = [f'microseconds={self.microseconds}']
        if self._tiny:
            kwargs.append(f'tiny={self.tiny}')
        return f'{type(self).__name__}({", ".join(kwargs)})'

    def __add__(self, other):
        # We can't use super() because timedelta hard-codes itself.
        if not isinstance(other, datetime.timedelta):
            return NotImplemented
        self_us = int(self.total_seconds()) * 10**6 + self.microseconds
        other_us = int(other.total_seconds()) * 10**6 + other.microseconds
        microseconds = self_us + other_us
        if not self._tiny:
            tiny = getattr(other, '_tiny', None)
        else:
            try:
                other_tiny = other._tiny
            except AttributeError:
                tiny = self._tiny
            else:
                tiny = _add_raw_timedeltas(self._tiny, other_tiny)
        return type(self)(microseconds, tiny)

    def __sub__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __floordiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, TinyTimeDelta):
            other_units = 'us'
            other = other.as_decimal('us')
        elif isinstance(other, datetime.timedelta):
            other_units = 'us'
            other = self._timedelta_as_decimal(other, 'us')
        elif isinstance(other, int):
            other_units = None
        else:
            return NotImplemented

        if other_units:
            return self.as_decimal(other_units) / other
        else:
            microseconds = self.as_decimal('us') / other
            return type(self)(microseconds)

    def __mod__(self, other):
        raise NotImplementedError

    def __divmod__(self, other):
        raise NotImplementedError

    def _cmp(self, other):
        raise NotImplementedError

    def __bool__(self):
        raise NotImplementedError

    def _getstate(self):
        raise NotImplementedError

    @property
    def tiny(self):
        return self._tiny

    def as_decimal(self, units='s'):
        value = self._timedelta_as_decimal(self, units)
        if self._tiny:
            tinyval, tinyunits = self._tiny
            value += _ensure_time_units(tinyval, tinyunits, units)
        return value

    def render(self, units='s'):
        value = str(self.as_decimal(units)).strip('0')
        return f'{value} {units}'


##################################
# uftrace

COLUMNS = ['Total time', 'Self time', 'Calls', 'Function']


def run_uftrace(datadir, *, graph=False):
    if graph:
        cmd = shlex.split(f'uftrace graph --data {datadir}')
    else:
        cmd = shlex.split(f'uftrace report --data {datadir} --sort self')
    return subprocess.check_output(cmd, text=True)


def parse_data(lines):
    '''
      Total time   Self time       Calls  Function
      ==========  ==========  ==========  ====================
      774.765 us  774.765 us       14874  memcpy
      416.849 us  416.849 us        7265  strlen
    '''
    if isinstance(lines, str):
        lines = lines.splitlines()
    lines = iter(lines)

    header = _parse_header(lines)

    totals = {
        'Total time': 0,
        'Self time': 0,
        'Calls': 0,
    }

    def iter_rows():
        colnames = [c[1] for c in header[0]]
        for row in _parse_rows(lines, header):
            for colname, data in zip(colnames, row):
                _, value = data
                if colname in totals:
                    if totals[colname] == 0:
                        totals[colname] = value
                    else:
                        totals[colname] += value
            yield row
    rows = iter_rows()

    return header, rows, totals


def _parse_header(lines):
    colsrest = colline = next(lines)
    divrest = divline = next(lines)
    if '===' not in divline:
        raise ValueError(f'bad div line {div!r}')
    assert divline.rstrip() == divline

    # Get the separator.
    sepwidth = len(divrest) - len(divrest.lstrip())
    sep = divline[:sepwidth]

    columns = []
    while divrest:
        # Strip the separator.
        assert len(divrest) - len(divrest.lstrip()) == sepwidth
        divrest = divrest.lstrip()

        # Get the column width.
        total = len(divrest)
        divrest = divrest.lstrip('=')
        width = total - len(divrest)

        # Get the column name.
        assert colsrest[:sepwidth] == sep, (colsrest, sep, columns[-1])
        colsrest = colsrest[sepwidth:]
        column = colsrest[:width]
        colsrest = colsrest[width:]
        colname = column.strip()

        columns.append((column, colname, width))
    assert colsrest == '', (colsrest, colline, columns)
    assert [v[1] for v in columns] == COLUMNS, (colline, columns)
    return columns, sep


def _parse_rows(lines, header):
    columns, sep = header
    sepwidth = len(sep)

    for line in lines:
        linerest = line
        row = []
        for _, colname, width in columns:
            # Strip the separator.
            assert linerest[:sepwidth] == sep, repr(line)  # It has the separator.
            linerest = linerest[sepwidth:]

            # Get the raw value.
            valraw = linerest[:width]
            assert valraw.strip(), line  # The value is there.
            linerest = linerest[width:]
            if linerest and linerest[0] != ' ':
                # The last column's value can be bigger than its width.
                assert linerest.rstrip() == linerest, repr(line)  # no trailing spaces
                valraw += linerest
                linerest = ''

            # Parse the raw value.
            value = _parse_value(valraw, colname)

            info = (valraw, value)
            row.append(info)
        assert linerest == '', repr(line)  # We used up the line.
        yield row


def _parse_value(raw, colname):
    if colname == 'Function':
        value = raw
    elif colname == 'Calls':
        value = int(raw)
    elif colname in ('Total time', 'Self time'):
        value = TinyTimeDelta.parse(raw)
        # There is always a 3-digit fractional part, even if trailing 0s.
#        assert value.fractional and len(value.fractional) == 3, repr(raw)
    else:
        raise NotImplementedError(colname)
    return value


def fix_rows(rows, totals, header):
    # XXX Add % columns to headers.
    # XXX Add % columns to rows.
    columns, sep = header
    assert (tuple(v[1] for v in columns) == ('Total time', 'Self time', 'Calls', 'Function')), columns
    columns = list(columns)
    columns.insert(3, ('   Calls %', 'Calls %', 10))
    columns.insert(2, ('    Self %', 'Self %', 10))
    columns.insert(1, ('   Total %', 'Total %', 10))
    header = (columns, sep)

    rows = list(rows)  # Force the totals to be summed up.
    total_all = totals['Total time']
    self_all = totals['Self time']
    calls_all = totals['Calls']

    def iter_rows():
        for total_row, self_row, calls_row, function in rows:
            total_fraction = total_row[1] / total_all * 100
            self_fraction = self_row[1] / self_all * 100
            calls_fraction = Decimal(calls_row[1]) / calls_all * 100
            yield (
                total_row,
                (f'{total_fraction:>8.1f} %', total_fraction),
                self_row,
                (f'{self_fraction:>8.1f} %', self_fraction),
                calls_row,
                (f'{calls_fraction:>8.1f} %', calls_fraction),
                function,
            )
    fixed = iter_rows()

    return header, fixed


def render_table(header, rows):
    columns, sep = header

    # Render the header.
    line = ''
    div = ''
    for column, _, width in columns:
        line += sep
        line += column
        div += sep
        div += '=' * width
    yield line
    yield div

    # Render the rows.
    for row in rows:
        line = ''
        assert len(row) == len(columns), (row, columns)
        for data, column in zip(row, columns):
            valuetext, value = data
#            if hasattr(value, 'as_decimal'):
#                print(value.as_decimal('us'))
            #assert len(valuetext) == width, (valuetext, width)
            line += sep + valuetext
        yield line


def render_info(totals, header):
    columns, _ = header
    yield 'totals:'
    for _, colname, _ in columns:
        try:
            value = totals[colname]
        except KeyError:
            continue
        if isinstance(value, TinyTimeDelta):
            value = value.render('us')
        yield f'{colname+":":11} {value}'


def fix_graph(lines):
    '''
    # Function Call Graph for 'python' (session: f03290c62782639b)
    ========== FUNCTION CALL GRAPH ==========
    # TOTAL TIME   FUNCTION
        3.978 ms : (1) python
      408.654 us :  +-(5907) memset
                 :  |
        0.067 us :  +-(1) pthread_condattr_init
                 :  |
    '''
    if isinstance(lines, str):
        lines = lines.splitlines()
    else:
        lines = list(lines)
    
    # Get the total:
    regex = re.compile(r'^\s*(\d+(?:\.\d+)? [a-z]+s) : .*')
    total = 0
    for i, line in enumerate(lines):
        m = regex.match(line)
        if m:
            raw, = m.groups()
            msec = TinyTimeDelta.parse(raw).as_decimal('us')
            total += msec
        else:
            msec = None
        lines[i] = (msec, line)

    # Add it into the output.
    running = 0
    for msec, line in lines:
        if line.startswith('# Function Call Graph for '):
            yield line
            yield f'# (total run time: {total:.3f} us)'
        elif line == '# TOTAL TIME   FUNCTION':
            yield f'# RUNNING %  TOTAL %  {line[2:]}'
        elif msec is not None:
            fraction = msec / total
            running += fraction
            yield f'    {running * 100:>5.1f} %   {fraction * 100:>4.1f} %{line}'
        elif line and line[0] == ' ':
            yield '                    ' + line
        else:
            yield line


##################################
# the script

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('datadir')
    args = parser.parse_args()
    return vars(args)


def main(datadir, *, graph=False):
    text = run_uftrace(datadir, graph=graph)

    if graph:
        for line in fix_graph(text):
            print(line)
    else:
        header, rows, totals = parse_data(text)
        header, fixed = fix_rows(rows, totals, header)

        print('### fixed uftrace report ###')
        print()
        for line in render_table(header, fixed):
            print(line)
        print()
        for line in render_info(totals, header):
            print(line)


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
