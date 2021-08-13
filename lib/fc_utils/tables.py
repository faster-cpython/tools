import itertools


_MINWIDTH = 3

COLUMNS = '<COLUMNS>'
DIVIDER = '<DIVIDER>'
_MARKERS = {
    COLUMNS,
    DIVIDER,
}


class TableError(Exception): pass
class TableParserError(TableError): pass
class TableRendererError(TableError): pass


# XXX Support tabs as column sep?
# XXX Support multi-line headers?

def parse_table(lines, *, minwidth=_MINWIDTH, endpreface=None):
    try:
        (_, header, rows,
         ) = _parse_table(lines, minwidth, endpreface, keeppreface=False)
    except TableParserError:
        raise ValueError('not a table')
    return header, rows


def parse_table_with_preface(lines, *, minwidth=_MINWIDTH, endpreface=None):
    try:
        return _parse_table(lines, minwidth, endpreface, keeppreface=True)
    except TableParserError:
        raise ValueError('not a table')


def render_table(header, rows):
    return _render_table(header, rows)


##################################
# low-level

def _parse_table(lines, minwidth, endpreface, keeppreface):
    if isinstance(lines, str):
        lines = lines.splitlines()
    lines = iter(lines)
#    def _iter(_lines=lines):
#        for line in _lines:
#            print(f'----- {line!r}')
#            yield line
#    lines = _iter(lines)

    (preface, header, divline, colsline, prediv,
     ) = _parse_header(lines, minwidth, endpreface, keeppreface)
    colspecs, _, _ = header
    rows = _parse_rows(lines, colspecs, divline, colsline, prediv)
#    rows = list(rows)
#    print(rows)
#    lines = list(lines)
#    print(lines)
    return preface, header, rows


def _parse_header(lines, minwidth, endpreface, keeppreface):
    if not minwidth or minwidth < 0:
        minwidth = None
    if isinstance(endpreface, str):
        def endpreface(line, *, _match=endpreface):
            return line == _match
    assert endpreface is None or callable(endpreface), endpraface

#    print('header')
    lines = _mark_ignored_lines(lines)

    # Go to the first div line, consuming the preface.
    preface = [] if keeppreface else None
    prevline = None
    divinfo = None
    for line, ignored in lines:
        if keeppreface:
            preface.append(line)
        if not ignored:
            if endpreface is None:
                divinfo = _parse_divline(line)
                if divinfo is not None:
                    break
            elif endpreface(line):
                prevline = None
                # Pop off any ignored lines.
                for line, ignored in lines:
                    if not ignored:
                        break
                    if keeppreface:
                        preface.append(line)
                else:
                    raise TableParserError('never matched end of preface')
                divinfo = _parse_divline(line)
                if divinfo is None:
                    prevline = line
                break
            prevline = line
    else:
        raise TableParserError('never matched end of preface')
    if divinfo is None:
        # The next line must be the first div line.
        for line, ignored in lines:
            if not ignored:
                divinfo = _parse_divline(line)
                if divinfo is None:
                    raise TableParserError(f'expected div line, got {line!r}')
                break
            else:
                prevline = None
    divline = line
    colspecs, div = divinfo

#    print('...')

    # Get the columns.
    prediv = False
    colsline = None
    matched = _parse_row(prevline, colspecs) if prevline else None
    if matched:
        colsline = prevline
    else:
        # Go to the second div line.
        prediv = True
        for line, ignored in lines:
            if ignored:
                raise NotImplementedError(repr(line))
            elif colsline:
                if line != divline:
                    raise TableParserError(f'div line mismatch: {line!r} != {divline!r}')
                break
            else:
                matched = _parse_row(line, colspecs)
                if not matched:
                    raise TableParserError(f'expected columns line, got {line!r}')
                colsline = line
    columns = matched
    header = colspecs, columns, div

#    print('done')
    return preface, header, divline, colsline, prediv


def _parse_divline(line, *, minwidth=_MINWIDTH):
    if not minwidth or minwidth < 0:
        minwidth = None

    nospaces = line.replace(' ', '')

    div = nospaces[0]
    if div.isalnum():
        return None
    if div * len(nospaces) != nospaces:
        return None

    widths = [len(v) for v in line.split()]
    sep = line[:len(line) - len(line.lstrip())] or None
    seps = [sep] * len(widths)

    rest = line
    for i, width in enumerate(widths):
        if minwidth and width < minwidth:
            return None
        sepwidth = len(rest) - len(rest.lstrip())
        seps[i] = rest[:sepwidth]
        #assert sep <= 1 or seps[i] == seps[1]
        rest = rest[sepwidth + width:]
    if rest != '':
        raise TableParserError(f'trailing spces not allowed in div line, got {line!r}')

    colspecs = list(zip(seps, widths))
    return colspecs, div


def _parse_rows(lines, colspecs, divline, colsline, prediv):
#    print('rows')
    if prediv:
        yield DIVIDER
    yield COLUMNS
    yield DIVIDER
    for line, ignored in _mark_ignored_lines(lines):
        if ignored:
            yield line
        elif line == divline:
            yield DIVIDER
        elif line == colsline:
            yield COLUMNS
        else:
            row = _parse_row(line, colspecs, allowlong=True)
            if not row:
                # We hit the end of the table.
                yield line
                break
            yield row
#    print('done')


def _parse_row(line, colspecs, allowlong=False):
    rest = line
    row = []
    for sep, width in colspecs:
#        print(f'+++ {rest!r}')
        if not rest.startswith(sep):
#            print(' (bad prefix)')
            return None
        rest = rest[len(sep):]
        row.append(rest[:width])
        rest = rest[width:]
    if rest and not allowlong:
        return None
    row[-1] += rest
    return row


def _mark_ignored_lines(lines):
#    print('marking')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            yield line, True
        elif stripped.startswith('#'):
            yield line, True
        else:
            yield line, False


def _render_table(header, rows):
    colspecs, columns, div = header

    colsline = _render_row(columns, colspecs)
    divline = ''.join(s + div * w for s, w in colspecs)

    rows = iter(rows)

    # Render the header.
    firstrow = None
    header = []
    for row in rows:
        if row is COLUMNS or row is DIVIDER:
            header.append(row)
        else:
            firstrow = row
            break
    if not header:
        yield colsline
        yield divline
    elif header == [COLUMNS, DIVIDER]:
        yield colsline
        yield divline
    elif header == [DIVIDER, COLUMNS, DIVIDER]:
        yield divline
        yield colsline
        yield divline
    else:
        raise NotImplementedError(header)

    # Render the rows.
    for row in itertools.chain([firstrow], rows):
        if row is COLUMNS:
            yield colsline
        elif row is DIVIDER:
            yield divline
        elif isinstance(row, str):
            assert '\n' not in row, repr(row)
            yield row
        else:
            yield _render_row(row, colspecs, allowlong=True)


def _render_row(row, colspecs, allowlong=False):
    rendered = ''
    ncols = len(colspecs)
    assert len(row) == ncols, (row, colspecs)
    for i, spec, value in zip(range(ncols), colspecs, row):
        sep, width = spec
        assert isinstance(value, str), repr(value)
        if i == ncols - 1:
            if not allowlong and len(value) > width:
                raise TableRendererError(f'value does not fit ({value!r} > {width})')
            rendered += sep + value
        else:
            if len(value) > width:
                print(row)
                raise TableRendererError(f'value does not fit ({value!r} > {width})')
            rendered += sep + value.rjust(width)
    return rendered
