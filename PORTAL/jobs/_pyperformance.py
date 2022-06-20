from collections import namedtuple
import gzip
import hashlib
import json
import logging
import os
import os.path
import platform
import re
import sys

from . import _utils


logger = logging.getLogger(__name__)


##################################
# pyperformance helpers

class Benchmarks:

    REPOS = os.path.join(_utils.HOME, 'repos')
    SUITES = {
        'pyperformance': {
            'url': 'https://github.com/python/pyperformance',
            'reldir': 'pyperformance/data-files/benchmarks',
        },
        'pyston': {
            'url': 'https://github.com/pyston/python-macrobenchmarks',
            'reldir': 'benchmarks',
        },
    }

    def __init__(self):
        self._cache = {}

    def load(self):
        """Return the per-suite lists of benchmarks."""
        benchmarks = {}
        for suite, info in self.SUITES.items():
            if suite in self._cache:
                benchmarks[suite] = list(self._cache[suite])
                continue
            url = info['url']
            reldir = info['reldir']
            reporoot = os.path.join(self.REPOS,
                                    os.path.basename(url))
            if not os.path.exists(reporoot):
                if not os.path.exists(self.REPOS):
                    os.makedirs(self.REPOS)
                _utils.git('clone', url, reporoot, cwd=None)
            names = self._get_names(os.path.join(reporoot, reldir))
            benchmarks[suite] = self._cache[suite] = names
        return benchmarks

    def _get_names(self, benchmarksdir):
        manifest = os.path.join(benchmarksdir, 'MANIFEST')
        if os.path.isfile(manifest):
            with open(manifest) as infile:
                for line in infile:
                    if line.strip() == '[benchmarks]':
                        for line in infile:
                            if line.strip() == 'name\tmetafile':
                                break
                        else:
                            raise NotImplementedError(manifest)
                        break
                else:
                    raise NotImplementedError(manifest)
                for line in infile:
                    if line.startswith('['):
                        break
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    name, _ = line.split('\t')
                    yield name
        else:
            for name in os.listdir(benchmarksdir):
                if name.startswith('bm_'):
                    yield name[3:]


class PyperfUploadID(namedtuple('PyperfUploadName',
                                'impl version commit host compatid suite')):
    # See https://github.com/faster-cpython/ideas/tree/main/benchmark-results/README.md
    # for details on this filename format.

    REGEX = re.compile(r'''
        # We do no strip leading/trailing whitespace in this regex.
        ^
        ( .*? )  # <prefix>
        ( \w+ )  # <impl>
        -
        ( main | \d\.\d+\. (?: 0a\d+ | 0b\d+ | 0rc\d+ | [1-9]\d* ) )  # <version>
        -
        ( [0-9a-f]{10} )  # <commit>
        -
        ( [^-\s]+ )  # <host>
        -
        ( [0-9a-f]{12} )  # <compatid>
        (?:
            -
            ( pyston )  # <suite>
        )?
        #( \.json (?: \.gz )? )?  # <suffix>
        ( \. .*? )?  # <suffix>
        $
    ''', re.VERBOSE)

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            self = cls.parse(raw)
            if not self:
                return cls.from_filename(raw)
        else:
            raise TypeError(raw)

    @classmethod
    def from_filename(cls, filename):
        # XXX Add a "checkexists" option?
        basename = os.path.basename(filename)
        self = cls._parse(basename)
        if self is None:
            return None
        if self.name != basename:
            filename = os.path.abspath(filename)
        self._filename = filename
        return self

    @classmethod
    def parse(cls, name, *, allowprefix=False, allowsuffix=False):
        self = cls._parse(name)
        if self:
            if not allowprefix and self._prefix:
                return None
            if not allowsuffix and self._suffix:
                return None
        return self

    @classmethod
    def _parse(cls, uploadid):
        m = cls.REGEX.match(uploadid)
        if not m:
            return None
        (prefix, impl, verstr, commit, host, compatid, suite, suffix,
         ) = m.groups()
        name = uploadid
        if prefix:
            name = name[len(prefix):]
        if suffix:
            name = name[:-len(suffix)]
        impl = _utils.resolve_python_implementation(impl)
        if verstr == 'main':
            version = impl.VERSION.resolve_main()
            name = name.replace('-main-', f'-{version}-')
        else:
            version = impl.parse_version(verstr)
        self = cls(impl, version, commit, host, compatid, suite)
        self._name = name
        self._prefix = prefix or None
        self._suffix = suffix or None
        return self

    @classmethod
    def from_metadata(cls, metadata, version=None,
                      commit=None,
                      host=None,
                      impl=None,
                      suite=None,
                      ):
        metadata = PyperfResultsMetadata.from_raw(metadata)
        impl = _utils.resolve_python_implementation(
            impl or metadata.python_implementation or 'cpython',
        )
        if not version or version == 'main':
            # We assume "main".
            version = impl.VERSION.resolve_main()
        else:
            version = impl.parse_version(version, requirestr=False)
        self = cls(
            impl=impl,
            version=version,
            commit=metadata.commit,
            host=host or metadata.host,
            compatid=metadata.compatid,
            suite=suite,
        )
        return self

    def __str__(self):
        return self.name

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            impl, version, commit, host, compatid, suite = self
            name = f'{impl}-{version}-{commit[:10]}-{host}-{compatid[:12]}'
            if suite and suite != 'pyperformance':
                name = f'{name}-{suite}'
            self._name = name
            return name

    @property
    def implementation(self):
        return self.impl

    @property
    def filename(self):
        try:
            return self._filename
        except AttributeError:
            filename, *_ = self.resolve_filenames()
            return filename

    def resolve_filenames(self, *, dirname=True, prefix=True, suffix=True):
        dirnames = []
        if dirname is True:
            if hasattr(self, '_dirname') and self._dirname:
                dirnames = [self._dirname]
        elif dirname:
            if isinstance(dirname, str):
                dirnames = [dirname]
            else:
                dirnames = list(dirname)
            if any(not d for d in dirnames):
                raise ValueError(f'blank dirname in {dirname}')

        prefixes = [None]
        if prefix is True:
            if hasattr(self, '_prefix') and self._prefix:
                prefixes = [self._prefix]
        elif prefix:
            if isinstance(prefix, str):
                prefixes = [prefix]
            else:
                prefixes = list(prefix)
            if any(not p for p in prefixes):
                raise ValueError(f'blank prefix in {prefix}')

        suffixes = [None]
        if suffix is True:
            if hasattr(self, '_suffix') and self._suffix:
                suffixes = [self._suffix]
        elif suffix:
            if isinstance(suffix, str):
                suffixes = [suffix]
            else:
                suffixes = list(suffix)
            if any(not s for s in suffixes):
                raise ValueError(f'blank suffix in {suffix}')

        name = self.name
        # XXX Ignore duplicates?
        for suffix in suffixes:
            for prefix in prefixes:
                filename = f'{prefix or ""}{name}{suffix or ""}'
                if dirnames:
                    for dirname in dirnames:
                        yield os.path.join(dirname, filename)
                else:
                    yield filename

    def copy(self, **replacements):
        if not replacements:
            return self
        kwargs = dict(self._asdict(), **replacements)
        # XXX Validate the replacements.
        cls = type(self)
        copied = cls(**kwargs)
        # We do not copy self._name.
        suffix = getattr(self, '_suffix', None)
        if suffix:
            copied._suffix = suffix
        dirname = getattr(self, '_dirname', None)
        if dirname:
            copied._dirname = dirname
        elif hasattr(self, '_filename') and self._filename:
            copied._dirname = os.path.dirname(filename)
        return copied


##################################
# results comparisons

class PyperfComparisonValue:
    """A single value reported by pyperf when comparing to sets of results."""

    IGNORED = 'not significant'
    BASELINE = '(ref)'

    REGEX = re.compile(rf'''
        ^
        (?:
            (?:
                {_utils.ElapsedTimeWithUnits.PAT}
             )  # <elapsed1> <units1>
            |
            (?:
                (?:
                    {_utils.ElapsedTimeWithUnits.PAT}
                    : \s+
                 )?  # <elapsed2> <units2>
                (?:
                    {_utils.ElapsedTimeComparison.PAT}
                 )  # <value> <direction>
             )
            |
            ( {re.escape(IGNORED)} )  # <ignored>
            |
            ( {re.escape(BASELINE)} )  # <baseline>
         )
        $
    ''', re.VERBOSE)

    @classmethod
    def parse(cls, valuestr, *, fail=False):
        m = cls.REGEX.match(valuestr)
        if not m:
            if fail:
                print(cls.REGEX.pattern)
                raise ValueError(f'could not parse {valuestr!r}')
            return None
        (elapsed1, units1,
         elapsed2, units2, value, direction,
         ignored,
         baseline,
         ) = m.groups()
        get_elapsed = _utils.ElapsedTimeWithUnits.from_values
        get_comparison = _utils.ElapsedTimeComparison.from_parsed_values
        if elapsed1:
            elapsed = get_elapsed(elapsed1, units1)
            comparison = None
        elif value:
            if elapsed2:
                elapsed = get_elapsed(elapsed2, units2)
            comparison = get_comparison(value, direction)
        elif ignored:
            elapsed = None
            comparison = None
        elif baseline:
            elapsed = None
            comparison = cls.BASELINE
        else:
            raise NotImplementedError(valuestr)
        return cls(elapsed, comparison)

    def __init__(self, elapsed, comparison):
        self._elapsed = elapsed
        self._comparison = comparison

    def __repr__(self):
        return f'{type(self).__name__}({self._elapsed!r}, {self._comparison!r})'

    def __str__(self):
        if self._elapsed:
            if self._comparison:
                return f'{self._elapsed}: {self._comparison}'
            else:
                return str(self._elapsed)
        else:
            if self._comparison:
                return str(self._comparison)
            else:
                return self.IGNORED

    @property
    def elapsed(self):
        return self._elapsed

    @property
    def comparison(self):
        return self._comparison

    @property
    def isbaseline(self):
        if self._elapsed and not self._comparison:
            return True
        return self._comparison == self.BASELINE


class PyperfComparisonBaseline:
    """The filename and set of result values for baseline results."""

    def __init__(self, source, byname):
        if not source:
            raise ValueError('missing source')
        # XXX Validate source as a filename.
        #elif not os.path.isabs(source):
        #    raise ValueError(f'expected an absolute source, got {source!r}')
        _byname = {}
        for name, value in byname.items():
            assert name and isinstance(name, str), (name, value, byname)
            assert value and isinstance(value, str), (name, value, byname)
            _byname[name] = _utils.ElapsedTimeWithUnits.parse(value, fail=True)
        self._source = source or None
        self._byname = byname

    def __repr__(self):
        return f'{type(self).__name__}({self._source!r}, {self._byname!r})'

    def __str__(self):
        return f'<baseline {self._source!r}>'

    @property
    def source(self):
        return self._source

    @property
    def byname(self):
        return dict(self._byname)


class PyperfComparison:
    """The per-benchmark differences between one results set and a baseline.

    The comparison values are a mapping from benchmark name to the
    relative differences (e.g. "1.04x faster").  The geometric mean
    is also provided.
    """

    _fields = 'baseline source byname mean'.split()

    Summary = namedtuple('Summary',
                         'bench baseline baseresult source result comparison')

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            raise ValueError('missing comparison')
        elif isinstance(raw, cls):
            return raw
        else:
            raise TypeError(raw)

    def __init__(self, baseline, source, byname, mean):
        if not baseline:
            raise ValueError('missing baseline')
        elif not isinstance(baseline, PyperfComparisonBaseline):
            raise TypeError(baseline)
        if not source:
            raise ValueError('missing source')
        # XXX Validate source as a filename.
        #elif not os.path.isabs(source):
        #    raise ValueError(f'expected an absolute source, got {source!r}')

        _byname = {}
        for name, value in byname.items():
            assert name and isinstance(name, str), (name, value, byname)
            assert value and isinstance(value, str), (name, value, byname)
            _byname[name] = PyperfComparisonValue.parse(value, fail=True)
        if sorted(_byname) != sorted(baseline.byname):
            raise ValueError(f'mismatch with baseline ({sorted(_byname)} != {sorted(baseline.byname)})')
        self._baseline = baseline
        self._source = source
        self._byname = _byname
        self._mean = _utils.ElapsedTimeComparison.parse(mean)

    def __repr__(self):
        values = [f'{a}={getattr(self, "_"+a)!r}' for a in self._fields]
        return f'{type(self).__name__}({", ".join(values)})'

    def __str__(self):
        return f'<{self._mean} ({self._source})>'

    @property
    def baseline(self):
        return self._baseline

    @property
    def source(self):
        return self._source

    @property
    def byname(self):
        return dict(self._byname)

    @property
    def mean(self):
        return self._mean

    def look_up(self, name):
        return self.Summary(
            name,
            self._baseline.source,
            self._baseline.byname[name],
            self._source,
            self._byname[name].elapsed,
            self._byname[name].comparison,
        )


class PyperfComparisons:
    """The baseline and comparisons for a set of results."""

    @classmethod
    def parse_table(cls, text):
        table = PyperfTable.parse(text)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        base_byname = {}
        bysource = {s: {} for s in table.header.others}
        means = {s: None for s in table.header.others}
        for row in table.rows:
            values = row.valuesdict
            if row.name == 'Geometric mean':
                for source in means:
                    means[source] = values[source]
            else:
                base_byname[row.name] = row.baseline
                for source, byname in bysource.items():
                    byname[row.name] = values[source]
        for source, byname in bysource.items():
            bysource[source] = (byname, means[source])
        self = cls(
            PyperfComparisonBaseline(table.header.baseline, base_byname),
            bysource,
        )
        self._table = table
        return self

    def __init__(self, baseline, bysource):
        # baseline is checked in PyperfComparison.__init__().
        if not bysource:
            raise ValueError('missing bysource')
        _bysource = {}
        for source, data in bysource.items():
            byname, mean = data
            _bysource[source] = PyperfComparison(baseline, source, byname, mean)
        self._baseline = baseline
        self._bysource = _bysource

    @property
    def baseline(self):
        return self._baseline

    @property
    def bysource(self):
        return dict(self._bysource)

    @property
    def table(self):
        try:
            return self._table
        except AttributeError:
            raise NotImplementedError


class PyperfTableParserError(ValueError):
    MSG = 'failed parsing results table'
    FIELDS = 'text reason'.split()
    def __init__(self, text, reason=None, msg=None):
        self.text = text
        self.reason = reason
        if not msg:
            msg = self.MSG
            if reason:
                msg = f'{msg} ({{reason}})'
        msgkwargs = {name: getattr(self, name) for name in self.FIELDS or ()}
        super().__init__(msg.format(**msgkwargs))


class PyperfTableRowParserError(PyperfTableParserError):
    MSG = 'failed parsing table row line {line!r}'
    FIELDS = 'line'.split()
    FIELDS = PyperfTableParserError.FIELDS + FIELDS
    def __init__(self, line, reason=None, msg=None):
        self.line = line
        super().__init__(line, reason, msg)


class PyperfTableRowUnsupportedLineError(PyperfTableRowParserError):
    MSG = 'unsupported table row line {line!r}'
    def __init__(self, line, msg=None):
        super().__init__(line, 'unsupported', msg)


class PyperfTableRowInvalidLineError(PyperfTableRowParserError):
    MSG = 'invalid table row line {line!r}'
    def __init__(self, line, msg=None):
        super().__init__(line, 'invalid', msg)


class PyperfTable:

    FORMATS = ['raw', 'meanonly']

    @classmethod
    def parse(cls, text):
        lines = iter(text.splitlines())
        # First parse the header.
        for line in lines:
            header = PyperfTableHeader.parse(line)
            if header:
                try:
                    headerdiv = next(lines)
                except StopIteration:
                    continue
                if not headerdiv.startswith('+='):
                #if headerdiv != header.div:
                    raise ValueError('bad table text')
                break
        else:
            return None
        # Then parse the rows.
        row_cls = PyperfTableRow.subclass_from_header(header)
        rows = []
        for line in lines:
            try:
                row = row_cls.parse(line, fail=True)
            except PyperfTableRowUnsupportedLineError as exc:
                if not line:
                    # end-of-table
                    break
                elif line.startswith('Ignored benchmarks '):
                    # end-of-table
                    ignored, _ = cls._parse_ignored(line, lines)
                    # XXX Add the names to the table.
                    line = _utils.get_next_line(lines, skipempty=True)
                    break
                elif not line.startswith('#'):
                    raise  # re-raise
            else:
                if row:
                    rows.append(row)
        hidden, _ = cls._parse_hidden(None, lines, required=False)
        if hidden:
            # XXX Add the names to the table.
            pass
        # Finally, create and return the table object.
        self = cls(rows, header)
        self._text = text
        return self

    @classmethod
    def _parse_names_list(cls, line, lines, prefix=None):
        while not line:
            try:
                line = next(lines)
            except StopIteration:
                return None, None
        pat = r'(\w+(?:[^:]*\w)?)'
        if prefix:
            pat = f'{prefix}: {pat}'
        m = re.match(f'^{pat}$', line)
        if not m:
            return None, line
        count, names = m.groups()
        names = names.replace(',', ' ').split()
        if names and len(names) != int(count):
            raise PyperfTableParserError(line, f'expected {count} names, got {names}')
        return names, line

    @classmethod
    def _parse_ignored(cls, line, lines, required=True):
        # Ignored benchmarks (2) of benchmark-results/cpython-3.10.4-9d38120e33-fc_linux-42d6dd4409cb.json: genshi_text, genshi_xml
        prefix = r'Ignored benchmarks \((\d+)\) of \w+.*\w'
        names, line = cls._parse_names_list(line, lines, prefix)
        if not names and required:
            raise PyperfTableParserError(line, 'expected "Ignored benchmarks..."')
        return names, line

    @classmethod
    def _parse_hidden(cls, line, lines, required=True):
        # Benchmark hidden because not significant (6): unpickle, scimark_sor, sqlalchemy_imperative, sqlite_synth, json_loads, xml_etree_parse
        prefix = r'Benchmark hidden because not significant \((\d+)\)'
        names, line = cls._parse_names_list(line, lines, prefix)
        if not names and required:
            raise PyperfTableParserError(line, 'expected "Benchmarks hidden..."')
        return names, line

    def __init__(self, rows, header=None):
        if not isinstance(rows, tuple):
            rows = tuple(rows)
        if not header:
            header = rows[0].header
        if header[0] != 'Benchmark':
            raise ValueError(f'unsupported header {header}')
        self.header = header
        self.rows = rows

    def __repr__(self):
        return f'{type(self).__name__}({self.rows}, {self.header})'

    @property
    def mean_row(self):
        try:
            return self._mean_row
        except AttributeError:
            for row in self.rows:
                if row.name == 'Geometric mean':
                    break
            else:
                row = None
            self._mean_row = row
            return self._mean_row

    def render(self, fmt=None):
        if not fmt:
            fmt = 'raw'
        if fmt == 'raw':
            text = getattr(self, '_text', None)
            if text:
                yield from text.splitlines()
            else:
                div = self.header.rowdiv
                yield div
                yield from self.header.render(fmt)
                yield self.header.div
                for row in self.rows:
                    yield from row.render(fmt)
                    yield div
        elif fmt == 'meanonly':
            text = getattr(self, '_text', None)
            if text:
                lines = text.splitlines()
                yield from lines[:3]
                yield from lines[-2:]
            else:
                div = self.header.rowdiv
                yield div
                yield from self.header.render('raw')
                yield self.header.div
                yield from self.mean_row.render('raw')
                yield div
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')


class _PyperfTableRowBase(tuple):

    @classmethod
    def parse(cls, line, *, fail=False):
        values = cls._parse(line, fail)
        if not values:
            return None
        self = tuple.__new__(cls, values)
        self._raw = line
        return self

    @classmethod
    def _parse(cls, line, fail=False):
        line = line.rstrip()
        if line.startswith('+'):
            return None
        elif not line.startswith('|') or not line.endswith('|'):
            if fail:
                raise PyperfTableRowUnsupportedLineError(line)
            return None
        values = tuple(v.strip() for v in line[1:-1].split('|'))
        if not values:
            raise PyperfTableRowInvalidLineError(line, 'missing name column')
        elif len(values) < 3:
            raise PyperfTableRowInvalidLineError(line, 'expected 2+ sources')
        return values

    def __new__(cls, name, baseline, *others):
        raise TypeError(f'not supported; use {cls.__name__}.parse() instead')

    @property
    def raw(self):
        return self._raw

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            self._name = self[0]
            return self._name

    @property
    def values(self):
        try:
            return self._values
        except AttributeError:
            self._values = self[1:]
            return self._values

    @property
    def baseline(self):
        try:
            return self._baseline
        except AttributeError:
            self._baseline = self[1]
            return self._baseline

    @property
    def others(self):
        try:
            return self._others
        except AttributeError:
            self._others = self[2:]
            return self._others

    def render(self, fmt=None):
        if not fmt:
            fmt = 'raw'
        if fmt == 'raw':
            raw = getattr(self, '_raw', None)
            if raw:
                return raw
            raise NotImplementedError
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')


class PyperfTableHeader(_PyperfTableRowBase):

    label = _PyperfTableRowBase.name
    sources = _PyperfTableRowBase.values

    @property
    def div(self):
        return '+=' + '=+='.join('=' * len(v) for v in self) + '=+'

    @property
    def rowdiv(self):
        return '+-' + '-+-'.join('-' * len(v) for v in self) + '-+'

    @property
    def indexpersource(self):
        return dict(zip(self.sources, range(len(self.sources))))


class PyperfTableRow(_PyperfTableRowBase):

    @classmethod
    def subclass_from_header(cls, header):
        if cls is not PyperfTableRow:
            raise TypeError('not supported for subclasses')
        if not header:
            raise ValueError('missing header')
        class _PyperfTableRow(PyperfTableRow):
            @classmethod
            def parse(cls, line, *, _header=header, fail=False):
                return super().parse(line, _header, fail=fail)
        return _PyperfTableRow

    @classmethod
    def parse(cls, line, header, fail=False):
        self = super().parse(line, fail=fail)
        if not self:
            return None
        if len(self) != len(header):
            raise ValueError(f'expected {len(header)} values, got {tuple(self)}')
        self._header = header
        return self

    @property
    def header(self):
        return self._header

    @property
    def valuesdict(self):
        return dict(zip(self.header.sources, self.values))

    def look_up(self, source):
        return self.valuesdict[source]


##################################
# results

def normalize_results_filename(filename, resultsroot=None):
    if not filename:
        raise ValueError('missing filename')
    resultsroot = os.path.abspath(resultsroot) if resultsroot else None
    if os.path.isabs(filename):
        if resultsroot:
            relfile = os.path.relpath(filename, resultsroot)
            if relfile.startswith('..' + os.path.sep):
                raise ValueError(f'filename does not match resultsroot ({filename!r}, {resultsroot!r})')
        else:
            resultsroot, relfile = os.path.split(filename)
    else:
        if resultsroot:
            relfile = filename
            filename = os.path.join(resultsroot, relfile)
        else:
            raise ValueError('missing resultsroot')
    return filename, relfile, resultsroot


class PyperfResultsFile:

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls(raw)
        else:
            raise TypeError(raw)

    def __init__(self, filename, uploadid=None, resultsroot=None):
        if not filename:
            raise ValueError('missing filename')
        (filename, relfile, resultsroot,
         ) = normalize_results_filename(filename, resultsroot)
        if uploadid:
            uploadid = PyperfUploadID.from_raw(uploadid)
        else:
            uploadid = PyperfUploadID.from_filename(filename)

        self._filename = filename
        self._relfile = relfile
        self._resultsroot = resultsroot
        self._uploadid = uploadid

    def __repr__(self):
        return f'{type(self).__name__}({self.filename!r})'

    def __str__(self):
        return self._filename

    @property
    def filename(self):
        return self._filename

    @property
    def relfile(self):
        return self._relfile

    @property
    def resultsroot(self):
        return self._resultsroot

    def read(self, host=None, version=None):
        filename = self._filename
        if filename.endswith('.json.gz'):
            _open = gzip.open
        elif filename.endswith('.json'):
            _open = open
        else:
            raise NotImplementedError(filename)
        with _open(filename) as infile:
            data = json.load(infile)
        return PyperfResults(
            data,
            self,
            version,
            host,
            uploadid=self._uploadid,
        )

    def compare(self, others):
        optional = []
        if len(others) == 1:
            optional.append('--group-by-speed')
        cwd = self._resultsroot
        proc = _utils.run_fg(
            sys.executable, '-m', 'pyperf', 'compare_to',
            *(optional),
            '--table',
            self._relfile,
            *(os.path.relpath(o.filename, cwd)
              for o in others),
            cwd=cwd,
        )
        if proc.returncode:
            logger.warn(proc.stdout)
            return None
        return PyperfComparisons.parse_table(proc.stdout)
#        return PyperfTable.parse(proc.stdout)


class PyperfResults:

    def __init__(self, data, resfile=None, version=None, host=None, *,
                 suite=None,
                 uploadid=None,
                 ):
        if data:
            self._set_data(data)
        elif not resfile:
            raise ValueError('missing resfile')
        self.resfile = PyperfResultsFile.from_raw(resfile) if resfile else None
        self.version = version
        if host:
            self._host = host
        self.suite = suite or None
        if uploadid:
            self._uploadid = PyperfUploadID.from_raw(uploadid)
            if not version:
                self.version = self._uploadid.version

    def _set_data(self, data):
        if hasattr(self, '_data'):
            raise Exception('already set')
        # XXX Validate?
        if data['version'] == '1.0':
            self._data = data
        else:
            raise NotImplementedError(data['version'])

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            data = self._read(self.filename)
            self._set_data(data)
            return self._data

    @property
    def metadata(self):
        return PyperfResultsMetadata.from_full_results(self.data)

    @property
    def filename(self):
        if not self.resfile:
            return None
        return self.resfile.filename

    @property
    def uploadid(self):
        try:
            return self._uploadid
        except AttributeError:
            self._uploadid = PyperfUploadID.from_metadata(
                self.metadata,
                version=self.version,
                host=getattr(self, '_host', None) or None,
                suite=self.suite,
            )
            return self._uploadid

    @property
    def host(self):
        try:
            return self._host
        except AttributeError:
            self._host = self.uploadid.host
            return self._host

    @property
    def build(self):
        # XXX Add to PyperfUploadID?
        # XXX Extract from metadata?
        return ['PGO', 'LTO']

    def split_benchmarks(self):
        """Return results collated by suite."""
        if self.suite:
            raise Exception(f'already split ({self.suite})')
        by_suite = {}
        benchmarks = Benchmarks().load()
        by_name = {}
        for suite, names in benchmarks.items():
            for name in names:
                if name in by_name:
                    raise NotImplementedError((suite, name))
                by_name[name] = suite
        results = self.data
        for data in results['benchmarks']:
            name = data['metadata']['name']
            try:
                suite = by_name[name]
            except KeyError:
                # Some benchmarks actually produce results for
                # sub-benchmarks (e.g. logging -> logging_simple).
                _name = name
                while '_' in _name:
                    _name, _, _ = _name.rpartition('_')
                    if _name in by_name:
                        suite = by_name[_name]
                        break
                else:
                    suite = 'unknown'
            if suite not in by_suite:
                by_suite[suite] = {k: v
                                   for k, v in results.items()
                                   if k != 'benchmarks'}
                by_suite[suite]['benchmarks'] = []
            by_suite[suite]['benchmarks'].append(data)
        cls = type(self)
        for suite, data in by_suite.items():
            host = getattr(self, '_host', None)
            results = cls(None, self.filename, self.version, host, suite=suite)
            results._data = data
            if hasattr(self, '_uploadid') and self._uploadid:
                results._uploadid = self._uploadid.copy(suite=suite)
            by_suite[suite] = results
        return by_suite

    #def compare(self, others):
    #    raise NotImplementedError


class PyperfResultsMetadata:

    EXPECTED = {
        # top-level
        "aslr",
        "boot_time",
        "commit_branch",
        "commit_date",
        "commit_id",
        "cpu_affinity",
        "cpu_config",
        "cpu_count",
        "cpu_model_name",
        "hostname",
        "performance_version",
        "platform",
        "unit",
        # per-benchmark
        "perf_version",
        "python_cflags",
        "python_compiler",
        "python_implementation",
        "python_version",
        "timer",
        "runnable_threads",
    }

    @classmethod
    def from_raw(cls, raw):
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        else:
            raise TypeError(raw)

    @classmethod
    def from_full_results(cls, data):
        metadata = dict(data['metadata'])
        for key, value in cls._merge_from_benchmarks(data['benchmarks']).items():
            if key in metadata:
                if value != metadata[key]:
                    logger.warn(f'metadata mismatch for {key} (top: {metadata[key]!r}, bench: {value!r}); ignoring')
            else:
                metadata[key] = value
        return cls(metadata, data['version'])

    @classmethod
    def _merge_from_benchmarks(cls, data):
        metadata = {}
        for bench in data:
            for key, value in bench['metadata'].items():
                if key not in cls.EXPECTED:
                    continue
                if not value:
                    continue
                if key in metadata:
                    if metadata[key] is None:
                        continue
                    if value != metadata[key]:
                        logger.warn(f'metadata mismatch for {key} ({value!r} != {metadata[key]!r}); ignoring')
                        metadata[key] = None
                else:
                    metadata[key] = value
        for key, value in list(metadata.items()):
            if value is None:
                del metadata[key]
        return metadata

    def __init__(self, data, version=None):
        self._data = data
        self._version = version

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)

    @property
    def version(self):
        return self._version

    @property
    def commit(self):
        return self._data['commit_id']

    @property
    def python_implementation(self):
        return self._data.get('python_implementation')

    @property
    def host(self):
        # We could use metadata['hostname'] but that doesn't
        # make a great label in the default case.
        host = self.os_name
        if self.arch in ('arm32', 'arm64'):
            host += '-arm'
        # Ignore everything else.
        return host

    @property
    def os_name(self):
        try:
            return self._os_name
        except AttributeError:
            platform = self._data['platform'].lower()
            if 'linux' in platform:
                name = 'linux'
            elif 'darwin' in platform or 'macos' in platform or 'osx' in platform:
                name = 'mac'
            elif 'win' in platform:
                name = 'windows'
            else:
                raise NotImplementedError(platform)
            self._os_name = name
            return name

    @property
    def arch(self):
        try:
            return self._arch
        except AttributeError:
            platform = metadata['platform'].lower()
            if 'x86_64' in platform:
                arch = 'x86_64'
            elif 'amd64' in platform:
                arch = 'amd64'

            procinfo = metadata['cpu_model_name'].lower()
            if 'aarch64' in procinfo:
                arch = 'arm64'
            elif 'arm' in procinfo:
                if '64' in procinfo:
                    arch = 'arm64'
                else:
                    arch = 'arm32'
            elif 'intel' in procinfo:
                arch = 'x86_64'
            else:
                raise NotImplementedError((platform, procinfo))
            self._arch = arch
            return arch

    @property
    def compatid(self):
        data = [
           self._data['hostname'],
           self._data['platform'],
           self._data.get('perf_version'),
           self._data['performance_version'],
           self._data['cpu_model_name'],
           self._data.get('cpu_freq'),
           self._data['cpu_config'],
           self._data.get('cpu_affinity'),
        ]
        h = hashlib.sha256()
        for value in data:
            if not value:
                continue
            h.update(value.encode('utf-8'))
        return h.hexdigest()


##################################
# results storage

class PyperfResultsStorage:

    def iter_all(self):
        raise NotImplementedError

    def get(self, uploadid):
        raise NotImplementedError

    def match(self, specifier):
        raise NotImplementedError

    def add(self, results, *, unzipped=True):
        raise NotImplementedError


class PyperfResultsRepo(PyperfResultsStorage):

    BRANCH = 'add-benchmark-results'

    SUFFIX = '.json'
    COMPRESSED_SUFFIX = '.json.gz'
    COMPRESSOR = gzip

    INDEX = 'index.json'
    BASELINE = '3.10.4'

    def __init__(self, root, remote=None, datadir=None):
        if root:
            root = os.path.abspath(root)
        if remote:
            if isinstance(remote, str):
                remote = _utils.GitHubTarget.resolve(remote, root)
            elif not isinstance(remote, _utils.GitHubTarget):
                raise TypeError(f'unsupported remote {remote!r}')
            root = remote.ensure_local(root)
        else:
            if root:
                if not os.path.exists(root):
                    raise FileNotFoundError(root)
                #_utils.verify_git_repo(root)
            else:
                raise ValueError('missing root')
            remote = None
        self.root = root
        self.remote = remote
        self.datadir = datadir or None

    def _git(self, *args, cfg=None):
        ec, text = _utils.git(*args, cwd=self.root, cfg=cfg)
        if ec:
            raise NotImplementedError((ec, text))
        return text

    @property
    def _suffixes(self):
        return [self.SUFFIX, self.COMPRESSED_SUFFIX]

    @property
    def _dataroot(self):
        if self.datadir:
            return os.path.join(self.root, self.datadir)
        else:
            return self.root

    @property
    def _indexfile(self):
        return os.path.join(self._dataroot, self.INDEX)

    def _get_index(self):
        filename = self._indexfile
        try:
            return PyperfResultsIndexFile.load(filename)
        except FileNotFoundError:
            index = PyperfResultsIndexFile.from_results_dir(
                self._dataroot,
                filename,
            )
            index.save()
            return index

    def iter_all(self):
        for name in os.listdir(self._dataroot):
            res = PyperfUploadID.parse(name, allowsuffix=True)
            if res:
                yield res

    def get(self, uploadid):
        if not uploadid:
            return None
        found = self._match_uploadid(uploadid)
        if not found:
            raise TypeError(uploadid)

        matched = None
        for filename in self._resolve_filenames(found):
            if not os.path.exists(filename):
                continue
            if matched:
                raise RuntimeError('matched multiple, consider using match()')
            matched = filename
        if matched:
            uploadid = found
            return PyperfResultsFile(filename, uploadid, self.root)
        return None

    def match(self, specifier, suites=None):
        for uploadid in self._match(specifier, suites):
            for filename in self._resolve_filenames(uploadid):
                if not os.path.exists(filename):
                    continue
                yield PyperfResultsFile(filename, uploadid, self.root)

    def _resolve_filenames(self, uploadid, suffix=None):
        return uploadid.resolve_filenames(
            dirname=self._dataroot,
            prefix=None,
            suffix=self._suffixes if suffix is None else suffix,
        )

    def _match(self, specifier, suites):
        # specifier: uploadID, version, filename
        if not specifier:
            return

        uploadid = self._match_uploadid(specifier)
        if uploadid:
            if suites:
                for suite in suites:
                    yield uploadid.copy(suite=suite)
            else:
                yield uploadid
            return

        matched = False
        for uploadid in self._match_versions(specifier):
            matched = True
            if suites and uploadid.suite not in suites:
                continue
            yield uploadid
        if matched:
            return

        #if isinstance(specifier, str):
        #    yield from self._match_uploadid_pattern(specifier)

        return None

    def _match_uploadid(self, uploadid):
        orig = uploadid
        if isinstance(orig, str):
            uploadid = PyperfUploadID.parse(orig)
            if not uploadid:
                uploadid = PyperfUploadID.from_filename(orig)
        elif isinstance(orig, PyperfUploadID):
            uploadid = orig
        else:
            return None
        return uploadid

    def _match_uploadid_pattern(self, pat):
        raise NotImplementedError

    def _match_versions(self, version):
        if isinstance(version, str):
            version = _utils.Version.parse(version)
            if not version:
                return
        elif not isinstance(version, _utils.Version):
            return
        # XXX Treat missing micro/release as wildcard?
        version = version.full
        for uploadid in self.iter_all():
            if version == uploadid.version.full:
                yield uploadid

    def add(self,
            results,
            branch=None,
            author=None,
            compressed=False,
            split=True,
            push=True,
            ):
        branch, gitcfg = self._prep_for_commit(branch, author)

        if not isinstance(results, PyperfResults):
            raise NotImplementedError(results)

        source = results.filename
        if source and not os.path.exists(source):
            logger.error(f'results not found at {source}')
            return False

        if split:
            by_suite = results.split_benchmarks()
            if 'other' in by_suite:
                raise NotImplementedError(sorted(by_suite))
        else:
            by_suite = {None: results}

        for suite in sorted(by_suite):
            suite_results = by_suite[suite]
            self._add_locally(suite_results, source, branch, gitcfg, compressed)

        if push:
            self._upload(reltarget)

    def _prep_for_commit(self, branch, author):
        if not branch:
            branch = self.BRANCH

        # We already ran self.remote.ensure_local() in __init__().

        cfg = {}
        if not author:
            pass
        elif isinstance(author, str):
            parsed = _utils.parse_email_address(author)
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

        return branch, cfg

    def _add_locally(self, results, source, branch, gitcfg, compressed=False):
        if results.suite:
            logger.info(f'adding results {source or "???"} ({results.suite})...')
        else:
            logger.info(f'adding results {source or "???"}...')

        reltarget = self._resolve_reltarget(results, compressed)
        logger.info(f'...as {reltarget}...')

        self._git('checkout', '-B', branch)
        self._save(results, reltarget, source, compressed)
        self._git('add', reltarget, self._indexfile)
        msg = f'Add Benchmark Results ({results.uploadid})'
        self._git('commit', '-m', msg, cfg=gitcfg)

        logger.info('...done adding')

    def _resolve_reltarget(self, results, compressed=False):
        reltarget, = results.uploadid.resolve_filenames(
            dirname=self.datadir if self.datadir else None,
            suffix=self.COMPRESSED_SUFFIX if compressed else self.SUFFIX,
        )
        return  reltarget

    def _save(self, results, reltarget, source=None, compressed=False):
        data = results.data
        #if compressed: assert reltarget.endswith(self.COMPRESSED_SUFFIX)
        #else: assert reltarget.endswith(self.SUFFIX)
        target = os.path.join(self.root, reltarget)
#        if source:
#            _, suffix = os.path.splitext(reltarget)
#            if os.path.splitext(source)[1] == suffix:
#                shutil.copyfile(source, target)
#                return
        _open = self.COMPRESSOR.open if compressed else open
        with _open(target, 'w') as outfile:
            json.dump(data, outfile, indent=2)

        # Update the index file.
        index = self._get_index()
        index.add_from_file(target, results)
        index.ensure_means(self.BASELINE)
        index.save()

    def _upload(self, reltarget):
        if not self.remote:
            raise Exception('missing remote')
        url = f'{self.remote.url}/tree/main/{reltarget}'
        logger.info(f'uploading results to {url}...')
        self._git('push', self.remote.push_url)
        logger.info('...done uploading')


class PyperfResultsIndex:

    BASELINE_MEAN = '(ref)'

    def __init__(self):
        self._entries = []

    def _collate_by_suite(self):
        by_suite = {}
        suite_baselines = {}
        for i, entry in enumerate(self._entries):
            suite = entry.uploadid.suite or 'pyperformance'
            try:
                indices = by_suite[suite]
            except KeyError:
                indices = by_suite[suite] = []
            indices.append(i)
            if entry.mean == self.BASELINE_MEAN:
                assert suite not in suite_baselines, suite
                suite_baselines[suite] = entry
            elif suite not in suite_baselines:
                suite_baselines[suite] = None
        return by_suite, suite_baselines

    @property
    def baseline(self):
        return self.get_baseline()

    def get_baseline(self, suite=None):
        for entry in self._entries:
            if entry.uploadid.suite != suite:
                continue
            if entry.mean == self.BASELINE_MEAN:
                return entry
        return None

    def add(self, entry):
        if not entry:
            raise ValueError('missing entry')
        elif isinstance(entry, str):
            raise NotImplementedError(entry)
        elif not isinstance(entry, PyperfResultsIndexEntry):
            raise TypeError(entry)
        self._add(entry)

    def _add(self, entry):
        #assert entry
        self._entries.append(entry)

    def add_from_results(self, results):
        entry = self._entry_from_results(results)
        self._add(entry)
        return entry

    def _entry_from_results(self, results):
        if not results:
            raise ValueError('missing results')
        return PyperfResultsIndexEntry.from_results(results)

    def ensure_means(self, baseline=None):
        requested = _utils.Version.from_raw(baseline).full if baseline else None
        requested_by_suite = {}
        outdated_by_suite = {}
        by_suite, suite_baselines = self._collate_by_suite()
        for suite in sorted(by_suite):
            indices = by_suite[suite]
            suite_base = suite_baselines[suite]
            if requested:
                for index in indices:
                    entry = self._entries[index]
                    if entry.uploadid.version.full == requested:
                        requested_entry = entry
                        break
                else:
                    raise ValueError(f'unknown baseline {baseline}')
                if not suite_base or suite_base != requested_entry:
                    outdated_by_suite[suite] = indices
                    suite_baselines[suite] = requested_entry
                    continue
            elif not suite_base:
                raise ValueError('missing baseline')
            # Fall back to checking each one.
            for index in indices:
                entry = self._entries[index]
                if not entry.mean and entry is not suite_base:
                    if suite not in outdated_by_suite:
                        outdated_by_suite[suite] = []
                    outedated_by_suite[suite].append(index)
        for suite, indices in outdated_by_suite.items():
            baseline = suite_baselines[suite]
            baselineforcomparison = self._entry_for_comparison(baseline)
            for i in indices:
                entry = self._entries[i]
                if entry is baseline:
                    continue
                mean = self._get_mean(entry, baselineforcomparison)
                self._entries[i] = entry._replace(mean=mean)

    def _entry_for_comparison(self, entry):
        raise NotImplementedError(entry)

    def _get_mean(self, entry, baseline):
        _entry = self._entry_for_comparison(entry)
        compared = baseline.compare([_entry])
        return compared.table.mean_row[-1]


class PyperfResultsIndexFile(PyperfResultsIndex):

    @classmethod
    def from_results_dir(cls, dirname, filename):
        self = cls(filename)
        self._resultsdir = dirname
        for name in os.listdir(dirname):
            self.add(os.path.join(dirname, name))
        return self

    @classmethod
    def load(cls, filename):
        with open(filename) as infile:
            text = infile.read()
        data = cls._parse(text)
        return cls.from_jsonable(data, filename)

    @classmethod
    def from_jsonable(cls, data, filename=None):
        self = cls(filename)
        if sorted(data) != ['entries']:
            raise ValueError(f'unsupported index data {data!r}')
        def entry_from_jsonable(data):
            if sorted(data) != ['build', 'geometric mean', 'uploadid']:
                raise ValueError(f'unsupported index entry data {data!r}')
            uploadid = PyperfUploadID.parse(data['uploadid'])
            if not uploadid:
                raise ValueError(f'bad uploadid in {data}')
            return PyperfResultsIndexEntry(
                uploadid,
                data['build'],
                data['geometric mean'],
            )
        for entrydata in data['entries']:
            entry = entry_from_jsonable(entrydata)
            # XXX The suffix might not be right.
            filename = f'{entry.uploadid}.json'
            self._add(entry, filename)
        return self

    @classmethod
    def _parse(cls, text):
        return json.loads(text)

    @classmethod
    def _unparse(cls, data):
        return json.dumps(data, indent=2)

    def __init__(self, filename):
        super().__init__()
        if not filename:
            raise ValueError('missing filename')
        self._filename = filename
        self._relfiles = {}

    @property
    def filename(self):
        return self._filename

    @property
    def resultsdir(self):
        try:
            return self._resultsdir
        except AttributeError:
            self._resultsdir = os.path.dirname(self._filename)
            return self._resultsdir

    def add(self, entry):
        if isinstance(entry, str) and entry:
            filename = entry
            return self.add_from_file(filename)
        return super().add(entry)

    def add_from_results(self, results):
        filename = None  # Handled in self._add()
        return self.add_from_file(filename, results)

    def add_from_file(self, filename, results=None):
        if not filename:
            return ValueError('missing filename')
        if results:
            entry = self._entry_from_results(results)
            # XXX Save to the file?
        else:
            entry = self._entry_from_file(filename)
        if not entry:
            return None
        self._add(entry, filename)
        return entry

    def _entry_from_file(self, filename):
        dirname, name = os.path.split(filename)
        if os.path.abspath(dirname) != self.resultsdir:
            raise ValueError(f'not in results dir ({filename})')
        uploadid = PyperfUploadID.parse(name, allowsuffix=True)
        if not uploadid:
            return None
        resfile = PyperfResultsFile(filename, uploadid)
        results = resfile.read()
        return PyperfResultsIndexEntry.from_results(results)

    def _add(self, entry, filename=None):
        if not filename:
            # XXX What is the right filename to use?
            raise NotImplementedError(entry)
            filename = f'{results.uploadid}.json'
        if os.path.isabs(filename):
            relfile = os.path.relpath(filename, self.resultsdir)
        elif os.path.basename(filename) == filename:
            relfile = filename
        else:
            raise NotImplementedError(repr(filename))
        if relfile in self._relfiles:
            raise KeyError(f'{relfile} already added ({self._relfiles[relfile]})')
        super()._add(entry)
        self._relfiles[entry] = relfile
        return relfile

    def _entry_for_comparison(self, entry):
        return PyperfResultsFile(
            self._relfiles[entry],
            entry.uploadid,
            self.resultsdir,
        )

    def save(self):
        data = self._as_jsonable()
        text = self._unparse(data)
        with open(self._filename, 'w', encoding='utf-8') as outfile:
            outfile.write(text)
            print(file=outfile)  # Add a blank line at the end.

    def _as_jsonable(self):
        def as_jsonable(entry):
            uploadid, build, mean = entry
            return {
                'uploadid': str(uploadid),
                'build': build,
                'geometric mean': str(mean) if mean else None,
            }
        return {
            'entries': [as_jsonable(e) for e in self._entries],
        }


class PyperfResultsIndexEntry(
        namedtuple('PyperfResultsIndexEntry', 'uploadid build mean')):

    @classmethod
    def from_results(cls, results):
        uploadid = results.uploadid
        build = results.build
        mean = None
        return cls(uploadid, build, mean)

    def __new__(cls, uploadid, build, mean):
        return super().__new__(
            cls,
            PyperfUploadID.from_raw(uploadid),
            tuple(build) if build else None,
            mean or None,
        )

    def __repr__(self):
        reprstr = super().__repr__()
        prefix, _, remainder = reprstr.partition('uploadid=')
        _, _, remainder = remainder.partition(', build=')
        return f'{prefix}uploadid={str(self.uploadid)!r}, build={remainder})'


##################################
# faster-cpython

class FasterCPythonResults(PyperfResultsRepo):

    REMOTE = _utils.GitHubTarget.from_origin('faster-cpython', 'ideas', ssh=True)
    DATADIR = 'benchmark-results'

    def __init__(self, root=None, remote=None):
        if not remote:
            remote = self.REMOTE
        super().__init__(root, remote, self.DATADIR)
