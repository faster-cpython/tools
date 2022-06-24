from collections import namedtuple
import gzip
import hashlib
import json
import logging
import os
import os.path
import platform
import re
import shutil
import sys

from . import _utils


logger = logging.getLogger(__name__)


##################################
# pyperformance helpers

class Benchmarks:

    REPOS = os.path.join(_utils.HOME, 'repos')

    PYPERFORMANCE = 'pyperformance'
    PYSTON = 'pyston'
    SUITES = {
        PYPERFORMANCE: {
            'url': 'https://github.com/python/pyperformance',
            'reldir': 'pyperformance/data-files/benchmarks',
        },
        PYSTON: {
            'url': 'https://github.com/pyston/python-macrobenchmarks',
            'reldir': 'benchmarks',
        },
    }

    @classmethod
    def _load_suite(cls, suite):
        info = cls.SUITES[suite]
        url = info['url']
        reldir = info['reldir']
        reporoot = os.path.join(cls.REPOS,
                                os.path.basename(url))
        if not os.path.exists(reporoot):
            if not os.path.exists(cls.REPOS):
                os.makedirs(cls.REPOS)
            _utils.git('clone', url, reporoot, cwd=None)
        names = cls._get_names(os.path.join(reporoot, reldir))
        return list(names)

    @classmethod
    def _get_names(cls, benchmarksdir):
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

    @classmethod
    def _iter_subcandidates(cls, bench):
        # Some benchmarks actually produce results for
        # sub-benchmarks (e.g. logging -> logging_simple).
        while '_' in bench:
            bench, _, _ = bench.rpartition('_')
            yield bench

    def __init__(self):
        self._cache = {}

    #def __repr__(self):
    #    raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def get_suites(self, benchmarks, default=None):
        mapped = {}
        by_bench = self.load('name')
        for bench in benchmarks:
            try:
                suite = by_bench[bench]
            except KeyError:
                for sub in self._iter_subcandidates(bench):
                    if sub in by_bench:
                        suite = by_bench[sub]
                        break
                else:
                    suite = default
            mapped[bench] = suite
        return mapped

    def get_suite(self, bench, default=None):
        by_suite = self._cache if self._cache else self._load()

        suite = self._get_suite(bench, by_suite)
        if suite:
            return suite

        for sub in self._iter_subcandidates(bench):
            suite = self._get_suite(sub, by_suite)
            if suite:
                return suite
        else:
            return default

    def _get_suite(self, bench, by_suite):
        for suite, names in by_suite.items():
            if bench in names:
                return suite
        else:
            return None

    def load(self, key='name'):
        """Return the per-suite lists of benchmarks."""
        by_suite = self._load()
        if key == 'suite':
            return {s: list(n) for s, n in by_suite.items()}
        elif key == 'name':
            by_name = {}
            for suite, names in by_suite.items():
                for name in names:
                    if name in by_name:
                        raise NotImplementedError((suite, name))
                    by_name[name] = suite
            return by_name
        else:
            raise ValueError(f'unsupported key {key!r}')

    def _load(self):
        for suite in self.SUITES:
            try:
                names = self._cache[suite]
            except KeyError:
                names = self._cache[suite] = self._load_suite(suite)
        return self._cache


class PyperfUploadID(namedtuple('PyperfUploadName',
                                'impl version commit host compatid suite')):
    # See https://github.com/faster-cpython/ideas/tree/main/benchmark-results/README.md
    # for details on this filename format.

    SUITE_NOT_KNOWN = None
    EMPTY = _utils.Sentinel('empty')
    MULTI_SUITE = _utils.Sentinel('multi-suite')
    SUITES = set(Benchmarks.SUITES)
    _SUITES = {s: s for s in SUITES}
    _SUITES[SUITE_NOT_KNOWN] = SUITE_NOT_KNOWN
    _SUITES[EMPTY] = EMPTY
    _SUITES[MULTI_SUITE] = MULTI_SUITE

    REGEX = re.compile(rf'''
        # We do no strip leading/trailing whitespace in this regex.
        ^
        ( .*? )  # <prefix>
        ( \w+ )  # <impl>
        -
        ( main | \d\.\d+\. (?: 0a\d+ | 0b\d+ | 0rc\d+ | [1-9]\d* ) )  # <version>
        -
        ( [0-9a-f]{{10}} )  # <commit>
        -
        ( [^-\s]+ )  # <host>
        -
        ( [0-9a-f]{{12}} )  # <compatid>
        (?:
            -
            ( {'|'.join(SUITES)} )  # <suite>
        )?
        #( \.json (?: \.gz )? )?  # <suffix>
        ( \. .*? )?  # <suffix>
        $
    ''', re.VERBOSE)

    @classmethod
    def from_raw(cls, raw, *, fail=None):
        self = None
        if not raw:
            if fail:
                raise ValueError('missing uploadid')
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            self = cls.parse(raw)
            if not self:
                self = cls.from_filename(raw)
        else:
            if fail or fail is None:
                raise TypeError(raw)
        if fail:
            raise ValueError(f'no match for {raw!r}')
        return self

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
    def from_metadata(cls, metadata, *,
                      version=None,
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
            commit=commit or metadata.commit,
            host=host or metadata.host,
            compatid=metadata.compatid,
            suite=suite,
        )
        return self

    @classmethod
    def build_compatid(cls, host, pyperformance_version, pyperf_version=None):
        if not host:
            raise ValueError('missing host')
        host = _utils.HostInfo.from_raw(host)
        if not pyperformance_version:
            raise ValueError('missing pyperformance_version')
        raw = host.as_metadata()
        data = [
            host.id,
            raw['platform'],
            str(pyperf_version),
            str(pyperformance_version),
            raw['cpu_model_name'],
            raw.get('cpu_freq'),
            raw['cpu_config'],
            raw.get('cpu_affinity'),
        ]
        h = hashlib.sha256()
        for value in data:
            if not value:
                continue
            h.update(value.encode('utf-8'))
        return h.hexdigest()

    @classmethod
    def normalize_suite(cls, suite):
        if not suite:
            return cls.SUITE_NOT_KNOWN

        if not isinstance(suite, str) and _utils.iterable(suite):
            suites = list(suite)
            if len(suites) == 1:
                suite, = suites
            else:
                for suite in suites:
                    suite = cls.normalize_suite(suite)
                    if suite not in cls.SUITES:
                        raise ValueError(f'unsupported suite in multisuite ({suite})')
                return cls.MULTI_SUITE

        try:
            return cls._SUITES[suite]
        except KeyError:
            raise ValueError(f'unsupported suite {suite!r}')

    def __new__(cls, impl, version, commit, host, compatid,
                suite=SUITE_NOT_KNOWN,
                ):
        return super().__new__(
            cls,
            impl=impl,
            version=version,
            commit=commit,
            host=host,
            compatid=compatid,
            suite=cls.normalize_suite(suite),
        )

    def __str__(self):
        return self.name

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            impl, version, commit, host, compatid, suite = self
            name = f'{impl}-{version}-{commit[:10]}-{host}-{compatid[:12]}'
            if suite in self.SUITES:
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

    def match(self, specifier, suites=None):
        # specifier: uploadID, version, filename
        matched = self._match(specifier, checksuite=(not suites))
        if matched and suites:
            assert suites is not self.SUITE_NOT_KNOWN  # ...since that is None.
            if suites is True:
                suites = (s for s in self._SUITES
                          if s is not self.SUITE_NOT_KNOWN)
            else:
                if isinstance(suites, str) or not _utils.iterable(suites):
                    suites = [suites]
                suites = {self.normalize_suite(s) for s in suites}
            if self.suite not in suites:
                return False
        return matched

    def _match(self, specifier, checksuite):
        requested = self.from_raw(specifier, fail=False)
        if requested:
            if not checksuite:
                requested = requested.copy(suite=self.suite)
            if requested == self:
                return True
            assert str(requested) != str(self), (requested, self)
        if self._match_version(specifier):
            return True
        #if self._match_pattern(specifier, checksuite):
        #    return True
        return False

    def _match_version(self, version):
        if isinstance(version, str):
            version = _utils.Version.parse(version)
            if not version:
                return False
        elif not isinstance(version, _utils.Version):
            return False
        # XXX Treat missing micro/release as wildcard?
        return version.full == self.version.full

    #def _match_pattern(self, pat, checksuite):
    #    raise NotImplementedError

    def copy(self, **replacements):
        if not replacements:
            return self
        # XXX Validate the replacements.
        kwargs = dict(self._asdict(), **replacements)
        cls = type(self)
        copied = cls(**kwargs)
        if copied == self:
            return self
        # Copy the internal attrs.
        # We do not copy self._name.
        suffix = getattr(self, '_suffix', None)
        if suffix:
            copied._suffix = suffix
        dirname = getattr(self, '_dirname', None)
        if dirname:
            copied._dirname = dirname
        elif hasattr(self, '_filename') and self._filename:
            copied._dirname = os.path.dirname(self._filename)
        # We do not copy self._filename.
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

    def __hash__(self):
        return hash((self._elapsed, self._comparison))

    def __eq__(self, other):
        raise NotImplementedError

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

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash((
                self._source,
                tuple(sorted(self._byname.items())),
            ))
            return self._hash

    def __eq__(self, other):
        raise NotImplementedError

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

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash((
                self._baseline,
                self._source,
                tuple(sorted(self._byname.items())),
                self._mean,
            ))
            return self._hash

    def __eq__(self, other):
        raise NotImplementedError

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

    def __eq__(self, other):
        raise NotImplementedError

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

    def __eq__(self, other):
        raise NotImplementedError

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
# results data

class PyperfResults:

    BENCHMARKS = Benchmarks()

    @classmethod
    def _validate_data(cls, data):
        if data['version'] == '1.0':
            for key in ('metadata', 'benchmarks', 'version'):
                if key not in data:
                    raise ValueError(f'invalid results data (missing {key})')
            # XXX Add other checks.
        else:
            raise NotImplementedError(data['version'])

    def __init__(self, data, resfile):
        if not data:
            raise ValueError('missing data')
        if not resfile:
            raise ValueError('missing refile')
        self._validate_data(data)
        self._data = data
        self._resfile = PyperfResultsFile.from_raw(resfile)

    def _copy(self):
        cls = type(self)
        copied = cls.__new__(cls)
        copied._data = self._data
        copied._resfile = self._resfile
        return copied

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        try:
            return self._metadata
        except AttributeError:
            self._metadata = PyperfResultsMetadata.from_full_results(self._data)
            return self._metadata

    @property
    def version(self):
        return self._data['version']

    @property
    def resfile(self):
        return self._resfile

    @property
    def filename(self):
        return self._resfile.filename

    @property
    def uploadid(self):
        try:
            return self._uploadid
        except AttributeError:
            if self._resfile.uploadid:
                # XXX Compare with what we get from the metadata?
                self._uploadid = self._resfile.uploadid
            else:
                self._uploadid = PyperfUploadID.from_metadata(
                    self.metadata,
                    suite=self.suites,
                )
            return self._uploadid

    @property
    def suite(self):
        return self.uploadid.suite

    @property
    def suites(self):
        return sorted(self.by_suite)

    @property
    def by_bench(self):
        try:
            return self._by_bench
        except AttributeError:
            self._by_bench = dict(self._iter_benchmarks())
            return self._by_bench

    @property
    def by_suite(self):
        try:
            return self._by_suite
        except AttributeError:
            self._by_suite = self._collate_suites()
            return self._by_suite

    def _collate_suites(self):
        by_suite = {}
        names = [n for n, _ in self._iter_benchmarks()]
        if names:
            bench_suites = self.BENCHMARKS.get_suites(names, 'unknown')
            for name, suite in bench_suites.items():
                suite = PyperfUploadID.normalize_suite(suite)
                if suite not in by_suite:
                    by_suite[suite] = []
                data = self.by_bench[name]
#                by_suite[suite][name] = data
                by_suite[suite].append(data)
        else:
            logger.warn(f'empty results {self}')
        return by_suite

    def _get_bench_name(self, benchdata):
        return benchdata['metadata']['name']

    def _iter_benchmarks(self):
        for data in self._data['benchmarks']:
            name = self._get_bench_name(data)
            yield name, data

    def split_benchmarks(self):
        """Return results collated by suite."""
        if self.suite is not PyperfUploadID.MULTI_SUITE:
            assert self.suite is not PyperfUploadID.SUITE_NOT_KNOWN
            raise Exception(f'already split ({self.suite})')
        by_suite = {}
        for suite, benchmarks in self.by_suite.items():
            by_suite[suite] = {k: v
                               for k, v in self._data.items()
                               if k != 'benchmarks'}
            by_suite[suite]['benchmarks'] = benchmarks
        cls = type(self)
        for suite, data in by_suite.items():
            results = self._copy()
            results._data = data
            results._by_suite = {suite: data['benchmarks'][0]}
            by_suite[suite] = results
        return by_suite

    #def compare(self, others):
    #    raise NotImplementedError

    def copy_to(self, filename, resultsroot=None, *, compressed=None):
        if os.path.exists(self._resfile.filename):
            resfile = self._resfile.copy_to(filename, resultsroot,
                                            compressed=compressed)
        else:
            resfile = PyperfResultsFile(filename, resultsroot,
                                        compressed=compressed)
            resfile.write(self)
        copied = self._copy()
        copied._resfile = resfile
        return copied


class PyperfResultsMetadata:

    EXPECTED = {
        # top-level
        "aslr",
        "boot_time",
        "commit_branch",
        "commit_date",
        "commit_id",
        "cpu_affinity",
        #"cpu_arch",
        "cpu_config",
        "cpu_count",
        #"cpu_freq",
        "cpu_model_name",
        #"dnsname",
        #"hostid",
        "hostname",
        "performance_version",
        "platform",
        "unit",
        # per-benchmark
        #"cpu_freq",
        "perf_version",
        "python_cflags",
        "python_compiler",
        "python_implementation",
        "python_version",
        "runnable_threads",
        "timer",
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

    def __eq__(self, other):
        raise NotImplementedError

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        yield from self._data

    @property
    def data(self):
        return self._data

    @property
    def version(self):
        return self._version

    @property
    def commit(self):
        return self._data['commit_id']

    @property
    def python_implementation(self):
        try:
            return self._python_implementation
        except AttributeError:
            impl = _utils.resolve_python_implementation(
                self._data.get('python_implementation'),
            )
            self._python_implementation = impl
            return impl

    @property
    def pyversion(self):
        try:
            return self._pyversion
        except AttributeError:
            impl = self.python_implementation
            self._pyversion = impl.parse_version(
                self._data.get('python_version'),
            )
            return self._pyversion
        raise NotImplementedError

    @property
    def build(self):
        # XXX Add to PyperfUploadID?
        # XXX Extract from self._data?
        return ['PGO', 'LTO']

    @property
    def host(self):
        try:
            return self._host
        except AttributeError:
            self._host = _utils.HostInfo.from_metadata(
                self._data.get('hostid'),
                self._data['hostname'],
                self._data.get('dnsname'),
                self._data['platform'],
                self._data['cpu_model_name'],
                self._data['cpu_config'],
                self._data.get('cpu_freq'),
                self._data.get('cpu_count'),
                self._data.get('cpu_affinity'),
            )
            return self._host

    @property
    def compatid(self):
        raw = self.host.as_metadata()
        return PyperfUploadID.build_compatid(
            self.host,
            self._data['performance_version'],
            self._data.get('perf_version'),
        )

    def overwrite(self, field, value):
        old = self._data.get(field)
        if not old:
            self._data[field] = value
        elif old != value:
            logger.warn(f'replacing {field} in results metadata ({old} -> {value})')
            self._data[field] = value
        return old


class PyperfResultsInfo(
        namedtuple('PyperfResultsInfo', 'uploadid build filename compared')):

    @classmethod
    def from_results(cls, results, compared=None):
        if not isinstance(results, PyperfResults):
            raise NotImplementedError(results)
        resfile = results.resfile
        assert resfile, results
        self = cls._from_values(
            results.uploadid,
            cls._normalize_build(results.metadata.build),  # XXX Use it as-is?
            resfile.filename,
            compared,
            resfile.resultsroot,
        )
        self._resfile = resfile
        return self

    @classmethod
    def from_resultsfile(cls, resfile, compared=None):
        if not resfile:
            raise ValueError('missing resfile')
        elif not isinstance(resfile, PyperfResultsFile):
            raise TypeError(resfile)
        if not resfile.uploadid:
            raise NotImplementedError(resfile)
        results = resfile.read()
        self = cls.from_results(results, compared)
        return self, results

    @classmethod
    def from_file(cls, filename, resultsroot=None, compared=None):
        resfile = PyperfResultsFile(filename, resultsroot)
        return cls.from_resultsfile(resfile, compared)

    @classmethod
    def from_values(cls, uploadid, build=None, filename=None, compared=None,
                    resultsroot=None):
        uploadid = PyperfUploadID.from_raw(uploadid)
        build = cls._normalize_build(build)
        return cls._from_values(
            uploadid, build, filename, compared, resultsroot)

    @classmethod
    def _from_values(cls, uploadid, build, filename, compared, resultsroot):
        if filename:
            (filename, relfile, resultsroot,
             ) = normalize_results_filename(filename, resultsroot)
        elif resultsroot or compared:
            raise ValueError('missing filename')
        if compared:
            compared = PyperfComparison.from_raw(compared)
        self = cls.__new__(cls, uploadid, build, filename, compared)
        if resultsroot:
            self._resultsroot = resultsroot
            self._relfile = relfile
        return self

    @classmethod
    def _normalize_build(cls, build):
        if not build:
            return build
        if isinstance(build, str):
            raise NotImplementedError(build)
        build = tuple(build)
        cls._validate_build_values(build)
        return build

    @classmethod
    def _validate_build_values(cls, values):
        for i, value in enumerate(values):
            if not value:
                raise ValueError(f'build[{i}] is empty')
            elif not isinstance(value, str):
                raise TypeError(f'expected str for build[{i}], got {value!r}')
            # XXX other checks?

    def __new__(cls, uploadid, build=None, filename=None, compared=None):
        return super().__new__(
            cls,
            uploadid=uploadid or None,
            build=build or None,
            filename=filename or None,
            compared=compared or None,
        )

    def __init__(self, *args, **kwargs):
        self._validate()

    def _validate(self):
        if not self.uploadid:
            raise ValueError('missing uploadid')
        elif not isinstance(self.uploadid, PyperfUploadID):
            raise TypeError(self.uploadid)

        if self.build:
            if not isinstance(self.build, tuple):
                raise TypeError(self.build)
            else:
                self._validate_build_items(self.build)

        if self.filename:
            if not isinstance(self.filename, str):
                raise TypeError(self.filename)
            if not os.path.isabs(self.filename):
                raise ValueError(f'expected an absolute filename, got {self.filename!r}')

        if self.compared:
            if not isinstance(self.compared, PyperfComparison):
                raise TypeError(self.compared)

    def __repr__(self):
        reprstr = super().__repr__()
        prefix, _, remainder = reprstr.partition('uploadid=')
        _, _, remainder = remainder.partition(', build=')
        return f'{prefix}uploadid={str(self.uploadid)!r}, build={remainder})'

    @property
    def resultsroot(self):
        try:
            return self._resultsroot
        except AttributeError:
            if not self.filename:
                return None
            return os.path.dirname(self.filename)
            #return None

    @property
    def relfile(self):
        try:
            return self._relfile
        except AttributeError:
            if not self.filename:
                return None
            if not hasattr(self, '_resultsroot'):
                return os.path.basename(self.filename)
            self._relfile = os.path.relpath(self.filename, self._resultsroot)
            return self._relfile

    @property
    def resfile(self):
        try:
            return self._resfile
        except AttributeError:
            if not self.filename:
                return None
            self._resfile = PyperfResultsFile(self.filename, self.resultsroot)
            return self._resfile

    @property
    def mean(self):
        return self.compared.mean if self.compared else None

    def match(self, specifier, suites=None, *, checkexists=False):
        # specifier: uploadID, version, filename
        if not specifier:
            return False
        matched = self._match(specifier, suites)
        if matched:
            if checkexists:
                if self.filename and not os.path.isfile(self.filename):
                    return False
        return matched

    def match_uploadid(self, uploadid, *, checkexists=False):
        if not self.uploadid:
            return False
        if not self.uploadid.match(uploadid):
            return False
        if checkexists:
            if self.filename and not os.path.isfile(self.filename):
                return False
        return True

    def _match(self, specifier, suites):
        if self._match_filename(specifier):
            return True
        if self.uploadid and self.uploadid.match(specifier, suites):
            return True
        return False

    def _match_filename(self, filename):
        if not self.filename:
            return False
        if not isinstance(filename, str):
            return False
        filename, _, _ = normalize_results_filename(
            filename,
            None if os.path.isabs(filename) else self.resultsroot,
        )
        return filename == self.filename

    def load_results(self):
        resfile = self.resfile
        if not resfile:
            return None
        return resfile.read()


class PyperfResultsIndex:

    BASELINE_MEAN = '(ref)'

#    iter_all()
#    add()
#    ensure_means()

    def __init__(self):
        self._entries = []

    def __eq__(self, other):
        raise NotImplementedError

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

    def iter_all(self, *, checkexists=False):
        if checkexists:
            for info in self._entries:
                if not info.filename or not os.path.isfile(info.filename):
                    continue
                yield info
        else:
            yield from self._entries

    def get_baseline(self, suite=None):
        for entry in self._entries:
            if entry.uploadid.suite != suite:
                continue
            if entry.mean == self.BASELINE_MEAN:
                return entry
        return None

    def get(self, uploadid, default=None, *, checkexists=False):
        raw = uploadid
        requested = PyperfUploadID.from_raw(raw)
        if not requested:
            return default
        found = None
        for info in self.iter_all(checkexists=checkexists):
            if not info.uploadid or info.uploadid != requested:
                continue
            if found:
                raise RuntimeError('matched multiple, consider using match()')
            found = info
        return found

    def match(self, specifier, suites=None, *, checkexists=False):
        # specifier: uploadID, version, filename
        if not specifier:
            return
        for info in self.iter_all(checkexists=checkexists):
            if not info.match(specifier, suites):
                continue
            yield info

    def match_uploadid(self, uploadid, *, checkexists=True):
        requested = PyperfUploadID.from_raw(raw)
        if not requested:
            return None
        for info in self.iter_all(checkexists=checkexists):
            if not info.match_uploadid(uploadid):
                continue
            yield info

    def add(self, info):
        if not info:
            raise ValueError('missing info')
        elif not isinstance(info, PyperfResultsInfo):
            raise TypeError(info)
        self._add(info)
        return info

    def _add(self, info):
        #assert info
        self._entries.append(info)

    def add_from_results(self, results, compared=None):
        info = PyperfResultsInfo.from_results(results, compared)
        return self.add(info)

    def ensure_means(self, baseline=None):
        return


##################################
# results files

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


class PyperfResultsDir:

    INDEX = 'index.json'

    @classmethod
    def _convert_to_uploadid(cls, uploadid):
        if not uploadid:
            return None
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

    def __init__(self, root):
        if not root:
            raise ValueError('missing root')
        self._root = os.path.abspath(root)
        self._indexfile = os.path.join(self._root, self.INDEX)

    def __repr__(self):
        return f'{type(self).__name__}({self._root!r})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def root(self):
        return self._root

    @property
    def indexfile(self):
        return self._indexfile

    def _info_from_values(self, filename, uploadid, build=None):
        if not build:
            build = ['PGO', 'LTO']
#            # XXX Get it from somewhere.
#            raise NotImplementedError
        compared = None  # XXX
        return PyperfResultsInfo.from_values(
            uploadid,
            build,
            filename,
            compared,
            self._root,
        )

    def _info_from_file(self, filename):
        compared = None  # XXX
        return PyperfResultsInfo.from_file(filename, self._root, compared)

    def _info_from_results(self, results):
        raise NotImplementedError  # XXX
        ...

    def _iter_results_files(self):
        raise NotImplementedError

    def iter_from_files(self):
        for filename in self._iter_results_files():
            info, _ = self._info_from_file(filename)
            yield info

    def iter_all(self):
        index = self.load_index()
        yield from index.iter_all()

    def index_from_files(self, *, baseline=None):
        index = PyperfResultsIndex()
        for info in self.iter_from_files():
            index.add(info)
        if baseline:
            index.ensure_means(baseline=baseline)
        return index

    def load_index(self, *,
                   baseline=None,
                   createifmissing=True,
                   saveifupdated=True,
                   ):
        save = False
        try:
            index = self._load_index()
        except FileNotFoundError:
            if not createifmissing:
                raise  # re-raise
            index = self.index_from_files()
            save = True
        if baseline:
            updated = index.ensure_means(baseline=baseline)
            if updated and saveifupdated:
                save = True
        if save:
            self.save_index(index)
        return index

    def _load_index(self):
        index = PyperfResultsIndex()
        with open(self._indexfile) as infile:
            text = infile.read()
        # We use a basic JSON format.
        data = json.loads(text)
        if sorted(data) != ['entries']:
            raise ValueError(f'unsupported index data {data!r}')
        fields = ['relative path', 'uploadid', 'build', 'geometric mean']
        expected = sorted(fields)
        for entrydata in data['entries']:
            if sorted(entrydata) != expected:
                raise ValueError(f'unsupported index entry data {data!r}')
            uploadid = PyperfUploadID.parse(entrydata['uploadid'])
            if not uploadid:
                raise ValueError(f'bad uploadid in {data}')
            relfile = entrydata['relative path'] or None
            if not relfile:
                raise ValueError(f'missing relative path for {uploadid}')
            elif os.path.isabs(relfile):
                raise ValueError(f'got absolute relative path {relfile!r}')
            build = entrydata['build'] or None
            mean = entrydata['geometric mean'] or None
            info = self._info_from_values(relfile, uploadid, build)
            index.add(info)
        return index

    def save_index(self, index):
        # We use a basic JSON format.
        entries = []
        for info in index.iter_all():
            if not info.filename:
                raise NotImplementedError(info)
            if info.resultsroot != self._root:
                raise NotImplementedError((info, self._root))
            entries.append({
                'relative path': os.path.relpath(info.filename, self._root),
                'uploadid': str(info.uploadid),
                'build': info.build or None,
                'geometric mean': str(info.mean) if info.mean else None,
            })
        data = {'entries': entries}
        text = json.dumps(data, indent=2)
        with open(self._indexfile, 'w', encoding='utf-8') as outfile:
            outfile.write(text)
            print(file=outfile)  # Add a blank line at the end.

    def get(self, uploadid, default=None, *, checkexists=True):
        index = self.load_index()
        return index.get(uploadid, default, checkexists=checkexists)

    def match(self, specifier, suites=None, *, checkexists=True):
        index = self.load_index()
        yield from index.match(specifier, suites, checkexists=checkexists)

    def match_uploadid(self, uploadid, *, checkexists=True):
        index = self.load_index()
        yield from index.match_uploadid(uploadid, checkexists=checkexists)

#    def add(self, info, *,
#            baseline=None,
#            compressed=False,
#            split=False,
#            ):
#        if isinstance(info, PyperfResultsInfo):
#            pass
#        else:
#            raise NotImplementedError(info)
#        raise NotImplementedError  # XXX
#        index = self.load_index(baseline=baseline)
#        ...
#        index.ensure_means(baseline)

    def add_from_results(self, results, *,
                         baseline=None,
                         compressed=False,
                         split=False,
                         ):
        if not isinstance(results, PyperfResults):
            raise NotImplementedError(results)

        source = results.filename
        if source and not os.path.exists(source):
            logger.error(f'results not found at {source}')
            return

        # First add the file(s).
        if split and results.suite is PyperfUploadID.MULTI_SUITE:
            by_suite = results.split_benchmarks()
        else:
            by_suite = {results.suite: results}
        copied = []
        for suite, suite_results in sorted(by_suite.items()):
            if results.suite in PyperfUploadID.SUITES:
                logger.info(f'adding results {source or "???"} ({suite})...')
            else:
                logger.info(f'adding results {source or "???"}...')
            resfile = PyperfResultsFile.from_uploadid(
                suite_results.uploadid,
                resultsroot=self._root,
                compressed=compressed,
            )
            logger.info(f'...as {resfile.relfile}...')
            #copied = suite_results.copy_to(resfile, self._root)
            copied.append(
                suite_results.copy_to(resfile, self._root)
            )
            logger.info('...done adding')

        # Then update the index.
        logger.info('updating index...')
        index = self.load_index(baseline=baseline)
        for results in copied:
            info = index.add_from_results(results)
            # XXX Do this after everything has been yielded.
            if baseline:
                index.ensure_means(baseline=baseline)
            self.save_index(index)
            yield info
        logger.info('...done updating index')

#    def add_from_file(self, filename):
#        ...


class PyperfUploadsDir(PyperfResultsDir):

    def _iter_results_files(self):
        for name in os.listdir(self._root):
            uploadid = PyperfUploadID.parse(name, allowsuffix=True)
            if not uploadid:
                continue
            filename = os.path.join(self._root, name)
            if not os.path.isfile(filename):
                continue
            yield filename


class PyperfResultsFile:

    SUFFIX = '.json'
    COMPRESSED_SUFFIX = '.json.gz'
    COMPRESSOR = gzip

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

    @classmethod
    def from_uploadid(cls, uploadid, resultsroot=None, *, compressed=False):
        uploadid = PyperfUploadID.from_raw(uploadid)
        if not uploadid:
            raise ValueError('missing uploadid')
        return cls(f'{uploadid}{cls.SUFFIX}', resultsroot,
                   compressed=compressed)

    #@classmethod
    #def split_suffix(cls, filename):
    #    for suffix in [cls.COMPRESSED_SUFFIX, cls.SUFFIX]:
    #        if filename.endswith(suffix):
    #            base = filename[:len(suffix)]
    #            return base, suffix
    #            break
    #    else:
    #        return filename, None

    @classmethod
    def _resolve_filename(cls, filename, resultsroot, compressed):
        if not filename:
            raise ValueError('missing filename')
        filename = cls._ensure_suffix(filename, compressed)
        return normalize_results_filename(filename, resultsroot)

    @classmethod
    def _ensure_suffix(cls, filename, compressed):
        if not filename.endswith((cls.SUFFIX, cls.COMPRESSED_SUFFIX)):
            raise ValueError(f'unsupported file suffix ({filename})')
        elif compressed is None:
            return filename
        elif compressed == cls._is_compressed(filename):
            return filename
        else:
            if compressed:
                old, new = cls.SUFFIX, cls.COMPRESSED_SUFFIX
            else:
                old, new = cls.COMPRESSED_SUFFIX, cls.SUFFIX
            return filename[:-len(old)] + new

    @classmethod
    def _is_compressed(cls, filename):
        return filename.endswith(cls.COMPRESSED_SUFFIX)

    def __init__(self, filename, resultsroot=None, *, compressed=None):
        (filename, relfile, resultsroot,
         ) = self._resolve_filename(filename, resultsroot, compressed)
        if os.path.isdir(filename):
            raise NotImplementedError(filename)

        self._filename = filename
        self._relfile = relfile
        self._resultsroot = resultsroot

    def __repr__(self):
        return f'{type(self).__name__}({self.filename!r})'

    def __str__(self):
        return self._filename

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def filename(self):
        return self._filename

    @property
    def relfile(self):
        return self._relfile

    @property
    def resultsroot(self):
        return self._resultsroot

    @property
    def uploadid(self):
        return PyperfUploadID.from_filename(self.filename)

    @property
    def iscompressed(self):
        return self._is_compressed(self._filename)

    def read(self):
        _open = self.COMPRESSOR.open if self.iscompressed else open
        with _open(self._filename) as infile:
            text = infile.read()
        if not text:
            raise RuntimeError(f'{self.filename} is empty')
        data = json.loads(text)
        return PyperfResults(data, self)

    def write(self, results):
        data = results.data
        _open = self.COMPRESSOR.open if self.iscompressed else open
        if self.iscompressed:
            text = json.dumps(data, indent=2)
            with _open(self._filename, 'w') as outfile:
                outfile.write(text.encode('utf-8'))
        else:
            with _open(self._filename, 'w') as outfile:
                json.dump(data, outfile, indent=2)

    def copy_to(self, filename, resultsroot=None, *, compressed=None):
        if isinstance(filename, PyperfResultsFile):
            copied = filename
            if (copied._resultsroot and resultsroot and
                    resultsroot != copied._resultsroot):
                raise ValueError(f'resultsroot mismatch ({resultsroot} != {copied._resultsroot})')
        else:
            if not filename:
                filename = self._filename
            elif os.path.isdir(filename):
                raise NotImplementedError(filename)
            elif not resultsroot and (not os.path.isabs(filename) or
                                      filename.startswith(self._resultsroot)):
                resultsroot = self._resultsroot
            (filename, relfile, resultsroot,
             ) = self._resolve_filename(filename, resultsroot, compressed)
            if filename == self._filename:
                raise ValueError(f'copying to self ({filename})')

            cls = type(self)
            copied = cls.__new__(cls)
            copied._filename = filename
            copied._relfile = relfile
            copied._resultsroot = resultsroot

        if copied.iscompressed == self.iscompressed:
            if copied._filename == self._filename:
                # XXX Fail?
                pass
            shutil.copyfile(self._filename, copied._filename)
        else:
            results = self.read()
            copied.write(results)
        return copied

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


##################################
# results storage

class PyperfResultsStorage:

    def __eq__(self, other):
        raise NotImplementedError

#    def iter_all(self):
#        raise NotImplementedError

#    def get(self, uploadid):
#        raise NotImplementedError

    def match(self, specifier):
        raise NotImplementedError

    def add(self, results, *, compressed=False, split=False):
        raise NotImplementedError


class PyperfResultsRepo(PyperfResultsStorage):

    BRANCH = 'add-benchmark-results'

    def __init__(self, root, remote=None, datadir=None, baseline=None):
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
        self.datadir = datadir or None
        self.remote = remote

        self._baseline = baseline
        self._resultsdir = PyperfUploadsDir(
            os.path.join(root, datadir) if datadir else root,
        )

    def _git(self, *args, cfg=None):
        ec, text = _utils.git(*args, cwd=self.root, cfg=cfg)
        if ec:
            raise NotImplementedError((ec, text))
        return text

    def iter_all(self):
        for info in self._resultsdir.iter_from_files():
            yield info.uploadid
        #yield from self._resultsdir.iter_from_files()
        #yield from self._resultsdir.iter_all()

    def get(self, uploadid, default=None):
        info = self._resultsdir.get(uploadid, default)
        return info.resfile if info else None
        #return self._resultsdir.get(uploadid)

    def match(self, specifier, suites=None):
        for info in self._resultsdir.match(specifier, suites):
            yield info.resfile
        #yield from self._resultsdir.match(specifier, suites)

    def add(self, results, *,
            branch=None,
            author=None,
            compressed=False,
            split=True,
            push=True,
            ):
        branch, gitcfg = self._prep_for_commit(branch, author)
        self._git('checkout', '-B', branch)

        #added = self._resultsdir.add(info, ...)
        added = self._resultsdir.add_from_results(
            results,
            baseline=self._baseline,
            compressed=compressed,
            split=split,
        )

        for info in added:
            logger.info('committing to the repo...')
            relfile = os.path.relpath(info.filename, self.root)
            self._git('add', relfile, self._resultsdir.indexfile)
            msg = f'Add Benchmark Results ({info.uploadid})'
            self._git('commit', '-m', msg, cfg=gitcfg)
            logger.info('...done committing')

        if push:
            self._upload(self.datadir or '.')

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

    def _upload(self, reltarget):
        if not self.remote:
            raise Exception('missing remote')
        url = f'{self.remote.url}/tree/main/{reltarget}'
        logger.info(f'uploading results to {url}...')
        self._git('push', self.remote.push_url)
        logger.info('...done uploading')


##################################
# faster-cpython

class FasterCPythonResults(PyperfResultsRepo):

    REMOTE = _utils.GitHubTarget.from_origin('faster-cpython', 'ideas', ssh=True)
    DATADIR = 'benchmark-results'
    BASELINE = '3.10.4'

    def __init__(self, root=None, remote=None, baseline=BASELINE):
        if not remote:
            remote = self.REMOTE
        super().__init__(root, remote, self.DATADIR, baseline)
