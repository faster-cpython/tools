from collections import namedtuple
import collections
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
        if not raw:
            if fail:
                raise ValueError('missing uploadid')
            return None
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            self = cls.parse(raw)
            if not self:
                self = cls.from_filename(raw)
            return self
        else:
            if fail or fail is None:
                raise TypeError(raw)
            return None

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
        if not version:
            # We assume "main" if it's missing.
            version = metadata.pyversion or 'main'
        if version == 'main':
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
            if isinstance(host, _utils.HostInfo):
                host = host.id
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

    @property
    def sortkey(self):
        # We leave commit and suite out.
        return (self.impl, self.version, self.host, self.compatid)

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
        if not isinstance(other, PyperfComparisonValue):
            return NotImplemented
        if self._elapsed != other._elapsed:
            return False
        if self._comparison != other._comparison:
            return False
        return True

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


class _PyperfComparison:

    kind = None

    @classmethod
    def from_raw(cls, raw, *, fail=None):
        if not raw:
            if fail:
                raise ValueError(f'missing {cls.kind}')
            return None
        elif isinstance(raw, cls):
            return raw
        else:
            if fail or fail is None:
                raise TypeError(raw)
            return None

    @classmethod
    def _parse_value(cls, valuestr):
        return _utils.ElapsedTimeWithUnits.parse(valuestr, fail=True)

    def __init__(self, source, byname=None):
        _utils.check_str(source, 'source', required=True, fail=True)
        if not os.path.isabs(source):
            raise ValueError(f'expected an absolute source, got {source!r}')
        # XXX Further validate source as a filename?

        _byname = {}
        if byname:
            for name, value in byname.items():
                assert name and isinstance(name, str), (name, value, byname)
                assert value and isinstance(value, str), (name, value, byname)
                _byname[name] = self._parse_value(value)

        self._source = source
        self._byname = _byname

    def __repr__(self):
        return f'{type(self).__name__}({self._source!r}, {self._byname!r})'

    def __str__(self):
        return f'<{self.kind} {self._source!r}>'

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self._as_hashable())
            return self._hash

    def __eq__(self, other):
        if not isinstance(other, _PyperfComparison):
            return NotImplemented
        if self._source != other._source:
            return False
        if self._byname != other._byname:
            return False
        return True

    def _as_hashable(self):
        return (
            self._source,
            tuple(sorted(self._byname.items())) if self._byname else (),
        )

    @property
    def source(self):
        return self._source

    @property
    def byname(self):
        return dict(self._byname)


class PyperfComparisonBaseline(_PyperfComparison):
    """The filename and set of result values for baseline results."""

    kind = 'baseline'


class PyperfComparison(_PyperfComparison):
    """The per-benchmark differences between one results set and a baseline.

    The comparison values are a mapping from benchmark name to the
    relative differences (e.g. "1.04x faster").  The geometric mean
    is also provided.
    """

    kind = 'comparison'

    Summary = namedtuple('Summary',
                         'bench baseline baseresult source result comparison')

    @classmethod
    def _parse_value(cls, valuestr):
        return PyperfComparisonValue.parse(valuestr, fail=True)

    def __init__(self, baseline, source, mean, byname=None):
        super().__init__(source, byname)
        baseline = PyperfComparisonBaseline.from_raw(baseline, fail=True)
        if self._byname and sorted(self._byname) != sorted(baseline.byname):
            raise ValueError(f'mismatch with baseline ({sorted(self._byname)} != {sorted(baseline.byname)})')
        if mean:
            mean = _utils.ElapsedTimeComparison.parse(mean, fail=True)

        self._baseline = baseline
        self._mean = mean or None

    def __repr__(self):
        fields = 'baseline source byname mean'.split()
        values = [f'{a}={getattr(self, "_"+a)!r}' for a in fields]
        return f'{type(self).__name__}({", ".join(values)})'

    def __str__(self):
        return f'<{self._mean} ({self._source})>'

    __hash__ = _PyperfComparison.__hash__

    def __eq__(self, other):
        if not isinstance(other, PyperfComparison):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self._baseline != other._baseline:
            return False
        if self._mean != other._mean:
            return False
        return True

    def _as_hashable(self):
        return (
            self._baseline,
            *super()._as_hashable(),
            self._mean,
        )

    @property
    def baseline(self):
        return self._baseline

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
    def parse_table(cls, text, filenames=None):
        table = PyperfTable.parse(text, filenames)
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
            assert source.endswith(PyperfResultsFile._SUFFIXES), repr(source)
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
            _bysource[source] = PyperfComparison(baseline, source, mean, byname)
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
    def parse(cls, text, filenames=None):
        lines = iter(text.splitlines())
        # First parse the header.
        for line in lines:
            header = PyperfTableHeader.parse(line, filenames)
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

    @classmethod
    def parse(cls, line, filenames=None, *, fail=False):
        self = super().parse(line, fail=fail)
        if not self:
            return None
        if filenames:
            values = (self.name, *filenames)
            if len(values) != len(self):
                raise ValueError(f'filenames mismatch ({values[1:]} != {self[1:]})')
            if values != self:
                # XXX Make sure they mostly match?
                self = tuple.__new__(cls, values)
        return self

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
    def get_metadata_raw(cls, data):
        return data['metadata']

    @classmethod
    def iter_benchmarks_from_data(cls, data):
        yield from data['benchmarks']

    @classmethod
    def get_benchmark_name(cls, benchdata):
        return benchdata['metadata']['name']

    @classmethod
    def get_benchmark_metadata_raw(cls, benchdata):
        return benchdata['metadata']

    @classmethod
    def iter_benchmark_runs_from_data(cls, benchdata):
        for rundata in benchdata['runs']:
            yield (
                rundata['metadata'],
                rundata.get('warmups'),
                rundata.get('values'),
            )

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
    def raw(self):
        return self._data

    @property
    def raw_metadata(self):
        return self._data['metadata']

    @property
    def raw_benchmarks(self):
        return self._data['benchmarks']

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
    def date(self):
        run0 = self.raw_benchmarks[0]['runs'][0]
        date = run0['metadata']['date']
        date, _ = _utils.get_utc_datetime(date)
        return date

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

    def _iter_benchmarks(self):
        for benchdata in self.iter_benchmarks_from_data(self._data):
            name = self.get_benchmark_name(benchdata)
            yield name, benchdata

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

    _EXPECTED_TOP = {
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
    }
    _EXPECTED_BENCH = {
        #"cpu_freq",
        "perf_version",
        "python_cflags",
        "python_compiler",
        "python_implementation",
        "python_version",
        "runnable_threads",
        "timer",
    }
    EXPECTED = _EXPECTED_TOP | _EXPECTED_BENCH

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
        topdata = data['metadata']
        benchdata = cls._merge_from_benchmarks(data['benchmarks'], topdata)
        metadata = collections.ChainMap(topdata, benchdata)
        self = cls(metadata, data['version'])
        self._topdata = topdata
        self._benchdata = benchdata
        self._benchmarks = data['benchmarks']
        return self

    @classmethod
    def overwrite_raw(cls, data, field, value, *, addifnotset=True):
        _, modified = cls._overwrite(data['metadata'], field, value, addifnotset)
        return modified

    @classmethod
    def overwrite_raw_all(cls, data, field, value):
        _, modified = cls._overwrite(data['metadata'], field, value)
        for benchdata in PyperfResults.iter_benchmarks_from_data(data):
            name = PyperfResults.get_benchmark_name(benchdata)
            benchmeta = PyperfResults.get_benchmark_metadata_raw(benchdata)
            context = f'benchmark {name!r}'
            _, _modified = cls._overwrite(benchmeta, field, value, context,
                                          addifnotset=False)
            if _modified:
                modified = True
            # XXX Update per-run metadata too?
        return modified

    @classmethod
    def _overwrite(cls, data, field, value, context=None, addifnotset=True):
        context = f' for {context}' if context else ''
        try:
            old = data[field]
        except KeyError:
            old = None
        modified = False
        if not old:
            if addifnotset:
                logger.debug(f'# initializing {field} in results metadata{context} ({value})')
                data[field] = value
                modified = True
            else:
                logger.debug(f'# {field} empty/missing in results metadata{context}; ignoring)')
        elif old != value:
             logger.warn(f'replacing {field} in results metadata{context} ({old} -> {value})')
             data[field] = value
             modified = True
        return old, modified

    @classmethod
    def _merge_from_benchmarks(cls, data, topdata):
        metadata = {}
        for bench in data:
            for key, value in bench['metadata'].items():
                if key not in cls.EXPECTED:
                    continue
                if not value:
                    continue
                if key in topdata:
                    if value != topdata[key]:
                        logger.warn(f'top/per-benchmark metadata mismatch for {key} (top: {topdata[key]!r}, bench: {value!r}); ignoring')
                elif key in metadata:
                    if metadata[key] is None:
                        continue
                    if value != metadata[key]:
                        logger.warn(f'per-benchmark metadata mismatch for {key} ({value!r} != {metadata[key]!r}); ignoring')
                        metadata[key] = None
                else:
                    metadata[key] = value
            # XXX Incorporate pre-run metadata?
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
            text = self._data.get('python_version')
            impl = self.python_implementation
            parsed = impl.parse_version(text)
            if not parsed and hasattr(impl.VERSION, 'parse_extended'):
                parsed = impl.VERSION.parse_extended(text)
                if parsed:
                    parsed, _, _ = parsed
                else:
                    # XXX This should have been covered by parse_extended().
                    parsed = impl.parse_version(text.split()[0])
            if not parsed:
                parsed = None
            self._pyversion = parsed
            return parsed

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
        old, _ = self._overwrite(self._data, field, value)
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
            results.date,
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
                    resultsroot=None, date=None):
        uploadid = PyperfUploadID.from_raw(uploadid, fail=True)
        build = cls._normalize_build(build)
        return cls._from_values(
            uploadid, build, filename, compared, resultsroot, date)

    @classmethod
    def _from_values(cls, uploadid, build, filename, compared,
                     resultsroot, date):
        if filename:
            (filename, relfile, resultsroot,
             ) = normalize_results_filename(filename, resultsroot)
        elif resultsroot or compared:
            raise ValueError('missing filename')
        if compared:
            compared = PyperfComparison.from_raw(compared, fail=True)
        self = cls.__new__(cls, uploadid, build, filename, compared)
        if resultsroot:
            self._resultsroot = resultsroot
            self._relfile = relfile
        if date:
            self._date = date
        return self

    @classmethod
    def _normalize_build(cls, build):
        if not build:
            return build
        if isinstance(build, str):
            # "PGO,LTO"
            build = build.split(',')
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
            self._relfile = _utils.strinct_relpath(self.filename,
                                                   self._resultsroot)
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
    def date(self):
        try:
            return self._date
        except AttributeError:
            results = self.resfile.read()
            self._date = results.date
            return self._date

    @property
    def baseline(self):
        if not self.compared:
            return None
        if not self.compared.baseline:
            return None
        return self.compared.baseline.source

    @property
    def mean(self):
        if not self.compared:
            return None
        if not self.compared.baseline:
            return None
        return self.compared.mean

    @property
    def isbaseline(self):
        return self.compared and not self.compared.baseline

    @property
    def sortkey(self):
        return (*self.uploadid.sortkey, self.date)

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

    def find_comparison(self, comparisons):
        try:
            return comparisons.bysource[self.filename]
        except KeyError:
            return None

    def load_results(self):
        resfile = self.resfile
        if not resfile:
            return None
        return resfile.read()

    def as_rendered_row(self, columns):
        results = None
        row = []
        for column in columns:
            if column == 'date':
                rendered = self.date.strftime('%Y-%m-%d (%H:%M UTC)')
            elif column == 'release':
                rendered = f'{self.uploadid.impl} {self.uploadid.version}'
            elif column == 'commit':
                rendered = self.uploadid.commit[:10]
            elif column == 'host':
                host = self.uploadid.host
                if isinstance(host, _utils.HostInfo):
                    rendered = str(host.id)
                else:
                    rendered = str(host)
            elif column == 'baseline':
                rendered = self.baseline or ''
            elif column == 'mean':
                if self.isbaseline:
                    rendered = self.BASELINE_REF
                else:
                    rendered = str(self.mean) if self.mean else ''
            else:
                raise NotImplementedError(column)
            row.append(rendered)
        return row


class PyperfResultsIndex:

#    iter_all()
#    add()
#    ensure_means()

    def __init__(self):
        self._entries = []

    def __eq__(self, other):
        raise NotImplementedError

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
            if entry.mean == PyperfComparisonValue.BASELINE:
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
        # XXX Do not add if already added.
        # XXX Fail if compatid is different but fails are same?
        self._entries.append(info)

    def add_from_results(self, results, compared=None):
        info = PyperfResultsInfo.from_results(results, compared)
        return self.add(info)

    def ensure_means(self, baseline=None):
        requested = _utils.Version.from_raw(baseline).full if baseline else None

        by_suite = {}
        baselines = {}
        entry_indices = {}
        for i, info in enumerate(self._entries):
            suite = info.uploadid.suite
            if suite not in PyperfUploadID.SUITES:
                raise NotImplementedError((suite, info))
            if info.uploadid.version.full == requested:
                assert suite not in baselines, info
                baselines[suite] = info
            else:
                if suite not in by_suite:
                    by_suite[suite] = []
                by_suite[suite].append(info)
            entry_indices[info] = i
        updated = []
        for suite, infos in by_suite.items():
            baseline = baselines[suite]
            comparisons = baseline.resfile.compare([i.resfile for i in infos])
#            means = comparisons.table.mean_row.others
            for info in infos:
                compared = info.find_comparison(comparisons)
                if not compared:
                    raise NotImplementedError(info)
                if info.mean == compared.mean:
                    continue
                i = entry_indices[info]
                copied = info._replace(compared=compared)
                entry_indices[copied] = i
                self._entries[i] = copied
                updated.append((info, copied))
        return updated

    def as_rendered_rows(self, columns):
        for info in self._entries:
            yield info.as_rendered_row(columns), info


##################################
# results files

def normalize_results_filename(filename, resultsroot=None):
    if not filename:
        raise ValueError('missing filename')
    if resultsroot and not os.path.isabs(resultsroot):
        raise ValueError(resultsroot)
    if os.path.isabs(filename):
        if resultsroot:
            relfile = _utils.strict_relpath(filename, resultsroot)
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

    SUFFIX = '.json'
    COMPRESSED_SUFFIX = '.json.gz'
    COMPRESSOR = gzip
    _SUFFIXES = (SUFFIX, COMPRESSED_SUFFIX)

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
        uploadid = PyperfUploadID.from_raw(uploadid, fail=True)
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
#            *(_utils.strict_relpath(o.filename, cwd)
#              for o in others),
            *(o._relfile for o in others),
            cwd=cwd,
        )
        if proc.returncode:
            logger.warn(proc.stdout)
            return None
        filenames = [
            self._filename,
#            *(os.path.join(cwd, o.filename) for o in others),
            *(o.filename for o in others),
        ]
        return PyperfComparisons.parse_table(proc.stdout, filenames)
#        return PyperfTable.parse(proc.stdout, filenames)


class PyperfResultsDir:

    INDEX = 'index.tsv'
    INDEX_FIELDS = [
        'relative path',
        'uploadid',
        'build',
        'baseline',
        'geometric mean',
    ]

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
        _utils.check_str(root, 'root', required=True, fail=True)
        if not os.path.isabs(root):
            raise ValueError(root)
        self._root = root
        self._indexfile = os.path.join(root, self.INDEX)

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

    def _info_from_values(self, relfile, uploadid, build=None,
                          baseline=None, mean=None, *,
                          baselines=None):
        assert not os.path.isabs(relfile), relfile
        filename = os.path.join(self._root, relfile)
        if not build:
            build = ['PGO', 'LTO']
#            # XXX Get it from somewhere.
#            raise NotImplementedError
        if baseline:
            assert not os.path.isabs(baseline), baseline
            baseline = os.path.join(self._root, baseline)
            if baselines is not None:
                try:
                    baseline = baselines[baseline]
                except KeyError:
                    baseline = PyperfComparisonBaseline(baseline)
                    baselines[baseline] = baseline
            else:
                baseline = PyperfComparisonBaseline(baseline)
            compared = PyperfComparison(baseline, source=filename, mean=mean)
        else:
            assert not mean, mean
            compared = None
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
        rows = self.iter_from_files()
        for info in sorted(rows, key=(lambda r: r.sortkey)):
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
        # We use a basic tab-separated values format.
        rows = []
        baselines = {}
        for row in self._read_rows():
            parsed = self._parse_row(row)
            info = self._info_from_values(*parsed, baselines=baselines)
            rows.append(info)
        index = PyperfResultsIndex()
        for info in rows:
            index.add(info)
        return index

    def _read_rows(self):
        with open(self._indexfile) as infile:
            text = infile.read()
        rows = iter(l
                    for l in text.splitlines()
                    if l and not l.startswith('#'))
        # First read the header.
        try:
            headerstr = next(rows)
        except StopIteration:
            raise NotImplementedError(self._indexfile)
        if headerstr != '\t'.join(self.INDEX_FIELDS):
            raise ValueError(header)
        # Now read the rows.
        return rows

    def _parse_row(self, row):
        rowstr = row
        row = rowstr.split('\t')
        if len(row) != len(self.INDEX_FIELDS):
            raise ValueError(rowstr)
        relfile, uploadid, build, baseline, mean = row
        uploadid = PyperfUploadID.parse(uploadid)
        if not uploadid:
            raise ValueError(f'bad uploadid in {rowstr}')
        if not relfile:
            raise ValueError(f'missing relative path for {uploadid}')
        elif os.path.isabs(relfile):
            raise ValueError(f'got absolute relative path {relfile!r}')
        if baseline:
            if os.path.isabs(baseline):
                raise ValueError(f'got absolute relative path {baseline!r}')
            if not mean:
                raise ValueError('missing mean')
        elif mean:
            raise ValueError('missing baseline')
        return relfile, uploadid, build or None, baseline or None, mean or None

    def save_index(self, index):
        # We use a basic tab-separated values format.
        rows = [self.INDEX_FIELDS]
        for info in sorted(index.iter_all(), key=(lambda v: v.sortkey)):
            row = self._render_as_row(info)
            rows.append(
                [(row[f] or '') for f in self.INDEX_FIELDS]
            )
        rows = ('\t'.join(row) for row in rows)
        text = os.linesep.join(rows)
        with open(self._indexfile, 'w', encoding='utf-8') as outfile:
            outfile.write(text)
            print(file=outfile)  # Add a blank line at the end.

    def _render_as_row(self, info):
        if not info.filename:
            raise NotImplementedError(info)
        if info.resultsroot != self._root:
            raise NotImplementedError((info, self._root))
        return {
            'relative path': _utils.strict_relpath(info.filename, self._root),
            'uploadid': str(info.uploadid),
            'build': ','.join(info.build) if info.build else None,
            'baseline': (_utils.strict_relpath(info.baseline, self._root)
                         if info.baseline
                         else None),
            'geometric mean': str(info.mean) if info.mean else None,
        }

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
        try:
            index = self.load_index(baseline=baseline, createifmissing=False)
        except FileNotFoundError:
            index = PyperfResultsIndex()
        for results in copied:
            info = index.add_from_results(results)
            yield info
        if baseline:
            index.ensure_means(baseline=baseline)
        self.save_index(index)
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

    @classmethod
    def from_remote(cls, remote, root, datadir=None, baseline=None):
        if not root or not _utils.check_str(root):
            root = None
        elif not os.path.isabs(root):
            raise ValueError(root)
        if isinstance(remote, str):
            remote = _utils.GitHubTarget.resolve(remote, root)
        elif not isinstance(remote, _utils.GitHubTarget):
            raise TypeError(f'unsupported remote {remote!r}')
        raw = remote.ensure_local(root)
#        raw.clean()
#        raw.switch_branch('main')
        kwargs = {}
        if datadir:
            kwargs['datadir'] = datadir
        if baseline:
            kwargs['baseline'] = baseline
        return cls(raw, remote, **kwargs)

    @classmethod
    def from_root(cls, root, datadir=None, baseline=None):
        if not root or not _utils.check_str(root):
            root = None
        elif not os.path.isabs(root):
            raise ValueError(root)
        raw = _utils.GitLocalRepo.ensure(root)
        if not raw.exists:
            raise FileNotFoundError(root)
        remote = None
        kwargs = {}
        if datadir:
            kwargs['datadir'] = datadir
        if baseline:
            kwargs['baseline'] = baseline
        return cls(raw, remote, **kwargs)

    def __init__(self, raw, remote=None, datadir=None, baseline=None):
        if not raw:
            raise ValueError('missing raw')
        elif not isinstance(raw, _utils.GitLocalRepo):
            raise TypeError(raw)
        if remote and not isinstance(remote, _utils.GitHubTarget):
            raise TypeError(f'unsupported remote {remote!r}')
        self._raw = raw
        self.datadir = datadir or None
        self.remote = remote or None

        self._baseline = baseline
        self._resultsdir = PyperfUploadsDir(
            raw.resolve(datadir) if datadir else raw.root,
        )

    @property
    def root(self):
        return self._raw.root

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
            clean=True,
            push=True,
            ):
        repo = self._raw.using_author(author)
        if clean:
            repo.refresh()
        repo.switch_branch(branch or self.BRANCH)

        #added = self._resultsdir.add(info, ...)
        added = self._resultsdir.add_from_results(
            results,
            baseline=self._baseline,
            compressed=compressed,
            split=split,
        )
        added = list(added)  # Force the iterator to complete.
        index = self._resultsdir.load_index()
        readme = self._update_table(index)

        logger.info('committing to the repo...')
        for info in added:
            repo.add(info.filename)
        repo.add(self._resultsdir.indexfile)
        repo.add(readme)
        msg = f'Add Benchmark Results ({info.uploadid.copy(suite=None)})'
        repo.commit(msg)
        logger.info('...done committing')

        if push:
            self._upload(self.datadir or '.')

    def _update_table(self, index):
        table_lines = self._render_markdown(index)
        MARKDOWN_START = '<!-- START results table -->'
        MARKDOWN_END = '<!-- END results table -->'
        filename = self._raw.resolve('README.md')
        with open(filename) as infile:
            text = infile.read()
        try:
            start = text.index(MARKDOWN_START)
        except ValueError:
            start = end = -1
            sep = os.linesep * 2
        else:
            end = text.index(MARKDOWN_END, start) + len(MARKDOWN_END)
            sep = ''
        text = (text[:start] +
                sep +
                os.linesep.join([
                    MARKDOWN_START,
                    *table_lines,
                    MARKDOWN_END,
                ]) +
                text[end:])
        with open(filename, 'w') as outfile:
            outfile.write(text)
        return filename

    def _render_markdown(self, index):
        def render_row(row):
            row = (f' {v} ' for v in row)
            return f'| {"|".join(row)} |'
        columns = 'date release commit host mean'.split()

        rows = index.as_rendered_rows(columns)
        by_suite = {}
        for row, info in sorted(rows, key=(lambda r: r[1].sortkey)):
            suite = info.uploadid.suite
            if suite not in by_suite:
                by_suite[suite] = []
            date, release, commit, host, mean = row
            relpath = self._raw.relpath(info.filename)
            relpath = relpath.replace('\/', '/')
            date = f'[{date}]({relpath})'
            if not mean:
#                assert info.isbaseline, repr(info)
#                assert not mean, repr(mean)
                mean = PyperfComparisonValue.BASELINE
            assert '3.10.4' not in release or mean == '(ref)', repr(mean)
            row = date, release, commit, host, mean
            by_suite[suite].append(row)

        for suite, rows in sorted(by_suite.items()):
            yield ''
            yield f'{suite or "???"}:'
            yield ''
            yield render_row(columns)
            yield render_row(['---'] * len(columns))
            for row in rows:
                yield render_row(row)
        yield ''

    def _upload(self, reltarget):
        if not self.remote:
            raise Exception('missing remote')
        url = f'{self.remote.url}/tree/main/{reltarget}'
        logger.info(f'uploading results to {url}...')
        self._raw.push(self.remote.push_url)
        logger.info('...done uploading')


##################################
# faster-cpython

class FasterCPythonResults(PyperfResultsRepo):

    REMOTE = _utils.GitHubTarget.from_origin('faster-cpython', 'ideas', ssh=True)
    DATADIR = 'benchmark-results'
    BASELINE = '3.10.4'

    @classmethod
    def from_remote(cls, remote=None, root=None, baseline=None):
        if not remote:
            remote = cls.REMOTE
        return super().from_remote(remote, root, baseline=baseline)

    @classmethod
    def from_root(cls, root, baseline=None):
        raise NotImplementedError

    def __init__(self, root, remote, baseline=BASELINE):
        if not remote:
            raise ValueError('missing remote')
        super().__init__(root, remote, self.DATADIR, baseline)
