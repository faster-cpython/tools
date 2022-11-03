from collections import namedtuple
import collections
import datetime
import gzip
import hashlib
import json
import logging
import os
import os.path
import re
import shutil
import sys
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional,
    Sequence, Tuple, Union
)

from . import _utils


logger = logging.getLogger(__name__)


SuiteType = Union[None, str, _utils.Sentinel]
SuitesType = Any  # This is a recursive iterable of SuiteType


##################################
# pyperformance helpers

class BenchmarkSuiteInfo(
        namedtuple('BenchmarkSuiteInfo', 'name url reldir show_results')):
    """A single benchmark suite."""

    def __new__(
            cls,
            name: str,
            url: str,
            reldir: str,
            show_results: bool = True
    ):
        return super().__new__(
            cls,
            name=name,
            url=url,
            reldir=reldir,
            show_results=show_results,
        )

    def __hash__(self):
        return hash(self.name)


class Benchmarks:

    REPOS = os.path.join(_utils.HOME, 'repos')

    PYPERFORMANCE = 'pyperformance'
    PYSTON = 'pyston'
    _SUITES: Dict[str, Dict[str, Any]] = {
        PYPERFORMANCE: {
            'url': 'https://github.com/python/pyperformance',
            'reldir': 'pyperformance/data-files/benchmarks',
            'show_results': True,
        },
        PYSTON: {
            'url': 'https://github.com/pyston/python-macrobenchmarks',
            'reldir': 'benchmarks',
            # We hide the pyston benchmarks for now, pending resolution
            # of https://github.com/faster-cpython/ideas/issues/434.
            'show_results': True,
        },
    }
    SUITES: Dict[str, BenchmarkSuiteInfo] = {}
    for _suitename in _SUITES:
        SUITES[_suitename] = BenchmarkSuiteInfo(
            _suitename,
            **_SUITES[_suitename]  # type: ignore[arg-type]
        )
    del _suitename
    del _SUITES

    @classmethod
    def _load_suite(cls, suite: str) -> List[str]:
        info = cls.SUITES[suite]
        url = info.url
        reldir = info.reldir
        reporoot = os.path.join(cls.REPOS,
                                os.path.basename(url))
        if not os.path.exists(reporoot):
            if not os.path.exists(cls.REPOS):
                os.makedirs(cls.REPOS)
            _utils.git('clone', url, reporoot, cwd=None)
        names = cls._get_names(os.path.join(reporoot, reldir))
        return list(names)

    @classmethod
    def _get_names(cls, benchmarksdir: str) -> Iterable[str]:
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
    def _iter_subcandidates(cls, bench: str) -> Iterable[str]:
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

    def get_suites(
            self,
            benchmarks: Iterable[str],
            default: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        mapped: Dict[str, Optional[str]] = {}
        suite: Any
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
            if suite is None or isinstance(suite, str):
                mapped[bench] = suite
            else:
                raise TypeError(f"Invalid suite type {type(suite)}")
        return mapped

    def get_suite(
            self,
            bench: str,
            default: Optional[str] = None
    ) -> Optional[str]:
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

    def load(
            self,
            key: str = 'name'
    ) -> Union[Dict[str, List[str]], Dict[str, str]]:
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
            if suite not in self._cache:
                self._cache[suite] = self._load_suite(suite)
        return self._cache


class PyperfUploadID(namedtuple('PyperfUploadName',
                                'impl version commit host compatid suite')):
    # See https://github.com/faster-cpython/ideas/tree/main/benchmark-results/README.md
    # for details on this filename format.

    SUITE_NOT_KNOWN = None
    EMPTY = _utils.Sentinel('empty')
    MULTI_SUITE = _utils.Sentinel('multi-suite')
    SUITES = set(Benchmarks.SUITES)
    _SUITES: Dict[SuiteType, SuiteType]
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

    _name: str
    _prefix: Optional[str] = None
    _suffix: Optional[str] = None
    _filename: str
    _dirname: Optional[str] = None

    @classmethod
    def from_raw(
            cls,
            raw: Any,
            *,
            fail: Optional[bool] = None
    ) -> Optional["PyperfUploadID"]:
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
    def from_filename(cls, filename: str) -> Optional["PyperfUploadID"]:
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
    def parse(
            cls,
            name,
            *,
            allowprefix: bool = False,
            allowsuffix: bool = False
    ) -> Optional["PyperfUploadID"]:
        self = cls._parse(name)
        if self:
            if not allowprefix and self._prefix:
                return None
            if not allowsuffix and self._suffix:
                return None
        return self

    @classmethod
    def _parse(cls, uploadid: str) -> Optional["PyperfUploadID"]:
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
        resolved_impl = _utils.resolve_python_implementation(impl)
        if verstr == 'main':
            version = resolved_impl.VERSION.resolve_main()
            name = name.replace('-main-', f'-{version}-')
        else:
            version = resolved_impl.parse_version(verstr)
        self = cls(resolved_impl, version, commit, host, compatid, suite)
        self._name = name
        self._prefix = prefix or None
        self._suffix = suffix or None
        return self

    @classmethod
    def from_metadata(
            cls,
            metadata: "PyperfResultsMetadata",
            *,
            version: Optional[Any] = None,
            commit: Optional[str] = None,
            host: Optional[_utils.HostInfo] = None,
            impl: Optional[str] = None,
            suite: Optional[Any] = None,
    ) -> "PyperfUploadID":
        resolved_metadata = PyperfResultsMetadata.from_raw(metadata)
        if resolved_metadata is None:
            raise ValueError("Couldn't load metadata")
        resolved_impl = _utils.resolve_python_implementation(
            impl or metadata.python_implementation or 'cpython',
        )
        if not version:
            # We assume "main" if it's missing.
            normalized_version = metadata.pyversion or 'main'
        else:
            normalized_version = version
        if normalized_version == 'main':
            resolved_version = resolved_impl.VERSION.resolve_main()
        else:
            resolved_version = resolved_impl.parse_version(
                normalized_version,
                requirestr=False
            )

        self = cls(
            impl=resolved_impl,
            version=resolved_version,
            commit=commit or resolved_metadata.commit,
            host=host or resolved_metadata.host,
            compatid=resolved_metadata.compatid,
            suite=suite,
        )
        return self

    @classmethod
    def build_compatid(
            cls,
            host: _utils.HostInfo,
            pyperformance_version: str,
            pyperf_version: Optional[str] = None
    ) -> str:
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
    def normalize_suite(cls, suite: SuitesType) -> SuiteType:
        if not suite:
            return cls.SUITE_NOT_KNOWN

        if (
                not isinstance(suite, str) and
                _utils.iterable(suite) and
                not isinstance(suite, _utils.Sentinel)
        ):
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

    def __new__(
            cls,
            impl: _utils.PythonImplementation,
            version: str,
            commit: str,
            host: Union[str, _utils.HostInfo],
            compatid: str,
            suite: SuiteType = SUITE_NOT_KNOWN,
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
    def name(self) -> str:
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
    def implementation(self) -> str:
        return self.impl

    @property
    def filename(self) -> str:
        try:
            return self._filename
        except AttributeError:
            filename, *_ = self.resolve_filenames()
            return filename

    @property
    def sortkey(self) -> Tuple:
        # We leave commit and suite out.
        return (self.impl, self.version, self.host, self.compatid)

    def resolve_filenames(
            self,
            *,
            dirname: Union[bool, str, Iterable[str]] = True,
            prefix: Optional[Union[bool, str, Iterable[str]]] = True,
            suffix: Optional[Union[bool, str, Iterable[str]]] = True
    ) -> Iterable[str]:
        dirnames: List[str] = []
        if dirname is True:
            if self._dirname:
                dirnames = [self._dirname]
        elif dirname:
            if isinstance(dirname, str):
                dirnames = [dirname]
            else:
                dirnames = list(dirname)
            if any(not d for d in dirnames):
                raise ValueError(f'blank dirname in {dirname}')

        prefixes: List[Optional[str]] = [None]
        if prefix is True:
            if self._prefix:
                prefixes = [self._prefix]
        elif prefix:
            if isinstance(prefix, str):
                prefixes = [prefix]
            else:
                prefixes = list(prefix)
            if any(not p for p in prefixes):
                raise ValueError(f'blank prefix in {prefix}')

        suffixes: List[Optional[str]] = [None]
        if suffix is True:
            if self._suffix:
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
                    yield from (os.path.join(d, filename) for d in dirnames)
                else:
                    yield filename

    def match(
            self,
            specifier: Any,
            suites=None
    ) -> bool:
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

    def _match(self, specifier: Any, checksuite) -> bool:
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

    def _match_version(self, version: Union[str, _utils.Version]) -> bool:
        if isinstance(version, str):
            version_obj = _utils.Version.parse(version)
            if not version_obj:
                return False
        elif not isinstance(version, _utils.Version):
            return False
        else:
            version_obj = version
        # XXX Treat missing micro/release as wildcard?
        return version_obj.full == self.version.full

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
    def parse(
            cls,
            valuestr: str,
            *,
            fail: bool = False
    ) -> Optional["PyperfComparisonValue"]:
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

    def __init__(
            self,
            elapsed: Optional[_utils.ElapsedTimeWithUnits] = None,
            comparison: Optional[_utils.ElapsedTimeComparison] = None
    ):
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
    def elapsed(self) -> Optional[_utils.ElapsedTimeWithUnits]:
        return self._elapsed

    @property
    def comparison(self) -> Optional[_utils.ElapsedTimeComparison]:
        return self._comparison

    @property
    def isbaseline(self) -> bool:
        if self._elapsed and not self._comparison:
            return True
        return self._comparison == self.BASELINE


class _PyperfComparison:

    kind: Optional[str] = None

    @classmethod
    def from_raw(
            cls,
            raw: Any,
            *,
            fail: Optional[bool] = None
    ) -> Optional["_PyperfComparison"]:
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
    def _parse_value(cls, valuestr: str) -> Any:
        # Each subclass has a different return type for this function
        return _utils.ElapsedTimeWithUnits.parse(valuestr, fail=True)

    def __init__(
            self,
            source: Any,
            byname: Optional[Dict[str, str]] = None
    ):
        _utils.check_str(source, 'source', required=True, fail=True)
        if not os.path.isabs(source):
            raise ValueError(f'expected an absolute source, got {source!r}')
        # XXX Further validate source as a filename?

        _byname: Dict[str, Any] = {}
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

    def _as_hashable(self) -> Tuple[Any, Tuple]:
        return (
            self._source,
            tuple(sorted(self._byname.items())) if self._byname else (),
        )

    @property
    def source(self) -> str:
        return self._source

    @property
    # Dict values match the return type of _parse_value()
    def byname(self) -> Dict[str, Any]:
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
    def _parse_value(cls, valuestr: str) -> PyperfComparisonValue:
        # Each subclass has a different return type for this function
        result = PyperfComparisonValue.parse(valuestr, fail=True)
        assert result is not None
        return result

    def __init__(
            self,
            baseline: Any,
            source: Any,
            mean: Any,
            byname: Optional[Dict[str, str]] = None
    ):
        super().__init__(source, byname)
        baseline = PyperfComparisonBaseline.from_raw(baseline, fail=True)
        if self._byname and sorted(self._byname) != sorted(baseline.byname):
            b1 = sorted(self._byname)
            b2 = sorted(baseline.byname)
            raise ValueError(f'mismatch with baseline ({b1} != {b2})')
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
    def baseline(self) -> PyperfComparisonBaseline:
        return self._baseline

    @property
    def mean(self) -> Optional[_utils.ElapsedTimeComparison]:
        return self._mean

    def look_up(self, name) -> "PyperfComparison.Summary":
        compared = self._byname[name]
        assert isinstance(compared, PyperfComparisonValue)

        return self.Summary(
            name,
            self._baseline.source,
            self._baseline.byname[name],
            self._source,
            compared.elapsed,
            compared.comparison,
        )


class PyperfComparisons:
    """The baseline and comparisons for a set of results."""

    _table: "PyperfTable"

    @classmethod
    def parse_table(
            cls,
            text: str,
            filenames: Optional[Iterable[str]] = None
    ) -> "PyperfComparisons":
        table = PyperfTable.parse(text, filenames)
        if table is None:
            raise ValueError("Could not parse table")
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table: "PyperfTable") -> "PyperfComparisons":
        base_byname = {}
        bysource: Dict[str, Any] = {s: {} for s in table.header.others}
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

    def __init__(
            self,
            baseline: PyperfComparisonBaseline,
            bysource: Mapping[str, dict]
    ):
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
    def baseline(self) -> PyperfComparisonBaseline:
        return self._baseline

    @property
    def bysource(self) -> Dict[str, Any]:
        return dict(self._bysource)

    @property
    def table(self) -> "PyperfTable":
        try:
            return self._table
        except AttributeError:
            raise NotImplementedError


class PyperfTableParserError(ValueError):
    MSG = 'failed parsing results table'
    FIELDS = 'text reason'.split()

    def __init__(
            self,
            text: Optional[str],
            reason: Optional[str] = None,
            msg: Optional[str] = None
    ):
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

    def __init__(
            self,
            line: str,
            reason: Optional[str] = None,
            msg: Optional[str] = None
    ):
        self.line = line
        super().__init__(line, reason, msg)


class PyperfTableRowUnsupportedLineError(PyperfTableRowParserError):
    MSG = 'unsupported table row line {line!r}'

    def __init__(self, line: str, msg: Optional[str] = None):
        super().__init__(line, 'unsupported', msg)


class PyperfTableRowInvalidLineError(PyperfTableRowParserError):
    MSG = 'invalid table row line {line!r}'

    def __init__(self, line: str, msg: Optional[str] = None):
        super().__init__(line, 'invalid', msg)


class PyperfTable:

    FORMATS = ['raw', 'meanonly']

    _mean_row: "PyperfTableRow"

    @classmethod
    def parse(
            cls,
            text: str,
            filenames: Optional[Iterable[str]] = None
    ) -> Optional["PyperfTable"]:
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
            except PyperfTableRowUnsupportedLineError:
                if not line:
                    # end-of-table
                    break
                elif line.startswith('Ignored benchmarks '):
                    # end-of-table
                    ignored, _ = cls._parse_ignored(line, lines)
                    # XXX Add the names to the table.
                    line = _utils.get_next_line(lines, skipempty=True) or ''
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
    def _parse_names_list(
            cls,
            line: Optional[str],
            lines: Iterator[str],
            prefix: Optional[str] = None
    ) -> Tuple[Optional[List[str]], Optional[str]]:
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
    def _parse_ignored(
            cls,
            line: Optional[str],
            lines: Iterator[str],
            required: bool = True
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        # Ignored benchmarks (2) of cpython-3.10.4-9d38120e33-fc_linux-42d6dd4409cb.json: genshi_text, genshi_xml
        prefix = r'Ignored benchmarks \((\d+)\) of \w+.*\w'
        names, line = cls._parse_names_list(line, lines, prefix)
        if not names and required:
            raise PyperfTableParserError(line, 'expected "Ignored benchmarks..."')
        return names, line

    @classmethod
    def _parse_hidden(
            cls,
            line: Optional[str],
            lines: Iterator[str],
            required: bool = True
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        # Benchmark hidden because not significant (6): unpickle, scimark_sor, sqlalchemy_imperative, sqlite_synth, json_loads, xml_etree_parse
        prefix = r'Benchmark hidden because not significant \((\d+)\)'
        names, line = cls._parse_names_list(line, lines, prefix)
        if not names and required:
            raise PyperfTableParserError(line, 'expected "Benchmarks hidden..."')
        return names, line

    def __init__(self, rows: Any, header: Any = None):
        if not isinstance(rows, tuple):
            rows = tuple(rows)
        if not header:
            header = rows[0].header
        if header[0] != 'Benchmark':
            raise ValueError(f'unsupported header {header}')
        self.header = header
        self.rows = rows
        self._text: Optional[str] = None

    def __repr__(self):
        return f'{type(self).__name__}({self.rows}, {self.header})'

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def mean_row(self) -> "PyperfTableRow":
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

    def render(self, fmt: Optional[str] = None) -> Iterable[str]:
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
    _raw: Optional[str] = None
    _header: "PyperfTableHeader"
    _name: str
    _values: Tuple
    _baseline: str
    _others: Tuple

    @classmethod
    def parse(
            cls,
            line: str,
            *,
            fail: bool = False
    ):  # TODO: Use Self in Python 3.11 and later
        values = cls._parse(line, fail)
        if not values:
            return None
        self = tuple.__new__(cls, values)
        self._raw = line
        return self

    @classmethod
    def _parse(cls, line: str, fail: bool = False) -> Optional[Tuple]:
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
    def raw(self) -> Optional[str]:
        return self._raw

    @property
    def name(self) -> str:
        try:
            return self._name
        except AttributeError:
            self._name = self[0]
            return self._name

    @property
    def values(self) -> Tuple:
        try:
            return self._values
        except AttributeError:
            self._values = self[1:]
            return self._values

    @property
    def baseline(self) -> str:
        try:
            return self._baseline
        except AttributeError:
            self._baseline = self[1]
            return self._baseline

    @property
    def others(self) -> Tuple:
        try:
            return self._others
        except AttributeError:
            self._others = self[2:]
            return self._others

    def render(self, fmt: Optional[str] = None) -> str:
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
    def parse(
            cls,
            line: str,
            filenames: Optional[Iterable[str]] = None,
            *,
            fail: bool = False
    ) -> Optional["PyperfTableHeader"]:
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
    def div(self) -> str:
        return '+=' + '=+='.join('=' * len(v) for v in self) + '=+'

    @property
    def rowdiv(self) -> str:
        return '+-' + '-+-'.join('-' * len(v) for v in self) + '-+'

    @property
    def indexpersource(self) -> Dict[str, Iterable[int]]:
        return dict(
            zip(  # type: ignore[call-overload]
                self.sources, range(len(self.sources))  # type: ignore[arg-type]
            )
        )


class PyperfTableRow(_PyperfTableRowBase):
    @classmethod
    def subclass_from_header(
            cls,
            header: PyperfTableHeader
    ):
        if cls is not PyperfTableRow:
            raise TypeError('not supported for subclasses')
        if not header:
            raise ValueError('missing header')

        class _PyperfTableRow(PyperfTableRow):
            @classmethod
            def parse(  # type: ignore[override]
                    cls,
                    line: str,
                    _header: PyperfTableHeader = header,
                    fail: bool = False
            ) -> Optional["_PyperfTableRowBase"]:
                return super().parse(line, header=_header, fail=fail)
        return _PyperfTableRow

    @classmethod
    def parse(  # type: ignore[override]
            cls,
            line: str,
            header: PyperfTableHeader,
            fail: bool = False
    ) -> Optional["PyperfTableRow"]:
        self = super().parse(line, fail=fail)
        if not self:
            return None
        if len(self) != len(header):
            raise ValueError(f'expected {len(header)} values, got {tuple(self)}')
        self._header = header
        return self

    @property
    def header(self) -> PyperfTableHeader:
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

    _metadata: "PyperfResultsMetadata"
    _uploadid: PyperfUploadID
    _by_bench: Mapping[str, Any]
    _by_suite: Mapping[SuiteType, Any]

    @classmethod
    def get_metadata_raw(cls, data) -> MutableMapping[str, str]:
        return data['metadata']

    @classmethod
    def iter_benchmarks_from_data(cls, data) -> Iterator[str]:
        yield from data['benchmarks']

    @classmethod
    def get_benchmark_name(cls, benchdata) -> str:
        return benchdata['metadata']['name']

    @classmethod
    def get_benchmark_metadata_raw(
            cls,
            benchdata
    ) -> MutableMapping[str, str]:
        return benchdata['metadata']

    @classmethod
    def iter_benchmark_runs_from_data(
            cls,
            benchdata
    ) -> Iterator[Tuple[Mapping[str, str], List[float], List[float]]]:
        for rundata in benchdata['runs']:
            yield (
                rundata['metadata'],
                rundata.get('warmups'),
                rundata.get('values'),
            )

    @classmethod
    def _validate_data(cls, data) -> None:
        if data['version'] == '1.0':
            for key in ('metadata', 'benchmarks', 'version'):
                if key not in data:
                    raise ValueError(f'invalid results data (missing {key})')
            # XXX Add other checks.
        else:
            raise NotImplementedError(data['version'])

    def __init__(self, data, resfile: Any):
        if not data:
            raise ValueError('missing data')
        if not resfile:
            raise ValueError('missing refile')
        self._validate_data(data)
        self._data = data
        self._resfile = PyperfResultsFile.from_raw(resfile)
        self._modified = False

    def _copy(self) -> "PyperfResults":
        cls = type(self)
        copied = cls.__new__(cls)
        copied._data = self._data
        copied._resfile = self._resfile
        return copied

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data

    @property
    def raw(self) -> Mapping[str, Any]:
        return self._data

    @property
    def raw_metadata(self) -> Mapping[str, str]:
        return self._data['metadata']

    @property
    def raw_benchmarks(self) -> List[Mapping[str, Any]]:
        return self._data['benchmarks']

    @property
    def metadata(self) -> "PyperfResultsMetadata":
        try:
            return self._metadata
        except AttributeError:
            self._metadata = PyperfResultsMetadata.from_full_results(self._data)
            return self._metadata

    @property
    def version(self) -> str:
        return self._data['version']

    @property
    def resfile(self) -> "PyperfResultsFile":
        return self._resfile

    @property
    def filename(self) -> str:
        return self._resfile.filename

    @property
    def date(self) -> datetime.datetime:
        run0 = self.raw_benchmarks[0]['runs'][0]
        date = run0['metadata']['date']
        date, _ = _utils.get_utc_datetime(date)
        return date

    @property
    def uploadid(self) -> PyperfUploadID:
        try:
            return self._uploadid
        except AttributeError:
            self._uploadid = PyperfUploadID.from_metadata(
                self.metadata,
                suite=self.suites,
            )
            assert (
                getattr(self, "_modified", False) or
                self._resfile.uploadid is None or
                self._uploadid == self._resfile.uploadid or
                (
                    str(self._uploadid._replace(suite=PyperfUploadID.SUITE_NOT_KNOWN)) ==
                    str(self._resfile.uploadid._replace(suite=PyperfUploadID.SUITE_NOT_KNOWN))
                )
            ), (self._uploadid, self._resfile.uploadid)
            return self._uploadid

    @property
    def suite(self) -> str:
        return self.uploadid.suite

    @property
    def suites(self) -> Iterable[SuiteType]:
        return sorted(self.by_suite)  # type: ignore[type-var]

    @property
    def by_bench(self) -> Mapping[str, Any]:
        try:
            return self._by_bench
        except AttributeError:
            self._by_bench = dict(self._iter_benchmarks())
            return self._by_bench

    @property
    def by_suite(self) -> Mapping[SuiteType, Any]:
        try:
            return self._by_suite
        except AttributeError:
            self._by_suite = self._collate_suites()
            return self._by_suite

    def _collate_suites(self) -> Mapping[SuiteType, Any]:
        by_suite: Dict[SuiteType, Any] = {}
        names = [n for n, _ in self._iter_benchmarks()]
        if names:
            bench_suites = self.BENCHMARKS.get_suites(names, 'unknown')
            for name, suite in bench_suites.items():
                normalized_suite = PyperfUploadID.normalize_suite(suite)
                if normalized_suite not in by_suite:
                    by_suite[normalized_suite] = []
                data = self.by_bench[name]
#                by_suite[suite][name] = data
                by_suite[normalized_suite].append(data)
        else:
            logger.warning(f'empty results {self}')
        return by_suite

    def _iter_benchmarks(self) -> Iterator[Tuple[str, Any]]:
        for benchdata in self.iter_benchmarks_from_data(self._data):
            name = self.get_benchmark_name(benchdata)
            yield name, benchdata

    def split_benchmarks(self) -> Mapping[SuiteType, "PyperfResults"]:
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
        by_suite_resolved = {}
        for suite, data in by_suite.items():
            results = self._copy()
            results._data = data
            results._by_suite = {suite: data['benchmarks'][0]}
            results._modified = True
            by_suite_resolved[suite] = results
        return by_suite_resolved

    #def compare(self, others):
    #    raise NotImplementedError

    def copy_to(
            self,
            filename: Union["PyperfResultsFile", str],
            resultsroot: Optional[str] = None,
            *,
            compressed: Optional[bool] = None,
    ) -> "PyperfResults":
        if self._resfile is None:
            raise ValueError
        if isinstance(filename, PyperfResultsFile):
            filename_str = filename.filename
        else:
            filename_str = filename
        if not self._modified and os.path.exists(self._resfile.filename):
            resfile = self._resfile.copy_to(filename_str, resultsroot,
                                            compressed=compressed)
        else:
            resfile = PyperfResultsFile(filename_str, resultsroot,
                                        compressed=compressed)
            resfile.write(self)
        copied = self._copy()
        copied._resfile = resfile
        return copied


class PyperfResultsMetadata:

    _topdata: Optional[Any]
    _benchdata: Optional[Any]
    _benchmarks: Optional[List[Any]]
    _python_implementation: _utils.PythonImplementation
    _pyversion: _utils.Version
    _host: _utils.HostInfo

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
    def from_raw(
            cls,
            raw: Optional["PyperfResultsMetadata"]
    ) -> Optional["PyperfResultsMetadata"]:
        if not raw:
            return None
        elif isinstance(raw, cls):
            return raw
        else:
            raise TypeError(raw)

    @classmethod
    def from_full_results(
            cls,
            data: Mapping[str, Any]
    ) -> "PyperfResultsMetadata":
        topdata = data['metadata']
        benchdata = cls._merge_from_benchmarks(data['benchmarks'], topdata)
        metadata = collections.ChainMap(topdata, benchdata)
        self = cls(metadata, data['version'])
        self._topdata = topdata
        self._benchdata = benchdata
        self._benchmarks = data['benchmarks']
        return self

    @classmethod
    def overwrite_raw(
            cls,
            data: Mapping[str, Any],
            field: str,
            value: Any,
            *,
            addifnotset: bool = True
    ) -> bool:
        _, modified = cls._overwrite(
            data['metadata'],
            field,
            value,
            addifnotset=addifnotset
        )
        return modified

    @classmethod
    def overwrite_raw_all(
            cls,
            data: Mapping[str, Any],
            field: str,
            value: Any
    ) -> bool:
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
    def _overwrite(
            cls,
            data: MutableMapping[str, Any],
            field: str,
            value: Any,
            context: Optional[str] = None,
            addifnotset: bool = True
    ) -> Tuple["PyperfResultsMetadata", bool]:
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
            logger.warning(f'replacing {field} in results metadata{context} ({old} -> {value})')
            data[field] = value
            modified = True
        return old, modified

    @classmethod
    def _merge_from_benchmarks(
            cls,
            data: List[Mapping[str, Any]],
            topdata: Mapping[str, Any]
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for bench in data:
            for key, value in bench['metadata'].items():
                if key not in cls.EXPECTED:
                    continue
                if not value:
                    continue
                if key in topdata:
                    if value != topdata[key]:
                        logger.warning(f'top/per-benchmark metadata mismatch for {key} (top: {topdata[key]!r}, bench: {value!r}); ignoring')
                elif key in metadata:
                    if metadata[key] is None:
                        continue
                    if value != metadata[key]:
                        logger.warning(f'per-benchmark metadata mismatch for {key} ({value!r} != {metadata[key]!r}); ignoring')
                        metadata[key] = None
                else:
                    metadata[key] = value
            # XXX Incorporate pre-run metadata?
        for key, value in list(metadata.items()):
            if value is None:
                del metadata[key]
        return metadata

    def __init__(self, data, version: Optional[str] = None):
        self._data = data
        self._version = version

    def __eq__(self, other):
        raise NotImplementedError

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        yield from self._data

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def commit(self) -> str:
        return self._data['commit_id']

    @property
    def python_implementation(self) -> _utils.PythonImplementation:
        try:
            return self._python_implementation
        except AttributeError:
            impl = _utils.resolve_python_implementation(
                self._data.get('python_implementation'),
            )
            self._python_implementation = impl
            return self._python_implementation

    @property
    def pyversion(self) -> _utils.Version:
        try:
            return self._pyversion
        except AttributeError:
            text = self._data.get('python_version')
            impl = self.python_implementation
            parsed = impl.parse_version(text)
            if not parsed and hasattr(impl.VERSION, "parse_extended"):
                parsed = impl.VERSION.parse_extended(text)  # type: ignore[attr-defined]
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
    def build(self) -> Tuple[str, ...]:
        # XXX Add to PyperfUploadID?
        # XXX Extract from self._data?
        return ('PGO', 'LTO')

    @property
    def host(self) -> _utils.HostInfo:
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
    def compatid(self) -> str:
        return PyperfUploadID.build_compatid(
            self.host,
            self._data['performance_version'],
            self._data.get('perf_version'),
        )

    def overwrite(self, field: str, value: Any) -> "PyperfResultsMetadata":
        old, _ = self._overwrite(self._data, field, value)
        return old


class PyperfResultsInfo(
        namedtuple('PyperfResultsInfo', 'uploadid build filename compared')):

    _resultsroot: str
    _relfile: str
    _resfile: "PyperfResultsFile"
    _date: datetime.datetime

    @classmethod
    def from_results(
            cls,
            results: PyperfResults,
            compared: Optional[_PyperfComparison] = None
    ) -> "PyperfResultsInfo":
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
    def from_resultsfile(
            cls,
            resfile: "PyperfResultsFile",
            compared: Optional[_PyperfComparison] = None
    ) -> Tuple["PyperfResultsInfo", "PyperfResults"]:
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
    def from_file(
            cls,
            filename: str,
            resultsroot: Optional[str] = None,
            compared: Optional[_PyperfComparison] = None
    ) -> Tuple["PyperfResultsInfo", "PyperfResults"]:
        resfile = PyperfResultsFile(filename, resultsroot)
        return cls.from_resultsfile(resfile, compared)

    @classmethod
    def from_values(
            cls,
            uploadid: Any,
            build: Optional[_utils.SequenceOrStrType] = None,
            filename: Optional[str] = None,
            compared: Optional[_PyperfComparison] = None,
            resultsroot: Optional[str] = None,
            date: Optional[datetime.datetime] = None
    ) -> "PyperfResultsInfo":
        uploadid = PyperfUploadID.from_raw(uploadid, fail=True)
        build = cls._normalize_build(build)
        return cls._from_values(
            uploadid, build, filename, compared, resultsroot, date)

    @classmethod
    def _from_values(
            cls,
            uploadid: Any,
            build: Optional[Tuple[str, ...]],
            filename: Optional[str],
            compared: Optional[_PyperfComparison],
            resultsroot: Optional[str],
            date: Optional[datetime.datetime]
    ) -> "PyperfResultsInfo":
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
    def _normalize_build(
            cls,
            build: Optional[_utils.SequenceOrStrType]
    ) -> Optional[Tuple[str, ...]]:
        if not build:
            return None
        if isinstance(build, str):
            # "PGO,LTO"
            build = build.split(',')
        cls._validate_build_values(build)
        return tuple(build)

    @classmethod
    def _validate_build_values(cls, values: _utils.SequenceOrStrType) -> None:
        for i, value in enumerate(values):
            if not value:
                raise ValueError(f'build[{i}] is empty')
            elif not isinstance(value, str):
                raise TypeError(f'expected str for build[{i}], got {value!r}')
            # XXX other checks?

    def __new__(
            cls,
            uploadid: Optional[PyperfUploadID],
            build: Optional[Tuple[str, ...]] = None,
            filename: Optional[str] = None,
            compared: Optional[_PyperfComparison] = None
    ):
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
            if not isinstance(self.build, list):
                raise TypeError(self.build)
            else:
                self._validate_build_items(self.build)

        if self.filename:
            if not isinstance(self.filename, str):
                raise TypeError(self.filename)
            if not os.path.isabs(self.filename):
                raise ValueError(f'expected an absolute filename, got {self.filename!r}')

        if self.compared:
            if not isinstance(self.compared, _PyperfComparison):
                raise TypeError(self.compared)

    def __repr__(self):
        reprstr = super().__repr__()
        prefix, _, remainder = reprstr.partition('uploadid=')
        _, _, remainder = remainder.partition(', build=')
        return f'{prefix}uploadid={str(self.uploadid)!r}, build={remainder})'

    @property
    def resultsroot(self) -> Optional[str]:
        try:
            return self._resultsroot
        except AttributeError:
            if not self.filename:
                return None
            return os.path.dirname(self.filename)

    @property
    def relfile(self) -> Optional[str]:
        try:
            return self._relfile
        except AttributeError:
            if not self.filename:
                return None
            if not hasattr(self, '_resultsroot'):
                return os.path.basename(self.filename)
            self._relfile = _utils.strict_relpath(
                self.filename,
                self._resultsroot
            )
            return self._relfile

    @property
    def resfile(self) -> Optional["PyperfResultsFile"]:
        try:
            return self._resfile
        except AttributeError:
            if not self.filename:
                return None
            self._resfile = PyperfResultsFile(self.filename, self.resultsroot)
            return self._resfile

    @property
    def date(self) -> Optional[datetime.datetime]:
        try:
            return self._date
        except AttributeError:
            if self.resfile is None:
                return None
            results = self.resfile.read()
            self._date = results.date
            return self._date

    @property
    def baseline(self) -> Optional[str]:
        if not self.compared:
            return None
        if not self.compared.baseline:
            return None
        return self.compared.baseline.source

    @property
    def mean(self) -> Optional[_utils.ElapsedTimeComparison]:
        if not self.compared:
            return None
        if not self.compared.baseline:
            return None
        return self.compared.mean

    @property
    def isbaseline(self) -> bool:
        return self.compared and not self.compared.baseline

    @property
    def sortkey(self) -> Any:
        return (self.uploadid.version, self.date)

    def match(
            self,
            specifier: str,
            suites: Optional[SuitesType] = None,
            *,
            checkexists=False
    ) -> bool:
        # specifier: uploadID, version, filename
        if not specifier:
            return False
        matched = self._match(specifier, suites)
        if matched:
            if checkexists:
                if self.filename and not os.path.isfile(self.filename):
                    return False
        return matched

    def match_uploadid(
            self,
            uploadid: Optional[str],
            *,
            checkexists=False
    ) -> bool:
        if not self.uploadid:
            return False
        if not self.uploadid.match(uploadid):
            return False
        if checkexists:
            if self.filename and not os.path.isfile(self.filename):
                return False
        return True

    def _match(
            self,
            specifier: str,
            suites: Optional[SuitesType]
    ) -> bool:
        if self._match_filename(specifier):
            return True
        if self.uploadid and self.uploadid.match(specifier, suites):
            return True
        return False

    def _match_filename(self, filename: str) -> bool:
        if not self.filename:
            return False
        if not isinstance(filename, str):
            return False
        filename, _, _ = normalize_results_filename(
            filename,
            None if os.path.isabs(filename) else self.resultsroot,
        )
        return filename == self.filename

    def find_comparison(
            self,
            comparisons: PyperfComparisons
    ) -> Optional[PyperfComparison]:
        try:
            return comparisons.bysource[self.filename]
        except KeyError:
            return None

    def load_results(self) -> Optional["PyperfResults"]:
        resfile = self.resfile
        if not resfile:
            return None
        return resfile.read()

    def write_compare_to_markdown(self, baseline: "PyperfResultsInfo") -> str:
        cwd = self._resultsroot
        proc = _utils.run_fg(
            sys.executable,
            '-m', 'pyperf', 'compare_to', '--group-by-speed',
            '--table', '--table-format', 'md',
            baseline.filename,
            self.filename,
            cwd=cwd,
        )
        if proc.returncode:
            raise RuntimeError(proc.stdout)
        md_filename = self.filename[:-5] + '.md'
        with open(md_filename, 'w') as fd:
            fd.write(proc.stdout)
        return md_filename

    def as_rendered_row(self, columns: Iterable[str]) -> List[str]:
        row = []
        for column in columns:
            if column == 'date':
                if self.date is None:
                    rendered = "(unknown)"
                else:
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
                    # TODO: This was self.BASELINE_REF, which isn't defined elsewhere
                    rendered = PyperfComparisonValue.BASELINE
                else:
                    rendered = str(self.mean) if self.mean else ''
            else:
                raise NotImplementedError(column)
            row.append(rendered)
        return row


class PyperfResultsIndex:

    # iter_all()
    # add()
    # ensure_means()

    def __init__(
            self,
            entries: Optional[List[PyperfResultsInfo]] = None
    ):
        if entries is None:
            entries = []
        self._entries = entries

    def __eq__(self, other):
        raise NotImplementedError

    @property
    def baseline(self):
        return self.get_baseline()

    def iter_all(
            self,
            *,
            checkexists: bool = False
    ) -> Iterator[PyperfResultsInfo]:
        if checkexists:
            for info in self._entries:
                if not info.filename or not os.path.isfile(info.filename):
                    continue
                yield info
        else:
            yield from self._entries

    def get_baseline(
            self,
            suite: Optional[SuiteType] = None
    ) -> Optional[PyperfResultsInfo]:
        for entry in self._entries:
            if entry.uploadid.suite != suite:
                continue
            if entry.mean == PyperfComparisonValue.BASELINE:
                return entry
        return None

    def get(
            self,
            uploadid: Any,
            default: Optional[PyperfResultsInfo] = None,
            *,
            checkexists: bool = False
    ) -> Optional[PyperfResultsInfo]:
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

    def match(
            self,
            specifier: Optional[str],
            suites: Optional[SuitesType] = None,
            *,
            checkexists: bool = False
    ) -> Iterator[PyperfResultsInfo]:
        # specifier: uploadID, version, filename
        if not specifier:
            return
        for info in self.iter_all(checkexists=checkexists):
            if not info.match(specifier, suites):
                continue
            yield info

    def match_uploadid(
            self,
            uploadid: Any,
            *,
            checkexists: bool = True
    ) -> Optional[Iterator[PyperfResultsInfo]]:
        requested = PyperfUploadID.from_raw(uploadid)
        if not requested:
            return None
        for info in self.iter_all(checkexists=checkexists):
            if not info.match_uploadid(uploadid):
                continue
            yield info

    def add(self, info: PyperfResultsInfo) -> PyperfResultsInfo:
        if not info:
            raise ValueError('missing info')
        elif not isinstance(info, PyperfResultsInfo):
            raise TypeError(info)
        self._add(info)
        return info

    def _add(self, info: PyperfResultsInfo) -> None:
        #assert info
        # XXX Do not add if already added.
        # XXX Fail if compatid is different but fails are same?
        self._entries.append(info)

    def add_from_results(
            self,
            results: PyperfResults,
            compared: Optional[_PyperfComparison] = None
    ) -> PyperfResultsInfo:
        info = PyperfResultsInfo.from_results(results, compared)
        return self.add(info)

    def get_baseline_by_version(
        self,
        requested_suite: str,
        requested_baseline: str
    ) -> Optional[PyperfResultsInfo]:
        """
        Get a PyperfResultsInfo for the given suite and baseline version.
        """
        baseline = _utils.CPythonVersion.parse(requested_baseline)
        for i, info in enumerate(self._entries):
            suite = info.uploadid.suite
            if suite not in PyperfUploadID.SUITES:
                raise NotImplementedError((suite, info))
            if (
                suite == requested_suite and
                info.uploadid.version.full == baseline.full
            ):
                return info
        return None

    def ensure_means(
            self,
            baseline: Optional[Any] = None
    ) -> List[Tuple[PyperfResultsInfo, PyperfResultsInfo]]:
        requested = _utils.Version.from_raw(baseline).full if baseline else None

        by_suite: Dict[SuiteType, List[Any]] = {}
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
            if baseline.resfile is None:
                raise KeyError
            comparisons = baseline.resfile.compare([i.resfile for i in infos])
            if comparisons is None:
                continue
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

    def as_rendered_rows(
            self,
            columns: Iterable[str]
    ) -> Iterator[Tuple[List[str], PyperfResultsInfo]]:
        for info in self._entries:
            yield info.as_rendered_row(columns), info

    def summarized(self) -> "PyperfResultsIndex":
        """
        Returns a table with only the latest result from a given
        suite/major.minor version.
        """
        entries = sorted(self._entries, key=lambda x: x.sortkey, reverse=True)
        summarized = []
        seen = set()
        for info in entries:
            suite = info.uploadid.suite
            version = _utils.Version.from_raw(info.uploadid.version)
            key = (suite, version.major, version.minor)
            if key not in seen:
                seen.add(key)
                summarized.append(info)
        return PyperfResultsIndex(summarized[::-1])


##################################
# results files

def normalize_results_filename(
        filename: str,
        resultsroot: Optional[str] = None
) -> Tuple[str, str, str]:
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
    def from_raw(cls, raw: Any) -> "PyperfResultsFile":
        if not raw:
            raise ValueError(raw)
        elif isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls(raw)
        else:
            raise TypeError(raw)

    @classmethod
    def from_uploadid(
            cls,
            uploadid: Any,
            resultsroot: Optional[str] = None,
            *,
            compressed: bool = False
    ) -> "PyperfResultsFile":
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
    def _resolve_filename(
            cls,
            filename: str,
            resultsroot: Optional[str],
            compressed: Optional[bool],
    ) -> Tuple[str, str, str]:
        if not filename:
            raise ValueError('missing filename')
        filename = cls._ensure_suffix(filename, compressed)
        return normalize_results_filename(filename, resultsroot)

    @classmethod
    def _ensure_suffix(
            cls,
            filename: str,
            compressed: Optional[bool],
    ) -> str:
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
    def _is_compressed(cls, filename: str) -> bool:
        return filename.endswith(cls.COMPRESSED_SUFFIX)

    def __init__(
            self,
            filename: str,
            resultsroot: Optional[str] = None,
            *,
            compressed: Optional[bool] = None,
    ):
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
    def filename(self) -> str:
        return self._filename

    @property
    def relfile(self) -> str:
        return self._relfile

    @property
    def resultsroot(self) -> Optional[str]:
        return self._resultsroot

    @property
    def uploadid(self) -> Optional[PyperfUploadID]:
        return PyperfUploadID.from_filename(self.filename)

    @property
    def iscompressed(self) -> bool:
        return self._is_compressed(self._filename)

    def read(self) -> PyperfResults:
        _open = self.COMPRESSOR.open if self.iscompressed else open
        with _open(self._filename) as infile:  # type: ignore[operator]
            text = infile.read()
        if not text:
            raise RuntimeError(f'{self.filename} is empty')
        data = json.loads(text)
        return PyperfResults(data, self)

    def write(self, results: PyperfResults) -> None:
        data = results.data
        _open = self.COMPRESSOR.open if self.iscompressed else open
        if self.iscompressed:
            text = json.dumps(data, indent=2)
            with _open(self._filename, 'w') as outfile:  # type: ignore[operator]
                outfile.write(text.encode('utf-8'))
        else:
            with _open(self._filename, 'w') as outfile:  # type: ignore[operator]
                json.dump(data, outfile, indent=2)

    def copy_to(
            self,
            filename: str,
            resultsroot: Optional[str] = None,
            *,
            compressed: Optional[bool] = None,
    ) -> "PyperfResultsFile":
        copied: PyperfResultsFile
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

    def compare(
            self,
            others: Sequence["PyperfResultsFile"]
    ) -> Optional[PyperfComparisons]:
        optional = []
        if len(others) == 1:
            optional.append('--group-by-speed')
        cwd = self._resultsroot
        proc = _utils.run_fg(
            sys.executable, '-m', 'pyperf', 'compare_to',
            *(optional),
            '--table',
            self._relfile,
            *(o._relfile for o in others),
            cwd=cwd,
        )
        if proc.returncode:
            logger.warning(proc.stdout)
            return None
        filenames = [
            self._filename,
            *(o.filename for o in others),
        ]
        return PyperfComparisons.parse_table(proc.stdout, filenames)


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
    def _convert_to_uploadid(
            cls,
            uploadid: Optional[Any] = None
    ) -> Optional[PyperfUploadID]:
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

    def __init__(self, root: str):
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
    def root(self) -> str:
        return self._root

    @property
    def indexfile(self) -> str:
        return self._indexfile

    def _info_from_values(
            self,
            relfile: str,
            uploadid: PyperfUploadID,
            build: Optional[_utils.SequenceOrStrType] = None,
            baseline: Optional[str] = None,
            mean: Optional[str] = None,
            *,
            baselines: Optional[MutableMapping[str, PyperfComparisonBaseline]] = None
    ) -> PyperfResultsInfo:
        assert not os.path.isabs(relfile), relfile
        filename = os.path.join(self._root, relfile)
        if not build:
            build = ['PGO', 'LTO']
#            # XXX Get it from somewhere.
#            raise NotImplementedError
        baseline_obj: PyperfComparisonBaseline
        if baseline:
            assert not os.path.isabs(baseline), baseline
            baseline = os.path.join(self._root, baseline)
            if baselines is not None:
                try:
                    baseline_obj = baselines[baseline]
                except KeyError:
                    baseline_obj = PyperfComparisonBaseline(baseline)
                    baselines[baseline] = baseline_obj
            else:
                baseline_obj = PyperfComparisonBaseline(baseline)
            compared = PyperfComparison(baseline_obj, source=filename, mean=mean)
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

    def _info_from_file(
            self,
            filename: str
    ) -> Tuple[PyperfResultsInfo, PyperfResults]:
        compared = None  # XXX
        return PyperfResultsInfo.from_file(filename, self._root, compared)

    def _info_from_results(self, results):
        raise NotImplementedError  # XXX
        ...

    def _iter_results_files(self):
        raise NotImplementedError

    def iter_from_files(self) -> Iterator[PyperfResultsInfo]:
        for filename in self._iter_results_files():
            info, _ = self._info_from_file(filename)
            yield info

    def iter_all(self) -> Iterator[PyperfResultsInfo]:
        index = self.load_index()
        yield from index.iter_all()

    def index_from_files(self, *, baseline=None) -> PyperfResultsIndex:
        index = PyperfResultsIndex()
        rows = self.iter_from_files()
        for info in sorted(rows, key=(lambda r: r.sortkey)):
            index.add(info)
        if baseline:
            index.ensure_means(baseline=baseline)
        return index

    def load_index(
            self,
            *,
            baseline=None,
            createifmissing: bool = True,
            saveifupdated: bool = True,
    ) -> PyperfResultsIndex:
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

    def _load_index(self) -> PyperfResultsIndex:
        # We use a basic tab-separated values format.
        rows = []
        baselines: MutableMapping[str, PyperfComparisonBaseline] = {}
        for row in self._read_rows():
            parsed = self._parse_row(row)
            info = self._info_from_values(*parsed, baselines=baselines)
            rows.append(info)
        index = PyperfResultsIndex()
        for info in rows:
            index.add(info)
        return index

    def _read_rows(self) -> Iterator[str]:
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
            raise ValueError(headerstr)
        # Now read the rows.
        return rows

    # TYPE_TODO
    def _parse_row(
            self,
            rowstr: str
    ) -> Tuple[
        str,
        PyperfUploadID,
        Optional[str],
        Optional[str],
        Optional[str]
    ]:
        row = rowstr.split('\t')
        if len(row) != len(self.INDEX_FIELDS):
            raise ValueError(rowstr)
        relfile, uploadid_str, build, baseline, mean = row
        uploadid = PyperfUploadID.parse(uploadid_str)
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
        return (
            relfile,
            uploadid,
            build or None,
            baseline or None,
            mean or None
        )

    def save_index(self, index: PyperfResultsIndex) -> None:
        # We use a basic tab-separated values format.
        rows = [self.INDEX_FIELDS]
        for info in sorted(index.iter_all(), key=(lambda v: v.sortkey)):
            row = self._render_as_row(info)
            rows.append(
                [(row[f] or '') for f in self.INDEX_FIELDS]
            )
        text = os.linesep.join('\t'.join(row) for row in rows)
        with open(self._indexfile, 'w', encoding='utf-8') as outfile:
            outfile.write(text)
            print(file=outfile)  # Add a blank line at the end.

    def _render_as_row(self, info: PyperfResultsInfo) -> Dict[str, Any]:
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

    def get(
            self,
            uploadid: Any,
            default: Optional[PyperfResultsInfo] = None,
            *,
            checkexists: bool = True
    ) -> Optional[PyperfResultsInfo]:
        index = self.load_index()
        return index.get(uploadid, default, checkexists=checkexists)

    def match(
            self,
            specifier: str,
            suites: Optional[SuitesType] = None,
            *,
            checkexists: bool = True
    ) -> Iterator[PyperfResultsInfo]:
        index = self.load_index()
        yield from index.match(specifier, suites, checkexists=checkexists)

    def match_uploadid(
            self,
            uploadid: Any,
            *,
            checkexists: bool = True
    ) -> Iterator[PyperfResultsInfo]:
        index = self.load_index()
        results = index.match_uploadid(uploadid, checkexists=checkexists)
        if results is not None:
            yield from results

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

    def add_from_results(
            self,
            results: PyperfResults,
            *,
            baseline: Optional[str] = None,
            compressed: bool = False,
            split: bool = False,
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
                suite_results.copy_to(resfile, self._root)  # type: ignore[arg-type]
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

    def _iter_results_files(self) -> Iterator[str]:
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

    def match(self, specifier: str):
        raise NotImplementedError

    def add(
            self,
            results: PyperfResults,
            *,
            compressed: bool = False,
            split: bool = False
    ):
        raise NotImplementedError


class PyperfResultsRepo(PyperfResultsStorage):

    BRANCH = 'add-benchmark-results'

    @classmethod
    def from_remote(
            cls,
            remote: Optional[Union[str, _utils.GitHubTarget]],
            root: Optional[str],
            datadir: Optional[str] = None,
            baseline: Optional[str] = None
    ) -> "PyperfResultsRepo":
        if not root or not _utils.check_str(root):
            root = None
        elif not os.path.isabs(root):
            raise ValueError(root)
        if isinstance(remote, str):
            remote_resolved = _utils.GitHubTarget.resolve(remote, root)
        elif not isinstance(remote, _utils.GitHubTarget):
            raise TypeError(f'unsupported remote {remote!r}')
        else:
            remote_resolved = remote
        raw = remote_resolved.ensure_local(root)
#        raw.clean()
#        raw.switch_branch('main')
        kwargs = {}
        if datadir:
            kwargs['datadir'] = datadir
        if baseline:
            kwargs['baseline'] = baseline
        return cls(raw, remote_resolved, **kwargs)

    @classmethod
    def from_root(
            cls,
            root: Optional[str],
            datadir: Optional[str] = None,
            baseline: Optional[str] = None
    ) -> "PyperfResultsRepo":
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

    def __init__(
            self,
            raw: Any,
            remote: Optional[_utils.GitHubTarget] = None,
            datadir: Optional[str] = None,
            baseline: Optional[str] = None
    ):
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
    def root(self) -> str:
        return self._raw.root

    def iter_all(self) -> Iterator[PyperfUploadID]:
        for info in self._resultsdir.iter_from_files():
            yield info.uploadid
        #yield from self._resultsdir.iter_from_files()
        #yield from self._resultsdir.iter_all()

    def get(
            self,
            uploadid: PyperfUploadID,
            default=None
    ) -> Optional[PyperfResultsFile]:
        info = self._resultsdir.get(uploadid, default)
        return info.resfile if info else None
        #return self._resultsdir.get(uploadid)

    def match(
            self,
            specifier: str,
            suites: Optional[SuitesType] = None
    ) -> Iterator[PyperfResultsFile]:
        for info in self._resultsdir.match(specifier, suites):
            if info.resfile:
                yield info.resfile
        #yield from self._resultsdir.match(specifier, suites)

    def add(
            self,
            results: PyperfResults,
            *,
            branch: Optional[str] = None,
            author: Optional[str] = None,
            compressed: bool = False,
            split: bool = True,
            clean: bool = True,
            push: bool = True,
    ) -> None:
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
        if self.datadir:
            readmes = [
                self._update_table(index.summarized(), "README.md"),
                self._update_table(index, os.path.join(self.datadir, "README.md"))
            ]
        else:
            readmes = [
                self._update_table(index, "README.md"),
            ]

        logger.info('committing to the repo...')
        for info in added:
            repo.add(info.filename)
            if self._baseline:
                baseline = index.get_baseline_by_version(info.uploadid.suite, self._baseline)
                assert baseline is not None
                repo.add(info.write_compare_to_markdown(baseline))
        repo.add(self._resultsdir.indexfile)
        for readme in readmes:
            repo.add(readme)
        msg = f'Add Benchmark Results ({info.uploadid.copy(suite=None)})'
        repo.commit(msg)
        logger.info('...done committing')

        if push:
            self._upload(self.datadir or '.')

    def _update_table(
            self,
            index: PyperfResultsIndex,
            filename: str
    ) -> str:
        table_lines = self._render_markdown(index, filename)
        MARKDOWN_START = '<!-- START results table -->'
        MARKDOWN_END = '<!-- END results table -->'
        filename = self._raw.resolve(filename)
        logger.debug('# writing results table to %s', filename)
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

    def _render_markdown(
        self,
        index: PyperfResultsIndex,
        output_filename: str
    ) -> Iterator[str]:
        def render_row(row):
            row = (f' {v} ' for v in row)
            return f'| {"|".join(row)} |'
        columns = 'date release commit host mean'.split()

        output_dir = os.path.dirname(os.path.join(self._raw.root, output_filename))

        rows = index.as_rendered_rows(columns)
        by_suite: Dict[SuiteType, List[List[str]]] = {}
        for row, info in sorted(rows, key=(lambda r: r[1].sortkey)):
            suite = info.uploadid.suite
            if suite not in by_suite:
                by_suite[suite] = []
            date, release, commit, host, mean = row
            relpath = os.path.relpath(info.filename, output_dir)
            relpath = relpath.replace(r'\/', '/')
            table_relpath = relpath[:-5] + ".md"
            date = f'[{date}]({relpath})'
            if not mean:
                mean = PyperfComparisonValue.BASELINE
            assert '3.10.4' not in release or mean == '(ref)', repr(mean)
            mean = f'[{mean}]({table_relpath})'
            row = [date, release, commit, host, mean]
            by_suite[suite].append(row)

        for suite2, rows2 in sorted(by_suite.items()):
            hidden = not Benchmarks.SUITES[suite].show_results
            yield ''
            if hidden:
                yield '<!--'
            yield f'{suite2 or "???"}:'
            yield ''
            yield render_row(columns)
            yield render_row(['---'] * len(columns))
            for row in rows2:
                yield render_row(row)
            if hidden:
                yield '-->'
        yield ''

    def _upload(self, reltarget: str) -> None:
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
    def from_remote(  # type: ignore[override]
            cls,
            remote: Optional[Union[str, _utils.GitHubTarget]] = None,
            root: Optional[str] = None,
            baseline: Optional[str] = None
    ) -> "PyperfResultsRepo":
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
