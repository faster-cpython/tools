import gzip
import hashlib
import json
import os
import os.path
import platform
import sys

from . import _utils


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


class PyperfResults:

    @classmethod
    def from_file(cls, filename, host=None, source=None):
        if not filename:
            raise ValueError('missing filename')
        data = cls._read(filename)
        self = cls(data, filename, host, source)
        return self

    @classmethod
    def _read(cls, filename):
        if filename.endswith('.json.gz'):
            _open = gzip.open
        else:
            raise NotImplementedError(filename)
        with _open(filename) as infile:
            return json.load(infile)

    @classmethod
    def _get_os_name(cls, metadata=None):
        if metadata:
            platform = metadata['platform'].lower()
            if 'linux' in platform:
                return 'linux'
            elif 'darwin' in platform or 'macos' in platform or 'osx' in platform:
                return 'mac'
            elif 'win' in platform:
                return 'windows'
            else:
                raise NotImplementedError(platform)
        else:
            if sys.platform == 'win32':
                return 'windows'
            elif sys.paltform == 'linux':
                return 'linux'
            elif sys.platform == 'darwin':
                return 'mac'
            else:
                raise NotImplementedError(sys.platform)

    @classmethod
    def _get_arch(cls, metadata=None):
        if metadata:
            platform = metadata['platform'].lower()
            if 'x86_64' in platform:
                return 'x86_64'
            elif 'amd64' in platform:
                return 'amd64'

            procinfo = metadata['cpu_model_name'].lower()
            if 'aarch64' in procinfo:
                return 'arm64'
            elif 'arm' in procinfo:
                if '64' in procinfo:
                    return 'arm64'
                else:
                    return 'arm32'
            elif 'intel' in procinfo:
                return 'x86_64'
            else:
                raise NotImplementedError((platform, procinfo))
        else:
            uname = _platform.uname()
            machine = uname.machine.lower()
            if machine in ('amd64', 'x86_64'):
                return machine
            elif machine == 'aarch64':
                return 'arm64'
            elif 'arm' in machine:
                return 'arm'
            else:
                raise NotImplementedError(machine)

    def __init__(self, data, filename=None, host=None, source=None, suite=None):
        if data:
            self._set_data(data)
        elif not filename:
            raise ValueError('missing filename')
        self.filename = filename or None
        if host:
            self._host = host
        self.source = source
        self.suite = suite or None

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
        return self.data['metadata']

    @property
    def host(self):
        try:
            return self._host
        except AttributeError:
            metadata = self.metadata
            # We could use metadata['hostname'] but that doesn't
            # make a great label in the default case.
            host = self._get_os_name(metadata)
            arch = self._get_arch(metadata)
            if arch in ('arm32', 'arm64'):
                host += '-arm'
            # Ignore everything else.
            return host
            return self._host

    @property
    def implementation(self):
        return 'cpython'

    @property
    def compat_id(self):
        return self._get_compat_id()

    def _get_compat_id(self, *, short=True):
        metadata = self.metadata
        data = [
            metadata['hostname'],
            metadata['platform'],
            metadata.get('perf_version'),
            metadata['performance_version'],
            metadata['cpu_model_name'],
            metadata.get('cpu_freq'),
            metadata['cpu_config'],
            metadata.get('cpu_affinity'),
        ]

        h = hashlib.sha256()
        for value in data:
            if not value:
                continue
            h.update(value.encode('utf-8'))
        compat = h.hexdigest()
        if short:
            compat = compat[:12]
        return compat

    def get_upload_name(self):
        # See https://github.com/faster-cpython/ideas/tree/main/benchmark-results/README.md
        # for details on this filename format.
        metadata = self.metadata
        commit = metadata.get('commit_id')
        if not commit:
            raise NotImplementedError
        compat = self._get_compat_id()
        source = self.source or 'main'
        implname = self.implementation
        host = self.host
        name = f'{implname}-{source}-{commit[:10]}-{host}-{compat}'
        if self.suite and self.suite != 'pyperformance':
            name = f'{name}-{self.suite}'
        return name

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
            results = cls(None, self.filename, host, self.source, suite)
            results._data = data
            by_suite[suite] = results
        return by_suite
