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

    MAIN_VERSION = '3.12a0'

    REGEX = re.compile(r'''
        ^
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
        ( \.json (?: \.gz)? )?  # <suffix>
        $
    ''', re.VERBOSE)

    @classmethod
    def parse(cls, name):
        m = cls.REGEX.match(name)
        if not m:
            return None
        impl, verstr, commit, host, compatid, suite, suffix = m.groups()
        impl = _utils.resolve_python_implementation(impl)
        if verstr == 'main':
            verstr = cls.MAIN_VERSION
            name = name.replace('-main-', f'-{verstr}-')
        version = impl.parse_version(verstr)
        self = cls(impl, version, commit, host, compatid, suite)
        self._name = name
        self._suffix = suffix
        return self

    @classmethod
    def from_filename(cls, filename):
        filename = os.path.abspath(filename)
        _, basename = os.path.split(filename)
        self = cls.parse(basename)
        if self is None:
            return None
        self._filename = filename
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
        self = cls(
            impl=impl,
            version=impl.parse_version(version, requirestr=False),
            commit=metadata.commit,
            host=host or metadata.host,
            compatid=metadata.compatid,
            suite=suite,
        )
        return self

    def __str__(self):
        return self.name

    @property
    def implementation(self):
        return self.impl

    @property
    def filename(self):
        try:
            return self._filename
        except AttributeError:
            suffix = getattr(self, '_suffix', None) or ''
            return f'{self.name}{suffix}'

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            impl, version, commit, host, compatid, suite = self
            name = f'{impl}-{version}-{commit[:10]}-{host}-{compatid}'
            if suite and suite != 'pyperformance':
                name = f'{name}-{suite}'
            self._name = name
            return name


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

    def __init__(self, filename):
        if not filename:
            raise ValueError('missing filename')
        self._filename = os.path.abspath(filename)

    def __str__(self):
        return self._filename

    @property
    def filename(self):
        return self._filename

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
        return PyperfResults(data, filename, version, host)

    def compare(self, others):
        argv = [
            sys.executable, '-m', 'pyperf', 'compare_to',
            '--group-by-speed',
            '--table',
            os.path.relpath(self._filename),
            *(os.path.relpath(o.filename)
              for o in others),
        ]
        proc = subprocess.run(argv, stdout=subprocess.PIPE, encoding='utf-8')
        if proc.returncode:
            return None
        return PyperfTable.parse(proc.stdout)


class PyperfResults:

    def __init__(self, data, resfile=None, version=None, host=None, suite=None):
        if data:
            self._set_data(data)
        elif not resfile:
            raise ValueError('missing resfile')
        self.resfile = PyperfResultsFile.from_raw(resfile) if resfile else None
        self.version = version
        if host:
            self._host = host
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
                version = self.version,
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
            results = cls(None, self.filename, host, suite)
            results._data = data
            by_suite[suite] = results
        return by_suite

    #def compare(self, others):
    #    raise NotImplementedError


class PyperfResultsMetadata:

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
        return cls(metadata, data['version'])

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

    def add(self, results, *, unzipped=True):
        raise NotImplementedError


class PyperfResultsRepo(PyperfResultsStorage):

    BRANCH = 'add-benchmark-results'

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

    def git(self, *args, cfg=None):
        ec, text = _utils.git(*args, cwd=self.root, cfg=cfg)
        if ec:
            raise NotImplementedError((ec, text))
        return text

    def iter_all(self):
        ...

    def add(self, results, *,
            branch=None,
            author=None,
            unzipped=True,
            split=True,
            push=True,
            ):
        if not branch:
            branch = self.BRANCH

        if not isinstance(results, PyperfResults):
            raise NotImplementedError(results)
        source = results.filename
        if source and not os.path.exists(source):
            logger.error(f'results not found at {source}')
            return
        if split:
            by_suite = results.split_benchmarks()
            if 'other' in by_suite:
                raise NotImplementedError(sorted(by_suite))
        else:
            by_suite = {None: results}

        if self.remote:
            self.remote.ensure_local(self.root)

        authorargs = ()
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
            #authorargs = ('--author', author)
        else:
            raise NotImplementedError(author)

        logger.info(f'adding results {source or "???"}...')
        for suite in by_suite:
            suite_results = by_suite[suite]
            name = suite_results.uploadid
            reltarget = f'{name}.json'
            if self.datadir:
                reltarget = f'{self.datadir}/{reltarget}'
            if not unzipped:
                reltarget += '.gz'
            target = os.path.join(self.root, reltarget)

            logger.info(f'...as {target}...')
            self.git('checkout', '-B', branch)
            if unzipped:
                data = suite_results.data
                with open(target, 'w') as outfile:
                    json.dump(data, outfile, indent=2)
            elif not source:
                data = suite_results.data
                with gzip.open(target, 'w') as outfile:
                    json.dump(data, outfile, indent=2)
            else:
                shutil.copyfile(source, target)
            self.git('add', reltarget)
            msg = f'Add Benchmark Results ({name})'
            self.git('commit', *authorargs, '-m', msg, cfg=cfg)
        logger.info('...done adding')

        if push:
            self._upload(reltarget)

    def _upload(self, reltarget):
        if not self.remote:
            raise Exception('missing remote')
        url = f'{self.remote.url}/tree/main/{reltarget}'
        logger.info(f'uploading results to {url}...')
        self.git('push', self.remote.push_url)
        logger.info('...done uploading')


class FasterCPythonResults(PyperfResultsRepo):

    REMOTE = _utils.GitHubTarget.from_origin('faster-cpython', 'ideas', ssh=True)
    DATADIR = 'benchmark-results'

    def __init__(self, root=None, remote=None):
        if not remote:
            remote = self.REMOTE
        super().__init__(root, remote, self.DATADIR)
