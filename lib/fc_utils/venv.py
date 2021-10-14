import logging
import os
import os.path
import subprocess
import sys

from . import (
    fs as _fs,
    packaging as _packaging,
    misc as _misc,
)


logger = logging.getLogger(__name__)


def prep_venv(name, *,
              venvsdir='.',
              on_exists='skip',
              python=None,
              failfast=True,
              ):
    if not callable(on_exists):
        on_exists = resolve_on_exists(on_exists)
    return _prep_venv(name, venvsdir, on_exists, python, failfast)


def create_venv(name, *,
                venvsdir='.',
                on_exists='skip',
                python=None,
                failfast=True,
                dryrun=False,
                ):
    venvroot, run, needreqs = prep_venv(
        name,
        venvsdir=venvsdir,
        on_exists=on_exists,
        python=python,
        failfast=failfast,
    )
    run(dryrun=dryrun)
    return venvroot, needreqs


def get_venv_factory(venvsdir='.', on_exists='skip', *,
                     python=None,
                     failfast=True,
                     ):
    if not callable(on_exists):
        on_exists = resolve_on_exists(on_exists)

    if not venvsdir:
        venvsdir = '.'
    venvsdir = os.path.abspath(venvsdir)
    if not os.path.exists(venvsdir):
        os.mkdirs(venvsdir)

    def prep_venv(name):
        return _prep_venv(name, venvsdir, on_exists, python, failfast)
    return prep


def _prep_venv(name, venvsdir, on_exists, python, failfast):
    venvroot = os.path.abspath(os.path.join(venvsdir, name))
    status, kind = get_venv_status(rootdir)
    if status != 'missing':
        try:
            op, venvroot = on_exists(venvroot, status, kind)
        except Exception as exc:
            op = exc
    else:
        op = 'create'

    # XXX below is a bit of a mess
    raise NotImplementedError

    run = None
    needreqs = False
    if isinstance(op, Exception):
        if failfast:
            raise op
        run = op
    elif op == 'create':
        def run(warn=False, dryrun=False, log=logger.info):
            if log is not None:
                log(f'creating venv {venvroot}')
            if not dryrun:
                _create_venv(venvroot, python)
        needreqs = True
    elif op == 'replace':
        if status == 'valid':
            def delete(warn=False, dryrun=False, log=logger.info):
                if log is not None:
                    log(f'replacing venv {venvroot}')
                if not dryrun:
                    shutil.rmtree(venvroot)
            def create(warn=False, dryrun=False, log=logger.info):
                if not dryrun:
                    _create_venv(venvinfo, python)
            run = [delete, create]
        elif status == 'invalid':
            def delete(warn=False, dryrun=False, log=logger.info):
                if log is not None:
                    log(f'replacing non-venv {venvroot} dir with venv')
                if not dryrun:
                    shutil.rmtree(venvroot)
            def create(warn=False, dryrun=False, log=logger.info):
                if not dryrun:
                    _create_venv(venvinfo, python)
            run = [delete, create]
        else:
            def delete(warn=False, dryrun=False, log=logger.info):
                if log is not None:
                    log(f'replacing {venvroot} file with venv dir')
                if not dryrun:
                    os.unlink(venvroot)
            def create(warn=False, dryrun=False, log=logger.info):
                if not dryrun:
                    _create_venv(venvinfo, python)
            run = [delete, create]
        needreqs = True
    elif op:
        raise NotImplementedError(op)
    elif status == 'valid':
        def run(warn=False, dryrun=False, log=logger.info):
            if warn:
                log('venv already exists:')
                log(f' {venvroot}')
                log('(skipping)')
        needreqs = True
    elif status == 'invalid':
        def run(warn=False, dryrun=False, log=logger.info):
            log('WARNING: an incomplete venv dir already exists:')
            log(f' {venvroot}')
            log('(skipping)')
    elif status == 'not-dir':
        def run(warn=False, dryrun=False, log=logger.info):
            log('WARNING: a conflicting file already exists:')
            log(f' {venvroot}')
            log('(skipping)')
    elif status != 'missing':
        raise NotImplementedError(status)

    return venvroot, run, needreqs


def _create_venv(rootdir, python=None):
    if not python:
        python = sys.executable
    subprocess.run(
        [python, '-m', 'venv', rootdir],
        check=True,
    )


def get_venv_status(rootdir):
    fskind = _fs.check_file(rootdir)
    if not fskind:
        return 'missing', None
    elif fskind not in ('dir', 'dir symlink'):
        return 'not-dir', fskind
    elif not resolve_venv_file(rootdir, 'bin', 'python', checkexists='exe'):
        return 'invalid', fskind
    else:
        return 'valid', fskind


def resolve_on_exists(action):
        return _misc.resolve_on_exists(
            on_exists,
            'venv',
            custom_status={'not-dir': "it isn't a dir"},
        )


def resolve_venv_file(rootdir, kind, name, *,
                      checkexists=False,
                      applysuffix=True,
                      ):
    if not kind:
        if not name:
            return rootdir
        dirname = name
        name = None

    if kind == 'bin':
        if os.name == "nt":
            dirname = 'Scripts'
            suffix = '.exe'
        else:
            dirname = 'bin'
            suffix = ''
    else:
        dirname = kind
        suffix = ''

    if name:
        basename = (name + suffix) if suffix and applysuffix else name
        resolved = os.path.join(rootdir, dirname, basename)
    else:
        resolved =  os.path.join(rootdir, dirname)

    if checkexists and not _fs.check_file(resolved, expected=checkexists):
        return None

    return resolved


def ensure_requirements(venv, reqs, *, dryrun=False):
    if os.path.isdir(venv):
        python = resolve_venv_file(venv, 'bin', 'python')
    else:
        python = venv
    _packaging.ensure_requirements(reqs, python, dryrun=dryrun)


def get_depdencies(venv, reqs):
    if os.path.isdir(venv):
        python = resolve_venv_file(venv, 'bin', 'python')
    else:
        python = venv
    return _packaging.get_dependencies(reqs, python)


def show_depdencies(venv, reqs, *, dryrun=False):
    if os.path.isdir(venv):
        python = resolve_venv_file(venv, 'bin', 'python')
    else:
        python = venv
    _packaging.show_dependencies(reqs, python, dryrun=dryrun)
