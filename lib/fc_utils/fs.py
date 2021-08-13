import os
import stat as _stat


def check_file(filename, *, expected=None):
    """Find the file and report its kind.

    The possible kinds are "dir", "file", and "other".  If it is a
    symlink then it will report the actual kind with " symlink" after.
    If the file is not found then None is returned.
    
    If "expected" is provided then it is used to decide if the file
    matches.  It can be a string/set/list of kinds or a callable.
    
    If a callable then "expected(filename, kinds)" is called.  If it
    returns False then None is returned here.

    Otherwise the detected kind must match one of the given kinds.
    It will also interpret "exe" to mean "it must be an executable".
    """
    kind, st = _get_file_kind(filename)
    if not kind:
        return None
    kinds = kind.split()

    if callable(expected):
        if not expected(filename, kinds):
            return None
    elif expected and expected is not True:
        if isinstance(expected, str):
            expected = expected.replace(',', ' ').split()
        expected = set(expected)

        # XXX Handle exclusion (prefix a kind with "-") too?

        if 'exe' in expected:
            expected.remove('exe')
            if not is_executable(filename, st):
                return None

        if 'symlink' in expected:
            expected.remove('symlink')
            if 'symlink' not in kinds:
                return None

        if expected and not any(k in kinds for k in expected):
            return None

    return kind


def get_file_kind(filename):
    kind, _ = _get_file_kind(filename)
    return kind


def _get_file_kind(filename):
    try:
        st = os.lstat(filename)
    except FileNotFoundError:
        return None

    kinds = []

    if _stat.S_ISDIR(st.st_mode):
        kinds.append('dir')
    elif _stat.S_ISREG(st.st_mode):
        kinds.append('file')
    else:
        kinds.append('other')

    if _stat.S_ISLNK(st.st_mode):
        kinds.append('symlink')

    return ' '.join(kinds), st


def is_executable(filename, st=None): 
    if not st:
        st = os.lstat(filename)
    # XXX This does not work on Windows.
    if os.name == 'nt':
        raise NotImplementedError(filename)
    return st.st_mode & _stat.S_IXUSR
