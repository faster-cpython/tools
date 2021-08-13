import importlib.util
import os.path
import sys
import types

from .runtime import resolve_context_name
from .text import format_item


class NamedSentinel:
    __slots__ = ['__name__']

    def __init__(self, name, *, resolve=False):
        if not name:
            raise ValueError('missing name')
        elif not name.isidentifier():
            raise ValueError(f'bad name {name!r}')
        if resolve:
            parent = resolve_context_name()
            name = f'{parent}.{name}'
        object.__setattr__(self, '__name__', name)

    def __repr__(self):
        return f'{type(self).__name__}({self.__name__})'

    def __str__(self):
        return self.__name__

    def __setattr__(self, name, value):
        raise AttributeError('readonly')

    def __delattr__(self, name):
        raise AttributeError('readonly')


_FAIL = NamedSentinel('FAIL')
_IGNORE = NamedSentinel('IGNORE')


##################################
# finding objects

def is_valid_qualname(value):
    if not value or not isinstance(value, str):
        return False
    return all(p.isidentifier() for p in value.split('.'))


def resolve_qualname(name, location):
    filename = locate_module(location)
    if not filename:
        filename = location

    try:
        srcfile = open(filename)
    except FileNotFoundError:
        return None
    # XXX Walk through closures?  Use the parser?
    prefix = f'def {name}('
    with srcfile:
        for line in srcfile:
            #if line.strip().startswith(prefix):
            if line.startswith(prefix):
                return name
    return None


def parse_qualname(value):
    if not value or not isinstance(value, str):
        return None, None, None
    location, sep, qualname = value.rpartition(':')
    if not is_valid_qualname(qualname):
        return None, None, None
    if not sep:
        return qualname, None, None
    if _looks_like_module_name(location):
        return qualname, location, None
    return qualname, None, location


def locate_module(module):
    if not _looks_like_module_name(module):
        return None
    try:
        spec = importlib.util.find_spec(module)
    except Exception:
        return None
    else:
        return spec.origin


def resolve_module(filename):
    if not filename.endswith('.py'):
        return '__main__'
    relpath, _, _ = filename.rpartition('.')
    if os.path.basename(filepath) == '__init__.py':
        relpath = os.path.dirname(relpath)

    for entry in sys.path:
        if relpath.startswith(f'{entry}/'):
            break
    else:
        if os.path.exists(filename):
            return '__main__'
        return None

    module = relpath.replace('/', '.')
    if not all(p.isidentifier() for p in module.split('.')):
        return '__main__'
    return module


def _looks_like_module_name(value):
    if not is_valid_qualname(value):
        return False
    # We trust there won't ever be a submodule name "py".
    return not value.endswith('.py')


##################################
# data about Python objects

def format_object_name(obj):
    try:
        qualname = obj.__qualname__
    except AttributeError:
        return repr(obj)
    return f'{type(obj).__qualname__} {qualname!r}'


#def render_list(items):
#    if not indent:
#        yield format_item(name, value[0], namewidth)
#        for line in value[1:]:
#            yield inline_indent + line
#        continue
#
#    yield ''
#    yield f'{name}:'
#    for line in value:
#        yield indent + line


def render_attrs(obj, names=None, *,
                 resolve_attr=getattr,
                 onmissing=_FAIL,
                 render_value=None,
                 namewidth=None,
                 ):
    if resolve_attr is None:
        resolve_attr = getattr
    elif not callable(resolve_attr):
        raise TypeError(f'expected resolve_attr to be callable, got {resolve_attr}')
    if render_value and not callable(render_value):
        raise TypeError(f'expected render_value to be callable, got {render_value}')

    if not namewidth:
        namewidth = 15

    if not names:
        names = (n for n in dir(obj) if not n.startswith('_'))
    elif names is obj:
        try:
            names = list(vars(obj))
        except TypeError:
            names = dir(obj)
        names = (n for n in names if not n.startswith('_'))

    inline_indent = ' ' * (namewidth + 1)
    needsep = False
    for name in names:
        try:
            value = resolve_attr(obj, name)
        except AttributeError:
            if onmissing is _FAIL:
                raise  # re-raise
            elif onmissing is _IGNORE:
                continue
            else:
                value = onmissing

        if render_value is not None:
            value, indent = render_value(obj, name, value)
            value = normalize_rendered(value)
            if not isinstance(value, str):
                if not indent:
                    yield format_item(name, value[0], namewidth)
                    for line in value[1:]:
                        yield inline_indent + line
                    continue

                yield ''
                yield f'{name}:'
                for line in value:
                    yield indent + line
                needsep = True
                continue
        if needsep:
            yield ''
            needsep = False
        value = normalize_rendered(value)
        yield format_item(name, value, namewidth)


def normalize_rendered(value):
    if isinstance(value, str):
        if not value:
            return repr(value)
        return value

    try:
        lines = list(value)
    except TypeError:
        return str(value)
    if not lines:
        return str(None)
    return lines
