from collections import namedtuple
import types


class classonly:
    """Like classmethod but does not show up on instances."""

    def __init__(self, wrapped, *, _functype=type(lambda:'')):
        if isinstance(wrapped, _functype):
            wrapped = classmethod(wrapped)
        self.wrapped = wrapped
        if hasattr(wrapped, '__get__'):
            self._getter = wrapped.__get__
        else:
            self._getter = None

    def __repr__(self):
        return f'{type(self).__name__}(wrapped={self.wrapped!r})'

    def __get__(self, obj, cls):
        if obj is not None:
            raise AttributeError
        if self._getter is None:
            return self.wrapped
        return self._getter(None, cls)


##################################
# readonly

def is_readonly(cls):
    if not isinstance(cls, type):
        raise TypeError(f'cls must be a type, got {cls!r}')

    res = _is_readonly(cls)
    if res is None:
        return False
    return res


def _is_readonly(cls):
    if cls.__setattr__ is Readonly.__setattr__:
        if cls.__delattr__ is Readonly.__delattr__:
            return True
    return False


#def readonly(cls, *, wrap=None):
#    if not isinstance(cls, type):
#        raise TypeError(f'cls must be a type, got {cls!r}')
#
#    okay = _is_readonly(cls)
#    if okay:
#        return cls
#
#    if wrap:
#        return Readonly._wrap(cls)
#    elif inject
#    elif wrap is None:
#        Readonly._inject(cls, fail=okay ) is not None:
#            return cls
#
#        inject(target, cls,
#               names=['__setattr__', '__delattr__'],
#               exclude
#               )
#
#
#
#    if okay is None:
#        cls.__setattr__ = Readonly.__setattr__
#        cls.__delattr__ = Readonly.__delattr__
#        #cls.__setattr_raw = Readonly.__setattr_raw
#        #cls.__delattr_raw = Readonly.__delattr_raw
#        return cls
#    if wrap is not None:


class Readonly:
    # XXX Use a metaclass for the sake of __subclasscheck__() and
    # __instancecheck__()?

    @classonly
    def __raw_setattr(cls, self, name, value):
        object.__setattr__(self, name, value)

    @classonly
    def __raw_delattr(cls, self, name):
        object.__delattr__(self, name)

    def __setattr__(self, name, value):
        raise TypeError(f'{type(self).__name__} is readonly')

    def __delattr__(self, name):
        raise TypeError(f'{type(self).__name__} is readonly')


##################################
# namespace types

def as_namedtuple(fields):
    NTBase = namedtuple('NTBase', fields)
    def decorator(cls):
        if cls.__base__ is not object:
            raise NotImplementedError
        if type(cls) is not type:
            raise NotImplementedError

        # If any decorators with side effects (e.g. registration) were
        # already applied then the following is problematic.
        class NT(cls, NTBase):
            __slots__ = ()
        NT.__name__ = cls.__name__
        # XXX Ensure that they are in the same file/scope?
        NT.__module__ = cls.__module__
        NT.__qualname__ = cls.__qualname__
        NT.__doc__ = cls.__doc__
        return NT
    return decorator


#def as_namespace(names, **defaults):
#    names = _parse_names(names)
#
#    ns = {}
#    exec(textwrap.dedent('''
#        def __init__(self, {
#    '''), ns, ns)
#
#
#    def decorator(cls):
#        sub = type(cls.__name__, (cls, Namespace), ns)
#
#        return sub
#    return decorator


class Namespace(types.SimpleNamespace):

    def __init__(self):
        super().__init__()
#class Namespace:
#    __slots__ = ()
#
#
#class OpenNamespace(Namespace, types.SimpleNamespace):
#    ...


#def _parse_names(names):
#    if isinstance(names, str):
#        names = tuple(names.replace(',', ' ').split())
#    else:
#        names = tuple(names)
#        if any(not isinstance(n, str) for n in names):
#            raise ValueError(f'expected str names, got {names}')
#        if any(not n for n in names):
#            raise ValueError(f'expected non-empty names, got {names}')
#    # XXX Check keywords too?
#    if any(not n.isidentifier() for n in names):
#        raise ValueError(f'names must be identifiers, got {names}')
#    if len(set(names)) != len(names):
#        raise ValueError(f'duplicate names in {names}')
#    return names


##################################
# other utils

def walk_attrs(cls):
    if not isinstance(cls, type):
        raise TypeError(f'cls must be a type, got {cls!r}')

    for cls in cls.__mro__:
        for meta in type(cls).__mro__:
            for name, value in vars(meta).items():
                if hasattr(cls, name):
                    yield cls, meta, name, value
            for name in dir(type(cls)):
                if hasattr(cls, name):
                    yield cls, meta, name, value

        for name, value in vars(cls).items():
            yield cls, None, name, value
        for name in dir(cls):
            try:
                value = getattr(cls, name)
            except AttributeError:
                continue
            yield cls, None, name, value


def walk_resolved_attrs(cls):
    seen = set()
    for _, _, name, _ in walk_attrs(cls):
        if name in seen:
            continue
        seen.add(name)
        try:
            value = getattr(cls, name)
        except AttributeError:
            continue
        yield name, value


def inject(cls, base, names=None, *, exclude=None, fail=True):
    if not isinstance(cls, type):
        raise TypeError(f'cls must be a type, got {cls!r}')

    if isinstance(base, type):
        # XXX Don't try inject_base() if any excluded names match.

        if _inject_base(cls, base, fail=False):
            return cls
        # Fall back to injecting attrs.

    ns = _resolve_base_namespace(base, names, exclude)
    if not _check_matching_attrs(cls, ns, base, fail):
        return None
    vars(cls).update(ns)
    return cls


def inject_base(cls, base, *, fail=True):
    if not isinstance(cls, type):
        raise TypeError(f'cls must be a type, got {cls!r}')
    if not isinstance(base, type):
        raise TypeError(f'base must be a type, got {cls!r}')
    return cls if _inject_base(cls, base, fail) else None


def _inject_base(cls, base, fail):
    if base in cls.__mro__:
        return True

    try:
        cls.__bases__ = (base, *cls.__bases__)
    except TypeError:
        if fail:
            raise  # re-reaise
        return False
    return True


def inject_attrs(cls, base, names=None, *, exclude=None, fail=True):
    if not isinstance(cls, type):
        raise TypeError(f'cls must be a type, got {cls!r}')

    ns = _resolve_base_namespace(base, names, exclude)
    if not _check_matching_attrs(cls, ns, base, fail):
        return None
    vars(cls).update(ns)
    return cls


def _resolve_name_checker(names):
    if callable(names):
        raise NotImplementedError(names)
    if not names:
        return None
    explicit = set()
    dunder = False
    public = False
    # XXX methods?
    # XXX non-methods?
    for name in names:
        if name.isidentifier():
            explicit.add(name)
        elif name == '<dunder>':
            dunder = True
        elif name == '<public>':
            public = True
        else:
            raise NotImplementedError(name)

    def check(name):
        if name in explicit:
            return True
        elif dunder and name.startwith('__') and name.endswith('__'):
            # XXX Support only the language-supported names.
            return True
        elif public and not name.startwith('_'):
            return True
        else:
            return False
    return check


def _resolve_base_namespace(base, names, exclude):
    if isinstance(base, type):
        base = dict(walk_resolved_attrs(base))
    include_name = _resolve_name_checker(names or base) or (lambda _: True)
    exclude_name = _resolve_name_checker(exclude) or (lambda _: False)

    resolved = {}
    for name in names or base:
        if not include_name(name):
            continue
        if exclude_name(name):
            continue
        try:
            resolved[name] = base[name]
        except KeyError:
            # XXX Fail?
            continue
    return resolved


def _check_matching_attrs(cls, ns, base, fail):
    _NOT_SET = object()

    mro = list(_common_mro(cls, base)) if isinstance(base, type) else ()
    for name, base_attr in base.items():
        cls_attr = getattr(cls, name, _NOT_SET)
        if cls_attr is _NOT_SET:
            continue
        elif cls_attr is base_attr:
            # XXX Do not override in this case?
            continue

        for _base in mro:
            _base_attr = getattr(_base, name, _NOT_SET)
            if _base_attr is not _NOT_SET:
                if cls_attr is _base_attr:
                    # XXX Do not override in this case?
                    break
                if fail:
                    raise TypeError(f'cannot override existing {name!r} attr in {cls!r}')
                return False
    return True


def _common_mro(cls1, cls2):
    mro1 = cls1.__mro__
    mro2 = cls2.__mro__
    for base in mro1:
        if base in mro2:
            yield base
