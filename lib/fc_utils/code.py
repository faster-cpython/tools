import ast
import contextlib
import dis
import inspect
import io
import re
import symtable
import sys
import textwrap
import types

from ._common import IGNORE
from .text import resolve_indent, indent_lines
from .classutil import as_namedtuple
from .objects import (
    format_object_name,
    render_attrs,
    is_valid_qualname,
    resolve_qualname,
    parse_qualname,
    locate_module,
    resolve_module,
)


#UNKNOWN_FILE = '<unknown>'

INDENT = ' ' * 3


KINDS = {
    types.CellType: 'cell',
    types.ModuleType: 'module',
    type: 'class',
    types.FunctionType: 'function',
    types.LambdaType: 'function',
    types.ClassMethodDescriptorType: 'classmethod',
    types.MethodType: 'method',
}


def _get_kind(obj):
    for base in KINDS:
        if isinstance(obj, base):
            return KINDS[base]
    raise TypeError(f'unsupported type of {obj!r}')


##################################
# top-level API

def resolve_object(obj):
    kind = _get_kind(obj)
    if kind == 'cell':
        # XXX Figure out the module/class/function?
        ...
    elif kind == 'module':
        pass
    elif kind == 'class':
        pass
    elif kind == 'function':
        pass
    elif kind == 'classmethod':
        obj = obj.__func__
        kind = 'function'
    elif kind == 'method':
        obj = obj.__func__
        kind = 'function'
    else:
        raise NotImplementedError((kind, obj))
    return obj, kind
    #return code, source, obj


def get_info(obj):
    obj, kind = resolve_object(obj)
    return Info._from_object(obj)


def get_source(obj):
    obj, kind = resolve_object(obj)
    return Source._from_object(obj)


def get_ast(obj):
    obj, kind = resolve_object(obj)
    return AST._from_object(obj)


def get_symtable(obj):
    obj, kind = resolve_object(obj)
    return SymbolTable._from_object(obj)


def get_bytecode(obj):
    obj, kind = resolve_object(obj)
    return Bytecode._from_object(obj)


def render_info(info, names=None, *,
                itemnamewidth=None,
                ):
    if not isinstance(info, Info):
        raise NotImplementedError(info)
    yield from info._render(names, itemnamewidth)


def render_source(source):
    yield from source.splitlines()


def render_ast(root):
    # XXX
    ...


def render_symtable(table):
    # XXX
    ...


def render_bytecode(bytecode):
    # XXX
    ...


def render_object(obj, *,
                  info=True,
                  source=False,
                  ast=False,
                  symtable=False,
                  bytecode=False,
                  ):
    yield format_object_name(obj)
    sections = _get_sections(info, source, ast, symtable, bytecode)
    yield from _render_sections(func, sections)


##################################
# code objects

CO_FLAGS = {
    0x0001: 'CO_OPTIMIZED',
    0x0002: 'CO_NEWLOCALS',
    0x0004: 'CO_VARARGS',
    0x0008: 'CO_VARKEYWORDS',
    0x0010: 'CO_NESTED',
    0x0020: 'CO_GENERATOR',
    0x0040: 'CO_NOFREE',

    0x0080: 'CO_COROUTINE',
    0x0100: 'CO_ITERABLE_COROUTINE',
    0x0200: 'CO_ASYNC_GENERATOR',

    0x20000: 'CO_FUTURE_DIVISION',
    0x40000: 'CO_FUTURE_ABSOLUTE_IMPORT',
    0x80000: 'CO_FUTURE_WITH_STATEMENT',
    0x100000: 'CO_FUTURE_PRINT_FUNCTION',
    0x200000: 'CO_FUTURE_UNICODE_LITERALS',

    0x400000: 'CO_FUTURE_BARRY_AS_BDFL',
    0x800000: 'CO_FUTURE_GENERATOR_STOP',
    0x1000000: 'CO_FUTURE_ANNOTATIONS',
}


def format_co_flags(co_flags):
    if not co_flags:
        return '0x00'
    names = []
    remaining = co_flags
    for name, flag in CO_FLAGS.items():
        if co_flags & flag:
            remaining -= flag
            names.append(name)
    if remaining:
        raise NotImplementedError(remaining)
    return f'0x{co_flags:08x} ({" | ".join(names)})'


def _get_code_object(obj):
    kind = _get_kind(obj)
    if kind == 'code':
        return obj

    try:
        return obj.__code__
    except AttributeError:
        pass

    if kind == 'function':
        # This should be unreachable.
        raise NotImplementedError
    elif kind in ('class', 'module'):
        filename, text = _get_source(obj, root=False)
        co = compile(text, filename, 'exec')
        obj.__code__ = co
        return co
    else:
        raise NotImplementedError(obj)


##################################
# object info

class Info(types.SimpleNamespace):

    ATTRS = None
    RENDER_NAMES = None

    CODE_ATTRS = [
        'co_name',
        'co_filename',
        'co_flags',
        'co_stacksize',
        'co_firstlineno',
        'co_lnotab',

        'co_code',

        'co_nlocals',
        'co_argcount',
        'co_kwonlyargcount',

        'co_consts',
        'co_names',
        'co_localslots',
        'co_varnames',
        'co_cellvars',
        'co_freevars',
    ]
    CODE_RENDER_NAMES = {
        'co_flags',
        'co_stacksize',
        #'co_nlocals',
        'co_names',
        'co_localslots',
        'co_varnames',
        'co_cellvars',
        'co_freevars',
    }

    @classmethod
    def _from_object(cls, obj):
        self = cls(
            __obj__ = obj,
            **{name.strip('_'): getattr(obj, name)
               for name in cls.ATTRS}
        )

        if cls.ATTRS is not cls.CODE_ATTRS:
            co = _get_code_object(obj)
            if not hasattr(obj, '__code__'):
                obj.__code__ = co
            for name in cls.CODE_ATTRS:
                value = getattr(co, name)
                setattr(self, name, value)

        return self

    @property
    def filename(self):
        return self.__code__.co_filename

    def _render(self, names, width):
        if not names:
            names = type(self).RENDER_NAMES
#        codeattrs = []
#        def resolve(obj, name):
#            try:
#                return getattr(obj, name)
#            except AttributeError:
#                codeattrs.append(name)
#                raise  # re-reiase
#        yield from render_attrs(
#            func,
#            names,
#            resolve_attr=resolve,
#            onmissing=IGNORE,
#            namewidth=width,
#        )
#        yield from _render_code_info(co, codeattrs, width)
#        # XXX
#        ...

        yield from render_attrs(
            self,
            names,
            resolve_attr=getattr(self, '_resolve_attr', None),
            #resolve_attr=resolve,
            #onmissing=IGNORE,
            namewidth=width,
        )


class CodeInfo(Info):

    ATTRS = Info.CODE_ATTRS
    RENDER_NAMES = ['__closure__']

    @property
    def name(self):
        return self.co_name

    @property
    def qualname(self):
        return self.co_name

    @property
    def filename(self):
        return self.co_filename

    def _resolve_attr(self, name):
        value = getattr(self, name)
        if name == 'co_flags':
            value = format_co_flags(value)
        return value


class FuncInfo(Info):

    ATTRS = [
        '__name__',
        '__qualname__',
        '__module__',
        '__doc__',
        '__annotations__',

        '__code__',
        '__closure__',
        '__defaults__',
        '__globals__',
        '__kwdefaults__',
    ]
    RENDER_NAMES = ['__closure__']


class ClassInfo(Info):

    ATTRS = [
        '__name__',
        '__qualname__',
        '__module__',
        '__doc__',
        '__annotations__',
    ]


class ModuleInfo(Info):

    ATTRS = [
        '__name__',
        '__spec__',
        '__file__',
        '__doc__',
        '__annotations__',
    ]

    @property
    def qualname(self):
        return self.name


##################################
# source

def _get_source(obj, root):
    filename = inspect.getsourcefile(obj)
    if root:
        with open(filename) as srcfile:
            text = srcfile.read()
    else:
        text = inspect.getsource(obj)
    return filename, text


@as_namedtuple('text filename')
class Source:

    @classmethod
    def _from_object(cls, obj):
        filename, text = _get_source(obj, root=False)
        return cls(text, filename)

    def __str__(self):
        return self.text


##################################
# AST

@as_namedtuple('node root')
class AST:

    @classmethod
    def _from_object(cls, obj):
        filename, text = _get_source(obj, root=True)
        root = ast.parse(text, filename)
        # XXX
        ...
        node = ...
        return cls(node, root)


##################################
# symbol table

@as_namedtuple('table root')
class SymbolTable:

    @classmethod
    def _from_object(cls, obj):
        filename, text = _get_source(obj, root=True)
        root = symtable.symtable(text, filename, 'exec')
        # XXX
        ...
        table = ...
        return cls(table, root)


##################################
# bytecode

@as_namedtuple('lno offset op arg argtext')
class Instruction:

    DIS_RE = re.compile(textwrap.dedent(r'''
        ^ \s*
        (?:
            ( \d+ )  # <lno>
            \s+
         )?
        ( \d+ )  # <offset>
        \s+
        ( \w+ )  # <op>
        (?:
            \s+
            ( \d+ )  # <arg>
            (?:
                [(]
                ( \S | \S.+\S )  # <argtext>
                [)]
             )?
         )?
        \s* $
    '''.strip()), re.VERBOSE)

    @classmethod
    def from_line(cls, line):
        m = cls.DIS_RE.match(line)
        if not m:
            return None
        lno, offset, op, arg, argtext = m.groups()

        self = cls(
            int(lno),
            int(offset),
            op,
            int(arg) if arg else None,
            argtext,
        )
        self._line = line.strip()
        return self

    def __new__(cls, lno, offset, name, arg=None, argtext=None):
        if not arg:
            if not isinstance(arg, int) or arg is False:
                arg = None
        self = super().__new__(
            cls,
            lno,
            offset,
            op or None,
            arg if arg or arg == 0 else None,
            argtext or None,
        )
        return self

    def __init__(self, *args, **kwargs):
        if not isinstance(self.lno, int):
            if not self.lno:
                raise TypeError('missing lno')
            raise TypeError(f'expected int lno, got {self.lno!r}')
        elif self.lno < 0:
            raise ValueError(f'expected non-negative lno, got {self.lno!r}')

        if not isinstance(self.offset, int):
            if not self.offset:
                raise TypeError('missing offset')
            raise TypeError(f'expected int offset, got {self.offset!r}')
        elif self.offset < 0:
            raise ValueError(f'expected non-negative offset, got {self.offset!r}')

        if not self.name:
            raise TypeError('missing name')
        elif not isinstance(self.name, str):
            raise TypeError(f'expected str name, got {self.name!r}')
        elif not self.name.isidentifier() or self.name.upper() != self.name:
            raise ValueError(f'name {name!r} does not look like an operator')

        if not isinstance(self.arg, int) or isinstance(self.arg, bool):
            if self.arg is not None:
                raise TypeError(f'expected int arg, got {self.arg!r}')
        elif self.arg < 0:
            raise ValueError(f'expected non-negative arg, got {self.arg!r}')

        if self.argtext:
            if self.arg is None:
                raise ValueError(f'missing arg (got argtext {self.argtext!r})')

    def __str__(self):
        line = f'{self.lno:>3} {self.offset:>5} {self.name:20}'
        if self.arg is not None:
            if self.argtext:
                line = f'{line} {self.arg:>3} ({self.argtext})'
            else:
                line = f'{line} {self.arg:>3}'
        return line

    def as_line(self, *, hidelno=False):
        try:
            line = self._line
        except AttributeError:
            line = self._line = str(self)

        if hidelno:
            lno = str(self.lno)
            line = line.replace(lno, ' ' * len(lno))

        return line


class Bytecode:

    @classmethod
    def from_object(cls, obj, depth=0):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            dis.dis(obj, depth=depth)
        text = stdout.getvalue()

        instructions = []
        for line in text.splitlines():
            if not line:
                continue
            instructions.append(
                Instruction.from_line(line)
            )

        self = cls(instructions, obj)
        self._text = text
        return self

    def __init__(self, instructions, obj=None):
        if isinstance(instructions, str):
            raise TypeError(f'expected a sequence of instructions, got {instructions!r}')
        self._instructions = tuple(instructions)
        self._obj = obj

    def __repr__(self):
        return f'{self.__type__.__name__}(<{len(self)} instructions>)'

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self._instructions)

    def __iter__(self):
        yield from self._instructions

    def __getitem__(self, index):
        return self._instructions[index]

    @property
    def __obj__(self):
        return self._obj

    @property
    def text(self):
        try:
            return self._text
        except AttributeError:
            self._text = '\n'.join(self._lines())
            return self._text

    @property
    def lines(self):
        return self.text.splitlines()

    def _lines(self):
        lastlno = None
        for instr in self._instructions:
            if instr.lno == lastlno:
                yield instr.as_line(hidelno=True)
            else:
                yield ''
                yield instr.as_line(hidelno=False)
            lastlno = instr.lno

    def at_byte(self, byte):
        return self._instructions[byte // 2]


##################################
# rendered

SECTIONS = {
    'info': (render_info, ('indent',)),
    'source': render_source,
    'ast': (render_ast, ('depth',)),
    'symtable': (render_symtable, ('depth',)),
    'bytecode': render_bytecode,
}


def _resolve_section(section, opts=()):
    if not isinstance(section, str) and section is not None:
        raise TypeError(f'expected str section, got {section!r}')

    if not opts:
        section, _, opts = section.partition(':')
        opts = opts.split(',') if opts else ()
    elif ':' in section:
        raise ValueError(f'opt provided but also embedded in section')
    elif isinstance(opts, str):
        _section, sep, _opts = opts.partition(':')
        if sep:
            if not section:
                section = _section
            elif _section != section:
                raise ValueError(f'section mismatch, {_section} (from opts) != {section}')
            opts = opts.split(',') if opts else ()
        elif not opts.strip():
            opts = (opts,)
        else:
            opts = opts.replace(',', ' ').split()
            assert opts and all(opts)

    try:
        spec = SECTIONS[section]
    except KeyError:
        raise ValueError(f'unsupported section {section!r}')
    else:
        if callable(spec):
            render = spec
            kwnames = ()
        else:
            render, kwnames = spec

    if opts:
        if not kwnames:
            opts = ', '.join(repr(o) for o in opts)
            raise ValueError(f'opts not supported for {section}, got {opts}')
        elif len(opts) > len(kwnames):
            opts = ', '.join(repr(o) for o in opts)
            if len(kwnames) == 1:
                raise ValueError(f'expected at most 1 opt, got {opts}')
            else:
                raise ValueError(f'expected at most {len(kwnames)} opts, got {opts}')
    kwargs = dict(zip(kwnames, opts))
    # We do not support required opts, else we would check here.

    return section, kwargs, render


def _resolve_section_str(section, opts=()):
    section, opts, render = _resolve_section(section, opts)
    if not opts:
        return section
    return f'{section}:{",".join(opts.values())}'


def _resolve_sections(sections, *, ondupe='fail', infofirst=True):
    if not ondupe:
        ondupe = 'fail'

    resolved = []
    seen = set()
    for section in sections:
        section, opts, render = _resolve_section(section)
        if section in seen:
            if ondupe == 'fail':
                raise ValueError(f'duplicate section {section!r}')
            elif ondupe == 'ignore':
                continue
            else:
                raise ValueError(f'unsupported ondupe {ondupe!r}')
        seen.add(section)

        request = (section, opts, render)
        if infofirst and section == 'info':
            resolved.insert(0, request)
        else:
            resolved.append(request)
    return resolved


def _get_sections(info, ast, symtable, bytecode):
    actual = dict(locals())

    sections = []
    for section, value in actual.items():
        if not value:
            continue
        if value is True:
            value = None
        resolved = _resolve_section_str(section, value)
        sections.append(resolved)
    return sections


def _resolve_section_header(header, indent, default=INDENT):
    if not header and isinstance(header, str):
        header = True
    if header is True:
        header = '== {section} ==\n'

    if not indent:
        return header
        #return header.splitlines()
    indent = resolve_indent(indent, default)
    return '\n'.join(indent_lines(header, indent))
    #return list(indent_lines(header, indent))


def _render_sections(obj, sections, *,
                     sectionheader=True,
                     headerindent=False,
                     indent=None,
                     foldinfofirst=True,
                     div=True,
                     ):
    indent = resolve_indent(indent, INDENT)
    sectionheader = _resolve_section_header(sectionheader, headerindent, indent)
    if not div:
        div = ()
    elif div is True:
        div = ('',)
    elif isinstance(div, str):
        div = div.splitlines()
    else:
        raise TypeError(f'expected div to be str, got {div!r}')

    sectionlines = iter(_render_sections_raw(obj, sections))
    # There will always be at least one line.
    section, line = next(sectionlines)

    if sectionheader and section == 'info' and foldinfofirst:
        infoindent = resolve_indent(foldinfofirst, indent or INDENT)
        yield f'{infoindent}{line}'
        for section, line in sectionlines:
            if section != 'info':
                break
            yield f'{infoindent}{line}'
        else:
            # It was the only section.
            return
        yield from div

    while True:
        if sectionheader:
            #for hline in sectionheader:
            #    yield hline.format(section=section)
            yield sectionheader.format(section=section)
        yield f'{indent}{line}'

        current = section
        for section, line in sectionlines:
            if section != current:
                break
            yield f'{indent}{line}'
        else:
            # We finished the last section.
            break

        yield from div


def _render_sections_raw(obj, sections):
    resolved = _resolve_sections(sections, ondupe='ignore', infofirst=True)
    if not resolved:
        resolved = [_resolve_section('info')]
    for section, opts, render in resolved:
        for line in render(obj, **opts):
            yield section, line

##################################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(
        prog=prog,
    )

    args = parser.parse_args(argv)
    ns = vars(args)

    cmd = None

    return cmd, ns



def main(cmd, cmd_kwargs):
    raise NotImplementedError


if __name__ == '__main__':
    cmd, cmd_kwargs = parse_args()
    main(cmd, cmd_kwargs)





































##################################

#def identify_object(obj):
#    name = getattr(obj, '__qualname__', None)
#    if not name:
#        name = getattr(obj, 'name', None)
#
#    filename = getattr(obj, 'filename', None)
#    #filename = inspect.getsourcefile(obj)
#
#    kind = _get_kind(obj)
#    if kind == 'func':
#        if not filename:
#            filename = obj.__code__.co_filename
#    elif kind == 'code':
#        if not name:
#            name = obj.co_name
#        if not filename:
#            filename = obj.co_filename
#    elif kind == 'source':
#        pass
#    elif kind == 'ast':
#        # XXX Can we find the filename?
#        ...
#    elif kind == 'symtable':
#        # XXX Can we find the filename?
#        ...
#    elif kind:
#        raise NotImplementedError(kind)
#
#    return kind, name, filename


#def resolve_source(obj, *, top=True):
#    kind, name, filename = identify_object(obj)
#    qualname = None
#
#    if kind == 'func':
#        qualname = name
#        if top:
#            mod = sys.modules[obj.__module__]
#            name, _, _ = qualname.partition('.')
#            obj = getattr(mod, name)
#        text = inspect.getsource(obj)
#    elif kind == 'code':
#        text = inspect.getsource(obj)
#    elif kind == 'source':
#        text = obj
#    elif kind == 'ast':
#        # XXX
#        raise NotImplementedError
#    elif kind == 'symtable':
#        # XXX
#        raise NotImplementedError
#    else:
#        raise NotImplementedError(obj)
#
#    text = textwrap.dedent(text)
#    return text, filename, name, qualname


#def _get_kind(obj):
#    kind = getattr(obj, 'kind', None)
#    if kind in KINDS:
#        return kind
#
#    if isinstance(obj, str):
#        # XXX Is this good enough?
#        return 'source'
#    elif hasattr(obj, '__code__'):
#        return 'func'
#    elif hasattr(obj, 'co_code'):
#        return 'code'
#    elif _is_ast_object(obj):
#        return 'ast'
#    elif _is_symtable_object(obj):
#        return 'symtable'
#    else:
#        return None


#def resolve_object(obj, kind=None):
#    curkind = _get_kind(obj)
#    if not curkind:
#        raise TypeError(f'could not determine kind for {obj!r}')
#    if not kind or kind == curkind:
#        # XXX If an info object, change to the actual object?
#        return obj, curkind, ...
#    if kind not in KINDS:
#        raise ValueError(f'unsupported kind {kind!r}')
#
#    if curkind == 'func':
#        filename = obj.__code__.co_filename
#        if kind == 'code':
#            resolved = obj.__code__
#        elif kind == 'ast':
#            text, filename, _, _ = resolve_source(obj)
#            resolved = ast.parse(text, filename)
#        elif kind == 'symtable':
#            text, filename, _, _ = resolve_source(obj)
#            resolved = symtable.symtable(text, filename, 'exec')
#        elif kind == 'source':
#            resolved = resolve_source(obj)
#        else:
#            raise NotImplementedError
#    elif curkind == 'code':
#        if kind == 'func':
#            raise NotImplementedError
#        elif kind == 'ast':
#            taxt = resolve_source(obj)
#            resolved = ast.parse(text, filename)
#        elif kind == 'symtable':
#            taxt = resolve_source(obj)
#            resolved = symtable.symtable(text, filename, 'exec')
#        elif kind == 'source':
#            resolved = resolve_source(obj)
#        else:
#            raise NotImplementedError
#    elif curkind == 'source':
#        filename = UNKNOWN_FILE
#        if kind == 'func':
#            raise NotImplementedError
#        elif kind == 'code':
#            resolved = compile(obj, filename, 'exec')
#        elif kind == 'ast':
#            resolved = ast.parse(obj, filename)
#        elif kind == 'symtable':
#            resolved = symtable.symtable(obj, filename, 'exec')
#        else:
#            raise NotImplementedError
#    elif curkind == 'ast':
#        if kind == 'func':
#            raise NotImplementedError
#        elif kind == 'code':
#            resolved = compile(obj, filename, 'exec')
#        elif kind == 'symtable':
#            taxt = resolve_source(obj)
#            resolved = symtable.symtable(text, filename, 'exec')
#        elif kind == 'source':
#            resolved = resolve_source(obj)
#        else:
#            raise NotImplementedError
#    elif curkind == 'symtable':
#        if kind == 'func':
#            raise NotImplementedError
#        elif kind == 'code':
#            resolved = compile(obj, filename, 'exec')
#        elif kind == 'ast':
#            taxt = resolve_source(obj)
#            resolved = ast.parse(text, filename)
#        elif kind == 'source':
#            resolved = resolve_source(obj)
#        else:
#            raise NotImplementedError
#    else:
#        raise NotImplementedError
#
#
#
#    # XXX Convert from obj to kind.
#    raise NotImplementedError
#
#
#
#
#    if isinstance(obj, str):
#        co = compile(obj, UNKNOWN_FILE, 'exec')
#        func = None
#    else:
#        co, func = resolve_code_object(obj)
#    ...


#def get_object_info(obj, kind=None):
#    ...


#def render_object(obj, *,
#                  info=True,
#                  source=False,
#                  ast=False,
#                  symtable=False,
#                  bytecode=True,
#                  ):
#    yield format_object_name(obj)
#    sections = _get_sections(info, ast, symtable, bytecode)
#    yield from _render_sections(func, sections)


#class FuncID(types.SimpleNamespace):
#
#    def __init__(self, name, filename, qualname=None, module=None):
#        if qualname and not is_valid_qualname(qualname):
#            raise ValueError(f'bad qualname {qualname!r}')
#        if not name:
#            if not qualname:
#                raise TypeError('missing name')
#            _, _, name = qualname.rpartition('.')
#        elif not name.isidentifier():
#            raise ValueError(f'expected name to be identifier, got {name!r}')
#        elif qualname and qualname.rpartition('.')[-1] != name:
#            raise ValueError(f'name {name!r} does not match qualname {qualname!r}')
#
#        if not filename:
#            if module:
#                filename = locate_module(module)
#                if not filename:
#                    filename = UNKNOWN_FILENAME
#                    #raise TypeError('missing filename')
#
#        super().__init__(
#            name=name,
#            filename=filename,
#        )
#        if qualname:
#            self.qualname = qualname
#        if module:
#            self.module = module
#
#    def __repr__(self):
#        self.qualname
#        self.module
#        return super().__repr__()
#
#    def __getattr__(self, name):
#        if name == 'module':
#            if self.filename == UNKNOWN_FILE:
#                raise TypeError('missing module')
#            self.module = resolve_module(self.filename)
#            return self.module
#        elif name == 'qualname':
#            if self.filename == UNKNOWN_FILE:
#                return self.name
#                #raise TypeError('missing qualname')
#            self.qualname = resolve_qualname(self.name, self.filename)
#            return self.qualname
#        else:
#            raise AttributeError(name)
#
#
#class CodeInfo(types.SimpleNamespace):
#
#    def __init__(self, __obj__, name, kind, filename, type, targetfunc=None,
#                 **kwargs):
#        if kind not in self.KINDS:
#            raise ValueError(f'unsupported kind {kind!r}')
#
#        if isinstance(targetfunc, str):
#            qualname, module, _filename = parse_filename(targetfunc)
#            targetfunc = FuncID(None, _filename or filename, qualname, module)
#        if not filename and targetfunc:
#            filename = targetfunc.filename
#        elif filename and targetfunc and filename != targetfunc.filename:
#            if filename == UNKNOWN_FILE:
#                filename = targetfunc.filename
#            elif targetfunc.filename == UNKNOWN_FILE:
#                targetfunc.filename = filename
#            else:
#                raise ValueError(f'filename mismatch ({filename!r} != {targetfunc.filename!r})')
#
#        kwargs.update(item for item in locals().items()
#                      if k not in ('self', 'kwargs', '__class__'))
#        super().__init__(**kwargs)
#
#    def __str__(self):
#        return f'<({self.kind}) {self.type} {self.name!r}>'


##################################
# function objects

#FUNC_ATTRS = [
#    '__annotations__',
#    '__closure__',
#    '__code__',
#    '__defaults__',
#    '__doc__',
#    '__globals__',
#    '__kwdefaults__',
#    '__module__',
#    '__qualname__',
#]
#
#
#def get_func_info(func):
#    func = resolve_object(func, 'function'
#    kwargs = {a.strip('_'): getattr(func, a) for a in FUNC_ATTRS}
#    return CodeInfo(
#        __obj__ = func,
#        kind='function',
#        type='function',
#        **kwargs
#    )


#def render_func_info(func, names=None, *,
#                     itemnamewidth=None,
#                     ):
#    co, _ = resolve_code_object(func)
#    yield from _render_func_info(func, co, names, itemnamewidth)


#def _render_func_info(func, co, names=None, width=None):
#    if not names:
#        names = ['__closure__']
#    codeattrs = []
#    def resolve(obj, name):
#        try:
#            return getattr(obj, name)
#        except AttributeError:
#            codeattrs.append(name)
#            raise  # re-reiase
#    yield from render_attrs(
#        func,
#        names,
#        resolve_attr=resolve,
#        onmissing=IGNORE,
#        namewidth=width,
#    )
#    yield from _render_code_info(co, codeattrs, width)


##################################
# code objects

#def resolve_code_object(obj):
#    if hasattr(obj, 'co_code'):
#        return obj, obj
#    try:
#        co = obj.__code__
#    except AttributeError:
#        raise TypeError(f'could not resolve code object from {obj!r}')
#    return co, obj


#CODE_ATTRS = [
#    'co_name',
#    'co_filename',
#    'co_flags',
#    'co_stacksize',
#    'co_firstlineno',
#    'co_lnotab',
#
#    'co_code',
#
#    'co_nlocals',
#    'co_argcount',
#    'co_kwonlyargcount',
#
#    'co_consts',
#    'co_names',
#    'co_varnames',
#    'co_cellvars',
#    'co_freevars',
#]


#def get_code_info(co):
#    kwargs = {a.strip('_'): getattr(func, a) for a in FUNC_ATTRS}
#    return CodeInfo(
#        kind='function',
#        type='function',
#        **kwargs
#    )
#    ...


#def render_code(co, *,
#                info=True,
#                ast=False,
#                symtable=False,
#                bytecode=True,
#                ):
#    co, _ = resolve_code_object(co)
#
#    yield format_object_name(co)
#
#    sections = _get_sections(info, ast, symtable, bytecode)
#    yield from _render_sections(co, sections)


#def render_code_info(co, names=None, *,
#                     itemnamewidth=None,
#                     ):
#    co = getattr(co, '__code__', co)
#    return _render_code_info(co, names, itemnamewidth)


#def _render_code_info(co, names=None, width=None):
#    if not names:
#        names = ('flags', 'names', 'nlocals', 'localslots', 'varnames', 'cellvars', 'freevars')
#        names = [f'co_{n}' for n in names]
#    yield from render_attrs(
#        co,
#        names,
#        resolve_attr=_resolve_code_attr,
#        onmissing=None,
#        namewidth=width,
#    )


#def _resolve_code_attr(co, name):
#    value = getattr(co, name)
#    if name == 'co_flags':
#        value = format_co_flags(value)
#    return value


##################################
# code data: AST

#def _is_ast_object(obj):
#    return isinstance(obj, ast.AST)


def render_ast(obj, *, depth='minimal'):
    if isinstance(obj, ast.AST):
        node = obj
    else:
        text, filename, _, qualname = resolve_source(obj)
        node = ast.parse(text, filename)
        assert isinstance(node, ast.Module)
        if qualname:
            if not depth:
                depth = 'minimal'

            if depth == 'top':
                node = node.body[0]
            elif depth in ('minimal', 'sub'):
                name, _, remainder = qualname.partition('.')
                while name:
                    if name != '<locals>':
                        for child in node.body:
                            if not isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                                continue
                            if child.name == name:
                                node = child
                                break
                        else:
                            raise NotImplementedError((qualname, name, remainder))
                    name, _, remainder = remainder.partition('.')
                if depth == 'minimal':
                    for i, child in enumerate(node.body or ()):
                        if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                            node.body[i] = f'<{type(child).__name__} {child.name!r} at 0x{id(child):x}>'
            elif depth != 'full':
                raise NotImplementedError(depth)
    yield from ast.dump(node, indent=3).splitlines()


##################################
# code data: symbol table

#def _is_symtable(obj):
#    return isinstance(obj, (symtable.SymbolTable,
#                            symtable.Function,
#                            symtable.Class,
#                            symtable.Symbol,
#                            ))


def symtable_snapshot(obj):
    if isinstance(obj, str):
        obj = symtable.symtable(obj, UNKNOWN_FILE, 'exec')

    if isinstance(obj, symtable.SymbolTable):
        snapshot = types.SimpleNamespace(
            __obj__=obj,
            id=obj.get_id(),
            type=obj.get_type(),
            name=obj.get_name(),
            nested=obj.is_nested(),
            optimized=obj.is_optimized(),
            has_children=obj.has_children(),
            lineno=obj.get_lineno(),

            identifiers=list(obj.get_identifiers() or ()),
        )
        if isinstance(obj, symtable.Function):
            snapshot.params = list(obj.get_parameters() or ())
            snapshot.local_vars = list(obj.get_locals() or ())
            snapshot.nonlocal_vars = list(obj.get_nonlocals() or ())
            snapshot.free_vars = list(obj.get_frees() or ())
            snapshot.global_vars = list(obj.get_globals() or ())
        elif isinstance(obj, symtable.Class):
            pass
        elif type(obj) is not symtable.SymbolTable:
            raise NotImplementedError(obj)

        snapshot.symbols = [symtable_snapshot(v)
                            for v in obj.get_symbols() or ()]
        snapshot.children = [symtable_snapshot(v)
                             for v in obj.get_children() or ()]

        return snapshot
    elif isinstance(obj, symtable.Symbol):
        if obj.is_parameter():
            scope = 'param'
        elif obj.is_local():
            scope = 'local'
        elif obj.is_nonlocal():
            scope = 'nonlocal (explicit)'
        elif obj.is_free():
            scope = 'free'
        elif obj.is_declared_global():
            scope = 'global (explicit)'
        elif obj.is_global():
            scope = 'global (implicit)'
        else:
            raise NotImplementedError
        snapshot = types.SimpleNamespace(
            __obj__=obj,
            type='symbol',
            name=obj.get_name(),

            scope=f'<{scope}>',
            #param=obj.is_parameter(),
            #global_var=obj.is_global(),
            #nonlocal_var=obj.is_nonlocal(),
            #declared=obj.is_declared_global(),
            #local_var=obj.is_local(),
            #free=obj.is_free(),

            used_locally=obj.is_referenced(),
            assigned=obj.is_assigned(),

            imported=obj.is_imported(),
            annotated=obj.is_annotated(),

            namespace=obj.is_namespace(),
            bound=str(obj.get_namespaces()),
        )
        return snapshot
    elif isinstance(obj, types.SimpleNamespace) and hasattr(obj, '__obj__'):
        return obj
    else:
        text, filename, name, qualname = resolve_source(obj)
        table = symtable.symtable(text, filename, 'exec')
        #table.lookup()
        snapshot = symtable_snapshot(table)
        #if qualname and name != qualname:
        #    # XXX Walk back dowo to the original object.
        #    raise NotImplementedError
        return snapshot


#def render_symtable(obj):
#    snapshot = symtable_snapshot(obj)
#    yield from _render_symtable(snapshot, '   ')


def _render_symtable(snapshot, indent):
    ignored = (
        'id', 'name', 'type',
        'has_children', 'lineno',
        'symbols',
        'imported', 'annotated',
    )
    def resolve(obj, name):
        if name in ignored:
            raise AttributeError(name)
        return getattr(obj, name)

    def render(_obj, _name, value):
        if not isinstance(value, list):
            return value, None
        elif not value:
            return None, None

        if isinstance(value[0], str):
            # XXX Deal with spaces in the middle?
            if len(value) < 6:
                return ', '.join(value), None
            return value, None

        lines = []
        for item in value:
            lines.extend(_render_symtable(item, '   '))
        return lines, '   '

    #yield str(snapshot.__obj__)
    #prefix = 'symbol_table for ' if snapshot.type != 'symbol' else ''
    #yield f'<{prefix}{snapshot.type} {snapshot.name}>'
    yield f'<{snapshot.type} {snapshot.name}>'
    for line in render_attrs(snapshot, snapshot,
                             resolve_attr=resolve,
                             render_value=render,
                             onmissing=IGNORE,
                             ):
        yield indent + line


##################################
# code data: bytecode

#def render_bytecode(obj, depth=0):
#    stdout = io.StringIO()
#    with contextlib.redirect_stdout(stdout):
#        dis.dis(obj, depth=depth)
#    yield from stdout.getvalue().splitlines()


##################################
# combined

#def render_info(obj, indent=None):
#    if isinstance(obj, str):
#        co = compile(obj, UNKNOWN_FILE, 'exec')
#        func = None
#    else:
#        co, func = resolve_code_object(obj)
#
#    if not func or func is co:
#        lines = _render_code_info(co)
#    else:
#        lines = _render_func_info(func, co)
#    if indent:
#        lines = indent_lines(lines, indent)
#    yield from lines
