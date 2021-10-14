import ast
import contextlib
import copy
import dis
import inspect
import io
import re
import symtable
import sys
import textwrap
import types

from .objects import (
    format_object_name,
    normalize_rendered,
    render_attrs,
)
from .text import indent_lines


INDENT = ' ' * 3

DIS_RE = re.compile(textwrap.dedent(r'''
    ^ \s*
    (?:
        ( \d+ )  # <lno>
        \s+
     )?
    (?:
        ( >> )  # <marker>
        \s+
     )?
    ( \d+ )  # <offset>
    \s+
    ( \w+ )  # <op>
    (?:
        \s+
        ( \d+ )  # <arg>
        (?:
            \s+
            [(]
            ( .* )  # <argtext>
            #( \S | \S.+\S )  # <argtext>
            [)]
         )?
     )?
    \s* $
'''.strip()), re.VERBOSE)


def render_func(func, sections=None, *,
                info=True,
                ):
    (sectionnames, sections, ignored,
     ) = _parse_sections(sections, ignored=('info',))
    if 'info' in ignored:
        if info is False:
            raise ValueError('info is False but found in sections')
        info = True
    elif not sectionnames:
        info = True

    yield format_object_name(func)

    if info:
        kwargs, = ignored.get('info') or ({},)
        data = _get_info(func, **kwargs)
        lines = _render_info(data, **kwargs)
        yield from indent_lines(lines, INDENT)

    for section in sectionnames:
        get_data, render = _resolve_section(section)
        kwargs, = sections[section]
        data = get_data(func, **kwargs)
        yield ''
        yield f'== {section} =='
        yield ''
        lines = render(data, **kwargs)
        yield from indent_lines(lines, INDENT)


##################################
# helpers

def _read_source(func, root=True):
    #filename = inspect.getsourcefile(func)
    filename = func.__code__.co_filename
    if root:
        with open(filename) as srcfile:
            text = srcfile.read()
    else:
        text = inspect.getsource(func)
    return filename, text


def _find_ast_node(root, qualname):
    curnode = root
    name, _, remainder = qualname.partition('.')
    while name:
        if name != '<locals>':
            for child in curnode.body:
                if not isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if child.name == name:
                    curnode = child
                    break
            else:
                raise NotImplementedError((curnode, qualname, name, remainder))
        name, _, remainder = remainder.partition('.')
    return curnode


def _find_symbol_table(root, qualname):
    assert isinstance(root, symtable.SymbolTable)

    curtable = root
    name, _, remainder = qualname.partition('.')
    while name:
        if name != '<locals>':
            for child in curtable.get_children() or ():
                if not isinstance(child, (symtable.SymbolTable)):
                    continue
                if child.get_name() == name:
                    curtable = child
                    break
            else:
                raise NotImplementedError((curtable, qualname, name, remainder))
        name, _, remainder = remainder.partition('.')
    return curtable


def _build_symtable_snapshot(obj):
    if isinstance(obj, symtable.SymbolTable):
        table = obj
        snapshot = types.SimpleNamespace(
            __obj__=table,
            id=table.get_id(),
            kind=table.get_type(),
            name=table.get_name(),
            nested=table.is_nested(),
            optimized=table.is_optimized(),
            has_children=table.has_children(),
            lineno=table.get_lineno(),

            identifiers=list(table.get_identifiers() or ()),
        )
        if isinstance(table, symtable.Function):
            snapshot.params = list(table.get_parameters() or ())
            snapshot.local_vars = list(table.get_locals() or ())
            snapshot.nonlocal_vars = list(table.get_nonlocals() or ())
            snapshot.free_vars = list(table.get_frees() or ())
            snapshot.global_vars = list(table.get_globals() or ())
        elif isinstance(table, symtable.Class):
            pass
        elif type(table) is not symtable.SymbolTable:
            raise NotImplementedError(table)
        # XXX Check for unexpected attrs?

        snapshot.symbols = [_build_symtable_snapshot(v)
                            for v in table.get_symbols() or ()]
        snapshot.children = [_build_symtable_snapshot(v)
                             for v in table.get_children() or ()]

        return snapshot
    elif isinstance(obj, symtable.Symbol):
        symbol = obj
        if symbol.is_parameter():
            scope = 'param'
        elif symbol.is_local():
            scope = 'local'
        elif symbol.is_nonlocal():
            scope = 'nonlocal (explicit)'
        elif symbol.is_free():
            scope = 'free'
        elif symbol.is_declared_global():
            scope = 'global (explicit)'
        elif symbol.is_global():
            scope = 'global (implicit)'
        else:
            raise NotImplementedError
        snapshot = types.SimpleNamespace(
            __obj__=symbol,
            kind='symbol',
            name=symbol.get_name(),

            scope=f'<{scope}>',
            #param=symbol.is_parameter(),
            #global_var=symbol.is_global(),
            #nonlocal_var=symbol.is_nonlocal(),
            #declared=symbol.is_declared_global(),
            #local_var=symbol.is_local(),
            #free=symbol.is_free(),

            used_locally=symbol.is_referenced(),
            assigned=symbol.is_assigned(),

            imported=symbol.is_imported(),
            annotated=symbol.is_annotated(),

            namespace=symbol.is_namespace(),
            bound=str(symbol.get_namespaces()),
        )
        # XXX Check for unexpected attrs?
        return snapshot
    else:
        raise NotImplementedError(obj)


##################################
# section-specific

def _get_info(func):
    info = types.SimpleNamespace()

    func_attrs = [
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
    for attr in func_attrs:
        value = getattr(func, attr, None)
        setattr(info, attr, value)

    code_attrs = [
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
    co = func.__code__
    for attr in code_attrs:
        value = getattr(co, attr, None)
        setattr(info, attr, value)

    return info


def _render_info(info):
    namewidth = 15

    names = [
        '__closure__',

        'co_flags',
        'co_stacksize',
        'co_names',
        'co_consts',
        'co_localslots',
        'co_varnames',
        'co_cellvars',
        'co_freevars',
    ]
    for name in names:
        value = getattr(info, name, None)
        value = normalize_rendered(value)
        #name = name.strip('_')
        yield f'{(name + ":").ljust(namewidth)} {value}'


def _get_source(func, *, root=False):
    _, text = _read_source(func, root)
    return text


def _render_source(text, **kwargs):
    text = textwrap.dedent(text)
    yield from text.splitlines()


def _get_ast(func, **kwargs):
    filename, text = _read_source(func, root=True)
    qualname = func.__qualname__
    topname = qualname.partition('.')[0]

    root = ast.parse(text, filename)
    top = _find_ast_node(root, topname)
    node = _find_ast_node(root, qualname)

    return node, top, root


def _render_ast(data, *, depth='minimal'):
    node, top, root = data
    if not depth:
        depth = 'minimal'

    if depth == 'full':
        node = root
    elif depth == 'top':
        node = top
    elif depth == 'minimal':
        node = copy.deepcopy(node)
        for i, child in enumerate(node.body or ()):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                node.body[i] = f'<{type(child).__name__} {child.name!r} at 0x{id(child):x}>'
    elif depth != 'sub':
        raise NotImplementedError(depth)

    text = ast.dump(node, indent=3)
    yield from text.splitlines()


def _get_symtable(func, **kwargs):
    filename, text = _read_source(func, root=True)
    qualname = func.__qualname__
    topname = qualname.partition('.')[0]

    root = symtable.symtable(text, filename, 'exec')
    top = _find_symbol_table(root, topname)
    table = _find_symbol_table(root, qualname)

    return table, top, root


def _render_symtable(data, *, depth='minimal'):
    namewidth = 15
    table, top, root = data
    if not depth:
        depth = 'minimal'

    if depth == 'full':
        table = _build_symtable_snapshot(root)
    elif depth == 'top':
        table = _build_symtable_snapshot(top)
    elif depth == 'minimal':
        table = _build_symtable_snapshot(table)
        for i, child in enumerate(table.children):
            assert not isinstance(child, symtable.SymbolTable)
            if isinstance(child, types.SimpleNamespace):
                name = child.name
                child = child.__obj__
                table.children[i] = f'<{type(child).__name__} {name!r} at 0x{id(child):x}>'
    elif depth == 'sub':
        table = _build_symtable_snapshot(table)
    else:
        raise NotImplementedError(depth)

    ignored = (
        'id', 'name', 'kind',
        'has_children', 'lineno',
        'symbols',
        'imported', 'annotated',
    )
    def render_simple(name, value, indent):
        value = normalize_rendered(value)
        return f'{indent}{(name + ":").ljust(namewidth)} {value}'

    def render(curobj, indent=''):
        assert isinstance(curobj, types.SimpleNamespace)

        yield f'{indent}<{curobj.kind} {curobj.name}>'
        indent += INDENT
        for name, value in vars(curobj).items():
            if name in ignored or name.startswith('_'):
                continue

            if not isinstance(value, list) or not value:
                yield render_simple(name, value, indent)
                continue

            if isinstance(value[0], str):
                # XXX Deal with spaces in the middle?
                if len(value) < 6:
                    value = ', '.join(value)
                    yield render_simple(name, value, indent)
                    continue

                yield f'{indent}{name}:'
                itemindent = indent + INDENT
                for item in value:
                    item = normalize_rendered(item)
                    yield f'{itemindent}{item}'
            else:
                yield f'{indent}{name}:'
                for item in value:
                    yield from render(item, indent + INDENT)

    yield from render(table)


#def _get_bytecode(func, *, depth=None):
def _get_bytecode(func, *, depth=0):
    if depth == 'full':
        target = sys.modules[func.__module__]
        depth = None
    elif depth == 'top':
        topname = func.__qualname__.partition('.')[0]
        target = getattr(sys.modules[func.__module__], topname)
        depth = None
    else:
        target = func
        if depth == 'minimal':
            depth = 0
        elif depth == 'sub':
            depth = None
        elif isinstance(depth, str):
            if not depth.isdigit():
                raise NotImplementedError
            depth = int(depth)
        elif depth is not None and type(depth) is not int:
            raise NotImplementedError(depth)

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        dis.dis(target, depth=depth)
    text = stdout.getvalue()

    instructions = []
    lastlno = None
    for line in text.splitlines():
        m = DIS_RE.match(line)
        if not m:
            instructions.append(line.strip())
            continue

        lno, marker, offset, op, arg, argtext = m.groups()
        if not lno:
            lno = lastlno
        instructions.append([
            line.rstrip(),
            int(lno) if lno is not None else None,
            marker or None,
            int(offset),
            op,
            int(arg) if arg else None,
            argtext,
        ])

        lastlno = lno
    return instructions


def _render_bytecode(instructions, **kwargs):
    for instr in instructions:
        if isinstance(instr, str):
            yield instr
        else:
            line, *_ = instr
            yield line


#def _render_bytecode(instructions, **kwargs):
#    for _, lno, offset, op, arg, argtext in instructions:
#        line = f'{lno:>3} {offset:>5} {name:20}'
#        if self.arg is not None:
#            if self.argtext:
#                line = f'{line} {self.arg:>3} ({self.argtext})'
#            else:
#                line = f'{line} {self.arg:>3}'
#        yield line


##################################
# rendered sections

SECTIONS = {
    'info': (_get_info, _render_info),
    'source': (_get_source, _render_source, 'root'),
    'ast': (_get_ast, _render_ast, 'depth'),
    'symtable': (_get_symtable, _render_symtable, 'depth'),
    'bytecode': (_get_bytecode, _render_bytecode, 'depth'),
}


def _parse_sections(raw, ignored=None):
    assert not isinstance(raw, str)

    names = []
    sections = {}
    for entry in raw:
        section, spec = _parse_section(entry)
        if section in sections:
            if spec == sections[section]:
                continue
            raise ValueError(f'duplicate section {section!r}')
        names.append(section)
        sections[section] = spec

    if ignored:
        ignored = {name: sections.pop(name)
                   for name in ignored
                   if name in sections}
        for name in ignored:
            names.remove(name)

    return names, sections, ignored or {}


def _parse_section(raw):
    section, sep, args = raw.partition(':')
    if not sep:
        args = ()
    elif not args or not args.strip():
        args = [args]
    else:
        args = args.split(',')
    args = [int(a) if a.isdigit() else a for a in args]

    try:
        _, _, *kwnames = SECTIONS[section]
    except KeyError:
        raise ValueError(f'unsupported section {section!r}')
    if len(args) > len(kwnames):
        raise ValueError(f'got extra args {args!r}')
    kwargs = dict(zip(kwnames, args))

    spec = (kwargs,)
    return section, spec


def _resolve_section(section):
    try:
        get_data, render, *_ = SECTIONS[section]
    except KeyError:
        raise ValueError(f'unsupported section {section!r}')
    return get_data, render
