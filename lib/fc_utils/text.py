

def resolve_indent(indent, default='    '):
    if not indent:
        return ''
    elif isinstance(indent, int):
        return ' ' * indent

    if not default and not isinstance(default, str):
        default = '    '
    if indent is True:
        indent = default

    if not isinstance(indent, str):
        raise TypeError(f'indent should be a str, got {indent!r}')
    elif indent.strip():
        raise ValueError(f'indent should be only whitespace, got {indent!r}')
    else:
        return indent


def indent_line(line, indent='    '):
    if not indent:
        return line
    indent = resolve_indent(indent)
    return f'{indent}{line}'


def indent_lines(lines, indent='    '):
    return format_lines(lines, indent=indent)


def format_lines(lines, *,
                 strip=False,
                 indent=None,
                 prefix=None,
                 ):
    if isinstance(lines, str):
        lines = lines.splitlines()

    if strip:
        # XXX Support 'left' and 'right'.
        lines = (l.strip() for l in lines)
    if prefix is not None:
        lines = (f'{prefix}{l}' for l in lines)
    if indent:
        indent = resolve_indent(indent)
        lines = (f'{indent}{l}' for l in lines)
    return lines


def format_item(name, value, namewidth=15):
    fmt = '{:%s} {}' % (namewidth or 15,)
    return fmt.format(name + ':', value)
