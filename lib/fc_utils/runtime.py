import re
import sys


_ROOT = __name__.rpartition('.')[0]
_UTILS_RE = re.compile(rf'^{_ROOT}$|^{_ROOT}[.].*')


def get_frame(reldepth=0):
    if not reldepth or reldepth < 0:
        reldepth = 0

    depth = 1
    f = sys._getframe(depth)
    while _UTILS_RE.match(f.f_globals['__name__']):
        f = f.f_back
        depth += 1
    # XXX Support skipping other frames?
    while reldepth > 0:
        f = f.f_back
        reldepth -= 1
        depth += 1
    return f, depth



def resolve_context_name(startdepth=None):
    raise NotImplementedError
    f, _ = get_frame(startdepth)
    # XXX Get the current function/class qualname.
    ...
