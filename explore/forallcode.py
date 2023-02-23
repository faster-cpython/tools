"""Grab code from the filesystem, compile it, and process it."""

import os
import sys
import types
from typing import Iterator


def recurse_code(code: types.CodeType, verbose: int) -> Iterator[types.CodeType]:
    if verbose >= 2:
        print(f"  Processing code {code}")
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from recurse_code(const, verbose)


def recurse_file(filename: str, verbose: int) -> Iterator[types.CodeType]:
    if verbose >= 1:
        print(f"Processing file {filename}")
    try:
        with open(filename, "r") as f:
            code = compile(f.read(), filename, "exec")
            yield from recurse_code(code, verbose)
    except OSError as e:
        if verbose >= 0:
            print(f"{filename}: {e!r}")
    except SyntaxError as e:
        if verbose >= 0:
            print(f"{filename}: {e!r}")

def recurse_dir(dirname: str, verbose: int) -> Iterator[types.CodeType]:
    if verbose >= 1:
        print(f"Processing dir {dirname}")
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(".py"):
                full = os.path.join(root, file)
                yield from recurse_file(full, verbose)


def recurse_tarball(filename: str, verbose: int) -> Iterator[types.CodeType]:
    if verbose >= 1:
        print(f"Processing tarball {filename}")
    import tarfile

    with tarfile.open(filename, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".py"):
                if verbose >= 1:
                    print(f"Processing tar member {member.name}")
                with tar.extractfile(member) as f:
                    code = compile(f.read(), member.name, "exec")
                    yield from recurse_code(code, verbose)


def expand_globs(filenames: list[str]) -> Iterator[str]:
    for filename in filenames:
        if "*" in filename and sys.platform == "win32":
            for fn in glob.glob(filename):
                yield fn
        else:
            yield filename


def find_all_code(paths: list[str], verbose: int = 1) -> Iterator[types.CodeType]:
    for filename in expand_globs(paths):
        if os.path.isfile(filename):
            if filename.endswith(".tar.gz"):
                yield from recurse_tarball(filename, verbose)
            else:
                yield from recurse_file(filename, verbose)
        elif os.path.isdir(filename):
            yield from recurse_dir(filename, verbose)
        else:
            print(f"{filename}: Cannot open")


def main():
    for code in find_all_code(sys.argv[1:]):
        print(f"    {code.co_name}:{code.co_firstlineno}: {len(code.co_code)} bytes")


if __name__ == "__main__":
    main()
