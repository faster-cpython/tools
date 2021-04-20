# Count opcode statistics in a collection of source code files

# TODO:
# - Count cases like LOAD_CONST <small int>, BINARY_ADD

import argparse
import glob
import opcode
import os
import sys
import tarfile
import warnings
from collections import Counter

# Special (non-opcode) counter keys
NOPCODES = "__nopcodes__"  # Number of opcodes
NPAIRS = "__npairs__"  # Number of opcode pairs
NFILES = "__nfiles__"  # Number of files
NLINES = "__nlines__"  # Number of lines
NCODEOBJS = "__ncodeobjs__"  # Number of code objects
NERRORS = "__nerrors__"  # Number of files with errors

# TODO: Make this list an option
OF_INTEREST_NAMES = [x for x in opcode.opname if not x.startswith("<")]
OF_INTEREST_NAMES = ["BINARY_ADD", "BINARY_SUBTRACT",
                     "INPLACE_ADD", "INPLACE_SUBTRACT"]

of_interest = set(opcode.opmap[x] for x in OF_INTEREST_NAMES)


def all_code_objects(code):
    yield code
    for x in code.co_consts:
        if hasattr(x, 'co_code'):
            yield x


SHOW_ITEMS = [
    (NERRORS, "errors"),
    (NFILES, "files"),
    (NCODEOBJS, "code objects"),
    (NLINES, "lines"),
    (NOPCODES, "opcodes"),
    (NPAIRS, "opcode pairs"),
]
def showstats(counter):
    res = []
    for key, name in SHOW_ITEMS:
        if key in counter:
            res.append(f"{counter[key]:>1,} {name}")
    return "; ".join(res)


def report(source, filename, verbose):
    counter = Counter()
    try:
        code = compile(source, filename, "exec")
    except Exception as err:
        if verbose > 0:
            print(f"{filename}: {err}")
        counter[NERRORS] += 1
        return counter

    for co in all_code_objects(code):
        counter[NCODEOBJS] += 1
        co_code = co.co_code
        for i in range(0, len(co_code), 2):
            counter[NOPCODES] += 1
            op = co_code[i]
            counter[op] += 1
            if i > 0:
                lastop = co_code[i-2]
                counter[NPAIRS] += 1
                counter[(lastop, op)] += 1

    counter[NLINES] += len(source.splitlines())
    if verbose > 0:
        print(f"{filename}: {showstats(counter)}")
    counter[NFILES] += 1
    return counter


def file_report(filename, verbose):
    try:
        with open(filename, "rb") as f:
            source = f.read()
        counter = report(source, filename, verbose)
    except Exception as err:
        if verbose > 0:
            print(filename + ":", err)
        counter = Counter()
        counter[NERRORS] += 1
    return counter


def tarball_report(filename, verbose):
    if verbose > 1:
        print(f"\nExamining tarball {filename}")
    counter = Counter()
    with tarfile.open(filename, "r") as tar:
        members = tar.getmembers()
        for m in members:
            info = m.get_info()
            name = info["name"]
            if name.endswith(".py"):
                try:
                    source = tar.extractfile(m).read()
                except Exception as err:
                    if verbose > 0:
                        print(f"{name}: {err}")
                    counter[NERRORS] += 1
                else:
                    counter.update(report(source, name, verbose-1))
    if verbose > 0:
        print(f"{filename}: {showstats(counter)}")
    return counter


def expand_globs(filenames):
    for filename in filenames:
        if "*" in filename and sys.platform == "win32":
            for fn in glob.glob(filename):
                yield fn
        else:
            yield filename


argparser = argparse.ArgumentParser()
argparser.add_argument("-q", "--quiet", action="store_true",
                       help="less verbose output")
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="more verbose output")
argparser.add_argument("--singles", type=int,
                      help="show N most common opcodes")
argparser.add_argument("--pairs", type=int,
                      help="show N most common opcode pairs")
argparser.add_argument("filenames", nargs="*", metavar="FILE",
                       help="files, directories or tarballs to count")


def main():
    args = argparser.parse_args()
    verbose = 1 + args.verbose - args.quiet
    if not args.pairs and not args.singles:
        args.pairs = 20

    filenames = args.filenames
    if not filenames:
        argparser.print_usage()
        sys.exit(0)

    if verbose < 2:
        warnings.filterwarnings("ignore", "", SyntaxWarning)

    if verbose >= 2:
        print("Looking for", ", ".join(OF_INTEREST_NAMES))
        print("In", filenames)

    counter = Counter()
    hits = 0
    for filename in expand_globs(filenames):
        hits += 1
        if os.path.isfile(filename):
            if filename.endswith(".tar.gz"):
                ctr = tarball_report(filename, verbose)
            else:
                ctr = file_report(filename, verbose)
            counter.update(ctr)
        elif os.path.isdir(filename):
            for root, dirs, files in os.walk(filename):
                for file in files:
                    if file.endswith(".py"):
                        full = os.path.join(root, file)
                        counter.update(file_report(full, verbose))
            if not counter[NFILES]:
                print(f"{filename}: No .py files")
        else:
            print(f"{filename}: Cannot open")
            counter[NERRORS] += 1
    if not hits:
        print("No files after expansion")
        sys.exit(1)

    nerrors = counter[NERRORS]
    nfiles = counter[NFILES]
    if not nfiles:
        sys.exit(bool(nerrors))

    nlines = counter[NLINES]
    nopcodes = counter[NOPCODES]
    npairs = counter[NPAIRS]
    print(f"Total: {showstats(counter)}")

    if args.singles:
        singles = []
        for key in counter:
            match key:
                case int():
                    count = counter[key]
                    fraction = count / nopcodes
                    singles.append((count, fraction, key))
        singles.sort(reverse=True)
        print(f"\nTop {args.singles} single opcodes:")
        for count, fraction, op in singles[:args.singles]:
            opname = opcode.opname[op]
            print(f"{opname}: {count:>1,} ({100.0*fraction:.2f}%)")

    if args.pairs:
        pairs = []
        for key in counter:
            match key:
                case (lastop, nextop):
                    count = counter[key]
                    fraction = count/npairs
                    pairs.append((count, fraction, lastop, nextop))
        pairs.sort(reverse=True)
        print(f"\nTop {args.pairs} opcode pairs:")
        for count, fraction, lastop, nextop in pairs[:args.pairs]:
            lastname = opcode.opname[lastop]
            nextname = opcode.opname[nextop]
            print(f"{lastname} => {nextname}: {count:>1,} ({100.0*fraction:.2f}%)")


if __name__ == "__main__":
    main()
