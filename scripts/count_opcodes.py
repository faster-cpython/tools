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
from importlib.util import MAGIC_NUMBER

# Magic number at which point jump targets start
# counting in (2-byte) instructions instead of bytes
JUMP_TARGETS_IN_INSTRS = (3435).to_bytes(2, 'little') + b'\r\n'

FOR_ITER = opcode.opmap["FOR_ITER"]

# Special (non-opcode) counter keys
NOPCODES = "__nopcodes__"  # Number of opcodes
NPAIRS = "__npairs__"  # Number of opcode pairs
NFILES = "__nfiles__"  # Number of files
NLINES = "__nlines__"  # Number of lines
NCODEOBJS = "__ncodeobjs__"  # Number of code objects
NERRORS = "__nerrors__"  # Number of files with errors
CACHE_SIZE = "__cache_size__" # quickening cache size
CACHE_WASTED = "__cache_wasted__" # Number of wasted cache entries
OPS_QUICKENED = "__ops_quickened__" # Number of ops that were quickened
SKIP_QUICKEN = "__skip_quicken__" # ops not quickened for lack of cache space
PREV_EXTENDED = "__prev_extended__" # skipped quickening as prev is extended

SHOW_ITEMS = [
    (NERRORS, "errors"),
    (NFILES, "files"),
    (NCODEOBJS, "code objects"),
    (NLINES, "lines"),
    (NOPCODES, "opcodes"),
    (NPAIRS, "opcode pairs"),
    (CACHE_SIZE, "cache_size"),
    (CACHE_WASTED, "cache wasted"),
    (OPS_QUICKENED, "ops quickened"),
    (SKIP_QUICKEN, "skip quicken"),
    (PREV_EXTENDED, "prev extended args")
]

# TODO: Make this list an option
CACHE_ENTRIES = {
    "LOAD_GLOBAL": 2,
    "LOAD_ATTR": 2,
    "CALL_FUNCTION": 2,
    "CALL_FUNCTION_KW": 2,
    "CALL_FUNCTION_EX": 2,  # Unsure
    "CALL_METHOD": 2,
    "CALL_METHOD_KW": 2,
}
OF_INTEREST_NAMES = CACHE_ENTRIES.keys()

of_interest = set(opcode.opmap[x] for x in OF_INTEREST_NAMES if x in opcode.opmap)


def all_code_objects(code):
    yield code
    for x in code.co_consts:
        if hasattr(x, 'co_code'):
            yield from all_code_objects(x)


def showstats(counter):
    res = []
    for key, name in SHOW_ITEMS:
        if key in counter:
            res.append(f"{counter[key]:>1,} {name}")
    return "; ".join(res)


def find_loops(co):
    loops = []
    extra = 0
    for i in range(0, len(co), 2):
        op = co[i]
        oparg = extra*256 + co[i+1]
        if op == opcode.EXTENDED_ARG:
            extra = oparg
            continue
        extra = 0
        if op == FOR_ITER:
            # Jumps used to count bytes, now they count opcodes
            if MAGIC_NUMBER >= JUMP_TARGETS_IN_INSTRS:
                oparg *= 2
            loops.append(range(i, i + 2 + oparg))  # Count from *next* opcode
    return loops


class CacheCounter:
    def __init__(self, counter):
        self.counter = counter
        self.offset = 0
        self.prev_is_extended_arg = False

    def update_offset(self, op, index):
        need = CACHE_ENTRIES.get(opcode.opname[op], 0)
        if need == 0:
            return
        oparg = self.offset - index // 2
        if oparg < 0:
            self.counter[CACHE_WASTED] += 0 - oparg
            oparg == 0
            self.offset = index // 2
        elif oparg > 255:
            self.counter[SKIP_QUICKEN] += 1
            return
        self.counter[OPS_QUICKENED] += 1
        self.offset += need

    def next_op(self, op, index):
        if not self.prev_is_extended_arg:
            self.update_offset(op, index)
        else:
            self.counter[PREV_EXTENDED] += 1
        self.prev_is_extended_arg = opcode.opname[op] == "EXTENDED_ARG"

    def end_code_block(self):
        self.counter[CACHE_SIZE] += self.offset


class Reporter:

    def report(self, source, filename, verbose, bias):
        counter = Counter()
        try:
            code = compile(source, filename, "exec")
        except Exception as err:
            if verbose > 0:
                print(f"{filename}: {err}")
            counter[NERRORS] += 1
            return counter

        for co in all_code_objects(code):
            cache_counter = CacheCounter(counter)
            counter[NCODEOBJS] += 1
            co_code = co.co_code
            loops = find_loops(co_code)
            for i in range(0, len(co_code), 2):
                inloops = sum(i in loop for loop in loops)
                count = 1 + bias*inloops  # Opcodes in loops are counted more
                counter[NOPCODES] += count
                op = co_code[i]
                counter[op] += count
                cache_counter.next_op(op, i/2)
                if i > 0:
                    lastop = co_code[i-2]
                    counter[NPAIRS] += count
                    counter[(lastop, op)] += count
            cache_counter.end_code_block()

        counter[NLINES] += len(source.splitlines())
        if verbose > 0:
            print(f"{filename}: {showstats(counter)}")
        counter[NFILES] += 1
        return counter


    def file_report(self, filename, verbose, bias):
        try:
            with open(filename, "rb") as f:
                source = f.read()
            counter = self.report(source, filename, verbose, bias)
        except Exception as err:
            if verbose > 0:
                print(filename + ":", err)
            counter = Counter()
            counter[NERRORS] += 1
        return counter


    def tarball_report(self, filename, verbose, bias):
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
                        counter.update(self.report(source, name, verbose-1, bias))
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
argparser.add_argument("--bias", type=int,
                       help="Add bias for opcodes inside for-loops")
argparser.add_argument("--cache-needs", action="store_true",
                       help="Show fraction of cache entries needed per opcode ")
argparser.add_argument("filenames", nargs="*", metavar="FILE",
                       help="files, directories or tarballs to count")


def main():
    args = argparser.parse_args()
    verbose = 1 + args.verbose - args.quiet
    bias = args.bias or 0
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
    reporter = Reporter()
    hits = 0
    for filename in expand_globs(filenames):
        hits += 1
        if os.path.isfile(filename):
            if filename.endswith(".tar.gz"):
                ctr = reporter.tarball_report(filename, verbose, bias)
            else:
                ctr = reporter.file_report(filename, verbose, bias)
            counter.update(ctr)
        elif os.path.isdir(filename):
            for root, dirs, files in os.walk(filename):
                for file in files:
                    if file.endswith(".py"):
                        full = os.path.join(root, file)
                        counter.update(reporter.file_report(full, verbose, bias))
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

    if args.cache_needs:
        print()
        print("Future cache needs")
        ncache = 0
        for key in counter:
            if key in of_interest:
                name = opcode.opname[key]
                need = CACHE_ENTRIES[name] * counter[key]
                ncache += need
                print(name, need)
        nops = counter[NOPCODES]
        print(f"{nops} opcodes, {2*nops} bytes,",
              f"{ncache} cache entries, {8*ncache} bytes,",
              f"{ncache/nops:.2f} ncache/nops")

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
            print(f"{opname:<20s} {count:>10,} {100.0*fraction:6.2f}%")

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
            print(f"{lastname:<20s} --> {nextname:<20s}",
                  f"{count:>10,} {100.0*fraction:6.2f}%")


if __name__ == "__main__":
    main()
