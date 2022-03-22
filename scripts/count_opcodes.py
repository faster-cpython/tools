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
NITEMS = "__nitems__"  # Number of constants/names
NFILES = "__nfiles__"  # Number of files
NLINES = "__nlines__"  # Number of lines
NCODEOBJS = "__ncodeobjs__"  # Number of code objects
NERRORS = "__nerrors__"  # Number of files with errors
CACHE_SIZE = "__cache_size__" # quickening cache size
CACHE_WASTED = "__cache_wasted__" # Number of wasted cache entries
OPS_QUICKENED = "__ops_quickened__" # Number of ops that were quickened
SKIP_QUICKEN = "__skip_quicken__" # ops not quickened for lack of cache space
PREV_EXTENDED = "__prev_extended__" # skipped quickening as prev is extended
NSTORE_FAST = "__nstore_fast__"  # Number of STORE_FAST opcodes
NSTORE_NONE_FAST = "__nstore_none_fast__"  # Number of STORE_FAST preceded by LOAD_CONST(None)
CO_CONSTS_SIZE = "__co_consts_size__"
CO_CONSTS_NUM = "__co_consts_num__"
NUM_JUMP_ABS = "__num_jump_abs__"
NUM_JUMP_REL = "__num_jump_rel__"
NUM_JUMP_ABS_EXT = "__num_jump_abs_extended__"
NUM_JUMP_REL_EXT = "__num_jump_rel_extended__"
NUM_JUMP_ABS_BACKWARDS = "__num_jump_abs_backwards__"
NUM_JUMP_ABS_BACKWARDS_EXT = "__num_jump_abs_extended_backwards__"
NUM_SHORT_ABS_JUMPS = "__num_short_abs_jumps__"

SHOW_ITEMS = [
    (NERRORS, "errors"),
    (NFILES, "files"),
    (NCODEOBJS, "code objects"),
    (NLINES, "lines"),
    (NOPCODES, "opcodes"),
    (NPAIRS, "opcode pairs"),
    (NITEMS, "constants"),
    (CACHE_SIZE, "cache_size"),
    (CACHE_WASTED, "cache wasted"),
    (OPS_QUICKENED, "ops quickened"),
    (SKIP_QUICKEN, "skip quicken"),
    (PREV_EXTENDED, "prev extended args"),
    (CO_CONSTS_SIZE, "total size of co_consts"),
    (CO_CONSTS_NUM, "number of co_consts"),
    (NUM_JUMP_ABS, "number of absolute jumps"),
    (NUM_JUMP_REL, "number of relative jumps"),
    (NUM_JUMP_ABS_EXT, "number of absolute jumps with extended args"),
    (NUM_JUMP_REL_EXT, "number of relative jumps with extended args"),
    (NUM_JUMP_ABS_BACKWARDS, "number of absolute jumps backwards"),
    (NUM_JUMP_ABS_BACKWARDS_EXT, "number of absolute jumps backwards with extended args"),
    (NUM_SHORT_ABS_JUMPS, "number of absolute jumps with delta < 256"),
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

    def reporting_guts(self, counter, co, bias):
        cache_counter = CacheCounter(counter)
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

    def report(self, source, filename, verbose, bias):
        counter = Counter()
        try:
            code = compile(source, filename, "exec")
        except Exception as err:
            if verbose > 0:
                print(f"{filename}: {err}")
            counter[NERRORS] += 1
            return counter

        const_ids = set()
        for co in all_code_objects(code):
            counter[NCODEOBJS] += 1
            if id(co.co_consts) not in const_ids:
                const_ids.add(id(co.co_consts))
                counter[CO_CONSTS_SIZE] += len(co.co_consts)
            self.reporting_guts(counter, co, bias)
        counter[CO_CONSTS_NUM] = len(const_ids)

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


class ConstantsReporter(Reporter):

    def reporting_guts(self, counter, co, bias):
        for const in co.co_consts:
            counter[NITEMS] += 1
            key = f"!{const!r}"
            counter[key] += 1


STORE_FAST = opcode.opmap["STORE_FAST"]
LOAD_CONST = opcode.opmap["LOAD_CONST"]

class StoreNoneReporter(Reporter):

    def reporting_guts(self, counter, co, bias):
        co_code = co.co_code
        for i in range(0, len(co_code), 2):
            counter[NOPCODES] += 1
            op = co_code[i]
            if op == STORE_FAST:
                counter[NSTORE_FAST] += 1
                if i > 0:
                    prevop = co_code[i-2]
                    if prevop == LOAD_CONST:
                        prevoparg = co_code[i-1]
                        val = co.co_consts[prevoparg]
                        if val is None:
                            counter[NSTORE_NONE_FAST] += 1

class NamesReporter(Reporter):

    def reporting_guts(self, counter, co, bias):
        for attr in [co.co_names, co.co_varnames, co.co_cellvars, co.co_freevars]:
            for name in attr:
                counter[NITEMS] += 1
                key = f"!{name}"
                counter[key] += 1

class JumpsReporter(Reporter):

    def reporting_guts(self, counter, co, bias):
        co_code = co.co_code
        extra = 0
        for i in range(0, len(co_code), 2):
            counter[NOPCODES] += 1
            op = co_code[i]
            oparg = extra*256 + co_code[i+1]
            if op == opcode.EXTENDED_ARG:
                extra = oparg
                continue
            extended = extra > 0
            extra = 0
            if op in opcode.hasjabs:
                counter[NUM_JUMP_ABS] += 1
                if extended:
                    counter[NUM_JUMP_ABS_EXT] += 1
                target = 2 * oparg
                if target < i:
                    counter[NUM_JUMP_ABS_BACKWARDS] += 1
                    if extended:
                        counter[NUM_JUMP_ABS_BACKWARDS_EXT] += 1
                if abs(target-i)//2 < 256:
                    counter[NUM_SHORT_ABS_JUMPS] += 1
            if op in opcode.hasjrel:
                counter[NUM_JUMP_REL] += 1
                if extended:
                    counter[NUM_JUMP_REL_EXT] += 1


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
argparser.add_argument("--singles", type=int, metavar="N",
                      help="show N most common opcodes")
argparser.add_argument("--pairs", type=int, metavar="N",
                      help="show N most common opcode pairs")
argparser.add_argument("--constants", type=int, metavar="N",
                       help="Show N most common constants")
argparser.add_argument("--names", type=int, metavar="N",
                       help="Show N most common names")
argparser.add_argument("--jumps", action="store_true",
                      help="counts jumps with and without extended args")
argparser.add_argument("--bias", type=int,
                       help="Add bias for opcodes inside for-loops")
argparser.add_argument("--cache-needs", action="store_true",
                       help="Show fraction of cache entries needed per opcode")
argparser.add_argument("--store-none", action="store_true",
                       help="count frequency of LOAD_CONST(None) + STORE_FAST")
argparser.add_argument("filenames", nargs="*", metavar="FILE",
                       help="files, directories or tarballs to count")


def main():
    args = argparser.parse_args()
    verbose = 1 + args.verbose - args.quiet
    bias = args.bias or 0
    if not (args.pairs or args.singles or args.constants or args.names or args.store_none):
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
    if args.store_none:
        reporter = StoreNoneReporter()
    elif args.constants:
        reporter = ConstantsReporter()
    elif args.names:
        reporter = NamesReporter()
    elif args.jumps:
        reporter = JumpsReporter()
    else:
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

    if args.store_none:
        total = counter[NOPCODES]
        value1 = counter[NSTORE_FAST]
        value2 = counter[NSTORE_NONE_FAST]
        fraction1 = value1 / total
        fraction2 = value2 / value1
        label1 = "Opcodes:"
        label2 = "STORE_FAST:"
        label3 = "STORE_FAST preceded by LOAD_CONST(None):"
        print(f"{label1:>40} {total:10,}")
        print(f"{label2:>40} {value1:10,} ({fraction1*100:.2g}%)")
        print(f"{label3:>40} {value2:10,} ({fraction2*100:.2g}%)")

    if args.constants or args.names:
        num = args.constants or args.names
        label = "constants" if args.constants else "names"
        print()
        nitems = counter[NITEMS]
        print(f"Top {num} {label}:")
        pairs = [(key[1:], value) for key, value in counter.items()
                                   if key.startswith("!")]
        pairs.sort(reverse=True, key=lambda a: a[1])
        for key, value in pairs[:num]:
            fraction = value / nitems
            print(f"{key:>40s} : {value:10d} ({100.0*fraction:.2f}%)")

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
