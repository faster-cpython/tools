# Count interesting opcodes in a collection of source code files

# TODO:
# - Support counting opcode pairs
# - Support counting special occurrences like
#   LOAD_CONST <small int> followed by BINARY_ADD
# - Command-line parsing

import glob
import opcode
import os
import sys
import tarfile
import tokenize
from collections import Counter

# Special (non-opcode) counter keys
TOTAL = "__total__"  # Total count
NFILES = "__nfiles__"  # Number of analyzed files
NLINES = "__nlines__"  # Number of analyzed lines
NERRORS = "__nerrors__"  # Number of files with errors

# TODO: Make this list an option
OF_INTEREST_NAMES = [x for x in opcode.opname if not x.startswith("<")]
# OF_INTEREST_NAMES = ["BINARY_ADD", "BINARY_SUBTRACT",
#                      "INPLACE_ADD", "INPLACE_SUBTRACT"]

of_interest = set(opcode.opmap[x] for x in OF_INTEREST_NAMES)


def all_code_objects(code):
    yield code
    for x in code.co_consts:
        if hasattr(x, 'co_code'):
            yield x


def report(source, filename):
    counter = Counter()
    try:
        code = compile(source, filename, "exec")
    except Exception as err:
        printf(f"{filename}: {err}")
        counter[NERRORS] += 1
        return counter
    for co in all_code_objects(code):
        co_code = co.co_code
        for i in range(0, len(co_code), 2):
            counter[TOTAL] += 1
            op = co_code[i]
            if op in of_interest:
                counter[op] += 1
                counter[op] += 1
    counter[NFILES] += 1
    counter[NLINES] += len(source.splitlines())
    if counter[TOTAL]:
        print(f"{filename}: {counter[NLINES]} lines, {counter[TOTAL]} opcodes")
    return counter


def file_report(filename):
    try:
        with open(filename, "rb") as f:
            source = f.read()
        counter = report(source, filename)
    except Exception as err:
        print(filename + ":", err)
        counter = Counter()
        counter[NERRORS] += 1
    return counter


def tarball_report(filename):
    print("\nExamining tarball", filename)
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
                    print(f"{name}: {err}")
                    counter[NERRORS] += 1
                else:
                    counter.update(report(source, name))
    return counter


def expand_globs(filenames):
    for filename in filenames:
        if "*" in filename and sys.platform == "win32":
            for fn in glob.glob(filename):
                yield fn
        else:
            yield filename


def main(filenames):
    # print("Looking for:", ", ".join(OF_INTEREST_NAMES))
    print("In", filenames)
    counter = Counter()
    for filename in expand_globs(filenames):
        if os.path.isfile(filename):
            if filename.endswith(".tar.gz"):
                ctr = tarball_report(filename)
            else:
                ctr = file_report(filename)
            counter.update(ctr)
        else:
            for root, dirs, files in os.walk(filename):
                for file in files:
                    if file.endswith(".py"):
                        full = os.path.join(root, file)
                        counter.update(file_report(full))

    nerrors = counter[NERRORS]
    nfiles = counter[NFILES]
    if nerrors:
        print(f"Errors reading or compiling {nerrors} files")
    if nfiles:
        print(f"Analyzed {nfiles} files")
    else:
        print("No files analyzed")
        return
    
    if not counter:
        print("No opcodes!")
    elif TOTAL not in counter:
        print("No total?!")
    elif not counter[TOTAL]:
        print("Zero total?!")
    else:
        total = counter[TOTAL]
        print(f"Total opcodes: {total}; total lines: {counter[NLINES]}")
        for name in OF_INTEREST_NAMES:
            key = opcode.opmap[name]
            if key in counter:
                print(f"{opcode.opname[key]}: {counter[key]}",
                      f"({100.0*counter[key]/total:.2f}%)")


if __name__ == "__main__":
    main(sys.argv[1:])
