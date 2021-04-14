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
NERRORS = "__nerrors__"  # Number of files with errors

# TODO: Make this list an option
OF_INTEREST_NAMES = ["BINARY_ADD", "BINARY_SUBTRACT",
                     "INPLACE_ADD", "INPLACE_SUBTRACT"]

of_interest = set(opcode.opmap[x] for x in OF_INTEREST_NAMES)


def all_code_objects(code):
    yield code
    for x in code.co_consts:
        if hasattr(x, 'co_code'):
            yield x


def report(code, filename, counter):
    local = Counter()
    for co in all_code_objects(code):
        co_code = co.co_code
        for i in range(0, len(co_code), 2):
            counter[TOTAL] += 1
            op = co_code[i]
            if op in of_interest:
                counter[op] += 1
                local[op] += 1
    if local:
        print(f"{filename}: {local}")


def file_report(filename, counter):
    try:
        with open(filename, "rb") as f:
            source = f.read()
        code = compile(source, filename, "exec")
        report(code, filename, counter)
        counter[NFILES] += 1
    except Exception as err:
        print(filename + ":", err, file=sys.stderr)
        counter[NERRORS] += 1


def tarball_report(filename, counter):
    print("\nExamining tarball", filename)
    with tarfile.open(filename, "r") as tar:
        members = tar.getmembers()
        for m in members:
            info = m.get_info()
            name = info["name"]
            if name.endswith(".py"):
                try:
                    source = tar.extractfile(m).read()
                    code = compile(source, name, "exec")
                except Exception as err:
                    print(f"{name}: {err}", file=sys.stderr)
                    counter[NERRORS] += 1
                else:
                    counter[NFILES] += 1
                    report(code, name, counter)


def expand_globs(filenames):
    for filename in filenames:
        if "*" in filename and sys.platform == "win32":
            for fn in glob.glob(filename):
                yield fn
        else:
            yield filename


def main(filenames):
    print("Looking for:", ", ".join(OF_INTEREST_NAMES))
    print("In", filenames)
    counter = Counter()
    for filename in expand_globs(filenames):
        if os.path.isfile(filename):
            if filename.endswith(".tar.gz"):
                tarball_report(filename, counter)
            else:
                file_report(filename, counter)
        else:
            for root, dirs, files in os.walk(filename):
                for file in files:
                    if file.endswith(".py"):
                        full = os.path.join(root, file)
                        file_report(full, counter)

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
        print(f"Total opcodes: {total}")
        for name in OF_INTEREST_NAMES:
            key = opcode.opmap[name]
            if key in counter:
                print(f"{opcode.opname[key]}: {counter[key]}",
                      f"({100.0*counter[key]/total:.2f}%)")


if __name__ == "__main__":
    main(sys.argv[1:])
