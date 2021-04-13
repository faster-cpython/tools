import opcode
import os
import sys
import tokenize
from collections import Counter

TOTAL = "__total__"  # Counter key for total count
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


def get_code(filename):
    try:
        with open(filename, "rb") as f:
            source = f.read()
        code = compile(source, filename, "exec")
        return code
    except Exception as err:
        print(filename + ":", err, file=sys.stderr)
        return None


def main(filename):
    print("Looking for:", ", ".join(OF_INTEREST_NAMES))
    counter = Counter()
    nfiles = 0
    nerrors = 0
    if not os.path.isdir(filename):
        code = get_code(filename)
        if code is not None:
            report(code, filename, counter)
            nfiles += 1
        else:
            nerrors += 1
    else:
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.endswith(".py"):
                    full = os.path.join(root, file)
                    code = get_code(full)
                    if code is not None:
                        report(code, full, counter)
                        nfiles += 1
                    else:
                        nerrors += 1

    if nerrors:
        print(f"Errors reading or compiling {nerrors} files")
    if nfiles:
        print(f"Compiled {nfiles} files")
    else:
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
    main(sys.argv[1])
