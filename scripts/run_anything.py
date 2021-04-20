# Run some other program, saving DXP data

import json
import os
import runpy
import sys


def dump_dxp(dxp, name):
    name = os.path.basename(name)
    if name.endswith(".py"):
        name = name[:-3]
    data = json.dumps(dxp)
    os.makedirs("data", exist_ok=True)
    datafile = os.path.join("data", name + ".json")
    with open(datafile, "w") as f:
        f.write(data + "\n")
    print("JSON data written to", datafile)


def main():
    args = sys.argv[1:]
    if not args:
        print(f"Usage:")
        print(f"    {sys.argv[0]} SCRIPTFILE [ARG ...]")
        print(f"    {sys.argv[0]} -m MODULE [ARG ...]")
        if not hasattr(sys, "getdxp"):
            print("NOTE: This Python is is not compiled with dxp enabled")
        sys.exit(0)

    if not hasattr(sys, "getdxp"):
        sys.exit("This Python is not compiled with dxp enabled")

    if len(args) >= 2 and args[0] == "-m":
        module = args[1]
        sys.argv[1:] = args[2:]
        print(f"Running module {module} with args: {sys.argv[1:]}")
        try:
            sys.getdxp()  # Reset
            runpy.run_module(module, run_name="__main__")
        finally:
            dxp = sys.getdxp()
            dump_dxp(dxp, module)
    else:
        file = args[0]
        sys.argv[1:] = args[1:]
        print(f"Running file {file} with args: {sys.argv[1:]}")
        try:
            sys.getdxp()  # Reset
            runpy.run_path(file, run_name="__main__")
        finally:
            dxp = sys.getdxp()
            dump_dxp(dxp, file)


if __name__ == "__main__":
    main()
