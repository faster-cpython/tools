"""Time interpreter startup."""

import argparse
import os
import statistics
import subprocess
import sys
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("n", nargs="?", type=int, default=0,
                       help="How many dummy modules to import")
argparser.add_argument("-k", type=int, default=10,
                       help="How many runs")
argparser.add_argument("--base", type=float,
                       help="Base time (for computing time per module)")


def make_dummy_modules(n: int):
    names = []
    for i in range(n):
        modname = f"dummy{i}"
        names.append(modname)
        filename = modname + ".py"
        with open(filename, "w") as f:
            for j in range(10):
                f.write(f"class C{j}:\n")
                for k in range(10):
                    f.write(f"    def func{k}(self): pass\n")
    with open("dummymain.py", "w") as f:
        for modname in names:
            f.write(f"import {modname}\n")
        f.write("print()\n")


def delete_dummy_modules(n: int):
    for i in range(n):
        modname = f"dummy{i}"
        filename = modname + ".py"
        os.unlink(filename)
    # os.unlink("dummymain.py")


def time_python(args: list[str]) -> float:
    """Measure time from start of process to first line on stdout.
    
    The command in args must contain a print(), else we wait
    until the process exits (which includes cleanup time).
    """
    t0 = time.time()
    p = subprocess.Popen([sys.executable] + args,stdout=subprocess.PIPE)
    t0a = time.time()
    # print(f"[{t0a-t0:.3f} for subprocess.Popen()]")
    p.stdout.readline()
    t1 = time.time()
    # print(f"[{t1-t0:.3f} until first line]")
    p.stdout.readlines()
    t2 = time.time()
    # print(f"[{t2-t1:.3f} until exit]")
    p.kill()
    p.wait()
    return t1 - t0


def main():
    args = argparser.parse_args()
    n: int = args.n
    k: int = args.k
    base: float = args.base
    print(f"Using {n} dummy modules")
    make_dummy_modules(n)
    dtimes: list[float] = []
    for i in range(-1, k):
        dt = time_python(["-S", "dummymain.py"])
        # print(f"{dt:.3f} sec")
        dtimes.append(dt)
    dt = dtimes.pop(0)
    assert len(dtimes) == k
    print(f"Initial run: {dt:.3f}")
    mean = statistics.mean(dtimes)
    stdev = statistics.stdev(dtimes)
    print(f"runs: {k}, mean: {mean:.3f}, stdev: {stdev:.3f}")
    if base is not None:
        per_mod = (mean - base) / n
        print(f"per mod: {per_mod:.4f}")
    delete_dummy_modules(n)


if __name__ == "__main__":
    main()
