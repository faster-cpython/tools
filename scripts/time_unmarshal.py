"""Time the cost of unmarshalling many objects vs. the bytecode interpreter."""

import array
import marshal
import time


def main():
    bias = 1000  # Numbers start at bias, bias+1, bias+2, ...
    n = 100_000  # How many numbers
    k = 1000  # How many times to unmarshal
    t = tuple(range(bias, bias + n))
    s = marshal.dumps(t)
    print(f"{n} values, {k} times, {len(s)/n:.3f} bytes per value")
    assert marshal.loads(s) == t
    t0 = time.time()
    for i in range(k):
        marshal.loads(s)
    t1 = time.time()
    dt = t1 - t0
    print(f"Unmarshal: total {dt:.3f} sec; {1e9*dt/n/k:.3f} nsec per value")

    # Now attempt to measure the speed of bytecode to create the same thing
    func = f"def f(x): return x, {repr(t)[1:-1]}"
    ns = {}
    exec(func, ns)
    f = ns["f"]
    t0 = time.time()
    for i in range(k):
        f(i)
    t1 = time.time()
    dt = t1 - t0
    print(f"Code: total {dt:.3f} sec; {1e9*dt/n/k:.3f} nsec per value")

    # And do it with arrays
    a = array.array("L", t)
    assert tuple(a) == t
    t0 = time.time()
    for i in range(k):
        marshal.loads(s)
    t1 = time.time()
    dt = t1 - t0
    print(f"Array: total {dt:.3f} sec; {1e9*dt/n/k:.3f} nsec per value")


main()
