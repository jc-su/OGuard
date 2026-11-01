import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory would shadow the numba package
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1024, 1025, 1000003)
dev = cuda.to_device


def sentinel(n, dtype, value):
    return dev(np.full(n, value, dtype=dtype))


def check(n):
    rng = np.random.default_rng(n + 1)
    values = rng.standard_normal(n)
    lens = rng.integers(1, 3000, n // 500 + 2)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    first = np.repeat(starts, lens)[:n].astype(np.int32)
    ok_all = True
    for k in (1, 3, 1024):
        j = np.arange(n) - k
        ref = np.full(n, np.nan)
        ok = j >= first
        ref[ok] = values[j[ok]]
        out = sentinel(n, np.float64, -12345.0)
        mod.run(dev(values), dev(first), out, n, k)
        ok_all &= np.array_equal(out.copy_to_host(), ref, equal_nan=True)
    return ok_all


def main():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    bad = 0
    for n in SIZES:
        ok = check(n)
        print(f"n={n} {'PASS' if ok else 'FAIL'}")
        bad += not ok
    sys.exit(1 if bad else 0)


main()
