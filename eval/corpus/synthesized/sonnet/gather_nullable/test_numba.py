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
    m = n + 5
    col = rng.standard_normal(n)
    idx = rng.integers(-1, n, m).astype(np.int32)
    ref = np.full(m, np.nan)
    ok = idx >= 0
    ref[ok] = col[idx[ok]]
    out = sentinel(m, np.float64, -12345.0)
    mod.run(dev(col), dev(idx), out, m, n)
    return np.array_equal(out.copy_to_host(), ref, equal_nan=True)


def main():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    bad = 0
    for n in SIZES:
        ok = check(n)
        print(f"n={n} {'PASS' if ok else 'FAIL'}")
        bad += not ok
    sys.exit(1 if bad else 0)


main()
