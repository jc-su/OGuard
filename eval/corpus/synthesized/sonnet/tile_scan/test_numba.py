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
    values = rng.integers(-1000, 1001, n).astype(np.int32)
    nt = (n + 1023) // 1024
    pad = np.zeros(nt * 1024, dtype=np.int64)
    pad[:n] = values
    tiles = pad.reshape(nt, 1024)
    ref = (np.cumsum(tiles, axis=1) - tiles).reshape(-1)[:n].astype(np.int32)
    ref_t = tiles.sum(axis=1).astype(np.int32)
    out = sentinel(n, np.int32, 777)
    totals = sentinel(nt, np.int32, 777)
    mod.run(dev(values), out, totals, n)
    return np.array_equal(out.copy_to_host(), ref) and np.array_equal(totals.copy_to_host(), ref_t)


def main():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    bad = 0
    for n in SIZES:
        ok = check(n)
        print(f"n={n} {'PASS' if ok else 'FAIL'}")
        bad += not ok
    sys.exit(1 if bad else 0)


main()
