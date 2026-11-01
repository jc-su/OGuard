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
    lens = rng.integers(0, 16, n)
    off = np.concatenate(([0], np.cumsum(lens))).astype(np.int32)
    chars = rng.integers(0, 256, int(off[n])).astype(np.uint8)
    lower = (chars >= 97) & (chars <= 122)
    ref = np.where(lower, chars - 32, chars).astype(np.uint8)
    out = sentinel(len(chars), np.uint8, 0)
    mod.run(dev(off), dev(chars), out, n)
    return np.array_equal(out.copy_to_host(), ref)


def main():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    bad = 0
    for n in SIZES:
        ok = check(n)
        print(f"n={n} {'PASS' if ok else 'FAIL'}")
        bad += not ok
    sys.exit(1 if bad else 0)


main()
