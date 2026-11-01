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
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    valid = rng.integers(0, 256, (n + 7) // 8, dtype=np.uint8)
    ref = np.where(np.unpackbits(valid, bitorder="little")[:n].astype(bool), a, b)
    out = sentinel(n, np.float64, -12345.0)
    mod.run(dev(a), dev(valid), dev(b), out, n)
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
