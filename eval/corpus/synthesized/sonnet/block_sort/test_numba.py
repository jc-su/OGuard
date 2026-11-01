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
    ok = True
    for lo, hi in ((-(2**31), 2**31 - 1), (0, 9)):
        keys = rng.integers(lo, hi, n, endpoint=True).astype(np.int32)
        if hi > 100 and n:
            keys[rng.integers(0, n, n // 50 + 1)] = 2**31 - 1
            keys[rng.integers(0, n, n // 50 + 1)] = -(2**31)
        nt = (n + 1023) // 1024
        pad = np.full(nt * 1024, 2**31 - 1, dtype=np.int32)
        pad[:n] = keys
        ref = np.sort(pad.reshape(nt, 1024), axis=1).reshape(-1)[:n]
        out = sentinel(n, np.int32, 777)
        mod.run(dev(keys), out, n)
        ok &= np.array_equal(out.copy_to_host(), ref)
    return ok


def main():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    bad = 0
    for n in SIZES:
        ok = check(n)
        print(f"n={n} {'PASS' if ok else 'FAIL'}")
        bad += not ok
    sys.exit(1 if bad else 0)


main()
