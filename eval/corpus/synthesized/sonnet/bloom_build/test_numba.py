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


H1 = np.uint64(0x9E3779B97F4A7C15)
H2 = np.uint64(0xC2B2AE3D27D4EB4F)


def build_ref(keys, words):
    u = keys.astype(np.uint64)
    ref = np.zeros(words, dtype=np.uint32)
    for h in (H1, H2):
        p = ((u * h) >> np.uint64(32)) % np.uint64(32 * words)
        np.bitwise_or.at(ref, (p >> np.uint64(5)).astype(np.int64),
                         (np.uint32(1) << (p & np.uint64(31)).astype(np.uint32)))
    return ref


def check(n):
    rng = np.random.default_rng(n + 1)
    keys = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
    ok = True
    for words in (1, 64, 4096, 1 << 16):
        bits = dev(np.zeros(words, dtype=np.uint32))
        mod.run(dev(keys), bits, n, words)
        ok &= np.array_equal(bits.copy_to_host(), build_ref(keys, words))
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
