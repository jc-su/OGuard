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
    lens_a = rng.integers(0, 12, n)
    mode = rng.integers(0, 4, n)  # 0 equal, 1 last byte differs, 2 longer, 3 independent
    lens_b = lens_a + (mode == 2)
    off_a = np.concatenate(([0], np.cumsum(lens_a))).astype(np.int32)
    off_b = np.concatenate(([0], np.cumsum(lens_b))).astype(np.int32)
    ta, tb = int(off_a[n]), int(off_b[n])
    chars_a = rng.integers(97, 100, ta).astype(np.uint8)
    chars_b = rng.integers(97, 100, tb).astype(np.uint8)
    row_b = np.repeat(np.arange(n), lens_b)
    pos_b = np.arange(tb) - off_b[row_b]
    cp = (mode[row_b] != 3) & (pos_b < lens_a[row_b])
    chars_b[cp] = chars_a[off_a[row_b][cp] + pos_b[cp]]
    r1 = np.nonzero((mode == 1) & (lens_a > 0))[0]
    chars_b[off_b[r1 + 1] - 1] ^= 1
    ref = (lens_a == lens_b)
    row_a = np.repeat(np.arange(n), lens_a)
    if ta:
        pos_a = np.arange(ta) - off_a[row_a]
        bad = ref[row_a] & (chars_a != chars_b[off_b[row_a] + pos_a])
        ref[row_a[bad]] = False
    ref = ref.astype(np.int8)
    out = sentinel(n, np.int8, 5)
    mod.run(dev(off_a), dev(chars_a), dev(off_b), dev(chars_b), out, n)
    return np.array_equal(out.copy_to_host(), ref) and (n < 1000 or 0 < ref.sum() < n)


def main():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    bad = 0
    for n in SIZES:
        ok = check(n)
        print(f"n={n} {'PASS' if ok else 'FAIL'}")
        bad += not ok
    sys.exit(1 if bad else 0)


main()
