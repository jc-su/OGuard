import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make(n, variant, rng):
    values = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    if variant == 0:
        bits = rng.random(n) < 0.5
    elif variant == 1:  # all null
        bits = np.zeros(n, dtype=bool)
    elif variant == 2:  # all valid
        bits = np.ones(n, dtype=bool)
    elif variant == 3:  # one valid row, then a null run as long as the column
        bits = np.zeros(n, dtype=bool)
        if n > 0:
            bits[0] = True
    else:  # sparse valid rows at tile-unaligned positions
        bits = (np.arange(n) % 50001) == 3
    valid = np.packbits(bits, bitorder="little")
    if n % 8 and len(valid):  # junk in the padding bits must be ignored
        valid[-1] |= np.uint8((0xFF << (n % 8)) & 0xFF)
    return values, valid


def reference(values, valid, n):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    idx = np.where(bits, np.arange(n, dtype=np.int64), -1)
    last = np.maximum.accumulate(idx) if n else idx
    return np.where(last >= 0, values[np.maximum(last, 0)], 0)


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values, valid = make(n, variant, rng)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int64))
    mod.run(cuda.to_device(values), cuda.to_device(valid), d_out, n)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(values, valid, n)
    assert np.array_equal(got, exp), f"n={n} variant={variant}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for variant in range(5):
        check(n, variant, n + variant)
print("fill_forward numba: ok")
