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


def reference(values, valid, n, lo, hi):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    ok = bits & (values >= lo) & (values <= hi)
    return np.packbits(ok, bitorder="little")


def check(n, seed):
    rng = np.random.default_rng(seed)
    nbytes = (n + 7) // 8
    lo = int(rng.integers(-(1 << 62), 1 << 62))
    hi = lo + int(rng.integers(0, 1 << 62))
    values = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    if n > 0:
        mix = rng.integers(0, 4, size=n)
        values[mix == 0] = lo
        values[mix == 1] = hi
        values[mix == 2] = lo + rng.integers(0, max(1, hi - lo), size=int((mix == 2).sum()))
    valid = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    out = cuda.to_device(np.full(nbytes, 0xA5, dtype=np.uint8))
    mod.run(cuda.to_device(values), cuda.to_device(valid), out, n, lo, hi)
    cuda.synchronize()
    got = out.copy_to_host()
    exp = reference(values, valid, n, lo, hi)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("between_bitmap numba: ok")
