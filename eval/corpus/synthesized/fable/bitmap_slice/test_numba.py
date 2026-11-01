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


def reference(source, n, offset):
    bits = np.unpackbits(source, bitorder="little")[offset:offset + n]
    return np.packbits(bits, bitorder="little")


def check(n, offset, seed):
    rng = np.random.default_rng(seed)
    src_bytes = (offset + n + 7) // 8
    source = rng.integers(0, 256, size=src_bytes, dtype=np.uint8)
    d_out = cuda.to_device(np.full((n + 7) // 8, 0xA5, dtype=np.uint8))
    mod.run(cuda.to_device(source), d_out, n, offset)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(source, n, offset)
    assert np.array_equal(got, exp), f"n={n} offset={offset}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for offset in (0, 3, 8, 13, 1000001):
        check(n, offset, n + offset)
print("bitmap_slice numba: ok")
