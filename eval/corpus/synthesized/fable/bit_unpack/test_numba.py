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


def reference(packed, n, bits):
    stream = np.unpackbits(packed.view(np.uint8), bitorder="little")[: n * bits].reshape(n, bits)
    weights = (np.uint64(1) << np.arange(bits, dtype=np.uint64))
    return (stream.astype(np.uint64) * weights).sum(axis=1).astype(np.uint32)


def check(n, bits, seed):
    rng = np.random.default_rng(seed)
    w = (n * bits + 31) // 32
    packed = rng.integers(0, 1 << 32, size=w, dtype=np.uint64).astype(np.uint32)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.uint32))
    mod.run(cuda.to_device(packed), d_out, n, w, bits)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(packed, n, bits)
    assert np.array_equal(got, exp), f"n={n} bits={bits}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for bits in (1, 3, 7, 8, 13, 31, 32):
        check(n, bits, n + bits)
print("bit_unpack numba: ok")
