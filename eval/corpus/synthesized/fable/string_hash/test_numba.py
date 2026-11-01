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


def make(n, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 24, size=n)
    if n > 100:
        lengths[rng.integers(0, n, size=5)] = 3000
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def reference(offsets, chars, n):
    lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)
    h = np.full(n, 2166136261, dtype=np.uint32)
    start = offsets[:-1].astype(np.int64)
    k = 0
    active = np.nonzero(lengths > k)[0]
    while len(active):
        c = chars[start[active] + k].astype(np.uint32)
        h[active] = (h[active] ^ c) * np.uint32(16777619)
        k += 1
        active = active[lengths[active] > k]
    return h


def check(n, seed):
    offsets, chars = make(n, seed)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.uint32))
    mod.run(cuda.to_device(offsets), cuda.to_device(chars), d_out, n)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(offsets, chars, n)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("string_hash numba: ok")
