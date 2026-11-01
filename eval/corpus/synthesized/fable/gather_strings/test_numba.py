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
        lengths[rng.integers(0, n, size=5)] = 5000
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    m = n if n < 10 else n + 7
    idx = rng.integers(0, n, size=m, dtype=np.int32) if n > 0 else np.zeros(0, dtype=np.int32)
    out_lengths = offsets[idx + 1] - offsets[idx]
    out_offsets = np.concatenate([[0], np.cumsum(out_lengths)]).astype(np.int32)
    return offsets, chars, idx, out_offsets, m


def reference(offsets, chars, idx, out_offsets, m):
    lengths = (out_offsets[1:] - out_offsets[:-1]).astype(np.int64)
    row = np.repeat(np.arange(m, dtype=np.int64), lengths)
    p = np.arange(int(out_offsets[m]), dtype=np.int64)
    src = offsets[idx[row]].astype(np.int64) + (p - out_offsets[row].astype(np.int64))
    return chars[src]


def check(n, seed):
    offsets, chars, idx, out_offsets, m = make(n, seed)
    total = int(out_offsets[m])
    out = cuda.to_device(np.full(total, 0xA5, dtype=np.uint8))
    mod.run(cuda.to_device(offsets), cuda.to_device(chars), cuda.to_device(idx),
            cuda.to_device(out_offsets), out, n, m)
    cuda.synchronize()
    got = out.copy_to_host()
    exp = reference(offsets, chars, idx, out_offsets, m)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("gather_strings numba: ok")
