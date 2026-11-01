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


def column(n, rng):
    lengths = rng.integers(0, 16, size=n)
    if n > 100:
        lengths[rng.integers(0, n, size=5)] = 4000
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def reference(a_offsets, a_chars, b_offsets, b_chars, out_offsets, n):
    la = (a_offsets[1:] - a_offsets[:-1]).astype(np.int64)
    lb = (b_offsets[1:] - b_offsets[:-1]).astype(np.int64)
    row = np.repeat(np.arange(n, dtype=np.int64), la + lb)
    p = np.arange(int(out_offsets[n]), dtype=np.int64)
    local = p - out_offsets[row].astype(np.int64)
    from_a = local < la[row]
    src_a = a_offsets[row].astype(np.int64) + local
    src_b = b_offsets[row].astype(np.int64) + (local - la[row])
    return np.where(from_a, a_chars[np.where(from_a, src_a, 0)], b_chars[np.where(from_a, 0, src_b)]) \
        if len(p) else np.zeros(0, dtype=np.uint8)


def check(n, seed):
    rng = np.random.default_rng(seed)
    a_offsets, a_chars = column(n, rng)
    b_offsets, b_chars = column(n, rng)
    out_offsets = (a_offsets + b_offsets).astype(np.int32)
    total = int(out_offsets[n])
    d_out = cuda.to_device(np.full(total, 0xA5, dtype=np.uint8))
    mod.run(cuda.to_device(a_offsets), cuda.to_device(a_chars), cuda.to_device(b_offsets),
            cuda.to_device(b_chars), cuda.to_device(out_offsets), d_out, n)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(a_offsets, a_chars, b_offsets, b_chars, out_offsets, n).astype(np.uint8)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("string_concat numba: ok")
