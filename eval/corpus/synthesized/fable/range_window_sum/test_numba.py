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
    if variant == 0:  # steps of 0 to 3, small range
        ts = np.cumsum(rng.integers(0, 4, size=n, dtype=np.int64))
        rng_ = 5
    elif variant == 1:  # all timestamps equal: every frame is the whole column
        ts = np.full(n, -(1 << 62), dtype=np.int64)
        rng_ = 0
    elif variant == 2:  # distinct timestamps, frames of a few thousand rows
        ts = np.cumsum(rng.integers(1, 3, size=n, dtype=np.int64)) - (1 << 61)
        rng_ = 7000
    else:  # the largest range: frames from the first row
        ts = np.sort(rng.integers(-(1 << 62), (1 << 62), size=n, dtype=np.int64))
        rng_ = 1 << 62
    return ts, values, rng_


def reference(ts, values, range_):
    with np.errstate(over="ignore"):
        prefix = np.concatenate([[0], np.cumsum(values)]).astype(np.int64)
        lo = np.searchsorted(ts, ts - range_, side="left")
        hi = np.searchsorted(ts, ts, side="right")
        return prefix[hi] - prefix[lo]


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    ts, values, range_ = make(n, variant, rng)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int64))
    mod.run(cuda.to_device(ts), cuda.to_device(values), d_out, n, range_)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(ts, values, range_)
    assert np.array_equal(got, exp), f"n={n} variant={variant}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for variant in range(4):
        check(n, variant, n + variant)
print("range_window_sum numba: ok")
