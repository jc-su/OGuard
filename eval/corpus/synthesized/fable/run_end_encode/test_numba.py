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
    if variant == 0:  # runs of geometric length
        starts = rng.random(n) < 0.3
        starts[:1] = True
        return np.cumsum(starts).astype(np.int64) * 7 - 3
    if variant == 1:  # one run for the whole column
        return np.full(n, -5, dtype=np.int64)
    if variant == 2:  # every row its own run
        return np.arange(n, dtype=np.int64)
    # long runs that cross tile boundaries
    return (np.arange(n, dtype=np.int64) // 50001) % 3


def reference(values, n):
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    ends = np.concatenate([np.nonzero(values[1:] != values[:-1])[0] + 1, [n]]).astype(np.int32)
    return ends, values[ends.astype(np.int64) - 1]


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values = make(n, variant, rng)
    ends0 = np.full(n, -7, dtype=np.int32)
    vals0 = np.full(n, -9, dtype=np.int64)
    d_ends = cuda.to_device(ends0)
    d_vals = cuda.to_device(vals0)
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    mod.run(cuda.to_device(values), d_ends, d_vals, d_count, n)
    cuda.synchronize()
    ends, vals = reference(values, n)
    r = len(ends)
    got_count = int(d_count.copy_to_host()[0])
    got_ends, got_vals = d_ends.copy_to_host(), d_vals.copy_to_host()
    assert got_count == r, f"n={n} variant={variant}: count {got_count} != {r}"
    assert np.array_equal(got_ends[:r], ends), f"n={n} variant={variant}: run_ends differ"
    assert np.array_equal(got_vals[:r], vals), f"n={n} variant={variant}: run_values differ"
    assert np.array_equal(got_ends[r:], ends0[r:]) and np.array_equal(got_vals[r:], vals0[r:]), \
        f"n={n} variant={variant}: tail changed"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for variant in range(4):
        check(n, variant, n + variant)
print("run_end_encode numba: ok")
