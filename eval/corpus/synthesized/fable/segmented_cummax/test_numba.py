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
        starts = rng.random(n) < 0.3
    elif variant == 1:  # one partition
        starts = np.zeros(n, dtype=bool)
    elif variant == 2:  # every row its own partition
        starts = np.ones(n, dtype=bool)
    else:  # long partitions crossing tile boundaries, increasing values
        starts = (np.arange(n) % 50001) == 0
        values = np.arange(n, dtype=np.int64) - (1 << 62)
    if n > 0:
        starts[0] = True
    idx = np.arange(n, dtype=np.int64)
    first = np.maximum.accumulate(np.where(starts, idx, 0)).astype(np.int32)
    return values, first


def reference(values, first, n):
    m = values.copy()
    idx = np.arange(n, dtype=np.int64)
    d = 1
    while d < n:
        ok = (idx[d:] - d) >= first[d:]
        m[d:] = np.where(ok, np.maximum(m[d:], m[:-d]), m[d:])
        d *= 2
    return m


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values, first = make(n, variant, rng)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int64))
    mod.run(cuda.to_device(values), cuda.to_device(first), d_out, n)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(values, first, n)
    assert np.array_equal(got, exp), f"n={n} variant={variant}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for variant in range(4):
        check(n, variant, n + variant)
print("segmented_cummax numba: ok")
