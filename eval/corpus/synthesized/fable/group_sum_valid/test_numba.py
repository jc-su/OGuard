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


def reference(group, value, valid, n, g):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    counts = np.zeros(g, dtype=np.int64)
    sums = np.zeros(g, dtype=np.int64)
    np.add.at(counts, group[bits], 1)
    with np.errstate(over="ignore"):
        np.add.at(sums, group[bits], value[bits])
    return counts, sums


def check(n, g, seed):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, g, size=n, dtype=np.int32)
    value = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    valid = rng.integers(0, 256, size=(n + 7) // 8, dtype=np.uint8)
    d_counts = cuda.to_device(np.zeros(g, dtype=np.int64))
    d_sums = cuda.to_device(np.zeros(g, dtype=np.int64))
    mod.run(cuda.to_device(group), cuda.to_device(value), cuda.to_device(valid), d_counts, d_sums, n, g)
    cuda.synchronize()
    counts, sums = reference(group, value, valid, n, g)
    got_c, got_s = d_counts.copy_to_host(), d_sums.copy_to_host()
    assert np.array_equal(got_c, counts), f"n={n} g={g}: counts differ"
    assert np.array_equal(got_s, sums), f"n={n} g={g}: sums differ"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for g in (1, 7, 1024):
        check(n, g, n + g)
print("group_sum_valid numba: ok")
