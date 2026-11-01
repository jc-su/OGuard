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


def make(n, g, variant, rng):
    group = rng.integers(0, g, size=n, dtype=np.int32)
    if variant == 0:
        value = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    elif variant == 1:  # few distinct values: many ties
        value = rng.integers(-2, 2, size=n, dtype=np.int64)
    else:  # increasing values: every row improves on the published one
        value = np.arange(n, dtype=np.int64)
    return group, value


def reference(group, value, n, g):
    best = np.full(g, -1, dtype=np.int32)
    if n:
        row = np.arange(n, dtype=np.int64)
        order = np.lexsort((-row, value, group))
        sg = group[order]
        last = np.concatenate([np.nonzero(sg[1:] != sg[:-1])[0], [n - 1]])
        best[sg[last]] = order[last]
    return best


def check(n, g, variant, seed):
    rng = np.random.default_rng(seed)
    group, value = make(n, g, variant, rng)
    d_best = cuda.to_device(np.full(g, -1, dtype=np.int32))
    mod.run(cuda.to_device(group), cuda.to_device(value), d_best, n, g)
    cuda.synchronize()
    got = d_best.copy_to_host()
    exp = reference(group, value, n, g)
    assert np.array_equal(got, exp), f"n={n} g={g} variant={variant}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for g in (1, 7, 1000, 1 << 20):
        for variant in range(3):
            check(n, g, variant, n + g + variant)
print("group_argmax numba: ok")
