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
    if variant == 0:
        return rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    if variant == 1:  # many ties on the minimum
        return rng.integers(-3, 3, size=n, dtype=np.int64)
    if variant == 2:  # minimum at the end, decreasing values
        return (np.arange(n, dtype=np.int64)[::-1] - n).copy()
    return np.full(n, -(1 << 63), dtype=np.int64)  # all equal to the smallest int64


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values = make(n, variant, rng)
    d_row = cuda.to_device(np.full(1, -1, dtype=np.int32))
    mod.run(cuda.to_device(values), d_row, n)
    cuda.synchronize()
    got = int(d_row.copy_to_host()[0])
    exp = int(np.argmin(values)) if n > 0 else -1
    assert got == exp, f"n={n} variant={variant}: {got} != {exp}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for variant in range(4):
        check(n, variant, n + variant)
print("argmin numba: ok")
