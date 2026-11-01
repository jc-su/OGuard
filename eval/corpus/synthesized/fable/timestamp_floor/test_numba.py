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


def reference(ts, unit):
    return ts - np.mod(ts, unit)  # numpy's mod has the sign of the divisor: floor semantics


def make(n, rng):
    ts = rng.integers(-(1 << 62), (1 << 62) + 1, size=n, dtype=np.int64)
    if n > 0:
        k = n // 2
        ts[:k] = rng.integers(-(1 << 44), 1 << 44, size=k)
        ts[0] = -(1 << 62)
        if n > 1:
            ts[1] = 1 << 62
    return ts


def check(n, unit, seed):
    rng = np.random.default_rng(seed)
    ts = make(n, rng)
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int64))
    mod.run(cuda.to_device(ts), d_out, n, unit)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(ts, unit)
    assert np.array_equal(got, exp), f"n={n} unit={unit}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for unit in (1, 7, 3600000000, 86400000000, 1 << 40):
        check(n, unit, n + unit % 1000)
print("timestamp_floor numba: ok")
