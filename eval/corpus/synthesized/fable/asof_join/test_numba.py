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


def reference(left, right, tolerance):
    j = np.searchsorted(right, left, side="right") - 1
    jj = np.maximum(j, 0)
    ok = (j >= 0) & ((left - right[jj]) <= tolerance) if len(right) else np.zeros(len(left), bool)
    return np.where(ok, j, -1).astype(np.int32)


def check(n, r, seed):
    rng = np.random.default_rng(seed)
    span = max(1, 4 * max(n, r))
    left = rng.integers(-span, span, size=n, dtype=np.int64)
    right = np.sort(rng.integers(-span, span, size=r, dtype=np.int64))
    if n > 0 and r > 0:
        left[: n // 4] = right[rng.integers(0, r, size=n // 4)]
    tolerance = int(rng.integers(0, 8))
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int32))
    mod.run(cuda.to_device(left), cuda.to_device(right), d_out, n, r, tolerance)
    cuda.synchronize()
    got = d_out.copy_to_host()
    exp = reference(left, right, tolerance)
    assert np.array_equal(got, exp), f"n={n} r={r}: mismatch at {np.nonzero(got != exp)[0][:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for r in (0, 1, 1000, 1000003):
        check(n, r, n + r)
print("asof_join numba: ok")
