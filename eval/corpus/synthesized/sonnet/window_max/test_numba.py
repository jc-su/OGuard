import importlib.util
import pathlib

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", pathlib.Path(__file__).with_name("numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cuda.compile_ptx(mod.kernel, mod.TYPES)
rng = np.random.default_rng(1)
def ref_window_max(v, w):
    n = len(v)
    if n == 0:
        return v.copy()
    # sparse-table style doubling: exact and O(n log w)
    out = v.copy()
    span = 1
    while span < w:
        step = min(span, w - span)
        shifted = np.concatenate([np.full(step, np.iinfo(np.int64).min, dtype=np.int64), out[:-step]]) if step < n else np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
        out = np.maximum(out, shifted)
        span += step
    return out


for n in (0, 1, 1000, 1000003):
    for w in (1, 2, 3, 63, 64, 65, 100, 256, 257, 1000, 1024):
        v = rng.integers(-(2**62), 2**62, n, dtype=np.int64)
        d = cuda.to_device(np.full(n, 5, dtype=np.int64))
        mod.run(cuda.to_device(v), d, n, w)
        cuda.synchronize()
        assert np.array_equal(d.copy_to_host(), ref_window_max(v, w)), (n, w)
    print("n", n, "ok")
print("PASS")
