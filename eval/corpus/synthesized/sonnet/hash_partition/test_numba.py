import importlib.util
import pathlib

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", pathlib.Path(__file__).with_name("numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cuda.compile_ptx(mod.kernel, mod.TYPES)
rng = np.random.default_rng(1)
for n in (0, 1, 1000, 1000003):
    for p in (1, 7, 256, 1000, 1024):
        keys = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64)
        if n > 10:
            keys[: n // 2] = rng.integers(-5, 5, n // 2)  # skewed, with negative keys
        h = (keys.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
        ref = (h % np.uint64(p)).astype(np.int32)
        d_part = cuda.to_device(np.full(n, -1, dtype=np.int32))
        d_counts = cuda.to_device(np.zeros(p, dtype=np.int32))
        mod.run(cuda.to_device(keys), d_part, d_counts, n, p)
        cuda.synchronize()
        assert np.array_equal(d_part.copy_to_host(), ref), (n, p)
        assert np.array_equal(d_counts.copy_to_host(), np.bincount(ref, minlength=p)), (n, p)
    print("n", n, "ok")
print("PASS")
