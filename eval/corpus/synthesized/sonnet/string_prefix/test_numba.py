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
    for k in (1, 7, 64):
        lens = rng.integers(0, 90, n).astype(np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(lens, out=offsets[1:])
        chars = rng.integers(1, 256, int(offsets[n]), dtype=np.uint8)
        ref = np.zeros((n, k), dtype=np.uint8)
        for j in range(k):
            m = lens > j
            ref[m, j] = chars[offsets[:n][m] + j]
        d_out = cuda.to_device(np.full(n * k, 77, dtype=np.uint8))
        mod.run(cuda.to_device(offsets), cuda.to_device(chars), d_out, n, k)
        cuda.synchronize()
        assert np.array_equal(d_out.copy_to_host(), ref.reshape(-1)), (n, k)
    print("n", n, "ok")
print("PASS")
