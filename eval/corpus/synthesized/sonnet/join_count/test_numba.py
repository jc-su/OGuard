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
    for b in (0, 1, n):
        build = np.sort(rng.integers(-50, 50 + n // 100, b, dtype=np.int64))
        probe = rng.integers(-60, 60 + n // 100, n, dtype=np.int64)
        ref = (np.searchsorted(build, probe, "right") - np.searchsorted(build, probe, "left")).astype(np.int32)
        d = cuda.to_device(np.full(n, -1, dtype=np.int32))
        mod.run(cuda.to_device(build), cuda.to_device(probe), d, b, n)
        cuda.synchronize()
        assert np.array_equal(d.copy_to_host(), ref), (n, b)
    print("n", n, "ok")
print("PASS")
