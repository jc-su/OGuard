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
    for l, r in ((1, 1), (1000, 37), (max(1, n), max(1, n // 3))):
        li = rng.integers(0, l, n).astype(np.int32)
        ri = rng.integers(0, r, n).astype(np.int32)
        left = rng.standard_normal(l)
        right = rng.standard_normal(r)
        ref = left[li] + right[ri]
        d = cuda.to_device(np.full(n, np.nan))
        mod.run(cuda.to_device(li), cuda.to_device(ri), cuda.to_device(left), cuda.to_device(right), d, n, l, r)
        cuda.synchronize()
        assert np.array_equal(d.copy_to_host(), ref), (n, l, r)
    print("n", n, "ok")
print("PASS")
