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
    for hi in (3, 2000):  # sums stay below 2^31
        lengths = rng.integers(0, hi + 1, n).astype(np.int32)
        ref = np.zeros(n + 1, dtype=np.int32)
        ref[1:] = np.cumsum(lengths, dtype=np.int64)
        d = cuda.to_device(np.full(n + 1, -5, dtype=np.int32))
        mod.run(cuda.to_device(lengths), d, n)
        cuda.synchronize()
        assert np.array_equal(d.copy_to_host(), ref), (n, hi)
    print("n", n, "ok")
print("PASS")
