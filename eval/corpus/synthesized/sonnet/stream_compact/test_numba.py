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
    values = rng.integers(-(2**62), 2**62, n, dtype=np.int64)
    keep = (rng.random(n) < 0.3).astype(np.int8)
    if n:
        keep[rng.integers(0, n, max(1, n // 20))] = rng.integers(-128, 128, max(1, n // 20)).astype(np.int8)
    d_out = cuda.to_device(np.full(n, -1, dtype=np.int64))
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    mod.run(cuda.to_device(values), cuda.to_device(keep), d_out, d_count, n)
    cuda.synchronize()
    c = int(d_count.copy_to_host()[0])
    ref = values[keep != 0]
    assert c == len(ref), (n, c, len(ref))
    got = d_out.copy_to_host()[:c]
    assert np.array_equal(np.sort(got), np.sort(ref)), n
    print("n", n, "ok")
print("PASS")
