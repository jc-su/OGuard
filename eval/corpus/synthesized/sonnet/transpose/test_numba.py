import importlib.util
import pathlib

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", pathlib.Path(__file__).with_name("numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cuda.compile_ptx(mod.kernel, mod.TYPES)
rng = np.random.default_rng(1)
shapes = [(0, 5), (5, 0), (0, 0), (1, 1), (1, 1000), (1000, 1), (1000003, 1), (1, 1000003),
          (31, 33), (1000, 1003), (1003, 997), (2, 500001)]
for rows, cols in shapes:
    src = rng.integers(-(2**31), 2**31, rows * cols, dtype=np.int64).astype(np.int32)
    ref = np.ascontiguousarray(src.reshape(rows, cols).T).reshape(-1)
    d = cuda.to_device(np.full(rows * cols, -1, dtype=np.int32))
    mod.run(cuda.to_device(src), d, rows, cols)
    cuda.synchronize()
    assert np.array_equal(d.copy_to_host(), ref), (rows, cols)
    print("shape", rows, cols, "ok")
print("PASS")
