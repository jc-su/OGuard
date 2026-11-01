import importlib.util
import pathlib

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", pathlib.Path(__file__).with_name("numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cuda.compile_ptx(mod.kernel, mod.TYPES)
rng = np.random.default_rng(1)
LO, HI = -719162, 2932896
all_days = np.arange(LO, HI + 1, dtype=np.int32)
for n in (0, 1, 1000, 1000003):
    days = rng.integers(LO, HI + 1, n).astype(np.int32)
    if n >= 4:
        days[:4] = (LO, HI, 0, 59)
    cases = [days]
    if n == 1000003:
        cases.append(all_days)  # every valid day
    for days in cases:
        m = len(days)
        ref = days.astype("datetime64[D]").astype("datetime64[M]").astype("datetime64[D]").astype(np.int32)
        d = cuda.to_device(np.full(m, -1, dtype=np.int32))
        mod.run(cuda.to_device(days), d, m)
        cuda.synchronize()
        assert np.array_equal(d.copy_to_host(), ref), n
    print("n", n, "ok")
print("PASS")
