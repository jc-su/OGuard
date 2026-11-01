import importlib.util
import pathlib

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", pathlib.Path(__file__).with_name("numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cuda.compile_ptx(mod.kernel, mod.TYPES)
rng = np.random.default_rng(1)
for n in (0, 1, 7, 8, 9, 1000, 1000003):
    for density in (0.0, 0.5, 1.0):
        bits = rng.random(n) < density
        valid = np.packbits(bits, bitorder="little")
        if n % 8:
            valid[-1] |= np.uint8((0xFF << (n % 8)) & 0xFF)  # garbage bits past n must be ignored
        d = cuda.to_device(np.zeros(1, dtype=np.int64))
        mod.run(cuda.to_device(valid), d, n)
        cuda.synchronize()
        assert int(d.copy_to_host()[0]) == int((~bits).sum()), (n, density)
    print("n", n, "ok")
print("PASS")
