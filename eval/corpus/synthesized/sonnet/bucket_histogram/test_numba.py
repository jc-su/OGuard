import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("bucket_histogram_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def test_bucket_histogram():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(7)
    for n in SIZES:
        for b in (0, 1, 10, 1023, 1024, 5000):
            bounds = np.sort(np.floor(rng.uniform(-100, 100, b)))  # integers, so duplicates and exact hits occur
            values = rng.uniform(-120, 120, n)
            kind = rng.integers(0, 4, n)
            if b > 0:
                values = np.where(kind == 0, bounds[rng.integers(0, b, n)], values)
            values = np.where(kind == 1, np.floor(values), values)
            if n > 3:
                values[:3] = [np.inf, -np.inf, 0.0]
            ref = np.bincount(np.searchsorted(bounds, values, side="right"), minlength=b + 1).astype(np.int32)
            dv = cuda.to_device(values)
            db = cuda.to_device(bounds)
            for launch in ("run", "three_blocks"):
                dc = cuda.to_device(np.zeros(b + 1, dtype=np.int32))
                if launch == "run":
                    mod.run(dv, db, dc, n, b)
                else:
                    jk[3, 256](dv, db, dc, n, b)
                assert np.array_equal(dc.copy_to_host(), ref), (n, b, launch)


if __name__ == "__main__":
    test_bucket_histogram()
    print("PASS")
