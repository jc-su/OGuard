import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("group_max_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min


def test_group_max():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(4)
    for n in SIZES:
        caps = [1, 7, 64, 1000, 1024, 1025, 5000, 1000003] + ([0] if n == 0 else [])
        for capacity in caps:
            group = rng.integers(0, max(capacity, 1), n, dtype=np.int32)
            value = rng.integers(INT64_MIN, np.iinfo(np.int64).max, n, dtype=np.int64, endpoint=True)
            ref = np.full(capacity, INT64_MIN, dtype=np.int64)
            np.maximum.at(ref, group, value)
            dg = cuda.to_device(group)
            dv = cuda.to_device(value)
            for launch in ("run", "three_blocks"):
                db = cuda.to_device(np.full(capacity, INT64_MIN, dtype=np.int64))
                if launch == "run":
                    mod.run(dg, dv, db, n, capacity)
                else:
                    jk[3, 256](dg, dv, db, n, capacity)
                assert np.array_equal(db.copy_to_host(), ref), (n, capacity, launch)


if __name__ == "__main__":
    test_group_max()
    print("PASS")
