import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("lower_bound_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def test_lower_bound():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(3)
    for n in SIZES:
        for s in SIZES:
            sorted_ = np.sort(rng.integers(-1000, 1001, s, dtype=np.int64))
            keys = rng.integers(-1100, 1101, n, dtype=np.int64)
            if n > 2:
                keys[0] = np.iinfo(np.int64).min
                keys[1] = np.iinfo(np.int64).max
            ref = np.searchsorted(sorted_, keys, side="left").astype(np.int32)
            dk = cuda.to_device(keys)
            ds = cuda.to_device(sorted_)
            for launch in ("run", "three_blocks"):
                dout = cuda.to_device(np.full(n, 9, dtype=np.int32))
                if launch == "run":
                    mod.run(dk, ds, dout, n, s)
                else:
                    jk[3, 256](dk, ds, dout, n, s)
                assert np.array_equal(dout.copy_to_host(), ref), (n, s, launch)


if __name__ == "__main__":
    test_lower_bound()
    print("PASS")
