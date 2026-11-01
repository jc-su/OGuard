import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("diff_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def reference(v):
    ref = v.copy()
    ref[1:] = v[1:] - v[:-1]  # int64 subtraction wraps
    return ref


def test_diff():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(8)
    for n in SIZES:
        for full_range in (False, True):
            if full_range:
                v = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, n, dtype=np.int64, endpoint=True)
            else:
                v = rng.integers(0, 1000, n, dtype=np.int64)
            ref = reference(v)
            dv = cuda.to_device(v)
            for launch in ("run", "three_blocks"):
                dout = cuda.to_device(np.full(n, 3, dtype=np.int64))
                if launch == "run":
                    mod.run(dv, dout, n)
                else:
                    jk[3, 256](dv, dout, n)
                assert np.array_equal(dout.copy_to_host(), ref), (n, full_range, launch)


if __name__ == "__main__":
    test_diff()
    print("PASS")
