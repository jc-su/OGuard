import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("run_starts_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def reference(v):
    ref = np.ones(len(v), dtype=np.int8)
    ref[1:] = v[1:] != v[:-1]
    return ref


def test_run_starts():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(1)
    for n in SIZES:
        for hi in (1, 3, 1 << 40):
            v = rng.integers(0, hi, n, dtype=np.int64)
            ref = reference(v)
            dv = cuda.to_device(v)
            for launch in ("run", "three_blocks"):
                df = cuda.to_device(np.full(n, 7, dtype=np.int8))
                if launch == "run":
                    mod.run(dv, df, n)
                else:
                    jk[3, 256](dv, df, n)
                got = df.copy_to_host()
                assert np.array_equal(got, ref), (n, hi, launch)


if __name__ == "__main__":
    test_run_starts()
    print("PASS")
