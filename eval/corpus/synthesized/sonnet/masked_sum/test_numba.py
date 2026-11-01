import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("masked_sum_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def test_masked_sum():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(6)
    for n in SIZES:
        for keep in (0, 1, 50, 100):
            values = rng.random(n)
            # Nonzero mask entries other than 1 also select the row.
            mask = np.where(rng.integers(0, 100, n) < keep, rng.integers(1, 101, n), 0).astype(np.int8)
            ref = values[mask != 0].sum(dtype=np.longdouble)
            dv = cuda.to_device(values)
            dm = cuda.to_device(mask)
            for launch in ("run", "three_blocks"):
                dr = cuda.to_device(np.zeros(1))
                if launch == "run":
                    mod.run(dv, dm, dr, n)
                else:
                    jk[3, 256](dv, dm, dr, n)
                got = dr.copy_to_host()[0]
                # Values are in [0, 1), so a relative tolerance covers the addition order.
                assert got == ref or abs(got - ref) <= 1e-9 * ref, (n, keep, launch, got, ref)


if __name__ == "__main__":
    test_masked_sum()
    print("PASS")
