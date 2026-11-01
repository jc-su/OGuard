import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("scatter_add_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def test_scatter_add():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(9)
    for m in SIZES:
        # out has at least one row whenever there is an index to scatter; n = 0 only pairs with m = 0.
        for n in (1, 10, 1000, 1000003) + ((0,) if m == 0 else ()):
            idx = rng.integers(0, max(n, 1), m, dtype=np.int32)
            values = rng.random(m)
            ref = np.zeros(n)
            np.add.at(ref, idx, values)
            di = cuda.to_device(idx)
            dv = cuda.to_device(values)
            for launch in ("run", "three_blocks"):
                dout = cuda.to_device(np.zeros(n))
                if launch == "run":
                    mod.run(di, dv, dout, m, n)
                else:
                    jk[3, 256](di, dv, dout, m, n)
                got = dout.copy_to_host()
                # Values are in [0, 1), so a relative tolerance covers the addition order.
                assert np.allclose(got, ref, rtol=1e-9, atol=0), (m, n, launch)


if __name__ == "__main__":
    test_scatter_add()
    print("PASS")
