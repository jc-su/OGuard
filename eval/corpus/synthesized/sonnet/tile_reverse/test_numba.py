import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("tile_reverse_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

# Sizes at and around a tile boundary are added to the required sizes.
SIZES = (0, 1, 1000, 1023, 1024, 1025, 2048, 3000, 1000003)
TILE = 1024


def reference(v):
    ref = np.empty_like(v)
    for base in range(0, len(v), TILE):
        ref[base : base + TILE] = v[base : base + TILE][::-1]
    return ref


def test_tile_reverse():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(10)
    for n in SIZES:
        v = rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, n, dtype=np.int32)
        ref = reference(v)
        dv = cuda.to_device(v)
        for launch in ("run", "three_blocks"):
            dout = cuda.to_device(np.full(n, 3, dtype=np.int32))
            if launch == "run":
                mod.run(dv, dout, n)
            else:
                jk[3, 256](dv, dout, n)
            assert np.array_equal(dout.copy_to_host(), ref), (n, launch)


if __name__ == "__main__":
    test_tile_reverse()
    print("PASS")
