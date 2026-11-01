import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("segment_max_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min


def make_offsets(rng, n, g):
    cuts = np.sort(rng.integers(0, n + 1, max(g - 1, 0)))
    return np.concatenate([[0], cuts, [n]])[: g + 1].astype(np.int32) if g > 0 else np.array([0], dtype=np.int32)


def reference(offsets, values, g):
    maxes = np.full(g, INT64_MIN, dtype=np.int64)
    seg = np.repeat(np.arange(g), np.diff(offsets.astype(np.int64)))
    np.maximum.at(maxes, seg, values)
    return maxes


def test_segment_max():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(2)
    for n in SIZES:
        values = rng.integers(INT64_MIN, np.iinfo(np.int64).max, n, dtype=np.int64, endpoint=True)
        gs = [1, 3, n // 10 + 1, 2 * n + 5] + ([0] if n == 0 else [])
        for g in gs:
            offsets = make_offsets(rng, n, g)
            assert len(offsets) == g + 1 and offsets[g] == n
            ref = reference(offsets, values, g)
            do = cuda.to_device(offsets)
            dv = cuda.to_device(values)
            for launch in ("run", "three_blocks"):
                dm = cuda.to_device(np.full(g, 5, dtype=np.int64))
                if launch == "run":
                    mod.run(do, dv, dm, g)
                else:
                    jk[3, 256](do, dv, dm, g)
                assert np.array_equal(dm.copy_to_host(), ref), (n, g, launch)


if __name__ == "__main__":
    test_segment_max()
    print("PASS")
