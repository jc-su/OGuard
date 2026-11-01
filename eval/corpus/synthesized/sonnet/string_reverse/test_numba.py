import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _here]

import numpy as np
from numba import cuda

_spec = importlib.util.spec_from_file_location("string_reverse_numba", os.path.join(_here, "numba.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

SIZES = (0, 1, 1000, 1000003)


def reference(offsets, chars, n):
    o = offsets.astype(np.int64)
    lens = np.diff(o)
    s = np.repeat(np.arange(n), lens)
    p = np.arange(len(chars))
    return chars[o[s] + o[s + 1] - 1 - p]


def test_string_reverse():
    cuda.compile_ptx(mod.kernel, mod.TYPES)
    jk = cuda.jit(mod.kernel)
    rng = np.random.default_rng(5)
    for n in SIZES:
        for max_len in (0, 1, 20, 100) + ((5000,) if n <= 1000 else ()):
            lens = rng.integers(0, max_len + 1, n)
            offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
            total = int(offsets[n])
            chars = rng.integers(0, 256, total, dtype=np.uint8)
            ref = reference(offsets, chars, n)
            do = cuda.to_device(offsets)
            dc = cuda.to_device(chars)
            for launch in ("run", "three_blocks"):
                dout = cuda.to_device(np.full(total, 0xAA, dtype=np.uint8))
                if launch == "run":
                    mod.run(do, dc, dout, n)
                else:
                    jk[3, 256](do, dc, dout, n)
                assert np.array_equal(dout.copy_to_host(), ref), (n, max_len, launch)


if __name__ == "__main__":
    test_string_reverse()
    print("PASS")
