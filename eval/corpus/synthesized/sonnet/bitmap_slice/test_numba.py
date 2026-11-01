import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(source, n, offset):
    first = offset >> 3
    last = (offset + n + 7) >> 3
    bits = np.unpackbits(source[first:last], bitorder="little")
    sel = bits[offset & 7 : (offset & 7) + n]
    return np.packbits(sel, bitorder="little")


def check(n, offset, rng):
    source = rng.integers(0, 256, size=(offset + n + 7) // 8, dtype=np.uint8)
    expected = reference(source, n, offset)
    d_out = cuda.to_device(np.full((n + 7) // 8, 0xA5, dtype=np.uint8))
    mod.run(cuda.to_device(source), d_out, n, offset)
    cuda.synchronize()
    ok = np.array_equal(d_out.copy_to_host(), expected)
    print(f"n={n} offset={offset}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    ok = True
    for n in (0, 1, 1000, 1000003):
        for offset in (0, 1, 3, 7, 8, 9, 13, 64, 12345):
            ok &= check(n, offset, rng)
    for n in (0, 1, 1000):
        ok &= check(n, 1 << 30, rng)
        ok &= check(n, (1 << 30) - 5, rng)
    sys.exit(0 if ok else 1)
