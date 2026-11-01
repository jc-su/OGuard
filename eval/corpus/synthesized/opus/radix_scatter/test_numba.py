"""GPU test of the Numba radix_scatter kernel against a NumPy reference.

Run from the model directory: python -P radix_scatter/test_numba.py
"""

import importlib.util
import pathlib
import warnings
from types import ModuleType

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Small test sizes launch few blocks; the low-occupancy warning is expected.
warnings.simplefilter("ignore", NumbaPerformanceWarning)

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
SHIFTS = (0, 8, 16, 24)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("radix_scatter_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(module: ModuleType, keys: np.ndarray, shift: int) -> None:
    n = keys.size
    digits = ((keys >> np.uint32(shift)) & np.uint32(255)).astype(np.int64)
    counts = np.bincount(digits, minlength=256)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int32)
    d_keys = cuda.to_device(keys)
    d_cursors = cuda.to_device(starts)
    d_out = cuda.to_device(np.full(max(n, 1), -1, dtype=np.int32))
    module.run(d_keys, d_cursors, d_out, n, shift)
    out = d_out.copy_to_host()[:n]
    # A permutation of the row ids, grouped by digit in the cursors' layout.
    np.testing.assert_array_equal(np.sort(out), np.arange(n, dtype=np.int32))
    np.testing.assert_array_equal(digits[out], np.repeat(np.arange(256), counts))
    np.testing.assert_array_equal(d_cursors.copy_to_host(), starts + counts)


def test_radix_scatter() -> None:
    module = load_module()
    rng = np.random.default_rng(7)
    for n in SIZES:
        keys = rng.integers(0, 2**32, size=n, dtype=np.uint32)
        for shift in SHIFTS:
            check(module, keys, shift)
        check(module, np.full(n, 0xDEADBEEF, dtype=np.uint32), 8)
        check(module, np.arange(n, dtype=np.uint32), 0)
        check(module, rng.integers(0, 3, size=n, dtype=np.uint32) << np.uint32(16), 16)
    for n in (255, 256, 257, 2047, 2048, 2049):
        check(module, rng.integers(0, 2**32, size=n, dtype=np.uint32), 24)


if __name__ == "__main__":
    test_radix_scatter()
    print("radix_scatter numba: passed")
