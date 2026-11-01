"""GPU test of the Numba valid_rows kernel against a NumPy reference.

Run from the model directory: python -P valid_rows/test_numba.py
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
DENSITIES = (0.0, 0.03, 0.5, 0.97, 1.0)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("valid_rows_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bitmap(rng: np.random.Generator, n: int, density: float) -> np.ndarray:
    nbytes = (n + 7) // 8
    bits = rng.random(nbytes * 8) < density
    # Set the padding bits past row n so the kernel must ignore them.
    bits[n:] = True
    return np.packbits(bits, bitorder="little")


def check(module: ModuleType, valid: np.ndarray, n: int) -> None:
    expected = np.flatnonzero(np.unpackbits(valid, bitorder="little")[:n]).astype(np.int32)
    d_valid = cuda.to_device(valid)
    d_rows = cuda.to_device(np.full(max(n, 1), -1, dtype=np.int32))
    d_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    module.run(d_valid, d_rows, d_count, n)
    count = int(d_count.copy_to_host()[0])
    rows = d_rows.copy_to_host()[:count]
    assert count == expected.size, (n, count, expected.size)
    np.testing.assert_array_equal(np.sort(rows), expected)


def test_valid_rows() -> None:
    module = load_module()
    rng = np.random.default_rng(1)
    for n in SIZES:
        for density in DENSITIES:
            check(module, make_bitmap(rng, n, density), n)
    for n in (7, 8, 9, 255, 256, 257, 2047, 2048, 2049, 65536 + 3):
        check(module, make_bitmap(rng, n, 0.5), n)


if __name__ == "__main__":
    test_valid_rows()
    print("valid_rows numba: passed")
