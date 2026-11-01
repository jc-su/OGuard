"""GPU test of the Numba merge_sorted kernel against a NumPy reference.

Run from the model directory: python -P merge_sorted/test_numba.py
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


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("merge_sorted_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sorted_values(rng: np.random.Generator, size: int, spread: int) -> np.ndarray:
    return np.sort(rng.integers(-spread, spread + 1, size=size, dtype=np.int64))


def check(module: ModuleType, a: np.ndarray, b: np.ndarray) -> None:
    m = a.size
    p = b.size
    expected = np.sort(np.concatenate((a, b)), kind="stable")
    d_a = cuda.to_device(a if m > 0 else np.zeros(1, dtype=np.int64))
    d_b = cuda.to_device(b if p > 0 else np.zeros(1, dtype=np.int64))
    d_out = cuda.to_device(np.full(max(m + p, 1), -1, dtype=np.int64))
    module.run(d_a, d_b, d_out, m, p)
    np.testing.assert_array_equal(d_out.copy_to_host()[: m + p], expected)


def test_merge_sorted() -> None:
    module = load_module()
    rng = np.random.default_rng(9)
    for n in SIZES:
        splits = {(n, 0), (0, n), (n // 2, n - n // 2), (n // 3, n - n // 3), (n, n), (n, 3)}
        for m, p in splits:
            # Many ties, and a wide range with few ties.
            check(module, sorted_values(rng, m, 50), sorted_values(rng, p, 50))
            check(module, sorted_values(rng, m, 2**62), sorted_values(rng, p, 2**62))
    # Disjoint ranges: all of a before all of b, and the reverse.
    a = np.arange(0, 5000, dtype=np.int64)
    b = np.arange(5000, 7000, dtype=np.int64)
    check(module, a, b)
    check(module, b, a)
    # Interleaved, all equal, and extreme values.
    check(module, np.arange(0, 20000, 2, dtype=np.int64), np.arange(1, 20001, 2, dtype=np.int64))
    check(module, np.full(3000, 7, dtype=np.int64), np.full(4000, 7, dtype=np.int64))
    info = np.iinfo(np.int64)
    ext = np.array([info.min, info.min, 0, info.max, info.max], dtype=np.int64)
    check(module, ext, ext[1:4].copy())


if __name__ == "__main__":
    test_merge_sorted()
    print("merge_sorted numba: passed")
