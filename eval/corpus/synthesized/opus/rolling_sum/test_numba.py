"""GPU test of the Numba rolling_sum kernel against a NumPy reference.

Run from the model directory: python -P rolling_sum/test_numba.py
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
SMALL_WINDOWS = (1, 2, 3, 7, 32, 255, 256, 257, 1000, 1023, 1024)
LARGE_WINDOWS = (1, 7, 256, 1024)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rolling_sum_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values: np.ndarray, window: int) -> np.ndarray:
    """Direct summation of every window."""
    if values.size == 0:
        return values.copy()
    return np.convolve(values, np.ones(window))[: values.size]


def check(module: ModuleType, values: np.ndarray, window: int) -> None:
    n = values.size
    d_values = cuda.to_device(values if n > 0 else np.zeros(1))
    d_out = cuda.to_device(np.full(max(n, 1), np.nan))
    module.run(d_values, d_out, n, window)
    out = d_out.copy_to_host()[:n]
    np.testing.assert_allclose(out, reference(values, window), rtol=1e-9, atol=1e-9)


def test_rolling_sum() -> None:
    module = load_module()
    rng = np.random.default_rng(6)
    for n in SIZES:
        values = rng.uniform(-1.0, 1.0, size=n)
        for window in SMALL_WINDOWS if n <= 1000 else LARGE_WINDOWS:
            check(module, values, window)
    for n in (1023, 1024, 1025, 2047, 2048, 2049, 5000):
        values = rng.uniform(-1.0, 1.0, size=n)
        for window in (1, 5, 1000, 1024):
            check(module, values, window)
    # A huge value early on must not disturb the windows that exclude it:
    # a difference of prefix sums would lose the small values after it.
    values = rng.uniform(-1.0, 1.0, size=100000)
    values[5] = 1e15
    for window in (1, 16, 1024):
        check(module, values, window)


if __name__ == "__main__":
    test_rolling_sum()
    print("rolling_sum numba: passed")
