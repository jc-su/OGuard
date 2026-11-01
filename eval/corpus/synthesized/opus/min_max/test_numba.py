"""GPU test of the Numba min_max kernel against a NumPy reference.

Run from the model directory: python -P min_max/test_numba.py
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
    spec = importlib.util.spec_from_file_location("min_max_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(module: ModuleType, values: np.ndarray) -> None:
    n = values.size
    d_values = cuda.to_device(values)
    d_result = cuda.to_device(np.array([np.inf, -np.inf]))
    module.run(d_values, d_result, n)
    result = d_result.copy_to_host()
    expected = [np.inf, -np.inf] if n == 0 else [values.min(), values.max()]
    np.testing.assert_array_equal(result, expected)


def test_min_max() -> None:
    module = load_module()
    rng = np.random.default_rng(5)
    for n in SIZES:
        check(module, rng.standard_normal(n) * 1e3)
        check(module, -rng.random(n) - 1.0)
        check(module, rng.random(n) + 1.0)
        check(module, np.full(n, 3.25))
    values = rng.standard_normal(1000003)
    values[123457] = np.inf
    values[999999] = -np.inf
    check(module, values)
    values = rng.standard_normal(1000003)
    values[0] = -1e300
    values[-1] = 1e300
    check(module, values)
    check(module, np.array([-0.0, 0.0, -0.0]))


if __name__ == "__main__":
    test_min_max()
    print("min_max numba: passed")
