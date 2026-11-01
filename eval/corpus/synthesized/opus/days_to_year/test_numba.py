"""GPU test of the Numba days_to_year kernel against a NumPy reference.

Run from the model directory: python -P days_to_year/test_numba.py
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
FIRST_DAY = -719162  # 0001-01-01
LAST_DAY = 2932896  # 9999-12-31


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("days_to_year_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(days: np.ndarray) -> np.ndarray:
    years = days.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    return years.astype(np.int32)


def check(module: ModuleType, days: np.ndarray) -> None:
    n = days.size
    d_days = cuda.to_device(days)
    d_year = cuda.to_device(np.full(n, -1, dtype=np.int32))
    module.run(d_days, d_year, n)
    np.testing.assert_array_equal(d_year.copy_to_host(), reference(days))


def test_days_to_year() -> None:
    module = load_module()
    rng = np.random.default_rng(8)
    assert reference(np.array([FIRST_DAY, LAST_DAY], dtype=np.int32)).tolist() == [1, 9999]
    for n in SIZES:
        check(module, rng.integers(FIRST_DAY, LAST_DAY + 1, size=n, dtype=np.int32))
    # Every date from 0001-01-01 to 9999-12-31.
    check(module, np.arange(FIRST_DAY, LAST_DAY + 1, dtype=np.int32))


if __name__ == "__main__":
    test_days_to_year()
    print("days_to_year numba: passed")
