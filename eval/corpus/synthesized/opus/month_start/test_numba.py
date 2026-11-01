"""Test the month_start Numba kernel on the GPU against a NumPy reference.

The reference converts with numpy datetime64, which uses the proleptic
Gregorian calendar. Besides the required sizes, the test checks every day of
the stated range [-719162, 2932896] (0001-01-01 to 9999-12-31).

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
FIRST_DAY = -719162
LAST_DAY = 2932896


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("month_start_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(days: np.ndarray) -> np.ndarray:
    months = days.astype(np.int64).astype("datetime64[D]").astype("datetime64[M]")
    return months.astype("datetime64[D]").astype(np.int64).astype(np.int32)


def check(days: np.ndarray) -> None:
    n = days.size
    d_out = cuda.to_device(np.full(n, -1, dtype=np.int32))
    TASK.run(cuda.to_device(days), d_out, n)
    assert np.array_equal(d_out.copy_to_host(), reference(days)), n


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(8)
    for n in SIZES:
        check(rng.integers(FIRST_DAY, LAST_DAY, size=n, dtype=np.int32, endpoint=True))


def test_whole_range() -> None:
    reference_check = reference(np.array([FIRST_DAY, 0, LAST_DAY], dtype=np.int32))
    assert reference_check.tolist() == [FIRST_DAY, 0, LAST_DAY - 30]
    check(np.arange(FIRST_DAY, LAST_DAY + 1, dtype=np.int32))


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    test_whole_range()
    print("month_start numba: all tests passed")
