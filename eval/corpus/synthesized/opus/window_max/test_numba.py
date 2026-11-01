"""Test the window_max Numba kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda
from numpy.lib.stride_tricks import sliding_window_view

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
WINDOWS: tuple[int, ...] = (1, 2, 3, 100, 512, 1000, 1024)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("window_max_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(values: np.ndarray, w: int) -> np.ndarray:
    # Pad on the left with the first value, which is in every window it enters.
    if values.size == 0:
        return values.copy()
    padded = np.concatenate([np.full(w - 1, values[0], dtype=np.int64), values])
    return sliding_window_view(padded, w).max(axis=1)


def check(values: np.ndarray, w: int) -> None:
    n = values.size
    d_out = cuda.to_device(np.full(n, 7, dtype=np.int64))
    TASK.run(cuda.to_device(values), d_out, n, w)
    assert np.array_equal(d_out.copy_to_host(), reference(values, w)), (n, w)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(9)
    for n in SIZES:
        for w in WINDOWS:
            check(rng.integers(INT64_MIN, INT64_MAX, size=n, dtype=np.int64, endpoint=True), w)
    descending = np.arange(1000003, 0, -1, dtype=np.int64)
    for w in (1, 777, 1024):
        check(descending, w)
        check(-descending, w)
    check(np.full(5000, INT64_MIN, dtype=np.int64), 1024)


def test_w_out_of_range() -> None:
    values = cuda.to_device(np.zeros(1, dtype=np.int64))
    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    for w in (0, 1025):
        try:
            TASK.run(values, out, 1, w)
        except ValueError:
            continue
        raise AssertionError(f"w={w} was accepted")


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    test_w_out_of_range()
    print("window_max numba: all tests passed")
