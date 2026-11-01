"""Test the join_count Numba kernel on the GPU against a NumPy reference.

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
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("join_count_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(build: np.ndarray, probe: np.ndarray) -> np.ndarray:
    first = np.searchsorted(build, probe, side="left")
    last = np.searchsorted(build, probe, side="right")
    return (last - first).astype(np.int32)


def check(build: np.ndarray, probe: np.ndarray) -> None:
    b = build.size
    p = probe.size
    d_counts = cuda.to_device(np.full(p, -1, dtype=np.int32))
    TASK.run(cuda.to_device(build), cuda.to_device(probe), d_counts, b, p)
    assert np.array_equal(d_counts.copy_to_host(), reference(build, probe)), (b, p)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(3)
    for b in SIZES:
        for p in SIZES:
            # Keys drawn from a range near b give many duplicates and misses.
            span = max(b // 4, 1)
            build = np.sort(rng.integers(-span, span, size=b, dtype=np.int64))
            probe = rng.integers(-2 * span, 2 * span, size=p, dtype=np.int64)
            check(build, probe)


def test_extreme_keys() -> None:
    build = np.array(
        [INT64_MIN, INT64_MIN, -1, 0, 0, 0, 7, INT64_MAX, INT64_MAX, INT64_MAX],
        dtype=np.int64,
    )
    probe = np.array([INT64_MIN, -1, 0, 1, 7, INT64_MAX, INT64_MAX - 1], dtype=np.int64)
    check(build, probe)
    same = np.full(1000003, 5, dtype=np.int64)
    check(same, np.array([4, 5, 6], dtype=np.int64))


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    test_extreme_keys()
    print("join_count numba: all tests passed")
