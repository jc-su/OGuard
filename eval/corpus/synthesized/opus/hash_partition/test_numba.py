"""Test the hash_partition Numba kernel on the GPU against a NumPy reference.

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
PARTITIONS: tuple[int, ...] = (1, 7, 1000, 1024)
MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("hash_partition_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(keys: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    hashed = (keys.view(np.uint64) * MULTIPLIER) >> np.uint64(32)
    part = (hashed % np.uint64(p)).astype(np.int32)
    counts = np.bincount(part, minlength=p).astype(np.int32)
    return part, counts


def check(keys: np.ndarray, p: int) -> None:
    n = keys.size
    d_part = cuda.to_device(np.full(n, -1, dtype=np.int32))
    d_counts = cuda.to_device(np.zeros(p, dtype=np.int32))
    TASK.run(cuda.to_device(keys), d_part, d_counts, n, p)
    part, counts = reference(keys, p)
    assert np.array_equal(d_part.copy_to_host(), part), (n, p)
    assert np.array_equal(d_counts.copy_to_host(), counts), (n, p)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(7)
    for n in SIZES:
        for p in PARTITIONS:
            keys = rng.integers(-(2**63), 2**63 - 1, size=n, dtype=np.int64, endpoint=True)
            check(keys, p)
    check(np.arange(-500, 500, dtype=np.int64), 16)
    check(np.full(1000003, 42, dtype=np.int64), 1024)
    extremes = np.array([np.iinfo(np.int64).min, -1, 0, 1, np.iinfo(np.int64).max])
    check(extremes.astype(np.int64), 3)


def test_p_out_of_range() -> None:
    keys = cuda.to_device(np.zeros(1, dtype=np.int64))
    part = cuda.to_device(np.zeros(1, dtype=np.int32))
    counts = cuda.to_device(np.zeros(1025, dtype=np.int32))
    for p in (0, 1025):
        try:
            TASK.run(keys, part, counts, 1, p)
        except ValueError:
            continue
        raise AssertionError(f"p={p} was accepted")


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    test_p_out_of_range()
    print("hash_partition numba: all tests passed")
