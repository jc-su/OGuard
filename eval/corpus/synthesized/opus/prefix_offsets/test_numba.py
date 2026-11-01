"""Test the prefix_offsets Numba kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES: tuple[int, ...] = (0, 1, 1000, 4095, 4096, 4097, 1000003)
MAX_LENGTH = 2**20


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("prefix_offsets_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(lengths: np.ndarray) -> np.ndarray:
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    return offsets.astype(np.int32)


def check(lengths: np.ndarray) -> None:
    n = lengths.size
    d_offsets = cuda.to_device(np.full(n + 1, -7, dtype=np.int32))
    TASK.run(cuda.to_device(lengths), d_offsets, n)
    assert np.array_equal(d_offsets.copy_to_host(), reference(lengths)), n


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(5)
    for n in SIZES:
        # The largest length that keeps offsets[n] within int32.
        top = min(MAX_LENGTH, (2**31 - 1) // max(n, 1))
        check(rng.integers(0, top, size=n, dtype=np.int32, endpoint=True))
        check(np.zeros(n, dtype=np.int32))
    check(np.full(2047, MAX_LENGTH, dtype=np.int32))


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    print("prefix_offsets numba: all tests passed")
