"""Test the null_count Numba kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES: tuple[int, ...] = (0, 1, 7, 8, 9, 1000, 1000003)
DENSITIES: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
INITIAL = 12345


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("null_count_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def check(n: int, null_density: float, rng: np.random.Generator) -> None:
    rows_valid = rng.random(n) >= null_density
    valid = np.packbits(rows_valid, bitorder="little")
    # Bits past row n - 1 in the last byte are set at random; they must not count.
    if n % 8 != 0:
        tail = rng.integers(0, 255, dtype=np.uint8, endpoint=True)
        valid[-1] |= tail & np.uint8(0xFF << (n % 8) & 0xFF)
    assert valid.size == (n + 7) // 8
    expected = INITIAL + int(n - np.count_nonzero(rows_valid))
    d_count = cuda.to_device(np.array([INITIAL], dtype=np.int64))
    TASK.run(cuda.to_device(valid), d_count, n)
    assert int(d_count.copy_to_host()[0]) == expected, (n, null_density)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(4)
    for n in SIZES:
        for null_density in DENSITIES:
            check(n, null_density, rng)


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    print("null_count numba: all tests passed")
