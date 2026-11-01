"""Test the string_prefix Numba kernel on the GPU against a NumPy reference.

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
KS: tuple[int, ...] = (1, 5, 64)
FIRST_OFFSETS: tuple[int, ...] = (0, 3)


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("string_prefix_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def make_column(n: int, first: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lengths = rng.integers(0, 100, size=n, dtype=np.int64)
    lengths[rng.random(n) < 0.1] = 0
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = first
    offsets[1:] = first + np.cumsum(lengths)
    chars = rng.integers(0, 255, size=int(offsets[n]), dtype=np.uint8, endpoint=True)
    return offsets, chars


def reference(offsets: np.ndarray, chars: np.ndarray, n: int, k: int) -> np.ndarray:
    starts = offsets[:-1].astype(np.int64)
    lengths = offsets[1:].astype(np.int64) - starts
    column = np.arange(k, dtype=np.int64)
    inside = column[None, :] < np.minimum(lengths, k)[:, None]
    positions = np.where(inside, starts[:, None] + column[None, :], 0)
    gathered = chars[positions] if chars.size else np.zeros((n, k), dtype=np.uint8)
    return np.where(inside, gathered, np.uint8(0)).astype(np.uint8).reshape(n * k)


def check(n: int, k: int, first: int, rng: np.random.Generator) -> None:
    offsets, chars = make_column(n, first, rng)
    d_out = cuda.to_device(np.full(n * k, 0xA5, dtype=np.uint8))
    TASK.run(cuda.to_device(offsets), cuda.to_device(chars), d_out, n, k)
    out = d_out.copy_to_host()
    assert np.array_equal(out, reference(offsets, chars, n, k)), (n, k, first)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(2)
    for n in SIZES:
        for k in KS:
            for first in FIRST_OFFSETS:
                check(n, k, first, rng)


def test_k_out_of_range() -> None:
    offsets = cuda.to_device(np.zeros(2, dtype=np.int32))
    chars = cuda.to_device(np.zeros(0, dtype=np.uint8))
    out = cuda.to_device(np.zeros(65, dtype=np.uint8))
    for k in (0, 65):
        try:
            TASK.run(offsets, chars, out, 1, k)
        except ValueError:
            continue
        raise AssertionError(f"k={k} was accepted")


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    test_k_out_of_range()
    print("string_prefix numba: all tests passed")
