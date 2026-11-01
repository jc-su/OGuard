"""GPU test of the Numba run_end_decode kernel against a NumPy reference.

Run from the model directory: python -P run_end_decode/test_numba.py
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
    spec = importlib.util.spec_from_file_location("run_end_decode_numba", HERE / "numba.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_run_ends(rng: np.random.Generator, n: int, r: int) -> np.ndarray:
    cuts = rng.choice(np.arange(1, n), size=r - 1, replace=False) if r > 1 else np.zeros(0)
    return np.concatenate((np.sort(cuts), [n])).astype(np.int32)


def skewed_run_ends(rng: np.random.Generator, n: int) -> np.ndarray:
    """One long run in the middle, short runs elsewhere, and runs of about a tile."""
    lengths: list[int] = []
    total = 0
    while total < n:
        u = rng.random()
        if u < 0.001:
            length = n // 3
        elif u < 0.01:
            length = int(rng.integers(2000, 2100))
        else:
            length = int(rng.integers(1, 5))
        length = min(length, n - total)
        lengths.append(length)
        total += length
    return np.cumsum(lengths).astype(np.int32)


def check(module: ModuleType, rng: np.random.Generator, run_ends: np.ndarray, n: int) -> None:
    r = run_ends.size
    run_values = rng.standard_normal(r)
    expected = np.repeat(run_values, np.diff(np.concatenate(([0], run_ends))))
    d_values = cuda.to_device(run_values if r > 0 else np.zeros(1))
    d_ends = cuda.to_device(run_ends if r > 0 else np.zeros(1, dtype=np.int32))
    d_out = cuda.to_device(np.full(max(n, 1), np.nan))
    module.run(d_values, d_ends, d_out, r, n)
    np.testing.assert_array_equal(d_out.copy_to_host()[:n], expected)


def test_run_end_decode() -> None:
    module = load_module()
    rng = np.random.default_rng(3)
    check(module, rng, np.zeros(0, dtype=np.int32), 0)
    for n in SIZES[1:]:
        check(module, rng, np.array([n], dtype=np.int32), n)
        check(module, rng, np.arange(1, n + 1, dtype=np.int32), n)
        for r in {max(1, n // 1000), max(1, n // 10), max(1, n // 2)}:
            check(module, rng, random_run_ends(rng, n, r), n)
        check(module, rng, skewed_run_ends(rng, n), n)
    for n in (1791, 1792, 1793, 2047, 2048, 2049, 4096, 300000):
        check(module, rng, random_run_ends(rng, n, max(1, n // 3)), n)
        check(module, rng, np.arange(1, n + 1, dtype=np.int32), n)


if __name__ == "__main__":
    test_run_end_decode()
    print("run_end_decode numba: passed")
