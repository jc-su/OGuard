"""Test the pair_sum Numba kernel on the GPU against a NumPy reference.

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
# (l, r) pairs: a small table and a large table on either side.
TABLES: tuple[tuple[int, int], ...] = ((1, 1), (777, 1000003), (1000003, 13))


def load_task() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("pair_sum_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def check(m: int, l: int, r: int, rng: np.random.Generator) -> None:  # noqa: E741
    left_idx = rng.integers(0, l, size=m, dtype=np.int32)
    right_idx = rng.integers(0, r, size=m, dtype=np.int32)
    left = rng.standard_normal(l) * 1e6
    right = rng.standard_normal(r) * 1e6
    d_out = cuda.to_device(np.full(m, np.nan))
    TASK.run(
        cuda.to_device(left_idx),
        cuda.to_device(right_idx),
        cuda.to_device(left),
        cuda.to_device(right),
        d_out,
        m,
        l,
        r,
    )
    expected = left[left_idx] + right[right_idx]
    # Both sides compute one IEEE double addition, so the bits must match.
    got = d_out.copy_to_host()
    assert np.array_equal(got.view(np.int64), expected.view(np.int64)), (m, l, r)


def test_compile_ptx() -> None:
    cuda.compile_ptx(TASK.kernel, TASK.TYPES)


def test_sizes() -> None:
    rng = np.random.default_rng(10)
    for m in SIZES:
        for l, r in TABLES:  # noqa: E741
            check(m, l, r, rng)
    check(0, 0, 0, rng)


if __name__ == "__main__":
    # Small test sizes launch few blocks; Numba warns about the low occupancy.
    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")
    test_compile_ptx()
    test_sizes()
    print("pair_sum numba: all tests passed")
