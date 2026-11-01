"""Test the filter_select Numba kernel on the GPU against a NumPy reference.

Run from the parent directory: python -m filter_select.test_numba
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
THRESHOLDS = (-1.0, 24.0, 51.0)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("filter_select_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, threshold: float, rng: np.random.Generator) -> None:
    quantity = rng.uniform(0.0, 50.0, n)
    quantity[rng.random(n) < 0.1] = threshold  # Rows equal to the threshold.
    d_quantity = cuda.to_device(quantity)
    d_sel = cuda.to_device(np.full(n, -1, dtype=np.int32))
    # run() must reset count, so start from a nonzero value.
    d_count = cuda.to_device(np.array([12345], dtype=np.int32))

    IMPL.run(d_quantity, threshold, d_sel, d_count, n)
    cuda.synchronize()

    expected = np.flatnonzero(quantity < threshold).astype(np.int32)
    count = int(d_count.copy_to_host()[0])
    sel = d_sel.copy_to_host()
    assert count == expected.size, f"n={n}: count {count} != {expected.size}"
    np.testing.assert_array_equal(np.sort(sel[:count]), expected)
    assert np.all(sel[count:] == -1), f"n={n}: write past count"


def test_filter_select() -> None:
    rng = np.random.default_rng(1)
    for n in SIZES:
        for threshold in THRESHOLDS:
            check(n, threshold, rng)


if __name__ == "__main__":
    test_filter_select()
    print(f"filter_select numba: pass for n in {SIZES}")
