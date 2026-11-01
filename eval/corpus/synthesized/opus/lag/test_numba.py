"""Test the Numba lag kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Each size runs with k in LAGS. A row starts a new partition with probability
1 / (2k), so partitions average 2k rows and both outcomes of the test
i - k >= first[i] occur.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
LAGS = (1, 7, 1024)


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("lag_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def make_first(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.random(n) < 1.0 / (2 * k)
    if n > 0:
        starts[0] = True
    rows = np.arange(n)
    return np.maximum.accumulate(np.where(starts, rows, 0)).astype(np.int32)


def reference(values: np.ndarray, first: np.ndarray, k: int) -> np.ndarray:
    source = np.arange(values.shape[0]) - k
    inside = source >= first
    out = np.full(values.shape[0], np.nan)
    out[inside] = values[source[inside]]
    return out


def check(n: int, k: int, rng: np.random.Generator) -> None:
    values = rng.standard_normal(n)
    first = make_first(n, k, rng)
    out = cuda.to_device(np.full(n, -np.inf))
    TASK.run(cuda.to_device(values), cuda.to_device(first), out, n, k)
    cuda.synchronize()
    got = out.copy_to_host()
    expected = reference(values, first, k)
    assert got.tobytes() == expected.tobytes(), f"lag mismatch at n={n} k={k}"


def test_lag() -> None:
    rng = np.random.default_rng(5)
    for n in SIZES:
        for k in LAGS:
            check(n, k, rng)


if __name__ == "__main__":
    test_lag()
    print("lag numba: passed for n in", SIZES, "and k in", LAGS)
