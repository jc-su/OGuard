"""Test the Numba bucketize kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.

Each size runs with b in BOUNDS. Bounds and values are rounded to one decimal,
so bounds repeat and values often equal a bound. The bounds start with -inf
when b >= 3, and a few values are -inf or +inf.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)
BOUNDS = (0, 1, 37, 100000)


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("bucketize_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def make_inputs(n: int, b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.sort(np.round(rng.standard_normal(b) * 3.0, 1))
    if b >= 3:
        bounds[0] = -np.inf
    values = np.round(rng.standard_normal(n) * 3.5, 1)
    special = rng.random(n)
    values[special < 0.01] = -np.inf
    values[special > 0.99] = np.inf
    return values, bounds


def reference(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.searchsorted(bounds, values, side="right").astype(np.int32)


def check(n: int, b: int, rng: np.random.Generator) -> None:
    values, bounds = make_inputs(n, b, rng)
    expected = reference(values, bounds)
    out = cuda.to_device(~expected)
    TASK.run(cuda.to_device(values), cuda.to_device(bounds), out, n, b)
    cuda.synchronize()
    got = out.copy_to_host()
    assert got.tobytes() == expected.tobytes(), f"bucketize mismatch at n={n} b={b}"


def test_bucketize() -> None:
    rng = np.random.default_rng(10)
    for n in SIZES:
        for b in BOUNDS:
            check(n, b, rng)


if __name__ == "__main__":
    test_bucketize()
    print("bucketize numba: passed for n in", SIZES, "and b in", BOUNDS)
