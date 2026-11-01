"""Test the Numba gather_nullable kernel on the GPU against a NumPy reference.

Run with `python -P test_numba.py` so that numba.py in this directory does not
shadow the numba package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load_task() -> ModuleType:
    path = Path(__file__).resolve().parent / "numba.py"
    spec = importlib.util.spec_from_file_location("gather_nullable_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(col: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(idx.shape[0], np.nan)
    matched = idx >= 0
    out[matched] = col[idx[matched]]
    return out


def check(m: int, n: int, rng: np.random.Generator) -> None:
    col = rng.standard_normal(n)
    idx = rng.integers(-1, n, size=m, dtype=np.int32)
    out = cuda.to_device(np.full(m, -np.inf))
    TASK.run(cuda.to_device(col), cuda.to_device(idx), out, m, n)
    cuda.synchronize()
    got = out.copy_to_host()
    expected = reference(col, idx)
    assert got.tobytes() == expected.tobytes(), f"gather_nullable mismatch at m={m} n={n}"


def test_gather_nullable() -> None:
    rng = np.random.default_rng(2)
    for size in SIZES:
        check(size, size, rng)
        check(size, size // 2, rng)


if __name__ == "__main__":
    test_gather_nullable()
    print("gather_nullable numba: passed for m in", SIZES)
