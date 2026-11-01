"""Test the Numba coalesce kernel on the GPU against a NumPy reference.

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
    spec = importlib.util.spec_from_file_location("coalesce_numba", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = load_task()


def reference(a: np.ndarray, valid: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    is_valid = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    return np.where(is_valid, a, b)


def check(n: int, rng: np.random.Generator) -> None:
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    valid = rng.integers(0, 256, size=(n + 7) // 8, dtype=np.uint8)
    out = cuda.to_device(np.full(n, -np.inf))
    TASK.run(cuda.to_device(a), cuda.to_device(valid), cuda.to_device(b), out, n)
    cuda.synchronize()
    got = out.copy_to_host()
    expected = reference(a, valid, b, n)
    assert got.tobytes() == expected.tobytes(), f"coalesce mismatch at n={n}"


def test_coalesce() -> None:
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(n, rng)


if __name__ == "__main__":
    test_coalesce()
    print("coalesce numba: passed for n in", SIZES)
