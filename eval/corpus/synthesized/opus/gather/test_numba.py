"""Test the gather Numba kernel on the GPU against a NumPy reference.

Run from the parent directory: python -m gather.test_numba
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("gather_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, m: int, rng: np.random.Generator) -> None:
    col = rng.standard_normal(n)
    sel = rng.integers(0, n, m, dtype=np.int32) if m > 0 else np.empty(0, np.int32)
    d_col = cuda.to_device(col)
    d_sel = cuda.to_device(sel)
    d_out = cuda.to_device(np.full(m, np.nan))

    IMPL.run(d_col, d_sel, d_out, m, n)
    cuda.synchronize()

    np.testing.assert_array_equal(d_out.copy_to_host(), col[sel])


def test_gather() -> None:
    rng = np.random.default_rng(2)
    for n in SIZES:
        # sel can hold row ids only when col has rows.
        ms = sorted({n, n // 2, 2 * n}) if n > 0 else [0]
        for m in ms:
            check(n, m, rng)


if __name__ == "__main__":
    test_gather()
    print(f"gather numba: pass for n in {SIZES}")
