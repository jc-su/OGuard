"""Test the projection Numba kernel on the GPU against a NumPy reference.

Run from the parent directory: python -m projection.test_numba
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from numba import cuda

SIZES = (0, 1, 1000, 1000003)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("projection_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, rng: np.random.Generator) -> None:
    price = rng.uniform(900.0, 105000.0, n)
    discount = rng.integers(0, 11, n) / 100.0
    d_price = cuda.to_device(price)
    d_discount = cuda.to_device(discount)
    d_out = cuda.to_device(np.full(n, np.nan))

    IMPL.run(d_price, d_discount, d_out, n)
    cuda.synchronize()

    # The kernel does one subtraction and one multiplication in float64, the
    # same operations as NumPy, so the results are equal bit for bit.
    np.testing.assert_array_equal(d_out.copy_to_host(), price * (1.0 - discount))


def test_projection() -> None:
    rng = np.random.default_rng(3)
    for n in SIZES:
        check(n, rng)


if __name__ == "__main__":
    test_projection()
    print(f"projection numba: pass for n in {SIZES}")
