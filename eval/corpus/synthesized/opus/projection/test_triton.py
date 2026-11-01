"""Test the projection Triton kernel on the GPU against a NumPy reference.

Run from the parent directory: python -P projection/test_triton.py. The -P
flag keeps this directory off sys.path, so its triton.py and numba.py do not
shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("projection_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, rng: np.random.Generator) -> None:
    price = rng.uniform(900.0, 105000.0, n)
    discount = rng.integers(0, 11, n) / 100.0
    d_price = torch.from_numpy(price).cuda()
    d_discount = torch.from_numpy(discount).cuda()
    d_out = torch.full((n,), np.nan, dtype=torch.float64, device="cuda")

    IMPL.run(d_price, d_discount, d_out, n)
    torch.cuda.synchronize()

    # The kernel does one subtraction and one multiplication in float64, the
    # same operations as NumPy, so the results are equal bit for bit.
    np.testing.assert_array_equal(d_out.cpu().numpy(), price * (1.0 - discount))


def test_projection() -> None:
    rng = np.random.default_rng(3)
    for n in SIZES:
        check(n, rng)


if __name__ == "__main__":
    test_projection()
    print(f"projection triton: pass for n in {SIZES}")
