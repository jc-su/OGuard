"""Test the q6_sum Triton kernel on the GPU against a NumPy reference.

Run from the parent directory: python -P q6_sum/test_triton.py. The -P flag
keeps this directory off sys.path, so its triton.py and numba.py do not
shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)


class Params(NamedTuple):
    date_lo: int
    date_hi: int
    disc_lo: float
    disc_hi: float
    qty_max: float


# TPC-H Q6 with dates as days since 1970-01-01: shipdate in
# [1994-01-01, 1995-01-01), discount in [0.05, 0.07], quantity < 24.
Q6 = Params(8766, 9131, 0.05, 0.07, 24.0)
# Bounds that keep every row.
ALL_ROWS = Params(-(2**31), 2**31 - 1, -np.inf, np.inf, np.inf)


def load_impl() -> ModuleType:
    path = Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("q6_sum_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = load_impl()


def check(n: int, p: Params, rng: np.random.Generator) -> None:
    # Days of 1992-01-01 to 1998-12-31, as in TPC-H lineitem.
    shipdate = rng.integers(8035, 10592, n, dtype=np.int32)
    discount = rng.integers(0, 11, n) / 100.0
    quantity = rng.integers(1, 51, n).astype(np.float64)
    price = rng.uniform(900.0, 105000.0, n)
    d_result = torch.zeros(1, dtype=torch.float64, device="cuda")

    IMPL.run(
        torch.from_numpy(shipdate).cuda(),
        torch.from_numpy(discount).cuda(),
        torch.from_numpy(quantity).cuda(),
        torch.from_numpy(price).cuda(),
        p.date_lo,
        p.date_hi,
        p.disc_lo,
        p.disc_hi,
        p.qty_max,
        d_result,
        n,
    )
    torch.cuda.synchronize()

    keep = (
        (p.date_lo <= shipdate)
        & (shipdate < p.date_hi)
        & (p.disc_lo <= discount)
        & (discount <= p.disc_hi)
        & (quantity < p.qty_max)
    )
    expected = np.sum(price[keep] * discount[keep])
    got = d_result.item()
    # The GPU adds the terms in a different order, so allow rounding error.
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


def test_q6_sum() -> None:
    rng = np.random.default_rng(4)
    for n in SIZES:
        for p in (Q6, ALL_ROWS):
            check(n, p, rng)


if __name__ == "__main__":
    test_q6_sum()
    print(f"q6_sum triton: pass for n in {SIZES}")
