"""Test the Triton kernel of lower_bound against a NumPy reference.

Run with `python -P test_triton.py` so that triton.py in this directory does not
shadow the triton package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lower_bound_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(n: int, s: int) -> None:
    keys = RNG.integers(-1200, 1200, n, dtype=np.int64)
    keys[::13] = np.iinfo(np.int64).min
    keys[5::13] = np.iinfo(np.int64).max
    sorted_ = np.sort(RNG.integers(-1000, 1000, s, dtype=np.int64))
    out = dev(np.full(n, -7, np.int32))
    MOD.run(dev(keys), dev(sorted_), out, n, s)
    ref = np.searchsorted(sorted_, keys, side="left").astype(np.int32)
    got = out.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n} s={s}"


def test_kernel() -> None:
    for n in SIZES:
        for s in (0, 1, 2, 7, 1000, 1000003):
            check(n, s)


if __name__ == "__main__":
    test_kernel()
    print("ok")
