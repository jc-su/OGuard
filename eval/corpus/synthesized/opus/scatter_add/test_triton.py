"""Test the Triton kernel of scatter_add against a NumPy reference.

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
        "scatter_add_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(m: int, n: int) -> None:
    idx = RNG.integers(0, n, m, dtype=np.int32)
    values = RNG.standard_normal(m)
    out = dev(np.zeros(n))
    MOD.run(dev(idx), dev(values), out, m, n)
    ref = np.bincount(idx, weights=values, minlength=n)
    scale = np.bincount(idx, weights=np.abs(values), minlength=n)
    got = out.cpu().numpy()
    assert np.all(np.abs(got - ref) <= 1e-12 * scale), f"m={m} n={n}"


def test_kernel() -> None:
    check(0, 0)
    for m in SIZES:
        for n in (1, 1000, 1000003):
            check(m, n)


if __name__ == "__main__":
    test_kernel()
    print("ok")
