"""Test the Triton kernel of bucket_histogram against a NumPy reference.

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
        "bucket_histogram_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(n: int, b: int) -> None:
    values = RNG.integers(-50, 50, n).astype(np.float64)
    values[::101] = np.inf
    values[7::101] = -np.inf
    values[3::5] += 0.5
    bounds = np.sort(RNG.integers(-40, 40, b)).astype(np.float64)
    counts = dev(np.zeros(b + 1, np.int32))
    MOD.run(dev(values), dev(bounds), counts, n, b)
    c = np.searchsorted(bounds, values, side="right")
    ref = np.bincount(c, minlength=b + 1).astype(np.int32)
    got = counts.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n} b={b}"


def test_kernel() -> None:
    for n in SIZES:
        for b in (0, 1, 10, 255, 256, 5000):
            check(n, b)


if __name__ == "__main__":
    test_kernel()
    print("ok")
