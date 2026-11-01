"""Test the Triton kernel of group_max against a NumPy reference.

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
        "group_max_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(n: int, capacity: int) -> None:
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    group = RNG.integers(0, capacity, n, dtype=np.int32)
    value = RNG.integers(lo, hi, n, dtype=np.int64, endpoint=True)
    value[::17] = hi
    best = dev(np.full(capacity, lo, np.int64))
    MOD.run(dev(group), dev(value), best, n, capacity)
    ref = np.full(capacity, lo, np.int64)
    np.maximum.at(ref, group, value)
    got = best.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n} capacity={capacity}"


def test_kernel() -> None:
    for n in SIZES:
        for capacity in (1, 1000, 1000003):
            check(n, capacity)


if __name__ == "__main__":
    test_kernel()
    print("ok")
