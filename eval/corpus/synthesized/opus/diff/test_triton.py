"""Test the Triton kernel of diff against a NumPy reference.

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
        "diff_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(n: int) -> None:
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    values = RNG.integers(lo, hi, n, dtype=np.int64, endpoint=True)
    out = dev(np.full(n, 99, np.int64))
    MOD.run(dev(values), out, n)
    ref = values.copy()
    ref[1:] = values[1:] - values[:-1]
    got = out.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n}"


def test_kernel() -> None:
    with np.errstate(over="ignore"):
        for n in SIZES:
            check(n)


if __name__ == "__main__":
    test_kernel()
    print("ok")
