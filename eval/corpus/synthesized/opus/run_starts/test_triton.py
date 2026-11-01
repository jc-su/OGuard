"""Test the Triton kernel of run_starts against a NumPy reference.

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
        "run_starts_triton", Path(__file__).with_name("triton.py")
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
    values = RNG.integers(-2, 2, n, dtype=np.int64)
    values[::7] = np.iinfo(np.int64).min
    flags = dev(np.full(n, 7, np.int8))
    MOD.run(dev(values), flags, n)
    ref = np.ones(n, np.int8)
    ref[1:] = (values[1:] != values[:-1]).astype(np.int8)
    got = flags.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n}"


def test_kernel() -> None:
    for n in SIZES:
        check(n)


if __name__ == "__main__":
    test_kernel()
    print("ok")
