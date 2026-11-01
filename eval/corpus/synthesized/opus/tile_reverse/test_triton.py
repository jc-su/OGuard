"""Test the Triton kernel of tile_reverse against a NumPy reference.

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
        "tile_reverse_triton", Path(__file__).with_name("triton.py")
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
    values = RNG.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, n, dtype=np.int32)
    out = dev(np.full(n, -1, np.int32))
    MOD.run(dev(values), out, n)
    i = np.arange(n)
    base = i // 1024 * 1024
    length = np.minimum(1024, n - base)
    ref = np.empty(n, np.int32)
    ref[base + length - 1 - (i - base)] = values
    got = out.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n}"


def test_kernel() -> None:
    for n in (*SIZES, 1023, 1024, 1025, 2048):
        check(n)


if __name__ == "__main__":
    test_kernel()
    print("ok")
