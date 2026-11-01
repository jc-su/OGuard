"""Test the Triton kernel of segment_max against a NumPy reference.

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
        "segment_max_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(g: int) -> None:
    lengths = RNG.integers(0, 9, g)
    lengths[RNG.random(g) < 0.2] = 0
    if g >= 1000:
        lengths[g // 2] = 5000
    offsets = np.zeros(g + 1, np.int32)
    offsets[1:] = np.cumsum(lengths)
    n = int(offsets[-1])
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    values = RNG.integers(lo, hi, n, dtype=np.int64, endpoint=True)
    values[::11] = lo
    maxes = dev(np.full(g, 12345, np.int64))
    MOD.run(dev(offsets), dev(values), maxes, g)
    ref = np.full(g, lo, np.int64)
    nonempty = lengths > 0
    if n > 0:
        ref[nonempty] = np.maximum.reduceat(values, offsets[:-1][nonempty])
    got = maxes.cpu().numpy()
    assert np.array_equal(got, ref), f"g={g}"


def test_kernel() -> None:
    for g in SIZES:
        check(g)


if __name__ == "__main__":
    test_kernel()
    print("ok")
