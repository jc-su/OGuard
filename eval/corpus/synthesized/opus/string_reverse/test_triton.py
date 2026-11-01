"""Test the Triton kernel of string_reverse against a NumPy reference.

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
        "string_reverse_triton", Path(__file__).with_name("triton.py")
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
    lengths = RNG.integers(0, 21, n)
    lengths[RNG.random(n) < 0.2] = 0
    if n >= 1000:
        lengths[n // 3] = 5000
    offsets = np.zeros(n + 1, np.int32)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])
    chars = RNG.integers(0, 256, total, dtype=np.uint8)
    out = dev(np.zeros(total, np.uint8))
    MOD.run(dev(offsets), dev(chars), out, n)
    p = np.arange(total)
    row = np.searchsorted(offsets, p, side="right") - 1
    ref = np.empty(total, np.uint8)
    ref[offsets[row] + offsets[row + 1] - 1 - p] = chars
    got = out.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n}"
    if n > 0:
        k = int(np.argmax(lengths))
        a, b = int(offsets[k]), int(offsets[k + 1])
        assert np.array_equal(got[a:b], chars[a:b][::-1])


def test_kernel() -> None:
    for n in SIZES:
        check(n)


if __name__ == "__main__":
    test_kernel()
    print("ok")
