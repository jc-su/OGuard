"""Test the Triton kernel of masked_sum against a NumPy reference.

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
        "masked_sum_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()
RNG = np.random.default_rng(3)


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def check(n: int, initial: float) -> None:
    values = RNG.standard_normal(n) * 1e3
    mask = RNG.choice(np.array([-3, 0, 0, 1, 5], np.int8), n)
    values[(mask == 0) & (RNG.random(n) < 0.1)] = np.nan
    result = dev(np.array([initial]))
    MOD.run(dev(values), dev(mask), result, n)
    keep = values[mask != 0]
    ref = initial + keep.sum()
    got = float(result.cpu().numpy()[0])
    scale = abs(initial) + np.abs(keep).sum()
    assert abs(got - ref) <= 1e-12 * scale, f"n={n} got={got} ref={ref}"


def test_kernel() -> None:
    for n in SIZES:
        check(n, 0.0)
    check(1000003, 2.5)


if __name__ == "__main__":
    test_kernel()
    print("ok")
