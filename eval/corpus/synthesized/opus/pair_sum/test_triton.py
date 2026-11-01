"""GPU test of the Triton pair_sum kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
# Sizes of the left and right columns.
TABLES = ((1, 1), (1000, 37), (1000003, 999983))


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("pair_sum_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task: ModuleType, rng: np.random.Generator, m: int, l: int, r: int) -> None:  # noqa: E741
    left_idx = rng.integers(0, l, m, dtype=np.int32)
    right_idx = rng.integers(0, r, m, dtype=np.int32)
    left = rng.standard_normal(l) * 1e6
    right = rng.standard_normal(r)
    expected = left[left_idx] + right[right_idx]

    d_left_idx = torch.from_numpy(left_idx).cuda()
    d_right_idx = torch.from_numpy(right_idx).cuda()
    d_left = torch.from_numpy(left).cuda()
    d_right = torch.from_numpy(right).cuda()
    d_out = torch.full((m,), np.nan, dtype=torch.float64, device="cuda")
    task.run(d_left_idx, d_right_idx, d_left, d_right, d_out, m, l, r)
    torch.cuda.synchronize()

    got = d_out.cpu().numpy()
    # Each output is one IEEE addition, so the results must match bit for bit.
    assert np.array_equal(got.view(np.int64), expected.view(np.int64)), f"m={m} l={l} r={r}: output differs"


def test_pair_sum() -> None:
    task = load_task()
    rng = np.random.default_rng(10)
    for m in SIZES:
        for l, r in TABLES:  # noqa: E741
            check(task, rng, m, l, r)
            print(f"pair_sum m={m} l={l} r={r}: ok")


if __name__ == "__main__":
    test_pair_sum()
    print("PASS")
