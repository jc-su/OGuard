"""GPU test of the Triton prefix_offsets kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
MAX_LENGTH = 1 << 20


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("prefix_offsets_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_lengths(rng: np.random.Generator, n: int, pattern: str) -> np.ndarray:
    match pattern:
        case "short":
            return rng.integers(0, 16, n, dtype=np.int32)
        case "full_range":
            # Large n makes the int32 offsets wrap; the reference wraps the same way.
            return rng.integers(0, MAX_LENGTH, n, dtype=np.int32, endpoint=True)
        case "max":
            return np.full(n, MAX_LENGTH, dtype=np.int32)
        case _:
            raise ValueError(f"unknown pattern {pattern}")


def check(task: ModuleType, rng: np.random.Generator, n: int, pattern: str) -> None:
    lengths = make_lengths(rng, n, pattern)
    expected = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, dtype=np.int64, out=expected[1:])
    expected = expected.astype(np.int32)

    d_lengths = torch.from_numpy(lengths).cuda()
    d_offsets = torch.full((n + 1,), -1, dtype=torch.int32, device="cuda")
    task.run(d_lengths, d_offsets, n)
    torch.cuda.synchronize()

    got = d_offsets.cpu().numpy()
    assert np.array_equal(got, expected), f"n={n} {pattern}: offsets differ"


def test_prefix_offsets() -> None:
    task = load_task()
    rng = np.random.default_rng(5)
    # 33 tiles and a partial tile split unevenly over the programs.
    block = task.CONSTANTS["BLOCK"]
    for n in SIZES + (block - 1, block, block + 1, 33 * block + 5):
        for pattern in ("short", "full_range", "max"):
            check(task, rng, n, pattern)
            print(f"prefix_offsets n={n} {pattern}: ok")


if __name__ == "__main__":
    test_prefix_offsets()
    print("PASS")
