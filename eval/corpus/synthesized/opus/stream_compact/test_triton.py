"""GPU test of the Triton stream_compact kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("stream_compact_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_keep(rng: np.random.Generator, n: int, pattern: str) -> np.ndarray:
    match pattern:
        case "random":
            # Nonzero flags other than 1 must also count as kept.
            return rng.choice(np.array([0, 0, 1, -1, 7, 127, -128], dtype=np.int8), n)
        case "none":
            return np.zeros(n, dtype=np.int8)
        case "all":
            return np.ones(n, dtype=np.int8)
        case "sparse":
            return (rng.random(n) < 0.01).astype(np.int8)
        case _:
            raise ValueError(f"unknown pattern {pattern}")


def check(task: ModuleType, rng: np.random.Generator, n: int, pattern: str) -> None:
    values = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, n, dtype=np.int64, endpoint=True)
    keep = make_keep(rng, n, pattern)
    expected = np.sort(values[keep != 0])

    d_values = torch.from_numpy(values).cuda()
    d_keep = torch.from_numpy(keep).cuda()
    d_out = torch.full((n,), -1, dtype=torch.int64, device="cuda")
    d_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    task.run(d_values, d_keep, d_out, d_count, n)
    torch.cuda.synchronize()

    count = int(d_count.cpu().item())
    assert count == expected.size, f"n={n} {pattern}: count {count} != {expected.size}"
    got = np.sort(d_out.cpu().numpy()[:count])
    assert np.array_equal(got, expected), f"n={n} {pattern}: kept values differ"


def test_stream_compact() -> None:
    task = load_task()
    rng = np.random.default_rng(1)
    block = task.CONSTANTS["BLOCK"]
    for n in SIZES + (block - 1, block, block + 1, 2 * block + 1):
        for pattern in ("random", "none", "all", "sparse"):
            check(task, rng, n, pattern)
            print(f"stream_compact n={n} keep={pattern}: ok")


if __name__ == "__main__":
    test_stream_compact()
    print("PASS")
