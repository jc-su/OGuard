"""GPU test of the Triton hash_partition kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
PARTITIONS = (1, 2, 7, 64, 1000, 1024)
MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("hash_partition_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task: ModuleType, rng: np.random.Generator, n: int, p: int) -> None:
    keys = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, n, dtype=np.int64, endpoint=True)
    keys[: min(n, 4)] = np.array([0, -1, np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)[
        : min(n, 4)
    ]
    # NumPy multiplies uint64 arrays modulo 2^64.
    hashed = (keys.view(np.uint64) * MULTIPLIER) >> np.uint64(32)
    expected_part = (hashed % np.uint64(p)).astype(np.int32)
    expected_counts = np.bincount(expected_part, minlength=p).astype(np.int32)

    d_keys = torch.from_numpy(keys).cuda()
    d_part = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    d_counts = torch.zeros(p, dtype=torch.int32, device="cuda")
    task.run(d_keys, d_part, d_counts, n, p)
    torch.cuda.synchronize()

    assert np.array_equal(d_part.cpu().numpy(), expected_part), f"n={n} p={p}: part differs"
    assert np.array_equal(d_counts.cpu().numpy(), expected_counts), f"n={n} p={p}: counts differ"


def test_hash_partition() -> None:
    task = load_task()
    rng = np.random.default_rng(7)
    rows_per_program = task.CONSTANTS["BLOCK"] * task.CONSTANTS["TILES"]
    for n in SIZES + (rows_per_program - 1, rows_per_program, rows_per_program + 1):
        for p in PARTITIONS:
            check(task, rng, n, p)
            print(f"hash_partition n={n} p={p}: ok")
    keys = torch.zeros(4, dtype=torch.int64, device="cuda")
    part = torch.empty(4, dtype=torch.int32, device="cuda")
    for p in (0, task.CONSTANTS["BINS"] + 1):
        counts = torch.zeros(max(p, 1), dtype=torch.int32, device="cuda")
        try:
            task.run(keys, part, counts, 4, p)
        except ValueError:
            print(f"hash_partition p={p} rejected: ok")
        else:
            raise AssertionError(f"p={p} was not rejected")


if __name__ == "__main__":
    test_hash_partition()
    print("PASS")
