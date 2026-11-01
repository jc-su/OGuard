"""GPU test of the Triton null_count kernel against a NumPy reference.

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
    spec = importlib.util.spec_from_file_location("null_count_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bitmap(rng: np.random.Generator, n: int, pattern: str) -> np.ndarray:
    # The bits after row n - 1 in the last byte are random in every pattern, so
    # the test sees whether the kernel ignores them.
    nbytes = (n + 7) // 8
    bitmap = rng.integers(0, 256, nbytes, dtype=np.uint8)
    match pattern:
        case "random":
            pass
        case "all_valid":
            bitmap[: n // 8] = 0xFF
        case "all_null":
            bitmap[: n // 8] = 0x00
        case _:
            raise ValueError(f"unknown pattern {pattern}")
    if n % 8 != 0:
        tail = np.uint8((1 << (n % 8)) - 1)
        match pattern:
            case "all_valid":
                bitmap[-1] |= tail
            case "all_null":
                bitmap[-1] &= ~tail
            case _:
                pass
    return bitmap


def check(task: ModuleType, rng: np.random.Generator, n: int, pattern: str) -> None:
    bitmap = make_bitmap(rng, n, pattern)
    bits = np.unpackbits(bitmap, bitorder="little")[:n]
    expected = int(n - bits.sum(dtype=np.int64))

    d_valid = torch.from_numpy(bitmap).cuda()
    d_count = torch.zeros(1, dtype=torch.int64, device="cuda")
    task.run(d_valid, d_count, n)
    torch.cuda.synchronize()

    got = int(d_count.cpu().item())
    assert got == expected, f"n={n} {pattern}: null count {got} != {expected}"


def test_null_count() -> None:
    task = load_task()
    rng = np.random.default_rng(4)
    rows_per_program = 8 * task.CONSTANTS["BLOCK"]
    for n in SIZES + (7, 8, 9, rows_per_program, rows_per_program + 1):
        for pattern in ("random", "all_valid", "all_null"):
            check(task, rng, n, pattern)
            print(f"null_count n={n} {pattern}: ok")


if __name__ == "__main__":
    test_null_count()
    print("PASS")
