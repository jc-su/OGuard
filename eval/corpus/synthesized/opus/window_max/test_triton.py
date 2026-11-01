"""GPU test of the Triton window_max kernel against a NumPy reference.

Run with python -P so that triton.py in this directory does not shadow the
triton package.
"""

import importlib.util
import pathlib
from types import ModuleType

import numpy as np
import torch

SIZES = (0, 1, 1000, 1000003)
WINDOWS = (1, 2, 3, 5, 64, 1000, 1023, 1024)
INT64_MIN = np.iinfo(np.int64).min
INT64_MAX = np.iinfo(np.int64).max


def load_task() -> ModuleType:
    path = pathlib.Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("window_max_triton", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(values: np.ndarray, w: int) -> np.ndarray:
    # Direct maximum over every window, with the window start clamped at row 0.
    if values.size == 0:
        return values.copy()
    padded = np.concatenate([np.full(w - 1, INT64_MIN, dtype=np.int64), values])
    return np.lib.stride_tricks.sliding_window_view(padded, w).max(axis=1)


def make_values(rng: np.random.Generator, n: int, pattern: str) -> np.ndarray:
    match pattern:
        case "random":
            values = rng.integers(INT64_MIN, INT64_MAX, n, dtype=np.int64, endpoint=True)
            extremes = rng.random(n) < 0.01
            values[extremes] = rng.choice(
                np.array([INT64_MIN, INT64_MAX], dtype=np.int64), int(extremes.sum())
            )
            return values
        case "descending":
            return np.arange(n, 0, -1, dtype=np.int64) - n // 2
        case _:
            raise ValueError(f"unknown pattern {pattern}")


def check(task: ModuleType, values: np.ndarray, w: int, label: str) -> None:
    n = values.size
    expected = reference(values, w)

    d_values = torch.from_numpy(values).cuda()
    d_out = torch.full((n,), -1, dtype=torch.int64, device="cuda")
    task.run(d_values, d_out, n, w)
    torch.cuda.synchronize()

    got = d_out.cpu().numpy()
    assert np.array_equal(got, expected), f"{label}: output differs"


def test_window_max() -> None:
    task = load_task()
    rng = np.random.default_rng(9)
    block = task.CONSTANTS["BLOCK"]
    for n in SIZES + (block - 1, block, block + 1, 2 * block + 1, 5000):
        for pattern in ("random", "descending"):
            values = make_values(rng, n, pattern)
            for w in WINDOWS:
                check(task, values, w, f"n={n} w={w} {pattern}")
                print(f"window_max n={n} w={w} {pattern}: ok")
    values = torch.zeros(4, dtype=torch.int64, device="cuda")
    out = torch.empty(4, dtype=torch.int64, device="cuda")
    for w in (0, task.CONSTANTS["HALO"] + 1):
        try:
            task.run(values, out, 4, w)
        except ValueError:
            print(f"window_max w={w} rejected: ok")
        else:
            raise AssertionError(f"w={w} was not rejected")


if __name__ == "__main__":
    test_window_max()
    print("PASS")
