"""Run on GPU 1 from any directory: CUDA_VISIBLE_DEVICES=1 python -P test_triton.py"""

import importlib.util
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("task_triton", HERE / "triton.py")
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)

SIZES = (0, 1, 1000, 1000003)
INT64_MIN = np.iinfo(np.int64).min


def dev(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def bytes_equal(got: torch.Tensor, want: np.ndarray, what: str) -> None:
    got_np = got.cpu().numpy()
    assert got_np.dtype == want.dtype, (what, got_np.dtype, want.dtype)
    assert got_np.tobytes() == want.tobytes(), what



def test(n: int, b: int) -> None:
    rng = np.random.default_rng(n * 11 + b)
    bounds = np.sort(rng.normal(0.0, 1.0, b))
    if b > 3:
        bounds[1] = bounds[2]
    values = rng.normal(0.0, 1.5, n)
    if n and b:
        take = min(n, 50)
        values[:take] = bounds[rng.integers(0, b, take)]
    if n > 60:
        values[50] = -np.inf
        values[51] = np.inf
    want = np.bincount(
        np.searchsorted(bounds, values, side="right"), minlength=b + 1
    ).astype(np.int32)
    counts = torch.zeros(b + 1, dtype=torch.int32, device="cuda")
    task.run(dev(values), dev(bounds), counts, n, b)
    torch.cuda.synchronize()
    bytes_equal(counts, want, f"bucket_histogram n={n} b={b}")


if __name__ == "__main__":
    for n in SIZES:
        for b in (0, 1, 2, 5, 255, 1000):
            test(n, b)
    print("bucket_histogram: all passed")
