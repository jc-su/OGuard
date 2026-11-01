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



def test(m: int, n: int) -> None:
    rng = np.random.default_rng(m * 17 + n)
    idx = rng.integers(0, max(n, 1), m, dtype=np.int32) if n else np.empty(0, np.int32)
    values = rng.normal(0.0, 100.0, m)
    want = np.zeros(n)
    scale = np.zeros(n)
    if m:
        np.add.at(want, idx, values)
        np.add.at(scale, idx, np.abs(values))
    out = torch.zeros(n, dtype=torch.float64, device="cuda")
    task.run(dev(idx), dev(values), out, m, n)
    torch.cuda.synchronize()
    got = out.cpu().numpy()
    assert np.all(np.abs(got - want) <= 1e-12 * scale), (m, n)


if __name__ == "__main__":
    for m in SIZES:
        for n in (1, 7, 1000, 200000):
            test(m, n)
    test(0, 0)
    print("scatter_add: all passed")
