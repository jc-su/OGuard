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



def test(n: int, s: int) -> None:
    rng = np.random.default_rng(n * 7 + s)
    sorted_ = np.sort(rng.integers(-1000, 1000, s, dtype=np.int64))
    if s > 2:
        sorted_[0] = INT64_MIN
        sorted_[-1] = 2**63 - 1
        sorted_ = np.sort(sorted_)
    keys = rng.integers(-1100, 1100, n, dtype=np.int64)
    if n > 4:
        keys[0], keys[1], keys[2] = INT64_MIN, 2**63 - 1, 0
        if s:
            keys[3 : 3 + min(s, n - 3)] = sorted_[: min(s, n - 3)]
    want = np.searchsorted(sorted_, keys, side="left").astype(np.int32)
    out = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    task.run(dev(keys), dev(sorted_), out, n, s)
    torch.cuda.synchronize()
    bytes_equal(out, want, f"lower_bound n={n} s={s}")


if __name__ == "__main__":
    for n in SIZES:
        for s in (0, 1, 2, 3, 1000, 1024, 100003):
            test(n, s)
    print("lower_bound: all passed")
