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



def test(n: int, span: int) -> None:
    rng = np.random.default_rng(n + span)
    values = rng.integers(0, span, n, dtype=np.int64)
    if n:
        values[rng.integers(0, n, n // 3 + 1)] = INT64_MIN
    want = np.ones(n, dtype=np.int8)
    want[1:] = values[1:] != values[:-1]
    flags = torch.full((n,), 7, dtype=torch.int8, device="cuda")
    task.run(dev(values), flags, n)
    torch.cuda.synchronize()
    bytes_equal(flags, want, f"run_starts n={n} span={span}")


if __name__ == "__main__":
    for n in SIZES:
        for span in (1, 3, 2**40):
            test(n, span)
    print("run_starts: all passed")
