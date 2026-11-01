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



def test(n: int) -> None:
    rng = np.random.default_rng(n)
    values = rng.integers(-(2**31), 2**31, n, dtype=np.int32)
    want = np.concatenate(
        [values[i : i + 1024][::-1] for i in range(0, n, 1024)]
        or [np.empty(0, np.int32)]
    )
    out = torch.full((n,), 5, dtype=torch.int32, device="cuda")
    task.run(dev(values), out, n)
    torch.cuda.synchronize()
    bytes_equal(out, want, f"tile_reverse n={n}")


if __name__ == "__main__":
    for n in (*SIZES, 1023, 1024, 1025, 2048, 5000):
        test(n)
    print("tile_reverse: all passed")
