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



def test(n: int, capacity: int) -> None:
    rng = np.random.default_rng(n * 13 + capacity)
    group = rng.integers(0, capacity, n, dtype=np.int32)
    value = rng.integers(INT64_MIN + 1, 2**63 - 1, n, dtype=np.int64)
    if n:
        value[rng.integers(0, n, n // 5 + 1)] = INT64_MIN
    want = np.full(capacity, INT64_MIN, dtype=np.int64)
    np.maximum.at(want, group, value)
    best = torch.full((capacity,), INT64_MIN, dtype=torch.int64, device="cuda")
    task.run(dev(group), dev(value), best, n, capacity)
    torch.cuda.synchronize()
    bytes_equal(best, want, f"group_max n={n} capacity={capacity}")


if __name__ == "__main__":
    for n in SIZES:
        for capacity in (1, 7, 1000, 200000):
            test(n, capacity)
    print("group_max: all passed")
