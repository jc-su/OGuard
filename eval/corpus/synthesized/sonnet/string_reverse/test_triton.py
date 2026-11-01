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



def test(n: int, max_len: int) -> None:
    rng = np.random.default_rng(n * 3 + max_len)
    lens = rng.integers(0, max_len + 1, n)
    if n > 10:
        lens[n // 2] = 5000
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lens)
    total = int(offsets[n])
    chars = rng.integers(0, 256, total, dtype=np.uint8)
    starts = offsets[:-1].astype(np.int64)
    src = np.repeat(starts + lens - 1, lens) - (
        np.arange(total) - np.repeat(starts, lens)
    )
    want = chars[src]
    for i in rng.integers(0, max(n, 1), 20) if n else ():
        assert want[offsets[i] : offsets[i + 1]].tobytes() == (
            chars[offsets[i] : offsets[i + 1]][::-1].tobytes()
        )
    out = torch.full((total,), 9, dtype=torch.uint8, device="cuda")
    task.run(dev(offsets), dev(chars), out, n)
    torch.cuda.synchronize()
    bytes_equal(out, want, f"string_reverse n={n} max_len={max_len}")


if __name__ == "__main__":
    for n in SIZES:
        for max_len in (0, 1, 30, 300):
            test(n, max_len)
    print("string_reverse: all passed")
