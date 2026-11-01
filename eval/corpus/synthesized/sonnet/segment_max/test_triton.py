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



def test(n: int, g: int, shape: str) -> None:
    rng = np.random.default_rng(n * 31 + g)
    if shape == "random":
        cuts = np.sort(rng.integers(0, n + 1, g - 1)) if g > 1 else np.empty(0, int)
    elif shape == "one_big":
        cuts = np.full(g - 1, n, dtype=np.int64)
    else:
        raise ValueError(shape)
    offsets = np.concatenate([[0], cuts, [n]]).astype(np.int32)
    assert len(offsets) == g + 1
    values = rng.integers(INT64_MIN, 2**63 - 1, n, dtype=np.int64, endpoint=True)
    want = np.full(g, INT64_MIN, dtype=np.int64)
    for j in range(g):
        seg = values[offsets[j] : offsets[j + 1]]
        if len(seg):
            want[j] = seg.max()
    maxes = torch.full((g,), 5, dtype=torch.int64, device="cuda")
    task.run(dev(offsets), dev(values), maxes, g)
    torch.cuda.synchronize()
    bytes_equal(maxes, want, f"segment_max n={n} g={g} {shape}")


if __name__ == "__main__":
    for n in SIZES:
        for g in (1, 7, 1000):
            test(n, g, "random")
            test(n, g, "one_big")
    test(1000003, 300000, "random")
    print("segment_max: all passed")
