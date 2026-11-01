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



def test(n: int, density: float) -> None:
    rng = np.random.default_rng(n * 5 + int(density * 10))
    values = rng.normal(0.0, 100.0, n)
    mask = (rng.random(n) < density).astype(np.int8)
    mask[mask != 0] = rng.integers(-128, 128, int(mask.sum()), dtype=np.int8) | 1
    values[mask == 0] = np.where(rng.random(int((mask == 0).sum())) < 0.1, np.nan, 1e300)
    sel = values[mask != 0]
    want = float(np.sum(sel, dtype=np.longdouble)) if n else 0.0
    scale = float(np.abs(sel).sum()) if len(sel) else 0.0
    result = torch.zeros(1, dtype=torch.float64, device="cuda")
    task.run(dev(values), dev(mask), result, n)
    torch.cuda.synchronize()
    got = result.item()
    assert abs(got - want) <= 1e-11 * scale + 1e-300, (n, density, got, want)


if __name__ == "__main__":
    for n in SIZES:
        for density in (0.0, 0.3, 1.0):
            test(n, density)
    print("masked_sum: all passed")
