import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory must not shadow the triton package.
sys.path = [p for p in sys.path if p not in ("", HERE)]

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


LIM = 2**61


def reference(left, right, n, r, tolerance):
    j = np.searchsorted(right, left, side="right").astype(np.int64) - 1
    if r == 0:
        return np.full(n, -1, dtype=np.int32)
    near = right[np.clip(j, 0, r - 1)]
    found = (j >= 0) & ((left - near) <= tolerance)
    return np.where(found, j, -1).astype(np.int32)


def make_case(rng, n, r, kind):
    if kind == "dense":
        right = np.sort(rng.integers(-LIM, LIM, r, dtype=np.int64, endpoint=True))
        left = rng.integers(-LIM, LIM, n, dtype=np.int64, endpoint=True)
        tolerance = 2**60
    elif kind == "near":
        right = np.sort(rng.integers(-1000, 1000, r, dtype=np.int64))
        left = rng.integers(-1100, 1100, n, dtype=np.int64)
        tolerance = 5
    elif kind == "zero_tolerance":
        right = np.sort(rng.integers(-100, 100, r, dtype=np.int64))
        left = rng.integers(-100, 100, n, dtype=np.int64)
        tolerance = 0
    elif kind == "max_tolerance":
        right = np.sort(rng.integers(-LIM, LIM, r, dtype=np.int64, endpoint=True))
        left = rng.integers(-LIM, LIM, n, dtype=np.int64, endpoint=True)
        tolerance = 2**62
    elif kind == "extremes":
        right = np.sort(np.concatenate([[-LIM, LIM], rng.integers(-LIM, LIM, max(r - 2, 0), dtype=np.int64)]))[:r]
        left = np.concatenate([[-LIM, LIM, 0], rng.integers(-LIM, LIM, max(n - 3, 0), dtype=np.int64)])[:n]
        tolerance = 2**62
    elif kind == "all_equal":
        right = np.full(r, 7, dtype=np.int64)
        left = np.full(n, 7, dtype=np.int64)
        tolerance = 0
    else:
        raise ValueError(kind)
    return left.astype(np.int64), right.astype(np.int64), tolerance


def main():
    rng = np.random.default_rng(4)
    failed = 0
    for n in (0, 1, 1000, 1000003):
        for r in (0, 1, 1000, 1000003):
            for kind in ("dense", "near", "zero_tolerance", "max_tolerance", "extremes", "all_equal"):
                left, right, tolerance = make_case(rng, n, r, kind)
                t_out = torch.full((n,), -5, dtype=torch.int32, device="cuda")
                task.run(
                    torch.from_numpy(left).cuda(), torch.from_numpy(right).cuda(), t_out, n, r, tolerance
                )
                torch.cuda.synchronize()
                ok = np.array_equal(t_out.cpu().numpy(), reference(left, right, n, r, tolerance))
                failed += not ok
                print(("PASS" if ok else "FAIL"), "asof_join triton", n, r, kind)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
