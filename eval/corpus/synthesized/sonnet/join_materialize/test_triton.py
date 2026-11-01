import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The task directory holds triton.py, which would shadow the triton package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

I64 = np.iinfo(np.int64)


def make_offsets(build, probe):
    left = np.searchsorted(build, probe, side="left")
    right = np.searchsorted(build, probe, side="right")
    counts = right - left
    offsets = np.zeros(len(probe) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    assert offsets[-1] <= 2**31 - 1
    return left, counts, offsets.astype(np.int32)


def reference(left, counts, offsets):
    total = int(offsets[-1])
    p = len(counts)
    out_probe = np.repeat(np.arange(p, dtype=np.int32), counts)
    j = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1].astype(np.int64), counts)
    out_build = (np.repeat(left, counts) + j).astype(np.int32)
    return out_build, out_probe


def cases(n, rng):
    # (name, build, probe)
    yield "unique", np.sort(rng.integers(-n, n + 1, size=n, dtype=np.int64)), rng.integers(-n, n + 1, size=n, dtype=np.int64)
    yield "empty_build", np.zeros(0, dtype=np.int64), rng.integers(0, 10, size=n, dtype=np.int64)
    yield "empty_probe", np.sort(rng.integers(0, 10, size=n, dtype=np.int64)), np.zeros(0, dtype=np.int64)
    dup = np.sort(rng.integers(0, max(1, n // 10), size=n, dtype=np.int64))
    yield "duplicates", dup, rng.integers(-1, max(1, n // 10) + 1, size=max(1, n // 10) if n else 0, dtype=np.int64)
    # Every key has 40 matches, so every row takes the cooperative path.
    keys = np.repeat(np.arange(max(1, n // 40), dtype=np.int64), 40)[:n]
    yield "wide_rows", keys, rng.integers(0, max(1, n // 40), size=min(n, 20000), dtype=np.int64)
    # One key with many matches among rows with none.
    heavy = np.sort(np.concatenate([np.full(min(n, 5000), 5, dtype=np.int64), rng.integers(100, 200, size=n, dtype=np.int64)]))[:n]
    probe = rng.integers(0, 10, size=n, dtype=np.int64)
    yield "skew", heavy, probe[: min(n, 3000)] if n else probe
    extreme = np.array([I64.min, I64.min, I64.max, I64.max, 0], dtype=np.int64)
    yield "extreme", np.sort(extreme), np.array([I64.min, I64.max, 0, 1, I64.min + 1], dtype=np.int64)


def main():
    rng = np.random.default_rng(1)
    failed = False
    for n in (0, 1, 1000, 1000003):
        for name, build, probe in cases(n, rng):
            left, counts, offsets = make_offsets(build, probe)
            total = int(offsets[-1])
            exp_build, exp_probe = reference(left, counts, offsets)
            t_build = torch.from_numpy(build).cuda()
            t_probe = torch.from_numpy(probe).cuda()
            t_offsets = torch.from_numpy(offsets).cuda()
            t_ob = torch.full((total,), -7, dtype=torch.int32, device="cuda")
            t_op = torch.full((total,), -7, dtype=torch.int32, device="cuda")
            task.run(t_build, t_probe, t_offsets, t_ob, t_op, len(build), len(probe))
            torch.cuda.synchronize()
            ok = np.array_equal(t_ob.cpu().numpy(), exp_build) and np.array_equal(t_op.cpu().numpy(), exp_probe)
            print(f"join_materialize triton n={n} {name} b={len(build)} p={len(probe)} total={total}: {'PASS' if ok else 'FAIL'}")
            failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
