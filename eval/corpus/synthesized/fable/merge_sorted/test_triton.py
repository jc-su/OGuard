import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# the task's triton.py must not shadow the triton package
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]


def load_task():
    spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task, a_np: np.ndarray, b_np: np.ndarray, label: str) -> None:
    m = len(a_np)
    p = len(b_np)
    a = torch.from_numpy(a_np).cuda()
    b = torch.from_numpy(b_np).cuda()
    out = torch.full((m + p,), -7, dtype=torch.int64, device="cuda")
    task.run(a, b, out, m, p)
    torch.cuda.synchronize()
    got = out.cpu().numpy()
    ref = np.sort(np.concatenate([a_np, b_np]), kind="stable")
    bad = np.flatnonzero(got != ref)
    assert len(bad) == 0, (m, p, label, bad[:5], got[bad[:5]], ref[bad[:5]])
    print(f"PASS merge_sorted m={m} p={p} {label}")


def sorted_ints(rng: np.random.Generator, size: int, lo: int, hi: int) -> np.ndarray:
    return np.sort(rng.integers(lo, hi, size=size, dtype=np.int64))


def main() -> None:
    rng = np.random.default_rng(9)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        splits = {(n, 0), (0, n), (n // 2, n - n // 2), (n // 10, n - n // 10)}
        for m, p in sorted(splits):
            # wide range: mostly distinct values
            check(task, sorted_ints(rng, m, -(2**62), 2**62), sorted_ints(rng, p, -(2**62), 2**62), "wide")
            # narrow range: many ties inside and across the inputs
            check(task, sorted_ints(rng, m, 0, 50), sorted_ints(rng, p, 0, 50), "ties")
            # disjoint ranges: one input entirely before the other
            check(task, sorted_ints(rng, m, 0, 1000), sorted_ints(rng, p, 1000, 2000), "a_before_b")
            check(task, sorted_ints(rng, m, 1000, 2000), sorted_ints(rng, p, 0, 1000), "b_before_a")
    check(task, np.zeros(1000, dtype=np.int64), np.zeros(1000003, dtype=np.int64), "all_equal")


if __name__ == "__main__":
    main()
