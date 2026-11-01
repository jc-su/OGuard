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


def check(task, keys_np: np.ndarray, members_np: np.ndarray, label: str) -> None:
    n = len(keys_np)
    s = len(members_np)
    keys = torch.from_numpy(keys_np).cuda()
    members = torch.from_numpy(members_np).cuda()
    match = torch.full((n,), 7, dtype=torch.int8, device="cuda")
    task.run(keys, members, match, n, s)
    torch.cuda.synchronize()
    got = match.cpu().numpy()
    ref = np.isin(keys_np, members_np).astype(np.int8)
    assert np.array_equal(got, ref), (n, s, label)
    print(f"PASS sorted_contains n={n} s={s} {label} hits={int(ref.sum())}")


def main() -> None:
    rng = np.random.default_rng(10)
    task = load_task()
    lo, hi = -(2**62), 2**62
    for n, s in ((0, 0), (0, 5), (1, 1), (1, 0), (1000, 500), (1000003, 300000), (1000003, 1)):
        members_np = np.unique(rng.integers(lo, hi, size=s, dtype=np.int64))
        keys_np = rng.integers(lo, hi, size=n, dtype=np.int64)
        if s > 0 and n > 0:
            pick = rng.random(n) < 0.5
            keys_np[pick] = members_np[rng.integers(0, len(members_np), size=int(pick.sum()))]
        check(task, keys_np, members_np, "random")
        if n > 0 and s > 0:
            edge = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)
            members_edge = np.unique(np.concatenate([members_np, edge]))
            keys_edge = keys_np.copy()
            keys_edge[: min(n, 2)] = edge[: min(n, 2)]
            check(task, keys_edge, members_edge, "int64_limits")


if __name__ == "__main__":
    main()
