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


def make_runs(rng: np.random.Generator, n: int, r: int) -> np.ndarray:
    """r strictly increasing run ends, the last one n."""
    if r == 0:
        return np.zeros(0, dtype=np.int32)
    cuts = np.sort(rng.choice(np.arange(1, n), size=r - 1, replace=False))
    return np.concatenate([cuts, [n]]).astype(np.int32)


def check(task, n: int, run_ends_np: np.ndarray, rng: np.random.Generator) -> None:
    r = len(run_ends_np)
    run_values_np = rng.standard_normal(r)
    run_values = torch.from_numpy(run_values_np).cuda()
    run_ends = torch.from_numpy(run_ends_np).cuda()
    out = torch.full((n,), float("nan"), dtype=torch.float64, device="cuda")
    task.run(run_values, run_ends, out, r, n)
    torch.cuda.synchronize()
    lengths = np.diff(np.concatenate([[0], run_ends_np]))
    ref = np.repeat(run_values_np, lengths)
    assert np.array_equal(out.cpu().numpy(), ref), (n, r)
    print(f"PASS run_end_decode n={n} r={r}")


def main() -> None:
    rng = np.random.default_rng(3)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        if n == 0:
            check(task, n, make_runs(rng, n, 0), rng)
            continue
        check(task, n, make_runs(rng, n, 1), rng)
        check(task, n, make_runs(rng, n, n), rng)
        if n > 1:
            check(task, n, make_runs(rng, n, min(n, max(2, n // 5))), rng)
            check(task, n, make_runs(rng, n, min(n, 100)), rng)


if __name__ == "__main__":
    main()
