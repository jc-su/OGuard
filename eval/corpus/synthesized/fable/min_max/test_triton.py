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


def check(task, values_np: np.ndarray, label: str) -> None:
    n = len(values_np)
    values = torch.from_numpy(values_np).cuda()
    result = torch.tensor([float("inf"), float("-inf")], dtype=torch.float64, device="cuda")
    task.run(values, result, n)
    torch.cuda.synchronize()
    got = result.cpu().numpy()
    if n == 0:
        ref = np.array([np.inf, -np.inf])
    else:
        ref = np.array([values_np.min(), values_np.max()])
    assert np.array_equal(got, ref), (n, label, got, ref)
    print(f"PASS min_max n={n} {label} min={got[0]} max={got[1]}")


def main() -> None:
    rng = np.random.default_rng(5)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        check(task, rng.standard_normal(n) * 1e6, "normal")
        check(task, -np.abs(rng.standard_normal(n)), "all_negative")
        check(task, np.abs(rng.standard_normal(n)) + 1.0, "all_positive")
        if n > 0:
            x = rng.standard_normal(n)
            x[n - 1] = -1e300
            x[0] = 1e300
            check(task, x, "extremes_at_ends")


if __name__ == "__main__":
    main()
