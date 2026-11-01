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


def reference(values_np: np.ndarray, window: int) -> np.ndarray:
    n = len(values_np)
    if n == 0:
        return np.zeros(0)
    padded = np.concatenate([np.zeros(window - 1), values_np])
    return np.lib.stride_tricks.sliding_window_view(padded, window).sum(axis=1)


def check(task, values_np: np.ndarray, window: int, label: str) -> None:
    n = len(values_np)
    values = torch.from_numpy(values_np).cuda()
    out = torch.full((n,), float("nan"), dtype=torch.float64, device="cuda")
    task.run(values, out, n, window)
    torch.cuda.synchronize()
    got = out.cpu().numpy()
    ref = reference(values_np, window)
    scale = max(1.0, float(np.abs(ref).max()) if n > 0 else 1.0)
    err = float(np.abs(got - ref).max()) if n > 0 else 0.0
    assert err <= 1e-11 * scale, (n, window, label, err, scale)
    print(f"PASS rolling_sum n={n} window={window} {label} max_abs_err={err:.3e}")


def main() -> None:
    rng = np.random.default_rng(6)
    task = load_task()
    for n in (0, 1, 1000, 1000003):
        values_np = rng.uniform(-1.0, 1.0, size=n)
        for window in (1, 2, 7, 100, 1023, 1024):
            check(task, values_np, window, "uniform")
        check(task, rng.standard_normal(n) * 1e6, 1024, "large_values")
        check(task, np.ones(n), 1024, "ones")


if __name__ == "__main__":
    main()
