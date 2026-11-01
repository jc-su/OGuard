"""Test projection's triton.py.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the result against a NumPy reference.
"""

import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "projection_triton", os.path.join(HERE, "triton.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(task, n, rng):
    price = rng.uniform(0.0, 100.0, size=max(n, 1)).astype(np.float64)
    discount = rng.uniform(0.0, 1.0, size=max(n, 1)).astype(np.float64)
    ref = price[:n] * (1.0 - discount[:n])

    d_price = torch.from_numpy(price).cuda()
    d_discount = torch.from_numpy(discount).cuda()
    d_out = torch.empty(max(n, 1), dtype=torch.float64, device="cuda")

    task.run(d_price, d_discount, d_out, n)
    torch.cuda.synchronize()

    got = d_out.cpu().numpy()[:n]
    assert np.allclose(got, ref), f"n={n}: mismatch"
    print(f"n={n}: PASS")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)


if __name__ == "__main__":
    main()
