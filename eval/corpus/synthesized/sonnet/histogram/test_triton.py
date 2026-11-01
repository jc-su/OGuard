"""Test histogram's triton.py.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003}, with a
couple of shift values, and checks the result against a NumPy
reference.
"""

import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "histogram_triton", os.path.join(HERE, "triton.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(keys, shift):
    bins = (keys.astype(np.uint64) >> np.uint64(shift)) & np.uint64(255)
    return np.bincount(bins.astype(np.int64), minlength=256).astype(np.int32)


def check(task, n, shift, rng):
    keys = rng.integers(0, 2**32, size=max(n, 1), dtype=np.int64).astype(np.uint32)

    ref = reference(keys[:n], shift) if n > 0 else np.zeros(256, dtype=np.int32)

    d_keys = torch.from_numpy(keys).cuda()
    d_counts = torch.zeros(256, dtype=torch.int32, device="cuda")

    task.run(d_keys, d_counts, shift, n)
    torch.cuda.synchronize()

    got = d_counts.cpu().numpy()
    assert np.array_equal(got, ref), f"n={n}, shift={shift}: mismatch"
    print(f"n={n}, shift={shift}: PASS (total={int(got.sum())})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        for shift in (0, 24):
            check(task, n, shift, rng)


if __name__ == "__main__":
    main()
