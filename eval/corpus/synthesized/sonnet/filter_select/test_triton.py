"""Test filter_select's triton.py.

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
        "filter_select_triton", os.path.join(HERE, "triton.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(quantity, threshold):
    ids = np.nonzero(quantity < threshold)[0].astype(np.int32)
    return ids, np.int32(ids.shape[0])


def check(task, n, rng):
    quantity = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    threshold = 0.0

    ref_ids, ref_count = reference(quantity, threshold)

    d_quantity = torch.from_numpy(
        quantity if n > 0 else np.zeros(1, dtype=np.float64)
    ).cuda()
    d_sel = torch.empty(max(n, 1), dtype=torch.int32, device="cuda")
    d_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    task.run(d_quantity, d_sel, d_count, threshold, n)
    torch.cuda.synchronize()

    got_count = d_count.cpu().numpy()[0]
    assert got_count == ref_count, f"n={n}: count {got_count} != {ref_count}"

    if got_count > 0:
        got_ids = np.sort(d_sel.cpu().numpy()[:got_count])
        assert np.array_equal(got_ids, np.sort(ref_ids)), f"n={n}: ids mismatch"

    print(f"n={n}: PASS (count={got_count})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)


if __name__ == "__main__":
    main()
