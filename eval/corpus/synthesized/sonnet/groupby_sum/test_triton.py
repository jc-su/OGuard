"""Test groupby_sum's triton.py.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the resulting table against a NumPy reference (sum of value per
group).
"""

import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "groupby_sum_triton", os.path.join(HERE, "triton.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def check(task, n, rng):
    num_groups = max(1, min(n, 20000)) if n > 0 else 1
    capacity = max(2, next_pow2(2 * num_groups))

    group = (
        rng.integers(0, num_groups, size=n).astype(np.int32)
        if n > 0
        else np.zeros(0, dtype=np.int32)
    )
    value = (
        rng.uniform(-10.0, 10.0, size=n).astype(np.float64)
        if n > 0
        else np.zeros(0, dtype=np.float64)
    )

    ref_sums = np.zeros(num_groups, dtype=np.float64)
    if n > 0:
        np.add.at(ref_sums, group, value)

    d_group = torch.from_numpy(group if n > 0 else np.zeros(1, dtype=np.int32)).cuda()
    d_value = torch.from_numpy(
        value if n > 0 else np.zeros(1, dtype=np.float64)
    ).cuda()
    d_table_keys = torch.from_numpy(np.full(capacity, -1, dtype=np.int32)).cuda()
    d_table_sums = torch.zeros(capacity, dtype=torch.float64, device="cuda")

    task.run(d_group, d_value, d_table_keys, d_table_sums, capacity, n)
    torch.cuda.synchronize()

    table_keys = d_table_keys.cpu().numpy()
    table_sums = d_table_sums.cpu().numpy()

    occupied = np.nonzero(table_keys != -1)[0]
    seen_groups = np.unique(group) if n > 0 else np.zeros(0, dtype=np.int32)

    assert occupied.shape[0] == seen_groups.shape[0], (
        f"n={n}: {occupied.shape[0]} occupied slots, "
        f"expected {seen_groups.shape[0]} distinct groups"
    )
    assert np.array_equal(np.sort(table_keys[occupied]), np.sort(seen_groups)), (
        f"n={n}: table keys don't match the distinct groups"
    )

    for slot in occupied:
        g = table_keys[slot]
        got = table_sums[slot]
        ref = ref_sums[g]
        assert abs(got - ref) <= 1e-6 * max(1.0, abs(ref)), (
            f"n={n}: group {g} sum {got} != {ref}"
        )

    print(f"n={n}: PASS (groups={occupied.shape[0]}, capacity={capacity})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)


if __name__ == "__main__":
    main()
