"""Test and compile groupby_sum's numba.py to PTX.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the resulting table against a NumPy reference (sum of value per
group). Also writes numba.ptx.
"""

import importlib.util
import os

import numpy as np
from numba import cuda

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "groupby_sum_numba", os.path.join(HERE, "numba.py")
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

    d_group = cuda.to_device(group if n > 0 else np.zeros(1, dtype=np.int32))
    d_value = cuda.to_device(value if n > 0 else np.zeros(1, dtype=np.float64))
    d_table_keys = cuda.to_device(np.full(capacity, -1, dtype=np.int32))
    d_table_sums = cuda.to_device(np.zeros(capacity, dtype=np.float64))

    task.run(d_group, d_value, d_table_keys, d_table_sums, capacity, n)
    cuda.synchronize()

    table_keys = d_table_keys.copy_to_host()
    table_sums = d_table_sums.copy_to_host()

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

    ptx, resty = cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    with open(os.path.join(HERE, "numba.ptx"), "w") as f:
        f.write(ptx)
    print("wrote numba.ptx")


if __name__ == "__main__":
    main()
