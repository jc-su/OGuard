"""Test hash_build's triton.py.

Runs the kernel on the GPU for n in {0, 1, 1000, 1000003} and checks
the resulting table against the invariants of a correct build: every
input key is present exactly once, at a slot whose table_rows entry
points back to the row that holds it.
"""

import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "hash_build_triton", os.path.join(HERE, "triton.py")
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
    capacity = max(2, next_pow2(2 * n))

    # Distinct keys, never -1.
    keys = rng.choice(
        np.arange(1, 4 * n + 4, dtype=np.int64), size=n, replace=False
    )

    d_keys = torch.from_numpy(keys if n > 0 else np.zeros(1, dtype=np.int64)).cuda()
    d_table_keys = torch.from_numpy(np.full(capacity, -1, dtype=np.int64)).cuda()
    d_table_rows = torch.from_numpy(np.zeros(capacity, dtype=np.int32)).cuda()

    task.run(d_keys, d_table_keys, d_table_rows, capacity, n)
    torch.cuda.synchronize()

    table_keys = d_table_keys.cpu().numpy()
    table_rows = d_table_rows.cpu().numpy()

    occupied = np.nonzero(table_keys != -1)[0]
    assert occupied.shape[0] == n, (
        f"n={n}: {occupied.shape[0]} occupied slots, expected {n}"
    )

    rows = table_rows[occupied]
    assert np.array_equal(np.sort(rows), np.arange(n)), (
        f"n={n}: row ids in table are not exactly 0..n-1"
    )
    assert np.array_equal(table_keys[occupied], keys[rows]), (
        f"n={n}: table_keys[slot] != keys[table_rows[slot]] for some slot"
    )

    print(f"n={n}: PASS (capacity={capacity})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)
    for n in (0, 1, 1000, 1000003):
        check(task, n, rng)


if __name__ == "__main__":
    main()
