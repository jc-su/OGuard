"""GPU test for groupby_sum: Numba kernel against a NumPy reference, plus PTX export.

Two group cardinalities per size exercise both kernel paths: 1000 groups
(capacity 2048, block-private shared-memory tables) and 100000 groups
(capacity 262144, global table only). The check compares the set of keys and
the per-key sums, and walks every key's probe chain.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory shadows the numba package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
CARDINALITIES = (1000, 100000)


def load_task():
    spec = importlib.util.spec_from_file_location("groupby_sum_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(launch, reset, reps=20):
    """Average kernel time in microseconds; reset runs before every timed launch."""
    start, end = cuda.event(), cuda.event()
    total = 0.0
    for _ in range(reps):
        reset()
        start.record()
        launch()
        end.record()
        end.synchronize()
        total += cuda.event_elapsed_time(start, end)
    return total / reps * 1e3


def home_slot(groups, capacity):
    h = groups.astype(np.int64).astype(np.uint64)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return h & np.uint64(capacity - 1)


def capacity_for(groups):
    c = 2
    while c < 2 * groups:
        c *= 2
    return c


def validate(group, value, table_keys, table_sums, capacity, cardinality):
    exp_keys = np.unique(group)
    exp_sums = np.bincount(group, weights=value, minlength=cardinality)[exp_keys]
    filled = table_keys != -1
    got_keys = table_keys[filled]
    order = np.argsort(got_keys)
    assert np.array_equal(got_keys[order], exp_keys), "key set"
    assert np.allclose(table_sums[filled][order], exp_sums, rtol=1e-9, atol=1e-9), "sums"
    assert np.all(table_sums[~filled] == 0.0), "sums of empty slots"
    # Probe chain integrity, vectorized: walk every key from its home slot.
    mask = np.uint64(capacity - 1)
    pos = home_slot(exp_keys, capacity)
    pending = np.arange(exp_keys.size)
    steps = 0
    while pending.size:
        at = table_keys[pos[pending]]
        assert not np.any(at == -1), "empty slot inside a probe chain"
        pending = pending[at != exp_keys[pending]]
        pos[pending] = (pos[pending] + np.uint64(1)) & mask
        steps += 1
        assert steps <= capacity, "probe chain longer than the table"


def check(task, n, cardinality, rng):
    capacity = capacity_for(cardinality)
    group = rng.integers(0, cardinality, n).astype(np.int32)
    value = rng.uniform(0.0, 1.0, n).astype(np.float64)

    d_group = cuda.to_device(group)
    d_value = cuda.to_device(value)
    d_table_keys = cuda.to_device(np.full(capacity, -1, dtype=np.int32))
    d_table_sums = cuda.to_device(np.zeros(capacity, dtype=np.float64))
    task.run(d_group, d_value, d_table_keys, d_table_sums, capacity, n)
    cuda.synchronize()

    validate(group, value, d_table_keys.copy_to_host(), d_table_sums.copy_to_host(), capacity, cardinality)

    if n == max(SIZES):

        def reset():
            d_table_keys.copy_to_device(np.full(capacity, -1, dtype=np.int32))
            d_table_sums.copy_to_device(np.zeros(capacity, dtype=np.float64))

        us = bench(lambda: task.run(d_group, d_value, d_table_keys, d_table_sums, capacity, n), reset)
        print(f"  n={n} groups={cardinality}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        for cardinality in CARDINALITIES:
            check(task, n, cardinality, rng)
        print(f"  n={n}: ok")
    ptx, _ = cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    with open(os.path.join(HERE, "numba.ptx"), "w") as f:
        f.write(ptx)
    print("PASS groupby_sum")


if __name__ == "__main__":
    main()
