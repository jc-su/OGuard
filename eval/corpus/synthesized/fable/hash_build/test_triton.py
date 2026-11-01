"""GPU test for hash_build: Triton kernel validated with NumPy.

The GPU places colliding keys in an order that depends on scheduling, so the
check validates the table's properties instead of comparing slot by slot:
every row appears exactly once, each slot is consistent, and a linear probe
from every key's home slot reaches the key without crossing an empty slot.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# triton.py in this directory shadows the triton package while this directory is on sys.path.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
DEV = "cuda"


def load_task():
    spec = importlib.util.spec_from_file_location(
        "hash_build_triton", os.path.join(HERE, "triton.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(launch, reset, reps=20):
    """Average kernel time in microseconds; reset runs before every timed launch."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(reps):
        reset()
        start.record()
        launch()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
    return total / reps * 1e3


def mix64(keys):
    h = keys.astype(np.uint64)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return h


def capacity_for(n):
    c = 2
    while c < 2 * n:
        c *= 2
    return c


def validate(keys, table_keys, table_rows, capacity):
    n = keys.size
    filled = table_keys != -1
    assert int(filled.sum()) == n, "filled slot count"
    rows = table_rows[filled]
    assert np.array_equal(np.sort(rows), np.arange(n, dtype=np.int32)), "each row once"
    assert np.array_equal(keys[rows], table_keys[filled]), "slot consistency"
    # Probe chain integrity, vectorized: walk every key from its home slot.
    mask = np.uint64(capacity - 1)
    pos = mix64(keys) & mask
    pending = np.arange(n)
    steps = 0
    while pending.size:
        at = table_keys[pos[pending]]
        assert not np.any(at == -1), "empty slot inside a probe chain"
        pending = pending[at != keys[pending]]
        pos[pending] = (pos[pending] + np.uint64(1)) & mask
        steps += 1
        assert steps <= capacity, "probe chain longer than the table"


def check(task, n, rng):
    capacity = capacity_for(n)
    keys = (rng.permutation(n).astype(np.int64) * 1000003 + 17).astype(np.int64)

    d_keys = torch.from_numpy(keys).to(DEV)
    d_table_keys = torch.full((capacity,), -1, dtype=torch.int64, device=DEV)
    d_table_rows = torch.empty(capacity, dtype=torch.int32, device=DEV)
    task.run(d_keys, d_table_keys, d_table_rows, capacity, n)
    torch.cuda.synchronize()

    validate(keys, d_table_keys.cpu().numpy(), d_table_rows.cpu().numpy(), capacity)

    if n == max(SIZES):
        us = bench(
            lambda: task.run(d_keys, d_table_keys, d_table_rows, capacity, n),
            lambda: d_table_keys.fill_(-1),
        )
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    print("PASS hash_build")


if __name__ == "__main__":
    main()
