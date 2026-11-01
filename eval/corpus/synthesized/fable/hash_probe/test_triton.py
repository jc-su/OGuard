"""GPU test for hash_probe: Triton kernel against a NumPy reference.

The table is built on the host with the same hash and linear probing as
hash_build. Half of the probe keys hit, half miss. The expected output is the
set of (build row, probe row) pairs; the kernel may emit them in any order.
A second run with out_cap below the match count checks that out_count still
reports every match and that only valid pairs are written.
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
        "hash_probe_triton", os.path.join(HERE, "triton.py")
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


def build_table(keys, capacity):
    """Linear-probing insert of all keys, vectorized round by round."""
    n = keys.size
    mask = np.uint64(capacity - 1)
    table_keys = np.full(capacity, -1, dtype=np.int64)
    table_rows = np.full(capacity, -1, dtype=np.int32)
    pos = mix64(keys) & mask
    pending = np.arange(n)
    while pending.size:
        target = pos[pending]
        free = table_keys[target] == -1
        cand, cand_pos = pending[free], target[free]
        slots, first = np.unique(cand_pos, return_index=True)
        winners = cand[first]
        table_keys[slots] = keys[winners]
        table_rows[slots] = winners
        placed = np.zeros(n, dtype=bool)
        placed[winners] = True
        pending = pending[~placed[pending]]
        pos[pending] = (pos[pending] + np.uint64(1)) & mask
    return table_keys, table_rows


def sorted_pairs(build, probe):
    order = np.lexsort((build, probe))
    return np.stack([probe[order], build[order]], axis=1)


def check(task, n, rng):
    m = n
    capacity = capacity_for(n)
    keys = (rng.permutation(n).astype(np.int64) * 1000003 + 17).astype(np.int64)
    table_keys, table_rows = build_table(keys, capacity)

    hit = rng.integers(0, 2, m).astype(bool)
    probe_keys = -(np.arange(m, dtype=np.int64) + 2)  # misses: never in the table
    if n > 0:
        probe_keys[hit] = keys[rng.integers(0, n, int(hit.sum()))]
    order = np.argsort(keys)
    idx = np.searchsorted(keys[order], probe_keys[hit])
    exp_build = order[idx].astype(np.int32)
    exp_probe = np.nonzero(hit)[0].astype(np.int32)
    expected = sorted_pairs(exp_build, exp_probe)
    matches = exp_probe.size

    d_probe_keys = torch.from_numpy(probe_keys).to(DEV)
    d_table_keys = torch.from_numpy(table_keys).to(DEV)
    d_table_rows = torch.from_numpy(table_rows).to(DEV)
    d_out_count = torch.zeros(1, dtype=torch.int32, device=DEV)

    out_cap = max(matches, 1)
    d_out_build = torch.empty(out_cap, dtype=torch.int32, device=DEV)
    d_out_probe = torch.empty(out_cap, dtype=torch.int32, device=DEV)
    args = (
        d_probe_keys,
        d_table_keys,
        d_table_rows,
        capacity,
        d_out_build,
        d_out_probe,
        d_out_count,
        m,
        out_cap,
    )
    task.run(*args)
    torch.cuda.synchronize()
    count = int(d_out_count.item())
    assert count == matches, f"n={n}: out_count {count} != {matches}"
    got = sorted_pairs(d_out_build[:count].cpu().numpy(), d_out_probe[:count].cpu().numpy())
    assert np.array_equal(got, expected), f"n={n}: pairs mismatch"

    if matches >= 2:
        small_cap = matches // 2
        d_out_count.zero_()
        d_small_build = torch.full((small_cap,), -1, dtype=torch.int32, device=DEV)
        d_small_probe = torch.full((small_cap,), -1, dtype=torch.int32, device=DEV)
        task.run(
            d_probe_keys,
            d_table_keys,
            d_table_rows,
            capacity,
            d_small_build,
            d_small_probe,
            d_out_count,
            m,
            small_cap,
        )
        torch.cuda.synchronize()
        count = int(d_out_count.item())
        assert count == matches, f"n={n}: out_count with small out_cap {count} != {matches}"
        b, p = d_small_build.cpu().numpy(), d_small_probe.cpu().numpy()
        assert np.all(b >= 0) and np.all(p >= 0), f"n={n}: unwritten slot below out_cap"
        assert np.array_equal(keys[b], probe_keys[p]), f"n={n}: invalid pair below out_cap"
        assert np.unique(p).size == small_cap, f"n={n}: duplicate probe row below out_cap"

    if n == max(SIZES):
        us = bench(lambda: task.run(*args), lambda: d_out_count.zero_())
        print(f"  n={n}: {us:.1f} us per launch")


def main():
    task = load_task()
    rng = np.random.default_rng(1)
    for n in SIZES:
        check(task, n, rng)
        print(f"  n={n}: ok")
    print("PASS hash_probe")


if __name__ == "__main__":
    main()
