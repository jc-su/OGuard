"""Test and compile hash_probe's numba.py to PTX.

Builds a hash table with a CPU reference implementation of hash_build
(this task's own test data, not a cross-directory dependency), then
runs the hash_probe kernel on the GPU for m in {0, 1, 1000, 1000003}
and checks the result against a CPU reference probe. Also writes
numba.ptx.
"""

import importlib.util
import os

import numpy as np
from numba import cuda

HERE = os.path.dirname(os.path.abspath(__file__))

MASK64 = (1 << 64) - 1


def load_task_module():
    spec = importlib.util.spec_from_file_location(
        "hash_probe_numba", os.path.join(HERE, "numba.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_hash64(key):
    h = key & MASK64
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & MASK64
    h = h ^ (h >> 31)
    return h & MASK64


def next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def cpu_build(keys, capacity):
    table_keys = np.full(capacity, -1, dtype=np.int64)
    table_rows = np.zeros(capacity, dtype=np.int32)
    mask = capacity - 1
    for i, k in enumerate(keys):
        slot = cpu_hash64(int(k)) & mask
        while table_keys[slot] != -1:
            slot = (slot + 1) & mask
        table_keys[slot] = k
        table_rows[slot] = i
    return table_keys, table_rows


def cpu_probe(probe_keys, table_keys, table_rows, capacity):
    mask = capacity - 1
    build_ids, probe_ids = [], []
    for j, k in enumerate(probe_keys):
        slot = cpu_hash64(int(k)) & mask
        while table_keys[slot] != -1:
            if table_keys[slot] == k:
                build_ids.append(int(table_rows[slot]))
                probe_ids.append(j)
                break
            slot = (slot + 1) & mask
    return np.array(build_ids, dtype=np.int32), np.array(probe_ids, dtype=np.int32)


def check(task, m, rng, build_keys, table_keys, table_rows, capacity):
    build_n = build_keys.shape[0]
    # Mix probe keys: half drawn from the build side (guaranteed
    # matches), half random values likely absent from it.
    if m > 0 and build_n > 0:
        from_build = rng.choice(build_keys, size=m // 2, replace=True)
        random_vals = rng.choice(
            np.arange(-(4 * build_n) - 4, -4, dtype=np.int64),
            size=m - m // 2,
            replace=True,
        )
        probe_keys = np.concatenate([from_build, random_vals]).astype(np.int64)
        rng.shuffle(probe_keys)
    else:
        probe_keys = np.zeros(max(m, 0), dtype=np.int64)

    ref_build, ref_probe = cpu_probe(probe_keys, table_keys, table_rows, capacity)
    out_cap = max(m, 1)

    d_probe_keys = cuda.to_device(
        probe_keys if m > 0 else np.zeros(1, dtype=np.int64)
    )
    d_table_keys = cuda.to_device(table_keys)
    d_table_rows = cuda.to_device(table_rows)
    d_out_build = cuda.device_array(out_cap, dtype=np.int32)
    d_out_probe = cuda.device_array(out_cap, dtype=np.int32)
    d_out_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    task.run(
        d_probe_keys,
        d_table_keys,
        d_table_rows,
        d_out_build,
        d_out_probe,
        d_out_count,
        capacity,
        m,
        out_cap,
    )
    cuda.synchronize()

    got_count = d_out_count.copy_to_host()[0]
    assert got_count == ref_build.shape[0], (
        f"m={m}: count {got_count} != {ref_build.shape[0]}"
    )

    if got_count > 0:
        got_build = d_out_build.copy_to_host()[:got_count]
        got_probe = d_out_probe.copy_to_host()[:got_count]
        got_pairs = sorted(zip(got_probe.tolist(), got_build.tolist()))
        ref_pairs = sorted(zip(ref_probe.tolist(), ref_build.tolist()))
        assert got_pairs == ref_pairs, f"m={m}: pairs mismatch"

    print(f"m={m}: PASS (matches={got_count})")


def main():
    task = load_task_module()
    rng = np.random.default_rng(0)

    build_n = 5000
    capacity = max(2, next_pow2(2 * build_n))
    build_keys = rng.choice(
        np.arange(1, 4 * build_n + 4, dtype=np.int64), size=build_n, replace=False
    )
    table_keys, table_rows = cpu_build(build_keys, capacity)

    for m in (0, 1, 1000, 1000003):
        check(task, m, rng, build_keys, table_keys, table_rows, capacity)

    ptx, resty = cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    with open(os.path.join(HERE, "numba.ptx"), "w") as f:
        f.write(ptx)
    print("wrote numba.ptx")


if __name__ == "__main__":
    main()
