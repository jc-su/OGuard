import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The task directory holds triton.py, which would shadow the triton package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

I64 = np.iinfo(np.int64)
ROW_EMPTY = 2**31 - 1


def check(keys, table_keys, table_rows, capacity):
    """Returns an error text, or None when the table meets the contract."""
    mask = capacity - 1
    occupied = table_keys != -1
    if not np.array_equal(table_rows[~occupied], np.full(int((~occupied).sum()), ROW_EMPTY, np.int32)):
        return "an empty slot has a row"
    uniq, first = np.unique(keys, return_index=True)
    slots = np.flatnonzero(occupied)
    stored = table_keys[slots]
    if len(slots) != len(uniq) or not np.array_equal(np.sort(stored), uniq):
        return "slots do not hold each distinct key once"
    # Probing from the start slot reaches the key without passing an empty slot.
    start = ((stored.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)).astype(np.int64) & mask
    dist = (slots - start) & mask
    empties = np.concatenate([~occupied, ~occupied]).astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(empties)])
    if np.any(prefix[start + dist + 1] - prefix[start] != 0):
        return "a key is not reachable by probing"
    expect_rows = first[np.searchsorted(uniq, stored)]
    if not np.array_equal(table_rows[slots], expect_rows.astype(np.int32)):
        return "a slot holds a row that is not the first of its key"
    return None


def capacity_for(n, factor):
    c = 1
    while c < 2 * n * factor:
        c *= 2
    return c


def cases(n, rng):
    keys = rng.integers(I64.min, I64.max, size=n, dtype=np.int64, endpoint=True)
    keys[keys == -1] = 0
    yield "random", keys
    yield "few_distinct", rng.integers(0, 10, size=n, dtype=np.int64)
    yield "negative", -2 - rng.integers(0, 1000, size=n, dtype=np.int64)
    yield "equal", np.full(n, I64.min, dtype=np.int64)
    yield "high_words", rng.integers(0, 1 << 30, size=n, dtype=np.int64) << 32
    yield "sequential", np.arange(n, dtype=np.int64)
    yield "distinct_half", rng.integers(0, max(1, n // 2), size=n, dtype=np.int64)


def main():
    rng = np.random.default_rng(1)
    failed = False
    for n in (0, 1, 1000, 1000003):
        for name, keys in cases(n, rng):
            for factor in (1, 4):
                capacity = capacity_for(n, factor)
                t_keys = torch.from_numpy(keys).cuda()
                t_tk = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
                t_tr = torch.full((capacity,), ROW_EMPTY, dtype=torch.int32, device="cuda")
                task.run(t_keys, t_tk, t_tr, n, capacity)
                torch.cuda.synchronize()
                err = check(keys, t_tk.cpu().numpy(), t_tr.cpu().numpy(), capacity)
                print(f"hash_distinct triton n={n} capacity={capacity} {name}: {'PASS' if err is None else 'FAIL ' + err}")
                failed |= err is not None
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
