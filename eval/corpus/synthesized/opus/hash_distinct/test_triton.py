"""Test the Triton kernel of hash_distinct against a NumPy reference."""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py and triton.py in this directory would shadow the packages.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]

SIZES = (0, 1, 1000, 1000003)
rng = np.random.default_rng(20261003)


def load(name):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{os.path.basename(HERE)}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(got, want):
    for name, expected in want.items():
        actual = got[name]
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            bad = np.flatnonzero(actual != expected) if actual.shape == expected.shape else []
            where = f" first mismatch at {bad[0]}: got {actual[bad[0]]} want {expected[bad[0]]}" if len(bad) else ""
            return f"{name} differs{where}"
    return None

ARGS = ("keys", "table_keys", "table_rows", "n", "capacity")
EMPTY_ROW = 2**31 - 1


def make_keys(n, mode):
    if mode == "random":
        keys = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
    elif mode == "duplicates":
        keys = rng.integers(0, n // 10 + 1, n).astype(np.int64) - 5
    elif mode == "one key":
        keys = np.full(n, -2, dtype=np.int64)
    else:  # multiples of 2^32 share low bits and crowd the start slots
        keys = rng.integers(0, n // 2 + 1, n).astype(np.int64) << 32
    keys[keys == -1] = 0
    return keys


def make_cases(n):
    for mode in ("random", "duplicates", "one key", "shared starts"):
        for spread in (1, 4):
            capacity = 1
            while capacity < 2 * n * spread:
                capacity *= 2
            case = dict(
                keys=make_keys(n, mode),
                table_keys=np.full(capacity, -1, dtype=np.int64),
                table_rows=np.full(capacity, EMPTY_ROW, dtype=np.int32),
                n=n,
                capacity=capacity,
            )
            yield f"{mode} capacity={capacity}", case


def start_slots(keys, capacity):
    h = (keys.view(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
    return (h & np.uint64(capacity - 1)).astype(np.int64)


def check(case, got):
    keys, capacity = case["keys"], case["capacity"]
    tk, tr = got["table_keys"], got["table_rows"]
    used = tk != -1
    if np.any(tr[~used] != EMPTY_ROW):
        return "an empty slot has a row"
    distinct, first = np.unique(keys, return_index=True)
    stored = tk[used]
    if len(stored) != len(distinct) or not np.array_equal(np.sort(stored), distinct):
        return "stored keys differ from the distinct keys"
    if not np.array_equal(tr[used], first[np.searchsorted(distinct, stored)].astype(np.int32)):
        return "a slot has the wrong row"
    # Linear probing: the slots from the start slot up to the key's slot are all used.
    # run[s] = number of consecutive used slots ending at s, cyclically.
    slots = np.flatnonzero(used)
    if len(slots):
        empty = int(np.flatnonzero(~used)[0])
        order = (np.arange(capacity) + empty + 1) % capacity  # starts after an empty slot
        u = used[order].astype(np.int64)
        breaks = np.where(u == 0, np.arange(capacity), -1)
        np.maximum.accumulate(breaks, out=breaks)
        run = np.empty(capacity, dtype=np.int64)
        run[order] = np.arange(capacity) - breaks
        distance = (slots - start_slots(tk[slots], capacity)) % capacity
        if np.any(distance >= run[slots]):
            return "a key is not reachable from its start slot"
    return None


def execute(case):
    import torch

    module = load("triton")
    dev = {k: torch.from_numpy(v.copy()).to("cuda") if isinstance(v, np.ndarray) else v for k, v in case.items()}
    module.run(*[dev[name] for name in ARGS])
    torch.cuda.synchronize()
    return {k: dev[k].cpu().numpy() for k, v in case.items() if isinstance(v, np.ndarray)}


def main():
    failures = 0
    for n in SIZES:
        for label, case in make_cases(n):
            error = check(case, execute(case))
            if error:
                failures += 1
                print(f"FAIL n={n} {label}: {error}")
        print(f"n={n} done")
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
