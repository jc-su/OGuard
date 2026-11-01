import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

EMPTY_ROW = (1 << 31) - 1


def make(n, variant, rng):
    if variant == 0:
        keys = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    elif variant == 1:  # heavy duplication
        keys = rng.integers(0, max(1, n // 50), size=n, dtype=np.int64)
    else:  # one distinct key, plus keys that collide in the top hash bits
        keys = np.full(n, 12345, dtype=np.int64)
        keys[: n // 2] = np.arange(n // 2, dtype=np.int64) * 4096
    keys[keys == -1] = -2
    return keys


def verify(keys, table_keys, table_rows, n, capacity):
    distinct, first = np.unique(keys, return_index=True)
    occupied = table_keys != -1
    assert int(occupied.sum()) == len(distinct), "number of occupied slots"
    assert np.array_equal(np.sort(table_keys[occupied]), distinct), "set of keys"
    expect_row = dict(zip(distinct.tolist(), first.tolist()))
    slots = np.nonzero(occupied)[0]
    rows = np.array([expect_row[k] for k in table_keys[slots].tolist()], dtype=np.int64)
    assert np.array_equal(table_rows[slots].astype(np.int64), rows), "first rows"
    assert np.all(table_rows[~occupied] == EMPTY_ROW), "rows of empty slots"
    mask = capacity - 1
    h = ((table_keys[slots].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)).astype(np.int64)
    start = h & mask
    dist = (slots - start) & mask
    empty_cum = np.concatenate([[0], np.cumsum(np.concatenate([~occupied, ~occupied]).astype(np.int64))])
    passed_empty = empty_cum[start + dist] - empty_cum[start]
    assert np.all(passed_empty == 0), "probe sequence passes an empty slot"


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    keys = make(n, variant, rng)
    capacity = 1
    while capacity < 2 * n:
        capacity *= 2
    d_keys_t = cuda.to_device(np.full(capacity, -1, dtype=np.int64))
    d_rows_t = cuda.to_device(np.full(capacity, EMPTY_ROW, dtype=np.int32))
    mod.run(cuda.to_device(keys), d_keys_t, d_rows_t, n, capacity)
    cuda.synchronize()
    verify(keys, d_keys_t.copy_to_host(), d_rows_t.copy_to_host(), n, capacity)


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for variant in range(3):
        check(n, variant, n + variant)
print("hash_distinct numba: ok")
