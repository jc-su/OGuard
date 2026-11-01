import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
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
    d_keys_t = torch.full((capacity,), -1, dtype=torch.int64, device="cuda")
    d_rows_t = torch.full((capacity,), EMPTY_ROW, dtype=torch.int32, device="cuda")
    mod.run(torch.from_numpy(keys).cuda(), d_keys_t, d_rows_t, n, capacity)
    torch.cuda.synchronize()
    verify(keys, d_keys_t.cpu().numpy(), d_rows_t.cpu().numpy(), n, capacity)


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for variant in range(3):
        check(n, variant, n + variant)
print("hash_distinct triton: ok")
