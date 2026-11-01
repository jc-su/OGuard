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


def make(n, variant, rng):
    if variant == 0:  # runs of geometric length
        starts = rng.random(n) < 0.3
        starts[:1] = True
        return np.cumsum(starts).astype(np.int64) * 7 - 3
    if variant == 1:  # one run for the whole column
        return np.full(n, -5, dtype=np.int64)
    if variant == 2:  # every row its own run
        return np.arange(n, dtype=np.int64)
    # long runs that cross tile boundaries
    return (np.arange(n, dtype=np.int64) // 50001) % 3


def reference(values, n):
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    ends = np.concatenate([np.nonzero(values[1:] != values[:-1])[0] + 1, [n]]).astype(np.int32)
    return ends, values[ends.astype(np.int64) - 1]


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values = make(n, variant, rng)
    ends0 = np.full(n, -7, dtype=np.int32)
    vals0 = np.full(n, -9, dtype=np.int64)
    d_ends = torch.from_numpy(ends0).cuda()
    d_vals = torch.from_numpy(vals0).cuda()
    d_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    mod.run(torch.from_numpy(values).cuda(), d_ends, d_vals, d_count, n)
    torch.cuda.synchronize()
    ends, vals = reference(values, n)
    r = len(ends)
    got_count = int(d_count.cpu()[0])
    got_ends, got_vals = d_ends.cpu().numpy(), d_vals.cpu().numpy()
    assert got_count == r, f"n={n} variant={variant}: count {got_count} != {r}"
    assert np.array_equal(got_ends[:r], ends), f"n={n} variant={variant}: run_ends differ"
    assert np.array_equal(got_vals[:r], vals), f"n={n} variant={variant}: run_values differ"
    assert np.array_equal(got_ends[r:], ends0[r:]) and np.array_equal(got_vals[r:], vals0[r:]), \
        f"n={n} variant={variant}: tail changed"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for variant in range(4):
        check(n, variant, n + variant)
print("run_end_encode triton: ok")
