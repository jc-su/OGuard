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


def make(n, g, variant, rng):
    group = rng.integers(0, g, size=n, dtype=np.int32)
    if variant == 0:
        value = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    elif variant == 1:  # few distinct values: many ties
        value = rng.integers(-2, 2, size=n, dtype=np.int64)
    else:  # increasing values: every row improves on the published one
        value = np.arange(n, dtype=np.int64)
    return group, value


def reference(group, value, n, g):
    best = np.full(g, -1, dtype=np.int32)
    if n:
        row = np.arange(n, dtype=np.int64)
        order = np.lexsort((-row, value, group))
        sg = group[order]
        last = np.concatenate([np.nonzero(sg[1:] != sg[:-1])[0], [n - 1]])
        best[sg[last]] = order[last]
    return best


def check(n, g, variant, seed):
    rng = np.random.default_rng(seed)
    group, value = make(n, g, variant, rng)
    d_best = torch.full((g,), -1, dtype=torch.int32, device="cuda")
    mod.run(torch.from_numpy(group).cuda(), torch.from_numpy(value).cuda(), d_best, n, g)
    torch.cuda.synchronize()
    got = d_best.cpu().numpy()
    exp = reference(group, value, n, g)
    assert np.array_equal(got, exp), f"n={n} g={g} variant={variant}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for g in (1, 7, 1000, 1 << 20):
        for variant in range(3):
            check(n, g, variant, n + g + variant)
print("group_argmax triton: ok")
