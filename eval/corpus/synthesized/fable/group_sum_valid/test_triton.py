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


def reference(group, value, valid, n, g):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    counts = np.zeros(g, dtype=np.int64)
    sums = np.zeros(g, dtype=np.int64)
    np.add.at(counts, group[bits], 1)
    with np.errstate(over="ignore"):
        np.add.at(sums, group[bits], value[bits])
    return counts, sums


def check(n, g, seed):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, g, size=n, dtype=np.int32)
    value = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    valid = rng.integers(0, 256, size=(n + 7) // 8, dtype=np.uint8)
    d_counts = torch.zeros(g, dtype=torch.int64, device="cuda")
    d_sums = torch.zeros(g, dtype=torch.int64, device="cuda")
    mod.run(torch.from_numpy(group).cuda(), torch.from_numpy(value).cuda(), torch.from_numpy(valid).cuda(),
            d_counts, d_sums, n, g)
    torch.cuda.synchronize()
    counts, sums = reference(group, value, valid, n, g)
    assert np.array_equal(d_counts.cpu().numpy(), counts), f"n={n} g={g}: counts differ"
    assert np.array_equal(d_sums.cpu().numpy(), sums), f"n={n} g={g}: sums differ"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for g in (1, 7, 1024):
        check(n, g, n + g)
print("group_sum_valid triton: ok")
