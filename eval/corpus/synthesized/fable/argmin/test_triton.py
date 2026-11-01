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
    if variant == 0:
        return rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    if variant == 1:  # many ties on the minimum
        return rng.integers(-3, 3, size=n, dtype=np.int64)
    if variant == 2:  # minimum at the end, decreasing values
        return (np.arange(n, dtype=np.int64)[::-1] - n).copy()
    return np.full(n, -(1 << 63), dtype=np.int64)  # all equal to the smallest int64


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values = make(n, variant, rng)
    d_row = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    mod.run(torch.from_numpy(values).cuda(), d_row, n)
    torch.cuda.synchronize()
    got = int(d_row.cpu()[0])
    exp = int(np.argmin(values)) if n > 0 else -1
    assert got == exp, f"n={n} variant={variant}: {got} != {exp}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for variant in range(4):
        check(n, variant, n + variant)
print("argmin triton: ok")
