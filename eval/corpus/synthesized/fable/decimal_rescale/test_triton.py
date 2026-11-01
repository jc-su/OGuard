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

INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1


def reference(values, shift):
    vo = values.astype(object)
    if shift >= 0:
        prod = vo * (10 ** shift)
        ovf = (prod > INT64_MAX) | (prod < INT64_MIN)
        out = np.where(ovf, 0, prod)
    else:
        q = 10 ** (-shift)
        mag = abs(vo)
        qt = mag // q + ((2 * (mag % q)) >= q)
        out = np.where(vo < 0, -qt, qt)
        ovf = np.zeros(len(values), dtype=bool)
    return out.astype(np.int64), ovf.astype(np.int8)


def make(n, rng):
    values = rng.integers(INT64_MIN, INT64_MAX, size=n, dtype=np.int64)
    if n > 0:
        mix = rng.integers(0, 4, size=n)
        values[mix == 0] = rng.integers(-1000000, 1000000, size=int((mix == 0).sum()))
        values[mix == 1] = rng.integers(-(1 << 40), 1 << 40, size=int((mix == 1).sum()))
        values[0] = INT64_MIN
        if n > 1:
            values[1] = INT64_MAX
        if n > 2:
            values[2] = -922337203685477580
    return values


def check(n, shift, seed):
    rng = np.random.default_rng(seed)
    values = make(n, rng)
    d_out = torch.full((n,), 7, dtype=torch.int64, device="cuda")
    d_ovf = torch.full((n,), 7, dtype=torch.int8, device="cuda")
    mod.run(torch.from_numpy(values).cuda(), d_out, d_ovf, n, shift)
    torch.cuda.synchronize()
    out, ovf = reference(values, shift)
    got_out, got_ovf = d_out.cpu().numpy(), d_ovf.cpu().numpy()
    bad = np.nonzero((got_out != out) | (got_ovf != ovf))[0]
    assert len(bad) == 0, f"n={n} shift={shift}: mismatch at rows {bad[:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for shift in (-18, -5, -1, 0, 1, 3, 18):
        check(n, shift, n + shift + 100)
print("decimal_rescale triton: ok")
