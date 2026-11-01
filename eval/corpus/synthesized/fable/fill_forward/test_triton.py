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
    values = rng.integers(-(1 << 63), (1 << 63) - 1, size=n, dtype=np.int64)
    if variant == 0:
        bits = rng.random(n) < 0.5
    elif variant == 1:  # all null
        bits = np.zeros(n, dtype=bool)
    elif variant == 2:  # all valid
        bits = np.ones(n, dtype=bool)
    elif variant == 3:  # one valid row, then a null run as long as the column
        bits = np.zeros(n, dtype=bool)
        if n > 0:
            bits[0] = True
    else:  # sparse valid rows at tile-unaligned positions
        bits = (np.arange(n) % 50001) == 3
    valid = np.packbits(bits, bitorder="little")
    if n % 8 and len(valid):  # junk in the padding bits must be ignored
        valid[-1] |= np.uint8((0xFF << (n % 8)) & 0xFF)
    return values, valid


def reference(values, valid, n):
    bits = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
    idx = np.where(bits, np.arange(n, dtype=np.int64), -1)
    last = np.maximum.accumulate(idx) if n else idx
    return np.where(last >= 0, values[np.maximum(last, 0)], 0)


def check(n, variant, seed):
    rng = np.random.default_rng(seed)
    values, valid = make(n, variant, rng)
    d_out = torch.full((n,), 7, dtype=torch.int64, device="cuda")
    mod.run(torch.from_numpy(values).cuda(), torch.from_numpy(valid).cuda(), d_out, n)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy()
    exp = reference(values, valid, n)
    assert np.array_equal(got, exp), f"n={n} variant={variant}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for variant in range(5):
        check(n, variant, n + variant)
print("fill_forward triton: ok")
