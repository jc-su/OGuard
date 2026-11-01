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


def make(n, seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 24, size=n)
    if n > 100:
        lengths[rng.integers(0, n, size=5)] = 3000
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    return offsets, chars


def reference(offsets, chars, n):
    lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)
    h = np.full(n, 2166136261, dtype=np.uint32)
    start = offsets[:-1].astype(np.int64)
    k = 0
    active = np.nonzero(lengths > k)[0]
    while len(active):
        c = chars[start[active] + k].astype(np.uint32)
        h[active] = (h[active] ^ c) * np.uint32(16777619)
        k += 1
        active = active[lengths[active] > k]
    return h


def check(n, seed):
    offsets, chars = make(n, seed)
    # uint32 output viewed as int32 for PyTorch; compared as uint32.
    d_out = torch.full((n,), 7, dtype=torch.int32, device="cuda")
    mod.run(torch.from_numpy(offsets).cuda(), torch.from_numpy(chars).cuda(), d_out, n)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy().view(np.uint32)
    exp = reference(offsets, chars, n)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("string_hash triton: ok")
