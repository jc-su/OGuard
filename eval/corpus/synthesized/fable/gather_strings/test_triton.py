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
        lengths[rng.integers(0, n, size=5)] = 5000
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    chars = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    m = n if n < 10 else n + 7
    idx = rng.integers(0, n, size=m, dtype=np.int32) if n > 0 else np.zeros(0, dtype=np.int32)
    out_lengths = offsets[idx + 1] - offsets[idx]
    out_offsets = np.concatenate([[0], np.cumsum(out_lengths)]).astype(np.int32)
    return offsets, chars, idx, out_offsets, m


def reference(offsets, chars, idx, out_offsets, m):
    lengths = (out_offsets[1:] - out_offsets[:-1]).astype(np.int64)
    row = np.repeat(np.arange(m, dtype=np.int64), lengths)
    p = np.arange(int(out_offsets[m]), dtype=np.int64)
    src = offsets[idx[row]].astype(np.int64) + (p - out_offsets[row].astype(np.int64))
    return chars[src]


def check(n, seed):
    offsets, chars, idx, out_offsets, m = make(n, seed)
    total = int(out_offsets[m])
    d_out = torch.full((total,), 0xA5, dtype=torch.uint8, device="cuda")
    mod.run(torch.from_numpy(offsets).cuda(), torch.from_numpy(chars).cuda(), torch.from_numpy(idx).cuda(),
            torch.from_numpy(out_offsets).cuda(), d_out, n, m)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy()
    exp = reference(offsets, chars, idx, out_offsets, m)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("gather_strings triton: ok")
