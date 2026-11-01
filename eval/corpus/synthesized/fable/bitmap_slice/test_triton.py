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


def reference(source, n, offset):
    bits = np.unpackbits(source, bitorder="little")[offset:offset + n]
    return np.packbits(bits, bitorder="little")


def check(n, offset, seed):
    rng = np.random.default_rng(seed)
    src_bytes = (offset + n + 7) // 8
    source = rng.integers(0, 256, size=src_bytes, dtype=np.uint8)
    d_out = torch.full(((n + 7) // 8,), 0xA5, dtype=torch.uint8, device="cuda")
    mod.run(torch.from_numpy(source).cuda(), d_out, n, offset)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy()
    exp = reference(source, n, offset)
    assert np.array_equal(got, exp), f"n={n} offset={offset}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for offset in (0, 3, 8, 13, 1000001):
        check(n, offset, n + offset)
print("bitmap_slice triton: ok")
