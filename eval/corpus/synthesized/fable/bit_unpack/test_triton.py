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


def reference(packed, n, bits):
    stream = np.unpackbits(packed.view(np.uint8), bitorder="little")[: n * bits].reshape(n, bits)
    weights = (np.uint64(1) << np.arange(bits, dtype=np.uint64))
    return (stream.astype(np.uint64) * weights).sum(axis=1).astype(np.uint32)


def check(n, bits, seed):
    rng = np.random.default_rng(seed)
    w = (n * bits + 31) // 32
    packed = rng.integers(0, 1 << 32, size=w, dtype=np.uint64).astype(np.uint32)
    # uint32 arrays travel as int32 tensors (same bytes) and are compared as uint32.
    d_packed = torch.from_numpy(packed.view(np.int32)).cuda()
    d_out = torch.full((n,), 7, dtype=torch.int32, device="cuda")
    mod.run(d_packed, d_out, n, w, bits)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy().view(np.uint32)
    exp = reference(packed, n, bits)
    assert np.array_equal(got, exp), f"n={n} bits={bits}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    for bits in (1, 3, 7, 8, 13, 31, 32):
        check(n, bits, n + bits)
print("bit_unpack triton: ok")
