import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np
from numba import cuda

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make(b, p, seed):
    rng = np.random.default_rng(seed)
    span = max(1, max(b, p) // 4)
    build = np.sort(rng.integers(0, span, size=b, dtype=np.int64))
    probe = rng.integers(-2, span + 2, size=p, dtype=np.int64)
    if b > 0 and p > 0:
        probe[: p // 2] = build[rng.integers(0, b, size=p // 2)]
    first = np.searchsorted(build, probe, side="left")
    last = np.searchsorted(build, probe, side="right")
    offsets = np.concatenate([[0], np.cumsum(last - first)]).astype(np.int32)
    return build, probe, offsets, first


def reference(offsets, first, p):
    counts = (offsets[1:] - offsets[:-1]).astype(np.int64)
    i = np.repeat(np.arange(p, dtype=np.int64), counts)
    q = np.arange(int(offsets[p]), dtype=np.int64)
    out_build = first[i] + (q - offsets[i].astype(np.int64))
    return out_build.astype(np.int32), i.astype(np.int32)


def check(b, p, seed):
    build, probe, offsets, first = make(b, p, seed)
    total = int(offsets[p])
    d_ob = cuda.to_device(np.full(total, 7, dtype=np.int32))
    d_op = cuda.to_device(np.full(total, 7, dtype=np.int32))
    mod.run(cuda.to_device(build), cuda.to_device(probe), cuda.to_device(offsets), d_ob, d_op, b, p)
    cuda.synchronize()
    eb, ep = reference(offsets, first, p)
    gb, gp = d_ob.copy_to_host(), d_op.copy_to_host()
    bad = np.nonzero((gb != eb) | (gp != ep))[0]
    assert len(bad) == 0, f"b={b} p={p}: mismatch at {bad[:10]}"


ptx, _ = cuda.compile_ptx(mod.kernel, mod.TYPES, cc=(8, 6))
assert ".entry" in ptx
for n in (0, 1, 1000, 1000003):
    for b in (0, 1, 1000, 1000003):
        check(b, n, n + b)
print("join_materialize numba: ok")
