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

DAY_MIN = int(np.datetime64("0001-01-01", "D").astype(np.int64))
DAY_MAX = int(np.datetime64("9999-12-31", "D").astype(np.int64))


def reference(days, months):
    d = days.astype("datetime64[D]")
    mo = d.astype("datetime64[M]")
    dom = (d - mo).astype(np.int64)
    target = mo + months.astype("timedelta64[M]")
    t0 = target.astype("datetime64[D]")
    length = ((target + np.timedelta64(1, "M")).astype("datetime64[D]") - t0).astype(np.int64)
    return (t0 + np.minimum(dom, length - 1).astype("timedelta64[D]")).astype(np.int64).astype(np.int32)


def make(n, rng):
    days = rng.integers(DAY_MIN, DAY_MAX + 1, size=n, dtype=np.int64)
    if n > 0:
        k = n // 3
        days[:k] = rng.integers(np.datetime64("1996-01-01", "D").astype(np.int64),
                                np.datetime64("2032-12-31", "D").astype(np.int64), size=k)
        specials = [np.datetime64(s, "D").astype(np.int64)
                    for s in ("2024-02-29", "2023-01-31", "0001-01-01", "9999-12-31", "2000-02-29", "1900-03-31")]
        days[: min(n, len(specials))] = specials[:n]
    months = rng.integers(-120000, 120001, size=n, dtype=np.int64)
    mo = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    lo_m = np.datetime64("0001-01", "M").astype(np.int64)
    hi_m = np.datetime64("9999-12", "M").astype(np.int64)
    months = np.clip(mo + months, lo_m, hi_m) - mo
    return days.astype(np.int32), months.astype(np.int32)


def check(n, seed):
    rng = np.random.default_rng(seed)
    days, months = make(n, rng)
    d_out = torch.full((n,), 7, dtype=torch.int32, device="cuda")
    mod.run(torch.from_numpy(days).cuda(), torch.from_numpy(months).cuda(), d_out, n)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy()
    exp = reference(days, months)
    assert np.array_equal(got, exp), f"n={n}: mismatch at {np.nonzero(got != exp)[0][:10]}"


sig = dict(mod.TYPES)
sig.update({k: "constexpr" for k in mod.CONSTANTS})
src = ASTSource(fn=mod.kernel, signature=sig, constexprs=dict(mod.CONSTANTS))
triton.compile(src, target=GPUTarget("cuda", 86, 32), options={"num_warps": mod.NUM_WARPS})
for n in (0, 1, 1000, 1000003):
    check(n, n + 1)
print("add_months triton: ok")
