import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(0)


def d(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


for n in (0, 1, 1000, 1000003):
    values = rng.standard_normal(n)
    starts = np.zeros(n, np.int32)
    if n:
        brk = rng.random(n) < 0.01
        brk[0] = True
        starts = np.maximum.accumulate(np.where(brk, np.arange(n), 0)).astype(np.int32)
    for k in (1, 2, 7, 1024):
        i = np.arange(n)
        ok = i - k >= starts
        ref = np.where(ok, values[np.maximum(i - k, 0)] if n else 0.0, np.nan)
        out = torch.zeros(n, dtype=torch.float64, device="cuda")
        mod.run(d(values), d(starts), out, n, k)
        torch.cuda.synchronize()
        assert np.array_equal(out.cpu().numpy(), ref, equal_nan=True), (n, k)
    print("lag n=%d ok" % n)
