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
    for m in (0, 1, 1000, 1000003):
        col = rng.standard_normal(n)
        if n:
            idx = rng.integers(-1, n, m).astype(np.int32)
        else:
            idx = np.full(m, -1, np.int32)
        ref = np.where(idx >= 0, col[np.maximum(idx, 0)] if n else np.nan, np.nan)
        out = torch.zeros(m, dtype=torch.float64, device="cuda")
        mod.run(d(col), d(idx), out, m, n)
        torch.cuda.synchronize()
        got = out.cpu().numpy()
        assert np.array_equal(got, ref, equal_nan=True), (n, m)
    print("gather_nullable n=%d ok" % n)
