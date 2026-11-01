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
    for b in (0, 1, 5, 1000, 100003):
        bounds = np.sort(np.round(rng.standard_normal(b), 2))
        values = np.round(rng.standard_normal(n) * 1.2, 2)
        if b and n:
            values[: min(n, b)] = bounds[: min(n, b)]
        ref = np.searchsorted(bounds, values, side="right").astype(np.int32)
        out = torch.full((n,), -1, dtype=torch.int32, device="cuda")
        mod.run(d(values), d(bounds), out, n, b)
        torch.cuda.synchronize()
        assert out.cpu().numpy().tobytes() == ref.tobytes(), (n, b)
    print("bucketize n=%d ok" % n)
