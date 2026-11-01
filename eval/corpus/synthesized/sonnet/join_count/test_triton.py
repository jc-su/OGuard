import importlib.util
import os
import sys

sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(os.path.dirname(os.path.abspath(__file__)), "triton.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def check(n):
    g = torch.Generator().manual_seed(n)
    for b, p in ((n, n), (0, n), (n, 0), (max(n // 3, 1), n)):
        build = torch.randint(0, max(b // 2, 1), (b,), generator=g, dtype=torch.int64).sort().values
        probe = torch.randint(-3, max(b // 2, 1) + 3, (p,), generator=g, dtype=torch.int64)
        counts = torch.full((p,), -1, dtype=torch.int32, device="cuda")
        mod.run(build.cuda(), probe.cuda(), counts, b, p)
        torch.cuda.synchronize()
        bn, pn = build.numpy(), probe.numpy()
        ref = (np.searchsorted(bn, pn, "right") - np.searchsorted(bn, pn, "left")).astype(np.int32)
        assert np.array_equal(counts.cpu().numpy(), ref), (b, p)


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
