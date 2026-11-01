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
    for p in (1, 7, 64, 1000, 1024):
        keys = torch.randint(-2**63, 2**63 - 1, (n,), generator=g, dtype=torch.int64)
        if n > 10:
            keys[:5] = torch.tensor([0, -1, 2**63 - 1, -2**63, 12345])
        part = torch.full((n,), -1, dtype=torch.int32, device="cuda")
        counts = torch.zeros(p, dtype=torch.int32, device="cuda")
        mod.run(keys.cuda(), part, counts, n, p)
        torch.cuda.synchronize()
        u = keys.numpy().view(np.uint64)
        with np.errstate(over="ignore"):
            h = (u * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)
        ref = (h % np.uint64(p)).astype(np.int32)
        assert np.array_equal(part.cpu().numpy(), ref), p
        assert np.array_equal(counts.cpu().numpy(), np.bincount(ref, minlength=p).astype(np.int32)), p


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
