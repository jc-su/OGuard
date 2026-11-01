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
    for l, r in ((1, 1), (1000, 37), (max(n, 1), max(n // 2, 1))):
        li = torch.randint(0, l, (n,), generator=g, dtype=torch.int32)
        ri = torch.randint(0, r, (n,), generator=g, dtype=torch.int32)
        left = torch.randn(l, generator=g, dtype=torch.float64)
        right = torch.randn(r, generator=g, dtype=torch.float64)
        out = torch.zeros(n, dtype=torch.float64, device="cuda")
        mod.run(li.cuda(), ri.cuda(), left.cuda(), right.cuda(), out, n, l, r)
        torch.cuda.synchronize()
        ref = left.numpy()[li.numpy()] + right.numpy()[ri.numpy()]
        assert np.array_equal(out.cpu().numpy(), ref), (l, r)


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
