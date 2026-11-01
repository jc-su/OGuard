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
    for w in (1, 2, 3, 17, 255, 256, 1000, 1023, 1024):
        values = torch.randint(-2**63, 2**63 - 1, (n,), generator=g, dtype=torch.int64)
        if n > 3:
            values[0] = -2**63
            values[n - 1] = 2**63 - 1
        out = torch.zeros(n, dtype=torch.int64, device="cuda")
        mod.run(values.cuda(), out, n, w)
        torch.cuda.synchronize()
        v = values.numpy()
        ref = v.copy()
        for d in range(1, min(w, n)):
            np.maximum(ref[d:], v[: n - d], out=ref[d:])
        assert np.array_equal(out.cpu().numpy(), ref), w


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
