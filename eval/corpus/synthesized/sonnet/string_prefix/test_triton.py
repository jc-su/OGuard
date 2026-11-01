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
    for k in (1, 5, 17, 64):
        lens = torch.randint(0, 100, (n,), generator=g, dtype=torch.int32)
        offsets = torch.zeros(n + 1, dtype=torch.int32)
        offsets[1:] = torch.cumsum(lens, 0).to(torch.int32)
        total = int(offsets[n])
        chars = torch.randint(1, 256, (total,), generator=g, dtype=torch.int32).to(torch.uint8)
        out = torch.full((n * k,), 99, dtype=torch.uint8, device="cuda")
        mod.run(offsets.cuda(), chars.cuda(), out, n, k)
        torch.cuda.synchronize()
        ref = np.zeros((n, k), dtype=np.uint8)
        o = offsets.numpy()
        c = chars.numpy()
        for i in range(n):
            m = min(k, int(o[i + 1] - o[i]))
            ref[i, :m] = c[o[i]:o[i] + m]
        assert np.array_equal(out.cpu().numpy(), ref.reshape(-1)), k


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
