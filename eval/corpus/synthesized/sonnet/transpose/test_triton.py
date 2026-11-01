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
    shapes = {0: [(0, 7), (7, 0), (0, 0)], 1: [(1, 1)], 1000: [(40, 25), (1, 1000), (1000, 1), (10, 100), (100, 10)],
              1000003: [(1000003, 1), (1, 1000003), (1001, 999), (1237, 809), (65, 64)]}[n]
    g = torch.Generator().manual_seed(n)
    for rows, cols in shapes:
        src = torch.randint(-2**31, 2**31, (rows * cols,), generator=g, dtype=torch.int64).to(torch.int32)
        out = torch.zeros(rows * cols, dtype=torch.int32, device="cuda")
        mod.run(src.cuda(), out, rows, cols)
        torch.cuda.synchronize()
        ref = src.numpy().reshape(rows, cols).T.reshape(-1)
        assert np.array_equal(out.cpu().numpy(), ref), (rows, cols)


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
