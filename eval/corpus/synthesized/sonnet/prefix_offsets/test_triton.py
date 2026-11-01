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
    for hi in (2**20, 3):
        lengths = torch.randint(0, hi + 1, (n,), generator=g, dtype=torch.int32)
        offsets = torch.full((n + 1,), -5, dtype=torch.int32, device="cuda")
        mod.run(lengths.cuda(), offsets, n)
        torch.cuda.synchronize()
        ref = np.zeros(n + 1, dtype=np.int64)
        ref[1:] = np.cumsum(lengths.numpy().astype(np.int64))
        ref = ref.astype(np.uint32).view(np.int32)
        assert np.array_equal(offsets.cpu().numpy(), ref), hi


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
