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
    nb = (n + 7) // 8
    for density in (0, 128, 256):
        valid = torch.randint(0, 256, (nb,), generator=g, dtype=torch.int32)
        if density == 0:
            valid[:] = 0
        elif density == 256:
            valid[:] = 255
        valid = valid.to(torch.uint8)
        count = torch.full((1,), 0, dtype=torch.int64, device="cuda")
        mod.run(valid.cuda(), count, n)
        torch.cuda.synchronize()
        bits = np.unpackbits(valid.numpy(), bitorder="little")[:n]
        assert int(count.item()) == int((bits == 0).sum()), density


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
