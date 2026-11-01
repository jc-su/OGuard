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
    values = torch.randint(-2**62, 2**62, (n,), generator=g, dtype=torch.int64)
    keep = torch.randint(0, 3, (n,), generator=g, dtype=torch.int8)
    out = torch.full((n,), -7, dtype=torch.int64, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    mod.run(values.cuda(), keep.cuda(), out, count, n)
    torch.cuda.synchronize()
    sel = values[keep != 0]
    c = int(count.item())
    assert c == sel.numel(), (c, sel.numel())
    assert torch.equal(out[:c].cpu().sort().values, sel.sort().values)


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
