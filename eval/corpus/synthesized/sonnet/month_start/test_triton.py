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
    days = torch.randint(-719162, 2932897, (n,), generator=g, dtype=torch.int32)
    if n == 1000003:
        days[0], days[1], days[2], days[3] = -719162, 2932896, 0, 11016  # 2000-02-29
    out = torch.full((n,), 1, dtype=torch.int32, device="cuda")
    mod.run(days.cuda(), out, n)
    torch.cuda.synchronize()
    dn = days.numpy()
    ref = dn.astype("datetime64[D]").astype("datetime64[M]").astype("datetime64[D]").astype(np.int32) if n else dn
    assert np.array_equal(out.cpu().numpy(), ref)
    if n == 1000003:
        full = np.arange(-719162, 2932897, dtype=np.int32)
        o = torch.zeros(full.size, dtype=torch.int32, device="cuda")
        mod.run(torch.from_numpy(full).cuda(), o, full.size)
        r = full.astype("datetime64[D]").astype("datetime64[M]").astype("datetime64[D]").astype(np.int32)
        assert np.array_equal(o.cpu().numpy(), r)


if __name__ == "__main__":
    for n in (0, 1, 1000, 1000003):
        check(n)
        print("ok", n)
