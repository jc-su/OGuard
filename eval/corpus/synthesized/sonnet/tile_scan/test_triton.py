import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("task_triton", os.path.join(HERE, "triton.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rng = np.random.default_rng(0)


def d(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


for n in (0, 1, 1000, 1024, 1025, 1000003):
    values = rng.integers(-1000000, 1000000, n).astype(np.int32)
    nt = (n + 1023) // 1024
    ref = np.zeros(n, np.int32)
    tot = np.zeros(nt, np.int32)
    for t in range(nt):
        s = values[t * 1024 : (t + 1) * 1024].astype(np.int64)
        c = np.cumsum(s)
        ref[t * 1024 : t * 1024 + len(s)] = (c - s).astype(np.int32)
        tot[t] = np.int32(c[-1])
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    totals = torch.zeros(nt, dtype=torch.int32, device="cuda")
    mod.run(d(values), out, totals, n)
    torch.cuda.synchronize()
    assert out.cpu().numpy().tobytes() == ref.tobytes(), n
    assert totals.cpu().numpy().tobytes() == tot.tobytes(), n
    print("tile_scan n=%d ok" % n)
