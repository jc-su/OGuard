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
for n in (0, 1, 1000, 1000003):
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    bits = rng.integers(0, 2, n).astype(np.uint8)
    valid = np.packbits(bits, bitorder="little") if n else np.zeros(0, np.uint8)
    assert valid.size == (n + 7) // 8
    ref = np.where(bits == 1, a, b)
    d = lambda x: torch.from_numpy(x).cuda()
    out = torch.zeros(n, dtype=torch.float64, device="cuda")
    mod.run(d(a), d(valid), d(b), out, n)
    torch.cuda.synchronize()
    assert out.cpu().numpy().tobytes() == ref.tobytes(), n
    print("coalesce n=%d ok" % n)
