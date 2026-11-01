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


for n in (0, 1, 1000, 1000003):
    lens = rng.integers(0, 12, n).astype(np.int32)
    off = np.zeros(n + 1, np.int32)
    np.cumsum(lens, out=off[1:])
    chars = rng.integers(0, 256, int(off[n]), dtype=np.uint8)
    ref = np.where((chars >= 97) & (chars <= 122), chars - 32, chars).astype(np.uint8)
    out = torch.zeros(int(off[n]), dtype=torch.uint8, device="cuda")
    mod.run(d(off), d(chars), out, n)
    torch.cuda.synchronize()
    assert out.cpu().numpy().tobytes() == ref.tobytes(), n
    print("ascii_upper n=%d ok" % n)
