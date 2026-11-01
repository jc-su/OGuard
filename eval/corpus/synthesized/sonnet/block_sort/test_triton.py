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
    keys = rng.integers(-(2**31), 2**31, n, dtype=np.int64).astype(np.int32)
    if n >= 1000:
        keys[::7] = np.iinfo(np.int32).max
        keys[3::11] = np.iinfo(np.int32).min
    ref = keys.copy()
    for s in range(0, n, 1024):
        ref[s : s + 1024].sort()
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    mod.run(d(keys), out, n)
    torch.cuda.synchronize()
    assert out.cpu().numpy().tobytes() == ref.tobytes(), n
    print("block_sort n=%d ok" % n)
