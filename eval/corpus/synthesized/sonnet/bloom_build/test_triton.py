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


sys.path.insert(1, "/tmp/heldout2-sonnet-triton")
import bloomref

for n in (0, 1, 1000, 1000003):
    for words in (1, 64, 1 << 15, 1 << 22, 1 << 26):
        keys = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64)
        if n:
            keys[: n // 4] = keys[n // 2 : n // 2 + n // 4]
        ref = bloomref.build(keys, words)
        bits = torch.zeros(words, dtype=torch.uint32, device="cuda")
        mod.run(d(keys), bits, n, words)
        torch.cuda.synchronize()
        assert bits.cpu().numpy().tobytes() == ref.tobytes(), (n, words)
    print("bloom_build n=%d ok" % n)
