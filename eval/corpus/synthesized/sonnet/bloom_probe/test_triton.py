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
        build = rng.integers(-(2**63), 2**63 - 1, max(n, 1), dtype=np.int64)
        bits_np = bloomref.build(build, words)
        keys = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64)
        if n:
            keys[::2] = build[: len(keys[::2])] if len(build) >= len(keys[::2]) else keys[::2]
        ref = bloomref.probe(keys, bits_np, words)
        match = torch.full((n,), 5, dtype=torch.int8, device="cuda")
        mod.run(d(keys), d(bits_np), match, n, words)
        torch.cuda.synchronize()
        assert match.cpu().numpy().tobytes() == ref.tobytes(), (n, words)
    print("bloom_probe n=%d ok" % n)
