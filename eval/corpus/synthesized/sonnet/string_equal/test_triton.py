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


def column(strs):
    lens = np.array([len(s) for s in strs], dtype=np.int32)
    off = np.zeros(len(strs) + 1, np.int32)
    np.cumsum(lens, out=off[1:])
    chars = np.frombuffer(b"".join(strs), dtype=np.uint8).copy()
    return off, chars


for n in (0, 1, 1000, 1000003):
    lens = rng.integers(0, 12, n)
    if n >= 1000:
        lens[:5] = [0, 300, 128, 129, 1000]
    a = [bytes(rng.integers(97, 100, l, dtype=np.uint8)) for l in lens]
    b = []
    for s in a:
        c = rng.integers(0, 4)
        if c == 0:
            b.append(s)
        elif c == 1 and len(s):
            p = rng.integers(0, len(s))
            b.append(s[:p] + bytes([(s[p] - 96) % 3 + 97]) + s[p + 1:])
        elif c == 2:
            b.append(s + b"a")
        else:
            b.append(bytes(rng.integers(97, 100, len(s), dtype=np.uint8)))
    ref = np.array([x == y for x, y in zip(a, b)], dtype=np.int8)
    ao, ac = column(a)
    bo, bc = column(b)
    out = torch.full((n,), 5, dtype=torch.int8, device="cuda")
    mod.run(d(ao), d(ac), d(bo), d(bc), out, n)
    torch.cuda.synchronize()
    if n:
        assert out.cpu().numpy().tobytes() == ref.tobytes(), n
    print("string_equal n=%d ok (%d equal)" % (n, ref.sum()))
