import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("gather_nullable_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(col, idx):
    out = np.full(idx.shape[0], np.nan)
    sel = idx >= 0
    out[sel] = col[idx[sel]]
    return out


def check(m, n, rng):
    col = rng.standard_normal(n)
    if n == 0:
        idx = np.full(m, -1, dtype=np.int32)
    else:
        idx = rng.integers(0, n, m, dtype=np.int32)
        idx[rng.random(m) < 0.3] = -1
    ref = reference(col, idx)
    dev = "cuda"
    out = torch.empty(m, dtype=torch.float64, device=dev)
    mod.run(torch.from_numpy(col).to(dev), torch.from_numpy(idx).to(dev), out, m, n)
    got = out.cpu().numpy()
    assert np.array_equal(got, ref, equal_nan=True), f"gather_nullable mismatch at m={m} n={n}"


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    for n in (0, 1, 1000, 1000003):
        check(n, n, rng)
        print(f"gather_nullable triton m={n} n={n} ok")
    for m, n in ((1000, 0), (0, 1000), (1000003, 1000), (1000, 1000003)):
        check(m, n, rng)
        print(f"gather_nullable triton m={m} n={n} ok")
