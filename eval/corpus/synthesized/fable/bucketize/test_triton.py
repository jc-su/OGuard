import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("bucketize_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(values, bounds):
    return np.searchsorted(bounds, values, side="right").astype(np.int32)


def check(n, b, rng):
    bounds = np.sort(rng.integers(-500, 501, b).astype(np.float64) / 4.0)
    values = rng.integers(-600, 601, n).astype(np.float64) / 4.0
    ref = reference(values, bounds)
    dev = "cuda"
    out = torch.empty(n, dtype=torch.int32, device=dev)
    mod.run(torch.from_numpy(values).to(dev), torch.from_numpy(bounds).to(dev), out, n, b)
    assert np.array_equal(out.cpu().numpy(), ref), f"bucketize mismatch at n={n} b={b}"


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    for n in (0, 1, 1000, 1000003):
        for b in (0, 1, 7, 1000, 100000):
            check(n, b, rng)
            print(f"bucketize triton n={n} b={b} ok")
