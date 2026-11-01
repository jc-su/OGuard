import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("lag_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def partition_first(n, rng):
    """first[i] for partitions of random lengths between 1 and 40 rows."""
    lengths = rng.integers(1, 41, n + 1)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    starts = starts[starts < n]
    first = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    return first.astype(np.int32)


def reference(values, first, n, k):
    i = np.arange(n)
    src = i - k
    inside = src >= first
    out = np.full(n, np.nan)
    out[inside] = values[src[inside]]
    return out


def check(n, k, rng):
    values = rng.standard_normal(n)
    first = partition_first(n, rng)
    ref = reference(values, first, n, k)
    dev = "cuda"
    out = torch.empty(n, dtype=torch.float64, device=dev)
    mod.run(torch.from_numpy(values).to(dev), torch.from_numpy(first).to(dev), out, n, k)
    got = out.cpu().numpy()
    assert np.array_equal(got, ref, equal_nan=True), f"lag mismatch at n={n} k={k}"


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    for n in (0, 1, 1000, 1000003):
        for k in (1, 3, 1024):
            check(n, k, rng)
            print(f"lag triton n={n} k={k} ok")
