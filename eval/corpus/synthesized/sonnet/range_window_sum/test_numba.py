import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

B62 = 1 << 62

# (timestamp generator, ranges to try)
CASES = {
    "dense": (5, 100, 5000),
    "all_equal": (0, 10, B62),
    "wide": (0, B62 // 2, B62),
    "groups": (0, 1, 3000),
    "extreme": (0, B62),
    "gaps": (0, 7, 1 << 40),
}


def make_ts(n, name, rng):
    if name == "dense":
        return np.cumsum(rng.integers(0, 4, size=n)).astype(np.int64) - 1000
    if name == "all_equal":
        return np.full(n, -12345, dtype=np.int64)
    if name == "wide":
        return np.sort(rng.integers(-B62, B62 + 1, size=n, dtype=np.int64))
    if name == "groups":
        # runs of equal timestamps of about 3000 rows, crossing tile boundaries
        return np.repeat(np.arange((n + 2999) // 3000, dtype=np.int64) * 2, 3000)[:n] - 5
    if name == "extreme":
        ts = np.where(rng.random(n) < 0.5, -B62, B62).astype(np.int64)
        return np.sort(ts)
    if name == "gaps":
        step = np.where(rng.random(n) < 0.001, 1 << 38, rng.integers(0, 3, size=n))
        return np.cumsum(step).astype(np.int64)
    raise ValueError(name)


def reference(ts, values, rng_):
    n = len(ts)
    prefix = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(values.view(np.uint64), dtype=np.uint64)])
    lo = np.searchsorted(ts, ts - rng_, side="left")
    hi = np.searchsorted(ts, ts, side="right")
    return (prefix[hi] - prefix[lo]).view(np.int64)


def check(n, name, rng_, rng):
    ts = make_ts(n, name, rng)
    values = rng.integers(-(1 << 63), 1 << 63, size=n, dtype=np.int64)
    expected = reference(ts, values, rng_)
    d_out = cuda.to_device(np.full(n, 99, dtype=np.int64))
    mod.run(cuda.to_device(ts), cuda.to_device(values), d_out, n, rng_)
    cuda.synchronize()
    ok = np.array_equal(d_out.copy_to_host(), expected)
    print(f"n={n} ts={name} range={rng_}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(15)
    ok = True
    for n in (0, 1, 1000, 2048, 2049, 1000003):
        for name, ranges in CASES.items():
            for r in ranges:
                ok &= check(n, name, r, rng)
    sys.exit(0 if ok else 1)
