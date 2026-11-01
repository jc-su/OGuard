"""Test the Numba kernel of string_hash against a NumPy reference."""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py and triton.py in this directory would shadow the packages.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]

SIZES = (0, 1, 1000, 1000003)
rng = np.random.default_rng(20261003)


def load(name):
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{os.path.basename(HERE)}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(got, want):
    for name, expected in want.items():
        actual = got[name]
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            bad = np.flatnonzero(actual != expected) if actual.shape == expected.shape else []
            where = f" first mismatch at {bad[0]}: got {actual[bad[0]]} want {expected[bad[0]]}" if len(bad) else ""
            return f"{name} differs{where}"
    return None

ARGS = ("offsets", "chars", "out", "n")


def make_cases(n):
    for max_len, first in ((20, 0), (100, 7)):
        lengths = rng.integers(0, max_len + 1, n)
        lengths[rng.random(n) < 0.125] = 0
        lengths[rng.random(n) < 0.0002] = 5000
        offsets = np.full(n + 1, first, dtype=np.int32)
        offsets[1:] += np.cumsum(lengths).astype(np.int32)
        chars = rng.integers(0, 256, int(offsets[-1]), dtype=np.uint8)
        out = np.full(n, 0xDEADBEEF, dtype=np.uint32)
        yield f"max_len={max_len} first={first}", dict(offsets=offsets, chars=chars, out=out, n=n)


def check(case, got):
    offsets, chars, n = case["offsets"], case["chars"], case["n"]
    # Vectorized FNV-1a over all strings, one byte position at a time.
    lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)
    h = np.full(n, 2166136261, dtype=np.uint64)
    order = np.argsort(-lengths, kind="stable")
    descending = -lengths[order]
    for k in range(int(lengths.max()) if n else 0):
        live = order[: np.searchsorted(descending, -k, side="left")]
        c = chars[offsets[live] + k].astype(np.uint64)
        h[live] = ((h[live] ^ c) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)
    return compare(got, {"out": h.astype(np.uint32)})


def execute(case):
    from numba import cuda

    module = load("numba")
    dev = {k: cuda.to_device(v) if isinstance(v, np.ndarray) else v for k, v in case.items()}
    module.run(*[dev[name] for name in ARGS])
    cuda.synchronize()
    return {k: dev[k].copy_to_host() for k, v in case.items() if isinstance(v, np.ndarray)}


def main():
    failures = 0
    for n in SIZES:
        for label, case in make_cases(n):
            error = check(case, execute(case))
            if error:
                failures += 1
                print(f"FAIL n={n} {label}: {error}")
        print(f"n={n} done")
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
