"""Test the Triton kernel of group_sum_valid against a NumPy reference."""

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

ARGS = ("group", "value", "valid", "counts", "sums", "n", "g")


def make_cases(n):
    for g in (1, 7, 1024):
        for order in ("random", "sorted"):
            if order == "random":
                group = rng.integers(0, g, n).astype(np.int32)
            else:
                group = (np.arange(n, dtype=np.int64) * g // max(n, 1)).astype(np.int32)
            value = rng.integers(-(2**63), 2**63 - 1, n, dtype=np.int64, endpoint=True)
            valid = rng.integers(0, 256, (n + 7) // 8, dtype=np.uint8)
            counts = np.zeros(g, dtype=np.int64)
            sums = np.zeros(g, dtype=np.int64)
            yield f"g={g} {order}", dict(group=group, value=value, valid=valid, counts=counts, sums=sums, n=n, g=g)


def check(case, got):
    n, g = case["n"], case["g"]
    use = np.unpackbits(case["valid"], count=n, bitorder="little").astype(bool)
    grp = case["group"][use]
    counts = np.bincount(grp, minlength=g).astype(np.int64)
    sums = np.zeros(g, dtype=np.uint64)
    np.add.at(sums, grp, case["value"][use].view(np.uint64))
    return compare(got, {"counts": counts, "sums": sums.view(np.int64)})


def execute(case):
    import torch

    module = load("triton")
    dev = {k: torch.from_numpy(v.copy()).to("cuda") if isinstance(v, np.ndarray) else v for k, v in case.items()}
    module.run(*[dev[name] for name in ARGS])
    torch.cuda.synchronize()
    return {k: dev[k].cpu().numpy() for k, v in case.items() if isinstance(v, np.ndarray)}


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
