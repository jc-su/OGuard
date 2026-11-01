"""Test the Triton kernel of bit_unpack against a NumPy reference."""

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

ARGS = ("packed", "out", "n", "w", "bits")


def make_cases(n):
    for bits in range(1, 33):
        w = (n * bits + 31) // 32
        packed = rng.integers(0, 2**32, w, dtype=np.uint64).astype(np.uint32)
        out = np.full(n, 0xDEADBEEF, dtype=np.uint32)
        yield f"bits={bits}", dict(packed=packed, out=out, n=n, w=w, bits=bits)


def check(case, got):
    n, bits = case["n"], case["bits"]
    stream = np.unpackbits(case["packed"].view(np.uint8), bitorder="little")[: n * bits]
    codes = stream.reshape(n, bits).astype(np.uint64) << np.arange(bits, dtype=np.uint64)
    return compare(got, {"out": codes.sum(axis=1, dtype=np.uint64).astype(np.uint32)})


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
