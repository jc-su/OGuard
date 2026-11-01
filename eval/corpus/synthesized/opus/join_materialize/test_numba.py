"""Test the Numba kernel of join_materialize against a NumPy reference."""

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

ARGS = ("build", "probe", "offsets", "out_build", "out_probe", "b", "p")


def make_case(b, p, key_range, heavy):
    build = rng.integers(0, key_range, b, dtype=np.int64)
    if heavy:
        build[::2] = 42
    build.sort()
    probe = rng.integers(0, key_range + key_range // 4 + 1, p, dtype=np.int64)
    if heavy:
        probe[::10] = 42
    counts = np.searchsorted(build, probe, side="right") - np.searchsorted(build, probe, side="left")
    offsets = np.zeros(p + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    out_build = np.full(total, -7, dtype=np.int32)
    out_probe = np.full(total, -7, dtype=np.int32)
    return dict(build=build, probe=probe, offsets=offsets, out_build=out_build, out_probe=out_probe, b=b, p=p)


def make_cases(n):
    yield "unique", make_case(n, n, n + 1, False)
    yield "duplicates", make_case(n, n, n // 8 + 1, False)
    yield "heavy", make_case(n, n // 5000 + 2, 1000, True)
    yield "sparse", make_case(n, n // 3 + 1, 4 * n + 1, False)


def check(case, got):
    build, probe, offsets = case["build"], case["probe"], case["offsets"]
    counts = np.diff(offsets).astype(np.int64)
    rows = np.repeat(np.arange(len(probe), dtype=np.int64), counts)
    first = np.searchsorted(build, probe, side="left")
    j = np.arange(int(offsets[-1]), dtype=np.int64) - np.repeat(offsets[:-1].astype(np.int64), counts)
    want_build = (np.repeat(first, counts) + j).astype(np.int32)
    return compare(got, {"out_build": want_build, "out_probe": rows.astype(np.int32)})


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
