"""Test the Numba kernel of string_to_int against a NumPy reference."""

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

import re

ARGS = ("offsets", "chars", "out", "valid", "n")
FIXED = [b"0", b"-0", b"+0", b"123", b"-123", b"+00042", b"9223372036854775807", b"-9223372036854775808",
         b"9223372036854775808", b"-9223372036854775809", b"+9223372036854775807", b"18446744073709551616",
         b"99999999999999999999", b"", b"+", b"-", b"12a", b" 12", b"1 ", b"--1", b"+-1", b"0x10", b"1e5",
         b"-" + b"0" * 30 + b"9223372036854775808"]
NUMBER = re.compile(rb"[+-]?[0-9]+")


def make_string(kind):
    if kind == 0:
        return FIXED[rng.integers(len(FIXED))]
    if kind in (1, 2):
        sign = (b"", b"-", b"+")[rng.integers(3)]
        digits = bytes(rng.integers(48, 58, rng.integers(1, 21)).astype(np.uint8))
        return sign + b"0" * int(rng.integers(4)) + digits
    if kind == 3:
        sign = (b"", b"-")[rng.integers(2)]
        return sign + b"0" * int(rng.integers(200, 500)) + str(int(rng.integers(0, 2**63))).encode()
    if kind == 4:
        s = bytearray(str(int(rng.integers(-(2**63), 2**63 - 1, endpoint=True))).encode())
        s[rng.integers(len(s))] = rng.integers(256)
        return bytes(s)
    return bytes(rng.integers(0, 256, rng.integers(6)).astype(np.uint8))


def make_cases(n):
    for rep in range(2):
        strs = [make_string(k) for k in rng.integers(0, 6, n)]
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum([len(s) for s in strs], out=offsets[1:])
        chars = np.frombuffer(b"".join(strs), dtype=np.uint8).copy()
        case = dict(offsets=offsets, chars=chars, out=np.full(n, 77, dtype=np.int64),
                    valid=np.zeros((n + 7) // 8, dtype=np.uint8), n=n)
        yield f"rep={rep}", case


def check(case, got):
    data = case["chars"].tobytes()
    offsets = case["offsets"]
    n = case["n"]
    out = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        s = data[offsets[i] : offsets[i + 1]]
        if NUMBER.fullmatch(s):
            v = int(s)
            if -(2**63) <= v < 2**63:
                out[i], ok[i] = v, True
    return compare(got, {"out": out, "valid": np.packbits(ok, bitorder="little")})


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
