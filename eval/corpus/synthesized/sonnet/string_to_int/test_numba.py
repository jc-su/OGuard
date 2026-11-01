import importlib.util
import os
import re
import sys

sys.set_int_max_str_digits(0)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NUMBER = re.compile(rb"[+-]?[0-9]+")

SPECIALS = [
    b"-9223372036854775808",
    b"9223372036854775807",
    b"-9223372036854775809",
    b"9223372036854775808",
    b"+9223372036854775807",
    b"+9223372036854775808",
    b"-9223372036854775807",
    b"18446744073709551616",
    b"99999999999999999999",
    b"0",
    b"-0",
    b"+0",
    b"1",
    b"-1",
    b"+1",
    b"",
    b"+",
    b"-",
    b" 1",
    b"1 ",
    b"1_0",
    b"+-1",
    b"-+1",
    b"--1",
    b"0x1",
    b"1e5",
    b"1.0",
    b"\xd9\xa1",
    b"\x001",
    b"/1",
    b":1",
    b"0" * 100000,
    b"-" + b"0" * 100000 + b"9223372036854775808",
    b"+" + b"0" * 50000 + b"9223372036854775807",
    b"0" * 100000 + b"1" + b"0" * 5,
    b"1" * 25,
    b"007",
    b"-000000000000000000000000000012345",
]


def parse(s):
    if not NUMBER.fullmatch(s):
        return None
    v = int(s)
    return v if INT64_MIN <= v <= INT64_MAX else None


def random_strings(rng, n):
    junk = b"0123456789+- _x.a\x00\xff/:"
    result = []
    for _ in range(n):
        kind = int(rng.integers(0, 10))
        if kind < 6:
            digits = int(rng.integers(1, 21))
            s = bytes((48 + rng.integers(0, 10, digits)).astype(np.uint8))
            if kind == 1:
                s = b"-" + s
            elif kind == 2:
                s = b"+" + s
            elif kind == 3:
                s = b"0" * int(rng.integers(1, 30)) + s
        elif kind == 6:
            s = b""
        elif kind == 7:
            s = bytes([junk[int(rng.integers(0, len(junk)))] for _ in range(int(rng.integers(1, 8)))])
        else:
            s = str(int(rng.integers(INT64_MIN, INT64_MAX, dtype=np.int64))).encode()
            if kind == 9 and len(s) > 2:
                k = int(rng.integers(0, len(s)))
                s = s[:k] + junk[int(rng.integers(0, len(junk))) :][:1] + s[k + 1 :]
        result.append(s)
    return result


def expected_for(strings):
    n = len(strings)
    out = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=np.uint8)
    for i, s in enumerate(strings):
        v = parse(s)
        if v is not None:
            out[i] = v
            ok[i] = 1
    return out, np.packbits(ok, bitorder="little")


def column(strings):
    offsets = np.zeros(len(strings) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(s) for s in strings])
    return offsets, np.frombuffer(b"".join(strings), dtype=np.uint8).copy()


def cases():
    rng = np.random.default_rng(5)
    for n in (0, 1, 1000, 1000003):
        yield f"n={n}", random_strings(rng, n)
    yield "specials", SPECIALS
    yield "specials x9 (n not multiple of 8)", (SPECIALS * 9)[:331]
    yield "all empty", [b""] * 1000
    yield "all valid", [str(int(v)).encode() for v in rng.integers(INT64_MIN, INT64_MAX, 1003, dtype=np.int64)]


failed = 0
for name, strings in cases():
    n = len(strings)
    offsets, chars = column(strings)
    expected_out, expected_valid = expected_for(strings)
    d_out = cuda.to_device(np.full(n, 0x7B7B7B7B7B7B7B7B, dtype=np.int64))
    d_valid = cuda.to_device(np.zeros((n + 7) // 8, dtype=np.uint8))
    task.run(cuda.to_device(offsets), cuda.to_device(chars), d_out, d_valid, n)
    ok = np.array_equal(d_out.copy_to_host(), expected_out) and np.array_equal(d_valid.copy_to_host(), expected_valid)
    print(("PASS" if ok else "FAIL"), name)
    failed += not ok
sys.exit(1 if failed else 0)
