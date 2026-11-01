"""GPU test of the Numba string_contains kernel against a Python/NumPy reference."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# numba.py in this directory shadows the numba package while the script's
# directory is on sys.path, so drop it before importing numba.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import numpy as np  # noqa: E402
from numba import cuda  # noqa: E402

SIZES = (0, 1, 1000, 1000003)
PATTERN_LENGTHS = (0, 1, 2, 3, 6)
ALPHABET = 3  # bytes 'a', 'b', 'c'


def load_task():
    spec = importlib.util.spec_from_file_location("task_numba", os.path.join(HERE, "numba.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_column(n, rng):
    lengths = rng.integers(0, 25, size=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    chars = rng.integers(97, 97 + ALPHABET, size=int(offsets[n]), dtype=np.uint8)
    buf = chars.tobytes()
    strings = [buf[offsets[i] : offsets[i + 1]] for i in range(n)]
    return offsets, chars, strings


def check(task, n, k, d_offsets, d_chars, strings, rng):
    pattern = rng.integers(97, 97 + ALPHABET, size=k, dtype=np.uint8)
    pat = pattern.tobytes()
    expected = np.fromiter((pat in s for s in strings), dtype=np.int8, count=n)

    d_pattern = cuda.to_device(pattern)
    d_match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    task.run(d_offsets, d_chars, d_pattern, d_match, n, k)
    cuda.synchronize()
    got = d_match.copy_to_host()
    assert np.array_equal(got, expected), f"n={n} k={k}: match differs at {np.flatnonzero(got != expected)[:5]}"


def main():
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES)
    rng = np.random.default_rng(4)
    for n in SIZES:
        offsets, chars, strings = make_column(n, rng)
        d_offsets = cuda.to_device(offsets)
        d_chars = cuda.to_device(chars)
        for k in PATTERN_LENGTHS:
            check(task, n, k, d_offsets, d_chars, strings, rng)
        print(f"string_contains numba n={n}: ok")
    print("PASS")


if __name__ == "__main__":
    main()
