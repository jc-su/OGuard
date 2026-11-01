"""Test for the Numba string_contains kernel against a NumPy/Python
reference, for n in {0, 1, 1000, 1000003}.

Run as: CUDA_VISIBLE_DEVICES=1 python -P test_numba.py
"""

import importlib.util
import pathlib
import sys

import numpy as np
from numba import cuda

_HERE = pathlib.Path(__file__).resolve().parent
ALPHABET = "ab c"  # biased small alphabet so "ab" often occurs


def _load_task_module():
    spec = importlib.util.spec_from_file_location(
        "string_contains_numba_kernel", _HERE / "numba.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_strings(n, rng):
    lengths = rng.integers(0, 21, size=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1]) if n > 0 else 0
    idx = rng.integers(0, len(ALPHABET), size=total)
    chars = np.frombuffer("".join(ALPHABET[i] for i in idx).encode(), dtype=np.uint8).copy()
    return offsets, chars


def reference(offsets, chars, pattern, n):
    pat = bytes(pattern)
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        s = bytes(chars[offsets[i]:offsets[i + 1]])
        out[i] = 1 if (pat == b"" or pat in s) else 0
    return out


def run_case(mod, n, pattern_str, rng):
    offsets, chars = make_strings(n, rng)
    pattern = np.frombuffer(pattern_str.encode(), dtype=np.uint8).copy()
    k = pattern.size
    expected = reference(offsets, chars, pattern, n) if n > 0 else np.zeros(0, dtype=np.int8)

    d_offsets = cuda.to_device(offsets)
    d_chars = cuda.to_device(chars if chars.size > 0 else np.zeros(1, dtype=np.uint8))
    d_pattern = cuda.to_device(pattern if k > 0 else np.zeros(1, dtype=np.uint8))
    d_match = cuda.to_device(np.zeros(max(n, 1), dtype=np.int8))

    mod.run(d_offsets, d_chars, d_pattern, d_match, n, k)
    cuda.synchronize()

    match = d_match.copy_to_host()[:n]
    ok = np.array_equal(match, expected)
    print(f"n={n} pattern={pattern_str!r}: hits={int(expected.sum())} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    mod = _load_task_module()
    rng = np.random.default_rng(3)
    sizes = [0, 1, 1000, 1000003]
    ok = all(run_case(mod, n, "ab", rng) for n in sizes)
    ok &= run_case(mod, 1000, "", rng)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
