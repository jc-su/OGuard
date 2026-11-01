import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("bloom_probe_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def bit_positions(keys, words):
    k = keys.astype(np.uint64)
    nbits = np.uint64(32 * words)
    h1 = ((k * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)) % nbits
    h2 = ((k * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(32)) % nbits
    return h1, h2


def build(keys, words):
    bits = np.zeros(words, dtype=np.uint32)
    for p in bit_positions(keys, words):
        np.bitwise_or.at(
            bits,
            (p >> np.uint64(5)).astype(np.int64),
            np.uint32(1) << (p & np.uint64(31)).astype(np.uint32),
        )
    return bits


def reference(keys, bits, words):
    h1, h2 = bit_positions(keys, words)
    hit1 = (
        bits[(h1 >> np.uint64(5)).astype(np.int64)] >> (h1 & np.uint64(31)).astype(np.uint32)
    ) & 1
    hit2 = (
        bits[(h2 >> np.uint64(5)).astype(np.int64)] >> (h2 & np.uint64(31)).astype(np.uint32)
    ) & 1
    return (hit1 & hit2).astype(np.int8)


def check(n, words, rng):
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    inserted = rng.integers(lo, hi, max(n // 2, 1), dtype=np.int64, endpoint=True)
    bits = build(inserted, words)
    keys = rng.integers(lo, hi, n, dtype=np.int64, endpoint=True)
    from_inserted = rng.random(n) < 0.5
    keys[from_inserted] = inserted[rng.integers(0, inserted.shape[0], from_inserted.sum())]
    ref = reference(keys, bits, words)
    assert np.all(ref[from_inserted] == 1), "inserted keys must match"
    dev = "cuda"
    match = torch.empty(n, dtype=torch.int8, device=dev)
    mod.run(torch.from_numpy(keys).to(dev), torch.from_numpy(bits).to(dev), match, n, words)
    got = match.cpu().numpy()
    assert np.array_equal(got, ref), f"bloom_probe mismatch at n={n} words={words}"
    if n >= 1000 and words >= 1 << 16:
        assert ref.sum() < n, "test data should include keys the filter rejects"


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    for n in (0, 1, 1000, 1000003):
        for words in (1, 64, 1 << 16, 1 << 26):
            check(n, words, rng)
            print(f"bloom_probe triton n={n} words={words} ok")
