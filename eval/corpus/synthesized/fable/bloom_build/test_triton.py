import importlib.util
import pathlib
import sys

import numpy as np
import torch

# The task's own triton.py must not shadow the triton package when it is imported below.
HERE = pathlib.Path(__file__).resolve().parent
sys.path = [p for p in sys.path if pathlib.Path(p or ".").resolve() != HERE]
spec = importlib.util.spec_from_file_location("bloom_build_triton", HERE / "triton.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def bit_positions(keys, words):
    k = keys.astype(np.uint64)
    nbits = np.uint64(32 * words)
    h1 = ((k * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)) % nbits
    h2 = ((k * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(32)) % nbits
    return h1, h2


def reference(keys, words):
    bits = np.zeros(words, dtype=np.uint32)
    for p in bit_positions(keys, words):
        np.bitwise_or.at(
            bits,
            (p >> np.uint64(5)).astype(np.int64),
            np.uint32(1) << (p & np.uint64(31)).astype(np.uint32),
        )
    return bits


def check(n, words, rng):
    keys = rng.integers(
        np.iinfo(np.int64).min, np.iinfo(np.int64).max, n, dtype=np.int64, endpoint=True
    )
    ref = reference(keys, words)
    dev = "cuda"
    bits = torch.zeros(words, dtype=torch.uint32, device=dev)
    mod.run(torch.from_numpy(keys).to(dev), bits, n, words)
    got = bits.cpu().numpy()
    assert np.array_equal(got, ref), f"bloom_build mismatch at n={n} words={words}"


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    for n in (0, 1, 1000, 1000003):
        for words in (1, 64, 1 << 16, 1 << 26):
            check(n, words, rng)
            print(f"bloom_build triton n={n} words={words} ok")
