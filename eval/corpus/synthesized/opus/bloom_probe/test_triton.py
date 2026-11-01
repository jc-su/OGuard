"""Test the Triton bloom_probe kernel on the GPU against a NumPy reference.

Run with `python -P bloom_probe/test_triton.py` so that triton.py in this
directory does not shadow the triton package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
WORDS = (1, 64, 2**20, 2**26)
MULTIPLIERS = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F)
GUARD = 64


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bloom_probe_triton", Path(__file__).resolve().parent / "triton.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load()


def test_declared_signature() -> None:
    assert list(MOD.TYPES) + list(MOD.CONSTANTS) == MOD.kernel.arg_names
    signature = {**MOD.TYPES, **dict.fromkeys(MOD.CONSTANTS, "constexpr")}
    source = ASTSource(fn=MOD.kernel, signature=signature, constexprs=MOD.CONSTANTS)
    triton.compile(source, target=GPUTarget("cuda", 86, 32), options={"num_warps": MOD.NUM_WARPS})


def bit_positions(keys: np.ndarray, words: int) -> list[np.ndarray]:
    nbits = np.uint64(32 * words)
    unsigned = keys.view(np.uint64)
    return [((unsigned * np.uint64(m)) >> np.uint64(32)) % nbits for m in MULTIPLIERS]


def build(keys: np.ndarray, words: int) -> np.ndarray:
    bits = np.zeros(words, dtype=np.uint32)
    for p in bit_positions(keys, words):
        shift = (p % np.uint64(32)).astype(np.uint32)
        np.bitwise_or.at(bits, p // np.uint64(32), np.uint32(1) << shift)
    return bits


def reference(keys: np.ndarray, bits: np.ndarray, words: int) -> np.ndarray:
    match = np.ones(keys.size, dtype=bool)
    for p in bit_positions(keys, words):
        shift = (p % np.uint64(32)).astype(np.uint32)
        match &= ((bits[p // np.uint64(32)] >> shift) & np.uint32(1)) == 1
    return match.astype(np.int8)


def test_bloom_probe() -> None:
    rng = np.random.default_rng(7)
    info = np.iinfo(np.int64)
    for words in WORDS:
        # About 4 inserted keys per 32 bits, and at most 2^20 keys.
        built = rng.integers(info.min, info.max, min(4 * words, 2**20), dtype=np.int64)
        bits = build(built, words)
        bits_dev = torch.from_numpy(bits).cuda()
        for n in SIZES:
            keys = rng.integers(info.min, info.max, n, dtype=np.int64, endpoint=True)
            inserted = rng.random(n) < 0.5
            keys[inserted] = rng.choice(built, int(inserted.sum()))
            expected = reference(keys, bits, words)
            if n >= 1000 and words > 1:
                assert 0 < expected.sum() < n

            buffer = torch.full((n + GUARD,), 7, dtype=torch.int8, device="cuda")
            MOD.run(torch.from_numpy(keys).cuda(), bits_dev, buffer[:n], n, words)
            got = buffer.cpu().numpy()
            assert got[:n].tobytes() == expected.tobytes(), f"n={n} words={words}"
            assert (got[n:] == 7).all(), f"n={n} words={words}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_bloom_probe()
    print("bloom_probe: triton tests passed")
