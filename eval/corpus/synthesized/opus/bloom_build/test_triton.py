"""Test the Triton bloom_build kernel on the GPU against a NumPy reference.

Run with `python -P bloom_build/test_triton.py` so that triton.py in this
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
GUARD = 64


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bloom_build_triton", Path(__file__).resolve().parent / "triton.py"
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


def random_keys(rng: np.random.Generator, n: int) -> np.ndarray:
    info = np.iinfo(np.int64)
    keys = rng.integers(info.min, info.max, n, dtype=np.int64, endpoint=True)
    # Repeated keys from a small range.
    repeated = rng.random(n) < 0.1
    keys[repeated] = rng.integers(-1000, 1000, int(repeated.sum()), dtype=np.int64)
    special = np.array([0, -1, 1, info.min, info.max], dtype=np.int64)
    keys[: min(n, special.size)] = special[: min(n, special.size)]
    return keys


def reference(keys: np.ndarray, words: int) -> np.ndarray:
    nbits = np.uint64(32 * words)
    bits = np.zeros(words, dtype=np.uint32)
    unsigned = keys.view(np.uint64)
    for multiplier in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F):
        p = ((unsigned * np.uint64(multiplier)) >> np.uint64(32)) % nbits
        np.bitwise_or.at(
            bits, p // np.uint64(32), np.uint32(1) << (p % np.uint64(32)).astype(np.uint32)
        )
    return bits


def test_bloom_build() -> None:
    rng = np.random.default_rng(6)
    for n in SIZES:
        keys = random_keys(rng, n)
        keys_dev = torch.from_numpy(keys).cuda()
        for words in WORDS:
            expected = reference(keys, words)
            buffer = torch.zeros(words + GUARD, dtype=torch.uint32, device="cuda")
            buffer[words:] = 0x5A5A5A5A
            MOD.run(keys_dev, buffer[:words], n, words)
            got = buffer.cpu().numpy()
            assert got[:words].tobytes() == expected.tobytes(), f"n={n} words={words}"
            assert (got[words:] == 0x5A5A5A5A).all(), f"n={n} words={words}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_bloom_build()
    print("bloom_build: triton tests passed")
