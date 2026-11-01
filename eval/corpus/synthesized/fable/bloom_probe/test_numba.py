"""Test the Numba bloom_probe kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
WORDS = (1, 1 << 10, 1 << 20, 1 << 26)
MULTS = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_bloom_probe_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positions(keys: np.ndarray, words: int, mult: int) -> np.ndarray:
    k = keys.astype(np.uint64)
    mask = np.uint64(32 * words - 1)
    return ((k * np.uint64(mult)) >> np.uint64(32)) & mask


def build(keys: np.ndarray, words: int) -> np.ndarray:
    bits = np.zeros(words, dtype=np.uint32)
    for mult in MULTS:
        p = positions(keys, words, mult)
        word = (p >> np.uint64(5)).astype(np.int64)
        bit = np.uint32(1) << (p & np.uint64(31)).astype(np.uint32)
        np.bitwise_or.at(bits, word, bit)
    return bits


def probe(keys: np.ndarray, bits: np.ndarray, words: int) -> np.ndarray:
    result = np.ones(keys.size, dtype=bool)
    for mult in MULTS:
        p = positions(keys, words, mult)
        word = (p >> np.uint64(5)).astype(np.int64)
        shift = (p & np.uint64(31)).astype(np.uint32)
        result &= ((bits[word] >> shift) & np.uint32(1)).astype(bool)
    return result.astype(np.int8)


def check(task: ModuleType, n: int, words: int, rng: np.random.Generator) -> bool:
    members = rng.integers(-(2**63), 2**63 - 1, size=max(n // 2, 1), dtype=np.int64)
    bits = build(members, words)
    keys = rng.integers(-(2**63), 2**63 - 1, size=n, dtype=np.int64)
    take = rng.random(n) < 0.5
    keys[take] = members[rng.integers(0, members.size, size=int(take.sum()))]
    expected = probe(keys, bits, words)

    d_match = cuda.to_device(np.full(n, -1, dtype=np.int8))
    task.run(cuda.to_device(keys), cuda.to_device(bits), d_match, n, words)
    cuda.synchronize()
    match = d_match.copy_to_host()
    ok = match.tobytes() == expected.tobytes()
    print(f"n={n} words={words}: {'PASS' if ok else 'FAIL'} ({int(expected.sum())} matches of {n})")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(7)
    failed = [(n, w) for n in SIZES for w in WORDS if not check(task, n, w, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
