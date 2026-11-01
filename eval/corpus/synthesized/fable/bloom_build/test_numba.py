"""Test the Numba bloom_build kernel on the GPU against NumPy."""

import importlib.util
import pathlib
import sys
from types import ModuleType

import numpy as np
from numba import cuda

HERE = pathlib.Path(__file__).resolve().parent
SIZES = (0, 1, 1000, 1000003)
WORDS = (1, 1 << 10, 1 << 20, 1 << 26)


def load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location("heldout2_bloom_build_numba", HERE / "numba.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load numba.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(keys: np.ndarray, words: int) -> np.ndarray:
    k = keys.astype(np.uint64)
    mask = np.uint64(32 * words - 1)
    bits = np.zeros(words, dtype=np.uint32)
    for mult in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F):
        p = ((k * np.uint64(mult)) >> np.uint64(32)) & mask
        word = (p >> np.uint64(5)).astype(np.int64)
        bit = np.uint32(1) << (p & np.uint64(31)).astype(np.uint32)
        np.bitwise_or.at(bits, word, bit)
    return bits


def check(task: ModuleType, n: int, words: int, rng: np.random.Generator) -> bool:
    keys = rng.integers(-(2**63), 2**63 - 1, size=n, dtype=np.int64)
    expected = reference(keys, words)

    d_bits = cuda.to_device(np.zeros(words, dtype=np.uint32))
    task.run(cuda.to_device(keys), d_bits, n, words)
    cuda.synchronize()
    bits = d_bits.copy_to_host()
    ok = bits.tobytes() == expected.tobytes()
    set_bits = int(np.unpackbits(expected.view(np.uint8)).sum())
    print(f"n={n} words={words}: {'PASS' if ok else 'FAIL'} ({set_bits} bits set)")
    return ok


def main() -> int:
    task = load_task()
    cuda.compile_ptx(task.kernel, task.TYPES, cc=(8, 6))
    rng = np.random.default_rng(6)
    failed = [(n, w) for n in SIZES for w in WORDS if not check(task, n, w, rng)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
