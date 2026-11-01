"""Test the Triton block_sort kernel on the GPU against a NumPy reference.

Run with `python -P block_sort/test_triton.py` so that triton.py in this
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
TILE = 1024
GUARD = 64


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "block_sort_triton", Path(__file__).resolve().parent / "triton.py"
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


def test_block_sort() -> None:
    rng = np.random.default_rng(9)
    info = np.iinfo(np.int32)
    for n in SIZES:
        keys = rng.integers(info.min, info.max, n, dtype=np.int32, endpoint=True)
        # Repeated keys, and the extreme values, which equal the kernel's padding.
        repeated = rng.random(n) < 0.3
        keys[repeated] = rng.integers(-5, 5, int(repeated.sum()), dtype=np.int32)
        extreme = rng.random(n) < 0.01
        keys[extreme] = rng.choice(
            np.array([info.min, info.max], dtype=np.int32), int(extreme.sum())
        )
        expected = keys.copy()
        for start in range(0, n, TILE):
            expected[start : start + TILE].sort()

        buffer = torch.full((n + GUARD,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
        MOD.run(torch.from_numpy(keys).cuda(), buffer[:n], n)
        got = buffer.cpu().numpy()
        assert got[:n].tobytes() == expected.tobytes(), f"n={n}"
        assert (got[n:] == 0x5A5A5A5A).all(), f"n={n}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_block_sort()
    print("block_sort: triton tests passed")
