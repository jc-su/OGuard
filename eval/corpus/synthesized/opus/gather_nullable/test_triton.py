"""Test the Triton gather_nullable kernel on the GPU against a NumPy reference.

Run with `python -P gather_nullable/test_triton.py` so that triton.py in this
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
GUARD = 64


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gather_nullable_triton", Path(__file__).resolve().parent / "triton.py"
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


def test_gather_nullable() -> None:
    rng = np.random.default_rng(2)
    for m in SIZES:
        for n in SIZES:
            col = rng.standard_normal(n)
            if n == 0:
                idx = np.full(m, -1, dtype=np.int32)
            else:
                idx = rng.integers(0, n, m, dtype=np.int32)
                idx[rng.random(m) < 0.2] = -1
            expected = np.full(m, np.nan)
            matched = idx >= 0
            expected[matched] = col[idx[matched]]

            buffer = torch.full((m + GUARD,), -7.0, dtype=torch.float64, device="cuda")
            MOD.run(torch.from_numpy(col).cuda(), torch.from_numpy(idx).cuda(), buffer[:m], m, n)
            got = buffer.cpu().numpy()
            assert got[:m].tobytes() == expected.tobytes(), f"m={m} n={n}"
            assert (got[m:] == -7.0).all(), f"m={m} n={n}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_gather_nullable()
    print("gather_nullable: triton tests passed")
