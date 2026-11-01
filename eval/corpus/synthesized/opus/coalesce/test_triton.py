"""Test the Triton coalesce kernel on the GPU against a NumPy reference.

Run with `python -P coalesce/test_triton.py` so that triton.py in this
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
        "coalesce_triton", Path(__file__).resolve().parent / "triton.py"
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


def test_coalesce() -> None:
    rng = np.random.default_rng(1)
    for n in SIZES:
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        nbytes = (n + 7) // 8
        valid = rng.integers(0, 256, nbytes, dtype=np.uint8)
        # Runs of all-valid and all-null bytes, besides mixed bytes.
        valid[rng.random(nbytes) < 0.3] = 0xFF
        valid[rng.random(nbytes) < 0.3] = 0x00
        is_valid = np.unpackbits(valid, bitorder="little")[:n].astype(bool)
        expected = np.where(is_valid, a, b)

        buffer = torch.full((n + GUARD,), -7.0, dtype=torch.float64, device="cuda")
        MOD.run(
            torch.from_numpy(a).cuda(),
            torch.from_numpy(valid).cuda(),
            torch.from_numpy(b).cuda(),
            buffer[:n],
            n,
        )
        got = buffer.cpu().numpy()
        assert got[:n].tobytes() == expected.tobytes(), f"n={n}"
        assert (got[n:] == -7.0).all(), f"n={n}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_coalesce()
    print("coalesce: triton tests passed")
