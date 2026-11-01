"""Test the Triton bucketize kernel on the GPU against a NumPy reference.

Run with `python -P bucketize/test_triton.py` so that triton.py in this
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
BOUNDS = (0, 1, 2, 3, 17, 1000, 100000)
GUARD = 64


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bucketize_triton", Path(__file__).resolve().parent / "triton.py"
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


def test_bucketize() -> None:
    rng = np.random.default_rng(10)
    for b in BOUNDS:
        # Rounded bounds, so that some bounds repeat.
        bounds = np.sort(np.round(rng.uniform(-100.0, 100.0, b), 1))
        bounds_dev = torch.from_numpy(bounds).cuda()
        for n in SIZES:
            values = rng.uniform(-120.0, 120.0, n)
            if b > 0:
                equal = rng.random(n) < 0.3
                values[equal] = rng.choice(bounds, int(equal.sum()))
            infinite = rng.random(n) < 0.01
            values[infinite] = rng.choice(np.array([-np.inf, np.inf]), int(infinite.sum()))
            expected = np.searchsorted(bounds, values, side="right").astype(np.int32)

            buffer = torch.full((n + GUARD,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
            MOD.run(torch.from_numpy(values).cuda(), bounds_dev, buffer[:n], n, b)
            got = buffer.cpu().numpy()
            assert got[:n].tobytes() == expected.tobytes(), f"n={n} b={b}"
            assert (got[n:] == 0x5A5A5A5A).all(), f"n={n} b={b}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_bucketize()
    print("bucketize: triton tests passed")
