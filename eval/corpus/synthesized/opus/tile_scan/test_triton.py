"""Test the Triton tile_scan kernel on the GPU against a NumPy reference.

Run with `python -P tile_scan/test_triton.py` so that triton.py in this
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
        "tile_scan_triton", Path(__file__).resolve().parent / "triton.py"
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


def test_tile_scan() -> None:
    rng = np.random.default_rng(8)
    for n in SIZES:
        # Sums of up to 1024 values of magnitude at most 10^6 fit in int32.
        values = rng.integers(-(10**6), 10**6, n, dtype=np.int32, endpoint=True)
        tiles = (n + TILE - 1) // TILE
        padded = np.zeros(tiles * TILE, dtype=np.int64)
        padded[:n] = values
        inclusive = np.cumsum(padded.reshape(tiles, TILE), axis=1)
        expected_out = (inclusive.reshape(-1)[:n] - values).astype(np.int32)
        expected_totals = inclusive[:, -1].astype(np.int32)

        out_buffer = torch.full((n + GUARD,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
        totals_buffer = torch.full((tiles + GUARD,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
        MOD.run(torch.from_numpy(values).cuda(), out_buffer[:n], totals_buffer[:tiles], n)
        out = out_buffer.cpu().numpy()
        totals = totals_buffer.cpu().numpy()
        assert out[:n].tobytes() == expected_out.tobytes(), f"n={n}"
        assert totals[:tiles].tobytes() == expected_totals.tobytes(), f"n={n}"
        assert (out[n:] == 0x5A5A5A5A).all(), f"n={n}: write past the end of out"
        assert (totals[tiles:] == 0x5A5A5A5A).all(), f"n={n}: write past the end of totals"


if __name__ == "__main__":
    test_declared_signature()
    test_tile_scan()
    print("tile_scan: triton tests passed")
