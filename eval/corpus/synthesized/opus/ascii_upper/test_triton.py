"""Test the Triton ascii_upper kernel on the GPU against a NumPy reference.

Run with `python -P ascii_upper/test_triton.py` so that triton.py in this
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
        "ascii_upper_triton", Path(__file__).resolve().parent / "triton.py"
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


def test_ascii_upper() -> None:
    rng = np.random.default_rng(4)
    for n in SIZES:
        lengths = rng.integers(0, 40, n)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
        total = int(offsets[-1])
        # Every byte value, with ASCII letters and the bytes next to them
        # ('`', '{', '@', '[') more frequent.
        chars = rng.integers(0, 256, total, dtype=np.uint8)
        pick = rng.random(total) < 0.5
        chars[pick] = rng.integers(0x40, 0x7C, int(pick.sum()), dtype=np.uint8)
        lower = (chars >= ord("a")) & (chars <= ord("z"))
        expected = np.where(lower, chars - 32, chars).astype(np.uint8)

        buffer = torch.full((total + GUARD,), 0xA5, dtype=torch.uint8, device="cuda")
        MOD.run(torch.from_numpy(offsets).cuda(), torch.from_numpy(chars).cuda(), buffer[:total], n)
        got = buffer.cpu().numpy()
        assert got[:total].tobytes() == expected.tobytes(), f"n={n}"
        assert (got[total:] == 0xA5).all(), f"n={n}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_ascii_upper()
    print("ascii_upper: triton tests passed")
