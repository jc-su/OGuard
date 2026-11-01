"""Test the Triton lag kernel on the GPU against a NumPy reference.

Run with `python -P lag/test_triton.py` so that triton.py in this directory
does not shadow the triton package.
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
LAGS = (1, 2, 37, 1024)
GUARD = 64


def load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lag_triton", Path(__file__).resolve().parent / "triton.py"
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


def partition_firsts(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return first[i] for partitions of 1 to 10 rows and of 1 to 3000 rows."""
    count = n + 1
    lengths = np.where(
        rng.random(count) < 0.5, rng.integers(1, 11, count), rng.integers(1, 3001, count)
    )
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.repeat(starts, lengths)[:n].astype(np.int32)


def test_lag() -> None:
    rng = np.random.default_rng(5)
    for n in SIZES:
        values = rng.standard_normal(n)
        first = partition_firsts(rng, n)
        rows = np.arange(n)
        for k in LAGS:
            src = rows - k
            has_source = src >= first
            expected = np.full(n, np.nan)
            expected[has_source] = values[src[has_source]]

            buffer = torch.full((n + GUARD,), -7.0, dtype=torch.float64, device="cuda")
            MOD.run(
                torch.from_numpy(values).cuda(), torch.from_numpy(first).cuda(), buffer[:n], n, k
            )
            got = buffer.cpu().numpy()
            assert got[:n].tobytes() == expected.tobytes(), f"n={n} k={k}"
            assert (got[n:] == -7.0).all(), f"n={n} k={k}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_lag()
    print("lag: triton tests passed")
