"""Test the Triton string_equal kernel on the GPU against a NumPy reference.

Run with `python -P string_equal/test_triton.py` so that triton.py in this
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
        "string_equal_triton", Path(__file__).resolve().parent / "triton.py"
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


def offsets_of(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)


def make_columns(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (a_offsets, a_chars, b_offsets, b_chars) for n rows.

    Row i of b is a copy of row i of a, a copy with one byte changed, a copy
    with one byte appended or removed, or a random string of the same length.
    Some rows are long enough to span several chunks of the kernel.
    """
    a_len = rng.integers(0, 40, n)
    long_rows = rng.random(n) < 0.02
    a_len[long_rows] = rng.integers(40, 400, int(long_rows.sum()))
    a_offsets = offsets_of(a_len)
    # A small alphabet, so that random strings of equal length sometimes match.
    a_chars = rng.integers(97, 100, int(a_offsets[-1]), dtype=np.uint8)

    mode = rng.integers(0, 5, n)
    b_len = a_len.copy()
    b_len[mode == 2] += 1
    shorter = (mode == 3) & (a_len > 0)
    b_len[shorter] -= 1
    b_offsets = offsets_of(b_len)
    b_row = np.repeat(np.arange(n), b_len)
    b_pos = np.arange(int(b_offsets[-1])) - b_offsets[b_row]
    from_a = b_pos < a_len[b_row]
    b_chars = rng.integers(97, 100, int(b_offsets[-1]), dtype=np.uint8)
    copied = from_a & (mode[b_row] != 4)
    b_chars[copied] = a_chars[a_offsets[b_row[copied]] + b_pos[copied]]

    changed = np.flatnonzero((mode == 1) & (a_len > 0))
    at = b_offsets[changed] + rng.integers(0, a_len[changed])
    b_chars[at] = b_chars[at] ^ rng.integers(1, 256, changed.size, dtype=np.uint8)
    return a_offsets, a_chars, b_offsets, b_chars


def reference(
    a_offsets: np.ndarray, a_chars: np.ndarray, b_offsets: np.ndarray, b_chars: np.ndarray
) -> np.ndarray:
    n = a_offsets.size - 1
    a_len = np.diff(a_offsets)
    same_length = a_len == np.diff(b_offsets)
    compared = np.where(same_length, a_len, 0)
    row = np.repeat(np.arange(n), compared)
    pos = np.arange(row.size) - offsets_of(compared)[row]
    differs = a_chars[a_offsets[row] + pos] != b_chars[b_offsets[row] + pos]
    mismatches = np.bincount(row[differs], minlength=n)
    return (same_length & (mismatches == 0)).astype(np.int8)


def test_string_equal() -> None:
    rng = np.random.default_rng(3)
    for n in SIZES:
        a_offsets, a_chars, b_offsets, b_chars = make_columns(rng, n)
        expected = reference(a_offsets, a_chars, b_offsets, b_chars)
        if n >= 1000:
            assert 0 < expected.sum() < n

        buffer = torch.full((n + GUARD,), 7, dtype=torch.int8, device="cuda")
        MOD.run(
            torch.from_numpy(a_offsets).cuda(),
            torch.from_numpy(a_chars).cuda(),
            torch.from_numpy(b_offsets).cuda(),
            torch.from_numpy(b_chars).cuda(),
            buffer[:n],
            n,
        )
        got = buffer.cpu().numpy()
        assert got[:n].tobytes() == expected.tobytes(), f"n={n}"
        assert (got[n:] == 7).all(), f"n={n}: write past the end"


if __name__ == "__main__":
    test_declared_signature()
    test_string_equal()
    print("string_equal: triton tests passed")
