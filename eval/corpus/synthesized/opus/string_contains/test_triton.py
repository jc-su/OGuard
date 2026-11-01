"""GPU test of the Triton string_contains kernel against a NumPy reference.

Run from the model directory with `python -P string_contains/test_triton.py`, or
with `python -m pytest --import-mode=importlib string_contains/test_triton.py`.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import triton
from triton.compiler import ASTSource

SIZES = (0, 1, 1000, 1000003)
# Lengths 0, 1, 2, 4, 5, 8, 9, 16, 17, 18 and 300 cover every compiled variant.
TEXT_PATTERNS = (
    b"",
    b"a",
    b"ab",
    b"abca",
    b"bcabc",
    b"abcabcab",
    b"cabcabcab",
    b"abcabcabcabcabca",
    b"bcabcabcabcabcabc",
    b"cab" * 6,
    b"x" * 300,
)
# Zero bytes must not match the zero-filled register before a string's first bytes.
BINARY_PATTERNS = (b"\x00", b"\x00\x00", b"\x00\x01\x00", b"\x00" * 9, b"\x01" + b"\x00" * 16)


def _load_task() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "string_contains_triton", Path(__file__).with_name("triton.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = _load_task()


def test_declared_signature_compiles() -> None:
    signature = dict(task.TYPES) | {name: "constexpr" for name in task.CONSTANTS}
    source = ASTSource(fn=task.kernel, signature=signature, constexprs=task.CONSTANTS)
    triton.compile(source, options={"num_warps": task.NUM_WARPS})


def _column(
    rng: np.random.Generator, n: int, layout: str, alphabet: bytes
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets and characters of n strings over `alphabet`."""
    match layout:
        case "short":
            lengths = rng.integers(0, 12, n)
        case "long":
            # Mostly short strings with a few long ones that span many chunks.
            lengths = np.where(
                rng.random(n) < 0.001, rng.integers(2000, 20000, n), rng.integers(0, 40, n)
            )
        case _:
            raise ValueError(layout)
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    chars = rng.choice(np.frombuffer(alphabet, dtype=np.uint8), size=int(offsets[-1]))
    return offsets, chars


def _check(offsets: np.ndarray, chars: np.ndarray, pattern: bytes) -> None:
    n = offsets.size - 1
    k = len(pattern)
    data = chars.tobytes()
    expected = np.array(
        [pattern in data[offsets[i] : offsets[i + 1]] for i in range(n)], dtype=np.int8
    )
    match = torch.full((n,), -1, dtype=torch.int8, device="cuda")
    task.run(
        torch.from_numpy(offsets).cuda(),
        torch.from_numpy(chars).cuda(),
        torch.tensor(list(pattern), dtype=torch.uint8, device="cuda"),
        match,
        n,
        k,
    )
    torch.cuda.synchronize()
    np.testing.assert_array_equal(match.cpu().numpy(), expected)


def test_string_contains() -> None:
    rng = np.random.default_rng(4)
    for n in SIZES:
        for layout in ("short", "long"):
            offsets, chars = _column(rng, n, layout, b"abc")
            for pattern in TEXT_PATTERNS:
                _check(offsets, chars, pattern)
            offsets, chars = _column(rng, n, layout, b"\x00\x01\x02")
            for pattern in BINARY_PATTERNS:
                _check(offsets, chars, pattern)


def test_planted_matches() -> None:
    """Patterns placed at every offset around the end of the per-lane walk."""
    rng = np.random.default_rng(11)
    for k in (1, 3, 5, 9, 17, 300):
        pattern = bytes(rng.choice(np.frombuffer(b"abc", dtype=np.uint8), size=k))
        strings = []
        for length in (k, 255, 256, 257, 300, 700, 5000):
            if length < k:
                continue
            for start in {0, 1, length - k, *range(max(0, 200 - k), min(length - k, 320) + 1)}:
                text = bytearray(b"z" * length)
                text[start : start + k] = pattern
                strings.append(bytes(text))
                # Cut one byte short, so that the match does not fit.
                strings.append(bytes(text[: start + k - 1]))
        offsets = np.zeros(len(strings) + 1, dtype=np.int32)
        np.cumsum([len(s) for s in strings], out=offsets[1:])
        chars = np.frombuffer(b"".join(strings), dtype=np.uint8).copy()
        _check(offsets, chars, pattern)


def test_match_does_not_cross_strings() -> None:
    strings = [b"", b"xab", b"ca", b"", b"abca", b"abc", b"a", b"", b"bca"]
    offsets = np.zeros(len(strings) + 1, dtype=np.int32)
    np.cumsum([len(s) for s in strings], out=offsets[1:])
    chars = np.frombuffer(b"".join(strings), dtype=np.uint8).copy()
    for pattern in (b"abca", b"ca", b"a", b""):
        _check(offsets, chars, pattern)


if __name__ == "__main__":
    test_declared_signature_compiles()
    test_match_does_not_cross_strings()
    test_planted_matches()
    test_string_contains()
    print("string_contains triton: passed")
