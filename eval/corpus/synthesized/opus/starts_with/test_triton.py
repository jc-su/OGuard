"""GPU test of starts_with/triton.py against a NumPy reference.

Run from the parent directory with `python -P starts_with/test_triton.py`.
The -P flag keeps this directory off sys.path, so triton.py and numba.py in
this directory do not shadow the packages.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import torch

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
MAX_SYMBOLS: int = 8
# A small alphabet with 1, 2, 3, and 4 byte UTF-8 characters, so that prefixes match often.
ALPHABET: tuple[str, ...] = ("a", "b", "é", "€", "😀")
PATTERNS: tuple[bytes, ...] = (
    b"",
    b"a",
    "é".encode(),
    "aé".encode(),
    "€😀".encode(),
    "ab€é".encode(),
    "€".encode()[:2],  # a partial UTF-8 sequence: the test is bytewise
    ("a" * 40).encode(),  # longer than every string
)


def load_module() -> ModuleType:
    path = Path(__file__).with_name("triton.py")
    spec = importlib.util.spec_from_file_location("starts_with_triton", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_column(
    n: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint8]]:
    """Return (offsets, chars) for n random strings over ALPHABET."""
    encoded = [s.encode() for s in ALPHABET]
    width = np.array([len(e) for e in encoded], dtype=np.int64)
    table = np.zeros((len(encoded), 4), dtype=np.uint8)
    for row, e in enumerate(encoded):
        table[row, : len(e)] = np.frombuffer(e, dtype=np.uint8)

    per_string = rng.integers(0, MAX_SYMBOLS + 1, n)
    symbols = rng.integers(0, len(encoded), int(per_string.sum()))
    symbol_width = width[symbols]
    starts = np.zeros(symbols.size + 1, dtype=np.int64)
    np.cumsum(symbol_width, out=starts[1:])
    chars = np.empty(int(starts[-1]), dtype=np.uint8)
    for b in range(4):
        sel = symbol_width > b
        chars[starts[:-1][sel] + b] = table[symbols[sel], b]
    symbol_index = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(per_string, out=symbol_index[1:])
    return starts[symbol_index].astype(np.int32), chars


def reference(
    offsets: npt.NDArray[np.int32], chars: npt.NDArray[np.uint8], prefix: npt.NDArray[np.uint8]
) -> npt.NDArray[np.int8]:
    n = offsets.size - 1
    k = prefix.size
    expected = np.zeros(n, dtype=np.int8)
    eligible = np.flatnonzero(np.diff(offsets) >= k)
    positions = offsets[:-1][eligible, None].astype(np.int64) + np.arange(k)[None, :]
    expected[eligible] = np.all(chars[positions] == prefix[None, :], axis=1)
    return expected


def check(
    module: ModuleType,
    offsets: npt.NDArray[np.int32],
    chars: npt.NDArray[np.uint8],
    pattern: bytes,
) -> int:
    """Run one pattern, raise on a mismatch, and return the number of matches."""
    n = offsets.size - 1
    prefix = np.frombuffer(pattern, dtype=np.uint8).copy()
    expected = reference(offsets, chars, prefix)
    if n <= 1000:
        # Check the NumPy reference against Python bytes.startswith.
        for i in range(n):
            text = chars[offsets[i] : offsets[i + 1]].tobytes()
            if expected[i] != int(text.startswith(pattern)):
                raise AssertionError(f"reference error at string {i}")
    d_offsets = torch.from_numpy(offsets).cuda()
    d_chars = torch.from_numpy(chars).cuda()
    d_prefix = torch.from_numpy(prefix).cuda()
    d_match = torch.full((n,), -1, dtype=torch.int8, device="cuda")
    module.run(d_offsets, d_chars, d_prefix, d_match, n, prefix.size)
    got = d_match.cpu().numpy()
    bad = np.flatnonzero(got != expected)
    if bad.size != 0:
        i = int(bad[0])
        raise AssertionError(
            f"n={n} pattern={pattern!r}: {bad.size} mismatches, first at {i}: "
            f"got {got[i]}, expected {expected[i]}"
        )
    return int(expected.sum())


def main() -> None:
    module = load_module()
    rng = np.random.default_rng(10)
    for n in SIZES:
        offsets, chars = make_column(n, rng)
        patterns = list(PATTERNS)
        if n > 0:
            # A pattern equal to a whole string, so that a string of exactly k bytes matches.
            i = int(np.argmax(np.diff(offsets)))
            patterns.append(chars[offsets[i] : offsets[i + 1]].tobytes())
        matches = [check(module, offsets, chars, p) for p in patterns]
        print(f"starts_with triton n={n} patterns={len(patterns)} matches={matches}: pass")


if __name__ == "__main__":
    main()
