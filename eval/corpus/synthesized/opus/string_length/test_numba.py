"""GPU test of string_length/numba.py against a NumPy reference.

Run from the parent directory with `python -P string_length/test_numba.py`.
The -P flag keeps this directory off sys.path, so numba.py in this directory
does not shadow the numba package.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
from numba import cuda

SIZES: tuple[int, ...] = (0, 1, 1000, 1000003)
MAX_CODE_POINTS: int = 16
LONG_STRING_CODE_POINTS: int = 5000
LONG_STRINGS: int = 10


def load_module() -> ModuleType:
    path = Path(__file__).with_name("numba.py")
    spec = importlib.util.spec_from_file_location("string_length_numba", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_code_points(count: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Return valid Unicode scalar values, with an equal share of 1, 2, 3, and 4 byte encodings."""
    width = rng.integers(1, 5, count)
    low = np.array([0, 0x0, 0x80, 0x800, 0x10000], dtype=np.int64)[width]
    high = np.array([0, 0x7F, 0x7FF, 0xFFFF - 0x800, 0x10FFFF], dtype=np.int64)[width]
    code_points = rng.integers(low, high + 1)
    # Move 3-byte values at or above the surrogate range (U+D800 to U+DFFF) past it.
    code_points += np.where((width == 3) & (code_points >= 0xD800), 0x800, 0)
    return code_points


def encode_utf8(
    code_points: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Return the UTF-8 bytes and the byte offset of each code point (with the total at the end)."""
    width = np.where(
        code_points < 0x80,
        1,
        np.where(code_points < 0x800, 2, np.where(code_points < 0x10000, 3, 4)),
    )
    starts = np.zeros(code_points.size + 1, dtype=np.int64)
    np.cumsum(width, out=starts[1:])
    out = np.empty(int(starts[-1]), dtype=np.uint8)
    lead_mark = np.array([0, 0x00, 0xC0, 0xE0, 0xF0], dtype=np.int64)
    for b in range(4):
        sel = width > b
        w = width[sel]
        payload = code_points[sel] >> (6 * (w - 1 - b))
        byte = lead_mark[w] | payload if b == 0 else 0x80 | (payload & 0x3F)
        out[starts[:-1][sel] + b] = byte
    return out, starts


def make_column(
    n: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint8], npt.NDArray[np.int32]]:
    """Return (offsets, chars, expected lengths) for n random UTF-8 strings."""
    per_string = rng.integers(0, MAX_CODE_POINTS + 1, n)
    if n >= 1000:
        per_string[rng.choice(n, LONG_STRINGS, replace=False)] = LONG_STRING_CODE_POINTS
    code_points = random_code_points(int(per_string.sum()), rng)
    chars, cp_starts = encode_utf8(code_points)
    cp_index = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(per_string, out=cp_index[1:])
    offsets = cp_starts[cp_index].astype(np.int32)
    # Check the generator: the bytes are valid UTF-8 with the expected code points.
    if len(chars.tobytes().decode("utf-8")) != code_points.size:
        raise AssertionError("generator produced a wrong code point count")
    if n <= 1000:
        for i in range(n):
            text = chars[offsets[i] : offsets[i + 1]].tobytes().decode("utf-8")
            if len(text) != per_string[i]:
                raise AssertionError(f"generator string {i} has the wrong length")
    return offsets, chars, per_string.astype(np.int32)


def check(module: ModuleType, n: int, rng: np.random.Generator) -> None:
    offsets, chars, expected = make_column(n, rng)
    d_offsets = cuda.to_device(offsets)
    d_chars = cuda.to_device(chars)
    d_lengths = cuda.to_device(np.full(n, -1, dtype=np.int32))
    module.run(d_offsets, d_chars, d_lengths, n)
    got = d_lengths.copy_to_host()
    bad = np.flatnonzero(got != expected)
    if bad.size != 0:
        i = int(bad[0])
        raise AssertionError(
            f"n={n}: {bad.size} mismatches, first at {i}: got {got[i]}, expected {expected[i]}"
        )
    print(f"string_length numba n={n} bytes={chars.size}: pass")


def main() -> None:
    module = load_module()
    rng = np.random.default_rng(9)
    for n in SIZES:
        check(module, n, rng)


if __name__ == "__main__":
    main()
