"""Task 10, starts_with: match[i] = 1 when string i starts with the byte pattern prefix.

The string column is in the Arrow layout: string i is the byte range
chars[offsets[i]:offsets[i + 1]]. The comparison is bytewise, so for a valid
UTF-8 pattern it is also a code point prefix test. An empty pattern (k == 0)
matches every string.

Each program takes a tile of ROWS strings. A string of at least k bytes is a
candidate. In step j, the program compares byte j of every candidate with
prefix[j], and a string with a different byte stops being a candidate. The
steps end after byte k - 1, or earlier when no candidate is left. The
candidates that remain match.

The scalar annotations (tl.int64) set the Triton types of the scalar
arguments at launch. Without them, Triton types a small Python int as i32.
ty does not accept a Triton dtype in an annotation, and reports each call of a
@triton.jit function (tl.max) as an argument error. The lines where this
occurs carry `ty: ignore` comments. A launch takes host values (tensors,
Python scalars, launch options), which the device-side annotations do not
describe, so run() casts the launcher to Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

ROWS = 128  # Strings per program.
NUM_WARPS = 4

TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "prefix": "*u8",
    "match": "*i8",
    "n": "i64",
    "k": "i64",
}
CONSTANTS = {"ROWS": ROWS}


@triton.jit
def kernel(
    offsets: tl.tensor,
    chars: tl.tensor,
    prefix: tl.tensor,
    match: tl.tensor,
    n: tl.int64,  # ty: ignore[invalid-type-form]
    k: tl.int64,  # ty: ignore[invalid-type-form]
    ROWS: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    valid = rows < n
    begin = tl.load(offsets + rows, mask=valid, other=0).to(tl.int64)
    end = tl.load(offsets + rows + 1, mask=valid, other=0).to(tl.int64)
    candidate = valid & (end - begin >= k)
    step = tl.full([], 0, tl.int64)
    while (step < k) & (tl.max(candidate.to(tl.int32), 0) > 0):  # ty: ignore[invalid-argument-type]
        byte = tl.load(chars + begin + step, mask=candidate, other=0)
        candidate = candidate & (byte == tl.load(prefix + step))
        step += 1
    tl.store(match + rows, candidate.to(tl.int8), mask=valid)


def check_array(name: str, array: torch.Tensor, dtype: torch.dtype, length: int) -> None:
    """Raise ValueError unless array is a contiguous CUDA tensor of dtype and shape (length,)."""
    if (
        array.dtype != dtype
        or tuple(array.shape) != (length,)
        or not array.is_cuda
        or not array.is_contiguous()
    ):
        raise ValueError(
            f"{name}: expected a contiguous CUDA tensor of {dtype} with shape ({length},), "
            f"got {array.dtype} with shape {tuple(array.shape)} on {array.device}"
        )


def run(
    offsets: torch.Tensor,
    chars: torch.Tensor,
    prefix: torch.Tensor,
    match: torch.Tensor,
    n: int,
    k: int,
) -> None:
    """Write 1 into match[i] when string i starts with prefix[0:k], else 0."""
    check_array("offsets", offsets, torch.int32, n + 1)
    check_array("chars", chars, torch.uint8, chars.shape[0])
    check_array("prefix", prefix, torch.uint8, k)
    check_array("match", match, torch.int8, n)
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[((n + ROWS - 1) // ROWS,)])
    launch(offsets, chars, prefix, match, n, k, ROWS=ROWS, num_warps=NUM_WARPS)
