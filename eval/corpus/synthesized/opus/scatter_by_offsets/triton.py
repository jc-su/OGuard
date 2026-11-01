"""Scatter by offsets: write the row id of each flagged row to its offset.

`offsets` is the exclusive prefix sum of `flags`, so with flags of 0 or 1 the
flagged rows land in `out[0:total]` in row order. This is the scatter step of
stream compaction. Each program handles one tile of BLOCK rows and reads the
offsets of the flagged rows only. The kernel does not read `total`; it is the
length of `out`.

The scalar annotations (tl.int64) set the Triton types of the scalar
arguments at launch. Without them, Triton types a small Python int as i32.
ty does not accept a Triton dtype in an annotation, so those lines carry
`ty: ignore` comments. A launch takes host values (tensors, Python scalars,
launch options), which the device-side annotations do not describe, so run()
casts the launcher to Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 1024  # Rows per program.
NUM_WARPS = 4

TYPES = {
    "flags": "*i8",
    "offsets": "*i32",
    "out": "*i32",
    "n": "i64",
    "total": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(
    flags: tl.tensor,
    offsets: tl.tensor,
    out: tl.tensor,
    n: tl.int64,  # ty: ignore[invalid-type-form]
    total: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    flagged = in_range & (tl.load(flags + rows, mask=in_range, other=0) != 0)
    position = tl.load(offsets + rows, mask=flagged, other=0)
    tl.store(out + position, rows.to(tl.int32), mask=flagged)


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
    flags: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
    n: int,
    total: int,
) -> None:
    """Scatter the flagged row ids of rows [0, n) into out[0:total]."""
    check_array("flags", flags, torch.int8, n)
    check_array("offsets", offsets, torch.int32, n)
    check_array("out", out, torch.int32, total)
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[((n + BLOCK - 1) // BLOCK,)])
    launch(flags, offsets, out, n, total, BLOCK=BLOCK, num_warps=NUM_WARPS)
