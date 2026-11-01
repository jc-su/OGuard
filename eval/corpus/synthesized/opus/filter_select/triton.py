"""filter_select: write the row ids i with quantity[i] < threshold into sel.

Each program handles one tile of BLOCK rows. It ranks the matching rows of
the tile with a prefix sum and reserves one range of sel with a single atomic
add on count, so the kernel requires count[0] == 0 on entry. run() sets
count[0] to zero before the launch, so a caller of run() can pass any initial
value in count. The row ids of one tile are in row order; the tiles are in
the order of their atomic adds.

The scalar annotations (tl.float64, tl.int64) set the Triton types of the
scalar arguments at launch. Without them, Triton types a Python float as fp32
and a small Python int as i32.

ty does not accept a Triton dtype in an annotation, and reports the calls of
the Triton library functions that are @triton.jit functions (tl.sum,
tl.cumsum) as argument errors. The lines where this occurs carry `ty: ignore`
comments. A launch takes host values (tensors, Python scalars, launch
options), which the device-side annotations do not describe, so run() casts
the launcher to Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 1024  # Rows per program.
NUM_WARPS = 4

TYPES = {
    "quantity": "*fp64",
    "threshold": "fp64",
    "sel": "*i32",
    "count": "*i32",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(
    quantity: tl.tensor,
    threshold: tl.float64,  # ty: ignore[invalid-type-form]
    sel: tl.tensor,
    count: tl.tensor,
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    keep = in_range & (tl.load(quantity + rows, mask=in_range, other=0.0) < threshold)
    flag = keep.to(tl.int32)
    matches = tl.sum(flag, 0)  # ty: ignore[invalid-argument-type]
    if matches > 0:
        slot = tl.atomic_add(count, matches, sem="relaxed")
        # The exclusive prefix sum of the flags ranks the matches of the tile.
        rank = tl.cumsum(flag, 0) - flag  # ty: ignore[invalid-argument-type]
        tl.store(sel + slot + rank, rows.to(tl.int32), mask=keep)


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
    quantity: torch.Tensor,
    threshold: float,
    sel: torch.Tensor,
    count: torch.Tensor,
    n: int,
) -> None:
    """Write the selected row ids into sel and their number into count."""
    check_array("quantity", quantity, torch.float64, n)
    check_array("sel", sel, torch.int32, n)
    check_array("count", count, torch.int32, 1)
    count.zero_()
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[((n + BLOCK - 1) // BLOCK,)])
    launch(quantity, threshold, sel, count, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
