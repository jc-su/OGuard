"""gather: out[i] = col[sel[i]] for i in [0, m).

Every value of sel must be a row id of col, that is, in [0, n). The kernel
does not read n; the argument gives the length of col. Each program handles
one tile of BLOCK output rows.

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

BLOCK = 1024  # Output rows per program.
NUM_WARPS = 4

TYPES = {
    "col": "*fp64",
    "sel": "*i32",
    "out": "*fp64",
    "m": "i64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(
    col: tl.tensor,
    sel: tl.tensor,
    out: tl.tensor,
    m: tl.int64,  # ty: ignore[invalid-type-form]
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < m
    source = tl.load(sel + rows, mask=in_range, other=0)
    tl.store(out + rows, tl.load(col + source, mask=in_range), mask=in_range)


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
    col: torch.Tensor,
    sel: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
) -> None:
    """Write col[sel[i]] into out[i] for every i in [0, m)."""
    check_array("col", col, torch.float64, n)
    check_array("sel", sel, torch.int32, m)
    check_array("out", out, torch.float64, m)
    if m == 0:
        return
    launch = cast(Callable[..., object], kernel[((m + BLOCK - 1) // BLOCK,)])
    launch(col, sel, out, m, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
