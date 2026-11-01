"""projection: out[i] = price[i] * (1 - discount[i]) for i in [0, n).

Each program handles one tile of BLOCK rows.

The scalar annotation (tl.int64) sets the Triton type of n at launch. Without
it, Triton types a small Python int as i32. ty does not accept a Triton dtype
in an annotation, so that line carries a `ty: ignore` comment. A launch takes
host values (tensors, Python scalars, launch options), which the device-side
annotations do not describe, so run() casts the launcher to
Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 1024  # Rows per program.
NUM_WARPS = 4

TYPES = {
    "price": "*fp64",
    "discount": "*fp64",
    "out": "*fp64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(
    price: tl.tensor,
    discount: tl.tensor,
    out: tl.tensor,
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < n
    p = tl.load(price + rows, mask=in_range)
    d = tl.load(discount + rows, mask=in_range)
    tl.store(out + rows, p * (1.0 - d), mask=in_range)


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
    price: torch.Tensor,
    discount: torch.Tensor,
    out: torch.Tensor,
    n: int,
) -> None:
    """Write price[i] * (1 - discount[i]) into out[i] for every i in [0, n)."""
    check_array("price", price, torch.float64, n)
    check_array("discount", discount, torch.float64, n)
    check_array("out", out, torch.float64, n)
    if n == 0:
        return
    launch = cast(Callable[..., object], kernel[((n + BLOCK - 1) // BLOCK,)])
    launch(price, discount, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
