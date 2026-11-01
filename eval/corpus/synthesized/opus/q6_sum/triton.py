"""q6_sum: the aggregate of TPC-H Q6.

result[0] += sum of price[i] * discount[i] over the rows i with
date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi, and
quantity[i] < qty_max.

The grid holds at most PROGRAMS_PER_SM programs per SM. Each program loops
over tiles of BLOCK rows with a grid stride, keeps one partial sum per lane,
and adds the sum of its lanes to result[0] with one atomic add, so result[0]
must be zero on entry. A program reads quantity only for the rows that pass
the date and discount predicates, and price only for the rows that pass all
predicates.

The scalar annotations (tl.int32, tl.float64, tl.int64) set the Triton types
of the scalar arguments at launch. Without them, Triton types a Python float
as fp32 and a small Python int as i32. ty does not accept a Triton dtype in an
annotation, and reports the calls of the Triton library functions that are
@triton.jit functions (tl.zeros, tl.sum) as argument errors. The lines where this occurs
carry `ty: ignore` comments. A launch takes host values (tensors, Python
scalars, launch options), which the device-side annotations do not describe,
so run() casts the launcher to Callable[..., object].
"""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl

BLOCK = 1024  # Rows per tile.
NUM_WARPS = 4
PROGRAMS_PER_SM = 8

TYPES = {
    "shipdate": "*i32",
    "discount": "*fp64",
    "quantity": "*fp64",
    "price": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "disc_lo": "fp64",
    "disc_hi": "fp64",
    "qty_max": "fp64",
    "result": "*fp64",
    "n": "i64",
}
CONSTANTS = {"BLOCK": BLOCK}


@triton.jit
def kernel(
    shipdate: tl.tensor,
    discount: tl.tensor,
    quantity: tl.tensor,
    price: tl.tensor,
    date_lo: tl.int32,  # ty: ignore[invalid-type-form]
    date_hi: tl.int32,  # ty: ignore[invalid-type-form]
    disc_lo: tl.float64,  # ty: ignore[invalid-type-form]
    disc_hi: tl.float64,  # ty: ignore[invalid-type-form]
    qty_max: tl.float64,  # ty: ignore[invalid-type-form]
    result: tl.tensor,
    n: tl.int64,  # ty: ignore[invalid-type-form]
    BLOCK: tl.constexpr,
) -> None:
    acc = tl.zeros([BLOCK], tl.float64)  # ty: ignore[invalid-argument-type]
    first = tl.program_id(0).to(tl.int64) * BLOCK
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    for start in tl.range(first, n, stride):
        rows = start + tl.arange(0, BLOCK)
        in_range = rows < n
        day = tl.load(shipdate + rows, mask=in_range, other=0)
        disc = tl.load(discount + rows, mask=in_range, other=0.0)
        keep = in_range & (date_lo <= day) & (day < date_hi) & (disc_lo <= disc) & (disc <= disc_hi)
        keep = keep & (tl.load(quantity + rows, mask=keep, other=0.0) < qty_max)
        revenue = tl.load(price + rows, mask=keep, other=0.0) * disc
        acc += tl.where(keep, revenue, 0.0)
    tl.atomic_add(result, tl.sum(acc, 0), sem="relaxed")  # ty: ignore[invalid-argument-type]


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
    shipdate: torch.Tensor,
    discount: torch.Tensor,
    quantity: torch.Tensor,
    price: torch.Tensor,
    date_lo: int,
    date_hi: int,
    disc_lo: float,
    disc_hi: float,
    qty_max: float,
    result: torch.Tensor,
    n: int,
) -> None:
    """Add the Q6 revenue of the n rows to result[0], which is zero on entry."""
    check_array("shipdate", shipdate, torch.int32, n)
    check_array("discount", discount, torch.float64, n)
    check_array("quantity", quantity, torch.float64, n)
    check_array("price", price, torch.float64, n)
    check_array("result", result, torch.float64, 1)
    if n == 0:
        return
    sms = torch.cuda.get_device_properties(result.device).multi_processor_count
    grid = (min((n + BLOCK - 1) // BLOCK, PROGRAMS_PER_SM * sms),)
    launch = cast(Callable[..., object], kernel[grid])
    launch(
        shipdate,
        discount,
        quantity,
        price,
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_max,
        result,
        n,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
