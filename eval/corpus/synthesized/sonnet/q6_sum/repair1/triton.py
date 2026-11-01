"""Triton kernel for q6_sum (TPC-H Q6).

Adds price[i] * discount[i] over the rows with
date_lo <= shipdate[i] < date_hi, disc_lo <= discount[i] <= disc_hi,
and quantity[i] < qty_max, into result.

Precondition: the caller zeroes result before each launch.

Each program reduces its block of rows with tl.sum and adds its
partial sum into result with one atomic add per program, instead of
one atomic add per matching row.

Repair: offs was computed as pid * BLOCK + arange in 32-bit
arithmetic. Since the checker must prove safety for any grid, pid is
not bounded by the grid this file happens to launch, so pid * BLOCK
can overflow a 32-bit index and wrap into a value that still passes
`offs < n` while addressing outside shipdate/discount/quantity/price.
offs is now built in 64-bit arithmetic throughout, so no such
wraparound is possible.
"""

import triton
import triton.language as tl

TYPES = {
    "shipdate": "*i32",
    "discount": "*fp64",
    "quantity": "*fp64",
    "price": "*fp64",
    "result": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "disc_lo": "fp64",
    "disc_hi": "fp64",
    "qty_max": "fp64",
    "n": "i64",
}

CONSTANTS = {"BLOCK": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    shipdate,
    discount,
    quantity,
    price,
    result,
    date_lo,
    date_hi,
    disc_lo,
    disc_hi,
    qty_max,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n

    sd = tl.load(shipdate + offs, mask=mask, other=0)
    d = tl.load(discount + offs, mask=mask, other=0.0)
    q = tl.load(quantity + offs, mask=mask, other=0.0)
    p = tl.load(price + offs, mask=mask, other=0.0)

    cond = (
        mask
        & (sd >= date_lo)
        & (sd < date_hi)
        & (d >= disc_lo)
        & (d <= disc_hi)
        & (q < qty_max)
    )
    val = tl.where(cond, p * d, 0.0)
    partial = tl.sum(val)
    tl.atomic_add(result, partial)


def run(
    shipdate,
    discount,
    quantity,
    price,
    result,
    date_lo,
    date_hi,
    disc_lo,
    disc_hi,
    qty_max,
    n,
):
    """Launch kernel on PyTorch CUDA tensors. result must already be zeroed."""
    if n == 0:
        return
    grid = (triton.cdiv(n, CONSTANTS["BLOCK"]),)
    kernel[grid](
        shipdate,
        discount,
        quantity,
        price,
        result,
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_max,
        n,
        BLOCK=CONSTANTS["BLOCK"],
        num_warps=NUM_WARPS,
    )
