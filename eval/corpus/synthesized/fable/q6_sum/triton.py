"""q6_sum: TPC-H Q6 filtered sum of price * discount.

Each program walks tiles of BLOCK rows with a grid stride and keeps a
tile-shaped accumulator. The cheap date predicate is evaluated first and masks
the loads of the other columns, so their bytes are only fetched for candidate
rows. One tile reduction and one atomicAdd per program accumulate into result.

result must be zero on entry.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024
NUM_WARPS = 4
# Programs per SM. Fewer, longer-running programs mean fewer atomics on
# result; 4 per SM measured fastest on the A6000 (26 us at n = 1000003
# against 34 us with one tile per program).
PROGRAMS_PER_SM = 4

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
    shipdate,
    discount,
    quantity,
    price,
    date_lo: tl.int32,
    date_hi: tl.int32,
    disc_lo: tl.float64,
    disc_hi: tl.float64,
    qty_max: tl.float64,
    result,
    n: tl.int64,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(n, BLOCK)
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float64)
    for tile in range(pid, num_tiles, num_programs):
        rows = tile.to(tl.int64) * BLOCK + lanes
        in_bounds = rows < n
        d = tl.load(shipdate + rows, mask=in_bounds, other=0)
        candidate = in_bounds & (d >= date_lo) & (d < date_hi)
        disc = tl.load(discount + rows, mask=candidate, other=0.0)
        qty = tl.load(quantity + rows, mask=candidate, other=0.0)
        selected = candidate & (disc >= disc_lo) & (disc <= disc_hi) & (qty < qty_max)
        p = tl.load(price + rows, mask=selected, other=0.0)
        acc += tl.where(selected, p * disc, 0.0)
    total = tl.sum(acc, axis=0)
    if total != 0.0:
        tl.atomic_add(result, total, sem="relaxed")


def run(
    shipdate, discount, quantity, price, date_lo, date_hi, disc_lo, disc_hi, qty_max, result, n
):
    max_programs = (
        PROGRAMS_PER_SM * torch.cuda.get_device_properties(result.device).multi_processor_count
    )
    grid = (max(1, min(triton.cdiv(n, BLOCK), max_programs)),)
    kernel[grid](
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
        **CONSTANTS,
        num_warps=NUM_WARPS,
    )
