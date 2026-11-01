import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_ptr": "*u8",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_total_ptr": "*fp64",
    "out_promo_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi,
    out_total_ptr, out_promo_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused scan:
      * filter by shipdate range
      * compute revenue = price * (1 - discount)
      * atomic sum of total revenue
      * atomic sum of revenue for promotional parts
    All arithmetic is fp64; promo flag is uint8 (0/1).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load shipdate (int32) for all active lanes.
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)

    # Conditional loads – only rows passing the date filter are fetched.
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    promo = tl.load(promo_ptr + offs, mask=keep, other=0)

    # Revenue per kept row (fp64)
    revenue = price * (1.0 - discount)

    # Reduce within the block and atomically add to global accumulators.
    total_sum = tl.sum(revenue)
    promo_sum = tl.sum(tl.where(promo != 0, revenue, 0.0))

    tl.atomic_add(out_total_ptr, total_sum)
    tl.atomic_add(out_promo_ptr, promo_sum)
