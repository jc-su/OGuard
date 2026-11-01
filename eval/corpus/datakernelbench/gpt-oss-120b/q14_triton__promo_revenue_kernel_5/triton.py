import triton
import triton.language as tl

TYPES = {
    "ext_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "promo_ptr": "*i8",
    "date_lo": "i32",
    "date_hi": "i32",
    "total_out_ptr": "*fp64",
    "promo_out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    ext_ptr,                # *float64* – l_extendedprice
    disc_ptr,               # *float64* – l_discount
    shipdate_ptr,           # *int32*   – l_shipdate (days)
    promo_ptr,              # *int8*    – p_type_is_promo (0/1)
    date_lo,                # int32    – lower bound (inclusive)
    date_hi,                # int32    – upper bound (exclusive)
    total_out_ptr,          # *float64* – atomic sum of all revenue
    promo_out_ptr,          # *float64* – atomic sum of promo revenue
    n_elements,             # total number of rows
    BLOCK_SIZE: tl.constexpr,
):
    """Fused scan:
    1. Filter rows by shipdate window.
    2. Compute revenue = extendedprice * (1 - discount).
    3. Accumulate total revenue.
    4. Conditionally accumulate promo revenue where promo flag == 1.
    All work is done in a single pass to minimise HBM traffic.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load shipdate and apply date filter
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep_date = (shipdate >= date_lo) & (shipdate < date_hi)

    # Load values needed for revenue computation
    discount = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    extprice = tl.load(ext_ptr + offs, mask=mask, other=0.0)

    revenue = extprice * (1.0 - discount)

    # Total revenue contribution (only rows that survive the date filter)
    total_contrib = tl.where(mask & keep_date, revenue, 0.0)

    # Load promo flag (stored as int8, 0 or 1)
    promo_flag = tl.load(promo_ptr + offs, mask=mask, other=0)

    # Promo revenue contribution
    promo_contrib = tl.where(mask & keep_date & (promo_flag != 0), revenue, 0.0)

    # Reduce within the block and atomically add to global scalars
    tl.atomic_add(total_out_ptr, tl.sum(total_contrib))
    tl.atomic_add(promo_out_ptr, tl.sum(promo_contrib))
