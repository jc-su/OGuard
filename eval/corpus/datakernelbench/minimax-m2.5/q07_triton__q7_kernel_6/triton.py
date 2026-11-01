import triton
import triton.language as tl

TYPES = {
    "supp_ptr": "*i32",
    "cust_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    supp_ptr, cust_ptr, shipdate_ptr,
    price_ptr, discount_ptr,
    date_lo, date_hi,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load filtering columns
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)

    # Filter by shipdate range
    keep = (shipdate >= date_lo) & (shipdate <= date_hi)

    # ----- year extraction (civil algorithm) -----
    days = shipdate.to(tl.int64) + 719468
    era = tl.where(days >= 0, days, days - 146096) // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y)
    year_offset = year - 1995

    # Compute group key
    key = tl.where(keep, supp * 4 + cust * 2 + year_offset, 0)

    # Load price & discount only where needed
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - discount)

    # ----- per-group reduction within the block -----
    # For each of the 8 possible groups, compute the sum for this block
    for i in tl.static_range(8):
        group_mask = (key == i)
        group_sum = tl.sum(tl.where(group_mask, volume, 0.0))
        # Only atomic add if there's something to add
        if group_sum != 0:
            tl.atomic_add(out_ptr + i, group_sum)
