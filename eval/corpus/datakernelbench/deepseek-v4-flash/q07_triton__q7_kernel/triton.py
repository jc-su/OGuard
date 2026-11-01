import triton
import triton.language as tl

TYPES = {
    "supp_code_ptr": "*i32",
    "cust_code_ptr": "*i32",
    "shipdate_days_ptr": "*i32",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "n_elements": "i32",
    "shipdate_lo": "i32",
    "shipdate_hi": "i32",
    "out_revenue_ptr": "*fp64",
}

CONSTANTS = {
    "BLOCK_SIZE": 1024,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    supp_code_ptr, cust_code_ptr, shipdate_days_ptr, price_ptr, disc_ptr,
    n_elements,
    shipdate_lo, shipdate_hi,
    out_revenue_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    supp = tl.load(supp_code_ptr + offs, mask=mask, other=2)
    cust = tl.load(cust_code_ptr + offs, mask=mask, other=2)
    ship = tl.load(shipdate_days_ptr + offs, mask=mask, other=shipdate_hi + 1)

    # filter: shipdate range and (supp=0,cust=1) or (supp=1,cust=0)
    date_ok = (ship >= shipdate_lo) & (ship <= shipdate_hi)
    pair_ok = ((supp == 0) & (cust == 1)) | ((supp == 1) & (cust == 0))
    keep = mask & date_ok & pair_ok

    # early exit if nothing to load
    if tl.sum(keep.to(tl.int64)) == 0:
        return

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    supp_sel = tl.load(supp_code_ptr + offs, mask=keep, other=0)
    cust_sel = tl.load(cust_code_ptr + offs, mask=keep, other=0)

    # compute year from shipdate days
    ship64 = ship.to(tl.int64)
    ship64 = ship64 + 719468
    era = tl.where(ship64 >= 0, ship64, ship64 - 146096) // 146097
    doe = ship64 - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y)
    year_offset = year - 1995

    volume = price * (1.0 - disc)

    # group key: supp*4 + cust*2 + year_offset
    key = supp_sel.to(tl.int64) * 4 + cust_sel.to(tl.int64) * 2 + year_offset.to(tl.int64)

    # per-warp block reduction into local group accumulators
    num_groups = 8
    # Use local accumulators for each group
    # We can't use tl.zeros with non-constexpr shape, use a loop over groups
    for g in tl.static_range(8):
        g_mask = keep & (key == g)
        g_sum = tl.sum(tl.where(g_mask, volume, 0.0))
        if g_sum != 0.0:
            tl.atomic_add(out_revenue_ptr + g, g_sum)
