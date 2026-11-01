import triton
import triton.language as tl

TYPES = {
    "supp_ptr": "*i32",
    "cust_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "out_ptr": "*fp64",
    "ker_n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    supp_ptr, cust_ptr, shipdate_ptr, price_ptr, disc_ptr,
    date_lo, date_hi, out_ptr, ker_n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < ker_n_elements

    # Load filter columns
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0)

    # Filter: date range and valid nation pairs
    ok_date = (shipdate >= date_lo) & (shipdate <= date_hi)
    ok_pair = ((supp == 0) & (cust == 1)) | ((supp == 1) & (cust == 0))
    keep = mask & ok_date & ok_pair

    # Conditional load of price and discount
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - disc)

    # Compute year from shipdate days using Howard Hinnant algorithm
    # Use tl.cast to convert and tl.where for conditional
    ship_int64 = shipdate.to(tl.int64)
    days = tl.where(keep, ship_int64, tl.zeros_like(ship_int64))
    days = days + 719468
    era = tl.where(days >= 0, days, days - 146096) // 146097
    doe = (days - era * 146097).to(tl.int64)
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y).to(tl.int32)

    # Compute key: supp*4 + cust*2 + (year - 1995)
    year_offset = year - 1995
    key = supp.to(tl.int64) * 4 + cust.to(tl.int64) * 2 + year_offset.to(tl.int64)

    # Scatter-add into output array (8 groups)
    vol_fp64 = volume.to(tl.float64)
    # Drop rows that don't contribute (key >= 8)
    tl.atomic_add(out_ptr + key, vol_fp64, mask=keep & (key < 8))
