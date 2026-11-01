import triton
import triton.language as tl

TYPES = {
    "supp_ptr": "*i32",
    "cust_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "date_lo": "i32",
    "date_hi": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    supp_ptr,
    cust_ptr,
    shipdate_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ship = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    keep_date = mask & (ship >= date_lo) & (ship <= date_hi)

    supp = tl.load(supp_ptr + offs, mask=keep_date, other=0)
    cust = tl.load(cust_ptr + offs, mask=keep_date, other=0)

    # Howard Hinnant civil_from_days year extraction, matching the Torch reference.
    z = ship + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    month = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(month <= 2, y + 1, y)
    year_off = year - 1995

    valid_year = (year_off >= 0) & (year_off < 2)
    keep = keep_date & valid_year

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    volume = price * (1.0 - disc)

    key = supp * 4 + cust * 2 + year_off

    s0 = tl.sum(tl.where(keep & (key == 0), volume, 0.0))
    s1 = tl.sum(tl.where(keep & (key == 1), volume, 0.0))
    s2 = tl.sum(tl.where(keep & (key == 2), volume, 0.0))
    s3 = tl.sum(tl.where(keep & (key == 3), volume, 0.0))
    s4 = tl.sum(tl.where(keep & (key == 4), volume, 0.0))
    s5 = tl.sum(tl.where(keep & (key == 5), volume, 0.0))
    s6 = tl.sum(tl.where(keep & (key == 6), volume, 0.0))
    s7 = tl.sum(tl.where(keep & (key == 7), volume, 0.0))

    tl.atomic_add(out_ptr + 0, s0, sem="relaxed")
    tl.atomic_add(out_ptr + 1, s1, sem="relaxed")
    tl.atomic_add(out_ptr + 2, s2, sem="relaxed")
    tl.atomic_add(out_ptr + 3, s3, sem="relaxed")
    tl.atomic_add(out_ptr + 4, s4, sem="relaxed")
    tl.atomic_add(out_ptr + 5, s5, sem="relaxed")
    tl.atomic_add(out_ptr + 6, s6, sem="relaxed")
    tl.atomic_add(out_ptr + 7, s7, sem="relaxed")
