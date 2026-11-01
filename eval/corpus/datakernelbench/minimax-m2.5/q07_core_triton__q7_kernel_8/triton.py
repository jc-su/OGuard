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

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    supp_ptr, cust_ptr, shipdate_ptr, price_ptr, discount_ptr,
    date_lo, date_hi, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: filter by shipdate, compute volume, extract year, aggregate."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load columns
    supp = tl.load(supp_ptr + offs, mask=mask, other=0)
    cust = tl.load(cust_ptr + offs, mask=mask, other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # filter by shipdate range
    keep = (shipdate >= date_lo) & (shipdate <= date_hi)

    # volume = price * (1 - discount) for kept rows, zero otherwise
    volume = tl.where(keep, price * (1.0 - discount), 0.0)

    # ---- year extraction (civil_from_days) ----
    # days are int32 epoch days; we compute year only for kept rows.
    days = tl.where(keep, shipdate, 0)
    days_i64 = days.to(tl.int64)
    days_i64 = days_i64 + 719468                     # shift to 0000‑03‑01 epoch
    era = tl.where(days_i64 >= 0, days_i64, days_i64 - 146096) // 146097
    doe = days_i64 - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = tl.where(mp < 10, mp + 3, mp - 9)
    year = tl.where(m <= 2, y + 1, y).to(tl.int32)

    year_offset = year - 1995   # 0 or 1 for the 1995‑1996 window

    # composite group key: supp*4 + cust*2 + year_offset
    key = (
        tl.where(keep, supp, 0).to(tl.int64) * 4 +
        tl.where(keep, cust, 0).to(tl.int64) * 2 +
        tl.where(keep, year_offset, 0).to(tl.int64)
    )

    # atomic add to the per‑group accumulator (size 8)
    tl.atomic_add(out_ptr + key, volume)
