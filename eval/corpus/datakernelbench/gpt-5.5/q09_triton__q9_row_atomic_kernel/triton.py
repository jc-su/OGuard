import triton
import triton.language as tl

TYPES = {
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "supplycost_ptr": "*fp64",
    "quantity_ptr": "*fp64",
    "nationkey_ptr": "*i64",
    "year_ptr": "*i32",
    "sum_out_ptr": "*fp64",
    "cnt_out_ptr": "*fp64",
    "n_elements": "i32",
    "min_year": "i32",
    "n_years": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    extendedprice_ptr,
    discount_ptr,
    supplycost_ptr,
    quantity_ptr,
    nationkey_ptr,
    year_ptr,
    sum_out_ptr,
    cnt_out_ptr,
    n_elements,
    min_year,
    n_years,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nation = tl.load(nationkey_ptr + offs, mask=mask, other=-1)
    year = tl.load(year_ptr + offs, mask=mask, other=-2147483648)
    yoff = year - min_year
    valid = mask & (nation >= 0) & (nation < 25) & (yoff >= 0) & (yoff < n_years)

    price = tl.load(extendedprice_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=valid, other=0.0)
    cost = tl.load(supplycost_ptr + offs, mask=valid, other=0.0)
    qty = tl.load(quantity_ptr + offs, mask=valid, other=0.0)

    amount = price * (1.0 - disc) - cost * qty
    out_idx = nation * n_years + yoff
    tl.atomic_add(sum_out_ptr + out_idx, amount, sem="relaxed", mask=valid)
    tl.atomic_add(cnt_out_ptr + out_idx, tl.full((BLOCK_SIZE,), 1.0, tl.float64), sem="relaxed", mask=valid)
