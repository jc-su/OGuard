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
}

CONSTANTS = {"N_YEARS": 7, "BLOCK_SIZE": 512}

NUM_WARPS = 8


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
    N_YEARS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nation = tl.load(nationkey_ptr + offs, mask=mask, other=-1)
    year = tl.load(year_ptr + offs, mask=mask, other=-2147483648)
    yoff = year - min_year

    price = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    cost = tl.load(supplycost_ptr + offs, mask=mask, other=0.0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    amount = price * (1.0 - disc) - cost * qty
    valid = mask & (nation >= 0) & (nation < 25) & (yoff >= 0) & (yoff < N_YEARS)

    for ni in tl.static_range(0, 25):
        for yi in tl.static_range(0, 16):
            if yi < N_YEARS:
                gmask = valid & (nation == ni) & (yoff == yi)
                partial_sum = tl.sum(tl.where(gmask, amount, 0.0), axis=0)
                partial_cnt = tl.sum(tl.where(gmask, 1.0, 0.0), axis=0)
                out_idx = ni * N_YEARS + yi
                has_rows = partial_cnt > 0.0
                tl.atomic_add(sum_out_ptr + out_idx, partial_sum, sem="relaxed", mask=has_rows)
                tl.atomic_add(cnt_out_ptr + out_idx, partial_cnt, sem="relaxed", mask=has_rows)
