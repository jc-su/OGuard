import triton
import triton.language as tl

TYPES = {
    "rf_ptr": "*i32",
    "ls_ptr": "*i32",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "tax_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "shipdate_hi": "i32",
    "sum_qty_ptr": "*fp64",
    "sum_base_price_ptr": "*fp64",
    "sum_disc_price_ptr": "*fp64",
    "sum_charge_ptr": "*fp64",
    "sum_disc_ptr": "*fp64",
    "count_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    rf_ptr, ls_ptr,
    qty_ptr, price_ptr, discount_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi,
    sum_qty_ptr, sum_base_price_ptr, sum_disc_price_ptr,
    sum_charge_ptr, sum_disc_ptr, count_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # load date threshold once per program
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load all columns (unconditional loads; filtered rows contribute zero)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    rf = tl.load(rf_ptr + offs, mask=mask, other=0)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=mask, other=0.0)

    # filter predicate: shipdate <= shipdate_hi
    keep = shipdate <= shipdate_hi

    # compute projected values only for kept rows (zero otherwise)
    qty_agg = tl.where(keep, qty, 0.0)
    price_agg = tl.where(keep, price, 0.0)
    discount_agg = tl.where(keep, discount, 0.0)
    # derived values
    disc_price_agg = tl.where(keep, price * (1.0 - discount), 0.0)
    charge_agg = tl.where(keep, disc_price_agg * (1.0 + tax), 0.0)
    count_agg = tl.where(keep, 1, 0)

    # group key = returnflag_code * 256 + linestatus_code
    # use inline constant 256 instead of global
    group_key = rf * 256 + ls

    # atomic adds for each aggregate
    tl.atomic_add(sum_qty_ptr + group_key, qty_agg)
    tl.atomic_add(sum_base_price_ptr + group_key, price_agg)
    tl.atomic_add(sum_disc_price_ptr + group_key, disc_price_agg)
    tl.atomic_add(sum_charge_ptr + group_key, charge_agg)
    tl.atomic_add(sum_disc_ptr + group_key, discount_agg)
    tl.atomic_add(count_ptr + group_key, count_agg)
