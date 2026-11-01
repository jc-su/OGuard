import triton
import triton.language as tl

TYPES = {
    "rf_ptr": "*i32",
    "ls_ptr": "*i32",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "tax_ptr": "*fp64",
    "shipdate_ptr": "*i32",
    "sum_qty_ptr": "*fp64",
    "sum_base_price_ptr": "*fp64",
    "sum_disc_price_ptr": "*fp64",
    "sum_charge_ptr": "*fp64",
    "sum_disc_ptr": "*fp64",
    "count_ptr": "*fp64",
    "n_elements": "i32",
    "shipdate_hi_days": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 4096}

NUM_WARPS = 8


@triton.jit
def kernel(
    rf_ptr,
    ls_ptr,
    qty_ptr,
    price_ptr,
    disc_ptr,
    tax_ptr,
    shipdate_ptr,
    sum_qty_ptr,
    sum_base_price_ptr,
    sum_disc_price_ptr,
    sum_charge_ptr,
    sum_disc_ptr,
    count_ptr,
    n_elements,
    shipdate_hi_days,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi_days + 1)
    keep_date = mask & (shipdate <= shipdate_hi_days)

    rf = tl.load(rf_ptr + offs, mask=keep_date, other=-1)
    ls = tl.load(ls_ptr + offs, mask=keep_date, other=-1)

    qty = tl.load(qty_ptr + offs, mask=keep_date, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep_date, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep_date, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep_date, other=0.0)

    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    for r in tl.static_range(0, 4):
        for s in tl.static_range(0, 4):
            group_mask = keep_date & (rf == r) & (ls == s)
            idx = r * 4 + s

            tl.atomic_add(sum_qty_ptr + idx, tl.sum(tl.where(group_mask, qty, 0.0)))
            tl.atomic_add(sum_base_price_ptr + idx, tl.sum(tl.where(group_mask, price, 0.0)))
            tl.atomic_add(sum_disc_price_ptr + idx, tl.sum(tl.where(group_mask, disc_price, 0.0)))
            tl.atomic_add(sum_charge_ptr + idx, tl.sum(tl.where(group_mask, charge, 0.0)))
            tl.atomic_add(sum_disc_ptr + idx, tl.sum(tl.where(group_mask, disc, 0.0)))
            tl.atomic_add(count_ptr + idx, tl.sum(tl.where(group_mask, 1.0, 0.0)))
