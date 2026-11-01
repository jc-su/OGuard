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
    "shipdate_hi": "i32",
    "acc_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    rf_ptr, ls_ptr,
    qty_ptr, price_ptr, disc_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi,
    acc_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load shipdate first for early filter
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    keep = mask & (shipdate <= shipdate_hi)

    # Load group keys (only meaningful if keep)
    rf = tl.load(rf_ptr + offs, mask=mask, other=0).to(tl.int64)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0).to(tl.int64)
    key = rf * 256 + ls

    # Load measure columns (conditional on keep)
    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    # Compute derived aggregates in registers
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Accumulate to global buffer using atomics
    # acc_ptr layout: [key * 6 + col_idx]
    # cols: 0:qty, 1:price, 2:disc_price, 3:charge, 4:disc, 5:count
    base = key * 6

    # Use tl.where to zero out non-kept rows for atomic_add
    tl.atomic_add(acc_ptr + base + 0, tl.where(keep, qty, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 1, tl.where(keep, price, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 2, tl.where(keep, disc_price, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 3, tl.where(keep, charge, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 4, tl.where(keep, disc, 0.0), sem="acq_rel")
    tl.atomic_add(acc_ptr + base + 5, tl.where(keep, 1.0, 0.0), sem="acq_rel")
