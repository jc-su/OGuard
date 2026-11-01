import triton
import triton.language as tl

TYPES = {
    "rf_ptr": "*i32",
    "ls_ptr": "*i32",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "tax_ptr": "*fp64",
    "ship_ptr": "*i32",
    "ship_hi": "i32",
    "n_elements": "i32",
    "sum_qty_ptr": "*fp64",
    "sum_base_price_ptr": "*fp64",
    "sum_disc_price_ptr": "*fp64",
    "sum_charge_ptr": "*fp64",
    "avg_qty_ptr": "*fp64",
    "avg_price_ptr": "*fp64",
    "avg_disc_ptr": "*fp64",
    "count_ptr": "*fp64",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    rf_ptr, ls_ptr, qty_ptr, price_ptr, disc_ptr, tax_ptr, ship_ptr,
    ship_hi, n_elements,
    sum_qty_ptr, sum_base_price_ptr, sum_disc_price_ptr, sum_charge_ptr,
    avg_qty_ptr, avg_price_ptr, avg_disc_ptr, count_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Filter
    ship_valid = tl.load(ship_ptr + offs, mask=mask, other=0).to(tl.int32)
    keep = mask & (ship_valid <= ship_hi)

    # Load values
    rf = tl.load(rf_ptr + offs, mask=keep, other=0).to(tl.int64)
    ls = tl.load(ls_ptr + offs, mask=keep, other=0).to(tl.int64)
    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    # Compute aggregates
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Store to output
    tl.store(sum_qty_ptr + offs, qty, mask=keep)
    tl.store(sum_base_price_ptr + offs, price, mask=keep)
    tl.store(sum_disc_price_ptr + offs, disc_price, mask=keep)
    tl.store(sum_charge_ptr + offs, charge, mask=keep)
    tl.store(avg_qty_ptr + offs, qty, mask=keep)
    tl.store(avg_price_ptr + offs, price, mask=keep)
    tl.store(avg_disc_ptr + offs, disc, mask=keep)
    tl.store(count_ptr + offs, 1.0, mask=keep)
