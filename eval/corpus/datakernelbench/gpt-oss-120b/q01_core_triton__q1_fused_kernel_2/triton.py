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
    "out_sum_qty": "*fp64",
    "out_sum_base": "*fp64",
    "out_sum_disc_price": "*fp64",
    "out_sum_charge": "*fp64",
    "out_sum_disc": "*fp64",
    "out_count": "*i64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 2048}
NUM_WARPS = 4


@triton.jit
def kernel(
    rf_ptr,                # int32 *
    ls_ptr,                # int32 *
    qty_ptr,               # fp64 *
    price_ptr,             # fp64 *
    disc_ptr,              # fp64 *
    tax_ptr,               # fp64 *
    shipdate_ptr,          # int32 *
    shipdate_hi,           # int32 (scalar)
    out_sum_qty,           # fp64 *
    out_sum_base,          # fp64 *
    out_sum_disc_price,    # fp64 *
    out_sum_charge,        # fp64 *
    out_sum_disc,          # fp64 *
    out_count,             # int64 *
    n_elements,            # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load cheap columns (always)
    rf = tl.load(rf_ptr + offs, mask=mask, other=0).to(tl.int32)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0).to(tl.int32)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0).to(tl.int32)

    # Apply date predicate
    keep = mask & (shipdate <= shipdate_hi)

    # Conditional loads of expensive columns (zero when not kept)
    qty = tl.load(qty_ptr + offs, mask=keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=keep, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=keep, other=0.0)

    # Group key (fits in int32)
    grp = rf * 256 + ls

    # Derived columns
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)

    # Atomic per‑group accumulation
    tl.atomic_add(out_sum_qty + grp, qty)
    tl.atomic_add(out_sum_base + grp, price)
    tl.atomic_add(out_sum_disc_price + grp, disc_price)
    tl.atomic_add(out_sum_charge + grp, charge)
    tl.atomic_add(out_sum_disc + grp, disc)

    # Conditional count increment (only for rows that satisfy predicate)
    add_one = tl.where(keep, 1, 0).to(tl.int64)
    tl.atomic_add(out_count + grp, add_one)
