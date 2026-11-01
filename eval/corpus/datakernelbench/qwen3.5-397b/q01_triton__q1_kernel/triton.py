import triton
import triton.language as tl

TYPES = {
    "rf_ptr": "*i32",
    "ls_ptr": "*i32",
    "shipdate_ptr": "*i32",
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "tax_ptr": "*fp64",
    "out_ptr": "*fp64",
    "shipdate_limit": "i32",
    "n_ls": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 2048
}

NUM_WARPS = 8


@triton.jit
def kernel(
    rf_ptr, ls_ptr, shipdate_ptr,
    qty_ptr, price_ptr, disc_ptr, tax_ptr,
    out_ptr,
    shipdate_limit,
    n_ls,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all columns for this block
    rf = tl.load(rf_ptr + offs, mask=mask, other=0)
    ls = tl.load(ls_ptr + offs, mask=mask, other=0)
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_limit + 1)
    
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=mask, other=0.0)

    # Date filter
    date_ok = shipdate <= shipdate_limit
    valid_mask = mask & date_ok
    
    # Compute group index for each element: rf * n_ls + ls
    group_idx = rf * n_ls + ls
    
    # Compute derived values
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)
    
    # Mask values for valid rows only
    qty_val = tl.where(valid_mask, qty, 0.0)
    price_val = tl.where(valid_mask, price, 0.0)
    disc_price_val = tl.where(valid_mask, disc_price, 0.0)
    charge_val = tl.where(valid_mask, charge, 0.0)
    disc_val = tl.where(valid_mask, disc, 0.0)
    count_val = valid_mask.to(tl.float64)
    
    # For each of the 6 groups, accumulate using tl.where + tl.sum
    # Then atomically add to global output
    # Group 0
    g0_mask = (group_idx == 0)
    tl.atomic_add(out_ptr + 0, tl.sum(tl.where(g0_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 1, tl.sum(tl.where(g0_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 2, tl.sum(tl.where(g0_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 3, tl.sum(tl.where(g0_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 4, tl.sum(tl.where(g0_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 5, tl.sum(tl.where(g0_mask, count_val, 0.0)))
    
    # Group 1
    g1_mask = (group_idx == 1)
    tl.atomic_add(out_ptr + 6, tl.sum(tl.where(g1_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 7, tl.sum(tl.where(g1_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 8, tl.sum(tl.where(g1_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 9, tl.sum(tl.where(g1_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 10, tl.sum(tl.where(g1_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 11, tl.sum(tl.where(g1_mask, count_val, 0.0)))
    
    # Group 2
    g2_mask = (group_idx == 2)
    tl.atomic_add(out_ptr + 12, tl.sum(tl.where(g2_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 13, tl.sum(tl.where(g2_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 14, tl.sum(tl.where(g2_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 15, tl.sum(tl.where(g2_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 16, tl.sum(tl.where(g2_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 17, tl.sum(tl.where(g2_mask, count_val, 0.0)))
    
    # Group 3
    g3_mask = (group_idx == 3)
    tl.atomic_add(out_ptr + 18, tl.sum(tl.where(g3_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 19, tl.sum(tl.where(g3_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 20, tl.sum(tl.where(g3_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 21, tl.sum(tl.where(g3_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 22, tl.sum(tl.where(g3_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 23, tl.sum(tl.where(g3_mask, count_val, 0.0)))
    
    # Group 4
    g4_mask = (group_idx == 4)
    tl.atomic_add(out_ptr + 24, tl.sum(tl.where(g4_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 25, tl.sum(tl.where(g4_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 26, tl.sum(tl.where(g4_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 27, tl.sum(tl.where(g4_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 28, tl.sum(tl.where(g4_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 29, tl.sum(tl.where(g4_mask, count_val, 0.0)))
    
    # Group 5
    g5_mask = (group_idx == 5)
    tl.atomic_add(out_ptr + 30, tl.sum(tl.where(g5_mask, qty_val, 0.0)))
    tl.atomic_add(out_ptr + 31, tl.sum(tl.where(g5_mask, price_val, 0.0)))
    tl.atomic_add(out_ptr + 32, tl.sum(tl.where(g5_mask, disc_price_val, 0.0)))
    tl.atomic_add(out_ptr + 33, tl.sum(tl.where(g5_mask, charge_val, 0.0)))
    tl.atomic_add(out_ptr + 34, tl.sum(tl.where(g5_mask, disc_val, 0.0)))
    tl.atomic_add(out_ptr + 35, tl.sum(tl.where(g5_mask, count_val, 0.0)))
