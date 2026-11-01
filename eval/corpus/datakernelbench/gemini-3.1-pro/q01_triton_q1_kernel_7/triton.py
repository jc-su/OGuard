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
    "num_ls": "i32",
    "n_elements": "i32",
    "out_qty": "*fp64",
    "out_base_price": "*fp64",
    "out_disc_price": "*fp64",
    "out_charge": "*fp64",
    "out_disc": "*fp64",
    "out_count": "*i64",
}

CONSTANTS = {"NUM_GROUPS": 6, "BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    rf_ptr, ls_ptr, qty_ptr, price_ptr, disc_ptr, tax_ptr, shipdate_ptr,
    shipdate_hi, num_ls, n_elements,
    out_qty, out_base_price, out_disc_price, out_charge, out_disc, out_count,
    NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate early to quickly evaluate the global filter
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=shipdate_hi + 1)
    valid = mask & (shipdate <= shipdate_hi)
    
    # Load category keys
    rf = tl.load(rf_ptr + offs, mask=valid, other=0)
    ls = tl.load(ls_ptr + offs, mask=valid, other=0)
    valid = valid & (rf >= 0) & (ls >= 0)
    
    group_id = rf * num_ls + ls
    
    # Load measure columns mapped strictly to valid lanes
    qty = tl.load(qty_ptr + offs, mask=valid, other=0.0)
    price = tl.load(price_ptr + offs, mask=valid, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=valid, other=0.0)
    tax = tl.load(tax_ptr + offs, mask=valid, other=0.0)
    
    # Derived expressions
    disc_price = price * (1.0 - disc)
    charge = disc_price * (1.0 + tax)
    
    zero_f64 = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float64)
    one_i64 = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)
    zero_i64 = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    
    # Block-local reduction: highly performant since NUM_GROUPS is tiny (typically ~4-6)
    # Loop unrolled by Triton, eliminating large amounts of atomic contention on global memory.
    for g in tl.static_range(0, NUM_GROUPS):
        g_mask = valid & (group_id == g)
        
        sum_qty = tl.sum(tl.where(g_mask, qty, zero_f64))
        sum_price = tl.sum(tl.where(g_mask, price, zero_f64))
        sum_disc_price = tl.sum(tl.where(g_mask, disc_price, zero_f64))
        sum_charge = tl.sum(tl.where(g_mask, charge, zero_f64))
        sum_disc = tl.sum(tl.where(g_mask, disc, zero_f64))
        sum_count = tl.sum(tl.where(g_mask, one_i64, zero_i64))
        
        # Only write to global memory if the block has data for this group
        if sum_count > 0:
            tl.atomic_add(out_qty + g, sum_qty)
            tl.atomic_add(out_base_price + g, sum_price)
            tl.atomic_add(out_disc_price + g, sum_disc_price)
            tl.atomic_add(out_charge + g, sum_charge)
            tl.atomic_add(out_disc + g, sum_disc)
            tl.atomic_add(out_count + g, sum_count)
