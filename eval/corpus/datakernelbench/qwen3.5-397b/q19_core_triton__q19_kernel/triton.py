import triton
import triton.language as tl

TYPES = {
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "brand_ptr": "*i32",
    "cont_ptr": "*i32",
    "size_ptr": "*i32",
    "brand12_id": "i32",
    "brand23_id": "i32",
    "brand34_id": "i32",
    "sm_grp": "i32",
    "md_grp": "i32",
    "lg_grp": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    qty_ptr, price_ptr, disc_ptr,
    brand_ptr, cont_ptr, size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load cheap predicate columns first (int32)
    brand = tl.load(brand_ptr + offs, mask=mask, other=0)
    cont = tl.load(cont_ptr + offs, mask=mask, other=0)
    size = tl.load(size_ptr + offs, mask=mask, other=0)
    
    # Evaluate brand/container/size predicates first
    # Condition 1: Brand#12, SM, size 1-5
    cond1_base = (brand == brand12_id) & (cont == sm_grp) & (size >= 1) & (size <= 5)
    # Condition 2: Brand#23, MED, size 1-10
    cond2_base = (brand == brand23_id) & (cont == md_grp) & (size >= 1) & (size <= 10)
    # Condition 3: Brand#34, LG, size 1-15
    cond3_base = (brand == brand34_id) & (cont == lg_grp) & (size >= 1) & (size <= 15)
    
    base_keep = cond1_base | cond2_base | cond3_base
    
    # Only load expensive float columns where base predicates pass
    qty = tl.load(qty_ptr + offs, mask=mask & base_keep, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask & base_keep, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask & base_keep, other=0.0)
    
    # Evaluate quantity predicates
    cond1_qty = (qty >= 1.0) & (qty <= 11.0)
    cond2_qty = (qty >= 10.0) & (qty <= 20.0)
    cond3_qty = (qty >= 20.0) & (qty <= 30.0)
    
    cond1 = cond1_base & cond1_qty
    cond2 = cond2_base & cond2_qty
    cond3 = cond3_base & cond3_qty
    
    keep = cond1 | cond2 | cond3
    
    # Revenue calculation
    revenue = price * (1.0 - disc)
    
    # Sum where keep is true
    block_sum = tl.sum(tl.where(keep, revenue, 0.0))
    
    tl.atomic_add(out_ptr, block_sum)
