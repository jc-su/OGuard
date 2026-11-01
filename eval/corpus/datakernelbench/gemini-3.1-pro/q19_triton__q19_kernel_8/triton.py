import triton
import triton.language as tl

TYPES = {
    "l_quantity_ptr": "*fp64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "p_brand_id_ptr": "*i32",
    "p_container_grp_ptr": "*i32",
    "p_size_ptr": "*i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_quantity_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    p_brand_id_ptr,
    p_container_grp_ptr,
    p_size_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load condition-determining fields first
    brand = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    cont = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    qty = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    size = tl.load(p_size_ptr + offs, mask=mask, other=0)
    
    # Evaluate disjoint conditions (hardcoded integer IDs match the host encoding mapping)
    cond1 = (brand == 12) & (cont == 1) & (qty >= 1.0) & (qty <= 11.0) & (size >= 1) & (size <= 5)
    cond2 = (brand == 23) & (cont == 2) & (qty >= 10.0) & (qty <= 20.0) & (size >= 1) & (size <= 10)
    cond3 = (brand == 34) & (cont == 3) & (qty >= 20.0) & (qty <= 30.0) & (size >= 1) & (size <= 15)
    
    keep = mask & (cond1 | cond2 | cond3)
    
    # Conditional load of price and discount saves HBM bandwidth for non-matching rows
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    
    revenue = tl.where(keep, price * (1.0 - discount), 0.0)
    
    tl.atomic_add(out_ptr, tl.sum(revenue))
