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
    "n_elements": "i32"
}

CONSTANTS = {
    "brand12_id": 12,
    "brand23_id": 23,
    "brand34_id": 34,
    "sm_grp": 1,
    "md_grp": 2,
    "lg_grp": 3,
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 16


@triton.jit
def kernel(
    l_quantity_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    p_brand_id_ptr,
    p_container_grp_ptr,
    p_size_ptr,
    brand12_id: tl.constexpr,
    brand23_id: tl.constexpr,
    brand34_id: tl.constexpr,
    sm_grp: tl.constexpr,
    md_grp: tl.constexpr,
    lg_grp: tl.constexpr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load filter columns first (int32 - cheaper)
    p_brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    p_container_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    p_size = tl.load(p_size_ptr + offs, mask=mask, other=0)
    l_quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    
    # Compute predicates
    cond1 = (
        (p_brand_id == brand12_id)
        & (p_container_grp == sm_grp)
        & (l_quantity >= 1.0)
        & (l_quantity <= 11.0)
        & (p_size >= 1)
        & (p_size <= 5)
    )
    
    cond2 = (
        (p_brand_id == brand23_id)
        & (p_container_grp == md_grp)
        & (l_quantity >= 10.0)
        & (l_quantity <= 20.0)
        & (p_size >= 1)
        & (p_size <= 10)
    )
    
    cond3 = (
        (p_brand_id == brand34_id)
        & (p_container_grp == lg_grp)
        & (l_quantity >= 20.0)
        & (l_quantity <= 30.0)
        & (p_size >= 1)
        & (p_size <= 15)
    )
    
    keep = cond1 | cond2 | cond3
    
    # Load expensive columns only for computation (still masked load)
    l_extendedprice = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    l_discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    
    # Compute revenue
    revenue = l_extendedprice * (1.0 - l_discount)
    
    # Sum masked revenue
    block_sum = tl.sum(tl.where(keep, revenue, 0.0))
    
    # Atomic add to global sum
    tl.atomic_add(out_ptr, block_sum)
