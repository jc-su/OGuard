import triton
import triton.language as tl

TYPES = {
    "l_quantity_ptr": "*fp64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "p_brand_id_ptr": "*i32",
    "p_container_grp_ptr": "*i32",
    "p_size_ptr": "*i32",
    "brand12_id": "i32",
    "brand23_id": "i32",
    "brand34_id": "i32",
    "sm_grp": "i32",
    "md_grp": "i32",
    "lg_grp": "i32",
    "thresh1_lo": "fp32",
    "thresh1_hi": "fp32",
    "thresh2_lo": "fp32",
    "thresh2_hi": "fp32",
    "thresh3_lo": "fp32",
    "thresh3_hi": "fp32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 1024,
}

NUM_WARPS = 8


@triton.jit
def kernel(
    l_quantity_ptr, l_extendedprice_ptr, l_discount_ptr,
    p_brand_id_ptr, p_container_grp_ptr, p_size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    thresh1_lo, thresh1_hi,
    thresh2_lo, thresh2_hi,
    thresh3_lo, thresh3_hi,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    cont_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    size_val = tl.load(p_size_ptr + offs, mask=mask, other=0)
    quantity = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    
    cond1 = (
        (brand_id == brand12_id)
        & (cont_grp == sm_grp)
        & (quantity >= 1.0)
        & (quantity <= 11.0)
        & (size_val >= 1)
        & (size_val <= 5)
    )
    cond2 = (
        (brand_id == brand23_id)
        & (cont_grp == md_grp)
        & (quantity >= 10.0)
        & (quantity <= 20.0)
        & (size_val >= 1)
        & (size_val <= 10)
    )
    cond3 = (
        (brand_id == brand34_id)
        & (cont_grp == lg_grp)
        & (quantity >= 20.0)
        & (quantity <= 30.0)
        & (size_val >= 1)
        & (size_val <= 15)
    )
    keep = mask & (cond1 | cond2 | cond3)
    
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    
    revenue = price * (1.0 - discount)
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, revenue, 0.0)))
