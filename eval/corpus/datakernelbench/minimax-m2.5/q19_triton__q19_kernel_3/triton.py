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
    "q1_lo": "fp32",
    "q1_hi": "fp32",
    "q2_lo": "fp32",
    "q2_hi": "fp32",
    "q3_lo": "fp32",
    "q3_hi": "fp32",
    "s1_lo": "i32",
    "s1_hi": "i32",
    "s2_lo": "i32",
    "s2_hi": "i32",
    "s3_lo": "i32",
    "s3_hi": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_quantity_ptr: tl.tensor,
    l_extendedprice_ptr: tl.tensor,
    l_discount_ptr: tl.tensor,
    p_brand_id_ptr: tl.tensor,
    p_container_grp_ptr: tl.tensor,
    p_size_ptr: tl.tensor,
    brand12_id,
    brand23_id,
    brand34_id,
    sm_grp,
    md_grp,
    lg_grp,
    q1_lo,
    q1_hi,
    q2_lo,
    q2_hi,
    q3_lo,
    q3_hi,
    s1_lo,
    s1_hi,
    s2_lo,
    s2_hi,
    s3_lo,
    s3_hi,
    out_ptr: tl.tensor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32)
    mask = offs < n_elements

    # Load column data
    qty = tl.load(l_quantity_ptr + offs, mask=mask, other=0.0)
    price = tl.load(l_extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=mask, other=0.0)
    brand_id = tl.load(p_brand_id_ptr + offs, mask=mask, other=0)
    container_grp = tl.load(p_container_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(p_size_ptr + offs, mask=mask, other=0)

    # Disjunct 1: Brand#12, SM containers, qty 1‑11, size 1‑5
    cond1 = (brand_id == brand12_id) & (container_grp == sm_grp) & (qty >= q1_lo) & (qty <= q1_hi) & (size >= s1_lo) & (size <= s1_hi)

    # Disjunct 2: Brand#23, MED containers, qty 10‑20, size 1‑10
    cond2 = (brand_id == brand23_id) & (container_grp == md_grp) & (qty >= q2_lo) & (qty <= q2_hi) & (size >= s2_lo) & (size <= s2_hi)

    # Disjunct 3: Brand#34, LG containers, qty 20‑30, size 1‑15
    cond3 = (brand_id == brand34_id) & (container_grp == lg_grp) & (qty >= q3_lo) & (qty <= q3_hi) & (size >= s3_lo) & (size <= s3_hi)

    keep = mask & (cond1 | cond2 | cond3)

    revenue = price * (1.0 - discount)
    revenue_keep = tl.where(keep, revenue, 0.0)

    block_sum = tl.sum(revenue_keep, axis=0)
    tl.atomic_add(out_ptr, block_sum)
