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
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 1024}
NUM_WARPS = 4


@triton.jit
def kernel(
    # column pointers
    l_quantity_ptr,          # fp64
    l_extendedprice_ptr,    # fp64
    l_discount_ptr,         # fp64
    p_brand_id_ptr,         # int32
    p_container_grp_ptr,    # int32
    p_size_ptr,             # int32
    # scalar ids
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    # output per‑block array
    out_ptr,                # fp64[]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (masked)
    l_q   = tl.load(l_quantity_ptr + offs,        mask=mask, other=0.0)
    l_ep  = tl.load(l_extendedprice_ptr + offs,  mask=mask, other=0.0)
    l_d   = tl.load(l_discount_ptr + offs,       mask=mask, other=0.0)
    b_id  = tl.load(p_brand_id_ptr + offs,       mask=mask, other=0)
    grp   = tl.load(p_container_grp_ptr + offs,  mask=mask, other=0)
    sz    = tl.load(p_size_ptr + offs,           mask=mask, other=0)

    # Three disjunctive predicates
    cond1 = (b_id == brand12_id) & (grp == sm_grp) & (l_q >= 1.0) & (l_q <= 11.0) & (sz >= 1) & (sz <= 5)
    cond2 = (b_id == brand23_id) & (grp == md_grp) & (l_q >= 10.0) & (l_q <= 20.0) & (sz >= 1) & (sz <= 10)
    cond3 = (b_id == brand34_id) & (grp == lg_grp) & (l_q >= 20.0) & (l_q <= 30.0) & (sz >= 1) & (sz <= 15)

    keep = cond1 | cond2 | cond3

    # Revenue for kept rows
    revenue = tl.where(keep, l_ep * (1.0 - l_d), 0.0)

    # Reduce within the block and write to per‑block output
    block_sum = tl.sum(revenue)
    tl.store(out_ptr + pid, block_sum)
