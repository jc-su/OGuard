import triton
import triton.language as tl

TYPES = {
    "qty_ptr": "*fp64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "brand_id_ptr": "*i32",
    "container_grp_ptr": "*i32",
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

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    qty_ptr, price_ptr, discount_ptr,
    brand_id_ptr, container_grp_ptr, size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load columns (rely on pointer dtype inference)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    brand = tl.load(brand_id_ptr + offs, mask=mask, other=0)
    container = tl.load(container_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(size_ptr + offs, mask=mask, other=0)

    # condition groups (exact logic as reference Torch plan)
    cond1 = (brand == brand12_id) & (container == sm_grp) & (qty >= 1.0) & (qty <= 11.0) & (size >= 1) & (size <= 5)
    cond2 = (brand == brand23_id) & (container == md_grp) & (qty >= 10.0) & (qty <= 20.0) & (size >= 1) & (size <= 10)
    cond3 = (brand == brand34_id) & (container == lg_grp) & (qty >= 20.0) & (qty <= 30.0) & (size >= 1) & (size <= 15)

    keep = mask & (cond1 | cond2 | cond3)

    revenue = price * (1.0 - discount)
    row_contrib = tl.where(keep, revenue, 0.0)
    block_sum = tl.sum(row_contrib)
    tl.atomic_add(out_ptr, block_sum)
