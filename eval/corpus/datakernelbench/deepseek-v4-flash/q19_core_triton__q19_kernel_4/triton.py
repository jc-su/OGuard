import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "quantity_ptr": "*fp64",
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
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 2048,
}

NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr, discount_ptr, quantity_ptr,
    brand_id_ptr, container_grp_ptr, size_ptr,
    brand12_id, brand23_id, brand34_id,
    sm_grp, md_grp, lg_grp,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    brand_id = tl.load(brand_id_ptr + offs, mask=mask, other=0)
    container_grp = tl.load(container_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(size_ptr + offs, mask=mask, other=0)
    quantity = tl.load(quantity_ptr + offs, mask=mask, other=0.0)

    c1 = (
        (brand_id == brand12_id)
        & (container_grp == sm_grp)
        & (quantity >= 1.0) & (quantity <= 11.0)
        & (size >= 1) & (size <= 5)
    )
    c2 = (
        (brand_id == brand23_id)
        & (container_grp == md_grp)
        & (quantity >= 10.0) & (quantity <= 20.0)
        & (size >= 1) & (size <= 10)
    )
    c3 = (
        (brand_id == brand34_id)
        & (container_grp == lg_grp)
        & (quantity >= 20.0) & (quantity <= 30.0)
        & (size >= 1) & (size <= 15)
    )
    keep = mask & (c1 | c2 | c3)

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    revenue = tl.where(keep, price * (1.0 - discount), 0.0)
    tl.atomic_add(out_ptr, tl.sum(revenue.to(tl.float64)))
