import triton
import triton.language as tl

TYPES = {
    "quantity_ptr": "*fp64",
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "brand_id_ptr": "*i32",
    "container_grp_ptr": "*i32",
    "size_ptr": "*i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
    "brand12_id": "i32",
    "brand23_id": "i32",
    "brand34_id": "i32",
    "sm_grp": "i32",
    "md_grp": "i32",
    "lg_grp": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 8


@triton.jit
def kernel(
    quantity_ptr,
    extendedprice_ptr,
    discount_ptr,
    brand_id_ptr,
    container_grp_ptr,
    size_ptr,
    out_ptr,
    n_elements,
    brand12_id,
    brand23_id,
    brand34_id,
    sm_grp,
    md_grp,
    lg_grp,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    qty = tl.load(quantity_ptr + offs, mask=valid, other=0.0)
    brand = tl.load(brand_id_ptr + offs, mask=valid, other=0)
    cont = tl.load(container_grp_ptr + offs, mask=valid, other=0)
    size = tl.load(size_ptr + offs, mask=valid, other=0)

    cond1 = (
        (brand == brand12_id)
        & (cont == sm_grp)
        & (qty >= 1.0)
        & (qty <= 11.0)
        & (size >= 1)
        & (size <= 5)
    )
    cond2 = (
        (brand == brand23_id)
        & (cont == md_grp)
        & (qty >= 10.0)
        & (qty <= 20.0)
        & (size >= 1)
        & (size <= 10)
    )
    cond3 = (
        (brand == brand34_id)
        & (cont == lg_grp)
        & (qty >= 20.0)
        & (qty <= 30.0)
        & (size >= 1)
        & (size <= 15)
    )
    keep = valid & (cond1 | cond2 | cond3)

    price = tl.load(extendedprice_ptr + offs, mask=keep, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=keep, other=0.0)
    revenue = tl.where(keep, price * (1.0 - disc), 0.0)
    block_sum = tl.sum(revenue, axis=0)
    tl.atomic_add(out_ptr, block_sum, sem="relaxed")
