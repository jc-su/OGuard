import triton
import triton.language as tl

TYPES = {
    "l_quantity_ptr": "*fp64",
    "l_extendedprice_ptr": "*fp64",
    "l_discount_ptr": "*fp64",
    "l_partkey_ptr": "*i64",
    "ship_ok_ptr": "*i8",
    "brand_map_ptr": "*i32",
    "container_map_ptr": "*i32",
    "size_map_ptr": "*i32",
    "max_partkey": "i32",
    "brand12_id": "i32",
    "brand23_id": "i32",
    "brand34_id": "i32",
    "sm_grp": "i32",
    "md_grp": "i32",
    "lg_grp": "i32",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_quantity_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_partkey_ptr,
    ship_ok_ptr,
    brand_map_ptr,
    container_map_ptr,
    size_map_ptr,
    max_partkey,
    brand12_id,
    brand23_id,
    brand34_id,
    sm_grp,
    md_grp,
    lg_grp,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = offs < n_elements

    ship_ok = tl.load(ship_ok_ptr + offs, mask=row_mask, other=0)
    ship_mask = row_mask & (ship_ok != 0)

    qty = tl.load(l_quantity_ptr + offs, mask=ship_mask, other=0.0)
    qty_candidate = ship_mask & (qty >= 1.0) & (qty <= 30.0)

    partkey = tl.load(l_partkey_ptr + offs, mask=qty_candidate, other=-1)
    part_mask = qty_candidate & (partkey >= 0) & (partkey <= max_partkey)

    brand_id = tl.load(brand_map_ptr + partkey, mask=part_mask, other=0)
    container_grp = tl.load(container_map_ptr + partkey, mask=part_mask, other=0)
    p_size = tl.load(size_map_ptr + partkey, mask=part_mask, other=0)

    cond1 = (
        (brand_id == brand12_id)
        & (container_grp == sm_grp)
        & (qty >= 1.0)
        & (qty <= 11.0)
        & (p_size >= 1)
        & (p_size <= 5)
    )
    cond2 = (
        (brand_id == brand23_id)
        & (container_grp == md_grp)
        & (qty >= 10.0)
        & (qty <= 20.0)
        & (p_size >= 1)
        & (p_size <= 10)
    )
    cond3 = (
        (brand_id == brand34_id)
        & (container_grp == lg_grp)
        & (qty >= 20.0)
        & (qty <= 30.0)
        & (p_size >= 1)
        & (p_size <= 15)
    )

    keep = part_mask & (cond1 | cond2 | cond3)

    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)

    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    contrib = tl.where(keep, price * (one - discount), zero)

    tl.atomic_add(out_ptr, tl.sum(contrib))
