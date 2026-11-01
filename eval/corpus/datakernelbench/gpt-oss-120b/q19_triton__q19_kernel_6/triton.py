import triton
import triton.language as tl

TYPES = {
    "qty_ptr": "*fp64",
    "extprice_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "brand_id_ptr": "*i32",
    "cont_grp_ptr": "*i32",
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
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 8


@triton.jit
def kernel(
    qty_ptr,                # float64 *
    extprice_ptr,           # float64 *
    disc_ptr,               # float64 *
    brand_id_ptr,           # int32 *
    cont_grp_ptr,           # int32 *
    size_ptr,               # int32 *
    brand12_id, brand23_id, brand34_id,   # int32 scalars
    sm_grp, md_grp, lg_grp,                # int32 scalars
    out_ptr,                # float64 *
    n_elements,             # int64
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all needed columns (masked)
    qty = tl.load(qty_ptr + offs, mask=mask, other=0.0)
    ext = tl.load(extprice_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    brand = tl.load(brand_id_ptr + offs, mask=mask, other=0)
    cont = tl.load(cont_grp_ptr + offs, mask=mask, other=0)
    size = tl.load(size_ptr + offs, mask=mask, other=0)

    # Three predicate groups
    cond1 = (brand == brand12_id) & (cont == sm_grp) & \
            (qty >= 1.0) & (qty <= 11.0) & (size >= 1) & (size <= 5)
    cond2 = (brand == brand23_id) & (cont == md_grp) & \
            (qty >= 10.0) & (qty <= 20.0) & (size >= 1) & (size <= 10)
    cond3 = (brand == brand34_id) & (cont == lg_grp) & \
            (qty >= 20.0) & (qty <= 30.0) & (size >= 1) & (size <= 15)

    keep = cond1 | cond2 | cond3

    # Revenue = l_extendedprice * (1 - l_discount)
    revenue = ext * (1.0 - disc)

    # Accumulate only rows that satisfy the combined mask
    tl.atomic_add(out_ptr, tl.sum(tl.where(keep, revenue, 0.0)))
