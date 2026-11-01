import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "extendedprice_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 2048, "NUM_NATIONS": 5}

NUM_WARPS = 8


@triton.jit
def kernel(
    nation_id_ptr,
    extendedprice_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_NATIONS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nation_id = tl.load(nation_id_ptr + offs, mask=mask, other=-1)
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    one = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    zero = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    revenue = tl.where(mask, extendedprice * (one - discount), zero)

    for g in tl.static_range(0, NUM_NATIONS):
        group_sum = tl.sum(tl.where(nation_id == g, revenue, zero), axis=0)
        tl.atomic_add(out_ptr + g, group_sum)
