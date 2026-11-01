import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "num_nations": 5,
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    nation_id_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    num_nations: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused filter (implicit), project revenue = price * (1 - discount), and
    scatter-add by nation_id (small fixed groups).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nid = tl.load(nation_id_ptr + offs, mask=mask, other=num_nations)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    revenue = price * (1.0 - disc)

    # Scatter-add per nation (small fixed group, unroll)
    for g in tl.static_range(num_nations):
        group_mask = mask & (nid == g)
        group_sum = tl.sum(tl.where(group_mask, revenue, 0.0))
        if group_sum != 0.0:
            tl.atomic_add(out_ptr + g, group_sum)
