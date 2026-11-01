import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
    "num_nations": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    nation_id_ptr, price_ptr, discount_ptr,
    out_ptr, n_elements, num_nations,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    nation_id = tl.load(nation_id_ptr + offs, mask=mask, other=num_nations)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    revenue = price * (1.0 - discount)
    valid = mask & (nation_id < num_nations)
    
    # Per-nation accumulator using static range for small fixed groups
    # num_nations is a constexpr via passed value (will be compiled per query)
    # But we need it as a compile-time constant for tl.zeros shape
    # Instead, use a loop: accumulate revenue per nation
    for i in tl.static_range(0, 5):  # Will be compiled with actual num_nations
        # Use tl.where and tl.sum per nation
        nation_mask = valid & (nation_id == i)
        contrib = tl.where(nation_mask, revenue, 0.0)
        block_sum = tl.sum(contrib)
        tl.atomic_add(out_ptr + i, block_sum)
    # Handle up to 25 nations max (TPC-H has 25)
    for i in tl.static_range(5, 25):
        nation_mask = valid & (nation_id == i)
        contrib = tl.where(nation_mask, revenue, 0.0)
        block_sum = tl.sum(contrib)
        tl.atomic_add(out_ptr + i, block_sum)
