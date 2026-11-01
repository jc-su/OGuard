import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 1024
}

NUM_WARPS = 4


@triton.jit
def kernel(
    nation_id_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q5 revenue aggregation.
    Single pass: load nation_id, price, discount; compute revenue; atomic_add to bucket.
    """
    pid = tl.program_id(0)
    base_offs = pid * BLOCK_SIZE
    
    for i in tl.static_range(BLOCK_SIZE):
        offs = base_offs + i
        if offs < n_elements:
            nation_id = tl.load(nation_id_ptr + offs)
            price = tl.load(price_ptr + offs)
            discount = tl.load(discount_ptr + offs)
            revenue = price * (1.0 - discount)
            tl.atomic_add(out_ptr + nation_id, revenue)
