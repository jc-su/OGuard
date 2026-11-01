import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
    "num_nations": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(
    nation_id_ptr,
    price_ptr,
    discount_ptr,
    out_ptr,
    n_elements,
    num_nations,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused filter → project → aggregate kernel for Q5.
    
    Each thread loads a row, computes revenue = price * (1 - discount),
    and atomically adds to the per‑nation accumulator.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load row data (masked loads ensure OOB lanes get neutral values)
    nation = tl.load(nation_id_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Project: compute revenue for kept rows
    revenue = price * (1.0 - discount)

    # Global per‑nation aggregation via atomic add
    tl.atomic_add(out_ptr + nation, revenue)
