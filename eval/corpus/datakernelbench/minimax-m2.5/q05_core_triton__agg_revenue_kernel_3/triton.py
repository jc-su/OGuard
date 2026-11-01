import triton
import triton.language as tl

TYPES = {
    "nation_id_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 2048}

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
    """Fused kernel: compute revenue = price * (1 - discount) and atomic‑add into per‑nation buckets."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load row data (int64 nation_id, float64 price, float64 discount)
    nation_id = tl.load(nation_id_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Compute revenue for valid rows
    revenue = price * (1.0 - discount)

    # Atomic add to the per‑nation accumulator; zero contribution for masked rows
    tl.atomic_add(out_ptr + nation_id, tl.where(mask, revenue, 0.0))
