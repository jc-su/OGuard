import triton
import triton.language as tl

TYPES = {
    "nation_ptr": "*i64",
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    nation_ptr,          # int64 *
    price_ptr,           # float64 *
    discount_ptr,        # float64 *
    out_ptr,             # float64 * (size = num_nations)
    n_elements,          # int64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load inputs; masked lanes get benign values
    nid = tl.load(nation_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # Zero out contributions from masked lanes to avoid spurious atomics
    revenue = tl.where(mask, revenue, 0.0)
    nid = tl.where(mask, nid, 0)

    # Atomic add per nation
    tl.atomic_add(out_ptr + nid, revenue)
