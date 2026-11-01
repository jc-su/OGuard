import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "discount_ptr": "*fp64",
    "inv_ptr": "*i64",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 4096}
NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr,       # float64 *
    discount_ptr,    # float64 *
    inv_ptr,         # int64 *
    out_ptr,         # float64 *
    n_elements,      # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (masked)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    inv = tl.load(inv_ptr + offs, mask=mask, other=0)

    # Compute revenue per row
    revenue = price * (1.0 - discount)

    # Atomic add into bucket identified by inverse index
    tl.atomic_add(out_ptr + inv, revenue, mask=mask)
