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
NUM_WARPS = 8


@triton.jit
def kernel(
    price_ptr,        # *float64
    discount_ptr,     # *float64
    inv_ptr,          # *int64   (inverse index from torch.unique)
    out_ptr,          # *float64 (output revenue per unique key)
    n_elements,       # i64
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)          # [BLOCK_SIZE]
    mask = offs < n_elements

    # Load the inverse bucket index and the price/discount columns
    inv = tl.load(inv_ptr + offs, mask=mask, other=0)          # int64
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)   # float64
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)  # float64

    # Compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # Atomically accumulate revenue into the appropriate bucket
    tl.atomic_add(out_ptr + inv, revenue, mask=mask)
