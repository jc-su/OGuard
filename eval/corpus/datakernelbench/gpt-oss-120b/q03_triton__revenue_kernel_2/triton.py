import triton
import triton.language as tl

TYPES = {
    "price_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "out_rev_ptr": "*fp64",
    "n_elements": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 2048}
NUM_WARPS = 4


@triton.jit
def kernel(
    price_ptr,            # float64 *
    disc_ptr,             # float64 *
    out_rev_ptr,          # float64 *
    n_elements,           # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)

    # 1.0 will be promoted to fp64 because price and disc are fp64
    rev = price * (1.0 - disc)

    tl.store(out_rev_ptr + offs, rev, mask=mask)
