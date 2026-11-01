import triton
import triton.language as tl

TYPES = {
    "partials_ptr": "*fp64",
    "out_ptr": "*fp64",
    "partial_stride": "i32",
    "n_blocks": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 8192}

NUM_WARPS = 4


@triton.jit
def kernel(
    partials_ptr,
    out_ptr,
    partial_stride,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_blocks

    total_vals = tl.load(partials_ptr + offs, mask=mask, other=0.0)
    promo_vals = tl.load(partials_ptr + partial_stride + offs, mask=mask, other=0.0)

    total = tl.sum(total_vals)
    promo = tl.sum(promo_vals)
    tl.store(out_ptr, (100.0 * promo) / total)
