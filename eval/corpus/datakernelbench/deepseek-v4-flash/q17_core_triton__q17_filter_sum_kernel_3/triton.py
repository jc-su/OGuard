import triton
import triton.language as tl

TYPES = {
    "l_price_ptr": "*fp64",
    "keep_mask_ptr": "*i1",
    "out_ptr": "*fp64",
    "n_elements": "i32",
}

CONSTANTS = {
    "BLOCK_SIZE": 8192,
}

NUM_WARPS = 4


@triton.jit
def kernel(
    l_price_ptr, keep_mask_ptr,
    out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keep = tl.load(keep_mask_ptr + offs, mask=mask, other=0)
    price = tl.load(l_price_ptr + offs, mask=keep.to(tl.int1), other=0.0)

    block_sum = tl.sum(price)
    tl.atomic_add(out_ptr, block_sum)
