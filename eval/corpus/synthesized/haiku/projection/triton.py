import torch
import triton
import triton.language as tl

TYPES = {
    "price": "*fp64",
    "discount": "*fp64",
    "out": "*fp64",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4

@triton.jit
def kernel(price, discount, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    price_vals = tl.load(price + offsets, mask=mask, other=0.0)
    discount_vals = tl.load(discount + offsets, mask=mask, other=0.0)

    result = price_vals * (1.0 - discount_vals)

    tl.store(out + offsets, result, mask=mask)

def run(price: torch.Tensor, discount: torch.Tensor, out: torch.Tensor) -> None:
    n = price.numel()
    BLOCK_SIZE = CONSTANTS["BLOCK_SIZE"]
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel[grid](price, discount, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)
