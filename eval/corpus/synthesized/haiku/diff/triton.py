import triton
import triton.language as tl
import torch


@triton.jit
def kernel(values, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load values for this block and the previous block
    mask = offsets < n
    vals = tl.load(values + offsets, mask=mask, other=0)

    # For the first element, copy the value
    # For other elements, compute difference
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    # First element of the array gets the original value
    result = tl.where(offsets == 0, vals, result)

    # For other elements, get difference from previous element
    prev_vals = tl.load(values + offsets - 1, mask=(offsets > 0) & mask, other=0)
    result = tl.where((offsets > 0) & mask, vals - prev_vals, result)

    # Store results
    tl.store(out + offsets, result, mask=mask)


TYPES = {
    "values": "*i64",
    "out": "*i64",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(values, out, n):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        values: torch.Tensor of shape (n,), dtype torch.int64
        out: torch.Tensor of shape (n,), dtype torch.int64, to be filled
        n: int, number of elements
    """
    assert values.is_cuda and out.is_cuda, "Tensors must be on CUDA"
    assert values.dtype == torch.int64, "values must be int64"
    assert out.dtype == torch.int64, "out must be int64"

    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](values, out, n, block_size, num_warps=NUM_WARPS)
