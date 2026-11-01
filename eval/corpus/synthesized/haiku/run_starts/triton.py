import triton
import triton.language as tl
import torch


@triton.jit
def kernel(values, flags, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load values for this block, with bounds checking
    mask = offsets < n
    vals = tl.load(values + offsets, mask=mask, other=0)

    # Compute flags: 1 if i=0 or values[i] != values[i-1]
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.int8)

    # First element of first block gets 1
    result = tl.where(offsets == 0, 1, result)

    # For other elements, check if different from previous
    # For elements within the block (not at block_start), we can check within block
    prev_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    prev_vals = tl.where(offsets > 0,
                         tl.load(values + offsets - 1, mask=(offsets > 0) & mask, other=0),
                         prev_vals)

    # Mark as 1 if different from previous (except for first element which is already marked)
    result = tl.where((offsets > 0) & (vals != prev_vals), 1, result)

    # Store results
    tl.store(flags + offsets, result, mask=mask)


TYPES = {
    "values": "*i64",
    "flags": "*i8",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(values, flags, n):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        values: torch.Tensor of shape (n,), dtype torch.int64
        flags: torch.Tensor of shape (n,), dtype torch.int8, to be filled
        n: int, length of arrays
    """
    assert values.is_cuda and flags.is_cuda, "Tensors must be on CUDA"
    assert values.dtype == torch.int64, "values must be int64"
    assert flags.dtype == torch.int8, "flags must be int8"

    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](values, flags, n, block_size, num_warps=NUM_WARPS)
