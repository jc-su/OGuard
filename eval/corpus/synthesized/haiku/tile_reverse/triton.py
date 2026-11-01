import triton
import triton.language as tl
import torch


@triton.jit
def kernel(values, out, n, TILE_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load values for this block
    mask = offsets < n
    vals = tl.load(values + offsets, mask=mask, other=0)

    # For each position, compute its reversed position within its tile
    # Each tile is TILE_SIZE elements
    tile_indices = offsets // TILE_SIZE
    within_tile = offsets % TILE_SIZE

    # The reversed position within the same tile
    tile_start = tile_indices * TILE_SIZE
    tile_end_or_n = tl.minimum((tile_indices + 1) * TILE_SIZE, n)
    tile_len = tile_end_or_n - tile_start
    reversed_within_tile = tile_len - 1 - within_tile

    # The reversed index
    reversed_indices = tile_start + reversed_within_tile

    # Load the value at the reversed position (need to be careful about bounds)
    reversed_mask = (reversed_indices < n) & mask
    reversed_vals = tl.load(values + reversed_indices, mask=reversed_mask, other=0)

    # Store results
    tl.store(out + offsets, reversed_vals, mask=mask)


TYPES = {
    "values": "*i32",
    "out": "*i32",
    "n": "i64",
}

CONSTANTS = {
    "TILE_SIZE": 1024,
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(values, out, n):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        values: torch.Tensor of shape (n,), dtype torch.int32
        out: torch.Tensor of shape (n,), dtype torch.int32, to be filled
        n: int, number of elements
    """
    assert values.is_cuda and out.is_cuda, "Tensors must be on CUDA"
    assert values.dtype == torch.int32, "values must be int32"
    assert out.dtype == torch.int32, "out must be int32"

    tile_size = 1024
    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](values, out, n, tile_size, block_size, num_warps=NUM_WARPS)
