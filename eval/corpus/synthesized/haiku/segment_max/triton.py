import triton
import triton.language as tl
import torch


@triton.jit
def kernel(offsets, values, maxes, g, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    # Each block processes one segment
    if pid >= g:
        return

    # Get segment boundaries
    offset_start = tl.load(offsets + pid)
    offset_end = tl.load(offsets + pid + 1)
    seg_len = offset_end - offset_start

    # Initialize with smallest int64
    max_val = -(2 ** 63)

    # Process values in this segment in chunks
    for i in tl.range(0, seg_len, BLOCK_SIZE):
        # How many elements in this chunk
        chunk_len = tl.minimum(BLOCK_SIZE, seg_len - i)

        # Load chunk of values
        chunk_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offsets < seg_len
        chunk_vals = tl.load(values + offset_start + chunk_offsets, mask=mask, other=-(2 ** 63))

        # Update max
        max_val = tl.maximum(max_val, tl.max(tl.where(mask, chunk_vals, -(2 ** 63))))

    # Store result
    tl.store(maxes + pid, max_val)


TYPES = {
    "offsets": "*i32",
    "values": "*i64",
    "maxes": "*i64",
    "g": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(offsets, values, maxes, g):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        offsets: torch.Tensor of shape (g + 1,), dtype torch.int32
        values: torch.Tensor of shape (n,), dtype torch.int64
        maxes: torch.Tensor of shape (g,), dtype torch.int64, to be filled
        g: int, number of segments
    """
    assert offsets.is_cuda and values.is_cuda and maxes.is_cuda, "Tensors must be on CUDA"
    assert offsets.dtype == torch.int32, "offsets must be int32"
    assert values.dtype == torch.int64, "values must be int64"
    assert maxes.dtype == torch.int64, "maxes must be int64"

    block_size = 256
    grid = (g,)
    kernel[grid](offsets, values, maxes, g, block_size, num_warps=NUM_WARPS)
