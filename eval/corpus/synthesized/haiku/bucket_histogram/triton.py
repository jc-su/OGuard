import triton
import triton.language as tl
import torch


@triton.jit
def kernel(values, bounds, counts, n, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load values for this block
    mask = offsets < n
    vals = tl.load(values + offsets, mask=mask, other=-1e10)

    # For each value, find which bucket it belongs to
    # Bucket = count of bounds that are <= value
    buckets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # Count bounds for each value - use a simpler method
    # For each bound, count how many values are >= that bound
    for i in tl.range(0, b):
        bound_val = tl.load(bounds + i)
        # If value >= bound, increment bucket
        buckets = tl.where(vals >= bound_val, buckets + 1, buckets)

    # Now atomically update counts
    # Note: We must ensure bucket indices are within bounds
    safe_buckets = tl.minimum(buckets, b)  # Clamp to valid range

    tl.atomic_add(counts + safe_buckets, 1, mask=mask)


TYPES = {
    "values": "*fp64",
    "bounds": "*fp64",
    "counts": "*i32",
    "n": "i64",
    "b": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(values, bounds, counts, n, b):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        values: torch.Tensor of shape (n,), dtype torch.float64
        bounds: torch.Tensor of shape (b,), dtype torch.float64, sorted ascending
        counts: torch.Tensor of shape (b + 1,), dtype torch.int32, zero on entry
        n: int, number of values
        b: int, number of bounds
    """
    assert values.is_cuda and bounds.is_cuda and counts.is_cuda, "Tensors must be on CUDA"
    assert values.dtype == torch.float64, "values must be float64"
    assert bounds.dtype == torch.float64, "bounds must be float64"
    assert counts.dtype == torch.int32, "counts must be int32"

    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](values, bounds, counts, n, b, block_size, num_warps=NUM_WARPS)
