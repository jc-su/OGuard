import triton
import triton.language as tl
import torch


@triton.jit
def kernel(values, mask, result, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load values and mask
    mask_cond = offsets < n
    vals = tl.load(values + offsets, mask=mask_cond, other=0.0)
    mask_vals = tl.load(mask + offsets, mask=mask_cond, other=0)

    # Compute sum of values where mask != 0
    # Create a mask for non-zero elements
    is_nonzero = mask_vals != 0
    masked_vals = tl.where(is_nonzero, vals, 0.0)

    # Sum the masked values
    block_sum = tl.sum(masked_vals)

    # Atomically add to result[0]
    tl.atomic_add(result, block_sum)


TYPES = {
    "values": "*fp64",
    "mask": "*i8",
    "result": "*fp64",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(values, mask, result, n):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        values: torch.Tensor of shape (n,), dtype torch.float64
        mask: torch.Tensor of shape (n,), dtype torch.int8
        result: torch.Tensor of shape (1,), dtype torch.float64, zero on entry
        n: int, number of rows
    """
    assert values.is_cuda and mask.is_cuda and result.is_cuda, "Tensors must be on CUDA"
    assert values.dtype == torch.float64, "values must be float64"
    assert mask.dtype == torch.int8, "mask must be int8"
    assert result.dtype == torch.float64, "result must be float64"

    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](values, mask, result, n, block_size, num_warps=NUM_WARPS)
