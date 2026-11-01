import triton
import triton.language as tl
import torch


@triton.jit
def kernel(group, value, best, n, capacity, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load group ids and values
    mask = offsets < n
    groups = tl.load(group + offsets, mask=mask, other=0)
    values = tl.load(value + offsets, mask=mask, other=-(2 ** 63))

    # Atomically update best array with maximum values per group
    # Use vectorized atomic operations
    tl.atomic_max(best + groups, values, mask=mask)


TYPES = {
    "group": "*i32",
    "value": "*i64",
    "best": "*i64",
    "n": "i64",
    "capacity": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(group, value, best, n, capacity):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        group: torch.Tensor of shape (n,), dtype torch.int32
        value: torch.Tensor of shape (n,), dtype torch.int64
        best: torch.Tensor of shape (capacity,), dtype torch.int64, pre-filled with min int64
        n: int, number of rows
        capacity: int, number of groups
    """
    assert group.is_cuda and value.is_cuda and best.is_cuda, "Tensors must be on CUDA"
    assert group.dtype == torch.int32, "group must be int32"
    assert value.dtype == torch.int64, "value must be int64"
    assert best.dtype == torch.int64, "best must be int64"

    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](group, value, best, n, capacity, block_size, num_warps=NUM_WARPS)
