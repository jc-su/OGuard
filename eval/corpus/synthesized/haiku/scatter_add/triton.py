import triton
import triton.language as tl
import torch


@triton.jit
def kernel(idx, values, out, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load indices and values for this block
    mask = offsets < m
    indices = tl.load(idx + offsets, mask=mask, other=0)
    vals = tl.load(values + offsets, mask=mask, other=0.0)

    # Atomically add values to output array based on indices
    tl.atomic_add(out + indices, vals, mask=mask)


TYPES = {
    "idx": "*i32",
    "values": "*fp64",
    "out": "*fp64",
    "m": "i64",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(idx, values, out, m, n):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        idx: torch.Tensor of shape (m,), dtype torch.int32, row indices
        values: torch.Tensor of shape (m,), dtype torch.float64
        out: torch.Tensor of shape (n,), dtype torch.float64, zero on entry
        m: int, number of elements to scatter
        n: int, size of output array
    """
    assert idx.is_cuda and values.is_cuda and out.is_cuda, "Tensors must be on CUDA"
    assert idx.dtype == torch.int32, "idx must be int32"
    assert values.dtype == torch.float64, "values must be float64"
    assert out.dtype == torch.float64, "out must be float64"

    block_size = 256
    grid = ((m + block_size - 1) // block_size,)
    kernel[grid](idx, values, out, m, n, block_size, num_warps=NUM_WARPS)
