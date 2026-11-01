import triton
import triton.language as tl
import torch


@triton.jit
def kernel(keys, sorted_arr, out, n, s, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load keys for this block, with bounds checking
    mask = offsets < n
    key_vals = tl.load(keys + offsets, mask=mask, other=0)

    # Initialize result array - count of elements less than each key
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # For each element in the sorted array, count how many keys are greater than it
    # This is equivalent to lower_bound for all keys at once
    for i in tl.range(0, s):
        sorted_val = tl.load(sorted_arr + i)
        # For each key, check if sorted_arr[i] < key_val
        is_less = sorted_val < key_vals
        result = tl.where(is_less, result + 1, result)

    # Store results
    tl.store(out + offsets, result, mask=mask)


TYPES = {
    "keys": "*i64",
    "sorted_arr": "*i64",
    "out": "*i32",
    "n": "i64",
    "s": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(keys, sorted_arr, out, n, s):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        keys: torch.Tensor of shape (n,), dtype torch.int64
        sorted_arr: torch.Tensor of shape (s,), dtype torch.int64, sorted ascending
        out: torch.Tensor of shape (n,), dtype torch.int32, to be filled
        n: int, number of keys
        s: int, size of sorted array
    """
    assert keys.is_cuda and sorted_arr.is_cuda and out.is_cuda, "Tensors must be on CUDA"
    assert keys.dtype == torch.int64, "keys must be int64"
    assert sorted_arr.dtype == torch.int64, "sorted_arr must be int64"
    assert out.dtype == torch.int32, "out must be int32"

    block_size = 256
    grid = ((n + block_size - 1) // block_size,)
    kernel[grid](keys, sorted_arr, out, n, s, block_size, num_warps=NUM_WARPS)
