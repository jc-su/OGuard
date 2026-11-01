import triton
import triton.language as tl
import torch


@triton.jit
def kernel(offsets, chars, out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    # Each thread block processes one string
    if pid >= n:
        return

    # Get string boundaries
    start_idx = tl.load(offsets + pid)
    end_idx = tl.load(offsets + pid + 1)
    str_len = end_idx - start_idx

    # Reverse the string: copy chars[start_idx:end_idx] to out in reverse order
    for i in tl.range(0, str_len, BLOCK_SIZE):
        # How many elements in this chunk
        chunk_size = tl.minimum(BLOCK_SIZE, str_len - i)

        # For each position in the chunk, copy from the reversed position
        indices = tl.arange(0, BLOCK_SIZE)
        mask = indices < chunk_size

        # Load characters from the original position (counting backwards from end)
        src_pos = start_idx + str_len - i - indices - 1
        chars_val = tl.load(chars + src_pos, mask=mask, other=0)

        # Store to output in forward order
        out_pos = start_idx + i + indices
        tl.store(out + out_pos, chars_val, mask=mask)


TYPES = {
    "offsets": "*i32",
    "chars": "*u8",
    "out": "*u8",
    "n": "i64",
}

CONSTANTS = {
    "BLOCK_SIZE": 256,
}

NUM_WARPS = 4


def run(offsets, chars, out, n):
    """
    Launch the kernel on PyTorch CUDA tensors.

    Args:
        offsets: torch.Tensor of shape (n + 1,), dtype torch.int32
        chars: torch.Tensor of shape (offsets[n],), dtype torch.uint8
        out: torch.Tensor of shape (offsets[n],), dtype torch.uint8, to be filled
        n: int, number of strings
    """
    assert offsets.is_cuda and chars.is_cuda and out.is_cuda, "Tensors must be on CUDA"
    assert offsets.dtype == torch.int32, "offsets must be int32"
    assert chars.dtype == torch.uint8, "chars must be uint8"
    assert out.dtype == torch.uint8, "out must be uint8"

    block_size = 256
    grid = (n,)
    kernel[grid](offsets, chars, out, n, block_size, num_warps=NUM_WARPS)
