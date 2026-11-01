# This module expects 'triton', 'tl', and 'torch' to be injected into its namespace

@triton.jit
def kernel(lengths_ptr, offsets_ptr, n):
    # Each program processes one offset
    idx = tl.program_id(0)

    if idx < n + 1:
        if idx == 0:
            # First offset is always 0
            tl.store(offsets_ptr, 0)
        else:
            # Compute offset as sum of previous lengths
            # Sum lengths[0..idx-1]
            total = 0
            for i in range(idx):
                length = tl.load(lengths_ptr + i)
                total += length

            tl.store(offsets_ptr + idx, total)


TYPES = {
    "lengths_ptr": "*i32",
    "offsets_ptr": "*i32",
    "n": "i64",
}

CONSTANTS = {}

NUM_WARPS = 1


def run(lengths: torch.Tensor, offsets: torch.Tensor, n: int) -> None:
    """
    Launch the prefix_offsets kernel.

    Args:
        lengths: element lengths (int32, n)
        offsets: output offsets (int32, n + 1)
        n: number of elements
    """
    grid = (n + 1,)
    kernel[grid](lengths, offsets, n, num_warps=NUM_WARPS)
