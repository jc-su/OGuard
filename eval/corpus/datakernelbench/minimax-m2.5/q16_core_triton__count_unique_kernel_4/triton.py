import triton
import triton.language as tl

TYPES = {
    "unique_group_ptr": "*i64",
    "output_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 512}

NUM_WARPS = 8


@triton.jit
def kernel(
    unique_group_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one unique (group, supp) pair
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load groups for valid indices only
    groups = tl.load(unique_group_ptr + idx, mask=mask, other=-1)
    
    # For each valid element, add 1 to its group's count
    for i in range(BLOCK_SIZE):
        local_idx = pid * BLOCK_SIZE + i
        if local_idx < n_elements:
            g = tl.load(unique_group_ptr + local_idx)
            tl.atomic_add(output_ptr + g, 1)
