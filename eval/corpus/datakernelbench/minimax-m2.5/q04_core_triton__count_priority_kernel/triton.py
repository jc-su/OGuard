import triton
import triton.language as tl

TYPES = {
    "priority_ptr": "*i32",
    "out_ptr": "*i64",
    "n_elements": "i32",
    "num_priorities": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(priority_ptr, out_ptr, n_elements, num_priorities, BLOCK_SIZE: tl.constexpr):
    """Count occurrences of each priority code in a block and atomically add to global counts."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Use num_priorities as sentinel for masked-out positions so they don't
    # get counted as priority 0 (which could be a valid priority code).
    priorities = tl.load(priority_ptr + offs, mask=mask, other=num_priorities)

    # Per‑priority block reduction: for each priority value compute the
    # number of matching elements in this block and atomically add to the
    # global histogram.
    for p in range(num_priorities):
        block_count = tl.sum(tl.where(priorities == p, 1, 0))
        tl.atomic_add(out_ptr + p, block_count)
