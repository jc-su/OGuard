import triton
import triton.language as tl

TYPES = {
    "sorted_keys_ptr": "*i64",
    "out_counts_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 4


@triton.jit
def kernel(sorted_keys_ptr, out_counts_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # FACTOR > max supplier id (supplier rows = 100k). 100001 is safe.
    GROUP_FACTOR = 100001
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load current composite key
    key = tl.load(sorted_keys_ptr + offs, mask=mask, other=0)

    # Load previous key (if any) – needed to detect a new unique pair
    prev_offs = offs - 1
    prev_mask = (offs > 0) & mask
    prev_key = tl.load(sorted_keys_ptr + prev_offs, mask=prev_mask, other=0)

    # Is this the very first element? If so, it's a new pair.
    is_first = offs == 0
    # Keep this element if it's the first or the key differs from the previous one.
    keep = is_first | (key != prev_key)

    # Decode group id from composite key (high bits)
    group_id = key // GROUP_FACTOR

    # Atomically increment the counter for this group if the pair is unique.
    tl.atomic_add(out_counts_ptr + group_id, tl.cast(keep, tl.int64))
