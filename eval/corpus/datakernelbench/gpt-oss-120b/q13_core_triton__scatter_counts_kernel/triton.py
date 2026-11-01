import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "out_ptr": "*i32",
    "n_elements": "i32",
    "max_key": "i32",
}
CONSTANTS = {"BLOCK_SIZE": 8192}
NUM_WARPS = 4


@triton.jit
def kernel(
    custkey_ptr,   # int64*  (orders' o_custkey)
    out_ptr,       # int32*  (per‑customer counters)
    n_elements,    # total number of orders
    max_key,       # highest valid customer key (num_customers)
    BLOCK_SIZE: tl.constexpr = 8192,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offs < n_elements
    # Load a customer key; for OOB lanes load 0 (invalid)
    key = tl.load(custkey_ptr + offs, mask=mask, other=0)

    # Valid keys are in the range (0, max_key]
    valid = mask & (key > 0) & (key <= max_key)

    # Increment the bucket by 1 for valid rows; 0 otherwise.
    incr = tl.where(valid, 1, 0)
    tl.atomic_add(out_ptr + key, incr)
