import triton
import triton.language as tl

TYPES = {
    "keys_ptr": "*i64",
    "flags_ptr": "*i8",
    "n_keys": "i32",
    "n_flags": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    keys_ptr,
    flags_ptr,
    n_keys,
    n_flags,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_keys
    key = tl.load(keys_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key < n_flags)
    safe_key = tl.where(valid, key, 0)
    tl.store(flags_ptr + safe_key, 1, mask=valid)
