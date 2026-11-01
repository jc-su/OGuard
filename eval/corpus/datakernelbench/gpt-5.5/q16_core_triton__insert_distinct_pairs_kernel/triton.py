import triton
import triton.language as tl

TYPES = {
    "group_ptr": "*i64",
    "supp_ptr": "*i64",
    "table_ptr": "*i64",
    "n_elements": "i32",
    "table_mask": "i32",
}

CONSTANTS = {"BLOCK_SIZE": 256}

NUM_WARPS = 4


@triton.jit
def kernel(
    group_ptr,
    supp_ptr,
    table_ptr,
    n_elements,
    table_mask,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < n_elements

    first_gid = tl.load(group_ptr + 0)
    first_supp = tl.load(supp_ptr + 0)

    gid_loaded = tl.load(group_ptr + offs, mask=valid, other=0)
    supp_loaded = tl.load(supp_ptr + offs, mask=valid, other=0)

    gid = tl.where(valid, gid_loaded, first_gid)
    supp = tl.where(valid, supp_loaded, first_supp)

    # TPC-H supplier keys and generated group ids are non-negative and fit in 32 bits.
    # Store one 64-bit composite key in an open-addressed hash set.
    key = gid * 4294967296 + supp

    h = (gid * 1315423911) ^ (supp * 2654435761)
    h = h ^ (h >> 16)
    slot = h & table_mask

    empty = tl.full((BLOCK_SIZE,), -1, tl.int64)
    done = tl.full((BLOCK_SIZE,), False, tl.int1)

    while tl.sum(tl.where(done, 0, 1)) > 0:
        old = tl.atomic_cas(table_ptr + slot, empty, key)
        hit = (old == empty) | (old == key)
        done = done | hit
        slot = tl.where(done, slot, (slot + 1) & table_mask)
