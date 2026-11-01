import triton
import triton.language as tl

TYPES = {
    "o_okey_ptr": "*i64",
    "o_odate_ptr": "*i32",
    "o_priority_ptr": "*i32",
    "exists_map_ptr": "*i32",
    "counts_out_ptr": "*i64",
    "date_lo": "i32",
    "date_hi": "i32",
    "max_key": "i32",
    "n_elements": "i32",
}

CONSTANTS = {"NUM_PRIORITIES": 5, "BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    o_okey_ptr, o_odate_ptr, o_priority_ptr, exists_map_ptr, counts_out_ptr,
    date_lo, date_hi, max_key, n_elements,
    NUM_PRIORITIES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    odate = tl.load(o_odate_ptr + offs, mask=mask, other=0)
    date_keep = mask & (odate >= date_lo) & (odate < date_hi)
    
    okey = tl.load(o_okey_ptr + offs, mask=date_keep, other=0)
    okey_valid = date_keep & (okey >= 0) & (okey <= max_key)
    
    # Check exists_map
    exists = tl.load(exists_map_ptr + okey, mask=okey_valid, other=0)
    final_keep = okey_valid & (exists == 1)
    
    prio = tl.load(o_priority_ptr + offs, mask=final_keep, other=0)
    
    # Accumulate locally and atomic add for small fixed group sizes
    for i in tl.static_range(NUM_PRIORITIES):
        p_mask = final_keep & (prio == i)
        count_i = tl.sum(tl.where(p_mask, 1, 0))
        if count_i > 0:
            tl.atomic_add(counts_out_ptr + i, count_i.to(tl.int64))
