import triton
import triton.language as tl

TYPES = {
    "lk_ptr": "*i64",
    "ep_ptr": "*fp64",
    "disc_ptr": "*fp64",
    "sd_ptr": "*i32",
    "od_ptr": "*i32",
    "sp_ptr": "*i32",
    "out_keys_ptr": "*i64",
    "out_rev_ptr": "*fp64",
    "block_offsets_ptr": "*i32",
    "sd_lo": "i32",
    "od_hi": "i32",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 4096
}

NUM_WARPS = 8


@triton.jit
def kernel(
    lk_ptr, ep_ptr, disc_ptr, sd_ptr, od_ptr, sp_ptr,
    out_keys_ptr, out_rev_ptr,
    block_offsets_ptr,
    sd_lo, od_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    sd = tl.load(sd_ptr + offs, mask=mask, other=sd_lo - 1)
    od = tl.load(od_ptr + offs, mask=mask, other=od_hi + 1)
    keep = (sd > sd_lo) & (od < od_hi) & mask
    
    # Load data needed for valid rows
    lk = tl.load(lk_ptr + offs, mask=mask, other=0)
    ep = tl.load(ep_ptr + offs, mask=mask, other=0.0)
    disc = tl.load(disc_ptr + offs, mask=mask, other=0.0)
    sp = tl.load(sp_ptr + offs, mask=mask, other=0)
    
    # Calculate revenue
    rev = ep * (1.0 - disc)
    
    # Pack Key: (lk << 32) | (od << 16) | sp
    # Cast to int64 for shifting
    lk_i64 = lk.to(tl.int64)
    od_i64 = od.to(tl.int64)
    sp_i64 = sp.to(tl.int64)
    
    packed_key = (lk_i64 << 32) | (od_i64 << 16) | sp_i64
    
    # Compaction using cumsum
    local_idx = tl.cumsum(keep.to(tl.int32)) - 1
    
    # Global index
    base_offset = tl.load(block_offsets_ptr + pid)
    global_idx = base_offset + local_idx
    
    # Store only if keep
    tl.store(out_keys_ptr + global_idx, packed_key, mask=keep)
    tl.store(out_rev_ptr + global_idx, rev, mask=keep)
