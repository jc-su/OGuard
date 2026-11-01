import triton
import triton.language as tl

TYPES = {
    "custkey_ptr": "*i64",
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i1",
    "orders_ptr": "*i64",
    "n_orders": "i32",
    "avg_val": "fp64",
    "code_map_ptr": "*i32",
    "group_counts_ptr": "*i64",
    "group_sums_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {
    "BLOCK_SIZE": 1024
}

NUM_WARPS = 4


@triton.jit
def kernel(
    custkey_ptr, acctbal_ptr, cntrycode_ptr, in_mask_ptr,
    orders_ptr, n_orders,
    avg_val,
    code_map_ptr,
    group_counts_ptr, group_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    ckey = tl.load(custkey_ptr + offs, mask=mask, other=0)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    code = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_m = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    
    # Filter 1 & 2: in_mask & bal > avg_val
    keep = in_m & (bal > avg_val)
    
    # Filter 3: NOT EXISTS (binary search in sorted orders)
    low = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    high = tl.full([BLOCK_SIZE], n_orders, dtype=tl.int64)
    
    # 22 iterations covers up to 4M elements (log2(1.5M) ~ 21)
    for _ in tl.static_range(22):
        mid = (low + high) // 2
        o_mask = mid < n_orders
        # Use a value that won't match valid custkey (>=1) for OOB
        o_val = tl.load(orders_ptr + mid, mask=o_mask, other=-1)
        
        # Update bounds
        # if o_val < ckey: low = mid + 1
        # else: high = mid
        # Treat OOB as +INF (so high = mid)
        is_less = (o_val < ckey) & o_mask
        is_ge = (o_val >= ckey) | (~o_mask)
        
        low = tl.where(is_less, mid + 1, low)
        high = tl.where(is_ge, mid, high)
        
    # Check if found
    check_mask = low < n_orders
    check_val = tl.load(orders_ptr + low, mask=check_mask, other=-1)
    is_in_orders = check_mask & (check_val == ckey)
    
    keep = keep & (~is_in_orders)
    
    # Map code to bin (0..6) using code_map
    # code is int32 (0..99). code_map is size 100.
    bin_idx = tl.load(code_map_ptr + code, mask=mask, other=-1)
    
    final_keep = keep & (bin_idx != -1) & mask
    
    # Safe bin index for atomic add (avoid OOB if bin_idx == -1)
    safe_bin = tl.where(final_keep, bin_idx, 0)
    
    v_count = tl.where(final_keep, 1, 0).to(tl.int64)
    v_sum = tl.where(final_keep, bal, 0.0)
    
    tl.atomic_add(group_counts_ptr + safe_bin, v_count)
    tl.atomic_add(group_sums_ptr + safe_bin, v_sum)
