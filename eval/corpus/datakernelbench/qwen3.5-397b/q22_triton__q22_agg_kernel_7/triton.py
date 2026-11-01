import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i1",
    "has_order_ptr": "*i1",
    "avg_thresh_ptr": "*fp64",
    "out_sum_ptr": "*fp64",
    "out_count_ptr": "*i64",
    "n_elements": "i32"
}

CONSTANTS = {
    "num_groups": 7,
    "BLOCK_SIZE": 8192
}

NUM_WARPS = 4


@triton.jit
def kernel(
    acctbal_ptr,
    cntrycode_ptr,
    in_mask_ptr,
    has_order_ptr,
    avg_thresh_ptr,
    out_sum_ptr,
    out_count_ptr,
    n_elements,
    num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Final aggregation: group by cntrycode, sum acctbal and count.
    Filters: in_mask AND acctbal > avg_thresh AND NOT has_order.
    num_groups = 7 (fixed country codes: 13, 17, 18, 23, 29, 30, 31)
    Group indices: 13->0, 17->1, 18->2, 23->3, 29->4, 30->5, 31->6
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load all filter columns
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=False)
    has_order = tl.load(has_order_ptr + offs, mask=mask, other=True)
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    avg_thresh = tl.load(avg_thresh_ptr)
    
    # Filter: in_set AND acctbal > avg_thresh AND NOT has_order
    keep = mask & in_set & (acctbal > avg_thresh) & (~has_order)
    
    # Map cntrycode to group index (13,17,18,23,29,30,31 -> 0,1,2,3,4,5,6)
    # Use tl.where chain for mapping
    code_13 = (cntrycode == 13)
    code_17 = (cntrycode == 17)
    code_18 = (cntrycode == 18)
    code_23 = (cntrycode == 23)
    code_29 = (cntrycode == 29)
    code_30 = (cntrycode == 30)
    code_31 = (cntrycode == 31)
    
    group_idx = (
        tl.where(code_13, 0,
        tl.where(code_17, 1,
        tl.where(code_18, 2,
        tl.where(code_23, 3,
        tl.where(code_29, 4,
        tl.where(code_30, 5,
        tl.where(code_31, 6, 7))))))))
    
    # Load acctbal only for kept rows
    acctbal_filtered = tl.load(acctbal_ptr + offs, mask=keep, other=0.0)
    
    # Accumulate per-group using atomics
    # For each group, atomic_add to the corresponding position
    for g in tl.static_range(num_groups):
        is_group = (group_idx == g) & keep
        group_sum = tl.sum(tl.where(is_group, acctbal_filtered, 0.0))
        group_count = tl.sum(tl.where(is_group, 1.0, 0.0))
        tl.atomic_add(out_sum_ptr + g, group_sum)
        tl.atomic_add(out_count_ptr + g, group_count)
