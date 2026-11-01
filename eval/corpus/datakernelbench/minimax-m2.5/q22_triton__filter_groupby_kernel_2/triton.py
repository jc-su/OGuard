import triton
import triton.language as tl

TYPES = {
    "acctbal_ptr": "*fp64",
    "cntrycode_ptr": "*i32",
    "in_mask_ptr": "*i32",
    "has_order_ptr": "*i32",
    "avg_ptr": "*fp64",
    "numcust_ptr": "*i64",
    "totacctbal_ptr": "*fp64",
    "n_elements": "i32"
}

CONSTANTS = {"BLOCK_SIZE": 1024}

NUM_WARPS = 8


@triton.jit
def kernel(
    acctbal_ptr, cntrycode_ptr, in_mask_ptr, has_order_ptr,
    avg_ptr, numcust_ptr, totacctbal_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load avg from scalar pointer
    avg_val = tl.load(avg_ptr)
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load masks
    in_mask = tl.load(in_mask_ptr + offs, mask=mask, other=0)
    has_order = tl.load(has_order_ptr + offs, mask=mask, other=1)  # 1 = has order
    
    # Load acctbal
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    
    # Load cntrycode (int32)
    cntrycode = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    
    # Full predicate: in_mask AND acctbal > avg AND NOT has_order
    # has_order == 0 means no order (our mask: 1=has order, 0=no order)
    no_order = (has_order == 0)
    keep = (in_mask != 0) & (acctbal > avg_val) & no_order
    
    # Get group index (cntrycode - 13) / 2 mapping to 0-6
    # Country codes: 13, 17, 18, 23, 29, 30, 31
    # We need to map to 0..6
    # Manual mapping - avoids runtime conditionals
    # 13->0, 17->1, 18->2, 23->3, 29->4, 30->5, 31->6
    gid = -1
    gid = tl.where(cntrycode == 13, 0, gid)
    gid = tl.where(cntrycode == 17, 1, gid)
    gid = tl.where(cntrycode == 18, 2, gid)
    gid = tl.where(cntrycode == 23, 3, gid)
    gid = tl.where(cntrycode == 29, 4, gid)
    gid = tl.where(cntrycode == 30, 5, gid)
    gid = tl.where(cntrycode == 31, 6, gid)
    
    valid_gid = (gid >= 0) & keep
    
    # Count and sum per group
    for i in tl.static_range(7):
        g_keep = valid_gid & (gid == i)
        g_count = tl.sum(tl.where(g_keep, 1, 0))
        g_sum = tl.sum(tl.where(g_keep, acctbal, 0.0))
        tl.atomic_add(numcust_ptr + i, g_count)
        tl.atomic_add(totacctbal_ptr + i, g_sum)
