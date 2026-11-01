import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import triton.language as tl
import cudf
import pandas as pd

# Triton kernel source must be in a separate string to be written to disk for JIT compilation.
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_l1"],
    reset_to_zero=["counts_ptr"],
)
@triton.jit
def _q21_kernel(
    l1_orderkey_ptr,
    l1_supp_id_ptr,
    agg_orderkey_ptr,
    agg_distinct_ptr,
    agg_late_distinct_ptr,
    counts_ptr,
    n_l1,
    n_agg,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_l1

    # Load l1 data
    l1_ok = tl.load(l1_orderkey_ptr + offs, mask=mask, other=0)
    l1_sid = tl.load(l1_supp_id_ptr + offs, mask=mask, other=0)

    # Binary Search for each element in the block to find index in agg_orderkey
    # Since agg_orderkey is sorted, we find the position where l1_ok == agg_orderkey
    # We implement a scalar binary search per lane.
    # To avoid dynamic loop divergence overhead, we use a fixed iteration count sufficient for max orders (~15M -> 24 iters)
    # However, standard while loop is supported in Triton.
    
    # We need to compute idx for each lane.
    # Initialize low/high
    low = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    high = tl.full([BLOCK_SIZE], n_agg, dtype=tl.int32)
    idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # 25 iterations covers up to 33M elements (2^25)
    for _ in tl.static_range(25):
        mid = (low + high) // 2
        # Prevent OOB load if mid >= n_agg (should not happen if high starts at n_agg and logic is correct, 
        # but mid can be n_agg if low=n_agg, high=n_agg+1? No high starts at n_agg.
        # If low=n_agg-1, high=n_agg, mid=n_agg-1.
        # If low=n_agg, high=n_agg, loop condition low<high fails.
        # So mid is always < n_agg inside loop if low < high.
        # But tl.load needs a pointer. We clamp mid to n_agg-1 for safety if logic drifts, 
        # but strict binary search logic ensures mid < high <= n_agg.
        # Wait, if high=n_agg, mid can be n_agg-1.
        # We must ensure we don't load out of bounds.
        # Safe load:
        val = tl.load(agg_orderkey_ptr + mid, mask=mid < n_agg, other=0)
        
        # Update bounds
        # if val < target: low = mid + 1
        # else: high = mid
        # Use tl.where for vectorization
        cond = val < l1_ok
        low = tl.where(cond, mid + 1, low)
        high = tl.where(cond, high, mid)

    idx = low # This is the searchsorted result (left insertion point)

    # Check if found (idx < n_agg and agg_orderkey[idx] == l1_ok)
    # In our logic, l1_ok must exist in agg.
    found_mask = mask & (idx < n_agg)
    
    # Load aggregate stats
    agg_dist = tl.load(agg_distinct_ptr + idx, mask=found_mask, other=0)
    agg_late_dist = tl.load(agg_late_distinct_ptr + idx, mask=found_mask, other=0)

    # Conditions:
    # 1. distinct_supp >= 2 (exists other supplier)
    # 2. late_distinct_supp <= 1 (no other late supplier, since l1 is late)
    # Note: l1 is already filtered for late in host, so we only check order stats.
    cond1 = agg_dist >= 2
    cond2 = agg_late_dist <= 1
    
    do_count = found_mask & cond1 & cond2

    # Atomic add to counts
    # l1_sid is int32 index into counts
    tl.atomic_add(counts_ptr + l1_sid, 1, mask=do_count)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q21_triton_")
    path = os.path.join(tmpdir, "_q21_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q21_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(tables: dict[str, cudf.DataFrame], nation_name: str = "SAUDI ARABIA", orderstatus: str = "F") -> cudf.DataFrame:
    """
    SQL: TPC-H Q21 - Suppliers Who Kept Orders Waiting.
    Description: 
      For each supplier in the given nation, count orders of status 'F' where the
      supplier's lineitem was late, another supplier also shipped the order, and no
      other supplier was late on that order.
      
      Optimization: 
      1. cuDF performs joins and initial filtering (Nation, Supplier, Orders).
      2. cuDF computes order-level aggregates (distinct suppliers, late distinct suppliers).
      3. Triton kernel fuses the lookup of order aggregates, condition evaluation, 
         and per-supplier counting into a single GPU pass, avoiding intermediate 
         PyTorch tensor allocations for masks and indices.
      4. Binary search for orderkey lookup is performed inside the Triton kernel 
         to maintain fusion and reduce HBM traffic.
    Params:
      nation_name (str): nation name filter.
      orderstatus (str): order status filter.
    """
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_nation = tables["nation"]

    # 1. Filter Nation -> Supplier
    nation_f = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supp_n = df_supplier.merge(nation_f, left_on="s_nationkey", right_on="n_nationkey")
    supp_n = supp_n[["s_suppkey", "s_name"]].sort_values("s_name").reset_index(drop=True)
    supp_n["supp_id"] = cudf.Series(range(len(supp_n)), dtype="int32")

    # 2. Filter Orders by status
    orders_f = df_orders[df_orders["o_orderstatus"] == orderstatus][["o_orderkey"]]

    # 3. Compute per-orderkey aggregates over FULL lineitem (restricted to orders_f)
    # This determines the EXISTS/NOT EXISTS conditions for all orders.
    li_all = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        orders_f, left_on="l_orderkey", right_on="o_orderkey"
    )
    li_all["late"] = (li_all["l_receiptdate"] > li_all["l_commitdate"]).astype("int8")

    # distinct suppliers per orderkey
    distinct_supp = (
        li_all[["l_orderkey", "l_suppkey"]]
        .drop_duplicates()
        .groupby("l_orderkey")
        .size()
        .reset_index()
    )
    distinct_supp.columns = ["l_orderkey", "distinct_supp"]

    # distinct late suppliers per orderkey
    late_rows = li_all[li_all["late"] == 1][["l_orderkey", "l_suppkey"]].drop_duplicates()
    late_distinct = late_rows.groupby("l_orderkey").size().reset_index()
    late_distinct.columns = ["l_orderkey", "late_distinct_supp"]

    agg = distinct_supp.merge(late_distinct, on="l_orderkey", how="left")
    agg["late_distinct_supp"] = agg["late_distinct_supp"].fillna(0).astype("int32")
    agg["distinct_supp"] = agg["distinct_supp"].astype("int32")
    # Sort for binary search in Triton
    agg = agg.sort_values("l_orderkey").reset_index(drop=True)

    # 4. Build l1 rows: supplier join + orders filter + late filter
    # We filter 'late' here in cuDF to reduce rows passed to Triton.
    l1 = df_lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]].merge(
        supp_n, left_on="l_suppkey", right_on="s_suppkey"
    )
    l1 = l1.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")
    l1 = l1[l1["l_receiptdate"] > l1["l_commitdate"]] # Filter late only

    # 5. Convert to Tensors
    l1_orderkey_t = torch.as_tensor(l1["l_orderkey"].to_cupy(), dtype=torch.int64)
    l1_supp_id_t = torch.as_tensor(l1["supp_id"].to_cupy(), dtype=torch.int32)

    lo_orderkeys_t = torch.as_tensor(agg["l_orderkey"].to_cupy(), dtype=torch.int64)
    lo_distinct_supp_t = torch.as_tensor(agg["distinct_supp"].to_cupy(), dtype=torch.int32)
    lo_late_distinct_supp_t = torch.as_tensor(agg["late_distinct_supp"].to_cupy(), dtype=torch.int32)

    num_suppliers = len(supp_n)
    n_l1 = l1_orderkey_t.numel()
    n_agg = lo_orderkeys_t.numel()

    # Output counts tensor
    counts_t = torch.zeros(num_suppliers, dtype=torch.int64, device=l1_orderkey_t.device)

    if n_l1 > 0 and n_agg > 0:
        grid = lambda meta: (triton.cdiv(n_l1, meta["BLOCK_SIZE"]),)
        _K._q21_kernel[grid](
            l1_orderkey_t,
            l1_supp_id_t,
            lo_orderkeys_t,
            lo_distinct_supp_t,
            lo_late_distinct_supp_t,
            counts_t,
            n_l1,
            n_agg,
        )

    # 6. Post-process
    # Filter suppliers with count > 0
    valid_mask = counts_t > 0
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
    
    if valid_indices.numel() == 0:
        return cudf.DataFrame({"s_name": [], "numwait": []})

    # Gather names and counts
    # supp_n is sorted by s_name. valid_indices are indices into supp_n.
    # We need to construct result DataFrame.
    # Convert to cuDF
    valid_indices_cpu = valid_indices.cpu().numpy()
    counts_cpu = counts_t[valid_indices].cpu().numpy()
    
    result = cudf.DataFrame({
        "s_name": supp_n["s_name"].iloc[valid_indices_cpu].reset_index(drop=True),
        "numwait": counts_cpu,
    })
    
    # Sort and Limit
    result = result.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100).reset_index(drop=True)
    return result
