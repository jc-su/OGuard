import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_valid_ptr"],
)
@triton.jit
def _filter_orders_kernel(
    order_key_ptr,
    order_date_ptr,
    priority_ptr,
    out_key_ptr,
    out_priority_ptr,
    out_valid_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Filter orders by date range using Triton.
    Fuses date filter with key/priority extraction in one kernel pass.
    Memory-bound optimization: single HBM pass for filter+project.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    order_date = tl.load(order_date_ptr + offs, mask=mask, other=date_hi)
    date_ok = (order_date >= date_lo) & (order_date < date_hi)
    keep = mask & date_ok
    
    order_key = tl.load(order_key_ptr + offs, mask=keep, other=0)
    priority = tl.load(priority_ptr + offs, mask=keep, other=0)
    
    tl.store(out_key_ptr + offs, order_key, mask=keep)
    tl.store(out_priority_ptr + offs, priority, mask=keep)
    tl.store(out_valid_ptr + offs, keep.to(tl.int32), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_valid_ptr"],
)
@triton.jit
def _filter_lineitems_kernel(
    line_key_ptr,
    line_commit_ptr,
    line_receipt_ptr,
    out_key_ptr,
    out_valid_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Filter lineitems where commit < receipt using Triton.
    Fuses comparison with key extraction in one kernel pass.
    Memory-bound optimization: single HBM pass for filter+project.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    l_commit = tl.load(line_commit_ptr + offs, mask=mask, other=0)
    l_receipt = tl.load(line_receipt_ptr + offs, mask=mask, other=0)
    keep = mask & (l_commit < l_receipt)
    
    line_key = tl.load(line_key_ptr + offs, mask=keep, other=0)
    tl.store(out_key_ptr + offs, line_key, mask=keep)
    tl.store(out_valid_ptr + offs, keep.to(tl.int32), mask=mask)
'''

def _load_kernels():
    """
    Load Triton kernels from a temporary file.
    Required because @triton.jit needs real on-disk source for compilation.
    """
    tmpdir = tempfile.mkdtemp(prefix="q4_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _launch_filter_orders(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_priority_codes: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Launch Triton kernel to filter orders by date range.
    Returns filtered keys, priorities, and validity mask.
    """
    n = o_orderkey.numel()
    out_key = torch.zeros(n, dtype=torch.int64, device=o_orderkey.device)
    out_priority = torch.zeros(n, dtype=torch.int32, device=o_priority_codes.device)
    out_valid = torch.zeros(n, dtype=torch.int32, device=o_orderkey.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._filter_orders_kernel[grid](
        o_orderkey, o_orderdate_days, o_priority_codes,
        out_key, out_priority, out_valid,
        date_lo_days, date_hi_days, n,
    )
    return out_key, out_priority, out_valid

def _launch_filter_lineitems(
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch Triton kernel to filter lineitems where commit < receipt.
    Returns filtered order keys and validity mask.
    """
    n = l_orderkey.numel()
    out_key = torch.zeros(n, dtype=torch.int64, device=l_orderkey.device)
    out_valid = torch.zeros(n, dtype=torch.int32, device=l_orderkey.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._filter_lineitems_kernel[grid](
        l_orderkey, l_commitdate_days, l_receiptdate_days,
        out_key, out_valid, n,
    )
    return out_key, out_valid

def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TPC-H Q4 core computation using Triton-optimized filtering.
    
    GPU Execution:
    - Triton kernels: filter orders by date, filter lineitems by commit<receipt
    - PyTorch tensors: existence check (isin), bincount aggregation
    
    Why faster:
    - Fused filter+project in Triton reduces HBM passes (memory-bound optimization)
    - Coalesced access patterns in Triton kernels
    - Autotuned block sizes for H100 architecture
    
    Args:
        o_orderkey: Order keys (int64)
        o_orderdate_days: Order dates as epoch days (int32)
        o_orderpriority_codes: Priority codes (int32)
        l_orderkey: Lineitem order keys (int64)
        l_commitdate_days: Commit dates as epoch days (int32)
        l_receiptdate_days: Receipt dates as epoch days (int32)
        date_lo_days: Inclusive lower date bound (epoch days)
        date_hi_days: Exclusive upper date bound (epoch days)
        num_priorities: Number of priority categories
    
    Returns:
        Tuple of (priority_ids, counts) for non-zero groups
    """
    # Filter orders by date range using Triton (fused filter+project)
    filt_orderkey, filt_priority, order_valid = _launch_filter_orders(
        o_orderkey, o_orderdate_days, o_orderpriority_codes,
        date_lo_days, date_hi_days,
    )
    
    # Filter lineitem rows where commitdate < receiptdate using Triton
    qualifying_l_orderkey, line_valid = _launch_filter_lineitems(
        l_orderkey, l_commitdate_days, l_receiptdate_days,
    )
    
    # Extract valid order keys and priorities
    order_mask = order_valid.bool()
    filt_orderkey = filt_orderkey[order_mask]
    filt_priority = filt_priority[order_mask]
    
    # Extract valid lineitem order keys
    line_mask = line_valid.bool()
    qualifying_l_orderkey = qualifying_l_orderkey[line_mask]
    
    # EXISTS: check which filt_orderkey values appear in qualifying_l_orderkey
    unique_keys = torch.unique(qualifying_l_orderkey)
    exists_mask = torch.isin(filt_orderkey, unique_keys)
    
    final_priority = filt_priority[exists_mask]
    
    # Group by priority code, count
    counts = torch.bincount(final_priority.to(torch.int64), minlength=num_priorities)
    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)
    
    # Keep only non-zero groups
    nz = counts > 0
    return priority_ids[nz], counts[nz]

def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_orderpriority, count(*) AS order_count FROM orders WHERE o_orderdate >= CAST('1993-07-01' AS date) AND o_orderdate < CAST('1993-10-01' AS date) AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority ORDER BY o_orderpriority;
    Description: TPC-H Q4 - Counts orders per priority within a date range where at least one lineitem shipped late (commit before receipt).
    Params:
        date_lo (str): inclusive lower bound for o_orderdate.
        date_hi (str): exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # Encode o_orderpriority as categorical for groupby on GPU
    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes  # int codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = torch.as_tensor(
        ((df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = torch.as_tensor(
        ((df_lineitem["l_commitdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_receiptdate_days_t = torch.as_tensor(
        ((df_lineitem["l_receiptdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
    )

    # Map codes back to priority strings and sort
    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[i] for i in priority_ids_host]

    result = cudf.DataFrame({
        "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
        "order_count": counts_t,
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
