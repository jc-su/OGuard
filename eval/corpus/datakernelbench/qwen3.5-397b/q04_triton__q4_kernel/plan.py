import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q4_kernel(
    priority_ptr,
    filter_mask_ptr,
    counts_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for TPC-H Q4 priority counting with block-level reduction.
    
    Each block accumulates local counts for all 5 priorities using tl.sum,
    then does ONE atomic add per priority. This reduces atomic contention
    from O(n) to O(num_blocks).
    
    Optimized for H100: large blocks, minimal atomics, coalesced access.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load priority codes and filter mask
    priority = tl.load(priority_ptr + offs, mask=mask, other=0)
    keep = tl.load(filter_mask_ptr + offs, mask=mask, other=0)
    
    # Block-level reduction: count per priority in this block
    # tl.sum reduces across the BLOCK_SIZE dimension efficiently
    count_0 = tl.sum(tl.where((priority == 0) & (keep == 1), 1, 0).to(tl.int64))
    count_1 = tl.sum(tl.where((priority == 1) & (keep == 1), 1, 0).to(tl.int64))
    count_2 = tl.sum(tl.where((priority == 2) & (keep == 1), 1, 0).to(tl.int64))
    count_3 = tl.sum(tl.where((priority == 3) & (keep == 1), 1, 0).to(tl.int64))
    count_4 = tl.sum(tl.where((priority == 4) & (keep == 1), 1, 0).to(tl.int64))
    
    # One atomic add per priority per block (5 atomics per block)
    tl.atomic_add(counts_ptr + 0, count_0, sem="relaxed")
    tl.atomic_add(counts_ptr + 1, count_1, sem="relaxed")
    tl.atomic_add(counts_ptr + 2, count_2, sem="relaxed")
    tl.atomic_add(counts_ptr + 3, count_3, sem="relaxed")
    tl.atomic_add(counts_ptr + 4, count_4, sem="relaxed")
'''

def _load_kernels():
    """Load Triton kernels from temporary on-disk file (required for @triton.jit)."""
    tmpdir = tempfile.mkdtemp(prefix="q4_triton_")
    path = os.path.join(tmpdir, "_q4_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q4_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


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
    Core query logic using fused Triton kernel with block reduction.
    
    GPU execution:
    - Host: Compute combined filter mask efficiently
    - Triton: Single-pass priority counting with block-level reduction
    
    Optimization: Use torch operations for filtering (already optimized),
    then Triton for counting with minimal atomic contention.
    """
    n_elements = o_orderkey.numel()
    
    # Step 1: Filter orders by date range (vectorized, fast)
    date_mask = (o_orderdate_days >= date_lo_days) & (o_orderdate_days < date_hi_days)
    
    # Step 2: Filter lineitem and get unique qualifying orderkeys
    # This is the expensive part - lineitem has 60M rows
    line_mask = l_commitdate_days < l_receiptdate_days
    qualifying_l_orderkey = l_orderkey[line_mask]
    unique_keys = torch.unique(qualifying_l_orderkey)
    
    # Step 3: EXISTS check - use sorted search for efficiency
    # Sort unique_keys for binary search
    unique_keys_sorted, _ = torch.sort(unique_keys)
    
    # Use searchsorted for faster lookup than isin
    positions = torch.searchsorted(unique_keys_sorted, o_orderkey)
    positions = positions.clamp(max=len(unique_keys_sorted) - 1)
    exists_mask = (unique_keys_sorted[positions] == o_orderkey)
    
    # Step 4: Combine masks
    filter_mask = (date_mask & exists_mask).to(torch.int32)
    
    # Step 5: Launch Triton kernel for priority counting
    counts = torch.zeros(num_priorities, dtype=torch.int64, device=o_orderkey.device)
    
    n_filtered = filter_mask.sum().item()
    if n_filtered > 0:
        # Use fixed large block size for H100 (avoid autotune overhead)
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _K._q4_kernel[grid](
            o_orderpriority_codes,
            filter_mask,
            counts,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Return non-zero priority counts
    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)
    nz = counts > 0
    return priority_ids[nz], counts[nz]


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT o_orderpriority, count(*) AS order_count
        FROM orders
        WHERE o_orderdate >= CAST('1993-07-01' AS date)
          AND o_orderdate < CAST('1993-10-01' AS date)
          AND EXISTS (
              SELECT * FROM lineitem
              WHERE l_orderkey = o_orderkey
                AND l_commitdate < l_receiptdate
          )
        GROUP BY o_orderpriority
        ORDER BY o_orderpriority;
    
    Description:
        TPC-H Q4 - Counts orders per priority within a date range where at least
        one lineitem shipped late (commit before receipt).
        
        GPU Optimization (Triton fused kernel with block reduction):
        - cuDF: Load tables, encode priority as categorical codes
        - Torch: Efficient filtering with searchsorted for EXISTS check
        - Triton: Single-pass priority counting with block-level reduction
        - Why faster: searchsorted is O(log n) vs isin O(n), block reduction
          reduces atomic contention from O(n) to O(num_blocks), fixed block
          size avoids autotune overhead
    
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables ("orders", "lineitem")
        date_lo (str): Inclusive lower bound for o_orderdate (default: "1993-07-01")
        date_hi (str): Exclusive upper bound for o_orderdate (default: "1993-10-01")
    
    Returns:
        cudf.DataFrame: Columns [o_orderpriority, order_count], sorted by priority
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # Convert date bounds to epoch days (int32)
    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    # Encode o_orderpriority as categorical for groupby on GPU
    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes  # int codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    # Transfer data to GPU tensors
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

    # Execute fused query core
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
        "order_count": counts_t.cpu(),
    })
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
