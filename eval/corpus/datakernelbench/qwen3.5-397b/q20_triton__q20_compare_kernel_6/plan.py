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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_mask_ptr"],
)
@triton.jit
def _q20_compare_kernel(
    ps_availqty_ptr, sum_qty_ptr,
    out_mask_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Q20 comparison: ps_availqty > 0.5 * sum_qty AND sum_qty > 0
    Fuses the comparison into a single GPU pass for memory-bound scan optimization.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data with safe defaults for OOB lanes
    avail = tl.load(ps_availqty_ptr + offs, mask=mask, other=0.0)
    sum_q = tl.load(sum_qty_ptr + offs, mask=mask, other=0.0)
    
    # Compare: ps_availqty > 0.5 * sum_qty AND sum_qty > 0
    keep = (avail > 0.5 * sum_q) & (sum_q > 0)
    
    # Store mask (int32 for CUDA compatibility)
    tl.store(out_mask_ptr + offs, keep.to(tl.int32), mask=mask)
'''

def _load_kernels():
    """Load Triton kernels from temporary file (required for @triton.jit)."""
    tmpdir = tempfile.mkdtemp(prefix="q20_triton_")
    path = os.path.join(tmpdir, "_q20_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q20_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    l_partkey: torch.Tensor,
    l_suppkey: torch.Tensor,
    l_quantity: torch.Tensor,
    ps_partkey: torch.Tensor,
    ps_suppkey: torch.Tensor,
    ps_availqty: torch.Tensor,
    s_suppkey: torch.Tensor,
    s_row_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Core query logic using Triton for hot path (comparison + filtering).
    
    GPU execution:
    - torch.unique for composite key grouping (matches reference semantics exactly)
    - scatter_add for quantity aggregation per (partkey, suppkey) group
    - Triton kernel for ps_availqty > 0.5 * sum_qty comparison (fused scan)
    - torch.isin for final supplier filtering
    
    Why faster:
    - Triton kernel fuses comparison into single memory-bound pass
    - Reduces HBM traffic compared to multiple elementwise operations
    - Maintains reference semantics for correctness
    
    Params:
        l_partkey, l_suppkey, l_quantity: Filtered lineitem tensors
        ps_partkey, ps_suppkey, ps_availqty: Filtered partsupp tensors
        s_suppkey, s_row_idx: Filtered supplier tensors
    
    Returns:
        torch.Tensor: Row indices of qualifying suppliers
    """
    # Build composite key (partkey, suppkey) for aggregation - matches reference exactly
    ps_pair = torch.stack([ps_partkey, ps_suppkey], dim=1)  # [Nps, 2]
    li_pair = torch.stack([l_partkey, l_suppkey], dim=1)    # [Nli, 2]

    # Concatenate to assign consistent group ids across both sets
    all_pairs = torch.cat([ps_pair, li_pair], dim=0)
    unique_pairs, inverse = torch.unique(all_pairs, dim=0, return_inverse=True)
    Nps = ps_pair.shape[0]
    ps_group = inverse[:Nps]
    li_group = inverse[Nps:]

    # Sum l_quantity per group
    num_groups = unique_pairs.shape[0]
    sum_qty = torch.zeros(num_groups, dtype=l_quantity.dtype, device=l_quantity.device)
    sum_qty.scatter_add_(0, li_group, l_quantity)

    # For each partsupp row, get the corresponding sum of quantities
    ps_sum_qty = sum_qty[ps_group]

    # Use Triton for the comparison (hot path - memory-bound scan)
    n_ps = len(ps_availqty)
    out_mask = torch.zeros(n_ps, dtype=torch.int32, device=ps_availqty.device)
    grid = lambda meta: (triton.cdiv(n_ps, meta["BLOCK_SIZE"]),)
    _K._q20_compare_kernel[grid](
        ps_availqty, ps_sum_qty, out_mask, n_ps,
    )
    
    qualifying_ps_mask = out_mask.to(torch.bool)
    qualifying_suppkeys = ps_suppkey[qualifying_ps_mask]

    if len(qualifying_suppkeys) == 0:
        return torch.tensor([], dtype=torch.int64, device=s_suppkey.device)
    
    unique_qual_supp = torch.unique(qualifying_suppkeys)

    # Use torch.isin for final filtering (optimized)
    supp_mask = torch.isin(s_suppkey, unique_qual_supp)
    return s_row_idx[supp_mask]


def run_query(
    tables: dict[str, cudf.DataFrame],
    name_prefix: str = "forest",
    nation_name: str = "CANADA",
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
) -> cudf.DataFrame:
    """
    SQL: SELECT s_name, s_address FROM supplier, nation WHERE s_suppkey IN (
            SELECT ps_suppkey FROM partsupp WHERE ps_partkey IN (
                SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
            AND ps_availqty > (SELECT 0.5 * sum(l_quantity) FROM lineitem
                               WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
                               AND l_shipdate >= DATE '1994-01-01'
                               AND l_shipdate <  DATE '1995-01-01'))
          AND s_nationkey = n_nationkey AND n_name = 'CANADA'
          ORDER BY s_name;
    
    Description: TPC-H Q20 — finds suppliers in a specific nation who have excess inventory
                 of parts whose name starts with a given prefix during a date range.
                 
    Execution plan:
    - cuDF: String filtering (p_name LIKE), nation join, date filtering
    - torch.unique: Composite key grouping for (partkey, suppkey) aggregation
    - Triton: Fused comparison kernel (ps_availqty > 0.5 * sum_qty) - memory-bound scan
    - torch: Final supplier filtering and result materialization
    
    Why faster than reference:
    - Triton kernel fuses comparison into single GPU pass (reduces HBM traffic)
    - Better memory coalescing for column scans
    - Maintains exact reference semantics for correctness
    
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables (supplier, nation, partsupp, part, lineitem)
        name_prefix (str): Prefix filter for p_name (LIKE '<prefix>%'). Default: 'forest'
        nation_name (str): Nation filter for supplier's nation. Default: 'CANADA'
        shipdate_lo (str): Inclusive lower bound on l_shipdate. Default: '1994-01-01'
        shipdate_hi (str): Exclusive upper bound on l_shipdate. Default: '1995-01-01'
    
    Returns:
        cudf.DataFrame: Result with columns [s_name, s_address], sorted by s_name
    """
    df_supplier = tables["supplier"]
    df_nation = tables["nation"]
    df_partsupp = tables["partsupp"]
    df_part = tables["part"]
    df_lineitem = tables["lineitem"]

    # Convert date parameters to epoch days (int32)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # --- Filter part by p_name LIKE 'prefix%' ---
    part_mask = df_part["p_name"].str.startswith(name_prefix)
    filtered_part = df_part[part_mask][["p_partkey"]]

    # --- Filter partsupp by ps_partkey IN filtered_part ---
    filtered_partsupp = df_partsupp.merge(
        filtered_part, left_on="ps_partkey", right_on="p_partkey", how="inner"
    )[["ps_partkey", "ps_suppkey", "ps_availqty"]]

    # --- Filter lineitem by shipdate range ---
    l_shipdate_days = (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    li_mask = (l_shipdate_days >= shipdate_lo_days) & (l_shipdate_days < shipdate_hi_days)
    filtered_lineitem = df_lineitem[li_mask][["l_partkey", "l_suppkey", "l_quantity"]]

    # --- Filter supplier by nation = CANADA ---
    nation_filtered = df_nation[df_nation["n_name"] == nation_name][["n_nationkey"]]
    supplier_in_nation = df_supplier.merge(
        nation_filtered, left_on="s_nationkey", right_on="n_nationkey", how="inner"
    ).reset_index(drop=True)

    # Add a row index column so we can recover s_name / s_address by tensor index
    supplier_in_nation = supplier_in_nation.reset_index(drop=True)
    s_row_idx_series = cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # Convert tensors
    l_partkey_t = torch.as_tensor(filtered_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_suppkey_t = torch.as_tensor(filtered_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(filtered_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_availqty_t = torch.as_tensor(filtered_partsupp["ps_availqty"].to_cupy(), dtype=torch.float64)

    s_suppkey_t = torch.as_tensor(supplier_in_nation["s_suppkey"].to_cupy(), dtype=torch.int64)
    s_row_idx_t = torch.as_tensor(s_row_idx_series.to_cupy(), dtype=torch.int64)

    selected_idx_t = _query_core(
        l_partkey_t, l_suppkey_t, l_quantity_t,
        ps_partkey_t, ps_suppkey_t, ps_availqty_t,
        s_suppkey_t, s_row_idx_t,
    )

    # Materialize result via gather on cuDF side
    selected_idx_cudf = cudf.Series(selected_idx_t)
    result = supplier_in_nation.iloc[selected_idx_cudf.to_pandas().tolist()][["s_name", "s_address"]]
    result = result.sort_values("s_name").reset_index(drop=True)
    return result
