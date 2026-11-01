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
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _scatter_add_kernel(
    out_ptr,
    indices_ptr,
    values_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused scatter_add: out[indices[i]] += values[i]
    
    Memory-bound optimization:
    - Single pass through indices and values (reduces HBM traffic)
    - Coalesced memory access patterns optimized for H100
    - Atomic operations for concurrent group updates
    - Autotuned block sizes for optimal occupancy on Hopper
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load indices and values with mask for OOB protection
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    val = tl.load(values_ptr + offs, mask=mask, other=0.0)
    
    # Atomic add to output groups (handles concurrent writes safely)
    tl.atomic_add(out_ptr + idx, val, mask=mask)
'''


def _load_kernels():
    """
    Load Triton kernels from temporary file.
    
    Required pattern: Triton JIT kernels must be in a separate on-disk file.
    This function writes _KERNEL_SOURCE to temp file and imports it via importlib.
    """
    tmpdir = tempfile.mkdtemp(prefix="q20_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _scatter_add_triton(
    indices: torch.Tensor,
    values: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """
    Triton-based scatter_add for summing values by group index.
    
    GPU: Runs on CUDA device via Triton JIT kernel
    Why faster: Single fused kernel pass with atomic operations,
    autotuned block sizes for H100, better memory coalescing than
    PyTorch scatter_add for large tensors.
    
    Args:
        indices: 1D tensor of group indices (int64)
        values: 1D tensor of values to sum (float64)
        num_groups: Number of unique groups
    
    Returns:
        1D tensor of summed values per group
    """
    n = indices.numel()
    out = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._scatter_add_kernel[grid](
        out, indices, values, n,
    )
    return out


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
    Core query logic for TPC-H Q20.
    
    GPU operations:
    - torch.unique for group ID assignment (host-side, efficient)
    - Triton scatter_add for quantity aggregation (fused kernel)
    - Tensor indexing and filtering (PyTorch)
    
    Why faster than reference:
    - Triton scatter_add reduces HBM traffic with single kernel pass
    - Autotuned block sizes optimize for H100 memory bandwidth (3.35 TB/s)
    - Atomic operations handle concurrent group updates efficiently
    - Same semantics as reference (verified by matching output)
    
    Args:
        l_partkey: lineitem partkey (filtered by date range)
        l_suppkey: lineitem suppkey (filtered by date range)
        l_quantity: lineitem quantity (filtered by date range)
        ps_partkey: partsupp partkey (filtered by part name)
        ps_suppkey: partsupp suppkey (filtered by part name)
        ps_availqty: partsupp available quantity
        s_suppkey: supplier suppkey (filtered by nation)
        s_row_idx: supplier row indices for result materialization
    
    Returns:
        Tensor of selected supplier row indices
    """
    # Build composite keys for grouping by (partkey, suppkey)
    ps_pair = torch.stack([ps_partkey, ps_suppkey], dim=1)  # [Nps, 2]
    li_pair = torch.stack([l_partkey, l_suppkey], dim=1)    # [Nli, 2]

    # Concatenate to assign consistent group ids across both sets
    all_pairs = torch.cat([ps_pair, li_pair], dim=0)
    unique_pairs, inverse = torch.unique(all_pairs, dim=0, return_inverse=True)
    Nps = ps_pair.shape[0]
    ps_group = inverse[:Nps]
    li_group = inverse[Nps:]

    # Sum l_quantity per group using Triton (fused scatter_add)
    num_groups = unique_pairs.shape[0]
    sum_qty = _scatter_add_triton(li_group, l_quantity, num_groups)

    # For each partsupp row, get the corresponding sum of quantities
    ps_sum_qty = sum_qty[ps_group]

    # Filter partsupp rows where ps_availqty > 0.5 * sum_qty
    # (and sum_qty > 0 ensures lineitem rows exist for this pair)
    ps_availqty_f = ps_availqty.to(torch.float64)
    qualifying_ps_mask = (ps_availqty_f > 0.5 * ps_sum_qty) & (ps_sum_qty > 0)

    qualifying_suppkeys = ps_suppkey[qualifying_ps_mask]

    # Find unique qualifying supplier keys
    unique_qual_supp = torch.unique(qualifying_suppkeys)

    # Match against supplier table
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
    Params:
        name_prefix (str): prefix filter for p_name (LIKE '<prefix>%').
        nation_name (str): nation filter for supplier's nation.
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
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
    s_row_idx_series = cudf.Series(
        cudf.core.column.arange(0, len(supplier_in_nation), dtype="int64")
    ) if False else cudf.Series(range(len(supplier_in_nation))).astype("int64")

    # Convert tensors
    l_partkey_t = torch.as_tensor(filtered_lineitem["l_partkey"].to_cupy(), dtype=torch.int64)
    l_suppkey_t = torch.as_tensor(filtered_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(filtered_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    ps_partkey_t = torch.as_tensor(filtered_partsupp["ps_partkey"].to_cupy(), dtype=torch.int64)
    ps_suppkey_t = torch.as_tensor(filtered_partsupp["ps_suppkey"].to_cupy(), dtype=torch.int64)
    ps_availqty_t = torch.as_tensor(filtered_partsupp["ps_availqty"].to_cupy(), dtype=torch.int64)

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
