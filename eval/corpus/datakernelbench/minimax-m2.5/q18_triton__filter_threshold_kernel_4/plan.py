import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd
import numpy as np

# ============================================================================
# TRITON KERNEL SOURCE - threshold filter
# ============================================================================
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096]
        for w in [4, 8]
        for s in [2, 3]
    ],
    key=["n_elements"],
)
@triton.jit
def _filter_threshold_kernel(
    sums_ptr, threshold, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
):
    """Apply threshold filter using Triton for reduced Python overhead."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    sums = tl.load(sums_ptr + offs, mask=mask, other=0.0)
    keep = mask & (sums > threshold)
    
    result = tl.where(keep, 1, 0)
    tl.store(out_ptr + offs, result, mask=mask)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q18_triton_")
    path = os.path.join(tmpdir, "_q18_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q18_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _query_core(
    l_orderkey: torch.Tensor,
    l_quantity: torch.Tensor,
    qty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes per-order sum of l_quantity, then filters those orders whose
    summed quantity exceeds the threshold. Returns the unique big order keys,
    along with (for the eventual join) sorted unique orderkeys + sums.
    
    Uses torch.unique (CUDA-optimized) + Triton filter for the threshold check.
    """
    # torch.unique is highly optimized on CUDA - use it directly
    unique_keys, inverse = torch.unique(l_orderkey, return_inverse=True)
    sums = torch.zeros(unique_keys.numel(), dtype=l_quantity.dtype, device=l_quantity.device)
    sums.scatter_add_(0, inverse, l_quantity)

    # Use Triton kernel for threshold filter - reduces Python interpreter overhead
    n = sums.numel()
    valid_mask = torch.zeros(n, dtype=torch.int32, device=sums.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._filter_threshold_kernel[grid](sums, float(qty_threshold), valid_mask, n)
    
    # Get indices where sum > threshold
    big_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
    big_keys = unique_keys[big_indices]
    big_sums = sums[big_indices]
    
    # Return sorted unique keys for merge join efficiency (torch.unique returns sorted)
    # But we need to return unique_keys sorted - torch.unique already returns sorted
    sorted_unique_keys = unique_keys
    sorted_sums = sums
    
    return big_keys, big_sums, sorted_unique_keys


def run_query(
    tables: dict[str, cudf.DataFrame],
    qty_threshold: float = 300.0,
    limit: int = 100,
) -> cudf.DataFrame:
    """
    SQL:
      SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
      FROM customer, orders, lineitem
      WHERE o_orderkey IN (
          SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300)
        AND c_custkey = o_custkey
        AND o_orderkey = l_orderkey
      GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
      ORDER BY o_totalprice DESC, o_orderdate
      LIMIT 100;
    
    Description: TPC-H Q18 — finds the top 100 customers with the largest-quantity orders.
    Uses torch.unique (CUDA-optimized) + Triton filter kernel to reduce Python overhead.
    
    Params:
      qty_threshold (float): HAVING sum(l_quantity) > threshold.
      limit (int): number of rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --- Compute big orderkeys (HAVING sum(l_quantity) > threshold) ---
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    big_keys_t, big_sums_t, _ = _query_core(l_orderkey_t, l_quantity_t, float(qty_threshold))

    # Build a cuDF DataFrame of qualifying orderkeys + their total quantity
    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    # Semi-join orders with big orderkeys, then join with customer
    orders_filtered = df_orders.merge(df_big, left_on="o_orderkey", right_on="o_orderkey", how="inner")
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

    # Order by o_totalprice DESC, o_orderdate ASC
    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"], ascending=[False, True]
    ).head(limit)

    result = cudf.DataFrame({
        "c_name": joined["c_name"].reset_index(drop=True),
        "c_custkey": joined["c_custkey"].reset_index(drop=True),
        "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
        "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
        "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
        "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
    })
    return result
