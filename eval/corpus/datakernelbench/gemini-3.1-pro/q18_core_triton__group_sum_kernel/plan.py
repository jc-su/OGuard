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
def _group_sum_kernel(
    orderkey_ptr,
    quantity_ptr,
    sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load orderkeys and quantities
    ok = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    # Atomic add directly into the pre-allocated index based on the orderkey
    tl.atomic_add(sums_ptr + ok, qty, mask=mask)
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
    
    Optimization: Avoids heavy sorting/hashing from torch.unique by taking advantage 
    of the fact that TPC-H order keys are integers bounded within a small maximum value. 
    It allocates an array of size (max_key + 1) and uses a Triton fused kernel to 
    perform hardware atomic additions in a single pass over HBM, drastically reducing 
    memory traffic and execution time.
    """
    n = l_orderkey.numel()
    
    # Handle empty lineitem bounds seamlessly
    if n == 0:
        return (
            torch.empty(0, dtype=l_orderkey.dtype, device=l_orderkey.device),
            torch.empty(0, dtype=l_quantity.dtype, device=l_quantity.device),
            torch.empty(0, dtype=l_orderkey.dtype, device=l_orderkey.device)
        )
        
    # Get the maximum orderkey to size our reduction array
    max_key = int(l_orderkey.max().item())
    
    # Pre-allocate contiguous memory for all possible keys up to max_key.
    # TPC-H orderkeys scale predictably (SF * 6,000,000) making this highly efficient.
    sums = torch.zeros(max_key + 1, dtype=torch.float64, device=l_orderkey.device)
    
    # Hardcoded robust config (avoids @triton.autotune inflating atomic adds due to multiple runs without exact reset logic)
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE),)
    
    _K._group_sum_kernel[grid](
        l_orderkey, l_quantity, sums, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
    )
    
    # In TPC-H, l_quantity is strictly positive (> 0). We can extract exactly 
    # those keys which actually appeared in the lineitem table via sums > 0.
    unique_mask = sums > 0
    big_mask = unique_mask & (sums > qty_threshold)
    
    # Compute output values matching the exact types and shapes of torch.unique references
    big_keys = torch.nonzero(big_mask).squeeze(-1)
    big_sums = sums[big_keys]
    
    unique_keys = torch.nonzero(unique_mask).squeeze(-1)
    
    return big_keys, big_sums, unique_keys


def run_query(tables: dict[str, cudf.DataFrame], qty_threshold: float = 300.0, limit: int = 100) -> cudf.DataFrame:
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
    Params:
      qty_threshold (float): HAVING sum(l_quantity) > threshold.
      limit (int): number of rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --- Compute big orderkeys (HAVING sum(l_quantity) > threshold) in the core ---
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
