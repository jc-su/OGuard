import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
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
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _group_sum_kernel(
    orderkey_ptr, quantity_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Scans the lineitem quantities and directly accumulates them into a global
    array indexed by orderkey using fast atomic additions.
    This replaces a multi-pass sort/unique/scatter_add with a single pass,
    significantly reducing memory bandwidth requirements and temporary allocations.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load orderkeys and quantities
    okey = tl.load(orderkey_ptr + offs, mask=mask, other=0)
    qty = tl.load(quantity_ptr + offs, mask=mask, other=0.0)
    
    # Hopper L2 atomics are extremely fast and sufficient for this 1D histogram pattern.
    tl.atomic_add(out_ptr + okey, qty, mask=mask)
'''

def _load_kernels():
    """
    Writes the Triton kernel string to a temporary file and loads it as a module
    to satisfy execution environment constraints without top-level decorators.
    """
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
      
    Description: 
      TPC-H Q18 — finds the top 100 customers with the largest-quantity orders.
      The heavy GROUP BY and filter is pushed down into a custom fused Triton kernel.
      Instead of expensive `torch.unique` and `scatter_add_`, it allocates a dense
      1D tensor sized by the maximum orderkey and performs a direct atomic reduction.
      
    Params:
      qty_threshold (float): HAVING sum(l_quantity) > threshold.
      limit (int): number of rows to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    # --- Compute big orderkeys (HAVING sum(l_quantity) > threshold) in Triton ---
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)

    n = l_orderkey_t.numel()
    
    # Dense allocation works well since TPC-H orderkeys are bounded and contiguous-ish
    # Max orderkey in SF10 is ~60,000,000, so this takes about 480MB (well within HBM)
    max_key = int(l_orderkey_t.max().item()) if n > 0 else 0
    out_sums = torch.zeros(max_key + 1, dtype=torch.float64, device=l_orderkey_t.device)
    
    if n > 0:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        
        # Fused single-pass atomic sum reduction
        _K._group_sum_kernel[grid](
            l_orderkey_t, 
            l_quantity_t, 
            out_sums, 
            n,
        )

    # Filter where sum > threshold using PyTorch primitives
    big_mask = out_sums > float(qty_threshold)
    
    # torch.nonzero with as_tuple=True avoids the need to squeeze and guarantees a 1D index
    big_keys_t = torch.nonzero(big_mask, as_tuple=True)[0]
    big_sums_t = out_sums[big_keys_t]

    # Build a cuDF DataFrame of qualifying orderkeys + their total quantity
    df_big = cudf.DataFrame({
        "o_orderkey": cudf.Series(big_keys_t),
        "sum_qty": cudf.Series(big_sums_t),
    })

    # Semi-join orders with big orderkeys, then join with customer
    orders_filtered = df_orders.merge(df_big, on="o_orderkey", how="inner")
    joined = orders_filtered.merge(
        df_customer[["c_custkey", "c_name"]],
        left_on="o_custkey", right_on="c_custkey", how="inner"
    )

    # Order by o_totalprice DESC, o_orderdate ASC
    joined = joined.sort_values(
        by=["o_totalprice", "o_orderdate"], ascending=[False, True]
    ).head(limit)

    # Format the final result
    result = cudf.DataFrame({
        "c_name": joined["c_name"].reset_index(drop=True),
        "c_custkey": joined["c_custkey"].reset_index(drop=True),
        "o_orderkey": joined["o_orderkey"].reset_index(drop=True),
        "o_orderdate": joined["o_orderdate"].reset_index(drop=True),
        "o_totalprice": joined["o_totalprice"].reset_index(drop=True),
        "sum(l_quantity)": joined["sum_qty"].reset_index(drop=True),
    })
    
    return result
