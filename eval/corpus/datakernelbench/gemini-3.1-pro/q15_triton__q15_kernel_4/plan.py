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
    reset_to_zero=["revenue_ptr", "present_ptr"],
)
@triton.jit
def _q15_kernel(
    l_suppkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_ptr,
    date_lo,
    date_hi,
    revenue_ptr,
    present_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load shipdate first to evaluate predicate
    shipdate = tl.load(l_shipdate_ptr + offs, mask=mask, other=date_hi)
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Conditionally load expensive columns (selectivity is ~3.5%)
    suppkey = tl.load(l_suppkey_ptr + offs, mask=keep, other=0)
    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    
    # Calculate revenue
    revenue = price * (1.0 - discount)
    
    # Scatter add to global accumulator array indexed directly by suppkey
    revenue_ptr_offset = revenue_ptr + suppkey
    present_ptr_offset = present_ptr + suppkey
    
    tl.atomic_add(revenue_ptr_offset, revenue, mask=keep)
    tl.atomic_max(present_ptr_offset, 1, mask=keep)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q15_triton_")
    path = os.path.join(tmpdir, "_q15_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q15_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def _query_core(
    l_suppkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    s_suppkey: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Core execution plan:
    1. Determine the maximum supplier key to allocate a direct lookup array.
    2. Launch a single Triton JIT kernel to scan lineitem, conditionally load payload,
       compute revenue, and globally scatter-add/record presence into arrays.
    3. Determine max revenue among actively present suppliers and return their keys.
    """
    n = l_suppkey.numel()
    
    if n == 0 or s_suppkey.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=l_suppkey.device),
            torch.empty(0, dtype=torch.int64, device=l_suppkey.device),
            torch.empty(0, dtype=torch.float64, device=l_suppkey.device),
        )

    # Referential integrity guarantees all l_suppkey exist in s_suppkey.
    # We allocate based on the max s_suppkey (usually dense, 1-indexed, fast lookup).
    max_suppkey = int(s_suppkey.max().item())
    
    revenue_arr = torch.zeros(max_suppkey + 1, dtype=torch.float64, device=l_suppkey.device)
    present_arr = torch.zeros(max_suppkey + 1, dtype=torch.int32, device=l_suppkey.device)
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._q15_kernel[grid](
        l_suppkey,
        l_extendedprice,
        l_discount,
        l_shipdate_days,
        shipdate_lo_days,
        shipdate_hi_days,
        revenue_arr,
        present_arr,
        n,
    )
    
    # We want to filter suppliers that are present and have revenue == max_revenue
    supplier_revenue = revenue_arr[s_suppkey]
    supplier_present = present_arr[s_suppkey] > 0
    
    if not supplier_present.any():
        return (
            torch.empty(0, dtype=torch.int64, device=s_suppkey.device),
            torch.empty(0, dtype=torch.int64, device=s_suppkey.device),
            torch.empty(0, dtype=torch.float64, device=s_suppkey.device),
        )
    
    max_rev = supplier_revenue[supplier_present].max()
    
    sel_mask = supplier_present & (supplier_revenue == max_rev)
    sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
    
    sel_keys = s_suppkey[sel_idx]
    order = torch.argsort(sel_keys)
    sel_idx = sel_idx[order]
    
    return sel_idx, s_suppkey[sel_idx], supplier_revenue[sel_idx]

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1996-01-01",
    shipdate_hi: str = "1996-04-01",
) -> cudf.DataFrame:
    """
    SQL:
      WITH revenue AS (
        SELECT l_suppkey AS supplier_no,
               SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
        GROUP BY l_suppkey
      )
      SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
      FROM supplier, revenue
      WHERE s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
      ORDER BY s_suppkey;
      
    Description: TPC-H Q15 — find supplier(s) with the top revenue in a given quarter.
    The cuDF data loading maps variables for GPU processing while a single fused Triton
    pass filters lineitem and aggregates directly to global memory arrays.
    
    Params:
      shipdate_lo (str): inclusive lower bound shipdate.
      shipdate_hi (str): exclusive upper bound shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_supplier = tables["supplier"]

    if df_lineitem.empty or df_supplier.empty:
        return cudf.DataFrame({
            "s_suppkey": cudf.Series(dtype="int64"),
            "s_name": cudf.Series(dtype="object"),
            "s_address": cudf.Series(dtype="object"),
            "s_phone": cudf.Series(dtype="object"),
            "total_revenue": cudf.Series(dtype="float64"),
        })

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_suppkey_t = torch.as_tensor(df_lineitem["l_suppkey"].to_cupy(), dtype=torch.int64)
    l_price_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_disc_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    s_suppkey_t = torch.as_tensor(df_supplier["s_suppkey"].to_cupy(), dtype=torch.int64)

    sel_idx_t, sel_keys_t, sel_rev_t = _query_core(
        l_suppkey_t,
        l_price_t,
        l_disc_t,
        l_shipdate_days_t,
        s_suppkey_t,
        shipdate_lo_days,
        shipdate_hi_days,
    )

    if sel_idx_t.numel() == 0:
        return cudf.DataFrame({
            "s_suppkey": cudf.Series(dtype="int64"),
            "s_name": cudf.Series(dtype="object"),
            "s_address": cudf.Series(dtype="object"),
            "s_phone": cudf.Series(dtype="object"),
            "total_revenue": cudf.Series(dtype="float64"),
        })

    # Use indices to gather string/phone columns from cuDF supplier table
    sel_idx_series = cudf.Series(sel_idx_t)
    supplier_sel = df_supplier.iloc[sel_idx_series].reset_index(drop=True)

    return cudf.DataFrame({
        "s_suppkey": supplier_sel["s_suppkey"],
        "s_name": supplier_sel["s_name"],
        "s_address": supplier_sel["s_address"],
        "s_phone": supplier_sel["s_phone"],
        "total_revenue": cudf.Series(sel_rev_t),
    })
