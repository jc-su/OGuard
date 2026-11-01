import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton
import triton.language as tl

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
    reset_to_zero=["out_total_ptr", "out_promo_ptr"],
)
@triton.jit
def _promo_kernel(
    shipdate_ptr,
    price_ptr,
    discount_ptr,
    promo_ptr,
    out_total_ptr,
    out_promo_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load columns
    # shipdate is int32 days
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=0)
    # price/discount are float64
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    # promo is bool
    promo = tl.load(promo_ptr + offs, mask=mask, other=False)
    
    # Date filter: shipdate_lo <= shipdate < shipdate_hi
    keep_date = (shipdate >= date_lo) & (shipdate < date_hi)
    
    # Revenue calculation: extendedprice * (1 - discount)
    revenue = price * (1.0 - discount)
    
    # Conditional accumulation
    # If keep_date is false, contribution is 0.0
    total_contrib = tl.where(keep_date, revenue, 0.0)
    # If keep_date AND promo are true, contribution is revenue, else 0.0
    promo_contrib = tl.where(keep_date & promo, revenue, 0.0)
    
    # Block-level reduction
    total_sum = tl.sum(total_contrib)
    promo_sum = tl.sum(promo_contrib)
    
    # Global accumulation
    tl.atomic_add(out_total_ptr, total_sum)
    tl.atomic_add(out_promo_ptr, promo_sum)
'''

def _load_kernels():
    """
    Writes Triton kernel source to a temp file and imports it.
    Required because @triton.jit functions must be defined in a real file.
    """
    tmpdir = tempfile.mkdtemp(prefix="q14_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load kernels at module import time
_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
                 Uses cuDF for join and string filtering, then a fused Triton kernel for 
                 date filtering, revenue projection, and aggregation to minimize HBM traffic.
    Params:
        tables (dict[str, cudf.DataFrame]): Input tables ('lineitem', 'part').
        shipdate_lo (str): inclusive lower bound on l_shipdate (YYYY-MM-DD).
        shipdate_hi (str): exclusive upper bound on l_shipdate (YYYY-MM-DD).
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    # Convert date bounds to int32 epoch days
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Precompute promo flag on part side before join (string op on CPU/cuDF)
    part_proj = cudf.DataFrame({
        "p_partkey": df_part["p_partkey"],
        "p_type_is_promo": df_part["p_type"].str.startswith("PROMO"),
    })

    # Join lineitem and part on partkey
    joined = df_lineitem.merge(part_proj, left_on="l_partkey", right_on="p_partkey")

    # Prepare GPU tensors
    # Shipdate: datetime64[s] -> int64 seconds -> int32 days
    l_shipdate_days_t = torch.as_tensor(
        ((joined["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    # Financial columns: float64 for precision
    l_extendedprice_t = torch.as_tensor(joined["l_extendedprice"].to_cupy(), dtype=torch.float64)
    l_discount_t = torch.as_tensor(joined["l_discount"].to_cupy(), dtype=torch.float64)
    # Promo flag: bool
    p_type_is_promo_t = torch.as_tensor(joined["p_type_is_promo"].astype("bool").to_cupy())

    n = l_extendedprice_t.numel()
    
    # Output buffers for global sums
    out_total = torch.zeros(1, dtype=torch.float64, device=l_extendedprice_t.device)
    out_promo = torch.zeros(1, dtype=torch.float64, device=l_extendedprice_t.device)
    
    # Launch grid
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    # Execute fused kernel
    _K._promo_kernel[grid](
        l_shipdate_days_t,
        l_extendedprice_t,
        l_discount_t,
        p_type_is_promo_t,
        out_total,
        out_promo,
        shipdate_lo_days,
        shipdate_hi_days,
        n,
    )
    
    total_revenue = float(out_total.item())
    promo_revenue = float(out_promo.item())
    
    if total_revenue == 0:
        return 0.0
    return 100.0 * promo_revenue / total_revenue
