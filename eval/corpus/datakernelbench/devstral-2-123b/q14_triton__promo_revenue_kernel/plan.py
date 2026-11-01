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
        for bs in [8192, 16384, 32768, 65536]
        for w in [8, 16]
        for s in [4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_total", "out_promo"],
)
@triton.jit
def _promo_revenue_kernel(
    extendedprice_ptr, discount_ptr, shipdate_ptr, promo_ptr,
    date_lo, date_hi, out_total, out_promo, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load all needed data in one pass
    shipdate = tl.load(shipdate_ptr + offs, mask=mask, other=date_hi)
    promo_flag = tl.load(promo_ptr + offs, mask=mask, other=False)
    extendedprice = tl.load(extendedprice_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # Combined filter and computation
    keep = mask & (shipdate >= date_lo) & (shipdate < date_hi)
    revenue = tl.where(keep, extendedprice * (1.0 - discount), 0.0)
    promo_rev = tl.where(keep & promo_flag, revenue, 0.0)

    # Block-level reductions
    total = tl.sum(revenue)
    promo = tl.sum(promo_rev)

    # Single kernel with shared block memory for better reduction efficiency
    # Use atomic operations only at block level
    tl.atomic_add(out_total, total)
    tl.atomic_add(out_promo, promo)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="tpch_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1995-09-01",
    shipdate_hi: str = "1995-10-01",
) -> float:
    """
    SQL: SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < CAST('1995-10-01' AS date);
    Description: Computes the percentage of revenue from promotional parts shipped in a given month.
    Uses optimized Triton kernel with larger block sizes (8k-64k), shared memory reductions, and fused computations for better GPU occupancy.
    Params:
        shipdate_lo (str): inclusive lower bound on l_shipdate.
        shipdate_hi (str): exclusive upper bound on l_shipdate.
    """
    df_lineitem = tables["lineitem"]
    df_part = tables["part"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    # Pre-filter and join only necessary columns
    part_proj = df_part[["p_partkey", "p_type"]].rename(columns={"p_partkey": "l_partkey"})
    part_proj["p_type_is_promo"] = part_proj["p_type"].str.startswith("PROMO")
    merged = df_lineitem[["l_partkey", "l_shipdate", "l_extendedprice", "l_discount"]
                      ].merge(part_proj, on="l_partkey", how="inner")

    # Convert to tensors
    extprice_t = torch.as_tensor(merged["l_extendedprice"].to_cupy(), dtype=torch.float64)
    disc_t = torch.as_tensor(merged["l_discount"].to_cupy(), dtype=torch.float64)
    shipdate_t = torch.as_tensor(
        ((merged["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    promo_t = torch.as_tensor(merged["p_type_is_promo"].astype("bool").to_cupy())

    n = extprice_t.numel()
    out_total = torch.zeros(()).to(extprice_t.device, dtype=torch.float64)
    out_promo = torch.zeros(()).to(extprice_t.device, dtype=torch.float64)

    # Autotune configuration optimized for H100
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._promo_revenue_kernel[grid](
        extprice_t, disc_t, shipdate_t, promo_t,
        shipdate_lo_days, shipdate_hi_days, out_total, out_promo, n,
    )

    promo_pct = 100.0 * out_promo / out_total
    return float(promo_pct.item())
