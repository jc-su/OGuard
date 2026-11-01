import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
from typing import Tuple

# -----------------------------------------------------------------------------
# Triton kernel source: fused filter + group‑by aggregation
# -----------------------------------------------------------------------------
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
    reset_to_zero=["count_ptr", "sum_ptr"],
)
@triton.jit
def _group_agg_kernel(
    cntrycode_ptr,
    acctbal_ptr,
    outer_mask_ptr,
    count_ptr,
    sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: for each row that passes outer_mask, atomically add
    its balance to the per‑country‑code count and sum."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    cntry = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    acctbal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    outer = tl.load(outer_mask_ptr + offs, mask=mask, other=False)

    keep = outer
    code = cntry

    # atomic add for count (int64)
    count_add = tl.where(keep, 1, 0)
    tl.atomic_add(count_ptr + code, count_add.to(tl.int64))

    # atomic add for sum (float64)
    sum_add = tl.where(keep, acctbal, 0.0)
    tl.atomic_add(sum_ptr + code, sum_add)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q22_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core GPU computation for TPC‑H Q22.
    Returns (cntrycode, numcust, totacctbal) sorted by cntrycode."""
    # ------------------------------------------------------------------
    # 1. average balance for target country codes with positive balance
    # ------------------------------------------------------------------
    avg_mask = in_cntrycode_mask & (c_acctbal > 0.0)
    avg_sum = torch.sum(c_acctbal[avg_mask])
    avg_count = torch.sum(avg_mask.to(torch.float64))
    avg_val = avg_sum / avg_count

    # ------------------------------------------------------------------
    # 2. NOT EXISTS: customers without any order
    # ------------------------------------------------------------------
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    idx = torch.searchsorted(sorted_okeys, c_custkey)
    idx_clamped = torch.clamp(idx, max=sorted_okeys.numel() - 1)
    has_order = (idx < sorted_okeys.numel()) & (sorted_okeys[idx_clamped] == c_custkey)
    no_order = ~has_order

    # ------------------------------------------------------------------
    # 3. outer filter: target country, above‑average balance, no order
    # ------------------------------------------------------------------
    outer_mask = in_cntrycode_mask & (c_acctbal > avg_val) & no_order

    # ------------------------------------------------------------------
    # 4. fused group‑by aggregation using Triton kernel
    # ------------------------------------------------------------------
    n = c_acctbal.numel()
    CODE_MAX = 100  # cntrycode is a 2‑digit integer (0‑99)
    counts = torch.zeros(CODE_MAX, dtype=torch.int64, device=c_custkey.device)
    sums = torch.zeros(CODE_MAX, dtype=torch.float64, device=c_acctbal.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _K._group_agg_kernel[grid](
        cntrycode, c_acctbal, outer_mask, counts, sums, n
    )

    # ------------------------------------------------------------------
    # 5. collect non‑zero groups and sort by country code
    # ------------------------------------------------------------------
    mask = counts > 0
    codes = torch.arange(CODE_MAX, dtype=torch.int32, device=c_custkey.device)[mask]
    numcust = counts[mask]
    totacctbal = sums[mask]

    order = torch.argsort(codes)
    codes = codes[order]
    numcust = numcust[order]
    totacctbal = totacctbal[order]

    return codes, numcust, totacctbal


def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17'),
) -> cudf.DataFrame:
    """
    SQL: SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
         FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
               FROM customer
               WHERE substring(c_phone FROM 1 FOR 2) IN (...)
                 AND c_acctbal > (SELECT avg(c_acctbal) FROM customer
                                  WHERE c_acctbal > 0.00
                                    AND substring(c_phone FROM 1 FOR 2) IN (...))
                 AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
              ) AS custsale
         GROUP BY cntrycode ORDER BY cntrycode;
    Description: TPC-H Q22 - Global sales opportunity query. Identifies customers in
                 target country codes with above-average balances who have placed no orders,
                 grouped and aggregated by country code.
    Params: cntrycodes (tuple[str,...]): target 2-digit country codes.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # Derive cntrycode: first 2 chars of c_phone, convert to int (codes are numeric strings)
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    # Build in-set mask on GPU
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)

    # Unique o_custkey from orders (for NOT EXISTS)
    o_custkey_unique_series = df_orders["o_custkey"].unique()

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t, c_acctbal_t, cntrycode_t, in_mask_t, o_custkey_t
    )

    # Format cntrycode back to 2-char zero-padded string
    codes_cudf = cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust": cudf.Series(numcust_t),
        "totacctbal": cudf.Series(totacctbal_t),
    })
