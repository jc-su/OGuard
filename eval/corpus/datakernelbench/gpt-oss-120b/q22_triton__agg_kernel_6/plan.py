import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton  # needed for triton.cdiv in host code

# ----------------------------------------------------------------------
# Triton kernels (must be written to a temporary .py file before JIT)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Kernel 1: average of c_acctbal for rows where mask == 1
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_sum", "out_cnt"],
)
@triton.jit
def _avg_kernel(
    bal_ptr,            # *float64   balance column
    mask_ptr,           # *uint8     1 where row participates in avg
    out_sum,            # *float64   scalar sum (global)
    out_cnt,            # *float64   scalar count (global)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load mask (0/1) as uint8
    m = tl.load(mask_ptr + offs, mask=mask, other=0)
    keep = mask & (m != 0)

    # load balance where keep, else 0.0
    bal = tl.load(bal_ptr + offs, mask=keep, other=0.0)

    # atomic reductions
    tl.atomic_add(out_sum, tl.sum(bal))
    tl.atomic_add(out_cnt, tl.sum(tl.where(keep, 1.0, 0.0)))


# ------------------------------------------------------------------
# Kernel 2: grouped aggregation (count + sum) per country code
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["out_cnt", "out_sum"],
)
@triton.jit
def _agg_kernel(
    bal_ptr,            # *float64   c_acctbal
    code_ptr,           # *int32    cntrycode as int (e.g. 13, 31 ...)
    mask_ptr,           # *uint8    1 where row passes final filter
    out_cnt,            # *float64  per‑code count array (size = MAX_CODE+1)
    out_sum,            # *float64  per‑code sum array (size = MAX_CODE+1)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    m = tl.load(mask_ptr + offs, mask=mask, other=0)
    keep = mask & (m != 0)

    # load only for kept rows
    code = tl.load(code_ptr + offs, mask=keep, other=0)
    bal  = tl.load(bal_ptr + offs, mask=keep, other=0.0)

    # atomic updates per country code (code values < 100)
    tl.atomic_add(out_cnt + code, tl.where(keep, 1.0, 0.0))
    tl.atomic_add(out_sum + code, tl.where(keep, bal, 0.0))
'''

def _load_kernels():
    """Write the Triton source to a temp file and import it as a module."""
    tmpdir = tempfile.mkdtemp(prefix="q22_triton_")
    path = os.path.join(tmpdir, "_q22_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q22_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()   # provides _K._avg_kernel and _K._agg_kernel


def _launch_avg(balance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute average of `balance` where `mask` == 1 using a fused Triton kernel.
    Returns a scalar torch.float64 tensor.
    """
    n = balance.numel()
    out_sum = torch.zeros(1, dtype=torch.float64, device=balance.device)
    out_cnt = torch.zeros(1, dtype=torch.float64, device=balance.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._avg_kernel[grid](
        balance,
        mask,
        out_sum,
        out_cnt,
        n,
    )
    torch.cuda.synchronize()
    return out_sum[0] / out_cnt[0]


def _launch_agg(
    balance: torch.Tensor,
    code: torch.Tensor,
    mask: torch.Tensor,
    max_code: int = 99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused per‑code count and sum.
    Returns tuple (counts_int64, sums_float64) each of length max_code+1.
    Unused entries stay zero.
    """
    n = balance.numel()
    out_cnt = torch.zeros(max_code + 1, dtype=torch.float64, device=balance.device)
    out_sum = torch.zeros(max_code + 1, dtype=torch.float64, device=balance.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _K._agg_kernel[grid](
        balance,
        code,
        mask,
        out_cnt,
        out_sum,
        n,
    )
    torch.cuda.synchronize()
    return out_cnt.to(torch.int64), out_sum


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Core GPU logic.
    Returns (codes, numcust, totacctbal):
        codes      – int32 tensor of distinct country codes (sorted)
        numcust    – int64 tensor count per code
        totacctbal – float64 tensor sum per code
    """
    # Step 1: average balance for rows with mask & positive balance
    mask_avg = in_cntrycode_mask & (c_acctbal > 0.0)
    avg_val = _launch_avg(c_acctbal, mask_avg.to(torch.uint8))

    # Step 2: NOT EXISTS – does custkey appear in orders?
    sorted_okeys, _ = torch.sort(o_custkey_unique)
    idx = torch.searchsorted(sorted_okeys, c_custkey)
    idx_clamped = torch.clamp(idx, max=sorted_okeys.numel() - 1)
    has_order = (idx < sorted_okeys.numel()) & (sorted_okeys[idx_clamped] == c_custkey)
    no_order = ~has_order

    # Step 3: final filter
    final_mask = in_cntrycode_mask & (c_acctbal > avg_val) & no_order

    # Step 4: fused aggregation per country code
    cnt_arr, sum_arr = _launch_agg(
        c_acctbal,
        cntrycode,
        final_mask.to(torch.uint8),
    )

    # Step 5: extract present codes, sort them, and slice results
    present_codes = cnt_arr.nonzero().squeeze(1)            # int64 indices
    sorted_codes, order = torch.sort(present_codes.to(torch.int32))
    numcust = cnt_arr[sorted_codes]
    totacctbal = sum_arr[sorted_codes]

    return sorted_codes, numcust, totacctbal


def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ('13', '31', '23', '29', '30', '18', '17'),
) -> cudf.DataFrame:
    """
    SQL:
    SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
    FROM (
        SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode,
               c_acctbal
        FROM customer
        WHERE substring(c_phone FROM 1 FOR 2) IN (...)
          AND c_acctbal > (SELECT avg(c_acctbal) FROM customer
                           WHERE c_acctbal > 0.00
                             AND substring(c_phone FROM 1 FOR 2) IN (...))
          AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
    ) AS custsale
    GROUP BY cntrycode
    ORDER BY cntrycode;

    Description:
        • cuDF handles inexpensive string slicing and the small set‑membership mask.
        • All heavy numeric work (average, NOT‑EXISTS filter, final aggregation)
          runs on the GPU inside two fused Triton kernels, collapsing the scan
          to a single pass over the `customer` columns.
        • Per‑code aggregates use atomics on a tiny 100‑element array,
          avoiding an additional group‑by pass.

    Params:
        cntrycodes (tuple[str, ...]): target two‑digit country codes.

    Returns:
        cudf.DataFrame with columns:
            cntrycode (str, zero‑padded, sorted),
            numcust   (int64),
            totacctbal (float64)
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    # ------------------------------------------------------------------
    # Prepare GPU tensors
    # ------------------------------------------------------------------
    # Extract first two chars of c_phone as int32
    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)

    # Build mask for target country codes
    codes_int = [int(c) for c in cntrycodes]
    in_mask_series = cntry_series.isin(codes_int)
    in_mask_t = torch.as_tensor(in_mask_series.to_cupy(), dtype=torch.bool)

    # Numeric columns
    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)

    # Unique order customer keys for NOT EXISTS
    o_custkey_unique_series = df_orders["o_custkey"].unique()
    o_custkey_t = torch.as_tensor(o_custkey_unique_series.to_cupy(), dtype=torch.int64)

    # ------------------------------------------------------------------
    # Run fused GPU core
    # ------------------------------------------------------------------
    codes, numcust, totacctbal = _query_core(
        c_custkey_t,
        c_acctbal_t,
        cntrycode_t,
        in_mask_t,
        o_custkey_t,
    )

    # ------------------------------------------------------------------
    # Build final cudf.DataFrame (zero‑pad country codes)
    # ------------------------------------------------------------------
    codes_cudf = cudf.Series(codes).astype("int32").astype("str").str.zfill(2)

    return cudf.DataFrame({
        "cntrycode": codes_cudf,
        "numcust":   cudf.Series(numcust),
        "totacctbal": cudf.Series(totacctbal),
    })
