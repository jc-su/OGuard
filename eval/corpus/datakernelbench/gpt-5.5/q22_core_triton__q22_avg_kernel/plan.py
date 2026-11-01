import os
import tempfile
import importlib.util
import linecache
from typing import Any, Tuple

import torch
import triton
import cudf


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q22_avg_kernel(
    acctbal_ptr,
    in_mask_ptr,
    stats_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=0) != 0
    keep = mask & in_set & (bal > 0.0)

    part_sum = tl.sum(tl.where(keep, bal, 0.0))
    part_cnt = tl.sum(keep.to(tl.float64))

    tl.atomic_add(stats_ptr + 0, part_sum, sem="relaxed")
    tl.atomic_add(stats_ptr + 1, part_cnt, sem="relaxed")


@triton.jit
def _q22_mark_orders_kernel(
    o_custkey_ptr,
    bitmap_ptr,
    n_orders,
    max_key,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders

    key = tl.load(o_custkey_ptr + offs, mask=mask, other=-1)
    valid = mask & (key >= 0) & (key <= max_key)
    safe_key = tl.where(valid, key, 0)
    tl.store(bitmap_ptr + safe_key, 1, mask=valid)


@triton.jit
def _q22_agg_kernel(
    c_custkey_ptr,
    acctbal_ptr,
    cntrycode_ptr,
    in_mask_ptr,
    bitmap_ptr,
    stats_ptr,
    counts_ptr,
    sums_ptr,
    n_elements,
    max_key,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    sum_pos = tl.load(stats_ptr + 0)
    cnt_pos = tl.load(stats_ptr + 1)
    avg_bal = sum_pos / cnt_pos

    key = tl.load(c_custkey_ptr + offs, mask=mask, other=-1)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)
    code = tl.load(cntrycode_ptr + offs, mask=mask, other=0)
    in_set = tl.load(in_mask_ptr + offs, mask=mask, other=0) != 0

    valid_key = mask & (key >= 0) & (key <= max_key)
    safe_key = tl.where(valid_key, key, 0)
    has_order = tl.load(bitmap_ptr + safe_key, mask=valid_key, other=0) != 0

    valid_code = (code >= 0) & (code < 100)
    keep = mask & in_set & valid_code & (bal > avg_bal) & (~has_order)

    one = tl.full((BLOCK_SIZE,), 1, tl.int64)
    tl.atomic_add(counts_ptr + code, one, sem="relaxed", mask=keep)
    tl.atomic_add(sums_ptr + code, bal, sem="relaxed", mask=keep)


@triton.jit
def _q22_extract_kernel(
    counts_ptr,
    sums_ptr,
    codes_out_ptr,
    counts_out_ptr,
    sums_out_ptr,
    num_groups_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    in_range = offs < 100

    cnt = tl.load(counts_ptr + offs, mask=in_range, other=0)
    sm = tl.load(sums_ptr + offs, mask=in_range, other=0.0)
    valid = in_range & (cnt > 0)

    valid_i = valid.to(tl.int32)
    prefix = tl.cumsum(valid_i, 0) - 1
    pos = tl.where(valid, prefix, 0)

    tl.store(codes_out_ptr + pos, offs.to(tl.int32), mask=valid)
    tl.store(counts_out_ptr + pos, cnt, mask=valid)
    tl.store(sums_out_ptr + pos, sm, mask=valid)

    total = tl.sum(valid_i, axis=0)
    tl.store(num_groups_ptr, total)
'''


def _load_kernels() -> Any:
    tmpdir = tempfile.mkdtemp(prefix="q22_triton_")
    path = os.path.join(tmpdir, "_q22_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q22_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create Triton kernel module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K: Any = _load_kernels()


def _launch_q22(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
    block_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs the numeric Q22 hot path on the GPU with Triton kernels.

    The customer table is scanned once to compute the positive-balance average, the
    order keys are materialized into a GPU bitmap, and the final customer filter plus
    fixed 0..99 country-code aggregation is fused into one Triton pass. This avoids
    the reference's GPU sort/searchsorted, boolean materializations, unique, and
    scatter pipeline, which is advantageous for this memory-bound scan.
    """
    n_customers = c_custkey.numel()
    device = c_custkey.device

    if n_customers == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
        )

    max_key = int(n_customers)
    stats = torch.zeros(2, dtype=torch.float64, device=device)
    bitmap = torch.zeros(max_key + 1, dtype=torch.uint8, device=device)
    counts = torch.zeros(100, dtype=torch.int64, device=device)
    sums = torch.zeros(100, dtype=torch.float64, device=device)

    grid_customers = (triton.cdiv(n_customers, block_size),)
    _K._q22_avg_kernel[grid_customers](
        c_acctbal,
        in_cntrycode_mask,
        stats,
        n_customers,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    n_orders = o_custkey_unique.numel()
    if n_orders > 0:
        grid_orders = (triton.cdiv(n_orders, block_size),)
        _K._q22_mark_orders_kernel[grid_orders](
            o_custkey_unique,
            bitmap,
            n_orders,
            max_key,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )

    _K._q22_agg_kernel[grid_customers](
        c_custkey,
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        bitmap,
        stats,
        counts,
        sums,
        n_customers,
        max_key,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    codes_out = torch.empty(100, dtype=torch.int32, device=device)
    counts_out = torch.empty(100, dtype=torch.int64, device=device)
    sums_out = torch.empty(100, dtype=torch.float64, device=device)
    num_groups = torch.empty(1, dtype=torch.int32, device=device)

    _K._q22_extract_kernel[(1,)](
        counts,
        sums,
        codes_out,
        counts_out,
        sums_out,
        num_groups,
        BLOCK_SIZE=128,
        num_warps=4,
    )

    m = int(num_groups.item())
    return codes_out[:m], counts_out[:m], sums_out[:m]


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    in_cntrycode_mask: torch.Tensor,
    o_custkey_unique: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes Q22 tensor compute on the GPU using Triton.

    cuDF/string preparation remains in run_query. This function replaces the eager
    PyTorch hot path with Triton kernels: a fused average scan, an order-key bitmap
    build for NOT EXISTS, a fused outer filter plus small fixed country aggregation,
    and a tiny ordered extraction. The query is mostly memory-bound, so eliminating
    sort/searchsorted intermediates and reducing customer scans/temporaries should
    improve runtime while preserving output dtypes and ordering.
    """
    return _launch_q22(
        c_custkey,
        c_acctbal,
        cntrycode,
        in_cntrycode_mask,
        o_custkey_unique,
    )


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
