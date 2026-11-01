import os
import tempfile
import importlib.util
import linecache
from typing import Any, Tuple

import cudf
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _build_order_bitmap_kernel(
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
    tl.store(bitmap_ptr + key, 1, mask=valid)


@triton.jit
def _avg_acctbal_kernel(
    cntrycode_ptr,
    acctbal_ptr,
    codes_ptr,
    out_sum_count_ptr,
    n_customers,
    N_CODES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    cntry = tl.load(cntrycode_ptr + offs, mask=mask, other=-999999)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=-1.0)

    in_set = tl.full((BLOCK_SIZE,), False, tl.int1)
    for i in tl.static_range(0, N_CODES):
        code = tl.load(codes_ptr + i)
        in_set = in_set | (cntry == code)

    keep = mask & in_set & (bal > 0.0)

    zf = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    of = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    part_sum = tl.sum(tl.where(keep, bal, zf))
    part_cnt = tl.sum(tl.where(keep, of, zf))

    tl.atomic_add(out_sum_count_ptr + 0, part_sum)
    tl.atomic_add(out_sum_count_ptr + 1, part_cnt)


@triton.jit
def _final_group_kernel(
    c_custkey_ptr,
    cntrycode_ptr,
    acctbal_ptr,
    bitmap_ptr,
    codes_ptr,
    avg_sum_count_ptr,
    out_count_f64_ptr,
    out_sum_ptr,
    n_customers,
    max_key,
    N_CODES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    avg_sum = tl.load(avg_sum_count_ptr + 0)
    avg_cnt = tl.load(avg_sum_count_ptr + 1)
    avg_val = avg_sum / avg_cnt

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_customers

    custkey = tl.load(c_custkey_ptr + offs, mask=mask, other=-1)
    cntry = tl.load(cntrycode_ptr + offs, mask=mask, other=-999999)
    bal = tl.load(acctbal_ptr + offs, mask=mask, other=0.0)

    key_valid = mask & (custkey >= 0) & (custkey <= max_key)
    has_order = tl.load(bitmap_ptr + custkey, mask=key_valid, other=1)
    base_keep = mask & key_valid & (has_order == 0) & (bal > avg_val)

    zf = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    of = tl.full((BLOCK_SIZE,), 1.0, tl.float64)

    for i in tl.static_range(0, N_CODES):
        code = tl.load(codes_ptr + i)
        keep = base_keep & (cntry == code)
        part_cnt = tl.sum(tl.where(keep, of, zf))
        part_sum = tl.sum(tl.where(keep, bal, zf))
        tl.atomic_add(out_count_f64_ptr + i, part_cnt)
        tl.atomic_add(out_sum_ptr + i, part_sum)
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q22_triton_")
    path = os.path.join(tmpdir, "_q22_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q22_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _query_core(
    c_custkey: torch.Tensor,
    c_acctbal: torch.Tensor,
    cntrycode: torch.Tensor,
    o_custkey: torch.Tensor,
    code_values: torch.Tensor,
    block_size_orders: int = 1024,
    block_size_customer: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU/Triton core for TPC-H Q22.

    Runs the large-table hot path on the GPU with Triton:
    1. scans orders once to build a customer-key presence bitmap,
    2. scans customer once to compute the positive-balance in-country average,
    3. scans customer once more to apply NOT EXISTS + balance threshold and aggregate
       directly into the small fixed country-code groups.

    This avoids the reference Torch sort/searchsorted/unique/scatter pipeline and reduces
    temporary tensors; the work is memory-bound, so fused scans and a bitmap lookup are
    faster than materializing masks and sorting order keys.
    """
    n_customers = int(c_custkey.numel())
    n_orders = int(o_custkey.numel())
    n_codes = int(code_values.numel())
    device = c_acctbal.device

    if n_codes == 0 or n_customers == 0:
        empty_codes = torch.empty((0,), dtype=torch.int32, device=device)
        empty_counts = torch.empty((0,), dtype=torch.int64, device=device)
        empty_sums = torch.empty((0,), dtype=torch.float64, device=device)
        return empty_codes, empty_counts, empty_sums

    max_key = int(torch.max(c_custkey).item())
    bitmap = torch.zeros((max_key + 1,), dtype=torch.uint8, device=device)

    if n_orders > 0:
        grid_orders = (triton.cdiv(n_orders, block_size_orders),)
        _K._build_order_bitmap_kernel[grid_orders](
            o_custkey,
            bitmap,
            n_orders,
            max_key,
            BLOCK_SIZE=block_size_orders,
        )

    avg_sum_count = torch.zeros((2,), dtype=torch.float64, device=device)
    grid_customers = (triton.cdiv(n_customers, block_size_customer),)
    _K._avg_acctbal_kernel[grid_customers](
        cntrycode,
        c_acctbal,
        code_values,
        avg_sum_count,
        n_customers,
        N_CODES=n_codes,
        BLOCK_SIZE=block_size_customer,
    )

    count_f64 = torch.zeros((n_codes,), dtype=torch.float64, device=device)
    sum_f64 = torch.zeros((n_codes,), dtype=torch.float64, device=device)
    _K._final_group_kernel[grid_customers](
        c_custkey,
        cntrycode,
        c_acctbal,
        bitmap,
        code_values,
        avg_sum_count,
        count_f64,
        sum_f64,
        n_customers,
        max_key,
        N_CODES=n_codes,
        BLOCK_SIZE=block_size_customer,
    )

    count_i64 = count_f64.to(torch.int64)
    return code_values, count_i64, sum_f64


def run_query(
    tables: dict[str, cudf.DataFrame],
    cntrycodes: tuple[str, ...] = ("13", "31", "23", "29", "30", "18", "17"),
) -> cudf.DataFrame:
    """
    SQL:
        SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal
        FROM (
            SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
            FROM customer
            WHERE substring(c_phone FROM 1 FOR 2) IN (...)
              AND c_acctbal > (
                    SELECT avg(c_acctbal)
                    FROM customer
                    WHERE c_acctbal > 0.00
                      AND substring(c_phone FROM 1 FOR 2) IN (...))
              AND NOT EXISTS (
                    SELECT *
                    FROM orders
                    WHERE o_custkey = c_custkey)
        ) AS custsale
        GROUP BY cntrycode
        ORDER BY cntrycode;

    Description:
        TPC-H Q22 global sales opportunity query. cuDF is used only for string
        extraction of the first two phone digits and final DataFrame formatting.
        The hot path runs on the GPU in Triton: orders are converted to a compact
        customer-key bitmap, customer rows are scanned to compute the average, and
        a fused customer scan performs NOT EXISTS, threshold filtering, and fixed
        small-group aggregation. This replaces Torch sort/searchsorted and multiple
        mask/materialization steps with coalesced memory-bound scans and minimal
        temporaries.

    Params:
        tables: dict[str, cudf.DataFrame]
            Must contain "customer" and "orders".
        cntrycodes: tuple[str, ...]
            Target two-digit country codes. Duplicates are ignored, matching SQL IN
            set semantics; output is ordered by numeric country code and formatted
            as zero-padded strings.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]

    codes_int = sorted({int(c) for c in cntrycodes})
    if len(codes_int) == 0:
        return cudf.DataFrame(
            {
                "cntrycode": cudf.Series([], dtype="object"),
                "numcust": cudf.Series([], dtype="int64"),
                "totacctbal": cudf.Series([], dtype="float64"),
            }
        )

    cntry_series = df_customer["c_phone"].str.slice(0, 2).astype("int32")

    c_custkey_t = torch.as_tensor(df_customer["c_custkey"].to_cupy(), dtype=torch.int64)
    c_acctbal_t = torch.as_tensor(df_customer["c_acctbal"].to_cupy(), dtype=torch.float64)
    cntrycode_t = torch.as_tensor(cntry_series.to_cupy(), dtype=torch.int32)
    o_custkey_t = torch.as_tensor(df_orders["o_custkey"].to_cupy(), dtype=torch.int64)
    code_values_t = torch.tensor(codes_int, dtype=torch.int32, device=c_acctbal_t.device)

    codes_t, numcust_t, totacctbal_t = _query_core(
        c_custkey_t,
        c_acctbal_t,
        cntrycode_t,
        o_custkey_t,
        code_values_t,
    )

    result = cudf.DataFrame(
        {
            "cntrycode": cudf.Series(codes_t).astype("int32").astype("str").str.zfill(2),
            "numcust": cudf.Series(numcust_t),
            "totacctbal": cudf.Series(totacctbal_t),
        }
    )
    result = result[result["numcust"] > 0]
    return result.reset_index(drop=True)
