import os
import tempfile
import importlib.util
import linecache
import types

import cudf
import pandas as pd
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _build_late_bitmap_kernel(
    l_orderkey_ptr,
    l_commitdate_days_ptr,
    l_receiptdate_days_ptr,
    bitmap_ptr,
    n_elements,
    bitmap_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    orderkey = tl.load(l_orderkey_ptr + offs, mask=mask, other=-1)
    commitdate = tl.load(l_commitdate_days_ptr + offs, mask=mask, other=0)
    receiptdate = tl.load(l_receiptdate_days_ptr + offs, mask=mask, other=0)

    keep = mask & (commitdate < receiptdate) & (orderkey >= 0) & (orderkey < bitmap_size)
    one = tl.full((BLOCK_SIZE,), 1, tl.uint8)
    tl.store(bitmap_ptr + orderkey, one, mask=keep)


@triton.jit
def _count_orders_kernel(
    o_orderkey_ptr,
    o_orderdate_days_ptr,
    o_priority_codes_ptr,
    bitmap_ptr,
    counts_ptr,
    n_elements,
    bitmap_size,
    date_lo_days,
    date_hi_days,
    BLOCK_SIZE: tl.constexpr,
    MAX_PRIORITIES: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    orderdate = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=date_hi_days)
    date_keep = mask & (orderdate >= date_lo_days) & (orderdate < date_hi_days)

    orderkey = tl.load(o_orderkey_ptr + offs, mask=date_keep, other=-1)
    key_ok = date_keep & (orderkey >= 0) & (orderkey < bitmap_size)

    late_flag = tl.load(bitmap_ptr + orderkey, mask=key_ok, other=0)
    exists_keep = key_ok & (late_flag != 0)

    prio = tl.load(o_priority_codes_ptr + offs, mask=exists_keep, other=-1)

    for g in tl.static_range(0, MAX_PRIORITIES):
        inc = tl.where(exists_keep & (prio == g), 1, 0)
        cnt = tl.sum(inc, axis=0).to(tl.int64)
        tl.atomic_add(counts_ptr + g, cnt, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton JIT kernels to a real Python file and import them for validator-safe compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q4_triton_")
    path = os.path.join(tmpdir, "_q4_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q4_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _datetime_days_tensor(series: cudf.Series) -> torch.Tensor:
    """
    Convert a cuDF datetime64[s] column to an int32 GPU tensor of epoch days.

    cuDF performs the datetime cast/divide on GPU; Triton kernels consume the compact int32
    representation so date predicates read half the bytes of int64 timestamps.
    """
    return torch.as_tensor(((series.astype("int64") // 86400).astype("int32")).to_cupy(), dtype=torch.int32)


def _launch_build_late_bitmap(
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    bitmap_size: int,
    block_size: int,
) -> torch.Tensor:
    """
    Build a GPU byte bitmap of orderkeys having at least one late lineitem.

    This is a single Triton pass over lineitem that fuses the l_commitdate < l_receiptdate
    predicate with bitmap materialization, avoiding torch.unique/torch.isin temporaries.
    """
    bitmap = torch.zeros((max(bitmap_size, 1),), dtype=torch.uint8, device=l_orderkey.device)
    n_elements = int(l_orderkey.numel())
    if n_elements == 0 or bitmap_size <= 0:
        return bitmap

    grid = (triton.cdiv(n_elements, block_size),)
    _K._build_late_bitmap_kernel[grid](
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        bitmap,
        n_elements,
        bitmap_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return bitmap


def _launch_count_orders(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    late_bitmap: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
    block_size: int,
) -> torch.Tensor:
    """
    Count qualifying orders per priority with one fused Triton scan over orders.

    The kernel filters by date, probes the late-order bitmap only for date-matching rows,
    conditionally loads priority codes only for rows satisfying EXISTS, and emits one
    atomic add per priority per block.
    """
    counts = torch.zeros((num_priorities,), dtype=torch.int64, device=o_orderkey.device)
    n_elements = int(o_orderkey.numel())
    if n_elements == 0 or num_priorities <= 0:
        return counts

    grid = (triton.cdiv(n_elements, block_size),)
    _K._count_orders_kernel[grid](
        o_orderkey,
        o_orderdate_days,
        o_orderpriority_codes,
        late_bitmap,
        counts,
        n_elements,
        int(late_bitmap.numel()),
        date_lo_days,
        date_hi_days,
        BLOCK_SIZE=block_size,
        MAX_PRIORITIES=num_priorities,
        num_warps=8,
    )
    return counts


def _query_core(
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_orderpriority_codes: torch.Tensor,
    l_orderkey: torch.Tensor,
    l_commitdate_days: torch.Tensor,
    l_receiptdate_days: torch.Tensor,
    date_lo_days: int,
    date_hi_days: int,
    num_priorities: int,
    bitmap_size: int | None = None,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute the numeric hot path on GPU with Triton.

    Compared with the reference eager PyTorch path, this replaces lineitem filtering,
    unique/is-in existence testing, order filtering, and small-group counting with two
    memory-bandwidth-oriented Triton passes and a compact byte bitmap.
    """
    if num_priorities <= 0:
        device = o_orderkey.device
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

    if bitmap_size is None:
        if int(l_orderkey.numel()) == 0:
            bitmap_size = 1
        else:
            bitmap_size = int(torch.max(l_orderkey).item()) + 1

    late_bitmap = _launch_build_late_bitmap(
        l_orderkey,
        l_commitdate_days,
        l_receiptdate_days,
        bitmap_size,
        block_size,
    )
    counts = _launch_count_orders(
        o_orderkey,
        o_orderdate_days,
        o_orderpriority_codes,
        late_bitmap,
        date_lo_days,
        date_hi_days,
        num_priorities,
        block_size,
    )

    priority_ids = torch.arange(num_priorities, device=counts.device, dtype=torch.int64)
    nz = counts > 0
    return priority_ids[nz], counts[nz]


def run_query(
    tables: dict[str, cudf.DataFrame],
    date_lo: str = "1993-07-01",
    date_hi: str = "1993-10-01",
) -> cudf.DataFrame:
    """
    SQL:
        SELECT
            o_orderpriority,
            count(*) AS order_count
        FROM
            orders
        WHERE
            o_orderdate >= CAST('1993-07-01' AS date)
            AND o_orderdate < CAST('1993-10-01' AS date)
            AND EXISTS (
                SELECT *
                FROM lineitem
                WHERE l_orderkey = o_orderkey
                  AND l_commitdate < l_receiptdate)
        GROUP BY
            o_orderpriority
        ORDER BY
            o_orderpriority;

    Description:
        TPC-H Q4. cuDF handles table access, datetime-day conversion, and the small string
        category mapping for o_orderpriority. The bulk numeric work runs on GPU in Triton:
        one fused lineitem scan builds a late-orderkey bitmap, and one fused orders scan
        applies the date predicate, probes EXISTS, and counts fixed priority groups. This
        avoids the reference torch.unique/torch.isin pipeline and reduces large temporary
        allocations and HBM passes.

    Params:
        tables (dict[str, cudf.DataFrame]): Input TPC-H tables containing "orders" and
            "lineitem".
        date_lo (str): Inclusive lower bound for o_orderdate.
        date_hi (str): Exclusive upper bound for o_orderdate.
    """
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    date_lo_days = int(pd.to_datetime(date_lo).timestamp() // 86400)
    date_hi_days = int(pd.to_datetime(date_hi).timestamp() // 86400)

    priority_cat = df_orders["o_orderpriority"].astype("category")
    priority_codes = priority_cat.cat.codes
    priority_categories = priority_cat.cat.categories.to_pandas().tolist()
    num_priorities = len(priority_categories)

    o_orderkey_t = torch.as_tensor(df_orders["o_orderkey"].to_cupy(), dtype=torch.int64)
    o_orderdate_days_t = _datetime_days_tensor(df_orders["o_orderdate"])
    o_priority_codes_t = torch.as_tensor(priority_codes.to_cupy(), dtype=torch.int32)

    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_commitdate_days_t = _datetime_days_tensor(df_lineitem["l_commitdate"])
    l_receiptdate_days_t = _datetime_days_tensor(df_lineitem["l_receiptdate"])

    if len(df_lineitem) == 0:
        bitmap_size = 1
    else:
        bitmap_size = int(df_lineitem["l_orderkey"].max()) + 1

    priority_ids_t, counts_t = _query_core(
        o_orderkey_t,
        o_orderdate_days_t,
        o_priority_codes_t,
        l_orderkey_t,
        l_commitdate_days_t,
        l_receiptdate_days_t,
        date_lo_days,
        date_hi_days,
        num_priorities,
        bitmap_size=bitmap_size,
        block_size=1024,
    )

    priority_ids_host = priority_ids_t.cpu().tolist()
    priority_strs = [priority_categories[int(i)] for i in priority_ids_host]

    result = cudf.DataFrame(
        {
            "o_orderpriority": cudf.Series(priority_strs, dtype="object"),
            "order_count": counts_t,
        }
    )
    result = result.sort_values("o_orderpriority").reset_index(drop=True)
    return result
