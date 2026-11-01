import os
import tempfile
import importlib.util
import linecache
from typing import Any

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _mark_orders_direct_kernel(
    o_orderkey_ptr,
    o_orderdate_days_ptr,
    o_shippriority_ptr,
    order_flag_ptr,
    orderdate_table_ptr,
    shippriority_table_ptr,
    n_orders,
    capacity,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_orders

    key = tl.load(o_orderkey_ptr + offs, mask=mask, other=-1)
    ok = mask & (key >= 0) & (key < capacity)

    od = tl.load(o_orderdate_days_ptr + offs, mask=mask, other=0)
    sp = tl.load(o_shippriority_ptr + offs, mask=mask, other=0)

    tl.store(order_flag_ptr + key, 1, mask=ok)
    tl.store(orderdate_table_ptr + key, od, mask=ok)
    tl.store(shippriority_table_ptr + key, sp, mask=ok)


@triton.jit
def _scan_lineitem_direct_kernel(
    l_orderkey_ptr,
    l_extendedprice_ptr,
    l_discount_ptr,
    l_shipdate_days_ptr,
    order_flag_ptr,
    revenue_table_ptr,
    n_lineitems,
    capacity,
    shipdate_lo_days,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_lineitems

    shipdate = tl.load(l_shipdate_days_ptr + offs, mask=mask, other=shipdate_lo_days)
    ship_ok = mask & (shipdate > shipdate_lo_days)

    key = tl.load(l_orderkey_ptr + offs, mask=ship_ok, other=0)
    in_range = ship_ok & (key >= 0) & (key < capacity)

    is_eligible_order = tl.load(order_flag_ptr + key, mask=in_range, other=0)
    keep = in_range & (is_eligible_order != 0)

    price = tl.load(l_extendedprice_ptr + offs, mask=keep, other=0.0)
    discount = tl.load(l_discount_ptr + offs, mask=keep, other=0.0)
    revenue = price * (1.0 - discount)

    tl.atomic_add(revenue_table_ptr + key, revenue, mask=keep, sem="relaxed")
'''


def _load_kernels() -> Any:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q3_triton_")
    path = os.path.join(tmpdir, "_q3_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q3_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import specification for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _empty_result() -> cudf.DataFrame:
    """Create an empty cuDF result with the exact SQL output columns and dtypes."""
    return cudf.DataFrame(
        {
            "l_orderkey": cudf.Series([], dtype="int64"),
            "revenue": cudf.Series([], dtype="float64"),
            "o_orderdate": cudf.Series([], dtype="datetime64[s]"),
            "o_shippriority": cudf.Series([], dtype="int32"),
        }
    )


def _launch_direct_aggregate(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    capacity: int,
    mark_block_size: int = 256,
    scan_block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU/Triton path: build a direct-address eligible-order table, then scan lineitem
    once with a fused shipdate filter, order lookup, revenue projection, and FP64
    atomic aggregation. This avoids materializing the large orders-lineitem join and
    avoids the reference Torch multi-column unique/scatter pipeline.
    """
    device = l_orderkey.device
    order_flag = torch.zeros((capacity,), dtype=torch.uint8, device=device)
    orderdate_table = torch.empty((capacity,), dtype=torch.int32, device=device)
    shippriority_table = torch.empty((capacity,), dtype=torch.int32, device=device)
    revenue_table = torch.zeros((capacity,), dtype=torch.float64, device=device)

    n_orders = int(o_orderkey.numel())
    if n_orders > 0:
        grid_mark = (triton.cdiv(n_orders, mark_block_size),)
        _K._mark_orders_direct_kernel[grid_mark](
            o_orderkey,
            o_orderdate_days,
            o_shippriority,
            order_flag,
            orderdate_table,
            shippriority_table,
            n_orders,
            int(capacity),
            BLOCK_SIZE=mark_block_size,
        )

    n_lineitems = int(l_orderkey.numel())
    if n_lineitems > 0:
        grid_scan = (triton.cdiv(n_lineitems, scan_block_size),)
        _K._scan_lineitem_direct_kernel[grid_scan](
            l_orderkey,
            l_extendedprice,
            l_discount,
            l_shipdate_days,
            order_flag,
            revenue_table,
            n_lineitems,
            int(capacity),
            int(shipdate_lo_days),
            BLOCK_SIZE=scan_block_size,
        )

    return order_flag, revenue_table, orderdate_table, shippriority_table


def _select_top(
    order_flag: torch.Tensor,
    revenue_table: torch.Tensor,
    orderdate_table: torch.Tensor,
    shippriority_table: torch.Tensor,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU tensor post-processing: compact orders with positive aggregated revenue and
    perform the SQL ORDER BY revenue DESC, o_orderdate ASC LIMIT top_n. The expensive
    scan/aggregation has already run in Triton; this final sort is over eligible
    groups only.
    """
    valid = (order_flag != 0) & (revenue_table != 0.0)
    order_idx = torch.nonzero(valid, as_tuple=False).flatten()

    if int(order_idx.numel()) == 0:
        device = revenue_table.device
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float64, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
        )

    lk = order_idx.to(torch.int64)
    rev = revenue_table[order_idx]
    od = orderdate_table[order_idx]
    sp = shippriority_table[order_idx]

    idx_sec = torch.argsort(od, stable=True)
    lk_s = lk[idx_sec]
    rev_s = rev[idx_sec]
    od_s = od[idx_sec]
    sp_s = sp[idx_sec]

    idx_pri = torch.argsort(rev_s, descending=True, stable=True)
    k = min(int(top_n), int(order_idx.numel()))

    return lk_s[idx_pri][:k], rev_s[idx_pri][:k], od_s[idx_pri][:k], sp_s[idx_pri][:k]


def _query_core(
    l_orderkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    l_shipdate_days: torch.Tensor,
    o_orderkey: torch.Tensor,
    o_orderdate_days: torch.Tensor,
    o_shippriority: torch.Tensor,
    shipdate_lo_days: int,
    capacity: int,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU/Triton core for TPC-H Q3 after cuDF has produced eligible orders. Triton
    performs the memory-bound lineitem scan, date predicate, direct order lookup,
    revenue projection, and FP64 aggregation in one fused pass, reducing HBM traffic
    and eliminating the large joined intermediate used by the reference plan.
    """
    order_flag, revenue_table, orderdate_table, shippriority_table = _launch_direct_aggregate(
        l_orderkey=l_orderkey,
        l_extendedprice=l_extendedprice,
        l_discount=l_discount,
        l_shipdate_days=l_shipdate_days,
        o_orderkey=o_orderkey,
        o_orderdate_days=o_orderdate_days,
        o_shippriority=o_shippriority,
        shipdate_lo_days=shipdate_lo_days,
        capacity=capacity,
    )
    return _select_top(order_flag, revenue_table, orderdate_table, shippriority_table, top_n)


def run_query(
    tables: dict[str, cudf.DataFrame],
    mktsegment: str = "BUILDING",
    orderdate_hi: str = "1995-03-15",
    shipdate_lo: str = "1995-03-15",
    top_n: int = 10,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT l_orderkey,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               o_orderdate,
               o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < CAST('1995-03-15' AS date)
          AND l_shipdate > CAST('1995-03-15' AS date)
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;

    Description:
        TPC-H Q3 - Shipping Priority. cuDF handles string filtering and the small
        customer-orders join to identify eligible orders. The hot path is a fused
        Triton GPU pipeline: eligible orders are marked in a direct-address table,
        then lineitem is scanned once to apply the shipdate predicate, look up the
        order, compute l_extendedprice * (1 - l_discount), and atomically aggregate
        FP64 revenue per orderkey. This avoids materializing the very large
        orders-lineitem join and replaces the reference Torch unique/scatter
        aggregation with a bandwidth-oriented fused GPU scan.

    Params:
        tables (dict[str, cudf.DataFrame]): Input cuDF tables keyed by TPC-H table name.
        mktsegment (str): customer market segment filter.
        orderdate_hi (str): exclusive upper bound on o_orderdate.
        shipdate_lo (str): exclusive lower bound on l_shipdate.
        top_n (int): LIMIT value.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]

    orderdate_hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)
    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)

    cust_f = df_customer.loc[
        df_customer["c_mktsegment"] == mktsegment,
        ["c_custkey"],
    ]

    if len(cust_f) == 0:
        return _empty_result()

    orders_orderdate_days_all = (
        (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    orders_f = df_orders.loc[
        orders_orderdate_days_all < orderdate_hi_days,
        ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
    ]

    if len(orders_f) == 0:
        return _empty_result()

    eligible_orders = orders_f.merge(
        cust_f,
        left_on="o_custkey",
        right_on="c_custkey",
        how="inner",
    )

    if len(eligible_orders) == 0:
        return _empty_result()

    max_eligible_orderkey = int(eligible_orders["o_orderkey"].max())
    capacity = max(1, max_eligible_orderkey + 1)

    eligible_orderdate_days = (
        (eligible_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    )
    o_orderkey_t = torch.as_tensor(
        eligible_orders["o_orderkey"].to_cupy(), dtype=torch.int64
    )
    o_orderdate_days_t = torch.as_tensor(
        eligible_orderdate_days.to_cupy(), dtype=torch.int32
    )
    o_shippriority_t = torch.as_tensor(
        eligible_orders["o_shippriority"].to_cupy(), dtype=torch.int32
    )

    l_shipdate_days = (
        (df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")
    )
    l_orderkey_t = torch.as_tensor(df_lineitem["l_orderkey"].to_cupy(), dtype=torch.int64)
    l_extendedprice_t = torch.as_tensor(
        df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64
    )
    l_discount_t = torch.as_tensor(
        df_lineitem["l_discount"].to_cupy(), dtype=torch.float64
    )
    l_shipdate_days_t = torch.as_tensor(l_shipdate_days.to_cupy(), dtype=torch.int32)

    lk_t, rev_t, od_t, sp_t = _query_core(
        l_orderkey=l_orderkey_t,
        l_extendedprice=l_extendedprice_t,
        l_discount=l_discount_t,
        l_shipdate_days=l_shipdate_days_t,
        o_orderkey=o_orderkey_t,
        o_orderdate_days=o_orderdate_days_t,
        o_shippriority=o_shippriority_t,
        shipdate_lo_days=shipdate_lo_days,
        capacity=capacity,
        top_n=int(top_n),
    )

    od_series = cudf.Series(od_t.to(torch.int64) * 86400).astype("datetime64[s]")

    return cudf.DataFrame(
        {
            "l_orderkey": lk_t,
            "revenue": rev_t,
            "o_orderdate": od_series,
            "o_shippriority": sp_t,
        }
    )
