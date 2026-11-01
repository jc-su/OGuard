import os
import tempfile
import importlib.util
import linecache
from types import ModuleType
from typing import Any

import cudf
import pandas as pd
import torch
import triton


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q10_revenue_by_custkey_kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    revenue_ptr,
    n_rows,
    n_revenue,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_rows

    custkey = tl.load(custkey_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    valid = mask & (custkey >= 0) & (custkey < n_revenue)
    revenue = price * (1.0 - discount)

    tl.atomic_add(revenue_ptr + custkey, revenue, sem="relaxed", mask=valid)
'''


def _load_kernels() -> ModuleType:
    """Write Triton kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q10_triton_")
    path = os.path.join(tmpdir, "_q10_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q10_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for Triton Q10 kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _to_torch(series: cudf.Series, dtype: torch.dtype) -> torch.Tensor:
    """Convert a cuDF GPU column to a PyTorch CUDA tensor without host-row transfer."""
    return torch.as_tensor(series.to_cupy(), dtype=dtype)


def _launch_revenue_by_custkey(
    o_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    n_revenue: int,
    block_size: int,
) -> torch.Tensor:
    """
    Run the fused Triton aggregation kernel on GPU.

    The kernel performs one coalesced pass over joined lineitem/order rows and computes
    l_extendedprice * (1 - l_discount) in registers before atomically accumulating into a
    dense customer-key revenue array. This avoids PyTorch unique/inverse materialization and
    scatter temporaries from the reference hot path.
    """
    revenue = torch.zeros(n_revenue, dtype=torch.float64, device=l_extendedprice.device)
    n_rows = int(l_extendedprice.numel())
    if n_rows == 0 or n_revenue <= 0:
        return revenue

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._q10_revenue_by_custkey_kernel[grid](
        o_custkey,
        l_extendedprice,
        l_discount,
        revenue,
        n_rows,
        n_revenue,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return revenue


def _query_core(
    o_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    n_revenue: int,
    block_size: int = 256,
) -> torch.Tensor:
    """
    GPU/Triton core for Q10 revenue aggregation.

    Runs entirely on GPU in a fused Triton pass over the joined numeric columns. It is faster
    than the reference Torch unique + scatter_add path because it avoids sorting/unique-style
    grouping intermediates and accumulates directly into a dense customer-key array.
    """
    return _launch_revenue_by_custkey(
        o_custkey=o_custkey,
        l_extendedprice=l_extendedprice,
        l_discount=l_discount,
        n_revenue=n_revenue,
        block_size=block_size,
    )


def _empty_result(df_customer: cudf.DataFrame, df_nation: cudf.DataFrame) -> cudf.DataFrame:
    """Build an empty result frame with the required Q10 columns and compatible dtypes."""
    return cudf.DataFrame(
        {
            "c_custkey": df_customer["c_custkey"].head(0).reset_index(drop=True),
            "c_name": df_customer["c_name"].head(0).reset_index(drop=True),
            "revenue": cudf.Series([], dtype="float64"),
            "c_acctbal": df_customer["c_acctbal"].head(0).reset_index(drop=True),
            "n_name": df_nation["n_name"].head(0).reset_index(drop=True),
            "c_address": df_customer["c_address"].head(0).reset_index(drop=True),
            "c_phone": df_customer["c_phone"].head(0).reset_index(drop=True),
            "c_comment": df_customer["c_comment"].head(0).reset_index(drop=True),
        }
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    orderdate_lo: str = "1993-10-01",
    orderdate_hi: str = "1994-01-01",
    returnflag_val: str = "R",
    top_n: int = 20,
    block_size: int = 256,
) -> cudf.DataFrame:
    """
    SQL:
        SELECT c_custkey, c_name,
               sum(l_extendedprice * (1 - l_discount)) AS revenue,
               c_acctbal, n_name, c_address, c_phone, c_comment
        FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate >= CAST('1993-10-01' AS date)
          AND o_orderdate < CAST('1994-01-01' AS date)
          AND l_returnflag = 'R'
          AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        ORDER BY revenue DESC
        LIMIT 20;

    Description:
        TPC-H Q10 returned item reporting. cuDF performs relational work that involves
        joins and strings: date/return-flag filters, the lineitem-orders join, and the final
        small top-customer joins to customer/nation descriptive columns. The numeric revenue
        aggregation hot path runs in Triton: after the lineitem-orders join, a single fused GPU
        kernel computes l_extendedprice * (1 - l_discount) and atomically accumulates directly
        into a dense customer-key revenue array. This avoids the reference plan's large
        customer/string join before aggregation and avoids PyTorch unique/inverse/scatter
        intermediates, reducing memory traffic and temporary allocations on this memory-bound
        analytic scan.

    Params:
        tables (dict[str, cudf.DataFrame]): Input cuDF tables keyed by TPCH table name.
        orderdate_lo (str): Inclusive lower bound for o_orderdate.
        orderdate_hi (str): Exclusive upper bound for o_orderdate.
        returnflag_val (str): Required l_returnflag value.
        top_n (int): Number of highest-revenue customers to return.
        block_size (int): Triton tile size for the fused revenue aggregation kernel.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation = tables["nation"]

    limit = int(top_n)
    if limit <= 0:
        return _empty_result(df_customer, df_nation)

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][
        ["o_orderkey", "o_custkey"]
    ]

    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    joined = lineitem_f.merge(
        orders_f,
        left_on="l_orderkey",
        right_on="o_orderkey",
        how="inner",
    )

    max_custkey = int(df_customer["c_custkey"].max())
    n_revenue = max_custkey + 1
    if n_revenue <= 0:
        return _empty_result(df_customer, df_nation)

    revenue_dense = _query_core(
        o_custkey=_to_torch(joined["o_custkey"], torch.int64),
        l_extendedprice=_to_torch(joined["l_extendedprice"], torch.float64),
        l_discount=_to_torch(joined["l_discount"], torch.float64),
        n_revenue=n_revenue,
        block_size=int(block_size),
    )

    customer_keys = _to_torch(df_customer["c_custkey"], torch.int64)
    customer_revenue = revenue_dense[customer_keys]

    positive_mask = customer_revenue > 0.0
    n_groups = int(torch.count_nonzero(positive_mask).item())
    if n_groups == 0:
        return _empty_result(df_customer, df_nation)

    grouped_keys = customer_keys[positive_mask]
    grouped_revenue = customer_revenue[positive_mask]

    k = min(limit, int(grouped_revenue.numel()))
    top_rev, top_idx = torch.topk(grouped_revenue, k)
    top_keys = grouped_keys[top_idx]

    agg_df = cudf.DataFrame(
        {
            "c_custkey": top_keys,
            "revenue": top_rev,
        }
    )

    cust_cols = df_customer[
        [
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "c_address",
            "c_comment",
            "c_nationkey",
        ]
    ]
    agg_df = agg_df.merge(cust_cols, on="c_custkey", how="inner")

    agg_df = agg_df.merge(
        df_nation[["n_nationkey", "n_name"]],
        left_on="c_nationkey",
        right_on="n_nationkey",
        how="inner",
    )

    agg_df = agg_df.sort_values("revenue", ascending=False).head(limit)

    return cudf.DataFrame(
        {
            "c_custkey": agg_df["c_custkey"].reset_index(drop=True),
            "c_name": agg_df["c_name"].reset_index(drop=True),
            "revenue": agg_df["revenue"].reset_index(drop=True),
            "c_acctbal": agg_df["c_acctbal"].reset_index(drop=True),
            "n_name": agg_df["n_name"].reset_index(drop=True),
            "c_address": agg_df["c_address"].reset_index(drop=True),
            "c_phone": agg_df["c_phone"].reset_index(drop=True),
            "c_comment": agg_df["c_comment"].reset_index(drop=True),
        }
    )
