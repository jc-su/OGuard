import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _q10_dense_agg_kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    revenue_ptr,
    flag_ptr,
    key_min,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    keys = tl.load(custkey_ptr + offs, mask=mask, other=key_min)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    dense_idx = keys - key_min
    revenue = price * (1.0 - discount)

    tl.atomic_add(revenue_ptr + dense_idx, revenue, sem="relaxed", mask=mask)
    tl.store(flag_ptr + dense_idx, 1, mask=mask)
'''


def _load_kernels() -> types.ModuleType:
    """
    Write Triton GPU kernels to a real on-disk Python file and import them.

    Triton compilation runs from that generated module, which avoids validator
    issues with JIT functions defined in an exec-loaded host module. The host
    module keeps cuDF/PyTorch orchestration, while the per-row aggregation scan
    runs in the Triton kernel.
    """
    tmpdir = tempfile.mkdtemp(prefix="q10_triton_")
    path = os.path.join(tmpdir, "_q10_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q10_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to create import spec for Triton kernels")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K = _load_kernels()


def _launch_q10_dense_agg(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate joined Q10 rows on the GPU with one Triton pass over row tensors.

    The reference computes revenue in one eager PyTorch op, then performs a
    general `torch.unique(..., return_inverse=True)` plus `scatter_add_`. For
    TPC-H customer keys, the key domain is dense and small relative to the table:
    this launcher allocates a dense revenue array over `[min(c_custkey), max]`,
    then a Triton kernel fuses revenue projection and atomic accumulation in a
    single coalesced scan. A compact PyTorch `nonzero` over the much smaller key
    domain restores the same sorted unique-key layout expected by `run_query`.
    """
    n = int(c_custkey.numel())
    if n == 0:
        empty_keys = torch.empty(0, dtype=torch.int64, device=c_custkey.device)
        empty_revenue = torch.empty(0, dtype=torch.float64, device=c_custkey.device)
        return empty_keys, empty_revenue

    key_min = int(torch.min(c_custkey).item())
    key_max = int(torch.max(c_custkey).item())
    key_range = key_max - key_min + 1

    revenue_dense = torch.zeros(key_range, dtype=torch.float64, device=l_extendedprice.device)
    present_flags = torch.zeros(key_range, dtype=torch.int8, device=l_extendedprice.device)

    grid = (triton.cdiv(n, block_size),)
    _K._q10_dense_agg_kernel[grid](
        c_custkey,
        l_extendedprice,
        l_discount,
        revenue_dense,
        present_flags,
        key_min,
        n,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    dense_idx = torch.nonzero(present_flags, as_tuple=False).flatten()
    unique_keys = dense_idx.to(torch.int64) + key_min
    revenue = revenue_dense[dense_idx]
    return unique_keys, revenue


def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU tensor core for TPC-H Q10 revenue aggregation.

    Inputs are already produced by cuDF filters/joins in `run_query`. This
    function replaces the eager PyTorch revenue projection plus scatter-add hot
    path with a Triton fused scan: each row loads customer key, price, and
    discount once, computes `l_extendedprice * (1 - l_discount)` in registers,
    and atomically accumulates into a dense per-customer revenue buffer. This is
    memory-bound, so fusing projection and aggregation reduces temporary tensor
    traffic versus the reference while preserving float64 aggregate output.
    """
    return _launch_q10_dense_agg(c_custkey, l_extendedprice, l_discount)


def run_query(
    tables: dict[str, cudf.DataFrame],
    orderdate_lo: str = "1993-10-01",
    orderdate_hi: str = "1994-01-01",
    returnflag_val: str = "R",
    top_n: int = 20,
) -> cudf.DataFrame:
    """
    SQL: SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue,
         c_acctbal, n_name, c_address, c_phone, c_comment
         FROM customer, orders, lineitem, nation
         WHERE c_custkey=o_custkey AND l_orderkey=o_orderkey
         AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01'
         AND l_returnflag='R' AND c_nationkey=n_nationkey
         GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
         ORDER BY revenue DESC LIMIT 20;
    Description: TPC-H Q10 - Returned item reporting. Finds top 20 customers by revenue
                 from returned lineitems within a quarter.
    Params:
        orderdate_lo (str): inclusive lower bound for o_orderdate.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        returnflag_val (str): lineitem return flag filter.
        top_n (int): number of top customers to return.
    """
    df_customer = tables["customer"]
    df_orders = tables["orders"]
    df_lineitem = tables["lineitem"]
    df_nation = tables["nation"]

    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter orders by date
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days < hi_days)][["o_orderkey", "o_custkey"]]

    # Filter lineitem by returnflag
    lineitem_f = df_lineitem[df_lineitem["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with orders
    lo = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # Join with customer
    loc = lo.merge(df_customer, left_on="o_custkey", right_on="c_custkey")

    # Aggregate revenue by c_custkey in torch
    custkey_t, revenue_t = _query_core(
        torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64),
    )

    # Top-N by revenue
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_t[top_idx]

    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to get descriptive cols
    cust_cols = df_customer[["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]], left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue DESC
    agg_df = agg_df.sort_values("revenue", ascending=False).head(int(top_n))

    result = cudf.DataFrame({
        "c_custkey": agg_df["c_custkey"].reset_index(drop=True),
        "c_name": agg_df["c_name"].reset_index(drop=True),
        "revenue": agg_df["revenue"].reset_index(drop=True),
        "c_acctbal": agg_df["c_acctbal"].reset_index(drop=True),
        "n_name": agg_df["n_name"].reset_index(drop=True),
        "c_address": agg_df["c_address"].reset_index(drop=True),
        "c_phone": agg_df["c_phone"].reset_index(drop=True),
        "c_comment": agg_df["c_comment"].reset_index(drop=True),
    })
    return result
