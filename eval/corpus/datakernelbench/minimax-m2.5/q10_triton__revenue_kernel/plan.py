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
        for bs in [1024, 2048, 4096, 8192]
        for w in [4, 8]
        for s in [2, 4]
    ],
    key=["n_elements"],
    reset_to_zero=["revenue_ptr"],
)
@triton.jit
def _revenue_kernel(
    custkey_ptr,
    price_ptr,
    discount_ptr,
    revenue_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # load columns – use a safe "other" so masked‑out lanes get zeros
    custkey = tl.load(custkey_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)

    # compute revenue = price * (1 - discount)
    revenue = price * (1.0 - discount)

    # atomic add to per‑customer bucket – masked lanes write zero
    tl.atomic_add(revenue_ptr + custkey, revenue)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q10_triton_")
    path = os.path.join(tmpdir, "_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()


def _compute_revenue(
    custkey_tensor: torch.Tensor,
    price_tensor: torch.Tensor,
    discount_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per‑customer revenue using a fused Triton kernel.
    The kernel loads custkey, price, discount; projects revenue;
    and atomically adds into a per‑customer array in a single pass.
    """
    device = custkey_tensor.device
    n_rows = custkey_tensor.shape[0]

    # size output array to cover the range of custkey values
    max_key = int(custkey_tensor.max().item()) + 1
    revenue = torch.zeros(max_key, dtype=torch.float64, device=device)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _K._revenue_kernel[grid](
        custkey_tensor,
        price_tensor,
        discount_tensor,
        revenue,
        n_rows,
    )
    return revenue


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

    Description: TPC‑H Q10 – Returned item reporting.
        Finds the top 20 customers by revenue from returned lineitems in a given quarter.
        This implementation uses cuDF for the join/filtering phases and a fused
        Triton kernel for the hot‑path aggregation (filter → project → per‑customer
        reduce). The kernel does a single pass over the filtered rows, performing
        coalesced loads and a single atomic‑add write per row, which is optimal for
        the memory‑bound scan.

    Params:
        orderdate_lo (str): inclusive lower bound for o_orderdate.
        orderdate_hi (str): exclusive upper bound for o_orderdate.
        returnflag_val (str): lineitem return flag filter.
        top_n (int): number of top customers to return.
    """
    # ------------------------------------------------------------------
    # 1.  Filter orders by date range (int32 epoch days)
    # ------------------------------------------------------------------
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    o_days = (tables["orders"]["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = tables["orders"][(o_days >= lo_days) & (o_days < hi_days)][
        ["o_orderkey", "o_custkey"]
    ]

    # ------------------------------------------------------------------
    # 2.  Filter lineitem by return flag
    # ------------------------------------------------------------------
    lineitem_f = tables["lineitem"][tables["lineitem"]["l_returnflag"] == returnflag_val][
        ["l_orderkey", "l_extendedprice", "l_discount"]
    ]

    # ------------------------------------------------------------------
    # 3.  Join lineitem with filtered orders
    # ------------------------------------------------------------------
    merged = lineitem_f.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")

    # ------------------------------------------------------------------
    # 4.  Move columns to GPU and run the fused Triton aggregation
    # ------------------------------------------------------------------
    custkey_t = torch.as_tensor(merged["o_custkey"].to_cupy(), dtype=torch.int64)
    price_t = torch.as_tensor(merged["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(merged["l_discount"].to_cupy(), dtype=torch.float64)

    revenue_per_cust = _compute_revenue(custkey_t, price_t, discount_t)

    # ------------------------------------------------------------------
    # 5.  Pick top N customers by revenue
    # ------------------------------------------------------------------
    k = min(int(top_n), revenue_per_cust.shape[0])
    top_rev, top_idx = torch.topk(revenue_per_cust, k)

    top_df = cudf.DataFrame({
        "c_custkey": top_idx.cpu().numpy(),
        "revenue": top_rev.cpu().numpy(),
    })

    # ------------------------------------------------------------------
    # 6.  Join back to customer & nation to retrieve descriptive columns
    # ------------------------------------------------------------------
    cust_cols = tables["customer"][
        ["c_custkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment", "c_nationkey"]
    ]
    top_df = top_df.merge(cust_cols, on="c_custkey", how="left")

    top_df = top_df.merge(
        tables["nation"][["n_nationkey", "n_name"]],
        left_on="c_nationkey",
        right_on="n_nationkey",
        how="left",
    )

    # ------------------------------------------------------------------
    # 7.  Order by revenue (descending) and format final result
    # ------------------------------------------------------------------
    top_df = top_df.sort_values("revenue", ascending=False)

    result = cudf.DataFrame({
        "c_custkey": top_df["c_custkey"].reset_index(drop=True),
        "c_name": top_df["c_name"].reset_index(drop=True),
        "revenue": top_df["revenue"].reset_index(drop=True),
        "c_acctbal": top_df["c_acctbal"].reset_index(drop=True),
        "n_name": top_df["n_name"].reset_index(drop=True),
        "c_address": top_df["c_address"].reset_index(drop=True),
        "c_phone": top_df["c_phone"].reset_index(drop=True),
        "c_comment": top_df["c_comment"].reset_index(drop=True),
    })
    return result
