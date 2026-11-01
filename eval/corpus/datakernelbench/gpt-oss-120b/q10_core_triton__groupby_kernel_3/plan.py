import os
import tempfile
import importlib.util
import linecache
import torch
import triton
import cudf
import pandas as pd

# ----------------------------------------------------------------------
# Triton kernel source (written to a temp file and imported)
# ----------------------------------------------------------------------
_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

# Autotune configurations for H100: vary block size and warps
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=4, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ] + [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=8, num_stages=2)
        for bs in [1024, 2048, 4096, 8192]
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],  # ensure accumulator is zeroed each launch
)
@triton.jit
def _groupby_kernel(
    price_ptr,       # float64 *
    discount_ptr,    # float64 *
    inv_ptr,         # int64 *
    out_ptr,         # float64 *
    n_elements,      # i32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load columns (masked)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    inv = tl.load(inv_ptr + offs, mask=mask, other=0)

    # Compute revenue per row
    revenue = price * (1.0 - discount)

    # Atomic add into bucket identified by inverse index
    tl.atomic_add(out_ptr + inv, revenue, mask=mask)
'''

def _load_kernels():
    """Write the Triton source to a temporary file and import it."""
    tmpdir = tempfile.mkdtemp(prefix="tpch_q10_triton_")
    path = os.path.join(tmpdir, "_q10_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q10_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load the compiled Triton module once at import time
_K = _load_kernels()

def _launch_groupby(
    inv: torch.Tensor,
    price: torch.Tensor,
    discount: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse revenue computation and per‑key scatter‑add using a single Triton kernel.
    """
    n = price.numel()
    out = torch.zeros(inv.max().item() + 1,
                      dtype=torch.float64,
                      device=price.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    # No explicit BLOCK_SIZE argument – it is supplied by autotune meta‑params
    _K._groupby_kernel[grid](
        price,
        discount,
        inv,
        out,
        n,
    )
    return out

def _query_core(
    c_custkey: torch.Tensor,
    l_extendedprice: torch.Tensor,
    l_discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per‑customer revenue for TPC‑H Q10.
    Steps:
      1. ``torch.unique`` obtains dense key IDs and the inverse mapping.
      2. Triton kernel fuses revenue = price * (1‑discount) and a scatter‑add
         into per‑key accumulators.
    This reduces two passes (elementwise multiplication + scatter_add) to a
    single memory‑bound kernel, cutting HBM traffic on H100.
    """
    unique_keys, inverse = torch.unique(c_custkey, return_inverse=True)
    revenue = _launch_groupby(inverse, l_extendedprice, l_discount)
    return unique_keys, revenue


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
    Description: TPC‑H Q10 – Returned item reporting. Finds top 20 customers by revenue
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

    # Aggregate revenue by c_custkey using the Triton‑accelerated core
    custkey_t, revenue_t = _query_core(
        torch.as_tensor(loc["c_custkey"].to_cupy(), dtype=torch.int64),
        torch.as_tensor(loc["l_extendedprice"].to_cupy(), dtype=torch.float64),
        torch.as_tensor(loc["l_discount"].to_cupy(), dtype=torch.float64),
    )

    # Top‑N by revenue
    k = min(int(top_n), revenue_t.shape[0])
    top_rev, top_idx = torch.topk(revenue_t, k)
    top_keys = custkey_t[top_idx]

    agg_df = cudf.DataFrame({
        "c_custkey": top_keys,
        "revenue": top_rev,
    })

    # Join back with customer to get the descriptive columns
    cust_cols = df_customer[[
        "c_custkey", "c_name", "c_acctbal", "c_phone",
        "c_address", "c_comment", "c_nationkey"
    ]]
    agg_df = agg_df.merge(cust_cols, on="c_custkey")

    # Join with nation for the nation name
    agg_df = agg_df.merge(df_nation[["n_nationkey", "n_name"]],
                          left_on="c_nationkey", right_on="n_nationkey")

    # Order by revenue DESC and keep only the requested number of rows
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
