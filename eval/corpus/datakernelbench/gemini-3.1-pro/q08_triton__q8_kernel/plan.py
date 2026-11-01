import os
import tempfile
import importlib.util
import linecache
import torch
import cudf
import pandas as pd
import triton

_KERNEL_SOURCE = r'''
import triton
import triton.language as tl

@triton.jit
def _q8_kernel(
    year_ptr, price_ptr, discount_ptr, is_target_ptr,
    sum_target_ptr, sum_total_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    year = tl.load(year_ptr + offs, mask=mask, other=0)
    price = tl.load(price_ptr + offs, mask=mask, other=0.0)
    discount = tl.load(discount_ptr + offs, mask=mask, other=0.0)
    is_target = tl.load(is_target_ptr + offs, mask=mask, other=0)

    volume = price * (1.0 - discount)
    target_volume = tl.where(is_target > 0, volume, 0.0)

    # Use year directly as index to sum arrays
    valid_idx = (year >= 1900) & (year < 2500) & mask

    tl.atomic_add(sum_total_ptr + year, volume, mask=valid_idx)
    tl.atomic_add(sum_target_ptr + year, target_volume, mask=valid_idx)
'''

def _load_kernels():
    tmpdir = tempfile.mkdtemp(prefix="q8_triton_")
    path = os.path.join(tmpdir, "_q8_kernels.py")
    with open(path, "w") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q8_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_K = _load_kernels()

def run_query(
    tables: dict[str, cudf.DataFrame],
    region_name: str = "AMERICA",
    part_type: str = "ECONOMY ANODIZED STEEL",
    nation_name: str = "BRAZIL",
    orderdate_lo: str = "1995-01-01",
    orderdate_hi: str = "1996-12-31",
) -> cudf.DataFrame:
    """
    SQL: SELECT o_year, sum(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END)/sum(volume) AS mkt_share
         FROM (SELECT extract(year FROM o_orderdate) AS o_year,
                      l_extendedprice*(1-l_discount) AS volume,
                      n2.n_name AS nation
               FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
               WHERE p_partkey=l_partkey AND s_suppkey=l_suppkey AND l_orderkey=o_orderkey
                 AND o_custkey=c_custkey AND c_nationkey=n1.n_nationkey
                 AND n1.n_regionkey=r_regionkey AND r_name='AMERICA'
                 AND s_nationkey=n2.n_nationkey
                 AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                 AND p_type='ECONOMY ANODIZED STEEL') AS all_nations
         GROUP BY o_year ORDER BY o_year;
    Description: TPC-H Q8 optimized with cuDF for joins and Triton for the final reduction by year.
    Params:
        region_name (str): region filter for customer's nation.
        part_type (str): part type filter.
        nation_name (str): nation whose market share is computed (supplier nation).
        orderdate_lo (str): order date lower bound (inclusive).
        orderdate_hi (str): order date upper bound (inclusive).
    """
    df_part = tables["part"]
    df_supplier = tables["supplier"]
    df_lineitem = tables["lineitem"]
    df_orders = tables["orders"]
    df_customer = tables["customer"]
    df_nation = tables["nation"]
    df_region = tables["region"]

    # Date bounds as epoch-day int32
    lo_days = int(pd.to_datetime(orderdate_lo).timestamp() // 86400)
    hi_days = int(pd.to_datetime(orderdate_hi).timestamp() // 86400)

    # Filter region -> nation n1 (customer side)
    region_f = df_region[df_region["r_name"] == region_name][["r_regionkey"]]
    n1 = df_nation.merge(region_f, left_on="n_regionkey", right_on="r_regionkey")[
        ["n_nationkey"]
    ].rename(columns={"n_nationkey": "n1_nationkey"})

    # Customers in that region
    cust_r = df_customer.merge(n1, left_on="c_nationkey", right_on="n1_nationkey")[
        ["c_custkey"]
    ]

    # Filter orders by date range
    o_days = (df_orders["o_orderdate"].astype("int64") // 86400).astype("int32")
    orders_f = df_orders[(o_days >= lo_days) & (o_days <= hi_days)][
        ["o_orderkey", "o_custkey", "o_orderdate"]
    ]
    # Orders by customers in region
    orders_f = orders_f.merge(cust_r, left_on="o_custkey", right_on="c_custkey")[
        ["o_orderkey", "o_orderdate"]
    ]

    # Filter part by type
    part_f = df_part[df_part["p_type"] == part_type][["p_partkey"]]

    # Lineitem joined with part (filter by p_partkey)
    li = df_lineitem.merge(part_f, left_on="l_partkey", right_on="p_partkey")[
        ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]
    ]

    # Join lineitem with filtered orders
    li = li.merge(orders_f, left_on="l_orderkey", right_on="o_orderkey")[
        ["l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"]
    ]

    # Join supplier -> nation n2 to get supplier nation name
    sup_n2 = df_supplier.merge(
        df_nation, left_on="s_nationkey", right_on="n_nationkey"
    )[["s_suppkey", "n_name"]]

    li = li.merge(sup_n2, left_on="l_suppkey", right_on="s_suppkey")[
        ["l_extendedprice", "l_discount", "o_orderdate", "n_name"]
    ]

    if len(li) == 0:
        return cudf.DataFrame({
            "o_year": cudf.Series([], dtype="int32"),
            "mkt_share": cudf.Series([], dtype="float64"),
        })

    # Build tensors
    o_year_series = li["o_orderdate"].dt.year.astype("int32")
    year_t = torch.as_tensor(o_year_series.to_cupy(), dtype=torch.int32)
    price_t = torch.as_tensor(li["l_extendedprice"].to_cupy(), dtype=torch.float64)
    discount_t = torch.as_tensor(li["l_discount"].to_cupy(), dtype=torch.float64)
    is_target_t = torch.as_tensor(
        (li["n_name"] == nation_name).astype("int32").to_cupy(), dtype=torch.int32
    )

    n_elements = year_t.numel()
    
    # We allocate up to 2500 for year indices to avoid global atomics overhead and use array offsets
    sum_total = torch.zeros(2500, dtype=torch.float64, device=year_t.device)
    sum_target = torch.zeros(2500, dtype=torch.float64, device=year_t.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _K._q8_kernel[grid](
        year_t, price_t, discount_t, is_target_t,
        sum_target, sum_total,
        n_elements,
        BLOCK_SIZE=4096,
    )

    # Extract valid years and compute target market share directly in PyTorch
    valid_years = torch.where(sum_total > 0)[0]
    mkt_share = sum_target[valid_years] / sum_total[valid_years]
    
    return cudf.DataFrame({
        "o_year": cudf.Series(valid_years.to(torch.int32).cpu().numpy(), dtype="int32"),
        "mkt_share": cudf.Series(mkt_share.cpu().numpy(), dtype="float64"),
    })
