"""The 22 TPC-H queries, each written once for pandas and cuDF.

A query is a function of the DataFrame library, the module `pandas` or
`cudf`, and the tables of one database as frames of that library
(`eval.tpch.database.Tables`). It returns a frame with the query's result
columns in the specification's order, sorted by its ORDER BY and cut by its
LIMIT. The parameters are the validation parameters of the specification,
clause 2.4; Q11's fraction is 0.0001 divided by the scale factor. A date is
the number of days since 1970-01-01, as in the tables.

The queries use only operations that both libraries have with the same
semantics: column selection, masks, arithmetic, `isin`, `where`, the string
methods `startswith`, `endswith`, `contains`, and `slice`, `merge`,
`groupby(..., as_index=False, sort=False).agg` with named aggregations
(`sum`, `mean`, `min`, `count`, `nunique`), `sort_values`, and `head`. The
libraries differ where these give rows in no specified order: pandas keeps
the left frame's order in a join and the order of first appearance in a
group-by, and cuDF gives the rows in an order of its hash tables. So no query
depends on the order of rows before its final sort, and where the
specification's ORDER BY leaves rows with equal keys unordered, the query adds
the remaining key columns as the last sort keys, which fixes the order of
those rows and the rows that a LIMIT keeps. Two further differences:

- A left join gives a missing key as NaN in a float column in pandas and as a
  null in an integer column in cuDF; Q13 only counts the non-missing keys,
  which `count` does in both.
- A reduction of a cuDF column, such as `sum`, is a NumPy scalar on the host,
  as it is in pandas; the single-row results of Q6, Q14, Q17, and Q19 are
  frames that the library builds from such a scalar.

SQL's `LIKE` patterns are `startswith`, `endswith`, `contains` of a literal,
or `contains` of a regular expression (`%special%requests%` is
`special.*requests`). An aggregate over no row is 0 here and NULL in SQL; with
the runner's seed, the database gives every query rows at each scale factor
from 0.01 to 1 that `eval/rq3_pipelines/tpch_queries.md` lists.
"""

import math
from collections.abc import Callable
from datetime import date, timedelta
from typing import Literal, Protocol, assert_never

import cudf
import pandas as pd

from eval.tpch.data import day
from eval.tpch.database import Tables

__all__ = ["QUERIES", "Frame", "Library", "Query", "host", "join", "mismatch", "year"]

Frame = pd.DataFrame | cudf.DataFrame
Column = pd.Series | cudf.Series


class Library(Protocol):
    """The module of a DataFrame library, `pandas` or `cudf`: the queries use its
    frame constructor."""

    @property
    def DataFrame(self) -> type[pd.DataFrame] | type[cudf.DataFrame]: ...


Query = Callable[[Library, Tables[Frame]], Frame]
How = Literal["inner", "left"]


def join(
    left: Frame, right: Frame, left_on: list[str], right_on: list[str], how: How = "inner"
) -> Frame:
    """The join of two frames of one library on equal keys. A key column of the
    same name on both sides appears once in the result, in both libraries."""
    match left, right:
        case pd.DataFrame(), pd.DataFrame():
            return left.merge(right, how=how, left_on=left_on, right_on=right_on)
        case cudf.DataFrame(), cudf.DataFrame():
            return left.merge(right, how=how, left_on=left_on, right_on=right_on)
        case _:
            raise TypeError(f"a join of a {type(left).__name__} and a {type(right).__name__}")


def year(days: Column) -> Column:
    """The year of each date."""
    return (days.astype("int64") * 86_400).astype("datetime64[s]").dt.year


def host(frame: Frame) -> pd.DataFrame:
    """A query's result in host memory, as pandas; for cuDF, after the device is done."""
    match frame:
        case pd.DataFrame():
            return frame
        case cudf.DataFrame():
            return frame.to_pandas()
        case _ as never:
            assert_never(never)


def mismatch(expected: pd.DataFrame, actual: pd.DataFrame) -> str | None:
    """The first difference between two results, or None if they are equal: the
    same columns in the same order and the same rows in the same order, a
    floating-point value within a relative tolerance of 1e-9 and any other value
    exactly."""
    if list(expected.columns) != list(actual.columns):
        return f"columns {list(actual.columns)}, expected {list(expected.columns)}"
    if len(expected) != len(actual):
        return f"{len(actual)} rows, expected {len(expected)}"
    for name in expected.columns:
        wanted, got = expected[name].tolist(), actual[name].tolist()
        floating = expected[name].dtype.kind == "f" or actual[name].dtype.kind == "f"
        for row, (a, b) in enumerate(zip(wanted, got, strict=True)):
            equal = math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0) if floating else a == b
            if not equal:
                return f"row {row} of {name!r}: {b!r}, expected {a!r}"
    return None


def _add_months(value: date, months: int) -> date:
    total = value.month - 1 + months
    return value.replace(year=value.year + total // 12, month=total % 12 + 1)


def _revenue(frame: Frame) -> Column:
    """Each line's `l_extendedprice * (1 - l_discount)`."""
    return frame["l_extendedprice"] * (1 - frame["l_discount"])


def _nations_in(t: Tables[Frame], region: str) -> Frame:
    """The key and name of each nation of the region."""
    regions = t.region[t.region["r_name"] == region][["r_regionkey"]]
    nations = join(
        t.nation[["n_nationkey", "n_name", "n_regionkey"]],
        regions,
        ["n_regionkey"],
        ["r_regionkey"],
    )
    return nations[["n_nationkey", "n_name"]]


def _nation_named(t: Tables[Frame], name: str) -> Frame:
    """The key of the nation of this name."""
    return t.nation[t.nation["n_name"] == name][["n_nationkey"]]


def q1(xd: Library, t: Tables[Frame]) -> Frame:
    """Pricing summary report."""
    li = t.lineitem[
        [
            "l_returnflag",
            "l_linestatus",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_shipdate",
        ]
    ]
    li = li[li["l_shipdate"] <= day(date(1998, 12, 1) - timedelta(days=90))]
    price = _revenue(li)
    li = li.assign(disc_price=price, charge=price * (1 + li["l_tax"]))
    out = li.groupby(["l_returnflag", "l_linestatus"], as_index=False, sort=False).agg(
        sum_qty=("l_quantity", "sum"),
        sum_base_price=("l_extendedprice", "sum"),
        sum_disc_price=("disc_price", "sum"),
        sum_charge=("charge", "sum"),
        avg_qty=("l_quantity", "mean"),
        avg_price=("l_extendedprice", "mean"),
        avg_disc=("l_discount", "mean"),
        count_order=("l_quantity", "count"),
    )
    return out.sort_values(["l_returnflag", "l_linestatus"])


def q2(xd: Library, t: Tables[Frame]) -> Frame:
    """Minimum cost supplier."""
    supplier = t.supplier[
        ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]
    ]
    supplier = join(supplier, _nations_in(t, "EUROPE"), ["s_nationkey"], ["n_nationkey"])
    offers = join(
        t.partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]],
        supplier,
        ["ps_suppkey"],
        ["s_suppkey"],
    )
    part = t.part[["p_partkey", "p_mfgr", "p_size", "p_type"]]
    part = part[(part["p_size"] == 15) & part["p_type"].str.endswith("BRASS")]
    rows = join(part, offers, ["p_partkey"], ["ps_partkey"])
    cheapest = rows.groupby("p_partkey", as_index=False, sort=False).agg(
        min_cost=("ps_supplycost", "min")
    )
    rows = join(rows, cheapest, ["p_partkey"], ["p_partkey"])
    rows = rows[rows["ps_supplycost"] == rows["min_cost"]]
    out = rows[
        [
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        ]
    ]
    out = out.sort_values(
        ["s_acctbal", "n_name", "s_name", "p_partkey"], ascending=[False, True, True, True]
    )
    return out.head(100)


def q3(xd: Library, t: Tables[Frame]) -> Frame:
    """Shipping priority."""
    cutoff = day(date(1995, 3, 15))
    customer = t.customer[["c_custkey", "c_mktsegment"]]
    customer = customer[customer["c_mktsegment"] == "BUILDING"][["c_custkey"]]
    orders = t.orders[["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"]]
    orders = orders[orders["o_orderdate"] < cutoff]
    li = t.lineitem[["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"]]
    li = li[li["l_shipdate"] > cutoff]
    rows = join(
        join(customer, orders, ["c_custkey"], ["o_custkey"]), li, ["o_orderkey"], ["l_orderkey"]
    )
    rows = rows.assign(revenue=_revenue(rows))
    out = rows.groupby(
        ["l_orderkey", "o_orderdate", "o_shippriority"], as_index=False, sort=False
    ).agg(revenue=("revenue", "sum"))
    out = out[["l_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    # the order key after the specification's keys
    out = out.sort_values(["revenue", "o_orderdate", "l_orderkey"], ascending=[False, True, True])
    return out.head(10)


def q4(xd: Library, t: Tables[Frame]) -> Frame:
    """Order priority checking."""
    start = date(1993, 7, 1)
    orders = t.orders[["o_orderkey", "o_orderdate", "o_orderpriority"]]
    orders = orders[
        (orders["o_orderdate"] >= day(start)) & (orders["o_orderdate"] < day(_add_months(start, 3)))
    ]
    li = t.lineitem[["l_orderkey", "l_commitdate", "l_receiptdate"]]
    late = li[li["l_commitdate"] < li["l_receiptdate"]]["l_orderkey"]
    orders = orders[orders["o_orderkey"].isin(late)]
    out = orders.groupby("o_orderpriority", as_index=False, sort=False).agg(
        order_count=("o_orderkey", "count")
    )
    return out.sort_values("o_orderpriority")


def q5(xd: Library, t: Tables[Frame]) -> Frame:
    """Local supplier volume."""
    start = date(1994, 1, 1)
    customer = join(
        t.customer[["c_custkey", "c_nationkey"]],
        _nations_in(t, "ASIA"),
        ["c_nationkey"],
        ["n_nationkey"],
    )
    orders = t.orders[["o_orderkey", "o_custkey", "o_orderdate"]]
    orders = orders[
        (orders["o_orderdate"] >= day(start))
        & (orders["o_orderdate"] < day(_add_months(start, 12)))
    ]
    rows = join(customer, orders[["o_orderkey", "o_custkey"]], ["c_custkey"], ["o_custkey"])
    li = t.lineitem[["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"]]
    rows = join(rows, li, ["o_orderkey"], ["l_orderkey"])
    supplier = t.supplier[["s_suppkey", "s_nationkey"]]
    rows = join(rows, supplier, ["l_suppkey", "c_nationkey"], ["s_suppkey", "s_nationkey"])
    rows = rows.assign(revenue=_revenue(rows))
    out = rows.groupby("n_name", as_index=False, sort=False).agg(revenue=("revenue", "sum"))
    # the nation's name after the specification's key
    return out.sort_values(["revenue", "n_name"], ascending=[False, True])


def q6(xd: Library, t: Tables[Frame]) -> Frame:
    """Forecasting revenue change."""
    start = date(1994, 1, 1)
    li = t.lineitem[["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"]]
    # the discount 0.06 plus or minus 0.01, as the decimal bounds: the float sum
    # 0.06 + 0.01 is below the float 0.07
    keep = (
        (li["l_shipdate"] >= day(start))
        & (li["l_shipdate"] < day(_add_months(start, 12)))
        & (li["l_discount"] >= 0.05)
        & (li["l_discount"] <= 0.07)
        & (li["l_quantity"] < 24)
    )
    li = li[keep]
    revenue = float((li["l_extendedprice"] * li["l_discount"]).sum())
    return xd.DataFrame({"revenue": [revenue]})


def q7(xd: Library, t: Tables[Frame]) -> Frame:
    """Volume shipping."""
    nation = t.nation[["n_nationkey", "n_name"]]
    nation = nation[nation["n_name"].isin(["FRANCE", "GERMANY"])]
    supplier = join(
        t.supplier[["s_suppkey", "s_nationkey"]],
        nation.rename(columns={"n_nationkey": "supp_nationkey", "n_name": "supp_nation"}),
        ["s_nationkey"],
        ["supp_nationkey"],
    )
    customer = join(
        t.customer[["c_custkey", "c_nationkey"]],
        nation.rename(columns={"n_nationkey": "cust_nationkey", "n_name": "cust_nation"}),
        ["c_nationkey"],
        ["cust_nationkey"],
    )
    li = t.lineitem[["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]]
    li = li[
        (li["l_shipdate"] >= day(date(1995, 1, 1))) & (li["l_shipdate"] <= day(date(1996, 12, 31)))
    ]
    rows = join(li, supplier, ["l_suppkey"], ["s_suppkey"])
    rows = join(rows, t.orders[["o_orderkey", "o_custkey"]], ["l_orderkey"], ["o_orderkey"])
    rows = join(rows, customer, ["o_custkey"], ["c_custkey"])
    # both nations are FRANCE or GERMANY, so the two pairs are the unequal ones
    rows = rows[rows["supp_nation"] != rows["cust_nation"]]
    rows = rows.assign(l_year=year(rows["l_shipdate"]), volume=_revenue(rows))
    out = rows.groupby(["supp_nation", "cust_nation", "l_year"], as_index=False, sort=False).agg(
        revenue=("volume", "sum")
    )
    return out.sort_values(["supp_nation", "cust_nation", "l_year"])


def q8(xd: Library, t: Tables[Frame]) -> Frame:
    """National market share."""
    customer = join(
        t.customer[["c_custkey", "c_nationkey"]],
        _nations_in(t, "AMERICA"),
        ["c_nationkey"],
        ["n_nationkey"],
    )
    orders = t.orders[["o_orderkey", "o_custkey", "o_orderdate"]]
    orders = orders[
        (orders["o_orderdate"] >= day(date(1995, 1, 1)))
        & (orders["o_orderdate"] <= day(date(1996, 12, 31)))
    ]
    orders = join(orders, customer[["c_custkey"]], ["o_custkey"], ["c_custkey"])
    part = t.part[["p_partkey", "p_type"]]
    part = part[part["p_type"] == "ECONOMY ANODIZED STEEL"][["p_partkey"]]
    li = t.lineitem[["l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"]]
    rows = join(li, part, ["l_partkey"], ["p_partkey"])
    rows = join(rows, orders[["o_orderkey", "o_orderdate"]], ["l_orderkey"], ["o_orderkey"])
    rows = join(rows, t.supplier[["s_suppkey", "s_nationkey"]], ["l_suppkey"], ["s_suppkey"])
    rows = join(rows, t.nation[["n_nationkey", "n_name"]], ["s_nationkey"], ["n_nationkey"])
    volume = _revenue(rows)
    rows = rows.assign(
        o_year=year(rows["o_orderdate"]),
        volume=volume,
        brazil=volume.where(rows["n_name"] == "BRAZIL", 0.0),
    )
    out = rows.groupby("o_year", as_index=False, sort=False).agg(
        brazil=("brazil", "sum"), volume=("volume", "sum")
    )
    out = out.assign(mkt_share=out["brazil"] / out["volume"])
    return out[["o_year", "mkt_share"]].sort_values("o_year")


def q9(xd: Library, t: Tables[Frame]) -> Frame:
    """Product type profit measure."""
    part = t.part[["p_partkey", "p_name"]]
    part = part[part["p_name"].str.contains("green", regex=False)][["p_partkey"]]
    li = t.lineitem[
        ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount"]
    ]
    rows = join(li, part, ["l_partkey"], ["p_partkey"])
    rows = join(
        rows,
        t.partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost"]],
        ["l_partkey", "l_suppkey"],
        ["ps_partkey", "ps_suppkey"],
    )
    rows = join(rows, t.supplier[["s_suppkey", "s_nationkey"]], ["l_suppkey"], ["s_suppkey"])
    rows = join(rows, t.nation[["n_nationkey", "n_name"]], ["s_nationkey"], ["n_nationkey"])
    rows = join(rows, t.orders[["o_orderkey", "o_orderdate"]], ["l_orderkey"], ["o_orderkey"])
    rows = rows.assign(
        nation=rows["n_name"],
        o_year=year(rows["o_orderdate"]),
        amount=_revenue(rows) - rows["ps_supplycost"] * rows["l_quantity"],
    )
    out = rows.groupby(["nation", "o_year"], as_index=False, sort=False).agg(
        sum_profit=("amount", "sum")
    )
    return out.sort_values(["nation", "o_year"], ascending=[True, False])


def q10(xd: Library, t: Tables[Frame]) -> Frame:
    """Returned item reporting."""
    start = date(1993, 10, 1)
    orders = t.orders[["o_orderkey", "o_custkey", "o_orderdate"]]
    orders = orders[
        (orders["o_orderdate"] >= day(start)) & (orders["o_orderdate"] < day(_add_months(start, 3)))
    ]
    li = t.lineitem[["l_orderkey", "l_extendedprice", "l_discount", "l_returnflag"]]
    li = li[li["l_returnflag"] == "R"]
    rows = join(li, orders[["o_orderkey", "o_custkey"]], ["l_orderkey"], ["o_orderkey"])
    rows = rows.assign(revenue=_revenue(rows))
    # grouped by the customer key alone: the other grouped columns are the customer's
    revenue = rows.groupby("o_custkey", as_index=False, sort=False).agg(revenue=("revenue", "sum"))
    customer = t.customer[
        ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_comment"]
    ]
    customer = join(customer, t.nation[["n_nationkey", "n_name"]], ["c_nationkey"], ["n_nationkey"])
    out = join(revenue, customer, ["o_custkey"], ["c_custkey"])
    out = out[
        [
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        ]
    ]
    # the customer key after the specification's key
    return out.sort_values(["revenue", "c_custkey"], ascending=[False, True]).head(20)


def q11(xd: Library, t: Tables[Frame]) -> Frame:
    """Important stock identification."""
    supplier = join(
        t.supplier[["s_suppkey", "s_nationkey"]],
        _nation_named(t, "GERMANY"),
        ["s_nationkey"],
        ["n_nationkey"],
    )
    stock = join(
        t.partsupp[["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]],
        supplier,
        ["ps_suppkey"],
        ["s_suppkey"],
    )
    stock = stock.assign(value=stock["ps_supplycost"] * stock["ps_availqty"])
    threshold = float(stock["value"].sum()) * 0.0001 / t.scale
    out = stock.groupby("ps_partkey", as_index=False, sort=False).agg(value=("value", "sum"))
    out = out[out["value"] > threshold]
    # the part key after the specification's key
    return out.sort_values(["value", "ps_partkey"], ascending=[False, True])


def q12(xd: Library, t: Tables[Frame]) -> Frame:
    """Shipping modes and order priority."""
    start = date(1994, 1, 1)
    li = t.lineitem[["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"]]
    li = li[
        li["l_shipmode"].isin(["MAIL", "SHIP"])
        & (li["l_commitdate"] < li["l_receiptdate"])
        & (li["l_shipdate"] < li["l_commitdate"])
        & (li["l_receiptdate"] >= day(start))
        & (li["l_receiptdate"] < day(_add_months(start, 12)))
    ]
    rows = join(li, t.orders[["o_orderkey", "o_orderpriority"]], ["l_orderkey"], ["o_orderkey"])
    high = rows["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])
    rows = rows.assign(high=high.astype("int64"), low=(~high).astype("int64"))
    out = rows.groupby("l_shipmode", as_index=False, sort=False).agg(
        high_line_count=("high", "sum"), low_line_count=("low", "sum")
    )
    return out.sort_values("l_shipmode")


def q13(xd: Library, t: Tables[Frame]) -> Frame:
    """Customer distribution."""
    orders = t.orders[["o_orderkey", "o_custkey", "o_comment"]]
    orders = orders[~orders["o_comment"].str.contains("special.*requests", regex=True)]
    rows = join(
        t.customer[["c_custkey"]],
        orders[["o_orderkey", "o_custkey"]],
        ["c_custkey"],
        ["o_custkey"],
        "left",
    )
    # `count` counts the orders that the left join found: pandas gives a missing
    # order key as NaN, cuDF as a null
    counts = rows.groupby("c_custkey", as_index=False, sort=False).agg(
        c_count=("o_orderkey", "count")
    )
    out = counts.groupby("c_count", as_index=False, sort=False).agg(custdist=("c_custkey", "count"))
    return out.sort_values(["custdist", "c_count"], ascending=[False, False])


def q14(xd: Library, t: Tables[Frame]) -> Frame:
    """Promotion effect."""
    start = date(1995, 9, 1)
    li = t.lineitem[["l_partkey", "l_extendedprice", "l_discount", "l_shipdate"]]
    li = li[(li["l_shipdate"] >= day(start)) & (li["l_shipdate"] < day(_add_months(start, 1)))]
    rows = join(li, t.part[["p_partkey", "p_type"]], ["l_partkey"], ["p_partkey"])
    revenue = _revenue(rows)
    promo = revenue.where(rows["p_type"].str.startswith("PROMO"), 0.0)
    share = 100.0 * float(promo.sum()) / float(revenue.sum())
    return xd.DataFrame({"promo_revenue": [share]})


def q15(xd: Library, t: Tables[Frame]) -> Frame:
    """Top supplier."""
    start = date(1996, 1, 1)
    li = t.lineitem[["l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"]]
    li = li[(li["l_shipdate"] >= day(start)) & (li["l_shipdate"] < day(_add_months(start, 3)))]
    li = li.assign(revenue=_revenue(li))
    revenue = li.groupby("l_suppkey", as_index=False, sort=False).agg(
        total_revenue=("revenue", "sum")
    )
    top = revenue[revenue["total_revenue"] == revenue["total_revenue"].max()]
    supplier = t.supplier[["s_suppkey", "s_name", "s_address", "s_phone"]]
    out = join(supplier, top, ["s_suppkey"], ["l_suppkey"])
    return out[["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"]].sort_values(
        "s_suppkey"
    )


def q16(xd: Library, t: Tables[Frame]) -> Frame:
    """Parts/supplier relationship."""
    part = t.part[["p_partkey", "p_brand", "p_type", "p_size"]]
    part = part[
        (part["p_brand"] != "Brand#45")
        & ~part["p_type"].str.startswith("MEDIUM POLISHED")
        & part["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])
    ]
    supplier = t.supplier[["s_suppkey", "s_comment"]]
    complaints = supplier[supplier["s_comment"].str.contains("Customer.*Complaints", regex=True)]
    offers = t.partsupp[["ps_partkey", "ps_suppkey"]]
    offers = offers[~offers["ps_suppkey"].isin(complaints["s_suppkey"])]
    rows = join(offers, part, ["ps_partkey"], ["p_partkey"])
    out = rows.groupby(["p_brand", "p_type", "p_size"], as_index=False, sort=False).agg(
        supplier_cnt=("ps_suppkey", "nunique")
    )
    return out.sort_values(
        ["supplier_cnt", "p_brand", "p_type", "p_size"], ascending=[False, True, True, True]
    )


def q17(xd: Library, t: Tables[Frame]) -> Frame:
    """Small-quantity-order revenue."""
    part = t.part[["p_partkey", "p_brand", "p_container"]]
    part = part[(part["p_brand"] == "Brand#23") & (part["p_container"] == "MED BOX")][["p_partkey"]]
    li = join(
        t.lineitem[["l_partkey", "l_quantity", "l_extendedprice"]],
        part,
        ["l_partkey"],
        ["p_partkey"],
    )
    # the average over all lines of the part: `li` holds every line of these parts
    average = li.groupby("l_partkey", as_index=False, sort=False).agg(
        avg_quantity=("l_quantity", "mean")
    )
    li = join(li, average, ["l_partkey"], ["l_partkey"])
    li = li[li["l_quantity"] < 0.2 * li["avg_quantity"]]
    return xd.DataFrame({"avg_yearly": [float(li["l_extendedprice"].sum()) / 7.0]})


def q18(xd: Library, t: Tables[Frame]) -> Frame:
    """Large volume customer."""
    li = t.lineitem[["l_orderkey", "l_quantity"]]
    quantity = li.groupby("l_orderkey", as_index=False, sort=False).agg(
        sum_quantity=("l_quantity", "sum")
    )
    large = quantity[quantity["sum_quantity"] > 300]
    orders = t.orders[["o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"]]
    rows = join(large, orders, ["l_orderkey"], ["o_orderkey"])
    rows = join(rows, t.customer[["c_custkey", "c_name"]], ["o_custkey"], ["c_custkey"])
    out = rows[["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice", "sum_quantity"]]
    # the order key after the specification's keys
    out = out.sort_values(
        ["o_totalprice", "o_orderdate", "o_orderkey"], ascending=[False, True, True]
    )
    return out.head(100)


def q19(xd: Library, t: Tables[Frame]) -> Frame:
    """Discounted revenue."""
    li = t.lineitem[
        ["l_partkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode", "l_shipinstruct"]
    ]
    # "AIR REG" is the specification's; the modes are "REG AIR" and "AIR", so it matches no line
    li = li[
        li["l_shipmode"].isin(["AIR", "AIR REG"]) & (li["l_shipinstruct"] == "DELIVER IN PERSON")
    ]
    rows = join(
        li, t.part[["p_partkey", "p_brand", "p_container", "p_size"]], ["l_partkey"], ["p_partkey"]
    )

    def term(brand: str, containers: list[str], quantity: int, size: int) -> Column:
        return (
            (rows["p_brand"] == brand)
            & rows["p_container"].isin(containers)
            & (rows["l_quantity"] >= quantity)
            & (rows["l_quantity"] <= quantity + 10)
            & (rows["p_size"] >= 1)
            & (rows["p_size"] <= size)
        )

    keep = (
        term("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
        | term("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
        | term("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
    )
    revenue = float(_revenue(rows[keep]).sum())
    return xd.DataFrame({"revenue": [revenue]})


def q20(xd: Library, t: Tables[Frame]) -> Frame:
    """Potential part promotion."""
    start = date(1994, 1, 1)
    part = t.part[["p_partkey", "p_name"]]
    part = part[part["p_name"].str.startswith("forest")][["p_partkey"]]
    li = t.lineitem[["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"]]
    li = li[(li["l_shipdate"] >= day(start)) & (li["l_shipdate"] < day(_add_months(start, 12)))]
    shipped = li.groupby(["l_partkey", "l_suppkey"], as_index=False, sort=False).agg(
        sum_quantity=("l_quantity", "sum")
    )
    offers = join(
        t.partsupp[["ps_partkey", "ps_suppkey", "ps_availqty"]], part, ["ps_partkey"], ["p_partkey"]
    )
    # an inner join: a part and supplier with no line in the year have a NULL sum,
    # which no quantity exceeds
    offers = join(offers, shipped, ["ps_partkey", "ps_suppkey"], ["l_partkey", "l_suppkey"])
    offers = offers[offers["ps_availqty"] > 0.5 * offers["sum_quantity"]]
    supplier = join(
        t.supplier[["s_suppkey", "s_name", "s_address", "s_nationkey"]],
        _nation_named(t, "CANADA"),
        ["s_nationkey"],
        ["n_nationkey"],
    )
    supplier = supplier[supplier["s_suppkey"].isin(offers["ps_suppkey"])]
    return supplier[["s_name", "s_address"]].sort_values("s_name")


def q21(xd: Library, t: Tables[Frame]) -> Frame:
    """Suppliers who kept orders waiting."""
    li = t.lineitem[["l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"]]
    late = li[li["l_receiptdate"] > li["l_commitdate"]][["l_orderkey", "l_suppkey"]]
    # another supplier exists in the order if it has more than one supplier, and no
    # other supplier is late if its late lines have one supplier, the late line's
    suppliers = li.groupby("l_orderkey", as_index=False, sort=False).agg(
        suppliers=("l_suppkey", "nunique")
    )
    late_suppliers = late.groupby("l_orderkey", as_index=False, sort=False).agg(
        late_suppliers=("l_suppkey", "nunique")
    )
    supplier = join(
        t.supplier[["s_suppkey", "s_name", "s_nationkey"]],
        _nation_named(t, "SAUDI ARABIA"),
        ["s_nationkey"],
        ["n_nationkey"],
    )
    rows = join(late, supplier, ["l_suppkey"], ["s_suppkey"])
    orders = t.orders[["o_orderkey", "o_orderstatus"]]
    orders = orders[orders["o_orderstatus"] == "F"][["o_orderkey"]]
    rows = join(rows, orders, ["l_orderkey"], ["o_orderkey"])
    rows = join(rows, suppliers, ["l_orderkey"], ["l_orderkey"])
    rows = join(rows, late_suppliers, ["l_orderkey"], ["l_orderkey"])
    rows = rows[(rows["suppliers"] > 1) & (rows["late_suppliers"] == 1)]
    out = rows.groupby("s_name", as_index=False, sort=False).agg(numwait=("l_orderkey", "count"))
    return out.sort_values(["numwait", "s_name"], ascending=[False, True]).head(100)


def q22(xd: Library, t: Tables[Frame]) -> Frame:
    """Global sales opportunity."""
    customer = t.customer[["c_custkey", "c_phone", "c_acctbal"]]
    customer = customer.assign(cntrycode=customer["c_phone"].str.slice(0, 2))
    customer = customer[customer["cntrycode"].isin(["13", "31", "23", "29", "30", "18", "17"])]
    average = float(customer[customer["c_acctbal"] > 0.0]["c_acctbal"].mean())
    customer = customer[
        (customer["c_acctbal"] > average) & ~customer["c_custkey"].isin(t.orders["o_custkey"])
    ]
    out = customer.groupby("cntrycode", as_index=False, sort=False).agg(
        numcust=("c_acctbal", "count"), totacctbal=("c_acctbal", "sum")
    )
    return out.sort_values("cntrycode")


QUERIES: dict[str, Query] = {
    "Q1": q1,
    "Q2": q2,
    "Q3": q3,
    "Q4": q4,
    "Q5": q5,
    "Q6": q6,
    "Q7": q7,
    "Q8": q8,
    "Q9": q9,
    "Q10": q10,
    "Q11": q11,
    "Q12": q12,
    "Q13": q13,
    "Q14": q14,
    "Q15": q15,
    "Q16": q16,
    "Q17": q17,
    "Q18": q18,
    "Q19": q19,
    "Q20": q20,
    "Q21": q21,
    "Q22": q22,
}
