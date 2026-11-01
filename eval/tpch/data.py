"""The TPC-H columns that the pipelines read.

`generate(scale, seed)` draws the orders and lineitem columns with the value
distributions of the TPC-H specification, clause 4.2.3: sparse order keys (8
of every 32), order dates uniform in [1992-01-01, 1998-12-31 - 151 days], 1 to
7 lines per order, quantities uniform in [1, 50], the extended price from the
part's retail price, discounts in [0.00, 0.10] and taxes in [0.00, 0.08] in
steps of 0.01, ship dates 1 to 121 days after the order date, receipt dates 1
to 30 days after the ship date, and the return flag and line status from the
current date 1995-06-17. It is not dbgen: it draws with NumPy's PCG64 from a
seed, so its rows differ from dbgen's rows, and it has no text columns. The
return flag and line status are coded as integers: flag R, A, N as 0, 1, 2 and
status O, F as 0, 1. A date is the number of days since 1970-01-01.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

__all__ = ["Lineitem", "Orders", "day", "generate"]

START = date(1992, 1, 1)
END = date(1998, 12, 31)
CURRENT = date(1995, 6, 17)
ORDERS_PER_SCALE = 1_500_000
PARTS_PER_SCALE = 200_000


def day(value: date) -> int:
    """The days since 1970-01-01."""
    return (value - date(1970, 1, 1)).days


@dataclass(frozen=True)
class Orders:
    orderkey: np.ndarray  # int64
    orderdate: np.ndarray  # int32 days


@dataclass(frozen=True)
class Lineitem:
    orderkey: np.ndarray  # int64
    quantity: np.ndarray  # float64
    extendedprice: np.ndarray  # float64
    discount: np.ndarray  # float64
    tax: np.ndarray  # float64
    shipdate: np.ndarray  # int32 days
    returnflag: np.ndarray  # int32: R, A, N as 0, 1, 2
    linestatus: np.ndarray  # int32: O, F as 0, 1


def _retail_price(partkey: np.ndarray) -> np.ndarray:
    return (90000 + (partkey // 10) % 20001 + 100 * (partkey % 1000)) / 100


def generate(scale: float, seed: int) -> tuple[Orders, Lineitem]:
    """The orders and lineitem columns at scale factor `scale`."""
    rng = np.random.Generator(np.random.PCG64(seed))
    count = max(1, round(ORDERS_PER_SCALE * scale))
    index = np.arange(count, dtype=np.int64)
    orderkey = (index // 8) * 32 + index % 8 + 1
    orderdate = rng.integers(day(START), day(END) - 151, count, endpoint=True).astype(np.int32)
    lines = rng.integers(1, 7, count, endpoint=True)
    owner = np.repeat(index, lines)
    rows = len(owner)
    quantity = rng.integers(1, 50, rows, endpoint=True).astype(np.float64)
    partkey = rng.integers(1, max(1, round(PARTS_PER_SCALE * scale)), rows, endpoint=True)
    discount = rng.integers(0, 10, rows, endpoint=True) / 100
    tax = rng.integers(0, 8, rows, endpoint=True) / 100
    shipdate = (orderdate[owner] + rng.integers(1, 121, rows, endpoint=True)).astype(np.int32)
    receiptdate = shipdate + rng.integers(1, 30, rows, endpoint=True)
    returned = receiptdate <= day(CURRENT)
    returnflag = np.where(returned, rng.integers(0, 1, rows, endpoint=True), 2).astype(np.int32)
    linestatus = np.where(shipdate > day(CURRENT), 0, 1).astype(np.int32)
    lineitem = Lineitem(
        orderkey[owner],
        quantity,
        quantity * _retail_price(partkey),
        discount,
        tax,
        shipdate,
        returnflag,
        linestatus,
    )
    return Orders(orderkey, orderdate), lineitem
