"""TPC-H Q1, Q6, and Q14 on PyTorch tensors, with model-written operators (RQ3).

A tensor engine in the manner of TQP: a table is a tensor for each column, a
string column is the dictionary codes that the host computes as it loads
the table, a filter is a mask and a compaction, and a group-by adds each
row's values into one element for each group number, which PyTorch computes
from the codes of the keys. Each query does its row-wise work with four
operators that model-written Triton kernels of the corpus implement
(`q6_sum`, `projection`, `scatter_add`, `gather`). `MODELS` names the model
whose kernel each task runs: Fable's `q6_sum`, `projection`, and `gather`,
and Opus's `scatter_add` (`out[idx[i]] += values[i]`, a task of
`eval/corpus/synthesized/HELDOUT3.md` for which Fable did not write a
kernel), which the gate accepts as written. Q1's group number is
`returnflag * statuses + linestatus` over the codes, at most 3 * 2 = 6
groups, and each of its sums and its count is a `scatter_add`. Q1 used
Fable's `groupby_sum` before, a hash table whose insertion retries a
compare-and-swap; the gate refuses such a loop, since it does not prove that
the loop ends (design decision 40).

`Operators` has three implementations:

- `TorchOperators`: PyTorch's own operators, the engine without model code;
- `DirectOperators`: the models' kernels, launched unchecked by Triton
  through their own `run` functions;
- `GatedOperators`: the same kernels' recorded PTX, verified by the gate,
  through the guard pass when the gate refuses an access or barrier
  participation (`opgate.hosts.register`), and launched through the executor
  on the session tenant's tensors (`opgate.hosts.torch`), with the grids of
  their `run` functions. A gather's row ids and a `scatter_add`'s group
  numbers are an index input, so they are copied into a buffer of their own,
  which the guarded launch checks, kept for the next index input of as many
  rows. A launch over no rows is skipped, since an empty tensor has no
  address. A failed check of the guard pass raises
  `opgate.hosts.CheckFailedError`.

Each query gives the frame of the query in `eval.tpch.queries`. The part
keys are the part table's rows plus one, as `eval.tpch.database` generates
them, so a line's part is the row `l_partkey - 1`; `load` checks it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from types import ModuleType
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.sources import source_module
from eval.corpus.workloads import encode
from eval.tpch.data import day
from eval.tpch.database import Tables
from opgate.executor import Executor, Module
from opgate.hosts import register
from opgate.hosts.torch import TorchHost

__all__ = [
    "MODELS",
    "QUERIES",
    "DirectOperators",
    "GatedOperators",
    "Operators",
    "TorchOperators",
    "TorchTables",
    "load",
    "q1",
    "q6",
    "q14",
]

# the model whose Triton kernel runs each task
MODELS: dict[str, str] = {
    "q6_sum": "fable",
    "projection": "fable",
    "scatter_add": "opus",
    "gather": "fable",
}
Q1_CUTOFF = day(date(1998, 12, 1) - timedelta(days=90))
Q6_DATES = (day(date(1994, 1, 1)), day(date(1995, 1, 1)))
Q14_DATES = (day(date(1995, 9, 1)), day(date(1995, 10, 1)))


@dataclass(frozen=True)
class Lineitem:
    shipdate: torch.Tensor  # int32 days
    quantity: torch.Tensor  # float64, as are the price, the discount, and the tax
    price: torch.Tensor
    discount: torch.Tensor
    tax: torch.Tensor
    returnflag: torch.Tensor  # int32 codes into `TorchTables.returnflags`
    linestatus: torch.Tensor  # int32 codes into `TorchTables.linestatuses`
    part: torch.Tensor  # int32 row of the line's part


@dataclass(frozen=True)
class TorchTables:
    """The columns that the three queries read, on the device."""

    lineitem: Lineitem
    promo: torch.Tensor  # float64 per part row: 1 for a type that starts with PROMO
    returnflags: tuple[str, ...]  # each code's string, in increasing order
    linestatuses: tuple[str, ...]


def _codes(column: pd.Series) -> tuple[torch.Tensor, tuple[str, ...]]:
    categorical = pd.Categorical(column)
    codes = torch.as_tensor(np.asarray(categorical.codes, dtype=np.int32), device="cuda")
    return codes, tuple(str(c) for c in categorical.categories)


def _column(values: pd.Series, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values.to_numpy(), device="cuda").to(dtype)


def load(database: Tables[pd.DataFrame]) -> TorchTables:
    """The columns of the three queries as tensors of the current device allocator."""
    li, part = database.lineitem, database.part
    keys = part["p_partkey"].to_numpy()
    if not np.array_equal(keys, np.arange(1, len(keys) + 1)):
        raise ValueError("the part keys are not the part rows plus one")
    returnflag, returnflags = _codes(li["l_returnflag"])
    linestatus, linestatuses = _codes(li["l_linestatus"])
    promo = part["p_type"].str.startswith("PROMO").to_numpy(dtype=np.float64)
    return TorchTables(
        Lineitem(
            _column(li["l_shipdate"], torch.int32),
            _column(li["l_quantity"], torch.float64),
            _column(li["l_extendedprice"], torch.float64),
            _column(li["l_discount"], torch.float64),
            _column(li["l_tax"], torch.float64),
            returnflag,
            linestatus,
            _column(li["l_partkey"] - 1, torch.int32),
        ),
        torch.as_tensor(promo, device="cuda"),
        returnflags,
        linestatuses,
    )


class Operators(Protocol):
    """The row-wise operators of the queries."""

    def q6_sum(
        self,
        shipdate: torch.Tensor,
        discount: torch.Tensor,
        quantity: torch.Tensor,
        price: torch.Tensor,
        dates: tuple[int, int],
        discounts: tuple[float, float],
        quantity_max: float,
    ) -> float: ...

    def projection(self, price: torch.Tensor, discount: torch.Tensor) -> torch.Tensor: ...

    def scatter_add(self, index: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
        """`n` sums from zero: `values[i]` added to the sum `index[i]` for each row `i`."""

    def gather(self, column: torch.Tensor, rows: torch.Tensor) -> torch.Tensor: ...


class TorchOperators:
    """PyTorch's own operators."""

    def q6_sum(
        self,
        shipdate: torch.Tensor,
        discount: torch.Tensor,
        quantity: torch.Tensor,
        price: torch.Tensor,
        dates: tuple[int, int],
        discounts: tuple[float, float],
        quantity_max: float,
    ) -> float:
        keep = (
            (shipdate >= dates[0])
            & (shipdate < dates[1])
            & (discount >= discounts[0])
            & (discount <= discounts[1])
            & (quantity < quantity_max)
        )
        return float((price * discount)[keep].sum())

    def projection(self, price: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        return price * (1 - discount)

    def scatter_add(self, index: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
        return torch.zeros(n, dtype=torch.float64, device="cuda").index_add_(
            0, index.long(), values
        )

    def gather(self, column: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return column[rows.long()]


def _kernel(task: str) -> ModuleType:
    model = MODELS[task]
    return source_module(f"{model}_{task}", directory_of(model, task) / "triton.py")


class DirectOperators:
    """The models' kernels, launched unchecked through their `run` functions."""

    def __init__(self) -> None:
        self._run: dict[str, Callable[..., None]] = {task: _kernel(task).run for task in MODELS}

    def q6_sum(
        self,
        shipdate: torch.Tensor,
        discount: torch.Tensor,
        quantity: torch.Tensor,
        price: torch.Tensor,
        dates: tuple[int, int],
        discounts: tuple[float, float],
        quantity_max: float,
    ) -> float:
        result = torch.zeros(1, dtype=torch.float64, device="cuda")
        self._run["q6_sum"](
            shipdate,
            discount,
            quantity,
            price,
            *dates,
            *discounts,
            quantity_max,
            result,
            len(price),
        )
        return float(result.item())

    def projection(self, price: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(price)
        self._run["projection"](price, discount, out, len(price))
        return out

    def scatter_add(self, index: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.zeros(n, dtype=torch.float64, device="cuda")
        self._run["scatter_add"](index, values, out, len(index), n)
        return out

    def gather(self, column: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        out = torch.empty(len(rows), dtype=torch.float64, device="cuda")
        self._run["gather"](column, rows, out, len(rows), len(column))
        return out


class GatedOperators:
    """The models' kernels through the executor, on the session tenant's tensors;
    `register` verifies and loads them."""

    def __init__(self, host: TorchHost, modules: dict[str, Module]) -> None:
        self._host = host
        self._modules = modules
        self._kernels = {task: _kernel(task) for task in MODELS}
        self._processors = torch.cuda.get_device_properties(0).multi_processor_count
        self._indexes: dict[int, torch.Tensor] = {}  # a buffer of an index input for each count

    @staticmethod
    def register(executor: Executor) -> dict[str, Module]:
        """Each task's recorded PTX, verified by the gate, with checks on the
        accesses that it does not prove (`opgate.frontends.guard`), and loaded."""
        modules: dict[str, Module] = {}
        for task, model in MODELS.items():
            text = (directory_of(model, task) / "triton.ptx").read_text()
            registered = register(executor, text, binding_of(model, task, "triton", text))
            modules[task] = registered.module
        return modules

    def _programs(self, task: str, rows: int) -> tuple[int, int, int]:
        kernel = self._kernels[task]
        programs = -(-rows // kernel.CONSTANTS["BLOCK"])
        if task == "q6_sum":
            programs = min(programs, kernel.PROGRAMS_PER_SM * self._processors)
        return (max(1, programs), 1, 1)

    def _launch(
        self,
        task: str,
        rows: int,
        tensors: Sequence[torch.Tensor],
        sizes: Sequence[int],
        scalars: Sequence[int] = (),
        *,
        guarded: bool = False,
    ) -> None:
        if rows == 0:
            return
        grid = self._programs(task, rows)
        ran = self._host.launch(self._modules[task], grid, tensors, sizes, scalars, guarded=guarded)
        if not ran:
            raise RuntimeError(f"{task}: the guarded launch's check failed")

    def _index(self, index: torch.Tensor) -> torch.Tensor:
        """`index` copied into the buffer of its own kept for an index input of as
        many rows."""
        buffer = self._indexes.get(len(index))
        if buffer is None:
            buffer = self._indexes[len(index)] = self._host.buffer(len(index), torch.int32)
        buffer.copy_(index)
        return buffer

    def q6_sum(
        self,
        shipdate: torch.Tensor,
        discount: torch.Tensor,
        quantity: torch.Tensor,
        price: torch.Tensor,
        dates: tuple[int, int],
        discounts: tuple[float, float],
        quantity_max: float,
    ) -> float:
        result = torch.zeros(1, dtype=torch.float64, device="cuda")
        scalars = (
            *(encode("int32", d) for d in dates),
            *(encode("float64", d) for d in discounts),
            encode("float64", quantity_max),
        )
        rows = len(price)
        self._launch(
            "q6_sum", rows, (shipdate, discount, quantity, price, result), (rows,), scalars
        )
        return float(result.item())

    def projection(self, price: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(price)
        self._launch("projection", len(price), (price, discount, out), (len(price),))
        return out

    def scatter_add(self, index: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.zeros(n, dtype=torch.float64, device="cuda")
        rows = len(index)
        if rows == 0:
            return out
        self._launch(
            "scatter_add", rows, (self._index(index), values, out), (rows, n), guarded=True
        )
        return out

    def gather(self, column: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        out = torch.empty(len(rows), dtype=torch.float64, device="cuda")
        if len(rows) == 0:
            return out
        self._launch(
            "gather",
            len(rows),
            (column, self._index(rows), out),
            (len(column), len(rows)),
            guarded=True,
        )
        return out


def q1(ops: Operators, t: TorchTables) -> pd.DataFrame:
    """Pricing summary report."""
    li = t.lineitem
    keep = li.shipdate <= Q1_CUTOFF
    quantity, price, discount, tax = (c[keep] for c in (li.quantity, li.price, li.discount, li.tax))
    statuses = len(t.linestatuses)
    groups = len(t.returnflags) * statuses
    group = (li.returnflag[keep] * statuses + li.linestatus[keep]).to(torch.int32)
    disc_price = ops.projection(price, discount)
    charge = disc_price * (1 + tax)
    sums = {
        name: ops.scatter_add(group, value, groups).tolist()
        for name, value in (
            ("quantity", quantity),
            ("price", price),
            ("disc_price", disc_price),
            ("charge", charge),
            ("discount", discount),
            ("count", torch.ones_like(quantity)),
        )
    }
    rows = []
    for key, count in enumerate(sums["count"]):
        if count == 0:  # the filter keeps none of the group's lines
            continue
        rows.append(
            {
                "l_returnflag": t.returnflags[key // statuses],
                "l_linestatus": t.linestatuses[key % statuses],
                "sum_qty": sums["quantity"][key],
                "sum_base_price": sums["price"][key],
                "sum_disc_price": sums["disc_price"][key],
                "sum_charge": sums["charge"][key],
                "avg_qty": sums["quantity"][key] / count,
                "avg_price": sums["price"][key] / count,
                "avg_disc": sums["discount"][key] / count,
                "count_order": round(count),
            }
        )
    return pd.DataFrame(rows)


def q6(ops: Operators, t: TorchTables) -> pd.DataFrame:
    """Forecasting revenue change."""
    li = t.lineitem
    revenue = ops.q6_sum(
        li.shipdate, li.discount, li.quantity, li.price, Q6_DATES, (0.05, 0.07), 24.0
    )
    return pd.DataFrame({"revenue": [revenue]})


def q14(ops: Operators, t: TorchTables) -> pd.DataFrame:
    """Promotion effect."""
    li = t.lineitem
    keep = (li.shipdate >= Q14_DATES[0]) & (li.shipdate < Q14_DATES[1])
    part, price, discount = li.part[keep], li.price[keep], li.discount[keep]
    promo = ops.gather(t.promo, part)
    revenue = ops.projection(price, discount)
    share = 100.0 * float((revenue * promo).sum()) / float(revenue.sum())
    return pd.DataFrame({"promo_revenue": [share]})


QUERIES: dict[str, Callable[[Operators, TorchTables], pd.DataFrame]] = {
    "Q1": q1,
    "Q6": q6,
    "Q14": q14,
}
