"""RQ2's corpus of cuDF's own UDFs: the examples of cuDF's guide to UDFs.

The guide, `docs/cudf/source/cudf/guide-to-udfs.ipynb` of cuDF 26.08, shows
each way to run a UDF in cuDF. Each `apply` example runs on the guide's frame
inside `install_cudf`, where the executor verifies and launches cuDF's kernel,
and its result is compared with the guide's; the hook refuses a UDF that it
does not bind, with the reason. Each custom Numba kernel that the guide
launches with `forall` is compiled by the Numba front end with the signature
of its arguments and decided by the gate, as an operator of the catalog is.

Run from the repository root:

    uv run python -m eval.rq2_coverage.cudf_udfs --output eval/records/coverage_cudf_udfs_<date>.json
"""

import argparse
import gc
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import cudf
import numpy as np
from numba import cuda, types

from opgate import signature as sig
from opgate.binding import ElementType
from opgate.catalog import TenantId
from opgate.executor import Executor, RefusedError
from opgate.frontends import Target
from opgate.frontends.numba_cuda import numba_operator
from opgate.gate import REGISTRATION_SECONDS, Accepted, Refused, VerificationBudget, verify, zeroed
from opgate.hosts.cudf import install_cudf
from opgate.hosts.rapids import CatalogResource
from opgate.signature import Affine, Direction

GUIDE = "cuDF 26.08, docs/cudf/source/cudf/guide-to-udfs.ipynb"
TARGET = Target(8, 6)
TENANT = TenantId(1)
BLOCK = (256, 1, 1)

Outcome = Literal["accepted", "refused"]


class Record(TypedDict):
    """One example of the guide: its path, its cell, and what the gate did."""

    name: str
    path: str
    cell: int
    outcome: Outcome
    reason: str
    result: str  # correct, wrong, or - for a refused example


@dataclass(frozen=True)
class Apply:
    """An `apply` example: the call on cuDF and the guide's result, None for a null."""

    name: str
    path: str
    cell: int
    call: Callable[[], object]
    expected: list[float | None]


def _add_one(x: int) -> int:
    return x + 1


def _add_const(x: int, const: int) -> int:
    return x + const


def _null_sensitive(x: object) -> object:
    if x is cudf.NA:
        return 42
    else:
        return x + 1  # ty: ignore[unsupported-operator] # the UDF sees a number or cudf.NA


def _string_test(st: str) -> int:
    if len(st) > 0:
        if st.startswith("a"):
            return 1
        elif "example" in st:
            return 2
        else:
            return -1
    else:
        return 42


def _row_sum(row: dict[str, float]) -> float:
    return row["A"] + row["B"]


def _row_null(row: dict[str, object]) -> object:
    x = row["a"]
    if x is cudf.NA:
        return 0
    else:
        return x + 1  # ty: ignore[unsupported-operator] # the UDF sees a number or cudf.NA


def _row_returns_null(row: dict[str, int]) -> object:
    x = row["a"]
    y = row["b"]
    if x + y > 3:
        return cudf.NA
    else:
        return x + y


def _row_mixed(row: dict[str, float]) -> float:
    return row["a"] + row["b"]


def _row_branch(row: dict[str, float]) -> float:
    x = row["a"]
    if x > 3:
        return x
    else:
        return 1.5


def _row_arithmetic(row: dict[str, float]) -> float:
    return row["a"] + (row["b"] - (row["c"] / row["d"])) % row["e"]


def _row_string(row: dict[str, object]) -> object:
    st = row["str_col"]
    scale = row["scale"]
    if len(st) > 5:  # ty: ignore[invalid-argument-type] # the UDF sees a string
        return len(st) + scale  # ty: ignore[invalid-argument-type, unsupported-operator]
    else:
        return len(st)  # ty: ignore[invalid-argument-type] # the UDF sees a string


def _udaf(df: cudf.DataFrame) -> float:
    return df["b"].max() - df["b"].min() / 2


def _window_max_sqrt(window: np.ndarray) -> float:
    b = 0.0
    for a in window:
        b = max(b, math.sqrt(a))
    if b == 8:
        return 100
    return b


APPLIES = (
    Apply(
        "series +1", "Series.apply", 8, lambda: cudf.Series([1, 2, 3]).apply(_add_one), [2, 3, 4]
    ),
    Apply(
        "series + argument",
        "Series.apply",
        13,
        lambda: cudf.Series([1, 2, 3]).apply(_add_const, args=(42,)),
        [43, 44, 45],
    ),
    Apply(
        "series with a null",
        "Series.apply",
        19,
        lambda: cudf.Series([1, cudf.NA, 3]).apply(_add_one),
        [2, None, 4],
    ),
    Apply(
        "null-sensitive series",
        "Series.apply",
        22,
        lambda: cudf.Series([1, cudf.NA, 3]).apply(_null_sensitive),
        [2, 42, 4],
    ),
    Apply(
        "string series",
        "Series.apply",
        28,
        lambda: cudf.Series(["", "abc", "some_example"]).apply(_string_test),
        [42, 1, 2],
    ),
    Apply(
        "row sum",
        "DataFrame.apply",
        45,
        lambda: cudf.DataFrame({"A": [1, 2, 3], "B": [4, cudf.NA, 6]}).apply(_row_sum, axis=1),
        [5, None, 9],
    ),
    Apply(
        "row with a null",
        "DataFrame.apply",
        54,
        lambda: cudf.DataFrame({"a": [1, cudf.NA, 3]}).apply(_row_null, axis=1),
        [2, 0, 4],
    ),
    Apply(
        "row returning a null",
        "DataFrame.apply",
        57,
        lambda: cudf.DataFrame({"a": [1, 2, 3], "b": [2, 1, 1]}).apply(_row_returns_null, axis=1),
        [3, 3, None],
    ),
    Apply(
        "row of mixed types",
        "DataFrame.apply",
        60,
        lambda: cudf.DataFrame({"a": [1, 2, 3], "b": [0.5, cudf.NA, 3.14]}).apply(
            _row_mixed, axis=1
        ),
        [1.5, None, 6.14],
    ),
    Apply(
        "row with a branch",
        "DataFrame.apply",
        63,
        lambda: cudf.DataFrame({"a": [1, 3, 5]}).apply(_row_branch, axis=1),
        [1.5, 1.5, 5.0],
    ),
    Apply(
        "row arithmetic",
        "DataFrame.apply",
        66,
        lambda: cudf.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [cudf.NA, 4, 4],
                "d": [8, 7, 8],
                "e": [7, 1, 6],
            }
        ).apply(_row_arithmetic, axis=1),
        [None, 2.428571428571429, 8.5],
    ),
    Apply(
        "row with a string",
        "DataFrame.apply",
        71,
        lambda: cudf.DataFrame({"str_col": ["abc", "ABC", "Example"], "scale": [1, 2, 3]}).apply(
            _row_string, axis=1
        ),
        [3, 3, 10],
    ),
    Apply(
        "group aggregate",
        "GroupBy.apply",
        76,
        lambda: (
            cudf.DataFrame(
                {"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 4, 5, 6], "c": [3, 4, 5, 6, 7, 8]}
            )
            .groupby("a")
            .apply(_udaf, engine="jit")
        ),
        [2.5, 4.0],
    ),
    Apply(
        "rolling window",
        "Rolling.apply",
        86,
        lambda: (
            cudf.Series([16, 25, 36, 49, 64, 81], dtype="float64")
            .rolling(window=3, min_periods=3, center=False)
            .apply(_window_max_sqrt)
        ),
        [None, None, 6.0, 7.0, 100.0, 9.0],
    ),
)


def _values(result: object) -> list[float | None]:
    """The result's values, None for a null."""
    if not isinstance(result, cudf.Series):
        raise TypeError(f"an apply result of type {type(result).__name__}")
    return [None if v is None or v is cudf.NA else float(v) for v in result.to_arrow().to_pylist()]


def _close(got: list[float | None], expected: list[float | None]) -> bool:
    return len(got) == len(expected) and all(
        (g is None and e is None) or (g is not None and e is not None and math.isclose(g, e))
        for g, e in zip(got, expected, strict=True)
    )


def _refusal(error: BaseException) -> RefusedError | None:
    """The hook's refusal that `error` is or that cuDF wrapped."""
    while error is not None:
        if isinstance(error, RefusedError):
            return error
        error = error.__cause__ or error.__context__  # ty: ignore[invalid-assignment]
    return None


def run_apply(example: Apply) -> Record:
    """Run the example inside the hook: its result, or the hook's refusal."""
    try:
        got = _values(example.call())
    except Exception as error:
        # cuDF wraps an error of the kernel's compilation; the refusal is its cause
        refusal = _refusal(error)
        if refusal is None:
            raise
        return Record(
            name=example.name,
            path=example.path,
            cell=example.cell,
            outcome="refused",
            reason=refusal.reason,
            result="-",
        )
    return Record(
        name=example.name,
        path=example.path,
        cell=example.cell,
        outcome="accepted",
        reason="",
        result="correct" if _close(got, example.expected) else "wrong",
    )


def _multiply(in_col: np.ndarray, out_col: np.ndarray, multiplier: float) -> None:
    i = cuda.grid(1)
    if i < in_col.size:  # boundary guard
        out_col[i] = in_col[i] * multiplier


def _multiply_by_5(x: np.ndarray, out: np.ndarray) -> None:
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] * 5


@dataclass(frozen=True)
class Kernel:
    """A custom kernel of the guide with the types and the signature of its arguments."""

    name: str
    cell: int
    function: Callable[..., None]
    argument_types: tuple[types.Type, ...]
    signature: sig.Signature
    elements: dict[str, ElementType]


ROWS = sig.SizeSymbol("n")
TWO_COLUMNS = sig.Signature(
    (ROWS,),
    (
        sig.Span("x", "x", Direction.READ, Affine.symbol(0)),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
    ),
)
KERNELS = (
    # the guide's frame of random integers, and a float64 column for the product
    Kernel(
        "multiply",
        36,
        _multiply,
        (types.int64[::1], types.float64[::1], types.float64),
        sig.Signature(
            (ROWS,),
            (
                sig.Span("in_col", "in_col", Direction.READ, Affine.symbol(0)),
                sig.Span("out_col", "out_col", Direction.WRITE, Affine.symbol(0)),
                sig.Scalar("multiplier", sig.Between(0, 2**64 - 1)),
            ),
        ),
        {"in_col": ElementType(8, True), "out_col": ElementType(8, False)},
    ),
    # a float64 series into an int32 series, then a CuPy array into one of its kind
    Kernel(
        "multiply_by_5 into a series",
        93,
        _multiply_by_5,
        (types.float64[::1], types.int32[::1]),
        TWO_COLUMNS,
        {"x": ElementType(8, False), "out": ElementType(4, True)},
    ),
    Kernel(
        "multiply_by_5 into an array",
        95,
        _multiply_by_5,
        (types.float64[::1], types.float64[::1]),
        TWO_COLUMNS,
        {"x": ElementType(8, False), "out": ElementType(8, False)},
    ),
)


def decide_kernel(kernel: Kernel) -> Record:
    """The gate's decision on the custom kernel, compiled by the Numba front end,
    under the service's budget of one registration."""
    operator = numba_operator(
        kernel.function, kernel.argument_types, kernel.signature, BLOCK, TARGET
    )
    binding = operator.binding(kernel.elements)
    match VerificationBudget(checker=verify)(zeroed(operator.text, binding), binding):
        case Accepted():
            outcome: Outcome = "accepted"
            reason = ""
        case Refused(reason=refused):
            outcome = "refused"
            reason = refused
        case _ as never:
            assert_never(never)
    return Record(
        name=kernel.name,
        path="forall",
        cell=kernel.cell,
        outcome=outcome,
        reason=reason,
        result="-",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: list[Record] = []
    with Executor(0, 1 << 28) as executor, CatalogResource(executor) as resource:
        with resource.current():
            with executor.tenant(TENANT), install_cudf(executor, resource, TARGET):
                executor.reserve(1 << 28)
                for example in APPLIES:
                    records.append(run_apply(example))
                    print(records[-1], flush=True)
                # free the examples' frames while their tenant's session lasts
                gc.collect()
    for kernel in KERNELS:
        records.append(decide_kernel(kernel))
        print(records[-1], flush=True)
    record = {
        "date": date.today().isoformat(),
        "source": GUIDE,
        "registration_seconds": REGISTRATION_SECONDS,
        "examples": records,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
