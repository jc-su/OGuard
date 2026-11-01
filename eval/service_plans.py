"""The plan schemas and query nodes that the evaluation's service deployments supply.

The service boundary experiment (`eval/rq1_safety/service_boundary.py`) and
the consolidation experiment (`eval/rq4_cost/consolidation.py`) start the
service with these:

- schemas `elementwise_f64` (the safe control's kernel: `output[i] = f(source[i])`
  over n float64 values), `read_one_i64` (one int64 read from a source of n
  values into one output, at a fixed index or at the one the kernel
  chooses: the paired replay's probe), and `tpch_q6_sum` (a TPC-H Q6 kernel
  over four lineitem columns into one float64 result);
- query nodes `cudf_linear` (a cuDF `Series.apply` UDF over request values),
  `cudf_tpch_q6` (Q6 as a cuDF `DataFrame.apply` row UDF and a cuDF sum over
  four of the session's columns), and `tpch_q6` (Q6 by a registered
  `tpch_q6_sum` operator over the same columns, launched through
  `opgate.hosts.execute`, which gives an operator with the guard pass's
  checks its status buffer).

Every service deployment of the evaluation runs `VERIFICATION_WORKERS`
verification threads (`Limits.verification_workers`): the checks that the
registration and guarded records ran at a time, so that the cold-query
table's schedule of a plan's checks (`eval.export.rq4.export_cold_query`)
uses times measured with as many checks running.
"""

import struct
from dataclasses import dataclass

import numpy as np

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Binding, Convention, Direction, ElementType
from opgate.catalog import Fill, Launch, Step, Write
from opgate.executor import RefusedError
from opgate.frontends import Operator, entry_kernels
from opgate.hosts import CheckFailedError, execute
from opgate.hosts.service import (
    Fields,
    LaunchRequest,
    NodeContext,
    PlanSchema,
    QueryNode,
    RequestError,
)

__all__ = ["MAX_ROWS", "NODES", "SCHEMAS", "VERIFICATION_WORKERS"]

MAX_ROWS = 1 << 27  # the rows of one query node's request
VERIFICATION_WORKERS = 28  # the service's verification threads
BLOCK = 256
DOUBLE = ElementType(8, False)  # float64 values, as their bits
LONG = ElementType(8, True)


def _kernel(text: str) -> str:
    kernels = entry_kernels(text)
    if len(kernels) != 1:
        raise RequestError(f"the module must define one entry kernel, not {len(kernels)}")
    return kernels[0]


def _grid(threads: int) -> tuple[int, int, int]:
    return (max(1, -(-threads // BLOCK)), 1, 1)


def _bits(value: float) -> int:
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


N = sig.Affine.symbol(0)


@dataclass(frozen=True)
class ElementwisePlan:
    binding: Binding
    kind: str = "elementwise_f64"

    def launch(self, request: Fields) -> LaunchRequest:
        request.exact("op", "operator", "source", "output", "n")
        n = request.integer("n", minimum=1, maximum=MAX_ROWS)
        return LaunchRequest(
            _grid(n), (request.text("source"), request.text("output")), (n,), (), False
        )


@dataclass(frozen=True)
class ElementwiseF64:
    kind: str = "elementwise_f64"

    def plan(self, text: str, fields: Fields) -> ElementwisePlan:
        fields.exact("kind")
        signature = sig.Signature(
            (sig.SizeSymbol("n", 1),),
            (
                sig.Span("source", "source", Direction.READ, N),
                sig.Span("output", "output", Direction.WRITE, N),
                sig.Scalar("n", sig.SizeOf(N)),
            ),
        )
        operator = Operator(
            text, _kernel(text), signature, Convention.C, (BLOCK, 1, 1), GRID_LIMIT, sig.Affine(0)
        )
        return ElementwisePlan(operator.binding({"source": DOUBLE, "output": DOUBLE}))


@dataclass(frozen=True)
class ReadOnePlan:
    binding: Binding
    index: int | None  # the fixed source index, a scalar of the launch
    kind: str = "read_one_i64"

    def launch(self, request: Fields) -> LaunchRequest:
        request.exact("op", "operator", "source", "output", "n")
        n = request.integer("n", minimum=1, maximum=MAX_ROWS)
        scalars = () if self.index is None else (self.index,)
        return LaunchRequest(
            (1, 1, 1), (request.text("source"), request.text("output")), (n,), scalars, False
        )


@dataclass(frozen=True)
class ReadOneI64:
    kind: str = "read_one_i64"

    def plan(self, text: str, fields: Fields) -> ReadOnePlan:
        fields.exact("kind", "source_index")
        index = fields.optional_integer("source_index", minimum=0, maximum=2**63 - 1)
        arguments: list[sig.Argument] = [
            sig.Span("source", "source", Direction.READ, N),
            sig.Span("output", "output", Direction.WRITE, sig.Affine(1)),
        ]
        if index is not None:
            arguments.append(sig.Scalar("source_index", sig.Between(index, index)))
        signature = sig.Signature((sig.SizeSymbol("n", 1),), tuple(arguments))
        operator = Operator(
            text, _kernel(text), signature, Convention.NUMBA, (1, 1, 1), GRID_LIMIT, sig.Affine(0)
        )
        return ReadOnePlan(operator.binding({"source": LONG, "output": LONG}), index)


@dataclass(frozen=True)
class TpchQ6Plan:
    binding: Binding
    kind: str = "tpch_q6_sum"

    def launch(self, request: Fields) -> LaunchRequest:
        raise RequestError("a tpch_q6_sum operator runs through the tpch_q6 query node")


@dataclass(frozen=True)
class TpchQ6Sum:
    kind: str = "tpch_q6_sum"

    def plan(self, text: str, fields: Fields) -> TpchQ6Plan:
        fields.exact("kind")
        signature = sig.Signature(
            (sig.SizeSymbol("n", 0),),
            (
                sig.Span("shipdate", "shipdate", Direction.READ, N),
                sig.Span("discount", "discount", Direction.READ, N),
                sig.Span("quantity", "quantity", Direction.READ, N),
                sig.Span("price", "price", Direction.READ, N),
                sig.Span("result", "result", Direction.READ_WRITE, sig.Affine(1)),
                sig.Scalar("date_lo", sig.Between(0, 2**32 - 1)),
                sig.Scalar("date_hi", sig.Between(0, 2**32 - 1)),
                sig.Scalar("disc_lo", sig.Between(0, 2**64 - 1)),
                sig.Scalar("disc_hi", sig.Between(0, 2**64 - 1)),
                sig.Scalar("qty_max", sig.Between(0, 2**64 - 1)),
                sig.Scalar("n", sig.SizeOf(N)),
            ),
        )
        operator = Operator(
            text,
            _kernel(text),
            signature,
            Convention.NUMBA,
            (BLOCK, 1, 1),
            GRID_LIMIT,
            sig.Affine(0),
        )
        elements = {
            "shipdate": ElementType(4, True),
            "discount": DOUBLE,
            "quantity": DOUBLE,
            "price": DOUBLE,
            "result": DOUBLE,
        }
        return TpchQ6Plan(operator.binding(elements))


@dataclass(frozen=True)
class Q6:
    """The parameters of a Q6 request."""

    rows: int
    date_lo: int
    date_hi: int
    discount_lo: float
    discount_hi: float
    quantity_max: float

    @staticmethod
    def of(request: Fields) -> "Q6":
        rows = request.integer("rows", minimum=1, maximum=MAX_ROWS)
        date_lo = request.integer("date_lo", minimum=-(2**31), maximum=2**31 - 1)
        date_hi = request.integer("date_hi", minimum=-(2**31), maximum=2**31 - 1)
        if date_hi <= date_lo:
            raise RequestError(f"{request.where}: the dates are not ordered")
        low, high = request.number("discount_lo"), request.number("discount_hi")
        if high < low:
            raise RequestError(f"{request.where}: the discount bounds are reversed")
        return Q6(rows, date_lo, date_hi, low, high, request.number("quantity_max"))


COLUMNS = ("shipdate", "discount", "quantity", "extendedprice")
Q6_FIELDS = ("rows", "date_lo", "date_hi", "discount_lo", "discount_hi", "quantity_max")


def _linear(value: float, slope: float, intercept: float) -> float:
    """The UDF of the `cudf_linear` node."""
    return value * slope + intercept


def _q6_row(
    row: dict[str, float],
    date_lo: int,
    date_hi: int,
    discount_lo: float,
    discount_hi: float,
    quantity_max: float,
) -> float:
    """The revenue of one lineitem row under Q6: the row UDF of the `cudf_tpch_q6` node."""
    if (
        row["shipdate"] >= date_lo
        and row["shipdate"] < date_hi
        and row["discount"] >= discount_lo
        and row["discount"] <= discount_hi
        and row["quantity"] < quantity_max
    ):
        return row["extendedprice"] * row["discount"]
    return 0.0


@dataclass(frozen=True)
class CudfLinear:
    op: str = "cudf_linear"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        import cudf

        from opgate.hosts.rapids import BufferView

        request.exact("op", "values", "slope", "intercept")
        raw = request.values("values")
        if not 0 < len(raw) <= MAX_ROWS:
            raise RequestError(f"cudf_linear.values must hold 1 to {MAX_ROWS} numbers")
        values = np.asarray(
            [_number(v, f"cudf_linear.values[{i}]") for i, v in enumerate(raw)], np.float64
        )
        slope, intercept = request.number("slope"), request.number("intercept")
        before = len(context.executor.catalog.operators)
        source = context.resource.allocate(values.nbytes)
        try:
            context.executor.execute(Step(context.tenant, (Write(source, values.tobytes(), None),)))
            series = cudf.Series(
                BufferView.of(context.resource, source, "<f8", len(values), access="read")
            )
            result = series.apply(_linear, args=(np.float64(slope), np.float64(intercept)))
            host = result.to_numpy().tolist()
            del result, series
        finally:
            context.resource.free(source)
        return {
            "values": host,
            "gated_operators_added": len(context.executor.catalog.operators) - before,
        }


def _number(value: object, where: str) -> float:
    return Fields({"value": value}, where).number("value")


@dataclass(frozen=True)
class CudfTpchQ6:
    op: str = "cudf_tpch_q6"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        import cudf

        from opgate.hosts.rapids import BufferView

        request.exact("op", *COLUMNS, *Q6_FIELDS)
        query = Q6.of(request)
        buffers = {name: context.buffer(request, name) for name in COLUMNS}
        before = len(context.executor.catalog.operators)
        types = {"shipdate": "<i4", "discount": "<f8", "quantity": "<f8", "extendedprice": "<f8"}
        frame = cudf.DataFrame(
            {
                name: cudf.Series(
                    BufferView.of(
                        context.resource, buffers[name], types[name], query.rows, access="read"
                    )
                )
                for name in COLUMNS
            }
        )
        result = frame.apply(
            _q6_row,
            axis=1,
            args=(
                np.int32(query.date_lo),
                np.int32(query.date_hi),
                np.float64(query.discount_lo),
                np.float64(query.discount_hi),
                np.float64(query.quantity_max),
            ),
        )
        revenue = float(result.sum())
        del result, frame
        return {
            "revenue": revenue,
            "rows": query.rows,
            "gated_operators_added": len(context.executor.catalog.operators) - before,
        }


@dataclass(frozen=True)
class TpchQ6:
    op: str = "tpch_q6"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        request.exact("op", "operator", *COLUMNS, *Q6_FIELDS)
        query = Q6.of(request)
        module = context.operator(request, "operator", "tpch_q6_sum")
        inputs = tuple(context.buffer(request, name) for name in COLUMNS)
        result = context.resource.allocate(8)
        try:
            scalars = (
                query.date_lo % 2**32,
                query.date_hi % 2**32,
                _bits(query.discount_lo),
                _bits(query.discount_hi),
                _bits(query.quantity_max),
            )
            launch = Launch(
                module.operator, _grid(query.rows), (*inputs, result), (query.rows,), scalars, False
            )
            step = Step(context.tenant, (Fill(result, DOUBLE, 0), launch))
            try:
                ran = execute(context.executor, context.resource, step)
            except CheckFailedError:
                context.launched(module)  # the kernel ran and stored its status
                raise
            context.executor.synchronize()
            if not all(ran):
                raise RefusedError("the Q6 launch did not run")
            context.launched(module)
            (revenue,) = struct.unpack("<d", context.executor.read(context.tenant, result))
        finally:
            context.resource.free(result)
        return {"revenue": revenue, "rows": query.rows, "ran": list(ran)}


SCHEMAS: tuple[PlanSchema, ...] = (ElementwiseF64(), ReadOneI64(), TpchQ6Sum())
NODES: tuple[QueryNode, ...] = (CudfLinear(), CudfTpchQ6(), TpchQ6())
