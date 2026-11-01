"""Complete shipping-priority query with verified projection and trusted cuDF.

The deployment fixes input column types and the query. Clients supply opaque
numeric buffer handles and bounded UTF-8 values, plus a registered projection
operator. cuDF constructs string columns, filters, joins, groups, and sorts
inside the owner's catalog resource. The submitted projection PTX alone is
untrusted generated code; it computes revenue through the ordinary executor.
"""

import base64
from dataclasses import dataclass
from datetime import date

import numpy as np

from opgate import signature as sig
from opgate.binding import Convention, Direction, ElementType
from opgate.catalog import Launch, Step
from opgate.executor import RefusedError
from opgate.hosts import CheckFailedError, execute
from opgate.hosts.service import Fields, NodeContext, OperatorSchema
from opgate.hosts.service.protocol import RequestError

MAX_ROWS = 1 << 24
MAX_STRING_BYTES = 16 << 20
BLOCK = (256, 1, 1)
N = sig.Affine.symbol(0)
REVENUE = OperatorSchema(
    "discounted_revenue",
    sig.Signature(
        (sig.SizeSymbol("n"),),
        (
            sig.Span("price", "price", Direction.READ, N),
            sig.Span("discount", "discount", Direction.READ, N),
            sig.Span("out", "out", Direction.WRITE, N),
            sig.Scalar("n", sig.SizeOf(N)),
        ),
    ),
    tuple((name, ElementType(8, False)) for name in ("price", "discount", "out")),
    Convention.NUMBA,
    BLOCK,
)

# None denotes explicitly supplied UTF-8 values, not unchecked device offsets.
Q3_COLUMNS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "customer": (("c_custkey", "<i8"), ("c_mktsegment", None)),
    "orders": (
        ("o_orderkey", "<i8"),
        ("o_custkey", "<i8"),
        ("o_orderdate", "<i4"),
        ("o_shippriority", "<i4"),
    ),
    "lineitem": (
        ("l_orderkey", "<i8"),
        ("l_extendedprice", "<f8"),
        ("l_discount", "<f8"),
        ("l_shipdate", "<i4"),
    ),
}
Q3_OUTPUT = (
    ("l_orderkey", "<i8"),
    ("revenue", "<f8"),
    ("o_orderdate", "<i4"),
    ("o_shippriority", "<i4"),
)


def utf8_values(fields: Fields, name: str, rows: int) -> list[str]:
    """Validate a complete non-null string column before the engine reads it."""
    values = fields.values(name)
    if len(values) != rows:
        raise RequestError(f"{name} must contain exactly {rows} strings")
    size = 0
    strings = []
    for value in values:
        if not isinstance(value, str):
            raise RequestError(f"{name} must contain exactly {rows} strings")
        try:
            size += len(value.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise RequestError(f"{name} contains invalid UTF-8") from error
        if size > MAX_STRING_BYTES:
            raise RequestError(f"{name} exceeds the UTF-8 byte limit")
        strings.append(value)
    return strings


def frame(context: NodeContext, request: Fields, columns: tuple[tuple[str, str | None], ...]):
    """A deployment-typed table: authorized views and bounded trusted strings."""
    import cudf

    from opgate.hosts.rapids import BufferView

    request.exact("rows", "columns")
    rows = request.integer("rows", minimum=0, maximum=MAX_ROWS)
    fields = request.fields("columns")
    fields.exact(*(name for name, _ in columns))
    # Resolve every input before materializing any engine column.
    inputs = {
        name: utf8_values(fields, name, rows)
        if dtype is None
        else BufferView.of(
            context.resource, context.buffer(fields, name), dtype, rows, access="read"
        )
        for name, dtype in columns
    }
    return cudf.DataFrame(
        {
            name: cudf.Series(inputs[name], dtype="str" if dtype is None else dtype)
            for name, dtype in columns
        }
    )


def result_columns(frame, columns: tuple[tuple[str, str], ...]) -> dict[str, object]:
    """Retain actual output bytes, including floating-point values, in the response."""
    result = []
    for name, dtype in columns:
        values = np.ascontiguousarray(frame[name].to_numpy(), dtype=dtype)
        result.append(
            {
                "name": name,
                "dtype": dtype,
                "data": base64.b64encode(values.tobytes()).decode("ascii"),
            }
        )
    return {"rows": len(frame), "columns": result}


def _engine_f64(context: NodeContext, column):
    """Name a contiguous, non-null engine allocation through the catalog."""
    if column.null_count or np.dtype(column.dtype) != np.dtype("<f8"):
        raise RefusedError("the revenue operator needs non-null float64 engine columns")
    interface = column.__cuda_array_interface__
    if interface.get("strides") not in (None, (8,)):
        raise RefusedError("the revenue operator needs contiguous engine columns")
    pointer = interface["data"][0]
    try:
        return context.resource.argument_at(pointer)
    except ValueError as error:
        raise RefusedError("the engine column must start at a catalog allocation") from error


@dataclass(frozen=True)
class ShippingPriority:
    """TPC-H Q3 with a verified generated revenue projection between join and sum."""

    op: str = "tpch_q3"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        import cudf

        from opgate.hosts.rapids import BufferView

        request.exact("op", "operator", *Q3_COLUMNS)
        module = context.operator(request, "operator", REVENUE.kind)
        customer, orders, lineitem = (
            frame(context, request.fields(name), columns) for name, columns in Q3_COLUMNS.items()
        )
        cutoff = (date(1995, 3, 15) - date(1970, 1, 1)).days
        customer = customer[customer["c_mktsegment"] == "BUILDING"][["c_custkey"]]
        orders = orders[orders["o_orderdate"] < cutoff]
        lineitem = lineitem[lineitem["l_shipdate"] > cutoff]
        joined = customer.merge(orders, left_on="c_custkey", right_on="o_custkey").merge(
            lineitem, left_on="o_orderkey", right_on="l_orderkey"
        )
        rows = len(joined)
        if not rows:
            return {
                "rows": 0,
                "columns": [
                    {"name": name, "dtype": dtype, "data": ""} for name, dtype in Q3_OUTPUT
                ],
                "projection_rows": 0,
                "generated_launches": 0,
            }
        price = _engine_f64(context, joined["l_extendedprice"])
        discount = _engine_f64(context, joined["l_discount"])
        result = context.resource.allocate(rows * 8)
        try:
            # cuDF and the executor own their scheduling; establish this boundary
            # before the generated kernel reads the engine's join result.
            context.executor.synchronize()
            step = Step(
                context.tenant,
                (
                    Launch(
                        module.operator,
                        (-(-rows // BLOCK[0]), 1, 1),
                        (price, discount, result),
                        (rows,),
                        (),
                        False,
                    ),
                ),
            )
            try:
                ran = execute(context.executor, context.resource, step)
            except CheckFailedError:
                context.launched(module)  # the kernel ran and stored its status
                raise
            context.executor.synchronize()
            if ran != (True,):
                raise RefusedError("the revenue projection did not run")
            context.launched(module)
            view = BufferView.of(context.resource, result, "<f8", rows, access="read")
            joined = joined.assign(revenue=cudf.Series(view, index=joined.index))
            out = joined.groupby(
                ["l_orderkey", "o_orderdate", "o_shippriority"], as_index=False, sort=False
            ).agg(revenue=("revenue", "sum"))
            out = out.sort_values(
                ["revenue", "o_orderdate", "l_orderkey"], ascending=[False, True, True]
            ).head(10)
            return {
                **result_columns(out, Q3_OUTPUT),
                "projection_rows": rows,
                "generated_launches": 1,
            }
        finally:
            context.resource.free(result)
