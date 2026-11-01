"""The engine's built-in operators that serve a query when the gate refuses its kernel.

When the gate refuses a model-written `string_contains` or `groupby_sum`, the
mixed experiment (`eval.rq4_cost.mixed`) runs the engine's own operator for
the task in its place: libcudf's substring search and hash group-by, through
pylibcudf, with CuPy for the copies around them. They are part of the
engine's trusted base, as cuDF's filters, joins, and groupings are on the
TPC-H path of `eval.rq3_pipelines.service_queries`; no model-written code
runs. Each built-in operator takes the kernel's inputs and writes the
kernel's outputs, so the task's reference (`eval.corpus.workloads.check`)
checks them as it checks the kernel's:

* `builtin_string_contains`: `match[i]` is 1 if the string `i` of `offsets`
  and `chars` contains `pattern`, else 0. libcudf reads a string column's
  offsets unchecked, so the operator first requires that they start at 0,
  never decrease, and end at the length of `chars`.
* `builtin_groupby_sum`: the distinct values of `group` and the sum of
  `value` over each, in the first slots of `table_keys` and `table_sums`,
  and -1 and 0 in the others; more groups than slots is a refusal.

The service runs them as query nodes (`NODES`) on its owner thread, over
views of the session's buffers (`opgate.hosts.rapids.BufferView`), and
answers with the bytes of the output that the query reads back; a tenant
process under MPS runs the same functions over its own device memory
(`DeviceArray`). cuDF, CuPy, and RMM are imported when an operator runs, and
the service's types only for type checking, so that the operators'
definitions load without a GPU.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from opgate.executor import RefusedError

if TYPE_CHECKING:
    from opgate.hosts.service import Fields, NodeContext, QueryNode

__all__ = [
    "BUILTINS",
    "MAX_ROWS",
    "NODES",
    "Builtin",
    "CudaArray",
    "DeviceArray",
    "Node",
    "Operand",
    "read_back",
]

MAX_ROWS = 1 << 27  # the elements of one operand of a request


class CudaArray(Protocol):
    """A device array as the CUDA array interface describes it."""

    @property
    def __cuda_array_interface__(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class DeviceArray:
    """`count` elements of NumPy type string `typestr` at device address `pointer`."""

    pointer: int
    typestr: str
    count: int
    readonly: bool

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        return {
            "shape": (self.count,),
            "strides": None,
            "typestr": self.typestr,
            "data": (self.pointer, self.readonly),
            "version": 3,
        }


@dataclass(frozen=True)
class Operand:
    """A buffer argument of a built-in operator: its NumPy type string, its
    length as a size symbol plus a constant, and whether the operator writes it."""

    name: str
    typestr: str
    length: tuple[str, int]
    access: Literal["read", "write"]

    def count(self, sizes: Mapping[str, int]) -> int:
        symbol, extra = self.length
        return sizes[symbol] + extra


@dataclass(frozen=True)
class Builtin:
    """A built-in operator for a corpus task: its operands in the kernel's
    argument order, its size symbols, the output that a query reads back, and
    the function that computes the outputs from the operands' arrays."""

    op: str
    task: str
    operands: tuple[Operand, ...]
    sizes: tuple[str, ...]
    read: str
    compute: Callable[[Mapping[str, CudaArray]], None]

    def counts(self, sizes: Mapping[str, int]) -> dict[str, int]:
        """Each operand's elements under `sizes`."""
        return {operand.name: operand.count(sizes) for operand in self.operands}


def _bytes_of(column: object, dtype: object, count: int) -> object:
    """The first `count` elements of a fresh libcudf result column as a CuPy array."""
    import cupy
    import pylibcudf as plc

    if not isinstance(column, plc.Column):
        raise TypeError(f"{column!r} is not a pylibcudf column")
    if column.offset() != 0 or column.null_count() != 0:
        raise RuntimeError("a libcudf result has an offset or nulls")
    data = column.data()
    if data is None:
        raise RuntimeError("a libcudf result has no data")
    return cupy.asarray(data).view(dtype)[:count]


def _string_contains(arrays: Mapping[str, CudaArray]) -> None:
    import cupy
    import pylibcudf as plc

    offsets, chars = cupy.asarray(arrays["offsets"]), cupy.asarray(arrays["chars"])
    pattern, match = cupy.asarray(arrays["pattern"]), cupy.asarray(arrays["match"])
    n = len(match)
    if len(offsets) != n + 1:
        raise RefusedError(f"{len(offsets)} offsets for {n} strings")
    if (
        int(offsets[0]) != 0
        or int(offsets[n]) != len(chars)
        or bool(cupy.any(offsets[1:] < offsets[:-1]))
    ):
        raise RefusedError("the offsets are not a layout of the characters")
    try:
        target = cupy.asnumpy(pattern).tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise RefusedError("the pattern is not UTF-8") from error
    strings = plc.Column(
        plc.DataType(plc.TypeId.STRING),
        n,
        plc.gpumemoryview(chars),
        None,
        0,
        0,
        [plc.Column.from_cuda_array_interface(offsets)],
    )
    found = plc.strings.find.contains(strings, plc.Scalar.from_py(target))
    # a BOOL8 element is one byte, 0 or 1
    match[:] = _bytes_of(found, cupy.int8, n)


def _groupby_sum(arrays: Mapping[str, CudaArray]) -> None:
    import cupy
    import pylibcudf as plc

    group = plc.Column.from_cuda_array_interface(arrays["group"])
    value = plc.Column.from_cuda_array_interface(arrays["value"])
    keys, sums = cupy.asarray(arrays["table_keys"]), cupy.asarray(arrays["table_sums"])
    if group.size() != value.size() or len(keys) != len(sums):
        raise RefusedError("the operands' lengths differ")
    grouped = plc.groupby.GroupBy(plc.Table([group]))
    request = plc.groupby.GroupByRequest(value, [plc.aggregation.sum()])
    unique, (summed,) = grouped.aggregate([request])
    count = unique.num_rows()
    if count > len(keys):
        raise RefusedError(f"{count} groups exceed the table's {len(keys)} slots")
    ((distinct,), (total,)) = unique.columns(), summed.columns()
    keys.fill(-1)
    sums.fill(0)
    keys[:count] = _bytes_of(distinct, cupy.int32, count)
    sums[:count] = _bytes_of(total, cupy.float64, count)


BUILTINS: dict[str, Builtin] = {
    builtin.task: builtin
    for builtin in (
        Builtin(
            "builtin_string_contains",
            "string_contains",
            (
                Operand("offsets", "<i4", ("n", 1), "read"),
                Operand("chars", "|u1", ("chars", 0), "read"),
                Operand("pattern", "|u1", ("k", 0), "read"),
                Operand("match", "|i1", ("n", 0), "write"),
            ),
            ("n", "chars", "k"),
            "match",
            _string_contains,
        ),
        Builtin(
            "builtin_groupby_sum",
            "groupby_sum",
            (
                Operand("group", "<i4", ("n", 0), "read"),
                Operand("value", "<f8", ("n", 0), "read"),
                Operand("table_keys", "<i4", ("capacity", 0), "write"),
                Operand("table_sums", "<f8", ("capacity", 0), "write"),
            ),
            ("n", "capacity"),
            "table_sums",
            _groupby_sum,
        ),
    )
}
"""The built-in operator of each task that has one, by task."""


def read_back(arrays: Mapping[str, CudaArray], name: str) -> bytes:
    """The bytes of the array `name`, after the work before it on the legacy stream."""
    import cupy

    return cupy.asnumpy(cupy.asarray(arrays[name])).tobytes()


@dataclass(frozen=True)
class Node:
    """A built-in operator as the service's query node. The request names each
    operand's buffer handle (`buffers`) and each size symbol's value (`sizes`);
    the response holds the bytes of the output that the query reads back, as
    base64 (`data`)."""

    builtin: Builtin

    @property
    def op(self) -> str:
        return self.builtin.op

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        from opgate.hosts.rapids import BufferView

        request.exact("op", "buffers", "sizes")
        given, handles = request.fields("sizes"), request.fields("buffers")
        given.exact(*self.builtin.sizes)
        handles.exact(*(operand.name for operand in self.builtin.operands))
        sizes = {s: given.integer(s, minimum=0, maximum=MAX_ROWS) for s in self.builtin.sizes}
        counts = self.builtin.counts(sizes)
        views: dict[str, CudaArray] = {
            operand.name: BufferView.of(
                context.resource,
                context.buffer(handles, operand.name),
                operand.typestr,
                counts[operand.name],
                access=operand.access,
            )
            for operand in self.builtin.operands
        }
        self.builtin.compute(views)
        data = read_back(views, self.builtin.read)
        del views  # the writable views keep their buffers raw while they live
        return {"data": base64.b64encode(data).decode("ascii")}


NODES: tuple[QueryNode, ...] = tuple(Node(builtin) for builtin in BUILTINS.values())
