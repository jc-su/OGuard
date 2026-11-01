"""TPC-H pipelines of one tenant through the executor.

Each pipeline is steps that the catalog checks: the tenant's columns enter
the pool by write actions, and every untrusted operator runs as a launch.

- `q6`: TPC-H Q6 as one launch of the corpus kernel `q6_sum`.
- `revenue`: the revenue `sum(price * (1 - discount))` of the lines with
  `quantity < 24`, the sum of Q1's `sum_disc_price` over those lines, as a
  pipeline with index intermediates: the filter writes the selected row
  ids, the host reads their number, two `gather` launches take the price and
  the discount of those rows, `projection` computes each row's revenue, and
  `segmented_sum` adds them in two levels: segments of `SEGMENT` rows, and
  then the segments' sums. On the static path, the filter's output is an
  index output and the gathers are plain launches; on the guarded path, the
  output is raw and the gathers are guarded launches, whose index input a
  trusted kernel checks when they run. With `graph`, the second step runs
  as a CUDA graph.
- `join_revenue`: the revenue of the lines whose order is placed before a
  date, as in Q3: the host filters the orders, pylibcudf's inner join, a
  trusted operator, gives the lines of those orders as a raw gather map, and
  the same gathers, as guarded launches, projection, and sum follow.

A session allocates through one allocator: the executor's own suballocator,
or the engine's, RMM through `CatalogResource`, never both, since each
carves the whole pool. The join needs RMM's. Each pipeline frees the
buffers it allocated.
"""

import struct
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

import numpy as np
import pylibcudf as plc

from eval.tpch.data import Lineitem, Orders, day
from eval.tpch.operators import BLOCK, INDEX, Operators
from opgate.binding import ElementType
from opgate.catalog import BufferId, Fill, Launch, Step, TenantId, Write
from opgate.executor import Executor, Module
from opgate.hosts.rapids import BufferView, CatalogResource

__all__ = [
    "THRESHOLD",
    "Allocator",
    "Columns",
    "Path",
    "Q6",
    "Q6_PARAMETERS",
    "Scratch",
    "Session",
    "free",
    "join_revenue",
    "load",
    "q6",
    "q6_launch",
    "revenue",
    "scratch_buffers",
    "select",
    "select_launch",
    "sum_modules",
    "sum_step",
    "unload",
    "write_segments",
]

DOUBLE = ElementType(8, False)
THRESHOLD = 24.0
# `segmented_sum` adds each segment with one warp, so a sum of many rows adds
# segments of this many rows first, and then the segments' sums
SEGMENT = 1024


class Allocator(Protocol):
    """A suballocator of the pool whose every region the catalog checks."""

    def allocate(self, nbytes: int) -> BufferId: ...

    def free(self, buffer: BufferId) -> None: ...


@dataclass(frozen=True)
class Session:
    """A tenant's pipelines on an executor, and the allocator of its buffers."""

    executor: Executor
    tenant: TenantId
    memory: Allocator


class Path(Enum):
    """How a consumer's index input is known to be in range."""

    STATIC = "static"  # the producer's index output, which the gate proves in range
    GUARDED = "guarded"  # a trusted kernel checks it when the consumer runs


@dataclass(frozen=True)
class Q6:
    """The parameters of TPC-H Q6: ship dates in [date_lo, date_hi), discounts in
    [disc_lo, disc_hi], quantities below qty_max."""

    date_lo: int
    date_hi: int
    disc_lo: float
    disc_hi: float
    qty_max: float


# the validation parameters of TPC-H's specification (clause 2.4.6.4), with the
# discount bounds DISCOUNT - 0.01 and DISCOUNT + 0.01 written as 0.05 and 0.07
Q6_PARAMETERS = Q6(day(date(1994, 1, 1)), day(date(1995, 1, 1)), 0.05, 0.07, 24.0)


@dataclass(frozen=True)
class Columns:
    """The lineitem columns of one tenant in the pool, and their rows."""

    shipdate: BufferId
    discount: BufferId
    quantity: BufferId
    price: BufferId
    orderkey: BufferId
    rows: int


def _bits(value: float) -> int:
    """A float64 scalar as the unsigned integer its bits spell."""
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


def _grid(threads: int) -> tuple[int, int, int]:
    """The blocks for `threads` threads, at least one, as the operators' own launchers compute them."""
    return (max(1, -(-threads // BLOCK[0])), 1, 1)


def _segments(rows: int) -> int:
    """The segments of `SEGMENT` rows that a sum over `rows` rows adds first, at least one."""
    return max(1, -(-rows // SEGMENT))


def _allocate(s: Session, *sizes: int) -> tuple[BufferId, ...]:
    with s.executor.tenant(s.tenant):
        return tuple(s.memory.allocate(max(1, n)) for n in sizes)


def free(s: Session, *buffers: BufferId) -> None:
    with s.executor.tenant(s.tenant):
        for buffer in buffers:
            s.memory.free(buffer)


def load(s: Session, lineitem: Lineitem) -> Columns:
    """Allocate the tenant's lineitem columns and write them."""
    with s.executor.tenant(s.tenant):
        arrays = (
            lineitem.shipdate,
            lineitem.discount,
            lineitem.quantity,
            lineitem.extendedprice,
            lineitem.orderkey,
        )
        buffers = _allocate(s, *(array.nbytes for array in arrays))
        writes = tuple(Write(b, a.tobytes(), None) for b, a in zip(buffers, arrays, strict=True))
        s.executor.execute(Step(s.tenant, writes))
        return Columns(*buffers, rows=len(lineitem.quantity))


def unload(s: Session, t: Columns) -> None:
    free(s, t.shipdate, t.discount, t.quantity, t.price, t.orderkey)


def _read_double(s: Session, buffer: BufferId) -> float:
    with s.executor.tenant(s.tenant):
        (value,) = struct.unpack("<d", s.executor.read(s.tenant, buffer))
        return value


def q6_launch(ops: Operators, t: Columns, result: BufferId, query: Q6) -> Launch:
    """The launch of Q6 that adds into `result`, which is zero on entry."""
    scalars = (
        query.date_lo % 2**32,
        query.date_hi % 2**32,
        _bits(query.disc_lo),
        _bits(query.disc_hi),
        _bits(query.qty_max),
    )
    buffers = (t.shipdate, t.discount, t.quantity, t.price, result)
    return Launch(ops.q6.operator, _grid(t.rows), buffers, (t.rows,), scalars, False)


def q6(s: Session, ops: Operators, t: Columns, query: Q6) -> float:
    with s.executor.tenant(s.tenant):
        (result,) = _allocate(s, 8)
        launch = q6_launch(ops, t, result, query)
        s.executor.execute(Step(s.tenant, (Fill(result, DOUBLE, 0), launch)))
        value = _read_double(s, result)
        free(s, result)
        return value


@dataclass(frozen=True)
class Scratch:
    """The buffers of a sum over `rows` rows."""

    rows: int
    price: BufferId  # the gathered price of each row
    discount: BufferId  # the gathered discount of each row
    revenue: BufferId  # the revenue of each row
    offsets: BufferId  # the offsets of the segments of `SEGMENT` rows
    partial: BufferId  # the sum of each segment
    whole: BufferId  # the offsets of the one segment of the partial sums
    total: BufferId  # the sum

    @property
    def buffers(self) -> tuple[BufferId, ...]:
        return (
            self.price,
            self.discount,
            self.revenue,
            self.offsets,
            self.partial,
            self.whole,
            self.total,
        )

    def segments(self) -> tuple[bytes, bytes]:
        """The data of the offsets and of the whole: segments of `SEGMENT` rows, and one segment."""
        segments = _segments(self.rows)
        bounds = np.minimum(np.arange(segments + 1) * SEGMENT, self.rows).astype("<i4")
        return bounds.tobytes(), struct.pack("<2i", 0, segments)


def sum_modules(ops: Operators) -> tuple[Module, ...]:
    """The module of each launch of `sum_step`, in order."""
    return (ops.gather, ops.gather, ops.projection, ops.segmented_sum, ops.segmented_sum)


def sum_step(
    s: Session, ops: Operators, t: Columns, sel: BufferId, scratch: Scratch, guarded: bool
) -> Step:
    """Gather the price and the discount of the rows that `sel` names, compute
    their revenue, and add it into the scratch's total: each segment of
    `SEGMENT` rows first, and then the segments' sums. The scratch's offsets
    hold the segments (`write_segments`)."""
    rows, segments = scratch.rows, _segments(scratch.rows)
    gather, grid = ops.gather.operator, _grid(rows)
    add = ops.segmented_sum.operator
    return Step(
        s.tenant,
        (
            Launch(gather, grid, (t.price, sel, scratch.price), (t.rows, rows), (), guarded),
            Launch(gather, grid, (t.discount, sel, scratch.discount), (t.rows, rows), (), guarded),
            Launch(
                ops.projection.operator,
                grid,
                (scratch.price, scratch.discount, scratch.revenue),
                (rows,),
                (),
                False,
            ),
            Launch(
                add,
                _grid(32 * segments),
                (scratch.offsets, scratch.revenue, scratch.partial),
                (segments, rows),
                (),
                False,
            ),
            Launch(
                add,
                _grid(32),
                (scratch.whole, scratch.partial, scratch.total),
                (1, segments),
                (),
                False,
            ),
        ),
    )


def write_segments(s: Session, scratch: Scratch) -> None:
    """Write the segments' offsets, before the sum step: a graph holds no upload."""
    with s.executor.tenant(s.tenant):
        offsets, whole = scratch.segments()
        s.executor.execute(
            Step(
                s.tenant,
                (Write(scratch.offsets, offsets, INDEX), Write(scratch.whole, whole, INDEX)),
            )
        )


def _sum(
    s: Session,
    ops: Operators,
    t: Columns,
    sel: BufferId,
    scratch: Scratch,
    guarded: bool,
    graph: bool,
) -> float:
    write_segments(s, scratch)
    _run(s, sum_step(s, ops, t, sel, scratch, guarded), graph)
    return _read_double(s, scratch.total)


def select_launch(
    ops: Operators, t: Columns, sel: BufferId, count: BufferId, path: Path, threshold: float
) -> Launch:
    """The launch of the filter of the lines with `quantity < threshold` into
    `sel`, which counts them into `count`, zero on entry."""
    module = ops.select_static if path is Path.STATIC else ops.select_raw
    scalars = (_bits(threshold),)
    return Launch(
        module.operator, _grid(t.rows), (t.quantity, sel, count), (t.rows,), scalars, False
    )


def select(
    s: Session, ops: Operators, t: Columns, path: Path, threshold: float
) -> tuple[BufferId, BufferId, int]:
    """Filter the lines with `quantity < threshold`: the selected row ids, their
    count's buffer, and their number, which the host reads."""
    with s.executor.tenant(s.tenant):
        sel, count = _allocate(s, 4 * t.rows, 4)
        # the output is filled first: a static index output must be checked in
        # range before its launch, and a guarded check reads every element
        launch = select_launch(ops, t, sel, count, path, threshold)
        s.executor.execute(Step(s.tenant, (Fill(sel, INDEX, 0), Fill(count, INDEX, 0), launch)))
        (rows,) = struct.unpack("<i", s.executor.read(s.tenant, count))
        return sel, count, rows


def scratch_buffers(s: Session, rows: int) -> Scratch:
    """The scratch buffers of a sum over `rows` rows."""
    segments = _segments(rows)
    sizes = (8 * rows, 8 * rows, 8 * rows, 4 * (segments + 1), 8 * segments, 8, 8)
    return Scratch(rows, *_allocate(s, *sizes))


def _run(s: Session, step: Step, graph: bool) -> None:
    """Perform `step` eagerly or as a graph. A guarded kernel whose index input is
    out of range does not run, and the pipeline has no result then."""
    with s.executor.tenant(s.tenant):
        if graph:
            captured = s.executor.capture(step)
            ran = s.executor.ran(captured)
            s.executor.release(captured)
        else:
            ran = s.executor.execute(step)
        if not all(ran):
            raise RuntimeError("a guarded launch did not run: its index input is out of range")


def revenue(
    s: Session, ops: Operators, t: Columns, path: Path, threshold: float, graph: bool
) -> float:
    """The revenue of the lines with `quantity < threshold`."""
    sel, count, rows = select(s, ops, t, path, threshold)
    scratch = scratch_buffers(s, rows)
    value = _sum(s, ops, t, sel, scratch, path is Path.GUARDED, graph)
    free(s, sel, count, *scratch.buffers)
    return value


def join_revenue(
    s: Session, ops: Operators, t: Columns, orders: Orders, before: int, rmm: CatalogResource
) -> float:
    """The revenue of the lines whose order is placed before day `before`. The
    join is pylibcudf's inner join, a trusted operator, which allocates its
    output in the pool through `rmm`, the session's allocator, as buffers of
    their own, since the gathers take its gather map with an index range."""
    with s.executor.tenant(s.tenant):
        if s.memory is not rmm:
            raise ValueError("the join's session allocates through the RMM wrapper it joins with")
        keys = np.ascontiguousarray(orders.orderkey[orders.orderdate < before])
        (placed,) = _allocate(s, keys.nbytes)
        s.executor.execute(Step(s.tenant, (Write(placed, keys.tobytes(), None),)))
        value = 0.0  # the sum of no line
        with s.executor.tenant(s.tenant):
            left = plc.Column.from_cuda_array_interface(
                BufferView.of(rmm, placed, "<i8", len(keys), access="read")
            )
            right = plc.Column.from_cuda_array_interface(
                BufferView.of(rmm, t.orderkey, "<i8", t.rows, access="read")
            )
            with rmm.dedicated():
                _, lines = plc.join.inner_join(
                    plc.Table([left]),
                    plc.Table([right]),
                    plc.types.NullEquality.EQUAL,
                    mr=rmm.resource,
                )
        rows = lines.size()
        if rows > 0:
            data = lines.data()
            if data is None:
                raise RuntimeError("the join's gather map of rows has no data")
            scratch = scratch_buffers(s, rows)
            value = _sum(s, ops, t, rmm.buffer_at(data.ptr), scratch, True, False)
            free(s, *scratch.buffers)
        with s.executor.tenant(s.tenant):
            del left, right, lines
        free(s, placed)
        return value
