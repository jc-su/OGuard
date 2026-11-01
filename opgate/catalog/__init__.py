"""The trusted state and the step check.

The catalog records the pool, each live buffer with its owner and status,
the grants, each freed region with its last owner, each tenant's
reservation, and each verified operator with its binding. A buffer's id is
its owner and a serial number of the owner's own, so the ids of two tenants
never meet. A tenant reserves one region of the pool as its session opens,
and each of its buffers lies in that region: where a tenant's buffers lie,
and whether its allocation fits, depend only on its own steps. A step is
one proposal from the planner or the
allocation wrapper: a tenant and a sequence of actions (allocate, free,
grant, write, fill, launch, reserve, unreserve). `check_step` checks the actions in order,
each against the state the previous ones leave, and returns the next
catalog with the effects the executor performs, or the reason for a
refusal; a refused step changes nothing (design section 6.1).
`check_launch`, the launch check, is its clause for one launch's arguments
against the operator's binding and the catalog. The Python `check_step`
implements the Rocq definition in `formal/CatalogStep.v`, and the conformance
test in `tests/formal/` requires the two to decide every generated step alike:
the same refusal, or the same next catalog (its buffers, grants, and freed
regions as sets) and effects.

A prepared step is a step of fills and one launch that a tenant repeats with
new scalars. `Catalog.prepare` checks it once; `Catalog.begin_prepared`
decides each use exactly as `begin` decides the step with the use's scalars.
A step's check reads only the records of the buffers it names (each buffer
and its grants), the operator's binding, which never changes, and the
scalars, which reach only the bounds check and the kernel's parameters. So
the catalog keeps, for each prepared step, the records its last full check
started from and the changes and effects it produced: when a use starts from
the same records, it checks the scalars' bounds and applies those changes
and effects with the new scalars (a `Use`); otherwise it checks the step in
full. A freed, granted, or rewritten buffer changes its record, so a use
after the change is checked in full and refused if the step no longer holds.
The catalog's epoch counts its changes: a use at the epoch at which the
step's records last equaled its memo's needs no comparison of records.

Rocq keeps the state in lists that a check scans. The executor's catalog is
one mutable `Catalog`, which keeps the live buffers by id and by offset and
the freed regions by offset, so that a step costs a few lookups however many
buffers are live: `apply` changes it in place and undoes every change if an
action is refused, and `check` applies and then undoes. The pure `check_step`
applies a step to a copy. The lookups rely on two invariants of the Rocq
state: live buffers are pairwise apart (`inv`), and a freed region lies apart
from every live buffer (`freed_apart`); freed regions are also pairwise
apart, since each comes from a buffer that was live, which the catalog checks
whenever it releases one. Code other than the executor reads the catalog
through `CatalogView`, which cannot change it.

Imports `opgate.signature` and `opgate.binding` only.
"""

import sys
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, NewType, Protocol, assert_never

from opgate.binding import (
    BASE_ALIGN,
    GRID_LIMIT,
    Base,
    Binding,
    Bounded,
    Constant,
    Direction,
    ElementType,
    Size,
    Span,
    Zero,
)
from opgate.signature import Affine

__all__ = [
    "ALIGNMENT",
    "GRID_LIMIT",
    "Action",
    "Allocate",
    "At",
    "Buffer",
    "BufferId",
    "Catalog",
    "CatalogView",
    "Check",
    "Checked",
    "Clear",
    "Decided",
    "Effect",
    "Fill",
    "Free",
    "Grant",
    "Kernel",
    "Launch",
    "Memset",
    "OperatorId",
    "Pending",
    "Prepared",
    "Raw",
    "Region",
    "Reservation",
    "Reserve",
    "Status",
    "Step",
    "StepAccepted",
    "StepRefused",
    "TenantId",
    "Traced",
    "Unreserve",
    "Upload",
    "Use",
    "Value",
    "Verified",
    "View",
    "Write",
    "check_launch",
    "check_step",
    "element_range",
]

TenantId = NewType("TenantId", int)
OperatorId = NewType("OperatorId", int)


class BufferId(NamedTuple):
    """A buffer's id: its owner, and a serial number that the owner's steps choose."""

    tenant: TenantId
    serial: int


ALIGNMENT = 2**BASE_ALIGN
"""Every buffer's offset in the pool is a multiple of 256 bytes; the pool's base is too."""

if sys.byteorder != "little":
    raise ImportError("the catalog reads uploaded elements in the host's byte order, the device's")

# the memoryview format of an element of each width and signedness
_ELEMENT_FORMATS: dict[tuple[int, bool], Literal["b", "B", "h", "H", "i", "I", "q", "Q"]] = {
    (1, True): "b",
    (1, False): "B",
    (2, True): "h",
    (2, False): "H",
    (4, True): "i",
    (4, False): "I",
    (8, True): "q",
    (8, False): "Q",
}


@dataclass(frozen=True)
class Raw:
    """The catalog knows nothing of the buffer's values."""


@dataclass(frozen=True)
class Checked:
    """Every element of the buffer, read as `element`, lies in `[lo, hi]`."""

    element: ElementType
    lo: int
    hi: int


Status = Raw | Checked


@dataclass(frozen=True)
class Buffer:
    id: BufferId
    offset: int  # bytes from the pool's base
    nbytes: int
    owner: TenantId
    status: Status


@dataclass(frozen=True)
class Region:
    """A freed region of the pool and its last owner."""

    offset: int
    nbytes: int
    owner: TenantId


@dataclass(frozen=True)
class Reservation:
    """The region of the pool that holds every buffer of `tenant`."""

    tenant: TenantId
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Grant:
    """The buffer is readable by `tenant`, which does not own it."""

    buffer: BufferId
    tenant: TenantId


@dataclass(frozen=True)
class Allocate:
    """A region the suballocator proposes, for a new buffer of the step's tenant."""

    buffer: BufferId
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Free:
    buffer: BufferId


@dataclass(frozen=True)
class Write:
    """The host writes `data` at the start of the buffer. With an element type,
    the data fills the buffer, and the buffer becomes checked with the range of
    its elements; without one, it becomes raw."""

    buffer: BufferId
    data: bytes
    element: ElementType | None


@dataclass(frozen=True)
class Fill:
    """The executor sets every element of the buffer to `value`."""

    buffer: BufferId
    element: ElementType
    value: int


@dataclass(frozen=True)
class View:
    """The bytes of a buffer from `offset`, a multiple of 256, as a launch's
    argument: an engine's column inside a tenant's arena. The argument of a
    span with an index range is a whole buffer, whose status the catalog
    records."""

    buffer: "BufferId"
    offset: int


@dataclass(frozen=True)
class Launch:
    """One launch of a verified operator, with a buffer, or a view of one, for each
    buffer argument of its binding, a value for each size symbol, and one for each
    bounded parameter.
    With `guarded`, the elements of an index input that the catalog has not
    checked are checked when the launch runs, and the kernel runs only if each
    lies in the range (design section 6.4)."""

    operator: OperatorId
    grid: tuple[int, int, int]
    buffers: tuple[BufferId | View, ...]
    sizes: tuple[int, ...]
    scalars: tuple[int, ...]
    guarded: bool


@dataclass(frozen=True)
class Reserve:
    """The step's tenant reserves `[offset, offset + nbytes)` as its session opens."""

    offset: int
    nbytes: int


@dataclass(frozen=True)
class Unreserve:
    """The step's tenant, holding no live buffer, ends its reservation and drops
    its freed regions, whose bytes are then cleared before any later use."""


Action = Allocate | Free | Grant | Write | Fill | Launch | Reserve | Unreserve


@dataclass(frozen=True)
class Step:
    tenant: TenantId
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class At:
    """A parameter that holds the address of the byte at `offset` in the pool."""

    offset: int


@dataclass(frozen=True)
class Value:
    """A parameter that holds `value`, modulo 2 to the parameter's width in bits."""

    value: int


@dataclass(frozen=True)
class Clear:
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Upload:
    offset: int
    data: bytes


@dataclass(frozen=True)
class Memset:
    """Set each element of `nbytes` bytes at `offset` to `value`."""

    offset: int
    nbytes: int
    element: ElementType
    value: int


@dataclass(frozen=True)
class Check:
    """Every whole element of the `nbytes` bytes at `offset`, read as `element`,
    lies in `[lo, hi]`."""

    offset: int
    nbytes: int
    element: ElementType
    lo: int
    hi: int


@dataclass(frozen=True)
class Kernel:
    """A checked launch: the operator, the grid, what each PTX parameter holds, and
    the checks that must hold when it runs; it runs only if every check holds."""

    operator: OperatorId
    grid: tuple[int, int, int]
    shared: int
    arguments: tuple[At | Value, ...]
    checks: tuple[Check, ...]


Effect = Clear | Upload | Memset | Kernel
"""What the executor does for an accepted step, in order."""


Undo = list[Callable[[], None]]
"""The inverse of each change a step made so far, in order."""


@dataclass(frozen=True)
class Prepared:
    """A step of fills and one launch, last, that `Catalog.prepare` checked; each
    use replaces the launch's scalars."""

    id: int
    tenant: TenantId
    actions: tuple[Action, ...]
    footprint: tuple[BufferId, ...]  # every buffer the actions name, once each
    parameters: tuple[int, ...]  # the index of each bounded parameter among the kernel's arguments


@dataclass(frozen=True)
class Decided:
    """A step that the catalog decided and its caller applied: the step, with a
    prepared use's scalars in its launch, and the effects of its full check, or
    its refusal."""

    step: "Step"
    decision: "tuple[Effect, ...] | StepRefused"


@dataclass(frozen=True)
class Verified:
    """An operator that the catalog records as verified, with its binding."""

    operator: "OperatorId"
    binding: Binding


Traced = Decided | Verified
"""What a traced catalog records, in order (`Catalog.trace`)."""


# the records of a prepared step's footprint: each buffer, or None, and its grantees
_Records = tuple[tuple["Buffer | None", tuple[TenantId, ...]], ...]


@dataclass(frozen=True)
class _Memo:
    """A prepared step's last full check: the records it started from, the status
    each buffer of the footprint had after it where that changed, and its effects."""

    records: _Records
    statuses: tuple[tuple[BufferId, "Status"], ...]
    effects: tuple["Effect", ...]


class CatalogView(Protocol):
    """What code other than the executor may do with the executor's catalog: read
    it, and check a step without applying it."""

    @property
    def pool(self) -> int: ...

    @property
    def buffers(self) -> tuple["Buffer", ...]: ...

    @property
    def grants(self) -> tuple["Grant", ...]: ...

    @property
    def freed(self) -> tuple["Region", ...]: ...

    @property
    def reserved(self) -> tuple["Reservation", ...]: ...

    @property
    def operators(self) -> tuple[tuple[OperatorId, Binding], ...]: ...

    def buffer(self, buffer: BufferId) -> "Buffer | None": ...

    def reservation(self, tenant: TenantId) -> "Reservation | None": ...

    def binding(self, operator: OperatorId) -> Binding | None: ...

    def granted(self, buffer: BufferId) -> bool: ...

    def readable(self, tenant: TenantId, buffer: BufferId) -> bool: ...

    def writable(self, tenant: TenantId, buffer: BufferId) -> bool: ...

    def check(self, step: "Step") -> "tuple[Effect, ...] | StepRefused": ...

    def copy(self) -> "Catalog": ...


def _region_offset(region: "Region") -> int:
    return region.offset


@dataclass(frozen=True)
class _Operator:
    """Immutable binding structure, prepared once without any launch authority.

    Fixed arguments can be shared because `Value` is immutable. Every buffer
    offset, size expression, and bounded scalar is filled from the newly checked
    launch; none of those values or their validation results is retained here.
    """

    binding: Binding
    uses: tuple[tuple[int, Span, bool], ...]  # each buffer parameter, its span, whether it writes
    sizes: tuple[tuple[int, Affine], ...]
    scalars: tuple[tuple[int, Bounded], ...]
    arguments: tuple[At | Value, ...]


def _prepare(binding: Binding) -> _Operator:
    uses: list[tuple[int, Span, bool]] = []
    sizes: list[tuple[int, Affine]] = []
    scalars: list[tuple[int, Bounded]] = []
    arguments: list[At | Value] = []
    for index, (_, parameter) in enumerate(binding.parameters):
        # Each dynamic position is replaced on every launch, after its checks.
        value = 0
        match parameter:
            case Base(span=span):
                uses.append(
                    (index, span, span.direction in (Direction.WRITE, Direction.READ_WRITE))
                )
            case Size(value=affine):
                if affine.terms:
                    sizes.append((index, affine))
                else:
                    value = affine.constant
            case Constant(value=constant):
                value = constant
            case Bounded():
                scalars.append((index, parameter))
            case Zero():
                pass
            case _ as never:
                assert_never(never)
        arguments.append(Value(value))
    return _Operator(binding, tuple(uses), tuple(sizes), tuple(scalars), tuple(arguments))


class Catalog:
    """The catalog's state: the pool's size, the live buffers, the grants, the
    freed regions with their last owners, and the verified operators."""

    __slots__ = (
        "_pool",
        "_buffers",
        "_starts",
        "_ids",
        "_grants",
        "_freed",
        "_reserved",
        "_operators",
        "_pending",
        "_memos",
        "_prepared",
        "_epoch",
        "_validated",
        "_trace",
    )

    _pool: int  # the bytes of the pool
    _buffers: dict[BufferId, Buffer]  # the live buffers by id
    _starts: list[int]  # the live buffers' offsets, in increasing order
    _ids: list[BufferId]  # the buffer at each of `_starts`
    _grants: dict[BufferId, tuple[TenantId, ...]]  # the grantees of each granted buffer
    _freed: list[Region]  # the freed regions, pairwise apart, by offset
    _reserved: dict[TenantId, Reservation]  # each tenant's reservation, in the order they were made
    _operators: dict[OperatorId, _Operator]
    _pending: bool  # a step is applied and neither committed nor rolled back
    _memos: dict[int, _Memo]  # each prepared step's last full check, by its id
    _prepared: int  # the number of steps prepared so far
    _epoch: int  # the number of changes to the state so far
    _validated: dict[int, int]  # the epoch at which a step's records last equaled its memo's
    _trace: list[Traced] | None  # what the catalog records while traced

    def __init__(self, pool: int, operators: tuple[tuple[OperatorId, Binding], ...] = ()) -> None:
        self._pool = pool
        self._pending = False
        self._buffers = {}
        self._starts = []
        self._ids = []
        self._grants = {}
        self._freed = []
        self._reserved = {}
        self._memos = {}
        self._prepared = 0
        self._epoch = 0
        self._validated = {}
        self._trace = None
        self._operators = {operator: _prepare(binding) for operator, binding in operators}
        if len(self._operators) != len(operators):
            raise ValueError("an operator id is registered twice")

    # -------------------------------------------------------------- reading

    @property
    def pool(self) -> int:
        return self._pool

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return tuple(self._buffers.values())

    @property
    def grants(self) -> tuple[Grant, ...]:
        return tuple(Grant(b, t) for b, tenants in self._grants.items() for t in tenants)

    @property
    def freed(self) -> tuple[Region, ...]:
        """The freed regions, by offset."""
        return tuple(self._freed)

    @property
    def reserved(self) -> tuple[Reservation, ...]:
        return tuple(self._reserved.values())

    @property
    def operators(self) -> tuple[tuple[OperatorId, Binding], ...]:
        return tuple((operator, prepared.binding) for operator, prepared in self._operators.items())

    def buffer(self, buffer: BufferId) -> Buffer | None:
        return self._buffers.get(buffer)

    def reservation(self, tenant: TenantId) -> Reservation | None:
        return self._reserved.get(tenant)

    def binding(self, operator: OperatorId) -> Binding | None:
        prepared = self._operators.get(operator)
        return None if prepared is None else prepared.binding

    def granted(self, buffer: BufferId) -> bool:
        """Whether a tenant other than the owner has a grant on the buffer."""
        return buffer in self._grants

    def readable(self, tenant: TenantId, buffer: BufferId) -> bool:
        """Whether `tenant` owns the buffer or has a grant on it."""
        found = self._buffers.get(buffer)
        return found is not None and (
            found.owner == tenant or tenant in self._grants.get(buffer, ())
        )

    def writable(self, tenant: TenantId, buffer: BufferId) -> bool:
        """Whether `tenant` owns the buffer and no tenant has a grant on it."""
        found = self._buffers.get(buffer)
        return found is not None and found.owner == tenant and buffer not in self._grants

    def _overlaps(self, offset: int, nbytes: int) -> bool:
        """Whether `[offset, offset + nbytes)` overlaps a live buffer: since live
        buffers are apart, only the last one starting at or before `offset` and
        the first one starting after it can."""
        i = bisect_right(self._starts, offset)
        if i > 0:
            before = self._buffers[self._ids[i - 1]]
            if before.offset + before.nbytes > offset:
                return True
        return i < len(self._starts) and self._starts[i] < offset + nbytes

    def _kept(self, tenant: TenantId, offset: int, nbytes: int) -> bool:
        """Whether one freed region of `tenant` holds all of `[offset, offset + nbytes)`:
        since freed regions are apart, only the last one starting at or before
        `offset` can."""
        k = bisect_right(self._freed, offset, key=_region_offset) - 1
        if k < 0:
            return False
        region = self._freed[k]
        return region.owner == tenant and offset + nbytes <= region.offset + region.nbytes

    # -------------------------------------------------------------- steps

    def copy(self) -> "Catalog":
        """A catalog with the same state, which changes apart from this one."""
        other = object.__new__(Catalog)
        other._pool = self._pool
        other._buffers = dict(self._buffers)
        other._starts = list(self._starts)
        other._ids = list(self._ids)
        other._grants = dict(self._grants)
        other._freed = list(self._freed)
        other._reserved = dict(self._reserved)
        other._operators = dict(self._operators)
        other._pending = False
        # the memos are a cache of checks, not state: a copy starts without them
        other._memos = {}
        other._prepared = self._prepared
        other._epoch = 0
        other._validated = {}
        other._trace = None
        return other

    def trace(self) -> list[Traced]:
        """Record, until `untrace`, each operator added and each step decided and
        applied: a refusal, and an accepted step when its caller commits it. A
        check (`check`, `prepare`) changes nothing and is not recorded."""
        if self._trace is not None:
            raise RuntimeError("the catalog is traced already")
        self._trace = []
        return self._trace

    def untrace(self) -> None:
        self._trace = None

    def _record(self, traced: Traced) -> None:
        if self._trace is not None:
            self._trace.append(traced)

    def add_operator(self, operator: OperatorId, binding: Binding) -> None:
        """Record a verified operator; the executor adds it after the gate accepts."""
        if operator in self._operators:
            raise ValueError(f"operator {operator} is registered")
        self._operators[operator] = _prepare(binding)
        self._record(Verified(operator, binding))

    def begin(self, step: "Step") -> "Pending | StepRefused":
        """Apply the step in place, until the caller commits or rolls it back; or
        refuse it and leave the catalog as it was."""
        if self._pending:
            raise RuntimeError("a step is applied and not committed or rolled back")
        undo: Undo = []
        result = _apply(self, step, undo)
        if isinstance(result, StepRefused):
            _roll_back(self, undo)
            self._record(Decided(step, result))
            return result
        self._pending = True
        return Pending(self, step, result, undo)

    def apply(self, step: "Step") -> "tuple[Effect, ...] | StepRefused":
        """Apply the step, or refuse it and leave the catalog as it was."""
        pending = self.begin(step)
        if isinstance(pending, StepRefused):
            return pending
        pending.commit()
        return pending.effects

    def check(self, step: "Step") -> "tuple[Effect, ...] | StepRefused":
        """The effects of the step, or its refusal, leaving the catalog as it was."""
        trace, self._trace = self._trace, None
        try:
            pending = self.begin(step)
        finally:
            self._trace = trace
        if isinstance(pending, StepRefused):
            return pending
        pending.roll_back()
        return pending.effects

    def prepare(self, step: "Step") -> "Prepared | StepRefused":
        """Check a step of fills and one launch, last, for repeated use, leaving the
        catalog as it was. A use submits no index check, so a launch that needs
        one cannot be prepared."""
        actions = step.actions
        launch = actions[-1] if actions else None
        if not isinstance(launch, Launch) or not all(isinstance(a, Fill) for a in actions[:-1]):
            return StepRefused("a prepared step is fills followed by one launch")
        effects = self.check(step)
        if isinstance(effects, StepRefused):
            return effects
        if _needs_checks(effects):
            return StepRefused(_PREPARED_CHECKS)
        buffers = [a.buffer for a in actions[:-1] if isinstance(a, Fill)]
        footprint = tuple(dict.fromkeys((*buffers, *map(_buffer_of, launch.buffers))))
        parameters = tuple(index for index, _ in self._operators[launch.operator].scalars)
        self._prepared += 1
        return Prepared(self._prepared, step.tenant, actions, footprint, parameters)

    def begin_prepared(self, prepared: Prepared, scalars: tuple[int, ...]) -> "Use | StepRefused":
        """Apply a use of `prepared` with the launch's `scalars` in place, as `begin`
        applies the step with those scalars, until the caller commits or rolls it
        back; or refuse it and leave the catalog as it was."""
        if self._pending:
            raise RuntimeError("a step is applied and not committed or rolled back")
        memo = self._memos.get(prepared.id)
        if memo is None or self._validated.get(prepared.id) != self._epoch:
            buffers, grants = self._buffers, self._grants
            records = tuple((buffers.get(b), grants.get(b, ())) for b in prepared.footprint)
            if memo is None or memo.records != records:
                return self._check_prepared(prepared, scalars, records)
            self._validated[prepared.id] = self._epoch
        launch = prepared.actions[-1]
        if not isinstance(launch, Launch):
            raise AssertionError("a prepared step ends with its launch")
        # the full check passed from these records: only the scalars' checks remain,
        # in the order `_check_launch` makes them
        where = f"action {len(prepared.actions) - 1}"
        bounded = self._operators[launch.operator].scalars
        refused = None
        if len(scalars) != len(bounded):
            refused = StepRefused(f"{where}: a scalar is missing or extra")
        elif not all(p.lower <= v <= p.upper for (_, p), v in zip(bounded, scalars, strict=True)):
            refused = StepRefused(f"{where}: a scalar lies outside its bounds")
        if refused is not None:
            self._record(Decided(_with_scalars(prepared, scalars), refused))
            return refused
        undo: Undo = []
        for buffer, status in memo.statuses:
            self._set_status(buffer, status, undo)
        self._pending = True
        return Use(self, _with_scalars(prepared, scalars), memo, prepared.parameters, scalars, undo)

    def hot(self, prepared: Prepared) -> object | None:
        """The memo of `prepared` if it decides a use at the current state without a
        change to the catalog: its records equal the current ones as of this epoch,
        and the step leaves them as it found them. A use that it decides then needs
        only the check of its scalars' bounds."""
        memo = self._memos.get(prepared.id)
        if memo is None or memo.statuses or self._validated.get(prepared.id) != self._epoch:
            return None
        return memo

    def unprepare(self, prepared: Prepared) -> None:
        """Forget a prepared step's last check."""
        self._memos.pop(prepared.id, None)

    def _check_prepared(
        self, prepared: Prepared, scalars: tuple[int, ...], records: _Records
    ) -> "Use | StepRefused":
        """A use of `prepared` checked in full; an accepted one becomes its memo."""
        launch = prepared.actions[-1]
        if not isinstance(launch, Launch):
            raise AssertionError("a prepared step ends with its launch")
        pending = self.begin(_with_scalars(prepared, scalars))
        if isinstance(pending, StepRefused):
            return pending
        if _needs_checks(pending.effects):
            pending.roll_back()
            return StepRefused(_PREPARED_CHECKS)
        after = self._buffers
        statuses = tuple(
            (b, after[b].status)
            for b, (before, _) in zip(prepared.footprint, records, strict=True)
            if before is not None and after[b].status != before.status
        )
        memo = _Memo(records, statuses, pending.effects)
        self._memos[prepared.id] = memo
        if not statuses:
            # the step left its records as it found them
            self._validated[prepared.id] = self._epoch
        step = _with_scalars(prepared, scalars)
        return Use(self, step, memo, prepared.parameters, scalars, pending._undo)

    # -------------------------------------------------------------- changes, each with its inverse

    def _allocate(self, buffer: Buffer, undo: Undo) -> None:
        """Add a live buffer, and carve its bytes out of the freed regions."""
        self._epoch += 1
        buffers, starts, ids, freed = self._buffers, self._starts, self._ids, self._freed
        buffers[buffer.id] = buffer
        position = bisect_right(starts, buffer.offset)
        starts.insert(position, buffer.offset)
        ids.insert(position, buffer.id)

        def unallocate() -> None:
            del buffers[buffer.id]
            del starts[position]
            del ids[position]

        undo.append(unallocate)
        offset, end = buffer.offset, buffer.offset + buffer.nbytes
        k = bisect_right(freed, offset, key=_region_offset) - 1
        first = k if k >= 0 and freed[k].offset + freed[k].nbytes > offset else k + 1
        last = first
        pieces: list[Region] = []
        while last < len(freed) and freed[last].offset < end:
            region = freed[last]
            if region.offset < offset:
                pieces.append(Region(region.offset, offset - region.offset, region.owner))
            if end < region.offset + region.nbytes:
                pieces.append(Region(end, region.offset + region.nbytes - end, region.owner))
            last += 1
        if last > first:
            _splice(freed, first, last, pieces, undo)

    def _free(self, buffer: Buffer, tenant: TenantId, undo: Undo) -> None:
        """Remove a live buffer and its grants, and release its bytes: they and the
        freed regions of `tenant` that touch them become one region of `tenant`."""
        self._epoch += 1
        buffers, starts, ids, freed, grants = (
            self._buffers,
            self._starts,
            self._ids,
            self._freed,
            self._grants,
        )
        position = bisect_left(starts, buffer.offset)
        if ids[position] != buffer.id:
            raise RuntimeError("the catalog's index of live buffers is inconsistent")
        del buffers[buffer.id]
        del starts[position]
        del ids[position]

        def unfree() -> None:
            buffers[buffer.id] = buffer
            starts.insert(position, buffer.offset)
            ids.insert(position, buffer.id)

        undo.append(unfree)
        if buffer.id in grants:
            grantees = grants.pop(buffer.id)
            undo.append(lambda: grants.__setitem__(buffer.id, grantees))
        offset, end = buffer.offset, buffer.offset + buffer.nbytes
        i = bisect_left(freed, offset, key=_region_offset)
        lo, hi, first, last = offset, end, i, i
        if (
            i > 0
            and freed[i - 1].owner == tenant
            and freed[i - 1].offset + freed[i - 1].nbytes == offset
        ):
            lo, first = freed[i - 1].offset, i - 1
        if i < len(freed) and freed[i].owner == tenant and freed[i].offset == end:
            hi, last = freed[i].offset + freed[i].nbytes, i + 1
        # the released region lies apart from its neighbors
        if (first > 0 and freed[first - 1].offset + freed[first - 1].nbytes > lo) or (
            last < len(freed) and freed[last].offset < hi
        ):
            raise RuntimeError("a freed region overlaps another")
        _splice(freed, first, last, [Region(lo, hi - lo, tenant)], undo)

    def _grant(self, buffer: BufferId, tenant: TenantId, undo: Undo) -> None:
        self._epoch += 1
        grants = self._grants
        before = grants.get(buffer)
        grants[buffer] = (*(before or ()), tenant)
        if before is None:
            undo.append(lambda: grants.__delitem__(buffer))
        else:
            undo.append(lambda: grants.__setitem__(buffer, before))

    def _reserve(self, reservation: Reservation, undo: Undo) -> None:
        self._epoch += 1
        reserved = self._reserved
        reserved[reservation.tenant] = reservation
        undo.append(lambda: reserved.__delitem__(reservation.tenant))

    def _unreserve(self, tenant: TenantId, undo: Undo) -> None:
        """End the tenant's reservation, and drop its freed regions."""
        self._epoch += 1
        reserved = self._reserved
        before = dict(reserved)
        del reserved[tenant]

        def restore() -> None:
            reserved.clear()
            reserved.update(before)

        undo.append(restore)
        freed = self._freed
        kept = [region for region in freed if region.owner != tenant]
        if len(kept) != len(freed):
            _splice(freed, 0, len(freed), kept, undo)

    def _set_status(self, buffer: BufferId, status: Status, undo: Undo) -> None:
        self._epoch += 1
        buffers = self._buffers
        found = buffers[buffer]
        buffers[buffer] = Buffer(found.id, found.offset, found.nbytes, found.owner, status)
        undo.append(lambda: buffers.__setitem__(buffer, found))


class Pending:
    """A step applied to a catalog, which the caller commits once the step's
    effects are done, or rolls back."""

    __slots__ = ("effects", "_catalog", "_step", "_undo")

    def __init__(
        self, catalog: Catalog, step: "Step", effects: "tuple[Effect, ...]", undo: Undo
    ) -> None:
        self.effects = effects
        self._catalog = catalog
        self._step = step
        self._undo = undo

    def commit(self) -> None:
        self._end()
        self._catalog._record(Decided(self._step, self._decided()))

    def _decided(self) -> "tuple[Effect, ...]":
        """The effects of the full check of the step."""
        return self.effects

    def roll_back(self) -> None:
        self._end()
        _roll_back(self._catalog, self._undo)

    def _end(self) -> None:
        if not self._catalog._pending:
            raise RuntimeError("the step is committed or rolled back already")
        self._catalog._pending = False


class Use(Pending):
    """A use of a prepared step applied to a catalog: its effects are the memo's,
    with the use's `scalars` at the kernel's bounded `parameters`."""

    __slots__ = ("memo", "parameters", "scalars")

    def __init__(
        self,
        catalog: Catalog,
        step: "Step",
        memo: _Memo,
        parameters: tuple[int, ...],
        scalars: tuple[int, ...],
        undo: Undo,
    ) -> None:
        super().__init__(catalog, step, memo.effects, undo)
        self.memo = memo  # the same object for every use the memo decides
        self.parameters = parameters
        self.scalars = scalars

    def checked(self) -> "tuple[Effect, ...]":
        """The use's effects, as the full check of the step with its scalars gives them."""
        *fills, kernel = self.memo.effects
        if not isinstance(kernel, Kernel):
            raise AssertionError("a prepared step's last effect is its kernel")
        arguments = list(kernel.arguments)
        for index, value in zip(self.parameters, self.scalars, strict=True):
            arguments[index] = Value(value)
        return (
            *fills,
            Kernel(kernel.operator, kernel.grid, kernel.shared, tuple(arguments), kernel.checks),
        )

    def _decided(self) -> "tuple[Effect, ...]":
        return self.checked()


def _splice(freed: list[Region], first: int, last: int, pieces: list[Region], undo: Undo) -> None:
    """Replace `freed[first:last]` with `pieces`, and record the inverse."""
    replaced = freed[first:last]
    freed[first:last] = pieces
    undo.append(lambda: freed.__setitem__(slice(first, first + len(pieces)), replaced))


def _roll_back(catalog: Catalog, undo: Undo) -> None:
    catalog._epoch += 1
    for inverse in reversed(undo):
        inverse()


_PREPARED_CHECKS = "a prepared launch cannot need index checks when it runs"


def _needs_checks(effects: "tuple[Effect, ...]") -> bool:
    return any(isinstance(effect, Kernel) and effect.checks for effect in effects)


@dataclass(frozen=True)
class StepAccepted:
    next: Catalog
    effects: tuple[Effect, ...]


@dataclass(frozen=True)
class StepRefused:
    reason: str


def element_range(element: ElementType) -> tuple[int, int]:
    """The values an element of this type holds."""
    bits = 8 * element.width
    if element.signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def evaluate(affine: Affine, sizes: tuple[int, ...]) -> int:
    return affine.constant + sum(c * sizes[k] for k, c in affine.terms)


def _write_status(buffer: Buffer, write: Write) -> Status | str:
    if write.element is None:
        return Raw() if len(write.data) <= buffer.nbytes else "the data is larger than the buffer"
    width = write.element.width
    if len(write.data) != buffer.nbytes or buffer.nbytes % width != 0:
        return "checked data must fill the buffer with whole elements"
    # the elements in the host's byte order, which is the device's (little-endian)
    values = memoryview(write.data).cast(_ELEMENT_FORMATS[width, write.element.signed])
    return Checked(write.element, min(values), max(values))


def check_launch(
    catalog: Catalog, tenant: TenantId, launch: Launch
) -> tuple[Catalog, Kernel] | str:
    """The launch check: the catalog after the launch and the checked launch, or why not."""
    after = catalog.copy()
    checked = _check_launch(after, tenant, launch, [])
    if isinstance(checked, str):
        return checked
    return after, checked


def _check_launch(catalog: Catalog, tenant: TenantId, launch: Launch, undo: Undo) -> Kernel | str:
    """The launch check, which changes the catalog only if it accepts the launch."""
    prepared = catalog._operators.get(launch.operator)
    if prepared is None:
        return "the operator is not verified"
    binding = prepared.binding
    if not all(1 <= g <= limit for g, limit in zip(launch.grid, GRID_LIMIT, strict=True)):
        return "the grid is outside the limits the gate assumes"
    if any(g > most for g, most in zip(launch.grid, binding.grid, strict=True)):
        return "the grid is larger than the binding's maxima"
    if len(launch.sizes) != len(binding.sizes):
        return "a size is missing or extra"
    if not all(
        s.minimum <= v <= s.maximum for s, v in zip(binding.sizes, launch.sizes, strict=True)
    ):
        return "a size lies outside its declared bounds"
    for relation in binding.derived_sizes:
        if (
            launch.sizes[relation.target]
            != evaluate(relation.numerator, launch.sizes) // relation.divisor
        ):
            return "a derived size does not equal its declared quotient"
    shared = evaluate(binding.shared, launch.sizes)
    if not 0 <= shared <= 49152:
        return "dynamic shared memory lies outside [0, 48 KiB]"
    uses = prepared.uses
    if len(launch.buffers) != len(uses):
        return "a buffer argument is missing or extra"
    bounded = prepared.scalars
    if len(launch.scalars) != len(bounded):
        return "a scalar is missing or extra"
    if not all(p.lower <= v <= p.upper for (_, p), v in zip(bounded, launch.scalars, strict=True)):
        return "a scalar lies outside its bounds"
    checks: list[Check] = []
    statuses: list[tuple[BufferId, Status]] = []
    offsets: list[int] = []
    ids = [_buffer_of(argument) for argument in launch.buffers]
    # the buffers that some argument passes with an index range or as a flag buffer
    indexed = {
        id_
        for id_, (_, span, _) in zip(ids, uses, strict=True)
        if span.index is not None or span.flag
    }
    for argument, id_, (_, span, writes) in zip(launch.buffers, ids, uses, strict=True):
        view = argument.offset if isinstance(argument, View) else 0
        buffer = catalog.buffer(id_)
        # one reason for a buffer the tenant may not read, live or not, so that the
        # refusal shows nothing of another tenant's buffers
        if buffer is None or not catalog.readable(tenant, id_):
            return "the tenant may not read a buffer argument"
        if view < 0 or view % ALIGNMENT != 0 or (span.index is not None and view != 0):
            return "a view is not at a multiple of 256 bytes, or an index span's view does not start its buffer"
        if buffer.nbytes < view + max(1, evaluate(span.extent, launch.sizes)):
            return "a buffer is smaller than its extent"
        if writes and not (
            catalog.writable(tenant, id_) and (ids.count(id_) == 1 or id_ not in indexed)
        ):
            return (
                "the tenant may not write a buffer argument, "
                "or passes a written buffer twice with an index range or a flag"
            )
        if span.index is not None:
            element = span.index.element
            lo = evaluate(span.index.lo, launch.sizes)
            hi = evaluate(span.index.hi, launch.sizes)
            if not _checked_inside(buffer.status, element, lo, hi):
                if not launch.guarded or writes:
                    return "an index buffer is not checked inside its declared range"
                checks.append(Check(buffer.offset, buffer.nbytes, element, lo, hi))
            if writes:
                statuses.append((id_, Checked(element, lo, hi)))
        elif writes:
            statuses.append((id_, Raw()))
        offsets.append(buffer.offset + view)
    # the arguments hold the addresses the catalog had before the launch changed any status
    arguments = _arguments(prepared, launch, offsets)
    for id_, status in statuses:
        catalog._set_status(id_, status, undo)
    return Kernel(launch.operator, launch.grid, shared, arguments, tuple(checks))


def _with_scalars(prepared: Prepared, scalars: tuple[int, ...]) -> Step:
    """The step of a use of `prepared`: its launch with the use's scalars."""
    launch = prepared.actions[-1]
    if not isinstance(launch, Launch):
        raise AssertionError("a prepared step ends with its launch")
    return Step(prepared.tenant, (*prepared.actions[:-1], replace(launch, scalars=scalars)))


def _buffer_of(argument: "BufferId | View") -> BufferId:
    return argument.buffer if isinstance(argument, View) else argument


def _checked_inside(status: Status, element: ElementType, lo: int, hi: int) -> bool:
    """Whether the catalog checked the buffer's elements, read as `element`, inside `[lo, hi]`."""
    match status:
        case Checked(element=checked, lo=blo, hi=bhi):
            return checked == element and lo <= blo and bhi <= hi
        case Raw():
            return False
        case _ as never:
            assert_never(never)


def _arguments(prepared: _Operator, launch: Launch, offsets: list[int]) -> tuple[At | Value, ...]:
    """What each PTX parameter holds: the buffer arguments, at `offsets`, and the scalars in order."""
    arguments = list(prepared.arguments)
    for (index, _, _), offset in zip(prepared.uses, offsets, strict=True):
        arguments[index] = At(offset)
    for index, affine in prepared.sizes:
        arguments[index] = Value(evaluate(affine, launch.sizes))
    for (index, _), value in zip(prepared.scalars, launch.scalars, strict=True):
        arguments[index] = Value(value)
    return tuple(arguments)


def _check_action(
    catalog: Catalog, tenant: TenantId, action: Action, undo: Undo
) -> tuple[Effect, ...] | str:
    """Check one action against the catalog, and apply it if it is accepted."""
    match action:
        case Allocate(buffer=id_, offset=offset, nbytes=nbytes):
            if id_.tenant != tenant:
                return "the buffer id is not the tenant's"
            if catalog.buffer(id_) is not None:
                return "the buffer id is in use"
            if nbytes < 1 or offset < 0 or offset + nbytes > catalog.pool:
                return "the region is empty or outside the pool"
            if offset % ALIGNMENT != 0:
                return "the region is not aligned to 256 bytes"
            reservation = catalog.reservation(tenant)
            if reservation is None or not (
                reservation.offset <= offset
                and offset + nbytes <= reservation.offset + reservation.nbytes
            ):
                return "the region lies outside the tenant's reservation"
            if catalog._overlaps(offset, nbytes):
                return "the region overlaps a live buffer"
            # the region keeps its contents only if the tenant last owned all of it
            kept = catalog._kept(tenant, offset, nbytes)
            catalog._allocate(Buffer(id_, offset, nbytes, tenant, Raw()), undo)
            return () if kept else (Clear(offset, nbytes),)
        case Free(buffer=id_):
            buffer = catalog.buffer(id_)
            if buffer is None or buffer.owner != tenant:
                return "the tenant does not own the buffer"
            catalog._free(buffer, tenant, undo)
            return ()
        case Grant(buffer=id_, tenant=grantee):
            buffer = catalog.buffer(id_)
            if buffer is None or buffer.owner != tenant or grantee == tenant:
                return "the tenant does not own the buffer, or grants it to itself"
            if grantee in catalog._grants.get(id_, ()):
                return "the grant exists"
            catalog._grant(id_, grantee, undo)
            return ()
        case Write(buffer=id_, data=data):
            buffer = catalog.buffer(id_)
            if buffer is None or not catalog.writable(tenant, id_):
                return "the tenant may not write the buffer"
            status = _write_status(buffer, action)
            if isinstance(status, str):
                return status
            catalog._set_status(id_, status, undo)
            return (Upload(buffer.offset, data),)
        case Fill(buffer=id_, element=element, value=value):
            buffer = catalog.buffer(id_)
            if buffer is None or not catalog.writable(tenant, id_):
                return "the tenant may not write the buffer"
            lo, hi = element_range(element)
            if not lo <= value <= hi or buffer.nbytes % element.width != 0:
                return "the value is not an element, or the buffer is not whole elements"
            catalog._set_status(id_, Checked(element, value, value), undo)
            return (Memset(buffer.offset, buffer.nbytes, element, value),)
        case Launch():
            checked = _check_launch(catalog, tenant, action, undo)
            if isinstance(checked, str):
                return checked
            return (checked,)
        case Reserve(offset=offset, nbytes=nbytes):
            if catalog.reservation(tenant) is not None:
                return "the tenant holds a reservation"
            if nbytes < 1 or offset < 0 or offset + nbytes > catalog.pool:
                return "the reservation is empty or outside the pool"
            if any(
                r.offset < offset + nbytes and offset < r.offset + r.nbytes
                for r in catalog._reserved.values()
            ):
                return "the reservation overlaps another"
            catalog._reserve(Reservation(tenant, offset, nbytes), undo)
            return ()
        case Unreserve():
            if catalog.reservation(tenant) is None:
                return "the tenant holds no reservation"
            if any(b.owner == tenant for b in catalog._buffers.values()):
                return "the tenant holds a live buffer"
            catalog._unreserve(tenant, undo)
            return ()
        case _ as never:
            assert_never(never)


def _apply(catalog: Catalog, step: Step, undo: Undo) -> tuple[Effect, ...] | StepRefused:
    """Apply the step's actions in order, each against the state the previous ones
    leave, recording the inverse of every change in `undo`."""
    effects: list[Effect] = []
    for position, action in enumerate(step.actions):
        checked = _check_action(catalog, step.tenant, action, undo)
        if isinstance(checked, str):
            return StepRefused(f"action {position}: {checked}")
        effects += checked
    return tuple(effects)


def check_step(catalog: CatalogView, step: Step) -> StepAccepted | StepRefused:
    """The next catalog and the effects of `step`, or the reason the first refused
    action fails; `catalog` stays as it is."""
    after = catalog.copy()
    result = after.apply(step)
    if isinstance(result, StepRefused):
        return result
    return StepAccepted(after, result)
