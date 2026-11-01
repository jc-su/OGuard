"""The mix of the mixed experiment (`eval.rq4_cost.mixed`), apart from the service and the GPU.

The templates and their kernels (`MIXES`), what serves each step in each
configuration (`served_by`), the recorded decisions (`recorded`), the
inputs (`save_inputs`), the requests of a kernel's prepared step and of a
built-in operator (`request_of`, `builtin_request`), the parameters and
fills of a direct launch under MPS (`checked_launch`, `marshal`,
`memset_of`), and a tenant's verification and closed loop over its steps'
uses (`verify`, `closed_loop`). This module imports neither the service nor
the driver's functions, so its tests run without a GPU.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import numpy as np

from eval.corpus.adapter import (
    ELEMENTS,
    Assigned,
    Pointer,
    Points,
    Role,
    directory_of,
    read_raw,
    read_roles,
)
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import Workload, _grid, check, encode, workload
from eval.rq4_cost.fallback import BUILTINS, Builtin
from opgate import catalog
from opgate.binding import Binding, ElementType
from opgate.catalog import (
    ALIGNMENT,
    Allocate,
    At,
    BufferId,
    Launch,
    OperatorId,
    Reserve,
    StepRefused,
    TenantId,
    Value,
    Write,
)
from opgate.executor import READ_BYTES

Model = Literal["opus", "sonnet"]
Task = Literal[
    "q6_sum",
    "starts_with",
    "masked_sum",
    "string_contains",
    "sorted_contains",
    "join_count",
    "groupby_sum",
    "string_equal",
    "bloom_probe",
]
TemplateName = Literal[
    "q6",
    "prefix_sum",
    "like_sum",
    "semi_join_sum",
    "join_count",
    "group_by",
    "string_equal",
    "bloom_probe",
]
Configuration = Literal["service", "service_refusals", "service_fallback", "mps", "mps_shared"]
Host = Literal["service", "mps"]
Status = Literal["as_written", "with_checks", "refused"]
Placement = Literal["shared", "output"]
Block = tuple[int, int, int]

CONFIGURATIONS: tuple[Configuration, ...] = (
    "service",
    "service_refusals",
    "service_fallback",
    "mps",
    "mps_shared",
)
SEED = 41  # each task's workload generator
# the reference check's tolerance, `eval.corpus.workloads.check`
TOLERANCE = 1e-9
# rows per task: each output that a step reads back fits in `READ_BYTES`
ROWS: dict[Task, int] = {
    "q6_sum": 1_000_000,
    "starts_with": 65_536,
    "masked_sum": 65_536,
    "string_contains": 65_536,
    "sorted_contains": 65_536,
    "join_count": 16_384,
    "groupby_sum": 32_768,
    "string_equal": 65_536,
    "bloom_probe": 65_536,
}
TASKS: tuple[Task, ...] = tuple(ROWS)
# the argument of `masked_sum` that is the first operator's output, and that output
LINK = ("mask", "match")
# the built-in operators by their op
BY_OP: dict[str, Builtin] = {builtin.op: builtin for builtin in BUILTINS.values()}


class Comparison(Enum):
    """How a use's read-back is compared with the verified use's read-back."""

    BYTES = "bytes"  # equal bytes
    CLOSE = "close"  # one float64, within `TOLERANCE`
    SORTED = "sorted"  # float64 slots whose order atomics decide, sorted, within `TOLERANCE`


# the output that each task's prepared step reads back, and how a use's read-back is compared
READS: dict[Task, tuple[str, Comparison]] = {
    "q6_sum": ("result", Comparison.CLOSE),
    "starts_with": ("match", Comparison.BYTES),
    "masked_sum": ("result", Comparison.CLOSE),
    "string_contains": ("match", Comparison.BYTES),
    "sorted_contains": ("match", Comparison.BYTES),
    "join_count": ("counts", Comparison.BYTES),
    "groupby_sum": ("table_sums", Comparison.SORTED),
    "string_equal": ("match", Comparison.BYTES),
    "bloom_probe": ("match", Comparison.BYTES),
}


@dataclass(frozen=True)
class Kernel:
    """A recorded Numba kernel of the synthesized corpus."""

    model: Model
    task: Task

    @property
    def kind(self) -> str:
        """The plan kind of its schema, which is also its corpus identifier."""
        return f"{self.model}/{self.task}/numba"

    @property
    def path(self) -> Path:
        return directory_of(self.model, self.task) / TOOLCHAINS["numba"]


MASKED_SUM = Kernel("opus", "masked_sum")


@dataclass(frozen=True)
class Template:
    """A query: `first`, and when `masked`, `masked_sum` over its `match` output."""

    name: TemplateName
    first: Kernel
    masked: bool

    @property
    def kernels(self) -> tuple[Kernel, ...]:
        return (self.first, MASKED_SUM) if self.masked else (self.first,)


def _templates(string_contains: Kernel, groupby_sum: Kernel) -> tuple[Template, ...]:
    return (
        Template("q6", Kernel("opus", "q6_sum"), False),
        Template("prefix_sum", Kernel("opus", "starts_with"), True),
        Template("like_sum", string_contains, True),
        Template("semi_join_sum", Kernel("opus", "sorted_contains"), True),
        Template("join_count", Kernel("opus", "join_count"), False),
        Template("group_by", groupby_sum, False),
        Template("string_equal", Kernel("opus", "string_equal"), False),
        Template("bloom_probe", Kernel("opus", "bloom_probe"), False),
    )


AS_WRITTEN = _templates(Kernel("opus", "string_contains"), Kernel("opus", "groupby_sum"))
MIXES: dict[Configuration, tuple[Template, ...]] = {
    "service": _templates(Kernel("sonnet", "string_contains"), Kernel("sonnet", "groupby_sum")),
    "service_refusals": AS_WRITTEN,
    "service_fallback": AS_WRITTEN,
    "mps": AS_WRITTEN,
    "mps_shared": AS_WRITTEN,
}


def host_of(configuration: Configuration) -> Host:
    """Whether the configuration's tenants run through one service or as processes under MPS."""
    match configuration:
        case "service" | "service_refusals" | "service_fallback":
            return "service"
        case "mps" | "mps_shared":
            return "mps"
        case _ as never:
            assert_never(never)


def serves_refused(configuration: Configuration) -> bool:
    """Whether a built-in operator takes the place of a kernel that the gate
    refuses, or the templates that use the kernel are skipped."""
    match configuration:
        case "service" | "service_refusals":
            return False
        case "service_fallback" | "mps" | "mps_shared":
            return True
        case _ as never:
            assert_never(never)


def served_by(configuration: Configuration, kernel: Kernel, accepted: bool) -> str | None:
    """What serves the kernel's step in the configuration: the kernel, named by
    its kind, if the gate accepts it; if the gate refuses it, the built-in
    operator of its task, named by its op, or nothing, and its templates are
    skipped."""
    if accepted:
        return kernel.kind
    if not serves_refused(configuration):
        return None
    builtin = BUILTINS.get(kernel.task)
    if builtin is None:
        raise RuntimeError(f"{kernel.kind}: the gate refuses it and its task has no built-in")
    return builtin.op


def runnable_templates(
    configuration: Configuration, accepted: Callable[[Kernel], bool]
) -> tuple[list[Template], list[str]]:
    """The templates whose every step something serves, and the names of the others."""
    mix = MIXES[configuration]
    runnable = [
        t
        for t in mix
        if all(served_by(configuration, k, accepted(k)) is not None for k in t.kernels)
    ]
    return runnable, [t.name for t in mix if t not in runnable]


def kernels_of(configuration: Configuration) -> tuple[Kernel, ...]:
    """The kernels of a configuration's templates, each once, in template order."""
    found: list[Kernel] = []
    for template in MIXES[configuration]:
        found.extend(kernel for kernel in template.kernels if kernel not in found)
    return tuple(found)


def every_kernel() -> tuple[Kernel, ...]:
    """The kernels of every configuration, each once."""
    found: list[Kernel] = []
    for configuration in CONFIGURATIONS:
        found.extend(kernel for kernel in kernels_of(configuration) if kernel not in found)
    return tuple(found)


# ------------------------------------------------------------------ decisions


class RecordedDecision(TypedDict):
    """A kernel's entries in the registration and guarded records, and the status they give."""

    status: Status
    registration: dict[str, object]
    guarded: dict[str, object]


def _entries(path: Path) -> dict[str, dict[str, object]]:
    loaded: object = json.loads(path.read_text())
    if not isinstance(loaded, dict) or loaded.get("complete") is not True:
        raise ValueError(f"{path}: not a complete record")
    results = loaded.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path}: the record has no results")
    entries: dict[str, dict[str, object]] = {}
    for entry in results:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError(f"{path}: a result has no identifier")
        entries[entry["id"]] = entry
    return entries


def recorded(registration: Path, guarded: Path) -> dict[str, RecordedDecision]:
    """Each kernel's status as the records give it: accepted as written by the
    registration record, accepted with the guard pass's checks or limits by the
    guarded record, or refused by both."""
    registered, checked = _entries(registration), _entries(guarded)
    decisions: dict[str, RecordedDecision] = {}
    for kernel in every_kernel():
        first, second = registered[kernel.kind], checked[kernel.kind]
        status: Status
        match first.get("accepted"), second.get("decision"):
            case True, "accepted":
                status = "as_written"
            case False, "accepted":
                status = "with_checks"
            case False, "refused":
                status = "refused"
            case accepted, decision:
                raise ValueError(
                    f"{kernel.kind}: the records give accepted={accepted!r} and decision={decision!r}"
                )
        decisions[kernel.kind] = {"status": status, "registration": first, "guarded": second}
    return decisions


# ------------------------------------------------------------------ inputs


def save_inputs(directory: Path, *, create: bool) -> None:
    """Write each task's workload, its arrays' bytes and its sizes and scalars;
    without `create`, require the bytes already there to equal them."""
    if create:
        directory.mkdir(parents=True, exist_ok=False)
    for task in TASKS:
        w = workload(task, ROWS[task], np.random.default_rng(SEED))
        if any(array.ndim != 1 for array in w.arrays.values()):
            raise RuntimeError(f"{task}: a workload array is not one-dimensional")
        description = {
            "rows": ROWS[task],
            "seed": SEED,
            "scalars": w.scalars,
            "sizes": w.sizes,
            "arrays": {name: array.dtype.str for name, array in w.arrays.items()},
        }
        files = {f"{name}.bin": array.tobytes() for name, array in w.arrays.items()}
        files["workload.json"] = (json.dumps(description, indent=1) + "\n").encode()
        folder = directory / task
        if create:
            folder.mkdir()
        for name, data in files.items():
            path = folder / name
            if create:
                path.write_bytes(data)
            elif path.read_bytes() != data:
                raise RuntimeError(f"the recorded input changed: {path}")


def read_workload(directory: Path, task: Task) -> Workload:
    """A task's workload as `save_inputs` wrote it."""
    folder = directory / task
    description: object = json.loads((folder / "workload.json").read_text())
    if not isinstance(description, dict):
        raise RuntimeError(f"{folder}: malformed workload description")
    arrays, scalars, sizes = (
        description.get("arrays"),
        description.get("scalars"),
        description.get("sizes"),
    )
    if not (isinstance(arrays, dict) and isinstance(scalars, dict) and isinstance(sizes, dict)):
        raise RuntimeError(f"{folder}: malformed workload description")
    read: dict[str, np.ndarray] = {}
    for name, dtype in arrays.items():
        if not isinstance(dtype, str):
            raise RuntimeError(f"{folder}: {name} has no element type")
        read[name] = np.frombuffer((folder / f"{name}.bin").read_bytes(), dtype=np.dtype(dtype))
    values: dict[str, int | float] = {}
    for name, value in scalars.items():
        if type(value) not in (int, float):
            raise RuntimeError(f"{folder}: the scalar {name} is not a number")
        values[name] = value
    counts: dict[str, int] = {}
    for name, value in sizes.items():
        if type(value) is not int:
            raise RuntimeError(f"{folder}: the size {name} is not an integer")
        counts[name] = value
    return Workload(read, values, counts)


# ------------------------------------------------------------------ requests


@dataclass(frozen=True)
class Argument:
    """A buffer argument of a kernel, in its signature's order."""

    name: str
    dtype: str
    role: Assigned


def buffers_of(kernel: Kernel) -> tuple[Argument, ...]:
    """The buffer arguments of a kernel, in the order that a launch names their buffers."""
    raws = read_raw(directory_of(kernel.model, kernel.task) / "signature.json")
    roles = read_roles(kernel.model, kernel.task)
    return tuple(
        Argument(raw.name, raw.dtype, roles.arguments[raw.name])
        for raw in raws
        if roles.arguments[raw.name] not in (Role.SIZE, Role.SCALAR)
    )


def placement(argument: Argument) -> Placement:
    """Whether the loader shares the argument's buffer, or each tenant allocates it."""
    match argument.role:
        case Role.READ:
            return "shared"
        case Role.WRITE | Role.READ_WRITE:
            return "output"
        case Role.SIZE | Role.SCALAR:
            raise ValueError(f"{argument.name} is not a buffer")
        case Points(pointer=Pointer.INDEX_OUT):
            raise ValueError(f"{argument.name}: an index output is outside the mix")
        case Points():
            return "shared"
        case _ as never:
            assert_never(never)


def linked(kernel: Kernel, argument: Argument) -> bool:
    """Whether the argument's buffer is the first operator's output: `masked_sum`
    runs second in every template that uses it, over that operator's `match`."""
    return kernel == MASKED_SUM and argument.name == LINK[0]


def shared_arguments(kernel: Kernel) -> tuple[Argument, ...]:
    """The arguments whose buffers the loader writes and grants to every tenant."""
    return tuple(
        argument
        for argument in buffers_of(kernel)
        if placement(argument) == "shared" and not linked(kernel, argument)
    )


def shared_inputs(configuration: Configuration) -> tuple[tuple[Task, Argument], ...]:
    """The shared inputs of the configuration's kernels, each task's argument
    once, whether the kernel or a built-in operator reads them."""
    found: list[tuple[Task, Argument]] = []
    for kernel in kernels_of(configuration):
        for argument in shared_arguments(kernel):
            if all((kernel.task, argument.name) != (t, a.name) for t, a in found):
                found.append((kernel.task, argument))
    return tuple(found)


def label_of(task: Task, name: str) -> str:
    """The name of a shared input in the record of an MPS tenant and its owner."""
    return f"{task}/{name}"


class Element(TypedDict):
    width: int
    signed: bool


class FillRequest(TypedDict):
    buffer: str
    element: Element
    value: int


class PrepareRequest(TypedDict):
    """The fields of a `prepare` request."""

    operator: str
    grid: list[int]
    buffers: list[str]
    sizes: list[int]
    scalars: list[int]
    fills: list[FillRequest]
    read: str


class BuiltinRequest(TypedDict):
    """The fields of a request to a built-in operator's query node (`fallback.Node`)."""

    op: str
    buffers: dict[str, str]
    sizes: dict[str, int]


def element_of(dtype: str) -> Element:
    element = ELEMENTS[dtype]
    return {"width": element.width, "signed": element.signed}


def fill_of(argument: Argument, initial: np.ndarray, buffer: str) -> FillRequest:
    """The fill that sets an output to its workload's initial contents before each
    use: the one value of all its elements."""
    if initial.dtype != np.dtype(argument.dtype):
        raise ValueError(f"{argument.name}: the workload's {initial.dtype} is not {argument.dtype}")
    element = element_of(argument.dtype)
    bits = initial.view(f"u{initial.itemsize}")
    if not len(bits) or not bool((bits == bits[0]).all()):
        raise ValueError(f"{argument.name}: a fill needs one initial value for every element")
    signed = initial.view(f"i{initial.itemsize}")
    value = int(signed[0]) if element["signed"] else int(bits[0])
    return {"buffer": buffer, "element": element, "value": value}


def request_of(
    kernel: Kernel,
    limit: Block,
    w: Workload,
    operator: str,
    handles: Mapping[str, str],
) -> PrepareRequest:
    """The prepared step of `kernel` on `w`: fills of its outputs, its launch over
    the buffers `handles` names for its arguments, with the corpus's grid within
    `limit`, the schema's grid, and the read-back of the output that `READS`
    names, which fits in `READ_BYTES`."""
    directory = directory_of(kernel.model, kernel.task)
    raws = read_raw(directory / "signature.json")
    roles = read_roles(kernel.model, kernel.task)
    buffers = buffers_of(kernel)
    for argument in buffers:
        if w.arrays[argument.name].dtype != np.dtype(argument.dtype):
            raise ValueError(f"{kernel.kind}: {argument.name} is not {argument.dtype}")
    sizes = tuple(w.sizes[name] for name in roles.symbols)
    scalars = tuple(
        encode(raw.dtype, w.scalars[raw.name])
        for raw in raws
        if roles.arguments[raw.name] is Role.SCALAR
    )
    read, _ = READS[kernel.task]
    if w.arrays[read].nbytes > READ_BYTES:
        raise ValueError(f"{kernel.kind}: {read} has more than {READ_BYTES} bytes")
    grid = [min(g, m) for g, m in zip(_grid("numba", sizes), limit, strict=True)]
    return {
        "operator": operator,
        "grid": grid,
        "buffers": [handles[argument.name] for argument in buffers],
        "sizes": list(sizes),
        "scalars": list(scalars),
        "fills": [
            fill_of(argument, w.arrays[argument.name], handles[argument.name])
            for argument in buffers
            if placement(argument) == "output"
        ],
        "read": handles[read],
    }


def builtin_request(builtin: Builtin, w: Workload, handles: Mapping[str, str]) -> BuiltinRequest:
    """The request of `builtin` on `w` over the buffers `handles` names for its
    operands: each operand holds the operand's elements of the workload."""
    sizes = {symbol: w.sizes[symbol] for symbol in builtin.sizes}
    counts = builtin.counts(sizes)
    for operand in builtin.operands:
        array = w.arrays[operand.name]
        if array.dtype != np.dtype(operand.typestr) or len(array) != counts[operand.name]:
            raise ValueError(f"{builtin.op}: {operand.name} is not {counts[operand.name]} elements")
    return {
        "op": builtin.op,
        "buffers": {operand.name: handles[operand.name] for operand in builtin.operands},
        "sizes": sizes,
    }


# ------------------------------------------------------------------ direct launches


@dataclass(frozen=True)
class Layout:
    """The buffers of a launch one after another, one `ALIGNMENT` unit apart: each
    handle's offset and bytes. An address inside a buffer, or one past its end,
    belongs to that buffer alone."""

    offsets: dict[str, int]
    nbytes: dict[str, int]

    @staticmethod
    def of(handles: Sequence[str], nbytes: Mapping[str, int]) -> Layout:
        offsets: dict[str, int] = {}
        end = 0
        for handle in handles:
            if handle in offsets:
                raise ValueError(f"the launch names {handle} twice")
            offsets[handle] = end
            end = (-(-(end + nbytes[handle]) // ALIGNMENT) + 1) * ALIGNMENT
        return Layout(offsets, {handle: nbytes[handle] for handle in handles})

    @property
    def end(self) -> int:
        last = max(self.offsets, key=lambda handle: self.offsets[handle])
        return self.offsets[last] + self.nbytes[last]

    def owner(self, offset: int) -> str:
        """The buffer whose bytes, or whose end, `offset` addresses."""
        found = [
            handle
            for handle, start in self.offsets.items()
            if start <= offset <= start + self.nbytes[handle]
        ]
        if len(found) != 1:
            raise ValueError(f"the offset {offset} lies in {len(found)} buffers")
        return found[0]


def checked_launch(
    binding: Binding,
    request: PrepareRequest,
    nbytes: Mapping[str, int],
    indexes: Mapping[str, tuple[bytes, ElementType]],
) -> tuple[catalog.Kernel, Layout]:
    """The catalog's check of the request's launch over a layout of its buffers
    (`Layout`), in which each index input is written with its contents, so that
    the catalog records their range: the checked kernel and the layout. A
    direct launch runs the kernel as written, so the kernel must need no check."""
    layout = Layout.of(request["buffers"], nbytes)
    tenant = TenantId(1)
    ids = {handle: BufferId(tenant, index) for index, handle in enumerate(layout.offsets)}
    operator = OperatorId(0)
    checker = catalog.Catalog(layout.end, ((operator, binding),))
    actions: list[catalog.Action] = [Reserve(0, layout.end)]
    actions.extend(
        Allocate(ids[handle], layout.offsets[handle], layout.nbytes[handle])
        for handle in layout.offsets
    )
    actions.extend(Write(ids[handle], data, element) for handle, (data, element) in indexes.items())
    x, y, z = request["grid"]
    actions.append(
        Launch(
            operator,
            (x, y, z),
            tuple(ids[handle] for handle in request["buffers"]),
            tuple(request["sizes"]),
            tuple(request["scalars"]),
            False,
        )
    )
    effects = checker.check(catalog.Step(tenant, tuple(actions)))
    if isinstance(effects, StepRefused):
        raise RuntimeError(f"the catalog refuses the direct launch: {effects.reason}")
    (kernel,) = (effect for effect in effects if isinstance(effect, catalog.Kernel))
    if kernel.checks:
        raise RuntimeError("a direct launch runs only kernels that need no checks")
    return kernel, layout


def marshal(
    kernel: catalog.Kernel, widths: Sequence[int], layout: Layout, addresses: Mapping[str, int]
) -> tuple[int, ...]:
    """The kernel's parameters as the executor marshals them, each address in
    the layout moved to the same byte of the device buffer at `addresses`."""
    values: list[int] = []
    for argument, width in zip(kernel.arguments, widths, strict=True):
        match argument:
            case At(offset=offset):
                handle = layout.owner(offset)
                values.append(addresses[handle] + offset - layout.offsets[handle])
            case Value(value=value):
                values.append(value % 2 ** (8 * width))
            case _ as never:
                assert_never(never)
    return tuple(values)


Period = Literal[1, 2, 4]


@dataclass(frozen=True)
class Memset:
    """A fill as the driver's memset: `count` units of `period` bytes, each `value`."""

    period: Period
    value: int
    count: int


def memset_of(fill: FillRequest, nbytes: int) -> Memset:
    """The memset that writes a fill's value to every element of `nbytes` bytes:
    one of the shortest unit whose repetition spells the element's bytes."""
    width = fill["element"]["width"]
    if nbytes % width:
        raise ValueError(f"{nbytes} bytes are not whole elements of {width} bytes")
    element = (fill["value"] % 2 ** (8 * width)).to_bytes(width, "little")
    periods: tuple[Period, ...] = (1, 2, 4)
    for period in periods:
        if period <= width and element == element[:period] * (width // period):
            return Memset(period, int.from_bytes(element[:period], "little"), nbytes // period)
    raise ValueError(f"no memset unit repeats the element {element.hex()}")


# ------------------------------------------------------------------ a tenant's uses


Use = Callable[[], bytes]
"""One use of a step: it runs the step and returns the bytes the step read back."""


@dataclass(frozen=True)
class Plan:
    """A template as a tenant runs it: each step's use, the kernel whose output
    the step reads back, what serves the step, and the verified use's read-back."""

    name: TemplateName
    uses: tuple[Use, ...]
    kernels: tuple[Kernel, ...]
    served: tuple[str, ...]
    verified: tuple[bytes, ...]


@dataclass(frozen=True)
class Loop:
    """The closed loop's queries in the measured window: each one's template, as
    its index in the plans, its seconds, each step's seconds, and each float64
    read-back that `Comparison.CLOSE` compares; and each use of the loop, by
    what served it."""

    template: list[int]
    latency: list[float]
    steps: list[list[float]]
    values: list[list[float]]
    calls: dict[str, int]


def check_reads(
    kernels: Sequence[Kernel],
    reads: list[dict[str, bytes]],
    workloads: Mapping[Task, Workload],
    where: str,
) -> None:
    """Check each step's outputs of a template with its task's reference;
    `masked_sum`'s reference takes the mask that the first step wrote."""
    for position, (kernel, read) in enumerate(zip(kernels, reads, strict=True)):
        w = workloads[kernel.task]
        if position:
            mask = np.frombuffer(reads[0][LINK[1]], dtype=w.arrays[LINK[0]].dtype)
            w = Workload({**w.arrays, LINK[0]: mask}, w.scalars, w.sizes)
        outputs = {
            name: np.frombuffer(data, dtype=w.arrays[name].dtype) for name, data in read.items()
        }
        failure = check(kernel.task, w, outputs)
        if failure is not None:
            raise RuntimeError(f"{where}, {kernel.kind}: {failure}")


def encoded(reads: list[dict[str, bytes]]) -> list[dict[str, str]]:
    return [
        {name: base64.b64encode(data).decode("ascii") for name, data in read.items()}
        for read in reads
    ]


def compare(comparison: Comparison, payload: bytes, verified: bytes, where: str) -> float | None:
    """Raise unless a use's read-back agrees with the verified use's; the float64
    that a `CLOSE` comparison reads."""
    match comparison:
        case Comparison.BYTES:
            if payload != verified:
                raise RuntimeError(f"{where}: the read-back differs from the verified bytes")
            return None
        case Comparison.CLOSE:
            (value,) = struct.unpack("<d", payload)
            (expected,) = struct.unpack("<d", verified)
            if not math.isclose(value, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
                raise RuntimeError(f"{where}: the read-back {value} differs from {expected}")
            return value
        case Comparison.SORTED:
            observed = np.sort(np.frombuffer(payload, dtype="<f8"))
            expected_slots = np.sort(np.frombuffer(verified, dtype="<f8"))
            if len(observed) != len(expected_slots) or not np.allclose(
                observed, expected_slots, rtol=TOLERANCE, atol=TOLERANCE
            ):
                raise RuntimeError(f"{where}: the read-back slots differ from the verified ones")
            return None
        case _ as never:
            assert_never(never)


def verify(
    name: TemplateName,
    uses: Sequence[Use],
    kernels: Sequence[Kernel],
    served: Sequence[str],
    read: Callable[[], list[dict[str, bytes]]],
    workloads: Mapping[Task, Workload],
    where: str,
) -> tuple[Plan, list[dict[str, bytes]]]:
    """Run each step of a template once, read and check all of its outputs, and
    require each step's read-back to be its output: the template's plan, and
    the outputs."""
    ran = [use() for use in uses]
    reads = read()
    check_reads(kernels, reads, workloads, f"{where}, {name}")
    for kernel, payload, outputs in zip(kernels, ran, reads, strict=True):
        if payload != outputs[READS[kernel.task][0]]:
            raise RuntimeError(f"{where}, {name}: the read-back is not the output")
    return Plan(name, tuple(uses), tuple(kernels), tuple(served), tuple(ran)), reads


def closed_loop(
    plans: Sequence[Plan], start: float, warmup: float, duration: float, where: str
) -> Loop:
    """Cycle through the templates from `start` until the window ends; the queries
    that began after the warm-up and ended in the window, and every use."""
    measured = start + warmup
    end = measured + duration
    loop = Loop([], [], [], [], {})
    index = 0
    while True:
        began = time.perf_counter()
        if began >= end:
            break
        plan = plans[index]
        marks = [began]
        ran: list[bytes] = []
        for use, served in zip(plan.uses, plan.served, strict=True):
            ran.append(use())
            marks.append(time.perf_counter())
            loop.calls[served] = loop.calls.get(served, 0) + 1
        finished = marks[-1]
        read = [
            compare(READS[kernel.task][1], payload, verified, f"{where}, {plan.name}, {served}")
            for kernel, served, payload, verified in zip(
                plan.kernels, plan.served, ran, plan.verified, strict=True
            )
        ]
        if began >= measured and finished <= end:
            loop.template.append(index)
            loop.latency.append(finished - began)
            loop.steps.append(
                [after - before for before, after in zip(marks, marks[1:], strict=False)]
            )
            loop.values.append([value for value in read if value is not None])
        index = (index + 1) % len(plans)
    if not loop.latency:
        raise RuntimeError(f"{where} completed no measured query")
    return loop


def calls_of(plans: Sequence[Plan], loop: Loop) -> dict[str, int]:
    """Every use by what served it: the verified use of each step, and the loop's."""
    calls = dict(loop.calls)
    for plan in plans:
        for served in plan.served:
            calls[served] = calls.get(served, 0) + 1
    return calls


def builtin_share(
    loops: Sequence[Loop], templates: Sequence[str], served: Mapping[str, Sequence[str]]
) -> float:
    """The built-in operators' share of the measured queries' time, over loops
    that ran `templates`, whose steps `served` names."""
    total = builtin = 0.0
    for loop in loops:
        for index, seconds, steps in zip(loop.template, loop.latency, loop.steps, strict=True):
            names = served[templates[index]]
            total += seconds
            builtin += sum(s for s, name in zip(steps, names, strict=True) if name in BY_OP)
    return builtin / total
