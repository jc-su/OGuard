"""The only execute path.

The executor owns the pool, one device allocation from which every buffer
is carved, and the catalog's current state. `Executor.register` puts the
zeroing prologue of shared and local memory at the start of the entry
kernel (`opgate.gate.zeroed`, design decision 46), verifies the result
against the binding of its entry kernel, loads exactly the text that the
gate accepted (`cuModuleLoadData`), and records the operator in the
catalog. It raises `RefusedError` if the gate refuses, if
the binding does not name the entry kernel's parameters in the text's
order, if the driver does not load the text, or if the loaded function
cannot take the binding's parameters, block, or shared memory. A load
submits no work, so a refused load leaves the executor usable.
`Executor.execute` has the catalog check a step and, if it accepts, submits
the step's effects in order, then adopts the next state: clears, uploads,
fills, and launches. Each launch's parameters are built from the checked
record and the pool's base, at the sizes the driver gives for the loaded
function, so no pointer comes from anywhere else (design section 6.3). A
kernel with checks, a guarded launch of an index input the catalog has not
checked, runs only if the trusted check kernel finds every element in its
range (section 6.4).

`Executor.capture` performs an accepted step as a CUDA graph: clears and
fills are memset nodes, launches are kernel nodes, and a kernel with checks
is the body of a conditional IF node whose condition the check kernels set.
`Executor.replay` has the catalog check the next step, which must have the
same shape of effects, updates every node's parameters from it, and
launches the graph again, so every replay is a checked step. Each launch of
a graph with checks overwrites its flags, so `replay` refuses a graph whose
last launch `ran` has not read.

`Executor.prepare` has the catalog check a step of fills and one launch that
a tenant repeats with new scalars, and gives a `Run`: the step, the buffer
each use reads back, a pinned host copy of it, and an event. `submit` has
the catalog decide one use (`Catalog.begin_prepared`) and submits its effects
and the copy on the tenant's own stream, without waiting; `finished` and
`result` collect the copy. `hot` describes a use that changes nothing in the
catalog as driver calls, which the service's compiled loop makes; such uses
of different tenants run concurrently, since each writes only buffers that
its tenant may write, which every other tenant may neither read nor write.

The device order follows the catalog's order (design decision 48). One
thread commits every change to the catalog, and every operation that
commits one first waits for the uses in progress (`quiesce`), so a free or a
status change never meets a running use. Every effect outside a use runs on
stream 0, which the executor requires to be the legacy stream: it refuses
per-thread default streams (`CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`). The
tenants' streams are blocking streams, checked when created, so the legacy
stream's work is ordered with theirs on the device.

Every buffer lies in its tenant's reservation, a region of the pool that
the tenant reserves as its session opens (`Executor.reserve`) and ends when
it holds no live buffer (`Executor.unreserve`). The executor's suballocators
are untrusted: one over the pool proposes each reservation, and one over
each reservation proposes the tenant's buffers (`Executor.allocate`), under
the tenant that trusted session code sets with `Executor.tenant`. A buffer's
id is its tenant and the tenant's own serial number. So where a tenant's
buffers lie, which ids they take, and whether an allocation fits depend only
on the tenant's own requests. Trusted host
code that writes a buffer outside the catalog, such as an engine's operator
through a writable view, has the executor keep the buffer raw while it holds
it (`Executor.keep_raw`).

Imports `opgate.binding`, `opgate.catalog`, and `opgate.gate`.
"""

import ctypes
import math
import string
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from threading import get_ident
from types import TracebackType
from typing import Self, assert_never

from opgate.binding import Base, Binding, Bounded, Constant, Parameter, Size, Zero, shared_bounds
from opgate.catalog import (
    ALIGNMENT,
    Allocate,
    At,
    BufferId,
    Catalog,
    CatalogView,
    Check,
    Clear,
    Effect,
    Free,
    Kernel,
    Launch,
    Memset,
    OperatorId,
    Pending,
    Prepared,
    Raw,
    Reserve,
    Step,
    StepRefused,
    TenantId,
    Traced,
    Unreserve,
    Upload,
    Value,
    Write,
)
from opgate.executor import trusted
from opgate.executor.driver import (
    DriverError,
    address,
    call,
    driver,
    integer,
    parameter_sizes,
)
from opgate.executor.suballocator import Suballocator
from opgate.gate import Accepted, Refused, verify, zeroed

# the catalog's identifiers, which a host names buffers and tenants by
__all__ = [
    "Hot",
    "READ_BYTES",
    "BufferId",
    "CheckFailedError",
    "Executor",
    "Graph",
    "Module",
    "RefusedError",
    "Run",
    "TenantId",
]

# the characters of a PTX name, which the gate's parser reads (`ident_cont`)
_NAME = frozenset(string.ascii_letters + string.digits + "_$")
_UNSIGNED: dict[int, trusted.CType] = {
    1: ctypes.c_uint8,
    2: ctypes.c_uint16,
    4: ctypes.c_uint32,
    8: ctypes.c_uint64,
}
_MEMSET = {1: driver.cuMemsetD8, 2: driver.cuMemsetD16, 4: driver.cuMemsetD32}
_MEMSET_ASYNC = {1: driver.cuMemsetD8Async, 2: driver.cuMemsetD16Async, 4: driver.cuMemsetD32Async}
READ_BYTES = 65536  # the most bytes that a use of a prepared step reads back
KEPT_LAUNCHES = 1024  # the most prepared launches that the executor keeps for `launch`
_SUCCESS = driver.CUresult.CUDA_SUCCESS
_NOT_READY = driver.CUresult.CUDA_ERROR_NOT_READY


class RefusedError(Exception):
    """The gate refused a kernel, or the catalog refused a step; the reason says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CheckFailedError(Exception):
    """A check that the guard pass added failed when the kernel ran: an access did
    not run, or a loop reached its iteration limit and ended, and the kernel
    stored 1 to its status buffer."""

    def __init__(self, kernel: str) -> None:
        super().__init__(
            f"{kernel}: a check failed: an access did not run, or a loop reached its limit"
        )
        self.kernel = kernel


def _find(text: str, part: str, i: int) -> int:
    """The index of `part` in `text` at or after `i`, or the text's length."""
    found = text.find(part, i)
    return len(text) if found < 0 else found


def _structure(text: str) -> str:
    """The text as the gate's parser reads its structure (`strip_go` in
    `formal/GateParse.v`): each comment blank but its line breaks, and the
    characters of each string blank."""
    out: list[str] = []
    # the next quote and the next slash at or after i, or the text's end
    i, quote, slash = 0, -1, -1
    while i < len(text):
        if quote < i:
            quote = _find(text, '"', i)
        if slash < i:
            slash = _find(text, "/", i)
        start = min(quote, slash)
        out.append(text[i:start])
        if start == len(text):
            break
        if start == quote:
            end = text.find('"', start + 1)
            if end < 0:
                raise RefusedError("a string of the text does not end")
            out.append('"' + " " * (end - start - 1) + '"')
        elif text.startswith("//", start):
            end = text.find("\n", start)
            if end < 0:
                break
            out.append(" " * (end - start) + "\n")
        elif text.startswith("/*", start):
            end = text.find("*/", start + 2) + 1
            if end < 1:
                raise RefusedError("a block comment of the text does not end")
            lines = text[start : end + 1].split("\n")
            out.append("\n".join(" " * len(line) for line in lines))
        else:
            end = start
            out.append("/")
        i = end + 1
    return "".join(out)


def _skip_space(structure: str, i: int) -> int:
    while i < len(structure) and structure[i].isspace():
        i += 1
    return i


def _declared_name(declaration: str, kernel: str) -> str:
    """The name of a `.param` declaration: its last word, without an array's length."""
    words = declaration.split()
    if not words or words[0] != ".param":
        raise RefusedError(f"{kernel}: a parameter declaration {declaration.strip()!r}")
    last = words[-1]
    name = last.removesuffix("]").partition("[")[0] if last.endswith("]") else last
    if not name or name[0].isdigit() or not set(name) <= _NAME:
        raise RefusedError(f"{kernel}: a parameter declaration {declaration.strip()!r}")
    return name


def _declared_names(structure: str, start: int, kernel: str) -> tuple[str, ...]:
    """The names of the parameter list whose `(` is at `start`, in order. A comma
    inside brackets does not end a declaration."""
    parts: list[str] = []
    depth, first = 0, start + 1
    for i in range(start, len(structure)):
        c = structure[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0 and c != ")":
                raise RefusedError(f"{kernel}: the parameter list closes with {c!r}")
            if depth == 0:
                parts.append(structure[first:i])
                break
        elif c == "," and depth == 1:
            parts.append(structure[first:i])
            first = i + 1
    else:
        raise RefusedError(f"{kernel}: the parameter list does not end")
    if len(parts) == 1 and not parts[0].strip():
        return ()
    return tuple(_declared_name(part, kernel) for part in parts)


def _entry_parameters(text: str, kernel: str) -> tuple[str, ...]:
    """The parameter names of the entry kernel `kernel` of `text`, in the order
    that the text declares them, which is the order of the loaded function's
    parameters.

    The gate finds a parameter by its name, and the executor marshals a
    parameter by its position, so `register` requires the binding to list
    the names in this order. The names are read as the gate's parser reads
    them: comments and strings are blank, `.entry` is a word, and a comma
    inside brackets does not end a declaration."""
    structure = _structure(text)
    found: list[tuple[str, ...]] = []
    at = structure.find(".entry")
    while at >= 0:
        end = at + len(".entry")
        if (at == 0 or structure[at - 1] not in _NAME) and structure[end : end + 1] not in _NAME:
            first = _skip_space(structure, end)
            last = first
            while last < len(structure) and structure[last] in _NAME:
                last += 1
            if structure[first:last] == kernel:
                opening = _skip_space(structure, last)
                found.append(
                    _declared_names(structure, opening, kernel)
                    if structure.startswith("(", opening)
                    else ()
                )
        at = structure.find(".entry", end)
    if len(found) != 1:
        raise RefusedError(f"{kernel}: the text declares {len(found)} entry kernels of that name")
    return found[0]


def _holds(parameter: Parameter, size: int) -> bool:
    """Whether a parameter of `size` bytes holds every value that the binding says
    `parameter` holds. A buffer's base is 8 bytes and a scalar 1, 2, 4, or 8;
    a constant and a bounded integer fit in the scalar's bits, and a size is
    passed modulo 2 to its bits, as the gate reads it."""
    if size not in _UNSIGNED:
        return False
    match parameter:
        case Base():
            return size == 8
        case Constant(value=value):
            return 0 <= value < 2 ** (8 * size)
        case Bounded(lower=lower, upper=upper):
            return 0 <= lower and upper < 2 ** (8 * size)
        case Size() | Zero():
            return True
        case _ as never:
            assert_never(never)


def _widths(binding: Binding, sizes: tuple[int, ...]) -> tuple[int, ...]:
    """`sizes`, the byte size of each parameter of the loaded function, if each
    parameter holds what the binding says it holds."""
    if len(sizes) != len(binding.parameters):
        raise RefusedError(
            f"{binding.kernel}: the loaded function has {len(sizes)} parameters, "
            f"the binding {len(binding.parameters)}"
        )
    for (name, parameter), size in zip(binding.parameters, sizes, strict=True):
        if not _holds(parameter, size):
            raise RefusedError(
                f"{binding.kernel}: parameter {name} of {size} bytes cannot hold {parameter}"
            )
    return sizes


@dataclass(frozen=True)
class Module:
    """A loaded operator: the text the gate accepted, the binding it accepted it
    against, which states the block and the dynamic shared memory of each
    launch, the driver's handles, and the byte size of each parameter as the
    driver lays out the loaded function."""

    operator: OperatorId
    text: str
    binding: Binding
    handle: object  # CUmodule
    function: object  # CUfunction
    widths: tuple[int, ...]
    _types: tuple[trusted.CType, ...] = field(init=False, repr=False)
    _masks: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The driver's parameter layout is fixed for the loaded function. Keep
        # only immutable layout metadata; addresses and values come from each
        # launch's freshly checked effect.
        object.__setattr__(self, "_types", tuple(_UNSIGNED[width] for width in self.widths))
        object.__setattr__(self, "_masks", tuple((1 << (8 * width)) - 1 for width in self.widths))


@dataclass(frozen=True)
class _Launch:
    """A kernel launch as the driver takes it."""

    function: object
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared: int
    values: tuple[int, ...]
    types: tuple[trusted.CType, ...]


@dataclass(frozen=True)
class _MemsetNode:
    node: object


@dataclass(frozen=True)
class _KernelNode:
    node: object


@dataclass(frozen=True)
class _CheckedNodes:
    """A kernel with checks: the reset of its flag, one node per check, the node
    that sets the condition, the conditional node, and the kernel node in the
    conditional's body."""

    reset: object
    checks: tuple[object, ...]
    decide: object
    conditional: object
    body: object
    flag: int  # the address of its flag


_Nodes = _MemsetNode | _KernelNode | _CheckedNodes

# The part of an effect that fixes its nodes: a memset of an element width, a
# fill of 8-byte elements, or a kernel with a number of checks.
_Shape = tuple[str, int]


@dataclass(frozen=True)
class Graph:
    """A step's effects as an instantiated CUDA graph, one node or group of
    nodes per effect, in a chain in the effects' order. A replay updates the
    nodes from a step whose effects have the same shape. `kernels` is the
    flag address of each kernel's checks, or None for a kernel without them."""

    shape: tuple[_Shape, ...]
    graph: object  # CUgraph
    instance: object  # CUgraphExec
    nodes: tuple[_Nodes, ...]
    flags: int  # the address of the flags, or 0 if no kernel has checks
    kernels: tuple[int | None, ...]


class _Parameters:
    """A loaded function's parameter block: an 8-byte slot per parameter, which
    holds the parameter's value in its low bytes, and the array of the slots'
    addresses that `cuLaunchKernel` reads. The driver copies each parameter's
    bytes, at the loaded function's sizes, when it takes a launch, so one block
    serves every launch."""

    def __init__(self, count: int) -> None:
        self.slots = (ctypes.c_uint64 * count)()
        base = ctypes.addressof(self.slots)
        self.pointers = (ctypes.c_void_p * count)(*(base + 8 * i for i in range(count)))
        self.address = ctypes.addressof(self.pointers)


@dataclass(eq=False)
class _Kept:
    """A launch that the executor keeps prepared for `Executor.launch`: the
    catalog's prepared step of the launch, its parameter block, and the driver
    call and parameter masks built for the catalog's memo `memo`."""

    prepared: Prepared
    parameters: _Parameters
    memo: object | None = None
    submit: Callable[[], tuple[object, ...]] | None = None
    masks: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Fill:
    """A fill of a use as a driver memset of `count` elements of `width` bytes."""

    width: int  # 1, 2, or 4
    address: int
    value: int
    count: int


@dataclass(frozen=True)
class Hot:
    """A use of a run that the catalog's memo of its step decides at the current
    state and that changes nothing in the catalog: the bounds of its scalars,
    the parameter that each scalar fills, and the use's driver calls, with
    their handles and addresses as integers. A native loop that checks the
    scalars against `bounds` and makes these calls performs the use that
    `submit` would perform."""

    bounds: tuple[tuple[int, int], ...]  # each scalar's lowest and highest value
    parameters: tuple[int, ...]  # the parameter that each scalar fills
    masks: tuple[int, ...]  # each parameter's mask, by its width
    block: ctypes.Array[ctypes.c_uint64]  # the parameter block's slots
    pointers: int  # the address of the slots' addresses
    fills: tuple[_Fill, ...]
    function: int
    grid: tuple[int, int, int]
    threads: tuple[int, int, int]
    shared: int
    stream: int
    event: int
    host: int  # the pinned copy of the read buffer
    source: int  # the read buffer's device address
    nbytes: int
    status_host: int  # zero if the run has no check status
    status_source: int


class _Session:
    """The block of trusted session code for one tenant (`Executor.tenant`)."""

    __slots__ = ("_executor", "_tenant", "_token")

    def __init__(self, executor: "Executor", tenant: TenantId) -> None:
        self._executor = executor
        self._tenant = tenant

    def __enter__(self) -> None:
        self._token = self._executor._tenant.set(self._tenant)
        self._executor._observe(self._tenant)

    def __exit__(self, *_: object) -> None:
        self._executor._tenant.reset(self._token)
        self._executor._observe(self._executor._tenant.get(None))


@dataclass(eq=False)
class Run:
    """A prepared step with its tenant's stream, the buffer each use reads back, a
    pinned host copy of that buffer, and the event that ends each use. One use
    runs at a time."""

    prepared: Prepared
    read: BufferId
    nbytes: int
    stream: object  # CUstream
    event: object  # CUevent
    host: int  # the address of the pinned copy
    parameters: _Parameters  # the launch's parameter block
    status: BufferId | None = None  # an internal four-byte check status
    status_source: int = 0
    running: bool = False
    memo: object = None  # the catalog memo that the fields below were built for
    calls: tuple[Callable[[], tuple[object, ...]], ...] = ()
    module: Module | None = None
    fills: tuple[_Fill, ...] | None = None  # None if a fill needs the trusted fill kernel
    kernel: Kernel | None = None
    source: int = 0  # the read buffer's device address


def _uniform(value: int) -> bool:
    """Whether the 8 bytes of a fill's `value` are all equal, so a byte memset sets it."""
    pattern = value % 2**64
    return pattern == (pattern & 0xFF) * 0x0101010101010101


def _memset_arguments(fill: _Fill, stream: object) -> tuple[object, ...]:
    """The arguments of the driver's memset of `fill`'s width, on `stream`."""
    return (fill.address, fill.value, fill.count, stream)


def _memset_node_params(destination: object, width: int, count: int, value: int) -> object:
    """A memset node of `count` elements of `width` bytes, each `value`, at `destination`."""
    params = driver.CUDA_MEMSET_NODE_PARAMS()
    params.dst = destination
    params.value = value
    params.elementSize = width
    params.width = count
    params.height = 1
    params.pitch = width * count
    return params


def _shape(effect: Effect) -> _Shape:
    match effect:
        case Clear():
            return ("memset", 1)
        case Memset(element=element) if element.width in _MEMSET:
            return ("memset", element.width)
        case Memset():
            return ("fill8", 8)
        case Kernel(checks=checks):
            return ("kernel", len(checks))
        case Upload():
            raise ValueError("a graph holds no upload: perform uploads with execute")
        case _ as never:
            assert_never(never)


class Executor:
    """The execute path on the primary context of one device, over one pool."""

    def __init__(self, device: int, pool_bytes: int) -> None:
        self._thread = get_ident()  # the owning thread
        self._closed = False
        self._failure: BaseException | None = None
        self._tenant: ContextVar[TenantId] = ContextVar("tenant")
        self._observer: Callable[[TenantId | None], None] | None = None
        call(driver.cuInit(0))
        (self._device,) = call(driver.cuDeviceGet(device))
        with ExitStack() as cleanup:
            (self._context,) = call(driver.cuDevicePrimaryCtxRetain(self._device))
            cleanup.callback(lambda: call(driver.cuDevicePrimaryCtxRelease(self._device)))
            call(driver.cuCtxSetCurrent(self._context))
            # Every effect outside a use goes to stream 0, which must be the legacy
            # stream: the tenants' blocking streams are ordered with it and not with
            # a per-thread default stream, which cuda-python makes stream 0 under
            # CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM.
            legacy = integer(driver.cuStreamGetId(driver.CU_STREAM_LEGACY))
            if integer(driver.cuStreamGetId(0)) != legacy:
                raise RuntimeError(
                    "stream 0 is a per-thread default stream "
                    "(CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM), not the legacy stream"
                )
            (pool,) = call(driver.cuMemAlloc(pool_bytes))
            cleanup.callback(lambda: call(driver.cuMemFree(pool)))
            self._pool = address(pool)
            if self._pool % ALIGNMENT != 0:
                raise RuntimeError("the pool's base is not aligned to 256 bytes")
            self._trusted = trusted.load(self._device)
            cleanup.callback(lambda: call(driver.cuModuleUnload(self._trusted.handle)))
            # the flag of an eager kernel's checks, outside the pool
            (flag,) = call(driver.cuMemAlloc(4))
            cleanup.callback(lambda: call(driver.cuMemFree(flag)))
            self._flag = address(flag)
            self._catalog = Catalog(pool_bytes)
            # proposes each tenant's reservation
            self._reservations = Suballocator(0, pool_bytes)
            # each tenant's suballocator, over its reservation
            self._suballocators: dict[TenantId, Suballocator] = {}
            # each live buffer, and its offset
            self._allocated: dict[BufferId, int] = {}
            # the serial number of each tenant's last buffer
            self._serials: dict[TenantId, int] = {}
            self._modules: dict[OperatorId, Module] = {}
            self._registered: dict[tuple[str, Binding], OperatorId] = {}
            self._graphs: list[Graph] = []
            # each graph with checks whose last launch `ran` has not read
            self._unread: list[Graph] = []
            self._streams: dict[TenantId, object] = {}  # each tenant's stream for its runs
            self._runs: set[Run] = set()
            # each buffer that a holder keeps raw (`keep_raw`), with its holders
            self._raw: dict[BufferId, list[weakref.ref[object]]] = {}
            # the launches of in-process hosts, by tenant and launch without its scalars
            self._kept: dict[tuple[TenantId, Launch], _Kept] = {}
            cleanup.pop_all()

    def _assert_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("the executor must be used on its owning thread")

    def _ensure_active(self) -> None:
        self._assert_thread()
        if self._closed:
            raise RuntimeError("the executor is closed")
        if self._failure is not None:
            raise RuntimeError("the executor failed and cannot be reused") from self._failure

    @contextmanager
    def _device_work(self) -> Iterator[None]:
        """A failed submission or synchronization makes all later work fail fast.

        Earlier effects may already have run, so rolling back a Python catalog
        cannot recover the device. Only resource cleanup remains allowed.
        """
        self._ensure_active()
        try:
            yield
        except BaseException as error:
            self._failure = error
            raise

    @property
    def catalog(self) -> CatalogView:
        """The catalog, which only the executor changes."""
        self._ensure_active()
        return self._catalog

    @property
    def pool_address(self) -> int:
        """The device address of the pool's first byte."""
        self._ensure_active()
        return self._pool

    def register(self, text: str, binding: Binding) -> Module:
        """Verify `text` with the zeroing prologue of its shared and local memory
        against `binding`, load that text, and record the operator in the
        catalog; the module's text is the loaded one. Raise `RefusedError` if the
        text does not have exactly one entry kernel of the binding's name with a
        body for the prologue, if the gate refuses, if the binding does not name the entry
        kernel's parameters in the text's order, if the driver does not load the text,
        or if the loaded function cannot take the binding's parameters, block,
        or shared memory; a refused text is unloaded. Loading submits no work,
        so a refusal leaves the executor usable."""
        self._ensure_active()
        key = (text, binding)
        if key in self._registered:
            return self._modules[self._registered[key]]
        try:
            text = zeroed(text, binding)
        except ValueError as error:
            raise RefusedError(
                f"{binding.kernel}: the zeroing prologue cannot be placed: {error}"
            ) from error
        match verify(text, [binding]):
            case Accepted():
                pass
            case Refused(reason=reason):
                raise RefusedError(reason)
            case _ as never:
                assert_never(never)
        declared = _entry_parameters(text, binding.kernel)
        named = tuple(name for name, _ in binding.parameters)
        if declared != named:
            raise RefusedError(
                f"{binding.kernel}: the binding names the parameters {named}, "
                f"the text declares {declared}"
            )
        call(driver.cuCtxSetCurrent(self._context))
        with ExitStack() as cleanup:
            try:
                # the driver reads the text up to its terminating zero byte
                (handle,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
            except DriverError as error:
                raise RefusedError(
                    f"{binding.kernel}: the driver does not load it: {error}"
                ) from error
            cleanup.callback(lambda: call(driver.cuModuleUnload(handle)))
            try:
                (function,) = call(
                    driver.cuModuleGetFunction(handle, binding.kernel.encode("ascii"))
                )
            except DriverError as error:
                raise RefusedError(
                    f"{binding.kernel}: the driver finds no entry: {error}"
                ) from error
            widths = _widths(binding, parameter_sizes(function))
            self._check_configuration(function, binding)
            operator = OperatorId(len(self._modules))
            module = Module(operator, text, binding, handle, function, widths)
            self._catalog.add_operator(operator, binding)
            self._modules[operator] = module
            self._registered[key] = operator
            cleanup.pop_all()
            return module

    def _check_configuration(self, function: object, binding: Binding) -> None:
        """Refuse a block or dynamic shared memory with which the device does not
        launch the loaded `function`, so that no launch fails for them."""
        device = driver.CUdevice_attribute
        attribute = driver.CUfunction_attribute
        dimensions = tuple(
            integer(driver.cuDeviceGetAttribute(limit, self._device))
            for limit in (
                device.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                device.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                device.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            )
        )
        if any(extent > most for extent, most in zip(binding.block, dimensions, strict=True)):
            raise RefusedError(
                f"{binding.kernel}: the block {binding.block} exceeds the device's {dimensions}"
            )
        threads = math.prod(binding.block)
        most = integer(
            driver.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function)
        )
        if threads > most:
            raise RefusedError(
                f"{binding.kernel}: a block of {threads} threads, and the loaded function "
                f"runs at most {most}"
            )
        static = integer(
            driver.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function)
        )
        limit = integer(
            driver.cuDeviceGetAttribute(
                device.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, self._device
            )
        )
        _, dynamic = shared_bounds(binding.sizes, binding.shared)
        if static + dynamic > limit:
            raise RefusedError(
                f"{binding.kernel}: {static} bytes of static and up to {dynamic} of dynamic "
                f"shared memory, and the device has {limit} for a block"
            )

    @property
    def session_tenant(self) -> TenantId:
        """The tenant that trusted session code set with `tenant`, which host
        adapters read and do not set."""
        self._ensure_active()
        tenant = self._tenant.get(None)
        if tenant is None:
            raise RefusedError("an operation outside a tenant's session")
        return tenant

    def session(self) -> TenantId | None:
        """The session's tenant, or None outside every session."""
        self._ensure_active()
        return self._tenant.get(None)

    def tenant(self, tenant: TenantId) -> "_Session":
        """Run the block for `tenant`: the allocation wrapper allocates for it."""
        self._ensure_active()
        return _Session(self, tenant)

    @contextmanager
    def trace(self) -> Iterator[tuple[Catalog, list[Traced]]]:
        """Record the catalog's decisions in the block (`Catalog.trace`): the catalog
        as the block begins, and what it records, for a replay through Rocq
        (`opgate.catalog.rocq.replay`)."""
        self._ensure_active()
        start = self._catalog.copy()
        recorded = self._catalog.trace()
        try:
            yield start, recorded
        finally:
            self._catalog.untrace()

    def observe_sessions(self, observer: Callable[[TenantId | None], None] | None) -> None:
        """Call `observer` with the session's tenant as each session begins and ends,
        and with None outside every session: the allocation wrapper makes the
        tenant's allocator the engine's. One observer at a time; None removes it."""
        if observer is not None and self._observer is not None:
            raise RuntimeError("the executor's sessions have an observer")
        self._observer = observer

    def _observe(self, tenant: TenantId | None) -> None:
        if self._observer is not None:
            self._observer(tenant)

    def _begin(self, step: Step) -> Pending:
        """Apply `step` to the catalog, pending its effects, if the catalog accepts it."""
        if step.tenant != self.session_tenant:
            raise RefusedError("the step's tenant is not the session's tenant")
        match self._catalog.begin(step):
            case Pending() as pending:
                self._keep_raw_buffers(pending)
                return pending
            case StepRefused(reason=reason):
                raise RefusedError(reason)
            case _ as never:
                assert_never(never)

    def keep_raw(self, buffer: BufferId, holder: object) -> None:
        """Make the session tenant's `buffer` raw and keep it raw while `holder` lives.

        Trusted host code writes the buffer outside the catalog through `holder`,
        such as a writable view that an engine's operator takes, so the buffer
        may hold any value until the holder is collected. The buffer becomes raw
        in a step (an empty write without an element type), which the tenant
        must be allowed to take; until the holder is collected, the executor
        refuses every step and every use of a prepared step that frees the buffer
        or records a status for it other than raw, as a fill or a write with an
        element type does."""
        holding = weakref.ref(holder)
        self.execute(Step(self.session_tenant, (Write(buffer, b"", None),)))
        self._raw.setdefault(buffer, []).append(holding)

    def _keep_raw_buffers(self, pending: Pending) -> None:
        """Roll back `pending` and refuse it if it frees a buffer that a live holder
        keeps raw (`keep_raw`) or records another status for it. A buffer whose
        holders are all collected leaves the kept buffers."""
        if not self._raw:
            return
        for buffer, holders in list(self._raw.items()):
            live = [holding for holding in holders if holding() is not None]
            if not live:
                del self._raw[buffer]
                continue
            self._raw[buffer] = live
            found = self._catalog.buffer(buffer)
            if found is None or not isinstance(found.status, Raw):
                pending.roll_back()
                raise RefusedError(
                    f"buffer {buffer} is written outside the catalog while its holder lives: "
                    "it stays live and raw"
                )

    def execute(self, step: Step) -> tuple[bool, ...]:
        """Check and submit the session's step, then commit its catalog state."""
        self.quiesce()
        pending = self._begin(step)
        if not pending.effects:
            # an allocation that keeps its bytes, a free, or a grant: nothing to submit
            pending.commit()
            return ()
        ran: list[bool] = []
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            for effect in pending.effects:
                match effect:
                    case Clear(offset=offset, nbytes=nbytes):
                        call(driver.cuMemsetD8(self._at(offset), 0, nbytes))
                    case Upload(offset=offset, data=data):
                        call(driver.cuMemcpyHtoD(self._at(offset), data, len(data)))
                    case Memset():
                        self._memset(effect)
                    case Kernel(checks=checks):
                        passed = self._checks_hold(checks)
                        if passed:
                            self._run(self._kernel_launch(effect), 0)
                        ran.append(passed)
                    case _ as never:
                        assert_never(never)
            pending.commit()
        return tuple(ran)

    def launch(self, launch: Launch) -> bool:
        """Check and submit one launch of the session's tenant, as `execute` does
        the step of the launch alone; whether the kernel ran.

        A launch without index checks goes through a prepared step of the launch,
        which the executor keeps for the tenant and the launch without its
        scalars (at most `KEPT_LAUNCHES`, the oldest forgotten first): while the
        records of the buffers that it names are those of its last full check,
        the catalog checks only the scalars (`Catalog.begin_prepared`), and the
        driver call, built once for that check, takes them. A launch that the
        catalog does not prepare or whose use it refuses, as one that needs index
        checks, is a step, which `execute` decides."""
        self.quiesce()
        tenant = self.session_tenant
        key = (tenant, replace(launch, scalars=()))
        kept = self._kept.get(key)
        if kept is None:
            prepared = self._catalog.prepare(Step(tenant, (launch,)))
            if isinstance(prepared, StepRefused):
                (ran,) = self.execute(Step(tenant, (launch,)))
                return ran
            module = self._modules[launch.operator]
            kept = self._kept[key] = _Kept(prepared, _Parameters(len(module._types)))
            if len(self._kept) > KEPT_LAUNCHES:
                oldest = next(iter(self._kept))
                self._catalog.unprepare(self._kept.pop(oldest).prepared)
        use = self._catalog.begin_prepared(kept.prepared, launch.scalars)
        if isinstance(use, StepRefused):
            del self._kept[key]
            self._catalog.unprepare(kept.prepared)
            (ran,) = self.execute(Step(tenant, (launch,)))
            return ran
        # A launch alone frees nothing and leaves a raw buffer raw, since the
        # catalog lets it write an index span of a checked buffer alone: a
        # buffer that `keep_raw` keeps stays raw through it.
        if use.memo is not kept.memo:
            try:
                self._compile_kept(kept, use.memo, use.effects)
            except BaseException:
                use.roll_back()
                raise
        slots, masks = kept.parameters.slots, kept.masks
        for index, value in zip(use.parameters, launch.scalars, strict=True):
            slots[index] = value & masks[index]
        submit = kept.submit
        if submit is None:
            raise AssertionError("a compiled kept launch has its driver call")
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            call(submit())
            use.commit()
        return True

    def _compile_kept(self, kept: _Kept, memo: object, effects: tuple[Effect, ...]) -> None:
        """The driver call of a kept launch whose catalog memo is `memo`, with the
        kernel's parameters other than its scalars written into the block."""
        (kernel,) = effects
        if not isinstance(kernel, Kernel) or kernel.checks:
            raise AssertionError("a kept launch's one effect is its kernel without checks")
        module = self._modules[kernel.operator]
        slots = kept.parameters.slots
        for index, (argument, mask) in enumerate(zip(kernel.arguments, module._masks, strict=True)):
            match argument:
                case At(offset=offset):
                    slots[index] = self._pool + offset
                case Value(value=number):
                    slots[index] = number & mask
                case _ as never:
                    assert_never(never)
        kept.submit = partial(
            driver.cuLaunchKernel,
            module.function,
            *kernel.grid,
            *module.binding.block,
            kernel.shared,
            0,
            kept.parameters.address,
            0,
        )
        kept.masks = module._masks
        kept.memo = memo

    def capture(self, step: Step) -> Graph:
        """Check `step`, launch its CUDA graph, then adopt the next state.

        `replay` launches the graph again with a later step's effects.
        The graph holds no upload.
        """
        self.quiesce()
        pending = self._begin(step)
        effects = pending.effects
        try:
            shape = tuple(_shape(effect) for effect in effects)
        except BaseException:
            pending.roll_back()
            raise
        with self._device_work(), ExitStack() as cleanup:
            call(driver.cuCtxSetCurrent(self._context))
            checked = sum(1 for kind, count in shape if kind == "kernel" and count > 0)
            flags = address(call(driver.cuMemAlloc(4 * checked))[0]) if checked else 0
            if flags:
                cleanup.callback(lambda: call(driver.cuMemFree(driver.CUdeviceptr(flags))))
            (graph,) = call(driver.cuGraphCreate(0))
            cleanup.callback(lambda: call(driver.cuGraphDestroy(graph)))
            last: list[object] = []
            nodes: list[_Nodes] = []
            kernels: list[int | None] = []
            for effect in effects:
                match effect:
                    case Clear():
                        node = self._add_memset(graph, last, self._memset_params(effect))
                        nodes.append(_MemsetNode(node))
                    case Memset(element=element) if element.width in _MEMSET:
                        node = self._add_memset(graph, last, self._memset_params(effect))
                        nodes.append(_MemsetNode(node))
                    case Memset():
                        node = self._add_kernel(graph, last, self._fill8_launch(effect))
                        nodes.append(_KernelNode(node))
                    case Kernel(checks=()):
                        node = self._add_kernel(graph, last, self._kernel_launch(effect))
                        nodes.append(_KernelNode(node))
                        kernels.append(None)
                    case Kernel(checks=checks):
                        flag = flags + 4 * sum(1 for k in kernels if k is not None)
                        group = self._add_checked(graph, last, effect, checks, flag)
                        nodes.append(group)
                        kernels.append(flag)
                        node = group.conditional
                    case Upload():
                        raise AssertionError("_shape refused the upload")
                    case _ as never:
                        assert_never(never)
                last = [node]
            (instance,) = call(driver.cuGraphInstantiate(graph, 0))
            cleanup.callback(lambda: call(driver.cuGraphExecDestroy(instance)))
            captured = Graph(shape, graph, instance, tuple(nodes), flags, tuple(kernels))
            call(driver.cuGraphLaunch(instance, 0))
            self._graphs.append(captured)
            if flags:
                self._unread.append(captured)
            pending.commit()
            cleanup.pop_all()
            return captured

    def replay(self, graph: Graph, step: Step) -> None:
        """Have the catalog check `step`, whose effects must have the graph's shape,
        update every node, launch, then adopt the next state.

        A launch overwrites the flags of the graph's checks, so a graph with
        checks replays only after `ran` has read its last launch's flags."""
        self._require_graph(graph)
        if any(unread is graph for unread in self._unread):
            raise RuntimeError("the graph's last launch is unread: call ran before its replay")
        self.quiesce()
        pending = self._begin(step)
        effects = pending.effects
        try:
            shape = tuple(_shape(effect) for effect in effects)
            if shape != graph.shape:
                raise ValueError(f"the step's effects have the shape {shape}, not the graph's")
        except BaseException:
            pending.roll_back()
            raise
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            for effect, nodes in zip(effects, graph.nodes, strict=True):
                self._update(graph.instance, effect, nodes)
            call(driver.cuGraphLaunch(graph.instance, 0))
            if graph.flags:
                self._unread.append(graph)
            pending.commit()

    def _update(self, instance: object, effect: Effect, nodes: _Nodes) -> None:
        """Set the parameters of an effect's nodes; the effect has the nodes' shape."""
        mismatch = AssertionError("an effect of the graph's shape has the nodes of that shape")
        match nodes:
            case _MemsetNode(node=node):
                if not isinstance(effect, Clear | Memset):
                    raise mismatch
                params = self._memset_params(effect)
                call(driver.cuGraphExecMemsetNodeSetParams(instance, node, params, self._context))
            case _KernelNode(node=node):
                match effect:
                    case Memset():
                        self._set_kernel(instance, node, self._fill8_launch(effect))
                    case Kernel():
                        self._set_kernel(instance, node, self._kernel_launch(effect))
                    case _:
                        raise mismatch
            case _CheckedNodes(checks=check_nodes, body=body, flag=flag):
                if not isinstance(effect, Kernel):
                    raise mismatch
                for check, node in zip(effect.checks, check_nodes, strict=True):
                    self._set_kernel(instance, node, self._check_launch(check, flag))
                self._set_kernel(instance, body, self._kernel_launch(effect))
            case _ as never:
                assert_never(never)

    def ran(self, graph: Graph) -> tuple[bool, ...]:
        """Whether each kernel of the graph's last launch ran, in order."""
        self._require_graph(graph)
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            call(driver.cuCtxSynchronize())
            count = sum(1 for flag in graph.kernels if flag is not None)
            words = bytearray(4 * count)
            if count:
                call(driver.cuMemcpyDtoH(words, driver.CUdeviceptr(graph.flags), len(words)))
            self._read(graph)
            return tuple(
                flag is None
                or int.from_bytes(words[flag - graph.flags : flag - graph.flags + 4], "little") == 1
                for flag in graph.kernels
            )

    def _read(self, graph: Graph) -> None:
        """The graph's last launch is read, or is discarded with the graph."""
        self._unread = [unread for unread in self._unread if unread is not graph]

    def release(self, graph: Graph) -> None:
        """Destroy a graph that `capture` made."""
        self._require_graph(graph)
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            call(driver.cuCtxSynchronize())
            call(driver.cuGraphExecDestroy(graph.instance))
            call(driver.cuGraphDestroy(graph.graph))
            if graph.flags:
                call(driver.cuMemFree(driver.CUdeviceptr(graph.flags)))
            self._read(graph)
            self._graphs.remove(graph)

    def _require_graph(self, graph: Graph) -> None:
        self._ensure_active()
        if not any(owned is graph for owned in self._graphs):
            raise ValueError("the graph is not live in this executor")

    def reserve(self, nbytes: int) -> None:
        """Reserve `nbytes` of the pool for the current tenant, which holds no
        reservation: a region that the suballocator over the pool proposes, in
        which every buffer of the tenant lies until `unreserve`. Raise
        `MemoryError` if no free region of the pool holds it."""
        tenant = self.session_tenant
        offset = self._reservations.propose(nbytes)
        try:
            self.execute(Step(tenant, (Reserve(offset, nbytes),)))
        except RefusedError:
            self._reservations.release(offset)
            raise
        self._suballocators[tenant] = Suballocator(offset, nbytes)

    def unreserve(self) -> None:
        """End the current tenant's reservation, once it holds no live buffer."""
        tenant = self.session_tenant
        self.execute(Step(tenant, (Unreserve(),)))
        suballocator = self._suballocators.pop(tenant)
        self._reservations.release(suballocator.base)

    def allocate(self, nbytes: int) -> BufferId:
        """A buffer of `nbytes` for the current tenant, at a region of its
        reservation that the tenant's suballocator proposes; `free` frees it.
        Raise `MemoryError` if no free region of the reservation holds it."""
        tenant = self.session_tenant
        suballocator = self._suballocators.get(tenant)
        if suballocator is None:
            raise RefusedError("the tenant holds no reservation")
        offset = suballocator.propose(nbytes)
        serial = self._serials.get(tenant, 0) + 1
        buffer = BufferId(tenant, serial)
        try:
            self.execute(Step(tenant, (Allocate(buffer, offset, nbytes),)))
        except RefusedError:
            suballocator.release(offset)
            raise
        self._serials[tenant] = serial
        self._allocated[buffer] = offset
        return buffer

    def free(self, buffer: BufferId, *more: BufferId) -> None:
        """Free buffers of the current tenant, in one step."""
        tenant = self.session_tenant
        buffers = (buffer, *more)
        self.quiesce()
        pending = self._begin(Step(tenant, tuple(Free(b) for b in buffers)))
        with self._device_work():
            suballocator = self._suballocators[tenant]
            for freed in buffers:
                suballocator.release(self._allocated.pop(freed))
            pending.commit()

    def read(self, tenant: TenantId, buffer: BufferId) -> bytes:
        """The contents of a buffer that `tenant` may read, after the work before it."""
        if tenant != self.session_tenant:
            raise RefusedError("the read's tenant is not the session's tenant")
        found = self._catalog.buffer(buffer)
        if found is None or not self._catalog.readable(tenant, buffer):
            raise RefusedError("the tenant may not read the buffer")
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            call(driver.cuCtxSynchronize())
            data = bytearray(found.nbytes)
            call(driver.cuMemcpyDtoH(data, self._at(found.offset), found.nbytes))
            return bytes(data)

    def synchronize(self) -> None:
        with self._device_work():
            call(driver.cuCtxSetCurrent(self._context))
            call(driver.cuCtxSynchronize())

    # ------------------------------------------------------------------
    # Prepared steps, submitted on each tenant's stream.

    def prepare(self, step: Step, read: BufferId, *, status: BufferId | None = None) -> Run:
        """Have the catalog check a step of fills and one launch of the session's
        tenant for repeated use; each use reads back `read`, a buffer the step
        names, of at most `READ_BYTES`."""
        self._ensure_active()
        if step.tenant != self.session_tenant:
            raise RefusedError("the step's tenant is not the session's tenant")
        prepared = self._catalog.prepare(step)
        if isinstance(prepared, StepRefused):
            raise RefusedError(prepared.reason)
        found = self._catalog.buffer(read)
        if read not in prepared.footprint or found is None or found.nbytes > READ_BYTES:
            self._catalog.unprepare(prepared)
            raise RefusedError(
                f"a prepared step reads back a buffer it names, of at most {READ_BYTES} bytes"
            )
        if status is not None:
            report = self._catalog.buffer(status)
            if (
                status == read
                or status not in prepared.footprint
                or report is None
                or report.nbytes != 4
                or not self._catalog.readable(step.tenant, status)
            ):
                self._catalog.unprepare(prepared)
                raise RefusedError(
                    "a prepared check status must be a separate readable four-byte buffer"
                )
        launch = step.actions[-1]
        if not isinstance(launch, Launch):
            raise AssertionError("the catalog prepares a step that ends with its launch")
        parameters = _Parameters(len(self._modules[launch.operator]._types))
        try:
            with self._device_work():
                call(driver.cuCtxSetCurrent(self._context))
                stream = self._streams.get(step.tenant)
                if stream is None:
                    stream = self._streams[step.tenant] = self._blocking_stream()
                (event,) = call(driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DISABLE_TIMING))
                try:
                    (host,) = call(
                        driver.cuMemAllocHost(found.nbytes + (4 if status is not None else 0))
                    )
                except BaseException:
                    call(driver.cuEventDestroy(event))
                    raise
        except BaseException:
            self._catalog.unprepare(prepared)
            raise
        run = Run(prepared, read, found.nbytes, stream, event, address(host), parameters, status)
        self._runs.add(run)
        return run

    def _blocking_stream(self) -> object:
        """A new stream of a tenant's runs: a blocking stream, whose flags are 0, so
        that the legacy stream's work is ordered with it on the device."""
        (stream,) = call(driver.cuStreamCreate(0))
        try:
            flags = integer(driver.cuStreamGetFlags(stream))
            if flags != 0:
                raise RuntimeError(
                    f"a tenant's stream has flags {flags}, not 0: it is not blocking"
                )
        except BaseException:
            call(driver.cuStreamDestroy(stream))
            raise
        return stream

    def submit(self, run: Run, scalars: tuple[int, ...]) -> None:
        """Have the catalog decide a use of `run` with the launch's `scalars`, and
        submit its effects and the copy of its read buffer on the tenant's stream.
        Like every operation that commits to the catalog, it first waits for the
        uses in progress; only the hot uses of the service's compiled loop, which
        commit nothing, overlap.

        The driver calls of a use depend only on the catalog's memo of the step
        and on the scalars: they are built once per memo (`_compile`), and each
        use writes its scalars into the parameter block and makes them."""
        self._require_run(run)
        if run.running:
            raise RuntimeError("the run's last use has not finished")
        if run.prepared.tenant != self.session_tenant:
            raise RefusedError("the run's tenant is not the session's tenant")
        self.quiesce()
        use = self._catalog.begin_prepared(run.prepared, scalars)
        if isinstance(use, StepRefused):
            raise RefusedError(use.reason)
        self._keep_raw_buffers(use)
        if use.memo is not run.memo:
            try:
                self._compile(run, use.memo, use.effects)
            except BaseException:
                use.roll_back()
                raise
        if run.module is None:
            raise AssertionError("a compiled run has its module")
        slots, masks = run.parameters.slots, run.module._masks
        for index, value in zip(use.parameters, scalars, strict=True):
            slots[index] = value & masks[index]
        try:
            for submission in run.calls:
                call(submission())
        except BaseException as error:
            self._failure = error
            raise
        use.commit()
        run.running = True

    def _compile(self, run: Run, memo: object, effects: tuple[Effect, ...]) -> None:
        """The driver calls of a use of `run` whose catalog memo is `memo`, with the
        kernel's parameters other than its scalars written into the block."""
        tenant = run.prepared.tenant
        found = self._catalog.buffer(run.read)
        if (
            found is None
            or found.nbytes != run.nbytes
            or not self._catalog.readable(tenant, run.read)
        ):
            raise RefusedError("the tenant may not read back the run's buffer")
        stream, pool = run.stream, self._pool
        calls: list[Callable[[], tuple[object, ...]]] = [
            partial(driver.cuCtxSetCurrent, self._context)
        ]
        fills: list[_Fill] | None = []
        kernel: Kernel | None = None
        for effect in effects:
            match effect:
                case Memset(offset=offset, nbytes=nbytes, element=element, value=value) if (
                    element.width in _MEMSET_ASYNC
                ):
                    width = element.width
                    fill = _Fill(width, pool + offset, value % 2 ** (8 * width), nbytes // width)
                    calls.append(partial(_MEMSET_ASYNC[width], *_memset_arguments(fill, stream)))
                    if fills is not None:
                        fills.append(fill)
                case Memset(offset=offset, nbytes=nbytes, value=value) if _uniform(value):
                    fill = _Fill(1, pool + offset, value & 0xFF, nbytes)
                    calls.append(partial(driver.cuMemsetD8Async, *_memset_arguments(fill, stream)))
                    if fills is not None:
                        fills.append(fill)
                case Memset():
                    calls.append(partial(self._launch, self._fill8_launch(effect), stream))
                    fills = None
                case Kernel(checks=()):
                    kernel = effect
                    module = self._modules[effect.operator]
                    slots = run.parameters.slots
                    for index, (argument, mask) in enumerate(
                        zip(effect.arguments, module._masks, strict=True)
                    ):
                        match argument:
                            case At(offset=offset):
                                slots[index] = pool + offset
                            case Value(value=number):
                                slots[index] = number & mask
                            case _ as never:
                                assert_never(never)
                    calls.append(
                        partial(
                            driver.cuLaunchKernel,
                            module.function,
                            *effect.grid,
                            *module.binding.block,
                            effect.shared,
                            stream,
                            run.parameters.address,
                            0,
                        )
                    )
                    run.module = module
                case Kernel() | Clear() | Upload():
                    raise AssertionError("a use has fills and one launch without checks")
                case _ as never:
                    assert_never(never)
        source = pool + found.offset
        calls.append(partial(driver.cuMemcpyDtoHAsync, run.host, source, run.nbytes, stream))
        if run.status is not None:
            status = self._catalog.buffer(run.status)
            if (
                status is None
                or status.nbytes != 4
                or not self._catalog.readable(tenant, run.status)
            ):
                raise RefusedError("the tenant may not read the run's check status")
            run.status_source = pool + status.offset
            calls.append(
                partial(
                    driver.cuMemcpyDtoHAsync, run.host + run.nbytes, run.status_source, 4, stream
                )
            )
        calls.append(partial(driver.cuEventRecord, run.event, stream))
        run.calls = tuple(calls)
        run.fills = None if fills is None else tuple(fills)
        run.kernel = kernel
        run.source = source
        run.memo = memo

    def hot(self, run: Run) -> Hot | None:
        """The hot form of the next use of `run`: when the catalog's memo of its step
        holds at the current state, a use changes nothing in the catalog, the
        driver calls were built for that memo, and its fills are driver memsets.
        Otherwise None, and the use goes through `submit`."""
        self._require_run(run)
        memo = self._catalog.hot(run.prepared)
        module, kernel, fills = run.module, run.kernel, run.fills
        if (
            memo is None
            or memo is not run.memo
            or run.running
            or module is None
            or kernel is None
            or fills is None
        ):
            return None
        bounded = [p for _, p in module.binding.parameters if isinstance(p, Bounded)]
        return Hot(
            tuple((p.lower, p.upper) for p in bounded),
            run.prepared.parameters,
            module._masks,
            run.parameters.slots,
            run.parameters.address,
            fills,
            address(module.function),
            kernel.grid,
            module.binding.block,
            kernel.shared,
            address(run.stream),
            address(run.event),
            run.host,
            run.source,
            run.nbytes,
            run.host + run.nbytes if run.status is not None else 0,
            run.status_source,
        )

    def finished(self, run: Run) -> bool:
        """Whether the use of `run` in progress has ended."""
        self._require_run(run)
        if not run.running:
            raise RuntimeError("the run has no use in progress")
        (status,) = driver.cuEventQuery(run.event)
        if status == _NOT_READY:
            return False
        if status != _SUCCESS:
            self._failure = DriverError(f"the driver returned {status}")
            raise self._failure
        return True

    def result(self, run: Run) -> bytes:
        """The bytes that the use of `run` in progress read back, once it ends."""
        self._require_run(run)
        if not run.running:
            raise RuntimeError("the run has no use in progress")
        try:
            call(driver.cuEventSynchronize(run.event))
        except BaseException as error:
            self._failure = error
            raise
        run.running = False
        if run.status is not None and any(ctypes.string_at(run.host + run.nbytes, 4)):
            if run.module is None:
                raise AssertionError("a submitted run has its module")
            raise CheckFailedError(run.module.binding.kernel)
        return ctypes.string_at(run.host, run.nbytes)

    def quiesce(self) -> None:
        """Wait for every use in progress to end; `result` still collects each."""
        with self._device_work():
            for run in self._runs:
                if run.running:
                    call(driver.cuEventSynchronize(run.event))

    def unprepare(self, run: Run) -> None:
        """Release a run whose use, if any, the caller collected."""
        self._require_run(run)
        if run.running:
            raise RuntimeError("the run's last use has not been collected")
        with self._device_work():
            call(driver.cuEventDestroy(run.event))
            call(driver.cuMemFreeHost(run.host))
        self._catalog.unprepare(run.prepared)
        self._runs.discard(run)

    def _require_run(self, run: Run) -> None:
        self._ensure_active()
        if run not in self._runs:
            raise ValueError("the run is not live in this executor")

    def _at(self, offset: int) -> object:
        return driver.CUdeviceptr(self._pool + offset)

    # ------------------------------------------------------------------
    # Launches, as eager calls and as graph nodes.

    def _kernel_launch(self, kernel: Kernel) -> _Launch:
        """The launch of a checked kernel: each parameter is the checked record's
        value at the size the driver gives for the loaded function, which
        `register` checked: 8 bytes for each buffer's base."""
        module = self._modules[kernel.operator]
        values: list[int] = []
        for argument, mask in zip(kernel.arguments, module._masks, strict=True):
            match argument:
                case At(offset=offset):
                    values.append(self._pool + offset)
                case Value(value=value):
                    values.append(value & mask)
                case _ as never:
                    assert_never(never)
        return _Launch(
            module.function,
            kernel.grid,
            module.binding.block,
            kernel.shared,
            tuple(values),
            module._types,
        )

    def _check_launch(self, check: Check, flag: int) -> _Launch:
        values = trusted.check_values(check, self._pool + check.offset, flag)
        count = check.nbytes // check.element.width
        grid = (trusted.blocks(count), 1, 1)
        return _Launch(
            self._trusted.check, grid, (trusted.THREADS, 1, 1), 0, values, trusted.CHECK_TYPES
        )

    def _fill8_launch(self, memset: Memset) -> _Launch:
        count = memset.nbytes // 8
        values = (self._pool + memset.offset, count, memset.value % 2**64)
        grid = (trusted.blocks(count), 1, 1)
        return _Launch(
            self._trusted.fill8, grid, (trusted.THREADS, 1, 1), 0, values, trusted.FILL8_TYPES
        )

    def _launch(self, launch: _Launch, stream: object) -> tuple[object, ...]:
        """The driver's result of launching `launch` on `stream`."""
        return driver.cuLaunchKernel(
            launch.function,
            *launch.grid,
            *launch.block,
            launch.shared,
            stream,
            (launch.values, launch.types),
            0,
        )

    def _run(self, launch: _Launch, stream: object) -> None:
        call(
            driver.cuLaunchKernel(
                launch.function,
                *launch.grid,
                *launch.block,
                launch.shared,
                stream,
                (launch.values, launch.types),
                0,
            )
        )

    def _node_params(self, launch: _Launch) -> object:
        params = driver.CUDA_KERNEL_NODE_PARAMS()
        params.func = launch.function
        params.gridDimX, params.gridDimY, params.gridDimZ = launch.grid
        params.blockDimX, params.blockDimY, params.blockDimZ = launch.block
        params.sharedMemBytes = launch.shared
        params.kernelParams = (launch.values, launch.types)
        return params

    def _add_memset(self, graph: object, last: Sequence[object], params: object) -> object:
        (node,) = call(
            driver.cuGraphAddMemsetNode(graph, list(last) or None, len(last), params, self._context)
        )
        return node

    def _add_kernel(self, graph: object, last: Sequence[object], launch: _Launch) -> object:
        (node,) = call(
            driver.cuGraphAddKernelNode(
                graph, list(last) or None, len(last), self._node_params(launch)
            )
        )
        return node

    def _set_kernel(self, instance: object, node: object, launch: _Launch) -> None:
        call(driver.cuGraphExecKernelNodeSetParams(instance, node, self._node_params(launch)))

    def _memset_params(self, effect: Clear | Memset) -> object:
        match effect:
            case Clear(offset=offset, nbytes=nbytes):
                return _memset_node_params(self._at(offset), 1, nbytes, 0)
            case Memset(offset=offset, nbytes=nbytes, element=element, value=value):
                width = element.width
                return _memset_node_params(
                    self._at(offset), width, nbytes // width, value % 2 ** (8 * width)
                )
            case _ as never:
                assert_never(never)

    def _add_checked(
        self,
        graph: object,
        last: Sequence[object],
        kernel: Kernel,
        checks: Sequence[Check],
        flag: int,
    ) -> _CheckedNodes:
        """Reset the flag, run each check, set the condition from the flag, and run
        the kernel in the body of a conditional IF node on the condition."""
        node = self._add_memset(graph, last, _memset_node_params(driver.CUdeviceptr(flag), 4, 1, 1))
        reset_node = node
        check_nodes: list[object] = []
        for check in checks:
            node = self._add_kernel(graph, [node], self._check_launch(check, flag))
            check_nodes.append(node)
        (handle,) = call(driver.cuGraphConditionalHandleCreate(graph, self._context, 0, 0))
        decide = _Launch(
            self._trusted.decide,
            (1, 1, 1),
            (1, 1, 1),
            0,
            (flag, address(handle)),
            trusted.DECIDE_TYPES,
        )
        decide_node = self._add_kernel(graph, [node], decide)
        params = driver.CUgraphNodeParams()
        params.type = driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.handle = handle
        params.conditional.type = driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
        params.conditional.size = 1
        params.conditional.ctx = self._context
        (conditional,) = call(driver.cuGraphAddNode(graph, [decide_node], None, 1, params))
        body = self._add_kernel(params.conditional.phGraph_out[0], [], self._kernel_launch(kernel))
        return _CheckedNodes(reset_node, tuple(check_nodes), decide_node, conditional, body, flag)

    def _checks_hold(self, checks: Sequence[Check]) -> bool:
        """Run each check eagerly; whether every element of each lies in its range."""
        if not checks:
            return True
        call(driver.cuMemsetD32(driver.CUdeviceptr(self._flag), 1, 1))
        for check in checks:
            self._run(self._check_launch(check, self._flag), 0)
        call(driver.cuCtxSynchronize())
        word = bytearray(4)
        call(driver.cuMemcpyDtoH(word, driver.CUdeviceptr(self._flag), 4))
        return int.from_bytes(word, "little") == 1

    def _memset(self, memset: Memset) -> None:
        width, count = memset.element.width, memset.nbytes // memset.element.width
        if width in _MEMSET:
            pattern = memset.value % 2 ** (8 * width)
            call(_MEMSET[width](self._at(memset.offset), pattern, count))
        else:
            self._run(self._fill8_launch(memset), 0)

    def close(self) -> None:
        """Destroy every graph, free the pool, unload every module, and release the
        primary context."""
        self._assert_thread()
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []

        def release(function: Callable[..., tuple[object, ...]], *args: object) -> None:
            try:
                call(function(*args))
            except Exception as error:
                errors.append(error)

        release(driver.cuCtxSetCurrent, self._context)
        release(driver.cuCtxSynchronize)
        for run in self._runs:
            release(driver.cuEventDestroy, run.event)
            release(driver.cuMemFreeHost, run.host)
        self._runs.clear()
        for stream in self._streams.values():
            release(driver.cuStreamDestroy, stream)
        self._streams.clear()
        for graph in self._graphs:
            release(driver.cuGraphExecDestroy, graph.instance)
            release(driver.cuGraphDestroy, graph.graph)
            if graph.flags:
                release(driver.cuMemFree, driver.CUdeviceptr(graph.flags))
        self._graphs.clear()
        self._unread.clear()
        self._raw.clear()
        for module in self._modules.values():
            release(driver.cuModuleUnload, module.handle)
        self._modules.clear()
        release(driver.cuModuleUnload, self._trusted.handle)
        release(driver.cuMemFree, driver.CUdeviceptr(self._flag))
        release(driver.cuMemFree, driver.CUdeviceptr(self._pool))
        release(driver.cuDevicePrimaryCtxRelease, self._device)
        if errors:
            raise ExceptionGroup("executor cleanup failed", errors)

    def __enter__(self) -> Self:
        self._ensure_active()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if error is None:
                raise
            error.add_note(f"Executor cleanup also failed: {cleanup!r}")
