"""The owner thread: the one thread that holds the executor, its RMM resource, and the cuDF hook.

Socket threads parse requests and hand them to the owner as work; the owner
performs each against the catalog and answers it. Every outcome has one
kind, decided by the type of the exception that ends the work, and nothing
else:

- `RequestError`: the request is malformed; nothing changed ("input").
- `RefusedError`: the catalog, the gate, the driver's load, or a quota
  refuses; nothing changed ("refused"). A full pool is a refusal too.
- `CheckFailedError`: a check that the guard pass added to the tenant's
  kernel failed when it ran ("check_failed"): an access did not run, or a
  loop reached its iteration limit and ended, and the catalog committed the launch's step, whose other effects stand.
- anything else: the service's state may be inconsistent ("fatal"); the owner
  answers every queued request as failed and stops, and the server stops
  accepting connections, for a supervisor to restart the process.

A registration whose text the gate refused for a reason that the guard pass
addresses is loaded as the pass's text and binding (`server`), and its record
keeps the text and binding as submitted beside the pass's result (`audit`).
A launch of such a module takes a status buffer that the tenant's resource
allocates in the tenant's memory, filled with zeros in the launch's step,
and read after the step (`opgate.hosts.execute`); a nonzero status answers
"check_failed" to the launching tenant. A prepared step retains its own status
buffer, clears it before each launch, and reads it on the same stream before
answering through either mailbox execution path.

A tenant's first open session reserves its principal's reservation, or
`Limits.tenant_bytes`, of the pool for the tenant (`Executor.reserve`), and every buffer and arena chunk of the
tenant lies in that reservation; a session whose reservation does not fit
is refused. The reservation ends when the tenant's last session closes, if
the tenant then holds no live buffer, and stays for its next session
otherwise.

The owner's loop is the compiled loop of `fast`, over the sessions'
mailboxes (`dataplane`). A mailbox request is a use of a step that the
session prepared (`prepare`). The compiled loop serves the uses whose steps
the executor describes as hot; it returns, with no use in progress, when work
is queued or when a request needs this module: a step's first use after a
change, a refusal, or a malformed record, which `_take` decides through the
executor and the catalog. Queued work therefore never meets a running use.

A query node runs inside a query boundary: afterwards cuDF's caches are
released, and if the set of live allocations differs from before, the
service fails with `InvariantError`, whatever the node's own outcome. The
engine allocates inside the tenant's arena, whose chunks stay the tenant's
buffers until its session closes or it asks for their release; chunks and
buffers are charged to the resource's tenant quotas.
"""

import base64
import binascii
import gc
import traceback
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, InvalidStateError
from contextlib import contextmanager
from dataclasses import dataclass, replace
from queue import Empty, Queue
from typing import TYPE_CHECKING, assert_never

from opgate.binding import Binding, Bounded, ElementType
from opgate.catalog import BufferId, Fill, Grant, Launch, Step, TenantId, Write
from opgate.executor import Executor, Module, RefusedError, Run
from opgate.executor.driver import DriverError
from opgate.frontends import Target
from opgate.frontends.guard import STATUS_BYTES, Guarded, reports
from opgate.gate import Accepted, Refused
from opgate.gate import Decision as GateDecision
from opgate.hosts import CheckFailedError, Registered, execute
from opgate.hosts.service.audit import Decision, Registration, Telemetry, parse_element
from opgate.hosts.service.dataplane import Mailbox, Record, Status
from opgate.hosts.service.fast import SEQUENCE, Exit, Tables, serve
from opgate.hosts.service.plans import Plan, QueryNode
from opgate.hosts.service.protocol import Fields, RequestError, encode
from opgate.hosts.service.sessions import (
    BufferEntry,
    Handles,
    Limits,
    OperatorEntry,
    Principal,
    SessionId,
)

if TYPE_CHECKING:
    from opgate.hosts.rapids import CatalogResource

__all__ = [
    "Checked",
    "Begin",
    "Close",
    "InvariantError",
    "Job",
    "Load",
    "Open",
    "Owner",
    "Request",
    "Response",
    "fatal",
]

MAX_PREPARED = 64  # the steps one session may prepare

Response = dict[str, object]


class InvariantError(RuntimeError):
    """The service's state broke an invariant; the service stops."""


@dataclass(frozen=True)
class Request:
    """A request of a session, other than a registration's check."""

    principal: Principal
    session: SessionId
    fields: Fields


@dataclass(frozen=True)
class Checked:
    """The gate's decision on a registration: on the text as submitted, or, when
    the guard pass ran, on the pass's result, which `guarded` holds."""

    decision: GateDecision
    guarded: Guarded | None


@dataclass(frozen=True)
class Load:
    """A registration and its checked decision, or the error that ended its check."""

    principal: Principal
    session: SessionId
    plan: Plan
    text: str
    outcome: Checked | str


@dataclass(frozen=True)
class CountRegistration:
    """A registration to count against the tenant's quotas before it is checked;
    a `Load` of the same principal follows each counted one."""

    principal: Principal


@dataclass(frozen=True)
class Begin:
    """The start of a session: the tenant's reservation, which its first session makes."""

    principal: Principal
    session: SessionId


@dataclass(frozen=True)
class Close:
    """The end of a session: its buffers are freed and its handles dropped."""

    principal: Principal
    session: SessionId


@dataclass(frozen=True)
class Open:
    """A session's mailbox, which the owner serves from now on."""

    principal: Principal
    session: SessionId
    mailbox: Mailbox


Work = Request | Load | CountRegistration | Begin | Close | Open


@dataclass(frozen=True)
class _Prepared:
    """A step a session prepared: the executor's run, its module for the audit's
    launch counts, and whether each scalar is signed."""

    run: Run
    module: Module
    signed: tuple[bool, ...]


@dataclass(frozen=True)
class Job:
    work: Work
    answer: Future[Response]


def fatal(reason: str) -> Response:
    return {"ok": False, "error": {"kind": "fatal", "reason": reason}}


def _error(kind: str, reason: str) -> Response:
    return {"ok": False, "error": {"kind": kind, "reason": reason}}


def _read_limit(nbytes: int, message_bytes: int) -> bool:
    """Whether a read of `nbytes` fits in a response: base64 and the envelope."""
    return 4 * -(-nbytes // 3) + 256 <= message_bytes


class Owner:
    """The owner thread's state and loop."""

    def __init__(
        self,
        *,
        device: int,
        pool_bytes: int,
        target: Target,
        principals: tuple[Principal, ...],
        nodes: Mapping[str, QueryNode],
        limits: Limits,
        queue: "Queue[Job | None]",
        wakeup: int,
        started: Future[None],
        failure: Future[str],
    ) -> None:
        self._device = device
        self._pool_bytes = pool_bytes
        self._target = target
        self._principals = {p.name: p for p in principals}
        self._nodes = nodes
        self._limits = limits
        self._queue = queue
        self._wakeup = wakeup  # an eventfd that the server writes after it queues work
        self._started = started
        self._failure = failure
        self._handles = Handles(limits)
        self._audit: list[Registration] = []
        self._by_operator: dict[int, int] = {}  # an operator id to its first registration
        self._telemetry = Telemetry()
        self._principal: Principal | None = None  # of the work in progress
        self._session: SessionId | None = None
        self._node: str | None = None  # the query node in progress
        self._executor: Executor | None = None
        self._resource: CatalogResource | None = None
        self._mailboxes: dict[SessionId, tuple[Principal, Mailbox]] = {}
        self._sessions: dict[Mailbox, SessionId] = {}  # each mailbox's session
        self._prepared: dict[SessionId, list[_Prepared]] = {}
        self._open: dict[TenantId, int] = {}  # the number of each tenant's open sessions
        # each mailbox's use in progress: its session, its step, and its request
        self._running: dict[Mailbox, tuple[SessionId, _Prepared, int]] = {}

    # ------------------------------------------------------------ the context of a node

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            raise InvariantError("the executor is used outside the owner's loop")
        return self._executor

    @property
    def resource(self) -> "CatalogResource":
        if self._resource is None:
            raise InvariantError("the RMM resource is used outside the owner's loop")
        return self._resource

    @property
    def tenant(self) -> TenantId:
        return self._current()[0].tenant

    def _current(self) -> tuple[Principal, SessionId]:
        if self._principal is None or self._session is None:
            raise InvariantError("a session's operation outside its request")
        return self._principal, self._session

    def buffer(self, request: Fields, name: str) -> BufferId:
        principal, session = self._current()
        return self._handles.readable(principal.tenant, session, request.text(name)).buffer

    def operator(self, request: Fields, name: str, kind: str) -> Module:
        principal, session = self._current()
        entry = self._handles.operator(principal.tenant, session, request.text(name))
        if entry.plan.kind != kind:
            raise RefusedError(
                f"the operator handle is registered as {entry.plan.kind}, not {kind}"
            )
        return entry.module

    def launched(self, module: Module) -> None:
        self._launches(module, 1)

    def _launches(self, module: Module, count: int) -> None:
        index = self._by_operator.get(int(module.operator))
        if index is None:
            raise InvariantError("a launch of a module with no registration record")
        self._audit[index].launch_count += count
        self._telemetry.launch_submissions += count

    # ------------------------------------------------------------ the cuDF hook's records

    def _cudf_registered(self, registered: Registered) -> None:
        principal, _ = self._current()
        if self._node is None:
            raise InvariantError("cuDF registered a module outside a query node")
        module = registered.module
        if int(module.operator) in self._by_operator:
            return
        self._record(
            principal,
            self._node,
            registered.text,
            registered.binding,
            registered.guarded,
            "accepted",
            None,
            1,
            module,
        )
        self._telemetry.accepted_registrations += 1
        self._telemetry.module_loads += 1

    def _cudf_launched(self, module: Module) -> None:
        self._telemetry.launch_requests += 1
        self.launched(module)

    def _record(
        self,
        principal: Principal,
        kind: str,
        text: str,
        binding: Binding,
        guarded: Guarded | None,
        decision: Decision,
        reason: str | None,
        loads: int,
        module: Module | None,
    ) -> int:
        """Append a registration record, and index it by its module's operator."""
        index = len(self._audit)
        loaded = None if module is None else module.text
        self._audit.append(
            Registration(
                index,
                principal.name,
                kind,
                text,
                binding,
                guarded,
                loaded,
                decision,
                reason,
                loads,
                0,
            )
        )
        if module is not None:
            self._by_operator.setdefault(int(module.operator), index)
        return index

    # ------------------------------------------------------------ the operations

    def _live(self) -> tuple[frozenset[int], int]:
        """The start of every buffer of its own, and the number of allocations in
        the arenas."""
        return self.resource.live()

    @contextmanager
    def _boundary(self, op: str) -> Iterator[None]:
        """A query node's boundary: cuDF's caches are cleared afterwards, and the
        live buffers that the resource does not keep must be those before; a
        node may change their status. The regions that the node's engine freed
        stay the tenant's until its session closes."""
        from opgate.hosts.cudf import clear_cudf_caches

        before = self._live()
        self._node = op
        try:
            yield
        finally:
            self._node = None
            clear_cudf_caches()
            after = self._live()
            if after != before:
                # a query's engine objects can hold allocations in reference cycles
                gc.collect()
                after = self._live()
            if after != before:
                raise InvariantError(
                    f"the query node {op} left buffers at {sorted(after[0] - before[0])} "
                    f"and {after[1] - before[1]} more arena allocations live, and freed "
                    f"buffers at {sorted(before[0] - after[0])}"
                )

    def _allocate(self, request: Fields) -> Response:
        principal, session = self._current()
        request.exact("op", "nbytes")
        nbytes = request.integer("nbytes", minimum=1, maximum=self._pool_bytes)
        try:
            buffer = self.resource.allocate(nbytes)
        except MemoryError as error:
            raise RefusedError(f"allocation of {nbytes} bytes refused: {error}") from error
        handle = self._handles.add_buffer(
            BufferEntry(buffer, nbytes, principal.tenant, session, set())
        )
        return {"buffer": handle, "nbytes": nbytes}

    def _write(self, request: Fields) -> Response:
        request.exact("op", "buffer", "data", "element")
        try:
            data = base64.b64decode(request.text("data"), validate=True)
        except (binascii.Error, ValueError) as error:
            raise RequestError("write.data is not canonical base64") from error
        element = (
            None if request.raw("element") is None else parse_element(request.fields("element"))
        )
        buffer = self.buffer(request, "buffer")
        self.executor.execute(Step(self.tenant, (Write(buffer, data, element),)))
        return {"written": len(data)}

    def _read(self, request: Fields) -> Response:
        principal, session = self._current()
        request.exact("op", "buffer")
        entry = self._handles.readable(principal.tenant, session, request.text("buffer"))
        if not _read_limit(entry.nbytes, self._limits.message_bytes):
            raise RefusedError(f"a buffer of {entry.nbytes} bytes does not fit in a response")
        data = self.executor.read(principal.tenant, entry.buffer)
        return {"data": base64.b64encode(data).decode("ascii"), "nbytes": len(data)}

    def _free(self, request: Fields) -> Response:
        principal, session = self._current()
        request.exact("op", "buffer")
        handle, entry = self._handles.owned(principal.tenant, session, request.text("buffer"))
        self.resource.free(entry.buffer)
        self._handles.remove_buffer(handle)
        return {"freed": True}

    def _grant(self, request: Fields) -> Response:
        principal, session = self._current()
        request.exact("op", "buffer", "principal")
        target = self._principals.get(request.text("principal"))
        if target is None:
            raise RequestError("grant_read.principal names no principal")
        _, entry = self._handles.owned(principal.tenant, session, request.text("buffer"))
        self.executor.execute(Step(principal.tenant, (Grant(entry.buffer, target.tenant),)))
        entry.grants.add(target.tenant)
        return {"granted_to": target.name}

    def _release_kept(self, request: Fields) -> Response:
        """Return this tenant's arena to the pool without ending its session."""
        request.exact("op")
        return {"released_chunks": self.resource.release(self.tenant)}

    def _run_operator(self, request: Fields) -> Response:
        principal, session = self._current()
        entry = self._handles.operator(principal.tenant, session, request.text("operator"))
        launch = entry.plan.launch(request)
        buffers = tuple(
            self._handles.readable(principal.tenant, session, handle).buffer
            for handle in launch.buffers
        )
        self._telemetry.launch_requests += 1
        step = Step(
            principal.tenant,
            (
                Launch(
                    entry.module.operator,
                    launch.grid,
                    buffers,
                    launch.sizes,
                    launch.scalars,
                    launch.guarded,
                ),
            ),
        )
        try:
            ran = execute(self.executor, self.resource, step)
        except CheckFailedError:
            self.launched(entry.module)  # the kernel ran and stored its status
            raise
        if len(ran) != 1 or type(ran[0]) is not bool:
            raise InvariantError("one operator launch must return exactly one boolean result")
        self.executor.synchronize()
        if ran[0]:
            self.launched(entry.module)
        return {"ran": list(ran)}

    def _prepare(self, request: Fields) -> Response:
        """Check a step of fills and one launch that the session's mailbox runs."""
        principal, session = self._current()
        request.exact("op", "operator", "grid", "buffers", "sizes", "scalars", "fills", "read")
        entry = self._handles.operator(principal.tenant, session, request.text("operator"))
        grid = _integers(request, "grid")
        if len(grid) != 3:
            raise RequestError("prepare.grid must hold three integers")
        buffers = tuple(
            self.buffer(Fields({"buffer": h}, "prepare.buffers"), "buffer")
            for h in request.values("buffers")
        )
        fills: list[Fill] = []
        for index, value in enumerate(request.values("fills")):
            fill = Fields(value, f"prepare.fills[{index}]")
            fill.exact("buffer", "element", "value")
            element = parse_element(fill.fields("element"))
            fills.append(Fill(self.buffer(fill, "buffer"), element, fill.integer("value")))
        steps = self._prepared.setdefault(session, [])
        if len(steps) >= MAX_PREPARED:
            raise RefusedError(f"a session prepares at most {MAX_PREPARED} steps")
        launch = Launch(
            entry.module.operator,
            (grid[0], grid[1], grid[2]),
            buffers,
            _integers(request, "sizes"),
            _integers(request, "scalars"),
            False,
        )
        read = self.buffer(request, "read")
        status = None
        try:
            if reports(entry.module.binding):
                status = self.resource.allocate(STATUS_BYTES)
                fills.append(Fill(status, ElementType(STATUS_BYTES, False), 0))
                launch = replace(launch, buffers=(*launch.buffers, status))
            step = Step(principal.tenant, (*fills, launch))
            run = self.executor.prepare(step, read, status=status)
        except BaseException:
            if status is not None:
                self.resource.free(status)
            raise
        bounded = self.executor.catalog.binding(entry.module.operator)
        if bounded is None:
            raise InvariantError("a prepared step of an operator the catalog lacks")
        signed = tuple(p.lower < 0 for _, p in bounded.parameters if isinstance(p, Bounded))
        steps.append(_Prepared(run, entry.module, signed))
        return {"prepared": len(steps) - 1}

    def _count_registration(self, principal: Principal) -> Response:
        self._handles.count_registration(principal.tenant)
        self._telemetry.register_requests += 1
        return {"counted": True}

    def _load(self, work: Load) -> Response:
        """Load the text that the gate accepted: the pass's text and binding when the
        guard pass ran, or else the text as submitted. The record keeps the text
        and binding as submitted and the pass's result."""
        match work.outcome:
            case str() as error:
                self._telemetry.internal_errors += 1
                return _error("internal", f"the registration's check failed: {error}")
            case Checked(decision=decision, guarded=guarded):
                pass
            case _ as never:
                assert_never(never)
        match decision:
            case Refused(reason=reason):
                return self._refused(work, guarded, reason)
            case Accepted():
                pass
            case _ as never:
                assert_never(never)
        text, binding = (
            (work.text, work.plan.binding) if guarded is None else (guarded.text, guarded.binding)
        )
        before = len(self.executor.catalog.operators)
        try:
            module = self.executor.register(text, binding)
        except RefusedError as error:
            return self._refused(work, guarded, error.reason)
        loaded = len(self.executor.catalog.operators) > before
        index = self._record(
            work.principal,
            work.plan.kind,
            work.text,
            work.plan.binding,
            guarded,
            "accepted",
            None,
            int(loaded),
            module,
        )
        self._telemetry.accepted_registrations += 1
        self._telemetry.module_loads += int(loaded)
        handle = self._handles.add_operator(
            OperatorEntry(module, work.plan, work.principal.tenant, work.session, index)
        )
        return {
            "ok": True,
            "operator": handle,
            "loaded": loaded,
            "kernel": module.binding.kernel,
            "checks": [] if guarded is None else list(guarded.guarded),
            "registration": index,
        }

    def _refused(self, work: Load, guarded: Guarded | None, reason: str) -> Response:
        """Record and answer a refused registration."""
        index = self._record(
            work.principal,
            work.plan.kind,
            work.text,
            work.plan.binding,
            guarded,
            "refused",
            reason,
            0,
            None,
        )
        self._telemetry.refused_requests += 1
        return {**_error("refused", reason), "registration": index}

    def _administer(self, principal: Principal, request: Fields) -> Response:
        if not principal.administrator:
            raise RefusedError(f"{request.text('op')} requires an administrator")
        match request.text("op"):
            case "telemetry":
                request.exact("op")
                return {
                    "telemetry": {
                        **self._telemetry.record(),
                        "live_buffers": len(self.executor.catalog.buffers)
                        - len(self.resource.every_chunk()),
                        "arena_chunks": len(self.resource.every_chunk()),
                        "verified_operators": len(self.executor.catalog.operators),
                        "service_buffer_handles": self._handles.buffer_count,
                        "service_operator_handles": self._handles.operator_count,
                        "registration_records": len(self._audit),
                    }
                }
            case "audit":
                request.exact("op", "start", "count")
                total = len(self._audit)
                start = request.integer("start", minimum=0, maximum=total)
                count = request.integer("count", minimum=1)
                end = min(total, start + count)
                response = {
                    "registrations": [r.record() for r in self._audit[start:end]],
                    "total": total,
                    "next": end if end < total else None,
                }
                try:
                    encode({"ok": True, **response}, self._limits.message_bytes)
                except ValueError as error:
                    raise RefusedError(
                        f"the requested audit page exceeds the message limit: {error}"
                    ) from error
                return response
            case other:
                raise InvariantError(f"an administration operation {other!r}")

    def _dispatch(self, principal: Principal, request: Fields) -> Response:
        op = request.text("op")
        match op:
            case "allocate":
                return self._allocate(request)
            case "write":
                return self._write(request)
            case "read":
                return self._read(request)
            case "free":
                return self._free(request)
            case "release":
                return self._release_kept(request)
            case "grant_read":
                return self._grant(request)
            case "run_operator":
                return self._run_operator(request)
            case "prepare":
                return self._prepare(request)
            case "telemetry" | "audit":
                return self._administer(principal, request)
            case _ if op in self._nodes:
                self._telemetry.node_requests[op] += 1
                with self._boundary(op):
                    try:
                        return self._nodes[op].run(self, request)
                    except (RequestError, RefusedError, CheckFailedError, MemoryError) as error:
                        # An exception's unwound node frames still own their
                        # temporary arrays. Release those locals before the
                        # boundary checks that no live allocation was leaked.
                        traceback.clear_frames(error.__traceback__)
                        if isinstance(error, MemoryError):
                            raise RefusedError(
                                f"allocation for query {op} refused: {error}"
                            ) from error
                        raise
            case _:
                raise RequestError(f"no operation {op!r}")

    def _close(self, principal: Principal, session: SessionId) -> Response:
        opened = self._mailboxes.pop(session, None)
        if opened is not None:
            del self._sessions[opened[1]]
            opened[1].close()
        for prepared in self._prepared.pop(session, []):
            self.executor.unprepare(prepared.run)
            if prepared.run.status is not None:
                self.resource.free(prepared.run.status)
        for entry in self._handles.close(principal.tenant, session):
            self.resource.free(entry.buffer)
        tenant = principal.tenant
        self.resource.release(tenant)
        self._open[tenant] -= 1
        if self._open[tenant] == 0:
            del self._open[tenant]
            if not any(b.owner == tenant for b in self.executor.catalog.buffers):
                self.executor.unreserve()
        self._telemetry.closed_sessions += 1
        return {"closed": True}

    def _begin(self, principal: Principal) -> Response:
        """Reserve the tenant's bytes if it holds no reservation."""
        tenant = principal.tenant
        if self.executor.catalog.reservation(tenant) is None:
            nbytes = (
                self._limits.tenant_bytes
                if principal.reservation is None
                else principal.reservation
            )
            try:
                self.executor.reserve(nbytes)
            except MemoryError as error:
                raise RefusedError(
                    f"no region of the pool holds a reservation of {nbytes} bytes: {error}"
                ) from error
        self._open[tenant] = self._open.get(tenant, 0) + 1
        return {"begun": True}

    def _answer(self, work: Work) -> Response:
        """Perform one piece of work; a fatal error propagates."""
        match work:
            case CountRegistration(principal=principal):
                self._principal = principal
                with self.executor.tenant(principal.tenant):
                    return {"ok": True, **self._count_registration(principal)}
            case Load(principal=principal, session=session):
                self._principal, self._session = principal, session
                # the load adds the operator handle whose place the count held
                self._handles.settle_registration(principal.tenant)
                with self.executor.tenant(principal.tenant):
                    return self._load(work)
            case Begin(principal=principal, session=session):
                self._principal, self._session = principal, session
                with self.executor.tenant(principal.tenant):
                    return {"ok": True, **self._begin(principal)}
            case Close(principal=principal, session=session):
                self._principal, self._session = principal, session
                with self.executor.tenant(principal.tenant):
                    return {"ok": True, **self._close(principal, session)}
            case Open(principal=principal, session=session, mailbox=mailbox):
                if session in self._mailboxes:
                    raise RefusedError("the session has a mailbox")
                self._mailboxes[session] = (principal, mailbox)
                self._sessions[mailbox] = session
                return {"ok": True}
            case Request(principal=principal, session=session, fields=fields):
                self._principal, self._session = principal, session
                self._telemetry.requests += 1
                with self.executor.tenant(principal.tenant):
                    return {"ok": True, **self._dispatch(principal, fields)}
            case _ as never:
                assert_never(never)

    def _classified(self, work: Work) -> Response:
        """`_answer`, with an input error, a refusal, or a failed check as its response."""
        try:
            return self._answer(work)
        except RequestError as error:
            self._telemetry.input_errors += 1
            return _error("input", str(error))
        except RefusedError as error:
            self._telemetry.refused_requests += 1
            return _error("refused", error.reason)
        except CheckFailedError as error:
            self._telemetry.failed_checks += 1
            return _error("check_failed", str(error))
        finally:
            self._principal = self._session = None

    # ------------------------------------------------------------ the loop

    def _publish_failure(self, reason: str) -> None:
        try:
            self._failure.set_result(reason)
        except InvalidStateError:
            # The acceptor can publish a failure while the owner is running.
            pass

    def run(self) -> None:
        try:
            from opgate.hosts.cudf import install_cudf
            from opgate.hosts.rapids import CatalogResource

            with Executor(self._device, self._pool_bytes) as executor:
                self._executor = executor
                self._resource = CatalogResource(
                    executor,
                    tenant_buffers=self._limits.tenant_buffers,
                    chunk_bytes=self._limits.chunk_bytes,
                )
                try:
                    with (
                        self._resource.current(),
                        install_cudf(
                            executor,
                            self._resource,
                            self._target,
                            on_register=self._cudf_registered,
                            on_launch=self._cudf_launched,
                        ),
                    ):
                        self._started.set_result(None)
                        self._serve()
                finally:
                    self._resource.close()
        except BaseException as error:
            if not self._started.done():
                self._started.set_exception(error)
            else:
                self._publish_failure(f"{type(error).__name__}: {error}")
        finally:
            self._executor = self._resource = None

    def _serve(self) -> None:
        try:
            while True:
                mailboxes = [mailbox for _, mailbox in self._mailboxes.values()]
                steps = [self._prepared.get(self._sessions[m], []) for m in mailboxes]
                forms = [[self.executor.hot(p.run) for p in prepared] for prepared in steps]
                tables = Tables.of(self._wakeup, mailboxes, forms)
                left, index = serve(tables, self._wakeup)
                self._account(mailboxes, steps, tables)
                match left:
                    case Exit.QUEUED:
                        if not self._jobs():
                            return
                    case Exit.PYTHON:
                        self._take(mailboxes[index])
                        self._finish()
                    case Exit.IPC:
                        raise InvariantError("a mailbox response notification failed")
                    case Exit.DRIVER:
                        raise DriverError(f"the driver returned {index}")
                    case _ as never:
                        assert_never(never)
        except BaseException as error:
            reason = f"{type(error).__name__}: {error}"
            self._publish_failure(reason)
            for _, mailbox in self._mailboxes.values():
                mailbox.respond(mailbox.sequence, Status.FATAL, reason.encode("utf-8"))
            raise
        finally:
            for _, mailbox in self._mailboxes.values():
                mailbox.close()
            self._mailboxes.clear()
            self._sessions.clear()

    def _account(
        self, mailboxes: list[Mailbox], steps: list[list[_Prepared]], tables: Tables
    ) -> None:
        """Take back what the compiled loop did: each mailbox's last request, and the
        launches it submitted, in the telemetry and the registrations' records."""
        for row, mailbox in enumerate(mailboxes):
            mailbox.sequence = int(tables.mail[row, SEQUENCE]) % 2**64
            for k, prepared in enumerate(steps[row]):
                hot = int(tables.steps[row, k])
                if hot >= 0 and tables.uses[hot]:
                    count = int(tables.uses[hot])
                    self._telemetry.launch_requests += count
                    self._launches(prepared.module, count)
                    self._telemetry.failed_checks += int(tables.failed_checks[hot])

    def _jobs(self) -> bool:
        """Answer every queued job, after the uses in progress; False at the stop."""
        while True:
            try:
                job = self._queue.get_nowait()
            except Empty:
                return True
            self._finish()
            if job is None:
                self._release()
                return False
            try:
                job.answer.set_result(self._classified(job.work))
            except BaseException as error:
                self._telemetry.fatal_errors += 1
                reason = f"{type(error).__name__}: {error}"
                self._publish_failure(reason)
                job.answer.set_result(fatal(reason))
                self._drain()
                raise

    # ------------------------------------------------------------ the mailboxes

    def _take(self, mailbox: Mailbox) -> None:
        """Copy out and submit a mailbox's request, or answer why not."""
        record = mailbox.take()
        session = self._sessions[mailbox]
        principal = self._mailboxes[session][0]
        if record.sequence != (mailbox.sequence + 1) % 2**64:
            mailbox.respond(
                record.sequence, Status.INPUT, b"the request's sequence number is not the next"
            )
            mailbox.sequence = record.sequence  # the client's numbering from here on
            return
        mailbox.sequence = record.sequence
        steps = self._prepared.get(session, [])
        if not 0 <= record.prepared < len(steps):
            mailbox.respond(record.sequence, Status.INPUT, b"the session prepared no such step")
            return
        prepared = steps[record.prepared]
        if record.count != len(prepared.signed):
            mailbox.respond(record.sequence, Status.REFUSED, b"a scalar is missing or extra")
            return
        self._telemetry.launch_requests += 1
        try:
            with self.executor.tenant(principal.tenant):
                self.executor.submit(prepared.run, _scalars(record, prepared.signed))
        except RefusedError as error:
            self._telemetry.refused_requests += 1
            mailbox.respond(record.sequence, Status.REFUSED, error.reason.encode("utf-8"))
            return
        self.launched(prepared.module)
        self._running[mailbox] = (session, prepared, record.sequence)

    def _finish(self) -> None:
        """Wait for every use in progress and answer it."""
        for mailbox, (_, prepared, sequence) in list(self._running.items()):
            self._answer_use(mailbox, prepared, sequence)

    def _answer_use(self, mailbox: Mailbox, prepared: _Prepared, sequence: int) -> None:
        try:
            data = self.executor.result(prepared.run)
        except CheckFailedError:
            self._telemetry.failed_checks += 1
            mailbox.respond(sequence, Status.CHECK_FAILED, b"")
        else:
            mailbox.respond(sequence, Status.ACCEPTED, data)
        del self._running[mailbox]

    def _drain(self) -> None:
        """Answer every queued job as failed."""
        while True:
            try:
                job = self._queue.get_nowait()
            except Empty:
                return
            if job is not None:
                job.answer.set_result(fatal("the service failed"))

    def _release(self) -> None:
        """Free every buffer that a session still holds, at a clean stop."""
        for steps in self._prepared.values():
            for prepared in steps:
                self.executor.unprepare(prepared.run)
                if prepared.run.status is not None:
                    with self.executor.tenant(prepared.run.prepared.tenant):
                        self.resource.free(prepared.run.status)
        self._prepared.clear()
        for _, entry in self._handles.every_buffer():
            with self.executor.tenant(entry.owner):
                self.resource.free(entry.buffer)
        self.resource.release_every()


def _integers(request: Fields, name: str) -> tuple[int, ...]:
    values = request.values(name)
    if not all(type(value) is int for value in values):
        raise RequestError(f"{request.where}.{name} must hold integers")
    return tuple(value for value in values if type(value) is int)


def _scalars(record: Record, signed: tuple[bool, ...]) -> tuple[int, ...]:
    """The record's 8-byte scalars as the catalog's integers: two's complement for a signed one."""
    if not any(signed):
        return record.scalars
    return tuple(
        word - 2**64 if negative and word >= 2**63 else word
        for word, negative in zip(record.scalars, signed, strict=True)
    )
