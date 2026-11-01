"""The Unix-socket server of the service, and its client.

The server accepts at most `Limits.connections` connections. A connection
must authenticate within `Limits.authentication_seconds`; with a principal's
`uid`, the connecting process must also belong to that user (`SO_PEERCRED`).
Each connection is one session, which begins once the owner holds the
tenant's reservation (`owner`); a connection whose reservation does not fit
is answered "refused" and closed. A socket thread parses a request and hands
it to the owner thread, except a registration's check, which a pool of
`Limits.verification_workers` threads performs outside the owner thread, so
that one tenant's check never stalls the others' queries. A register request
names one or more kernels: each is counted against its tenant's quotas on the
owner thread and then checked, all of them at once while threads are free and
in the request's order otherwise, and the owner loads or refuses each one in
that order; the response gives each kernel its own answer. A principal has at
most `Limits.registrations_in_flight` kernels counted and not yet answered,
over its sessions. If the gate refuses the
text for an access that it does not prove in bounds or aligned, for
barrier participation, or for a loop that it does not prove ends
(`opgate.frontends.guard.applies`), the check applies the guard pass
(`opgate.frontends.guard.guard`), and the gate's decision on the pass's
text decides the registration: an access whose check fails at a launch
does not run, a loop that reaches its iteration limit ends, and the launch
answers "check_failed" to its tenant (`owner`). The whole check, every round of the pass included, takes at
most `Limits.verification_seconds`. A text that itself declares the pass's
status parameter (`opgate.frontends.guard.declares_status`) is refused
without a check.

A session's `data_plane` request makes its mailbox (`dataplane`): the
server passes the mailbox's descriptors after the JSON response, and the
owner serves the session's prepared steps through it from then on.

When the owner thread fails, the server stops accepting connections, answers
every open connection's next request as failed, and exposes the failure in
`Service.failure` for a supervisor to restart the process.
"""

import os
import socket
import stat
import struct
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from types import TracebackType
from typing import Self, assert_never

from opgate.binding import Binding
from opgate.executor import RefusedError
from opgate.frontends import Target
from opgate.frontends.guard import STATUS_DECLARED, applies, declares_status, guard
from opgate.gate import Accepted, Refused, VerificationBudget, verify, zeroed
from opgate.hosts.service.dataplane import ClientMailbox, Mailbox
from opgate.hosts.service.owner import (
    Begin,
    Checked,
    Close,
    CountRegistration,
    Job,
    Load,
    Open,
    Owner,
    Request,
    Response,
    fatal,
)
from opgate.hosts.service.plans import Plan, PlanSchema, QueryNode
from opgate.hosts.service.protocol import Fields, RequestError, read_message, write_message
from opgate.hosts.service.sessions import (
    MESSAGE_BYTES,
    RESPONSE_SECONDS,
    Limits,
    Principal,
    SessionId,
    authenticate,
    new_session,
)

__all__ = ["Client", "Service"]

CORE_OPERATIONS = frozenset(
    {
        "authenticate",
        "register",
        "allocate",
        "write",
        "read",
        "free",
        "release",
        "grant_read",
        "run_operator",
        "prepare",
        "data_plane",
        "telemetry",
        "audit",
    }
)
_PEER = struct.Struct("3i")  # SO_PEERCRED: pid, uid, gid


def _peer_uid(connection: socket.socket) -> int:
    _, uid, _ = _PEER.unpack(
        connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEER.size)
    )
    return uid


def _check(text: str, binding: Binding, seconds: float, cancel: Event) -> Checked:
    """The gate's decision on `text` with its zeroing prologue and `binding`, which
    `Executor.register` loads; or, if the gate refuses them for a reason that
    the guard pass addresses, its decision on the pass's result.
    The whole check takes at most `seconds`: each of the gate's checks takes
    the time that remains, and once none remains the decision is a refusal
    for the time, without a check. A text that itself declares the guard pass's
    status parameter is refused without a check, since only the pass adds it."""
    if declares_status(text):
        return Checked(Refused(STATUS_DECLARED, ()), None)
    check = VerificationBudget(seconds, cancel=cancel, checker=verify)
    try:
        registered = zeroed(text, binding)
    except ValueError as error:
        return Checked(
            Refused(f"{binding.kernel}: the zeroing prologue cannot be placed: {error}", ()), None
        )
    decision = check(registered, binding)
    if isinstance(decision, Accepted) or not applies(decision.reason):
        return Checked(decision, None)
    guarded = guard(text, binding, check)
    return Checked(guarded.decision, guarded)


def _fatal(response: Response) -> bool:
    """Whether `response` answers that the service failed or is stopping."""
    error = response.get("error")
    return isinstance(error, dict) and error.get("kind") == "fatal"


def _check_directory(path: Path) -> None:
    """The socket's directory must be the service user's and writable by no one else."""
    directory = path.parent
    info = directory.stat()
    if info.st_uid != os.getuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(f"{directory} must belong to this user and be writable only by it")


class Service:
    """A multi-tenant service whose GPU state belongs to one owner thread."""

    def __init__(
        self,
        path: str | Path,
        principals: Sequence[Principal],
        *,
        device: int,
        pool_bytes: int,
        target: Target,
        schemas: Sequence[PlanSchema],
        nodes: Sequence[QueryNode],
        limits: Limits,
        socket_mode: int = 0o600,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_absolute() or self.path.exists():
            raise ValueError(f"{self.path} must be an absolute path that does not exist")
        _check_directory(self.path)
        people = tuple(principals)
        if not people:
            raise ValueError("the service needs a principal")
        for attribute in ("name", "token", "tenant"):
            if len({getattr(p, attribute) for p in people}) != len(people):
                raise ValueError(f"two principals share a {attribute}")
        self._schemas = {s.kind: s for s in schemas}
        if len(self._schemas) != len(schemas):
            raise ValueError("two plan schemas share a kind")
        by_op = {n.op: n for n in nodes}
        if len(by_op) != len(nodes) or by_op.keys() & CORE_OPERATIONS:
            raise ValueError("a query node's op is taken")
        self._principals = people
        self._limits = limits
        self._queue: Queue[Job | None] = Queue()
        # written after every queued item, so that the owner's loop wakes for it
        self._wakeup = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self._started: Future[None] = Future()
        self.failure: Future[str] = Future()
        self._stopped = False
        self._closing = Event()
        self._owner_stopping = False
        self._lock = Lock()
        self._connections: set[socket.socket] = set()
        self._handlers: set[Thread] = set()
        # each principal's kernels counted and not yet answered, over its sessions
        self._in_flight: dict[str, int] = {}
        self._verifiers = ThreadPoolExecutor(
            limits.verification_workers, thread_name_prefix="opgate-verify"
        )

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            listener.bind(str(self.path))
            bound = True
            os.chmod(self.path, socket_mode)
            listener.listen()
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            if bound:
                self.path.unlink(missing_ok=True)
            self._verifiers.shutdown(wait=True, cancel_futures=True)
            raise
        self._listener = listener
        owner = Owner(
            device=device,
            pool_bytes=pool_bytes,
            target=target,
            principals=people,
            nodes=by_op,
            limits=limits,
            queue=self._queue,
            wakeup=self._wakeup,
            started=self._started,
            failure=self.failure,
        )
        self._owner = Thread(target=owner.run, name="opgate-owner", daemon=True)
        owner_started = False
        try:
            self._owner.start()
            owner_started = True
            self._started.result(timeout=60)
            self._acceptor = Thread(target=self._accept, name="opgate-accept", daemon=True)
            self._acceptor.start()
        except BaseException:
            self._stopped = True
            self._closing.set()
            self._owner_stopping = True
            # A timed-out initializer can still finish later. Its first queue
            # read must stop the owner and release its executor and RMM state.
            self._enqueue(None)
            listener.close()
            self.path.unlink(missing_ok=True)
            if owner_started:
                self._owner.join(timeout=60)
            self._verifiers.shutdown(wait=True, cancel_futures=True)
            raise

    # ------------------------------------------------------------ connections

    def _accept(self) -> None:
        try:
            while not self._stopped and not self.failure.done():
                try:
                    connection, _ = self._listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stopped:
                        return
                    raise
                with self._lock:
                    if self._stopped:
                        connection.close()
                        return
                    self._handlers = {h for h in self._handlers if h.is_alive()}
                    if len(self._connections) >= self._limits.connections:
                        connection.close()
                        continue
                    handler: Thread | None = None
                    try:
                        handler = Thread(
                            target=self._serve,
                            args=(connection,),
                            name="opgate-client",
                            daemon=True,
                        )
                        self._connections.add(connection)
                        self._handlers.add(handler)
                        # close() takes this lock before collecting handlers:
                        # it must never observe one that has not started.
                        handler.start()
                    except BaseException:
                        self._connections.discard(connection)
                        if handler is not None:
                            self._handlers.discard(handler)
                        connection.close()
                        raise
        except BaseException as error:
            self._closing.set()
            try:
                self.failure.set_result(f"{type(error).__name__}: {error}")
            except InvalidStateError:
                # An owner failure can win the race; keep its first reason.
                pass
            with self._lock:
                self._owner_stopping = True
                self._enqueue(None)
        finally:
            self._listener.close()

    def _enqueue(self, item: Job | None) -> None:
        self._queue.put(item)
        os.eventfd_write(self._wakeup, 1)

    def _open_mailbox(
        self, principal: Principal, session: SessionId, request: Fields
    ) -> tuple[Response, Mailbox | None]:
        """Make the session's mailbox and have the owner serve it."""
        request.exact("op")
        mailbox = Mailbox()
        response = self._submit(Open(principal, session, mailbox))
        if response.get("ok") is not True:
            mailbox.close()
            return response, None
        return response, mailbox

    def _submit(self, work: CountRegistration | Load | Request | Begin | Close | Open) -> Response:
        """Hand `work` to the owner and wait for its answer, or for the owner's failure."""
        answer: Future[Response] = Future()
        with self._lock:
            if self.failure.done():
                return fatal("the service failed")
            if self._owner_stopping or (self._stopped and not isinstance(work, Close)):
                return fatal("the service is stopping")
            self._enqueue(Job(work, answer))
        while True:
            try:
                return answer.result(timeout=0.2)
            except FutureTimeout:
                if self.failure.done() and not answer.done():
                    return fatal("the service failed")

    def _kernel(self, request: Fields) -> tuple[str, Plan]:
        """A kernel of a register request: its text and the plan of its schema."""
        request.exact("ptx", "plan")
        text = request.text("ptx")
        if not text.isascii() or not 0 < len(text) <= self._limits.ptx_bytes:
            raise RequestError(
                f"{request.where}.ptx must be ASCII text of 1 to {self._limits.ptx_bytes} bytes"
            )
        fields = request.fields("plan")
        schema = self._schemas.get(fields.text("kind"))
        if schema is None:
            raise RequestError(
                f"{request.where}.plan.kind names no plan schema of {sorted(self._schemas)}"
            )
        try:
            return text, schema.plan(text, fields)
        except RequestError:
            raise
        except Exception as error:  # a schema reads untrusted text: its failure is the client's
            raise RequestError(
                f"{request.where}.ptx does not fit the plan {schema.kind}: {error}"
            ) from error

    def _register(self, principal: Principal, session: SessionId, request: Fields) -> Response:
        """Count, check, and load each kernel of one register request. The checks
        run at once on the verification threads; each kernel's answer is its own,
        a refusal of its count, of its check, or of its load included."""
        request.exact("op", "kernels")
        entries = request.values("kernels")
        limit = self._limits.registrations_in_flight
        if not 0 < len(entries) <= limit:
            raise RequestError(f"register.kernels must hold 1 to {limit} kernels")
        kernels = [
            self._kernel(Fields(entry, f"register.kernels[{index}]"))
            for index, entry in enumerate(entries)
        ]
        with self._lock:
            held = self._in_flight.get(principal.name, 0)
            if held + len(kernels) > limit:
                raise RefusedError(
                    f"the principal has {held} registrations in flight, and {len(kernels)} "
                    f"more exceed its limit of {limit}"
                )
            self._in_flight[principal.name] = held + len(kernels)
        try:
            # Each count precedes its check; the owner loads or refuses every
            # counted kernel, which releases the place that its count held.
            started: list[Future[Checked] | str | Response] = []
            for text, plan in kernels:
                counted = self._submit(CountRegistration(principal))
                if counted.get("ok") is not True:
                    started.append(counted)
                    continue
                try:
                    started.append(
                        self._verifiers.submit(
                            _check,
                            text,
                            plan.binding,
                            self._limits.verification_seconds,
                            self._closing,
                        )
                    )
                except RuntimeError as error:  # the threads stopped with the service
                    started.append(f"{type(error).__name__}: {error}")
            answers: list[Response] = []
            for (text, plan), check in zip(kernels, started, strict=True):
                outcome: Checked | str
                match check:
                    case dict():
                        answers.append(check)
                        continue
                    case str():
                        outcome = check
                    case Future():
                        try:
                            outcome = check.result()
                        except Exception as error:  # the check failed; the owner is untouched
                            outcome = f"{type(error).__name__}: {error}"
                    case _ as never:
                        assert_never(never)
                answers.append(self._submit(Load(principal, session, plan, text, outcome)))
        finally:
            with self._lock:
                remaining = self._in_flight[principal.name] - len(kernels)
                if remaining:
                    self._in_flight[principal.name] = remaining
                else:
                    del self._in_flight[principal.name]
        for answer in answers:
            if _fatal(answer):
                return answer
        return {"ok": True, "registrations": answers}

    def _serve(self, connection: socket.socket) -> None:
        principal: Principal | None = None
        session: SessionId | None = None
        maximum = self._limits.message_bytes
        try:
            first = read_message(
                connection,
                maximum,
                "authenticate",
                timeout=self._limits.authentication_seconds,
            )
            first.exact("op", "token")
            if first.text("op") != "authenticate":
                raise RequestError("the first message must authenticate")
            principal = authenticate(self._principals, first.text("token"), _peer_uid(connection))
            if principal is None:
                write_message(
                    connection,
                    {"ok": False, "error": {"kind": "authentication", "reason": "denied"}},
                    maximum,
                    timeout=self._limits.response_seconds,
                )
                return
            opened = new_session()
            begun = self._submit(Begin(principal, opened))
            if begun.get("ok") is not True:
                # the session holds nothing: there is nothing to close
                write_message(connection, begun, maximum, timeout=self._limits.response_seconds)
                return
            session = opened
            write_message(
                connection,
                {"ok": True, "principal": principal.name},
                maximum,
                timeout=self._limits.response_seconds,
            )
            while not self._stopped:
                request = read_message(
                    connection, maximum, "request", timeout=self._limits.request_seconds
                )
                mailbox: Mailbox | None = None
                response: Response
                try:
                    match request.text("op"):
                        case "register":
                            response = self._register(principal, session, request)
                        case "data_plane":
                            response, mailbox = self._open_mailbox(principal, session, request)
                        case _:
                            response = self._submit(Request(principal, session, request))
                except RequestError as error:
                    response = {"ok": False, "error": {"kind": "input", "reason": str(error)}}
                except RefusedError as error:
                    response = {"ok": False, "error": {"kind": "refused", "reason": error.reason}}
                write_message(connection, response, maximum, timeout=self._limits.response_seconds)
                if mailbox is not None:
                    # the descriptors ride on one byte after the response
                    socket.send_fds(connection, [b"M"], mailbox.descriptors)
                if _fatal(response):
                    return
        except (EOFError, OSError):
            pass
        except RequestError as error:
            try:
                write_message(
                    connection,
                    {"ok": False, "error": {"kind": "protocol", "reason": str(error)}},
                    maximum,
                    timeout=self._limits.response_seconds,
                )
            except OSError:
                pass
        finally:
            if principal is not None and session is not None:
                self._submit(Close(principal, session))
            with self._lock:
                self._connections.discard(connection)
            connection.close()

    # ------------------------------------------------------------ lifetime

    def close(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._closing.set()
        with self._lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._acceptor.join(timeout=5)
        self._listener.close()
        # Running checks observe cancellation; queued checks never start. Keep
        # the owner alive until every handler has submitted its session cleanup.
        self._verifiers.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            handler.join(timeout=30)
        if any(handler.is_alive() for handler in handlers) or self._acceptor.is_alive():
            raise RuntimeError("the service's connection threads did not stop")
        with self._lock:
            self._owner_stopping = True
            self._enqueue(None)
        self._owner.join(timeout=60)
        self.path.unlink(missing_ok=True)
        if self._owner.is_alive():
            raise RuntimeError("the service's owner thread did not stop")
        os.close(self._wakeup)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Client:
    """A client of the service, with serialized requests on its one session."""

    def __init__(self, path: str | Path, token: str, *, message_bytes: int = MESSAGE_BYTES) -> None:
        self._maximum = message_bytes
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(path))
            write_message(connection, {"op": "authenticate", "token": token}, message_bytes)
            response = read_message(connection, message_bytes, "response")
            if response.raw("ok") is not True:
                error = response.fields("error")
                if error.text("kind") == "refused":
                    # the session's reservation does not fit in the pool
                    raise RefusedError(error.text("reason"))
                raise PermissionError("the service denied authentication")
            self.principal = response.text("principal")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._request_lock = Lock()

    def request(self, op: str, **fields: object) -> Fields:
        """Send one request and return its response's fields."""
        if "op" in fields:
            raise ValueError("op is not a request field")
        with self._request_lock:
            try:
                write_message(self._connection, {"op": op, **fields}, self._maximum)
                return read_message(self._connection, self._maximum, op)
            except (EOFError, OSError, RequestError):
                # A partial or malformed frame cannot safely be reused for the
                # next response. Closing also interrupts a concurrently blocked read.
                self.close()
                raise

    def register(self, *kernels: tuple[str, Mapping[str, object]]) -> list[Fields]:
        """Register `kernels`, each a PTX text and its plan, in one request, whose
        checks the service runs at once; each kernel's answer in order: its
        operator handle, or its own error. A refusal of the request as a whole
        raises `RefusedError`."""
        response = self.request(
            "register", kernels=[{"ptx": text, "plan": dict(plan)} for text, plan in kernels]
        )
        if response.raw("ok") is not True:
            raise RefusedError(response.fields("error").text("reason"))
        response.exact("ok", "registrations")
        answers = response.values("registrations")
        if len(answers) != len(kernels):
            raise RequestError("the register response has another number of answers")
        return [
            Fields(answer, f"register.registrations[{index}]")
            for index, answer in enumerate(answers)
        ]

    def prepare(self, **fields: object) -> int:
        """Prepare a step for this session's mailbox; its handle."""
        response = self.request("prepare", **fields)
        if response.raw("ok") is not True:
            raise RefusedError(response.fields("error").text("reason"))
        return response.integer("prepared", minimum=0)

    def data_plane(self, *, timeout: float = RESPONSE_SECONDS) -> ClientMailbox:
        """This session's mailbox, which the service passes after its response."""
        with self._request_lock:
            write_message(self._connection, {"op": "data_plane"}, self._maximum)
            response = read_message(self._connection, self._maximum, "data_plane")
            if response.raw("ok") is not True:
                raise RefusedError(response.fields("error").text("reason"))
            data, descriptors, _, _ = socket.recv_fds(self._connection, 1, 3)
            if data != b"M" or len(descriptors) != 3:
                for descriptor in descriptors:
                    os.close(descriptor)
                raise RequestError("the service did not pass the mailbox's descriptors")
        return ClientMailbox(descriptors, timeout=timeout)

    def audit(self, *, page_size: int = 1) -> list[object]:
        """Read the raw registration records present at the first page.

        A page is an explicit bounded request; an oversized page is refused,
        never retried with a different size. Counters are observed per page.
        """
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be a positive integer")
        records: list[object] = []
        total: int | None = None
        while total is None or len(records) < total:
            count = page_size if total is None else min(page_size, total - len(records))
            page = self.request("audit", start=len(records), count=count)
            if page.raw("ok") is not True:
                raise RefusedError(page.fields("error").text("reason"))
            page.exact("ok", "registrations", "total", "next")
            observed = page.integer("total", minimum=0)
            if total is None:
                total = observed
            if observed < total:
                raise RequestError("the audit's append-only record count decreased")
            entries = page.values("registrations")
            end = len(records) + len(entries)
            if len(entries) != min(count, observed - len(records)):
                raise RequestError("the audit page has an unexpected record count")
            if page.raw("next") != (end if end < observed else None):
                raise RequestError("the audit page has an unexpected next index")
            records.extend(entries)
        return records

    def close(self) -> None:
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
