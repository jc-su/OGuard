"""Principals, sessions, and the opaque handles that name a session's buffers and operators.

A client authenticates as a principal, which the server maps to a tenant; a
connection is one session. A request names buffers and operators only by
handles, random strings that the server issued; a handle is valid in the
session that created it, and a buffer handle also for a tenant that holds a
read grant. `Handles` enforces this and the operator/registration quotas;
`CatalogResource` enforces the buffer quota, including engine and retained
allocations, and the tenant's reservation of `Limits.tenant_bytes` bounds
its bytes. A counted registration holds a place among its tenant's operator
handles until the owner loads or refuses it, so registrations that are
checked at once never together exceed `Limits.tenant_operators`.
"""

import hmac
import math
import secrets
from dataclasses import dataclass, field
from typing import NewType

from opgate.catalog import BufferId, TenantId
from opgate.executor import Module, RefusedError
from opgate.gate import REGISTRATION_SECONDS
from opgate.hosts.service.plans import Plan

SessionId = NewType("SessionId", str)
BufferHandle = NewType("BufferHandle", str)
OperatorHandle = NewType("OperatorHandle", str)

HANDLE_BYTES = 16
MESSAGE_BYTES = 64 << 20  # a request or response, unless `Limits.message_bytes` says otherwise
RESPONSE_SECONDS = 30.0  # to send a response, unless `Limits.response_seconds` says otherwise

__all__ = [
    "BufferEntry",
    "BufferHandle",
    "Handles",
    "Limits",
    "MESSAGE_BYTES",
    "OperatorEntry",
    "OperatorHandle",
    "Principal",
    "RESPONSE_SECONDS",
    "SessionId",
    "authenticate",
    "new_session",
]


@dataclass(frozen=True)
class Principal:
    """A server-side identity. The token is a credential, never a tenant id; with
    a `uid`, a connection must also come from a process of that user. The
    tenant's first session reserves `reservation` bytes of the pool, or
    `Limits.tenant_bytes` without it."""

    name: str
    token: str = field(repr=False)
    tenant: TenantId
    administrator: bool = False
    uid: int | None = None
    reservation: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or type(self.token) is not str
            or not self.name
            or len(self.token) < 16
        ):
            raise ValueError("a principal needs a name and a token of at least 16 characters")
        try:
            self.name.encode("utf-8")
            self.token.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("a principal's name and token must be UTF-8 text") from error
        if type(self.tenant) is not int or self.tenant < 0:
            raise ValueError("a tenant id must be a nonnegative integer")
        if type(self.administrator) is not bool:
            raise ValueError("administrator must be a boolean")
        if self.uid is not None and (type(self.uid) is not int or self.uid < 0):
            raise ValueError("uid must be a nonnegative integer or None")
        if self.reservation is not None and (
            type(self.reservation) is not int or self.reservation < 1
        ):
            raise ValueError("reservation must be a positive integer or None")


@dataclass(frozen=True, kw_only=True)
class Limits:
    """What the service accepts, as a whole and from each tenant.

    `verification_workers` is the number of threads that check registrations,
    each check one `coqc` process of the gate. It is a required field, which
    the deployment sets from the cores of the machine that it gives the checks. The
    evaluation's deployments set 28 (`eval.service_plans.VERIFICATION_WORKERS`),
    the checks that the registration and guarded records ran at a time. A
    principal has at most `registrations_in_flight` registrations counted
    and not yet answered, over its sessions; a register request of more
    kernels than its principal has places left is refused as a whole. The
    threads take the checks in the order of their requests, so a principal's
    check waits for at most `registrations_in_flight` checks of each other
    principal.
    """

    verification_workers: int  # registrations checked at once, over every principal
    registrations_in_flight: int = 16  # one principal's registrations counted and not answered
    message_bytes: int = MESSAGE_BYTES  # a request or response
    ptx_bytes: int = 16 << 20  # a registered module's text
    verification_seconds: float = (
        REGISTRATION_SECONDS  # a registration's check, from its thread's start; longer is a refusal
    )
    connections: int = 64  # open connections
    authentication_seconds: float = 5.0  # to send the first message
    request_seconds: float = 300.0  # to receive each complete authenticated request
    response_seconds: float = RESPONSE_SECONDS  # to send each complete response
    tenant_bytes: int = 1 << 30  # a tenant's reservation, which holds its buffers and arena chunks
    tenant_buffers: int = 4096  # a tenant's buffers and arena chunks
    chunk_bytes: int = 64 << 20  # the first chunk of a tenant's arena (`rapids.CHUNK`)
    tenant_operators: int = 256  # one tenant's operator handles
    tenant_registrations: int = 256  # one tenant's registrations, accepted or refused

    def __post_init__(self) -> None:
        for name in (
            "message_bytes",
            "ptx_bytes",
            "verification_workers",
            "registrations_in_flight",
            "connections",
            "tenant_bytes",
            "tenant_buffers",
            "chunk_bytes",
            "tenant_operators",
            "tenant_registrations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.message_bytes > (1 << 32) - 1:
            raise ValueError("message_bytes exceeds the wire protocol's four-byte length")
        if self.chunk_bytes % 256 or self.chunk_bytes > self.tenant_bytes:
            raise ValueError("chunk_bytes must be a multiple of 256 within tenant_bytes")
        for name in (
            "verification_seconds",
            "authentication_seconds",
            "request_seconds",
            "response_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class BufferEntry:
    buffer: BufferId
    nbytes: int
    owner: TenantId
    session: SessionId
    grants: set[TenantId]


@dataclass(frozen=True)
class OperatorEntry:
    module: Module
    plan: Plan
    owner: TenantId
    session: SessionId
    registration: int  # the index of its audit record


def new_session() -> SessionId:
    return SessionId(secrets.token_hex(HANDLE_BYTES))


def authenticate(principals: tuple[Principal, ...], token: str, uid: int) -> Principal | None:
    """The principal whose token this is, if the connecting user may use it; every
    token is compared, in constant time."""
    try:
        credential = token.encode("utf-8")
    except UnicodeEncodeError:
        return None
    found: Principal | None = None
    for principal in principals:
        if hmac.compare_digest(principal.token.encode("utf-8"), credential):
            found = principal
    if found is None or (found.uid is not None and found.uid != uid):
        return None
    return found


class Handles:
    """The live handles of every session, and each tenant's use of its quotas."""

    def __init__(self, limits: Limits) -> None:
        self._limits = limits
        self._buffers: dict[BufferHandle, BufferEntry] = {}
        self._operators: dict[OperatorHandle, OperatorEntry] = {}
        self._registrations: dict[TenantId, int] = {}
        # registrations counted and not yet loaded or refused, each holding a
        # place among its tenant's operator handles
        self._unloaded: dict[TenantId, int] = {}

    @property
    def buffer_count(self) -> int:
        return len(self._buffers)

    @property
    def operator_count(self) -> int:
        return len(self._operators)

    def count_registration(self, tenant: TenantId) -> None:
        """Count a registration against the tenant's quotas, or refuse it. The
        registration holds a place among the tenant's operator handles until
        `settle_registration`."""
        made = self._registrations.get(tenant, 0)
        if made >= self._limits.tenant_registrations:
            raise RefusedError(f"the tenant made {made} registrations, its limit")
        unloaded = self._unloaded.get(tenant, 0)
        if (
            sum(1 for e in self._operators.values() if e.owner == tenant) + unloaded
            >= self._limits.tenant_operators
        ):
            raise RefusedError(
                "the tenant's operator handles and counted registrations reach its limit"
            )
        self._registrations[tenant] = made + 1
        self._unloaded[tenant] = unloaded + 1

    def settle_registration(self, tenant: TenantId) -> None:
        """Release a counted registration's place, before its load adds an
        operator handle or its refusal adds none."""
        unloaded = self._unloaded.get(tenant, 0)
        if unloaded < 1:
            raise RuntimeError("a registration that was never counted was settled")
        if unloaded == 1:
            del self._unloaded[tenant]
        else:
            self._unloaded[tenant] = unloaded - 1

    def add_buffer(self, entry: BufferEntry) -> BufferHandle:
        handle = BufferHandle(secrets.token_hex(HANDLE_BYTES))
        self._buffers[handle] = entry
        return handle

    def add_operator(self, entry: OperatorEntry) -> OperatorHandle:
        handle = OperatorHandle(secrets.token_hex(HANDLE_BYTES))
        self._operators[handle] = entry
        return handle

    def readable(self, tenant: TenantId, session: SessionId, handle: str) -> BufferEntry:
        """A buffer that the session owns, or that its tenant holds a grant for."""
        found = self._buffers.get(BufferHandle(handle))
        if found is None:
            raise RefusedError("the buffer handle is unknown or stale")
        if found.owner == tenant:
            if found.session != session:
                raise RefusedError("the buffer handle belongs to another session")
        elif tenant not in found.grants:
            raise RefusedError("the tenant does not own or hold a grant for the buffer handle")
        return found

    def owned(
        self, tenant: TenantId, session: SessionId, handle: str
    ) -> tuple[BufferHandle, BufferEntry]:
        """A buffer that the session owns."""
        found = self._buffers.get(BufferHandle(handle))
        if found is None:
            raise RefusedError("the buffer handle is unknown or stale")
        if found.owner != tenant or found.session != session:
            raise RefusedError("only the owning session may free or grant a buffer")
        return BufferHandle(handle), found

    def remove_buffer(self, handle: BufferHandle) -> None:
        del self._buffers[handle]

    def operator(self, tenant: TenantId, session: SessionId, handle: str) -> OperatorEntry:
        found = self._operators.get(OperatorHandle(handle))
        if found is None:
            raise RefusedError("the operator handle is unknown or stale")
        if found.owner != tenant or found.session != session:
            raise RefusedError("the session does not own the operator handle")
        return found

    def close(self, tenant: TenantId, session: SessionId) -> list[BufferEntry]:
        """Drop the session's handles; the buffers it owned, which the caller frees."""
        buffers = [
            h for h, e in self._buffers.items() if e.owner == tenant and e.session == session
        ]
        entries = [self._buffers.pop(h) for h in buffers]
        for handle in [
            h for h, e in self._operators.items() if e.owner == tenant and e.session == session
        ]:
            del self._operators[handle]
        return entries

    def every_buffer(self) -> list[tuple[BufferHandle, BufferEntry]]:
        return list(self._buffers.items())
