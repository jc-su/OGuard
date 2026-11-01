"""Configuration, wire failure, and bounded raw-audit behavior without device work."""

import os
from concurrent.futures import Future
from queue import Queue
from threading import Event, Lock
from unittest.mock import Mock

import pytest

import opgate.hosts.service.server as server
from opgate.binding import GRID_LIMIT, Binding
from opgate.catalog import TenantId
from opgate.executor import RefusedError
from opgate.frontends import Target
from opgate.hosts.service.audit import Registration
from opgate.hosts.service.owner import Begin, Close, Owner, Request
from opgate.hosts.service.protocol import Fields, RequestError, encode
from opgate.hosts.service.sessions import Limits, Principal, SessionId, authenticate
from opgate.signature import Affine


@pytest.mark.parametrize(
    "values",
    [
        {"connections": True},
        {"verification_workers": 1.5},
        {"verification_workers": 0},
        {"registrations_in_flight": 0},
        {"tenant_bytes": 0},
        {"tenant_bytes": 1 << 20},  # smaller than the arena's first chunk
        {"chunk_bytes": 100},
        {"message_bytes": 1 << 32},
        {"authentication_seconds": float("nan")},
        {"verification_seconds": float("inf")},
        {"verification_seconds": True},
        {"request_seconds": -1},
        {"response_seconds": float("nan")},
        {"response_seconds": 10**400},
    ],
)
def test_invalid_limits_fail_before_service_start(values):
    with pytest.raises(ValueError):
        Limits(**({"verification_workers": 2} | values))


def test_the_verification_workers_have_no_default():
    with pytest.raises(TypeError, match="verification_workers"):
        Limits()  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "fields", [{"administrator": 1}, {"uid": True}, {"uid": -1}, {"token": "\ud800" * 16}]
)
def test_invalid_principal_configuration_is_rejected(fields):
    values = {"name": "alice", "token": "a sufficiently long credential", "tenant": TenantId(1)}
    with pytest.raises(ValueError):
        Principal(**(values | fields))


def test_a_non_utf8_credential_is_denied():
    principal = Principal("alice", "a sufficiently long credential", TenantId(1))
    assert authenticate((principal,), "\ud800" * 16, 0) is None


@pytest.mark.parametrize("failure", ["connect", "malformed", "denied", "refused"])
def test_client_initialization_closes_socket_on_every_failure(monkeypatch, failure):
    connection = Mock()
    monkeypatch.setattr(server.socket, "socket", lambda *args: connection)
    monkeypatch.setattr(server, "write_message", Mock())
    kind = "refused" if failure == "refused" else "authentication"
    response = Fields({"ok": False, "error": {"kind": kind, "reason": "no"}}, "authenticate")
    receive = Mock(return_value=response)
    monkeypatch.setattr(server, "read_message", receive)
    error: type[Exception]
    if failure == "connect":
        connection.connect.side_effect = OSError("connection refused")
        error = OSError
    elif failure == "malformed":
        receive.side_effect = RequestError("malformed authentication response")
        error = RequestError
    elif failure == "refused":
        # the session's reservation does not fit in the pool
        error = RefusedError
    else:
        error = PermissionError
    with pytest.raises(error):
        server.Client("/test.sock", "a sufficiently long credential")
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("error", [EOFError(), RequestError("malformed response")])
def test_client_closes_a_partial_or_invalid_response_stream(monkeypatch, error):
    client = server.Client.__new__(server.Client)
    client._maximum = 4096
    client._connection = Mock()
    client._request_lock = Lock()
    monkeypatch.setattr(server, "write_message", Mock())
    monkeypatch.setattr(server, "read_message", Mock(side_effect=error))
    with pytest.raises(type(error)):
        client.request("telemetry")
    client._connection.close.assert_called_once_with()


def test_a_failed_bind_does_not_remove_another_servers_path(tmp_path, monkeypatch):
    path = tmp_path / "service.sock"
    connection = Mock()

    def bind(address):
        # Another service won the race after the constructor's existence check.
        path.write_bytes(b"the other service's socket placeholder")
        raise OSError("address already in use")

    connection.bind.side_effect = bind
    monkeypatch.setattr(server.socket, "socket", lambda *args: connection)
    principal = Principal("alice", "a sufficiently long credential", TenantId(1))
    with pytest.raises(OSError, match="already in use"):
        server.Service(
            path,
            (principal,),
            device=0,
            pool_bytes=4096,
            target=Target(8, 6),
            schemas=(),
            nodes=(),
            limits=Limits(verification_workers=2),
        )
    assert path.read_bytes() == b"the other service's socket placeholder"
    connection.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["timeout", "initialization", "owner_start", "acceptor_start"])
def test_failed_service_startup_stops_a_late_owner_and_cleans_up(tmp_path, monkeypatch, failure):
    path = tmp_path / "service.sock"
    connection = Mock()
    connection.bind.side_effect = lambda address: path.touch()
    monkeypatch.setattr(server.socket, "socket", lambda *args: connection)
    verifiers = Mock()
    monkeypatch.setattr(server, "ThreadPoolExecutor", Mock(return_value=verifiers))
    started = Mock(spec=Future)
    monkeypatch.setattr(server, "Future", Mock(side_effect=[started, Future()]))
    owner = Mock()
    owner_type = Mock(return_value=owner)
    monkeypatch.setattr(server, "Owner", owner_type)
    owner_thread, acceptor_thread = Mock(), Mock()
    monkeypatch.setattr(server, "Thread", Mock(side_effect=[owner_thread, acceptor_thread]))
    error = TimeoutError("startup deadline") if failure == "timeout" else RuntimeError(failure)
    if failure in ("timeout", "initialization"):
        started.result.side_effect = error
    elif failure == "owner_start":
        owner_thread.start.side_effect = error
    else:
        acceptor_thread.start.side_effect = error

    def finish_delayed_initialization(*, timeout):
        # Model an owner finishing initialization only after construction fails.
        # It must find the stop request immediately, without waiting on an empty queue.
        queue = owner_type.call_args.kwargs["queue"]
        assert queue.get_nowait() is None

    owner_thread.join.side_effect = finish_delayed_initialization
    principal = Principal("alice", "a sufficiently long credential", TenantId(1))
    service = server.Service.__new__(server.Service)
    with pytest.raises(type(error), match=str(error)):
        service.__init__(
            path,
            (principal,),
            device=0,
            pool_bytes=4096,
            target=Target(8, 6),
            schemas=(),
            nodes=(),
            limits=Limits(verification_workers=2),
        )
    assert service._stopped and service._closing.is_set() and service._owner_stopping
    assert not path.exists()
    connection.close.assert_called_once_with()
    verifiers.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    if failure == "owner_start":
        owner_thread.join.assert_not_called()
    else:
        owner_thread.join.assert_called_once_with(timeout=60)


@pytest.mark.parametrize("failure", ["accept", "handler_create", "handler_start"])
def test_acceptor_failure_closes_resources_and_allows_service_shutdown(
    tmp_path, monkeypatch, failure
):
    service = server.Service.__new__(server.Service)
    service.path = tmp_path / "service.sock"
    service.path.touch()
    service._stopped = False
    service._closing = Event()
    service._owner_stopping = False
    service.failure = Future()
    service._queue = Queue()
    service._wakeup = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
    service._lock = Lock()
    service._limits = Limits(verification_workers=2)
    service._connections = set()
    service._handlers = set()
    service._listener = Mock()
    service._acceptor = Mock(is_alive=Mock(return_value=False))
    service._owner = Mock(is_alive=Mock(return_value=False))
    service._verifiers = Mock()
    connection, handler = Mock(), Mock()
    service._listener.accept.return_value = (connection, None)
    thread = Mock(return_value=handler)
    monkeypatch.setattr(server, "Thread", thread)
    error = RuntimeError(failure)
    if failure == "accept":
        service._listener.accept.side_effect = error
    elif failure == "handler_create":
        thread.side_effect = error
    else:

        def failed_start():
            # Shutdown must not collect this handler before start has resolved.
            acquired = service._lock.acquire(blocking=False)
            if acquired:
                service._lock.release()
            assert not acquired
            raise error

        handler.start.side_effect = failed_start

    service._accept()
    assert service.failure.result() == f"RuntimeError: {failure}"
    assert service._closing.is_set() and service._owner_stopping
    assert not service._connections and not service._handlers
    assert service._queue.get_nowait() is None
    service._listener.close.assert_called_once_with()
    if failure == "accept":
        connection.close.assert_not_called()
    else:
        connection.close.assert_called_once_with()
    service.close()
    handler.join.assert_not_called()
    service._owner.join.assert_called_once_with(timeout=60)
    service._verifiers.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    assert not service.path.exists()


def test_acceptor_failure_preserves_a_concurrent_owner_failure(monkeypatch):
    service = server.Service.__new__(server.Service)
    service._stopped = False
    service._closing = Event()
    service._owner_stopping = False
    service.failure = Future()
    service._queue = Queue()
    service._wakeup = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
    service._lock = Lock()
    service._listener = Mock()

    def owner_failed():
        service.failure.set_result("the owner failed first")
        raise OSError("the listener failed too")

    service._listener.accept.side_effect = owner_failed
    service._accept()
    assert service.failure.result() == "the owner failed first"
    assert service._closing.is_set() and service._owner_stopping
    assert service._queue.get_nowait() is None
    service._listener.close.assert_called_once_with()


def test_audit_pages_preserve_raw_records_and_refuse_an_oversized_page():
    principal = Principal("admin", "a sufficiently long credential", TenantId(1), True)
    owner = Owner(
        device=0,
        pool_bytes=4096,
        target=Target(8, 6),
        principals=(principal,),
        nodes={},
        limits=Limits(verification_workers=2, message_bytes=1024),
        queue=Queue(),
        wakeup=os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK),
        started=Future(),
        failure=Future(),
    )
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    texts = ["// " + letter * 200 for letter in "abc"]
    owner._audit = [
        Registration(index, "admin", "test", text, binding, None, text, "accepted", None, 1, index)
        for index, text in enumerate(texts)
    ]

    def page(start, count):
        return owner._administer(
            principal, Fields({"op": "audit", "start": start, "count": count}, "audit")
        )

    with pytest.raises(RefusedError, match="page exceeds"):
        page(0, 3)
    client = server.Client.__new__(server.Client)
    client.request = Mock(
        side_effect=lambda op, **fields: Fields({"ok": True, **page(**fields)}, op)
    )
    records = client.audit()
    assert records == [record.record() for record in owner._audit]
    assert [Fields(record, "registration").text("ptx").encode("ascii") for record in records] == [
        text.encode("ascii") for text in texts
    ]
    assert client.request.call_count == 3
    assert not owner._failure.done()
    for start in range(4):
        response = {"ok": True, **page(start, 1)}
        assert len(encode(response, 1024)) <= 1028
    with pytest.raises(RequestError):
        page(4, 1)


def test_audit_reads_only_records_present_in_the_first_page():
    client = server.Client.__new__(server.Client)
    client.request = Mock(
        side_effect=[
            Fields(
                {"ok": True, "registrations": [{"ptx": "first"}], "total": 2, "next": 1}, "audit"
            ),
            Fields(
                {"ok": True, "registrations": [{"ptx": "second"}], "total": 3, "next": 2}, "audit"
            ),
        ]
    )
    assert client.audit() == [{"ptx": "first"}, {"ptx": "second"}]
    assert client.request.call_count == 2


@pytest.mark.parametrize("phase", ["request", "response"])
def test_an_authenticated_timeout_closes_and_cleans_up_its_session(monkeypatch, phase):
    principal = Principal("alice", "a sufficiently long credential", TenantId(1))
    session = SessionId("the session")
    connection = Mock()
    service = server.Service.__new__(server.Service)
    service._limits = Limits(verification_workers=2)
    service._principals = (principal,)
    service._connections = {connection}
    service._lock = Lock()
    service._stopped = False
    service._submit = Mock(return_value={"ok": True, "value": 42})
    monkeypatch.setattr(server, "_peer_uid", lambda connection: 0)
    monkeypatch.setattr(server, "new_session", lambda: session)
    first = Fields({"op": "authenticate", "token": principal.token}, "authenticate")
    receive = Mock(
        side_effect=[first, TimeoutError()]
        if phase == "request"
        else [first, Fields({"op": "answer"}, "request")]
    )
    send = Mock(side_effect=None if phase == "request" else [None, TimeoutError()])
    monkeypatch.setattr(server, "read_message", receive)
    monkeypatch.setattr(server, "write_message", send)
    service._serve(connection)
    assert receive.call_args_list[0].kwargs == {"timeout": service._limits.authentication_seconds}
    assert receive.call_args_list[1].kwargs == {"timeout": service._limits.request_seconds}
    assert all(
        call.kwargs == {"timeout": service._limits.response_seconds} for call in send.call_args_list
    )
    submitted = [call.args[0] for call in service._submit.call_args_list]
    assert submitted[0] == Begin(principal, session)
    assert submitted[-1] == Close(principal, session)
    assert sum(isinstance(work, Request) for work in submitted) == (phase == "response")
    assert connection not in service._connections
    connection.close.assert_called_once_with()
