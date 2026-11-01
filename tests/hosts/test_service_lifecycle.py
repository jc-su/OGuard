"""Service lifecycle and input boundaries without device work."""

import os
from concurrent.futures import Future
from contextlib import nullcontext
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import opgate.hosts.service.owner as owner_module
import opgate.hosts.service.server as server
from opgate.binding import GRID_LIMIT, Binding
from opgate.catalog import OperatorId, TenantId
from opgate.frontends import Target
from opgate.frontends.guard import UNPROVED, Guarded
from opgate.gate import TIMEOUT_REASON, Refused
from opgate.hosts import CheckFailedError
from opgate.hosts.service.owner import Close, InvariantError, Job, Owner, Request
from opgate.hosts.service.plans import LaunchRequest
from opgate.hosts.service.protocol import Fields, RequestError
from opgate.hosts.service.sessions import Limits, OperatorEntry, Principal, SessionId
from opgate.signature import Affine


@pytest.fixture
def service_owner(fake_executor):
    executor, _ = fake_executor
    person = Principal("alice", "a sufficiently long credential", TenantId(1))
    owner = Owner(
        device=0,
        pool_bytes=4096,
        target=Target(8, 6),
        principals=(person,),
        nodes={},
        limits=Limits(verification_workers=2),
        queue=Queue(),
        wakeup=os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK),
        started=Future(),
        failure=Future(),
    )
    owner._executor = executor
    owner._resource = Mock()  # a status buffer's allocator, which a module with checks uses
    owner._principal, owner._session = person, SessionId("session")
    return owner


@pytest.mark.parametrize(
    ("result", "submissions"),
    [((True,), 1), ((False,), 0), ((), None), ((True, False), None), ((1,), None)],
)
def test_operator_audit_counts_only_a_submitted_launch(
    service_owner, monkeypatch, result, submissions
):
    owner = service_owner
    principal, session = owner._current()
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    module = Mock(operator=OperatorId(1), binding=binding, text="// retained PTX")
    plan = Mock(
        kind="test",
        binding=binding,
        launch=Mock(return_value=LaunchRequest((1, 1, 1), (), (), (), True)),
    )
    registration = owner._record(
        principal, "test", module.text, binding, None, "accepted", None, 1, module
    )
    handle = owner._handles.add_operator(
        OperatorEntry(module, plan, principal.tenant, session, registration)
    )
    monkeypatch.setattr(owner.executor, "execute", Mock(return_value=result))
    request = Fields({"op": "run_operator", "operator": handle}, "request")
    if submissions is None:
        with pytest.raises(InvariantError, match="exactly one boolean"):
            owner._run_operator(request)
    else:
        assert owner._run_operator(request) == {"ran": list(result)}
    assert owner._telemetry.launch_requests == 1
    assert owner._telemetry.launch_submissions == (submissions or 0)
    assert owner._audit[registration].launch_count == (submissions or 0)


def test_a_failed_check_is_the_tenants_answer_and_counts_its_launch(service_owner, monkeypatch):
    owner = service_owner
    principal, session = owner._current()
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    module = Mock(operator=OperatorId(1), binding=binding, text="// retained PTX")
    plan = Mock(
        kind="test",
        binding=binding,
        launch=Mock(return_value=LaunchRequest((1, 1, 1), (), (), (), False)),
    )
    registration = owner._record(
        principal, "test", module.text, binding, None, "accepted", None, 1, module
    )
    handle = owner._handles.add_operator(
        OperatorEntry(module, plan, principal.tenant, session, registration)
    )
    monkeypatch.setattr(owner_module, "execute", Mock(side_effect=CheckFailedError("kernel")))
    monkeypatch.setattr(owner.executor, "tenant", Mock(return_value=nullcontext()))
    request = Request(principal, session, Fields({"op": "run_operator", "operator": handle}, "r"))
    assert owner._classified(request) == {
        "ok": False,
        "error": {
            "kind": "check_failed",
            "reason": "kernel: a check failed: an access did not run, or a loop reached its limit",
        },
    }
    assert owner._audit[registration].launch_count == 1
    assert owner._telemetry.failed_checks == 1
    assert owner._telemetry.fatal_errors == 0


@pytest.mark.parametrize("prior_failure", [None, "the acceptor failed first"])
def test_owner_failure_answers_and_drains_after_an_acceptor_failure(
    service_owner, monkeypatch, prior_failure
):
    owner = service_owner
    principal, session = owner._current()
    request = Request(principal, session, Fields({"op": "telemetry"}, "request"))
    current, queued = Future(), Future()
    owner._queue.put(Job(request, current))
    owner._queue.put(Job(request, queued))
    owner._queue.put(None)
    os.eventfd_write(owner._wakeup, 1)  # as the server signals after it queues work
    if prior_failure is not None:
        owner._failure.set_result(prior_failure)
    original = RuntimeError("the owner operation failed")
    monkeypatch.setattr(owner, "_classified", Mock(side_effect=original))
    with pytest.raises(RuntimeError, match="the owner operation failed") as raised:
        owner._serve()
    assert raised.value is original
    assert owner._failure.result() == (prior_failure or "RuntimeError: the owner operation failed")
    assert current.result() == {
        "ok": False,
        "error": {"kind": "fatal", "reason": "RuntimeError: the owner operation failed"},
    }
    assert queued.result() == {
        "ok": False,
        "error": {"kind": "fatal", "reason": "the service failed"},
    }
    assert owner._queue.empty() and owner._telemetry.fatal_errors == 1


@pytest.mark.parametrize("value", [10**400, float("inf"), float("nan"), True])
def test_number_validation_is_a_request_error(value):
    with pytest.raises(RequestError):
        Fields({"value": value}, "request").number("value")


def test_invalid_numeric_input_does_not_stop_owner(fake_executor, monkeypatch):
    executor, _ = fake_executor
    person = Principal("alice", "a sufficiently long credential", TenantId(1))
    node = SimpleNamespace(
        op="number", run=lambda context, fields: {"value": fields.number("value")}
    )
    owner = Owner(
        device=0,
        pool_bytes=4096,
        target=Target(8, 6),
        principals=(person,),
        nodes={},
        limits=Limits(verification_workers=2),
        queue=Queue(),
        wakeup=os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK),
        started=Future(),
        failure=Future(),
    )
    owner._executor = executor
    # Parsing happens before a node has created any device objects.
    monkeypatch.setattr(owner, "_dispatch", lambda principal, fields: node.run(owner, fields))
    session = SessionId("session")
    bad = owner._classified(Request(person, session, Fields({"value": 10**400}, "request")))
    good = owner._classified(Request(person, session, Fields({"value": 3}, "request")))
    assert Fields(bad, "response").fields("error").text("kind") == "input"
    assert good == {"ok": True, "value": 3.0}
    assert not owner._failure.done()


def test_small_buffer_view_is_refused_before_engine_access(fake_executor, monkeypatch):
    from opgate.executor import RefusedError
    from opgate.hosts.rapids import BufferView, CatalogResource

    executor, _ = fake_executor
    resource = CatalogResource.__new__(CatalogResource)
    resource._executor = executor
    with executor.tenant(TenantId(1)):
        executor.reserve(1 << 12)
        buffer = executor.allocate(8)
        monkeypatch.setattr(
            resource, "_owned", {buffer: SimpleNamespace(size=8, ptr=4096)}, raising=False
        )
        with pytest.raises(RefusedError, match="fewer"):
            BufferView.of(resource, buffer, "<i8", 2, access="read")
        assert BufferView.of(resource, buffer, "<i8", 1, access="read").count == 1
        executor.free(buffer)
        executor.unreserve()


def test_close_cancels_verification_and_drains_sessions_before_owner_stops(tmp_path, monkeypatch):
    checking, owner_stopped = Event(), Event()
    closed = []

    def run(owner):
        owner._started.set_result(None)
        while (job := owner._queue.get()) is not None:
            if isinstance(job.work, Close):
                closed.append(job.work.session)
            job.answer.set_result({"ok": True})
        owner_stopped.set()

    def verify(text, bindings, *, timeout, cancel):
        checking.set()
        assert cancel.wait(3), "shutdown did not cancel verification"
        raise InterruptedError("verification cancelled")

    monkeypatch.setattr(Owner, "run", run)
    monkeypatch.setattr(server, "verify", verify)
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    schema = Mock(
        kind="test", plan=lambda text, fields: SimpleNamespace(kind="test", binding=binding)
    )
    person = Principal("alice", "a sufficiently long credential", TenantId(1))
    service = server.Service(
        tmp_path / "service.sock",
        (person,),
        device=0,
        pool_bytes=4096,
        target=Target(8, 6),
        schemas=(schema,),
        nodes=(),
        limits=Limits(verification_workers=2),
    )
    client = server.Client(service.path, person.token)
    errors = []

    def register():
        try:
            client.request(
                "register",
                kernels=[
                    {"ptx": ".visible .entry kernel()\n{\nret;\n}\n", "plan": {"kind": "test"}}
                ],
            )
        except (EOFError, OSError) as error:
            errors.append(error)

    thread = Thread(target=register)
    thread.start()
    try:
        assert checking.wait(3)
        service.close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert owner_stopped.is_set() and len(closed) == 1
        assert not service._connections
        assert all(not handler.is_alive() for handler in service._handlers)
        assert service._submit(Close(person, closed[0]))["ok"] is False
    finally:
        client.close()
        service.close()


def test_a_registration_check_and_its_guard_rounds_share_one_deadline(monkeypatch):
    """The gate's first decision refuses an unproved access, so the guard pass runs;
    its check starts after the deadline and is a refusal for the time, without a
    check by the gate."""
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    unproved = Refused(f"kernel: {UNPROVED[0]}ld.global.u64 %rd2, [%rd1+0]", ())
    verify = Mock(return_value=unproved)
    clock = iter((100.0, 100.0, 107.0))
    monkeypatch.setattr(server, "verify", verify)
    monkeypatch.setattr("opgate.gate.time.monotonic", lambda: next(clock))

    def guard(text, bound, check):
        return Guarded(text + "// checked", bound, (), (), False, check(text + "// checked", bound))

    monkeypatch.setattr(server, "guard", guard)
    text = ".visible .entry kernel()\n{\nret;\n}\n"
    checked = server._check(text, binding, 5.0, Event())
    assert verify.call_count == 1
    assert checked.decision == Refused(f"{TIMEOUT_REASON} 5 seconds", ())
    assert checked.guarded is not None and checked.guarded.text == text + "// checked"
