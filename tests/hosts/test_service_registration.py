"""Concurrent registrations without device work.

The service runs with its real server, sockets, verification threads, and the
owner's own handling of counts and loads (`Owner._classified`), over a mock
executor; the gate is a fake check. A register request's kernels are checked
at once, a principal's kernels in flight are bounded over its sessions, the
tenant's operator quota holds for registrations checked at once, and each
kernel's refusal or failed check is that kernel's own answer.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Barrier, Condition, Event, Semaphore, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import opgate.hosts.service.server as server
from opgate.binding import GRID_LIMIT, Binding
from opgate.catalog import OperatorId, TenantId
from opgate.executor import RefusedError
from opgate.frontends import Target
from opgate.gate import TIMEOUT_REASON, Accepted, Refused
from opgate.hosts.service.owner import Checked, CountRegistration, Load, Owner
from opgate.hosts.service.protocol import Fields
from opgate.hosts.service.sessions import Handles, Limits, Principal
from opgate.signature import Affine

ALICE = Principal("alice", "alice's sufficiently long credential", TenantId(1))
BOB = Principal("bob", "bob's sufficiently long credential", TenantId(2))
BINDING = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
PLAN = {"kind": "test"}

Check = Callable[[str, Binding, float, Event], Checked]


def _text(name: str) -> str:
    return f".visible .entry kernel()\n{{\nret;\n}}\n// {name}\n"


@contextmanager
def _service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check: Check, limits: Limits
) -> Iterator[tuple[server.Service, list[Owner]]]:
    """A service whose owner counts and loads registrations as the owner does,
    over a mock executor whose `register` loads every text as a new module."""
    owners: list[Owner] = []
    modules = iter(range(1, 1 << 20))

    def register(text: str, binding: Binding) -> SimpleNamespace:
        return SimpleNamespace(operator=OperatorId(next(modules)), binding=binding, text=text)

    def run(owner: Owner) -> None:
        owner._executor = Mock(
            tenant=Mock(return_value=nullcontext()),
            register=Mock(side_effect=register),
            catalog=Mock(operators=[]),
        )
        owners.append(owner)
        owner._started.set_result(None)
        while (job := owner._queue.get()) is not None:
            match job.work:
                case CountRegistration() | Load():
                    job.answer.set_result(owner._classified(job.work))
                case _:  # a session's start or end, which holds no device state here
                    job.answer.set_result({"ok": True})

    monkeypatch.setattr(Owner, "run", run)
    monkeypatch.setattr(server, "_check", check)
    schema = Mock(
        kind="test", plan=lambda text, fields: SimpleNamespace(kind="test", binding=BINDING)
    )
    service = server.Service(
        tmp_path / "service.sock",
        (ALICE, BOB),
        device=0,
        pool_bytes=4096,
        target=Target(8, 6),
        schemas=(schema,),
        nodes=(),
        limits=limits,
    )
    try:
        yield service, owners
    finally:
        service.close()


def _kind(answer: Fields) -> str | None:
    if answer.raw("ok") is True:
        return None
    return answer.fields("error").text("kind")


def test_the_kernels_of_one_request_are_checked_at_once(tmp_path, monkeypatch):
    workers = 4
    every = Barrier(workers, timeout=10)

    def check(text: str, binding: Binding, seconds: float, cancel: Event) -> Checked:
        every.wait()  # each check ends only once every kernel's check has started
        return Checked(Accepted(), None)

    limits = Limits(verification_workers=workers, registrations_in_flight=workers)
    with (
        _service(tmp_path, monkeypatch, check, limits) as (service, owners),
        server.Client(service.path, ALICE.token) as alice,
    ):
        answers = alice.register(*((_text(f"k{i}"), PLAN) for i in range(workers)))
        assert [_kind(a) for a in answers] == [None] * workers
        assert len({a.text("operator") for a in answers}) == workers
        # the owner loads them in the request's order, so the audit keeps that order
        assert [a.integer("registration") for a in answers] == list(range(workers))
        (owner,) = owners
        assert [r.ptx for r in owner._audit] == [_text(f"k{i}") for i in range(workers)]
        assert owner._handles.operator_count == workers and not owner._handles._unloaded


def test_a_principal_s_kernels_in_flight_are_bounded_over_its_sessions(tmp_path, monkeypatch):
    release, started = Event(), Semaphore(0)

    def check(text: str, binding: Binding, seconds: float, cancel: Event) -> Checked:
        if "slow" in text:
            started.release()
            assert release.wait(10)
        return Checked(Accepted(), None)

    limits = Limits(verification_workers=4, registrations_in_flight=3)
    with (
        _service(tmp_path, monkeypatch, check, limits) as (service, _),
        server.Client(service.path, ALICE.token) as first,
        server.Client(service.path, ALICE.token) as second,
        server.Client(service.path, BOB.token) as bob,
    ):
        slow: list[list[Fields]] = []
        thread = Thread(
            target=lambda: slow.append(
                first.register(*((_text(f"slow {i}"), PLAN) for i in range(3)))
            )
        )
        thread.start()
        try:
            for _ in range(3):
                assert started.acquire(timeout=10)
            # alice's three places are taken, in any of her sessions
            with pytest.raises(RefusedError, match="3 registrations in flight"):
                second.register((_text("fast"), PLAN))
            # another principal's kernel takes the fourth thread
            (answer,) = bob.register((_text("bob"), PLAN))
            assert _kind(answer) is None
        finally:
            release.set()
            thread.join(timeout=10)
        assert not thread.is_alive()
        (answers,) = slow
        assert [_kind(a) for a in answers] == [None, None, None]
        # the places return once the request is answered
        assert [_kind(a) for a in second.register(*((_text(f"f{i}"), PLAN) for i in range(3)))] == [
            None
        ] * 3
        # a request of more kernels than a principal's places is the client's error
        oversized = second.request(
            "register", kernels=[{"ptx": _text(f"o{i}"), "plan": PLAN} for i in range(4)]
        )
        assert oversized.fields("error").text("kind") == "input"


def test_registrations_checked_at_once_keep_the_operator_quota(tmp_path, monkeypatch):
    release = Event()
    counts = Condition()
    calls = [0]
    count_registration = Handles.count_registration

    def counting(self: Handles, tenant: TenantId) -> None:
        try:
            count_registration(self, tenant)
        finally:
            with counts:
                calls[0] += 1
                counts.notify_all()

    def check(text: str, binding: Binding, seconds: float, cancel: Event) -> Checked:
        assert release.wait(10)
        return Checked(Accepted(), None)

    monkeypatch.setattr(Handles, "count_registration", counting)
    limits = Limits(verification_workers=4, registrations_in_flight=4, tenant_operators=3)
    with (
        _service(tmp_path, monkeypatch, check, limits) as (service, owners),
        server.Client(service.path, ALICE.token) as first,
        server.Client(service.path, ALICE.token) as second,
    ):
        results: list[list[Fields]] = []
        threads = [
            Thread(
                target=lambda client=client, name=name: results.append(
                    client.register((_text(f"{name} 0"), PLAN), (_text(f"{name} 1"), PLAN))
                )
            )
            for client, name in ((first, "first"), (second, "second"))
        ]
        for thread in threads:
            thread.start()
        try:
            # every kernel is counted while no check has ended, so no load has
            # added an operator handle yet
            with counts:
                assert counts.wait_for(lambda: calls[0] == 4, timeout=10)
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=10)
        answers = [answer for result in results for answer in result]
        assert len(answers) == 4
        kinds = sorted((_kind(a) or "") for a in answers)
        assert kinds == ["", "", "", "refused"]
        (refused,) = [a for a in answers if _kind(a) == "refused"]
        assert "counted registrations reach its limit" in refused.fields("error").text("reason")
        (owner,) = owners
        assert owner._handles.operator_count == 3 and not owner._handles._unloaded


def test_each_kernel_s_refusal_or_failed_check_is_its_own_answer(tmp_path, monkeypatch):
    def check(text: str, binding: Binding, seconds: float, cancel: Event) -> Checked:
        if "late" in text:
            return Checked(Refused(f"{TIMEOUT_REASON} {seconds:g} seconds", ()), None)
        if "broken" in text:
            raise RuntimeError("the check failed")
        return Checked(Accepted(), None)

    limits = Limits(verification_workers=2, registrations_in_flight=4, verification_seconds=5.0)
    with (
        _service(tmp_path, monkeypatch, check, limits) as (service, owners),
        server.Client(service.path, ALICE.token) as alice,
    ):
        names = ("safe", "late", "broken", "other")
        answers = alice.register(*((_text(name), PLAN) for name in names))
        assert [_kind(a) for a in answers] == [None, "refused", "internal", None]
        assert answers[1].fields("error").text("reason") == f"{TIMEOUT_REASON} 5 seconds"
        (owner,) = owners
        assert [(r.ptx, r.decision) for r in owner._audit] == [
            (_text("safe"), "accepted"),
            (_text("late"), "refused"),
            (_text("other"), "accepted"),
        ]
        assert owner._handles.operator_count == 2 and not owner._handles._unloaded


def test_a_counted_registration_holds_an_operator_place_until_settled():
    tenant = TenantId(1)
    handles = Handles(Limits(verification_workers=1, tenant_operators=2))
    handles.count_registration(tenant)
    handles.count_registration(tenant)
    with pytest.raises(RefusedError, match="counted registrations reach its limit"):
        handles.count_registration(tenant)
    handles.settle_registration(tenant)  # its check refused it: it adds no handle
    handles.count_registration(tenant)
    with pytest.raises(RuntimeError, match="never counted"):
        handles.settle_registration(TenantId(2))
