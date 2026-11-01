"""A session's prepared steps run through its mailbox, and the catalog decides each use."""

import base64
import os
import random
import select
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cudf")
pytest.importorskip("rmm")

from eval.service_plans import NODES, SCHEMAS  # noqa: E402
from opgate.catalog import TenantId  # noqa: E402
from opgate.executor import RefusedError  # noqa: E402
from opgate.frontends import Target  # noqa: E402
from opgate.hosts.service import Client, Fields, Limits, Principal, Service  # noqa: E402
from opgate.hosts.service.dataplane import (  # noqa: E402
    MAX_SCALARS,
    REQUEST,
    RESPONSE,
    RESPONSE_AT,
    ClientMailbox,
    Mailbox,
    Status,
)
from opgate.hosts.service.owner import Owner  # noqa: E402
from opgate.hosts.service.protocol import RequestError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PTX = ROOT / "eval" / "corpus" / "synthesized" / "sonnet" / "q6_sum" / "numba.O1.ptx"
COLUMNS = {
    "shipdate": np.asarray([10, 20, 30, 40, 50], dtype="<i4"),
    "discount": np.asarray([0.06, 0.04, 0.07, 0.06, 0.06], dtype="<f8"),
    "quantity": np.asarray([10.0, 10.0, 24.0, 23.0, 10.0], dtype="<f8"),
    "extendedprice": np.asarray([100.0, 200.0, 300.0, 400.0, 500.0], dtype="<f8"),
}
DOUBLE = {"width": 8, "signed": False}


def _token(name: str) -> str:
    return f"{name}-token-0123456789"


def _service(path: Path) -> Service:
    path.parent.chmod(0o700)
    principals = (
        Principal("alice", _token("alice"), TenantId(1)),
        Principal("bob", _token("bob"), TenantId(2)),
        Principal("admin", _token("admin"), TenantId(99), administrator=True),
    )
    return Service(
        path,
        principals,
        device=0,
        pool_bytes=1 << 27,
        target=Target(8, 6),
        schemas=SCHEMAS,
        nodes=NODES,
        limits=Limits(verification_workers=2, tenant_bytes=1 << 24, chunk_bytes=1 << 20),
    )


def _bits(value: float) -> int:
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


def _scalars(date_lo: int, date_hi: int) -> list[int]:
    return [date_lo, date_hi, _bits(0.05), _bits(0.07), _bits(24.0)]


def _upload(client: Client, values: np.ndarray) -> str:
    handle = client.request("allocate", nbytes=values.nbytes).text("buffer")
    data = base64.b64encode(values.tobytes()).decode("ascii")
    assert client.request("write", buffer=handle, data=data, element=None).raw("ok") is True
    return handle


def _revenue(data: bytes) -> float:
    (value,) = struct.unpack("<d", data)
    return value


def test_a_mailbox_cannot_be_resized():
    mailbox = Mailbox()
    try:
        with pytest.raises(PermissionError):
            os.ftruncate(mailbox.memory, 0)
    finally:
        mailbox.close()


@pytest.mark.gpu
def test_prepared_q6_runs_through_the_mailbox(tmp_path: Path) -> None:
    text = PTX.read_text()
    path = tmp_path / "opgate.sock"
    with (
        _service(path),
        Client(path, _token("alice")) as alice,
        Client(path, _token("bob")) as bob,
        Client(path, _token("admin")) as admin,
    ):
        operator = alice.register((text, {"kind": "tpch_q6_sum"}))[0].text("operator")
        handles = {name: _upload(alice, values) for name, values in COLUMNS.items()}
        result = alice.request("allocate", nbytes=8).text("buffer")
        step = {
            "operator": operator,
            "grid": [1, 1, 1],
            "buffers": [*handles.values(), result],
            "sizes": [5],
            "scalars": _scalars(20, 50),
            "fills": [{"buffer": result, "element": DOUBLE, "value": 0}],
            "read": result,
        }
        prepared = alice.prepare(**step)
        # another session names none of alice's handles
        bob_operator = bob.register((text, {"kind": "tpch_q6_sum"}))[0].text("operator")
        with pytest.raises(RefusedError):
            bob.prepare(**{**step, "operator": bob_operator})
        with alice.data_plane() as mailbox:
            for date_hi, expected in ((50, 24.0), (60, 54.0), (30, 0.0), (50, 24.0)):
                assert _revenue(mailbox.run(prepared, _scalars(20, date_hi))) == pytest.approx(
                    expected
                )
            with pytest.raises(RefusedError, match="a scalar lies outside its bounds"):
                mailbox.run(prepared, _scalars(20, 2**32))
            with pytest.raises(RefusedError, match="a scalar is missing or extra"):
                mailbox.run(prepared, [20])
            with pytest.raises(RequestError, match="no such step"):
                mailbox.run(prepared + 1, _scalars(20, 50))
            # a control request between uses; the free of a named buffer revokes
            # the step, and a release of the arena changes nothing of it
            assert alice.request("read", buffer=result).raw("ok") is True
            assert alice.request("free", buffer=handles["discount"]).raw("ok") is True
            assert alice.request("release").raw("ok") is True
            with pytest.raises(RefusedError, match="may not read"):
                mailbox.run(prepared, _scalars(20, 50))
        with bob.data_plane() as other:
            with pytest.raises(RequestError, match="no such step"):
                other.run(0, _scalars(20, 50))
        telemetry = admin.request("telemetry").fields("telemetry")
        assert telemetry.integer("fatal_errors") == 0


def _raw(mailbox: ClientMailbox, record: bytes) -> tuple[int, int, bytes] | None:
    """Write `record` as the request, ring, and read the response, if one comes."""
    view = mailbox._view
    view[: len(record)] = record
    os.eventfd_write(mailbox._request, 1)
    waiting = select.poll()
    waiting.register(mailbox._response, select.POLLIN)
    if not waiting.poll(2000):
        return None
    os.eventfd_read(mailbox._response)
    sequence, status, length = RESPONSE.unpack(view[RESPONSE_AT : RESPONSE_AT + RESPONSE.size])
    end = RESPONSE_AT + RESPONSE.size
    return sequence, status, bytes(view[end : end + length])


@pytest.mark.gpu
def test_a_hostile_client_cannot_break_the_mailbox(tmp_path: Path) -> None:
    text = PTX.read_text()
    path = tmp_path / "opgate.sock"
    rng = random.Random(11)
    with (
        _service(path),
        Client(path, _token("alice")) as alice,
        Client(path, _token("admin")) as admin,
    ):
        operator = alice.register((text, {"kind": "tpch_q6_sum"}))[0].text("operator")
        handles = {name: _upload(alice, values) for name, values in COLUMNS.items()}
        result = alice.request("allocate", nbytes=8).text("buffer")
        prepared = alice.prepare(
            operator=operator,
            grid=[1, 1, 1],
            buffers=[*handles.values(), result],
            sizes=[5],
            scalars=_scalars(20, 50),
            fills=[{"buffer": result, "element": DOUBLE, "value": 0}],
            read=result,
        )
        with alice.data_plane() as mailbox:
            assert _revenue(mailbox.run(prepared, _scalars(20, 50))) == pytest.approx(24.0)
            last = 1
            accepted = 1
            # garbage: every record is answered, as an input error or a refusal
            for _ in range(200):
                answer = _raw(mailbox, rng.randbytes(REQUEST.size))
                assert answer is not None
                last, status, payload = answer
                assert status in (Status.INPUT, Status.REFUSED), payload
            hostile = [
                (prepared + 7, 5, _scalars(20, 50), Status.INPUT),  # no such step
                (prepared, 2**32 - 1, [], Status.REFUSED),  # a count past the record
                (prepared, 5, _scalars(20, 2**32), Status.REFUSED),  # a scalar past its bound
                (prepared, 4, _scalars(20, 50)[:4], Status.REFUSED),  # a scalar missing
            ]
            for step, count, scalars, expected in hostile:
                last += 1
                words = [*scalars, *[0] * (MAX_SCALARS - len(scalars))]
                answer = _raw(mailbox, REQUEST.pack(last, step, count, *words))
                assert answer is not None and answer[:2] == (last, expected)
            # a stale sequence number and a doorbell rung without a record: no answer
            assert (
                _raw(mailbox, REQUEST.pack(last, prepared, 5, *_scalars(20, 50), *[0] * 27)) is None
            )
            for _ in range(1000):
                os.eventfd_write(mailbox._request, 1)
            # the session's next request is served as before
            mailbox._sequence = last
            for date_hi, expected_revenue in ((50, 24.0), (60, 54.0)):
                revenue = _revenue(mailbox.run(prepared, _scalars(20, date_hi)))
                assert revenue == pytest.approx(expected_revenue)
                accepted += 1
        telemetry = admin.request("telemetry").fields("telemetry")
        assert telemetry.integer("fatal_errors") == 0
        audit = [r for r in admin.audit() if Fields(r, "row").text("principal") == "alice"]
        assert sum(Fields(r, "row").integer("launch_count") for r in audit) == accepted


@pytest.mark.gpu
def test_an_owner_failure_reaches_the_mailbox_and_ends_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed owner answers the pending use with the failure, every later use
    fails at once, and engine allocations returned after the executor closed
    change nothing (`CatalogResource.close`)."""
    text = PTX.read_text()
    path = tmp_path / "opgate.sock"
    with _service(path) as service, Client(path, _token("alice")) as alice:
        operator = alice.register((text, {"kind": "tpch_q6_sum"}))[0].text("operator")
        handles = {name: _upload(alice, values) for name, values in COLUMNS.items()}
        result = alice.request("allocate", nbytes=8).text("buffer")
        prepared = alice.prepare(
            operator=operator,
            grid=[1, 1, 1],
            buffers=[*handles.values(), result],
            sizes=[5],
            scalars=_scalars(20, 50),
            fills=[{"buffer": result, "element": DOUBLE, "value": 0}],
            read=result,
        )
        with alice.data_plane() as mailbox:

            def failing(*_: object) -> None:
                raise RuntimeError("the owner failed on purpose")

            monkeypatch.setattr(Owner, "_account", failing)
            for _ in range(2):
                with pytest.raises(RuntimeError, match="the owner failed on purpose"):
                    mailbox.run(prepared, _scalars(20, 50))
        assert "the owner failed on purpose" in service.failure.result(timeout=30)
