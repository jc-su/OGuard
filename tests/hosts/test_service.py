"""The multi-tenant service is a real multi-process tenant boundary.

Two OS clients share one owner thread, CUDA context, and RMM pool without
receiving tenant ids or device pointers. The cuDF query node releases every
query allocation before the next request. The paired replay's unsafe Numba
PTX goes through the guard pass: the gate refuses the one-past module's
checked text before load, and accepts the wild module with a check, whose
failure a launch answers to its tenant alone as "check_failed", while the
safe control runs; a prepared step of a module with checks is refused.
Opaque handles and read grants are checked across tenants and sessions. A tenant's malformed or
hostile requests are answered as its own errors, and the service keeps
serving the other tenants; a query node that leaves a buffer live fails the
service explicitly.
"""

import base64
import multiprocessing
import socket
import struct
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
numba = pytest.importorskip("numba")
pytest.importorskip("cudf")
pytest.importorskip("rmm")

from numba import cuda, int64  # noqa: E402

from eval.corpus.adapter import binding_of  # noqa: E402
from eval.service_plans import NODES, SCHEMAS, ReadOneI64  # noqa: E402
from opgate.catalog import TenantId  # noqa: E402
from opgate.executor import RefusedError  # noqa: E402
from opgate.frontends import Target  # noqa: E402
from opgate.hosts.service import (  # noqa: E402
    Client,
    Fields,
    Limits,
    NodeContext,
    Principal,
    QueryNode,
    Service,
    binding_record,
)
from opgate.hosts.service.protocol import HEADER  # noqa: E402

POOL = 1 << 27
WILD_INDEX = 1 << 32
LIMITS = Limits(verification_workers=2, tenant_bytes=1 << 24, chunk_bytes=1 << 20)


def _token(name: str) -> str:
    return f"{name}-token-0123456789"


def _principals() -> tuple[Principal, ...]:
    return (
        Principal("alice", _token("alice"), TenantId(1)),
        Principal("bob", _token("bob"), TenantId(2)),
        Principal("admin", _token("admin"), TenantId(99), administrator=True),
    )


def _service(path: Path, *, nodes: tuple[QueryNode, ...] = ()) -> Service:
    path.parent.chmod(0o700)
    return Service(
        path,
        _principals(),
        device=0,
        pool_bytes=POOL,
        target=Target(8, 6),
        schemas=SCHEMAS,
        nodes=(*NODES, *nodes),
        limits=LIMITS,
    )


def _kind(response: Fields) -> str:
    """The kind of a failed response's error."""
    assert response.raw("ok") is False, response.record()
    return response.fields("error").text("kind")


def _audit(admin: Client) -> list[Fields]:
    records = admin.audit()
    fields = [Fields(r, "registration") for r in records]
    for record in fields:
        assert record.keys() == {
            "registration",
            "principal",
            "plan_kind",
            "ptx",
            "ptx_bytes",
            "binding",
            "guarded",
            "loaded",
            "decision",
            "reason",
            "module_load_count",
            "launch_count",
        }
        assert record.integer("ptx_bytes") == len(record.text("ptx").encode("ascii"))
        # an accepted registration loaded its text with the zeroing prologue
        assert (record.raw("loaded") is None) == (record.text("decision") == "refused")
        if record.raw("guarded") is not None:
            guarded = record.fields("guarded")
            assert guarded.keys() == {
                "ptx",
                "ptx_bytes",
                "binding",
                "checks",
                "limits",
                "restructured",
            }
            assert guarded.integer("ptx_bytes") == len(guarded.text("ptx").encode("ascii"))
    return fields


def _telemetry(admin: Client) -> Fields:
    return admin.request("telemetry").fields("telemetry")


def _one_past(source: np.ndarray, output: np.ndarray) -> None:
    if cuda.grid(1) == 0:
        output[0] = source[source.size]


def _wild(source: np.ndarray, output: np.ndarray, source_index: int) -> None:
    if cuda.grid(1) == 0:
        output[0] = source[source_index]


def _safe(source: np.ndarray, output: np.ndarray) -> None:
    if cuda.grid(1) == 0:
        output[0] = source[0]


def _client_query(
    path: str,
    token: str,
    base: float,
    ready: "Queue[str]",
    start: Event,
    results: "Queue[tuple[str, dict[str, object]]]",
) -> None:
    """A client in a spawned process."""
    with Client(path, token) as client:
        ready.put(client.principal)
        if not start.wait(timeout=30):
            raise RuntimeError("the client start event timed out")
        response = client.request(
            "cudf_linear", values=[base, base + 1, base + 2], slope=2, intercept=1
        )
        results.put((client.principal, response.record()))


def _upload(client: Client, values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    handle = client.request("allocate", nbytes=contiguous.nbytes).text("buffer")
    written = client.request(
        "write",
        buffer=handle,
        data=base64.b64encode(contiguous.tobytes()).decode("ascii"),
        element=None,
    )
    assert written.raw("ok") is True, written.record()
    return handle


@pytest.mark.gpu
def test_two_client_processes_share_the_gated_cudf_worker(tmp_path: Path) -> None:
    socket_path = tmp_path / "opgate.sock"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    with _service(socket_path):
        clients = [
            context.Process(
                target=_client_query,
                args=(str(socket_path), _token(name), base, ready, start, results),
            )
            for name, base in (("alice", 10.0), ("bob", 100.0))
        ]
        for process in clients:
            process.start()
        assert {ready.get(timeout=30), ready.get(timeout=30)} == {"alice", "bob"}
        start.set()
        observed = dict([results.get(timeout=60), results.get(timeout=60)])
        for process in clients:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert observed["alice"]["values"] == [21.0, 23.0, 25.0]
        assert observed["bob"]["values"] == [201.0, 203.0, 205.0]
        assert sorted(
            [observed["alice"]["gated_operators_added"], observed["bob"]["gated_operators_added"]]
        ) == [0, 1]
        with Client(socket_path, _token("admin")) as admin:
            telemetry = _telemetry(admin)
            audit = _audit(admin)
        assert telemetry.integer("live_buffers") == 0
        assert telemetry.integer("service_buffer_handles") == 0
        assert telemetry.integer("module_loads") == 1
        assert telemetry.integer("launch_submissions") == 2
        assert [
            (r.text("plan_kind"), r.integer("module_load_count"), r.integer("launch_count"))
            for r in audit
        ] == [("cudf_linear", 1, 2)]


@pytest.mark.gpu
def test_unsafe_ptx_and_handles_are_checked_at_the_service_boundary(tmp_path: Path) -> None:
    texts = {
        "one_past": cuda.compile_ptx(_one_past, (int64[::1], int64[::1]), cc=(8, 6), opt=True)[0],
        "wild": cuda.compile_ptx(_wild, (int64[::1], int64[::1], int64), cc=(8, 6), opt=True)[0],
        "safe": cuda.compile_ptx(_safe, (int64[::1], int64[::1]), cc=(8, 6), opt=True)[0],
    }
    socket_path = tmp_path / "opgate.sock"
    with (
        _service(socket_path),
        Client(socket_path, _token("alice")) as alice,
        Client(socket_path, _token("bob")) as bob,
        Client(socket_path, _token("admin")) as admin,
    ):
        # one request, whose three checks run at once; each kernel has its own answer
        one_past, wild, safe = alice.register(
            (texts["one_past"], {"kind": "read_one_i64", "source_index": None}),
            (texts["wild"], {"kind": "read_one_i64", "source_index": WILD_INDEX}),
            (texts["safe"], {"kind": "read_one_i64", "source_index": None}),
        )
        # the gate does not prove the one-past module's checked load; the wild one's
        # check makes it acceptable, and every launch of it fails the check
        assert _kind(one_past) == "refused"
        assert wild.raw("ok") is True and wild.values("checks"), wild.record()
        assert safe.raw("ok") is True and safe.values("checks") == [], safe.record()
        operator = safe.text("operator")

        source = alice.request("allocate", nbytes=24).text("buffer")
        output = alice.request("allocate", nbytes=8).text("buffer")
        data = struct.pack("<3q", 11, 22, 33)
        encoded = base64.b64encode(data).decode("ascii")
        assert alice.request(
            "write", buffer=source, data=encoded, element={"width": 8, "signed": True}
        ).raw("ok")
        run = alice.request("run_operator", operator=operator, source=source, output=output, n=3)
        assert run.record() == {"ok": True, "ran": [True]}
        assert struct.unpack(
            "<q", base64.b64decode(alice.request("read", buffer=output).text("data"))
        ) == (11,)

        assert _kind(bob.request("read", buffer=source)) == "refused"
        assert _kind(bob.request("free", buffer=source)) == "refused"
        assert (
            _kind(bob.request("run_operator", operator=operator, source=source, output=output, n=3))
            == "refused"
        )
        assert alice.request("grant_read", buffer=source, principal="bob").raw("ok") is True
        assert base64.b64decode(bob.request("read", buffer=source).text("data")) == data
        assert _kind(bob.request("write", buffer=source, data=encoded, element=None)) == "refused"

        assert _kind(alice.request("read", buffer="0" * 32)) == "refused"
        assert _kind(alice.request("arbitrary_python", source="lambda: 1")) == "input"
        assert _kind(alice.request("register", ptx=texts["safe"], binding={})) == "input"
        assert _kind(alice.request("register", kernels=[])) == "input"
        assert (
            _kind(
                alice.request(
                    "register", kernels=[{"ptx": texts["safe"], "plan": {"kind": "read_one_i64"}}]
                )
            )
            == "input"
        )

        failed = alice.request(
            "run_operator", operator=wild.text("operator"), source=source, output=output, n=3
        )
        assert _kind(failed) == "check_failed", failed.record()

        audit = _audit(admin)
        assert [r.text("ptx") for r in audit] == [
            texts[name] for name in ("one_past", "wild", "safe")
        ]
        assert safe.integer("registration") == 2
        assert [r.text("decision") for r in audit] == ["refused", "accepted", "accepted"]
        assert [r.integer("module_load_count") for r in audit] == [0, 1, 1]
        assert [r.integer("launch_count") for r in audit] == [0, 1, 1]
        assert [r.raw("guarded") is None for r in audit] == [False, False, True]
        assert _telemetry(admin).integer("failed_checks") == 1

    with (
        _service(tmp_path / "second.sock"),
        Client(tmp_path / "second.sock", _token("alice")) as reconnected,
    ):
        assert _kind(reconnected.request("read", buffer=source)) == "refused"
        assert (
            _kind(
                reconnected.request(
                    "run_operator", operator=operator, source=source, output=source, n=3
                )
            )
            == "refused"
        )


@pytest.mark.gpu
def test_a_guarded_kernel_reports_a_failed_check_to_its_tenant_alone(tmp_path: Path) -> None:
    """A kernel that reads `source[5]` for any `n`: the gate does not prove the read,
    and accepts it with the guard pass's check. A launch with `n = 3` fails the
    check and answers "check_failed" to its tenant, whose session and the other
    tenant's keep running; a launch with `n = 8` runs."""
    text = cuda.compile_ptx(_wild, (int64[::1], int64[::1], int64), cc=(8, 6), opt=True)[0]
    plan = {"kind": "read_one_i64", "source_index": 5}
    socket_path = tmp_path / "opgate.sock"
    with (
        _service(socket_path),
        Client(socket_path, _token("alice")) as alice,
        Client(socket_path, _token("bob")) as bob,
        Client(socket_path, _token("admin")) as admin,
    ):
        (registered,) = alice.register((text, plan))
        assert registered.raw("ok") is True and registered.raw("loaded") is True
        checks = registered.values("checks")
        assert checks, registered.record()
        operator = registered.text("operator")
        (record,) = _audit(admin)
        guarded = record.fields("guarded")
        assert record.text("ptx") == text and record.text("decision") == "accepted"
        assert guarded.values("checks") == checks
        assert guarded.text("ptx") != text and "opgate_status" in guarded.text("ptx")
        assert record.raw("binding") == binding_record(
            ReadOneI64().plan(text, Fields(plan, "plan")).binding
        )
        last = guarded.fields("binding").values("parameters")[-1]
        assert Fields(last, "parameter").text("name") == "opgate_status"

        secret = 0x77665544332211
        victim = _upload(bob, np.asarray([secret], dtype="<i8"))
        short = _upload(alice, np.asarray([11, 22, 33], dtype="<i8"))
        long = _upload(alice, np.arange(100, 108, dtype="<i8"))
        output = alice.request("allocate", nbytes=8).text("buffer")
        failed = alice.request("run_operator", operator=operator, source=short, output=output, n=3)
        assert _kind(failed) == "check_failed", failed.record()
        assert bob.request("cudf_linear", values=[1.0], slope=2.0, intercept=0.0).raw("values") == [
            2.0
        ]
        assert base64.b64decode(bob.request("read", buffer=victim).text("data")) == struct.pack(
            "<q", secret
        )
        ran = alice.request("run_operator", operator=operator, source=long, output=output, n=8)
        assert ran.record() == {"ok": True, "ran": [True]}
        assert struct.unpack(
            "<q", base64.b64decode(alice.request("read", buffer=output).text("data"))
        ) == (105,)
        # prepared uses of a module with checks report a failed check through the
        # mailbox (tests/hosts/test_prepared_status.py)
        for handle in (short, long, output):
            assert alice.request("free", buffer=handle).raw("ok") is True
        assert bob.request("free", buffer=victim).raw("ok") is True
        telemetry = _telemetry(admin)
        (record,) = [r for r in _audit(admin) if r.text("plan_kind") == "read_one_i64"]
        assert record.integer("launch_count") == 2
        assert telemetry.integer("failed_checks") == 1
        assert telemetry.integer("fatal_errors") == 0
        assert telemetry.integer("live_buffers") == 0


LINEITEM = {
    "shipdate": np.asarray([10, 20, 30, 40, 50], dtype="<i4"),
    "discount": np.asarray([0.06, 0.04, 0.07, 0.06, 0.06], dtype="<f8"),
    "quantity": np.asarray([10.0, 10.0, 24.0, 23.0, 10.0], dtype="<f8"),
    "extendedprice": np.asarray([100.0, 200.0, 300.0, 400.0, 500.0], dtype="<f8"),
}
Q6 = {
    "rows": 5,
    "date_lo": 20,
    "date_hi": 50,
    "discount_lo": 0.05,
    "discount_hi": 0.07,
    "quantity_max": 24.0,
}


@pytest.mark.gpu
def test_cudf_q6_node_uses_catalog_buffers_and_releases_temporaries(tmp_path: Path) -> None:
    socket_path = tmp_path / "opgate.sock"
    with (
        _service(socket_path),
        Client(socket_path, _token("alice")) as alice,
        Client(socket_path, _token("bob")) as bob,
        Client(socket_path, _token("admin")) as admin,
    ):
        handles = {name: _upload(alice, values) for name, values in LINEITEM.items()}
        assert _kind(bob.request("cudf_tpch_q6", **handles, **Q6)) == "refused"
        result = alice.request("cudf_tpch_q6", **handles, **Q6)
        assert result.number("revenue") == pytest.approx(24.0)
        assert result.integer("gated_operators_added") == 1
        telemetry = _telemetry(admin)
        assert telemetry.integer("live_buffers") == 4
        assert telemetry.fields("node_requests").integer("cudf_tpch_q6") == 2
        rows = [r for r in _audit(admin) if r.text("plan_kind") == "cudf_tpch_q6"]
        assert [(r.integer("module_load_count"), r.integer("launch_count")) for r in rows] == [
            (1, 1)
        ]
        # the query's temporaries came from alice's arena, whose chunks stay hers
        assert telemetry.integer("arena_chunks") > 0
        for handle in handles.values():
            assert alice.request("free", buffer=handle).raw("ok") is True
        assert _telemetry(admin).integer("live_buffers") == 0
        # the owner closes the session after the connection ends
        alice.close()
        deadline = time.monotonic() + 10
        while _telemetry(admin).integer("arena_chunks") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _telemetry(admin).integer("arena_chunks") == 0


@pytest.mark.gpu
def test_recorded_model_q6_runs_through_the_q6_node(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    text = (
        root / "eval" / "corpus" / "synthesized" / "sonnet" / "q6_sum" / "numba.O1.ptx"
    ).read_text()
    expected = binding_record(binding_of("sonnet", "q6_sum", "numba", text))
    socket_path = tmp_path / "opgate.sock"
    with (
        _service(socket_path),
        Client(socket_path, _token("alice")) as alice,
        Client(socket_path, _token("bob")) as bob,
        Client(socket_path, _token("admin")) as admin,
    ):
        (registered,) = alice.register((text, {"kind": "tpch_q6_sum"}))
        (bob_registered,) = bob.register((text, {"kind": "tpch_q6_sum"}))
        assert registered.raw("ok") is True and bob_registered.raw("loaded") is False
        handles = {name: _upload(alice, values) for name, values in LINEITEM.items()}
        refused = bob.request("tpch_q6", operator=bob_registered.text("operator"), **handles, **Q6)
        assert _kind(refused) == "refused"
        result = alice.request("tpch_q6", operator=registered.text("operator"), **handles, **Q6)
        assert result.number("revenue") == pytest.approx(24.0)
        rows = [r for r in _audit(admin) if r.text("plan_kind") == "tpch_q6_sum"]
        assert all(r.raw("binding") == expected for r in rows)
        assert [
            (r.text("principal"), r.integer("module_load_count"), r.integer("launch_count"))
            for r in rows
        ] == [
            ("alice", 1, 1),
            ("bob", 0, 0),
        ]
        # an operator of another kind: the client's error, and the service keeps serving
        assert (
            _kind(
                alice.request(
                    "run_operator", operator=registered.text("operator"), source="", output="", n=1
                )
            )
            == "input"
        )
        for handle in handles.values():
            assert alice.request("free", buffer=handle).raw("ok") is True
        assert _telemetry(admin).integer("fatal_errors") == 0


@pytest.mark.gpu
def test_a_tenant_cannot_exhaust_or_break_the_service(tmp_path: Path) -> None:
    socket_path = tmp_path / "opgate.sock"
    with (
        _service(socket_path),
        Client(socket_path, _token("alice")) as alice,
        Client(socket_path, _token("admin")) as admin,
    ):
        # past the tenant's quota, and past the pool
        assert _kind(alice.request("allocate", nbytes=LIMITS.tenant_bytes + 1)) == "refused"
        assert _kind(alice.request("allocate", nbytes=POOL + 1)) == "input"
        # a malformed message ends that connection only
        raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        raw.connect(str(socket_path))
        payload = b'{"op": "authenticate", "token": ' + b"[" * 64 + b"]" * 64 + b"}"
        raw.sendall(HEADER.pack(len(payload)) + payload)
        (length,) = HEADER.unpack(raw.recv(HEADER.size))
        assert b"protocol" in raw.recv(length)
        raw.close()
        # a connection that never authenticates is closed
        idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        idle.connect(str(socket_path))
        idle.settimeout(LIMITS.authentication_seconds + 5)
        assert idle.recv(1) == b""
        idle.close()
        assert alice.request("allocate", nbytes=64).raw("ok") is True
        assert _telemetry(admin).integer("fatal_errors") == 0


@dataclass(frozen=True)
class TemporaryAllocationNode:
    """An ordinary engine allocation fails after one temporary was created."""

    op: str = "temporary_allocations"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        import rmm

        first = rmm.DeviceBuffer(size=256, mr=context.resource.resource)
        second = rmm.DeviceBuffer(size=512, mr=context.resource.resource)
        return {"bytes": first.size + second.size}


@pytest.mark.gpu
def test_a_session_whose_reservation_does_not_fit_is_refused(tmp_path: Path) -> None:
    """Each tenant's first session reserves `tenant_bytes` of the pool; a tenant
    whose reservation the pool cannot hold beside the others' gets no session."""
    socket_path = tmp_path / "opgate.sock"
    socket_path.parent.chmod(0o700)
    with (
        Service(
            socket_path,
            _principals(),
            device=0,
            pool_bytes=POOL,
            target=Target(8, 6),
            schemas=SCHEMAS,
            nodes=(),
            limits=Limits(
                verification_workers=2, tenant_bytes=POOL // 2 + 256, chunk_bytes=1 << 20
            ),
        ) as service,
        Client(socket_path, _token("alice")) as alice,
    ):
        with pytest.raises(RefusedError, match="reservation"):
            Client(socket_path, _token("bob"))
        # a second session of the same tenant shares its reservation
        with Client(socket_path, _token("alice")) as again:
            assert again.request("allocate", nbytes=256).raw("ok") is True
        assert alice.request("allocate", nbytes=256).raw("ok") is True
        assert not service.failure.done()


@pytest.mark.gpu
def test_query_quota_failure_releases_temporaries_and_keeps_service_usable(tmp_path: Path) -> None:
    socket_path = tmp_path / "opgate.sock"
    with (
        Service(
            socket_path,
            _principals(),
            device=0,
            pool_bytes=POOL,
            target=Target(8, 6),
            schemas=SCHEMAS,
            nodes=(TemporaryAllocationNode(),),
            limits=Limits(verification_workers=2, tenant_bytes=512, chunk_bytes=256),
        ) as service,
        Client(socket_path, _token("alice")) as alice,
        Client(socket_path, _token("admin")) as admin,
    ):
        assert _kind(alice.request("temporary_allocations")) == "refused"
        assert not service.failure.done()
        assert _telemetry(admin).integer("live_buffers") == 0
        # the arena keeps its first chunk of 256 bytes, which the second
        # allocation does not fit beside
        assert _telemetry(admin).integer("arena_chunks") == 1
        buffer = alice.request("allocate", nbytes=256).text("buffer")
        assert alice.request("free", buffer=buffer).raw("ok") is True
        assert alice.request("release").record() == {"ok": True, "released_chunks": 1}
        assert _telemetry(admin).integer("arena_chunks") == 0
        # The released chunk no longer consumes the budget, and the session
        # stays authenticated.
        assert alice.request("allocate", nbytes=512).raw("ok") is True


@dataclass(frozen=True)
class LeakingNode:
    """A deployment's faulty node, which leaves a buffer live."""

    op: str = "leak"

    def run(self, context: NodeContext, request: Fields) -> dict[str, object]:
        context.resource.allocate(64)
        return {}


@pytest.mark.gpu
def test_a_node_that_leaves_a_buffer_live_fails_the_service(tmp_path: Path) -> None:
    socket_path = tmp_path / "opgate.sock"
    with (
        _service(socket_path, nodes=(LeakingNode(),)) as service,
        Client(socket_path, _token("alice")) as alice,
    ):
        assert _kind(alice.request("leak")) == "fatal"
        assert "InvariantError" in service.failure.result(timeout=10)
