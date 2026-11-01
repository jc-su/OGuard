"""Random control-plane requests never fail the service.

Three sessions send a seeded mix of requests: each operation of the protocol
and unknown ones, with fields that are missing, extra, of the wrong type, out
of range, another session's handles, or valid, between valid sequences of
allocation, write, read, grant, and free. Every response is an answer, an
input error, or a refusal, never a fatal error; the service keeps serving;
and once the sessions close, no buffer and no arena chunk is live.
"""

import base64
import random
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cudf")
pytest.importorskip("rmm")

from eval.service_plans import NODES, SCHEMAS  # noqa: E402
from opgate.catalog import TenantId  # noqa: E402
from opgate.frontends import Target  # noqa: E402
from opgate.hosts.service import Client, Limits, Principal, Service  # noqa: E402

OPERATIONS = (
    "allocate",
    "write",
    "read",
    "free",
    "release",
    "grant_read",
    "run_operator",
    "prepare",
    "register",
    "telemetry",
    "audit",
    "cudf_tpch_q6",
    "launch",
    "",
)
FIELDS = (
    "nbytes",
    "buffer",
    "data",
    "element",
    "principal",
    "operator",
    "grid",
    "buffers",
    "sizes",
    "scalars",
    "fills",
    "read",
    "ptx",
    "plan",
    "kernels",
    "start",
    "count",
    "shipdate",
    "discount",
    "quantity",
    "extendedprice",
)
REQUESTS = 600


def _token(name: str) -> str:
    return f"{name}-token-0123456789"


def _value(rng: random.Random, handles: list[str]) -> object:
    """A field's value: valid-looking, of another type, or out of range."""
    choices = (
        lambda: rng.choice(handles) if handles else "h0",
        lambda: rng.randint(-(2**70), 2**70),
        lambda: rng.choice((0, 1, 8, 256, 4096, 2**31 - 1, 2**32, -1)),
        lambda: rng.random(),
        lambda: rng.choice((True, False, None)),
        lambda: "".join(rng.choice("abc{}[]\\\"' é") for _ in range(rng.randint(0, 12))),
        lambda: base64.b64encode(rng.randbytes(rng.randint(0, 64))).decode("ascii"),
        lambda: [rng.randint(-3, 3) for _ in range(rng.randint(0, 5))],
        lambda: [rng.choice(handles) if handles else "h0" for _ in range(rng.randint(0, 6))],
        lambda: {"width": rng.choice((1, 2, 4, 8, 3, -8)), "signed": rng.choice((True, 1, None))},
        lambda: {"kind": rng.choice(("tpch_q6_sum", "nope", ""))},
        lambda: [{"buffer": rng.choice(handles) if handles else "h0", "value": rng.randint(-9, 9)}],
    )
    return rng.choice(choices)()


def _random(rng: random.Random, handles: list[str]) -> tuple[str, dict[str, object]]:
    fields = rng.sample(FIELDS, rng.randint(0, 5))
    return rng.choice(OPERATIONS), {f: _value(rng, handles) for f in fields}


def _valid(rng: random.Random, client: Client, handles: list[str]) -> list[dict[str, object]]:
    """A valid sequence: allocate, write, read, and perhaps grant and free."""
    data = rng.randbytes(rng.choice((8, 64, 1000)))
    allocated = client.request("allocate", nbytes=len(data)).record()
    if allocated.get("ok") is not True:
        return [allocated]
    handle = str(allocated["buffer"])
    handles.append(handle)
    encoded = base64.b64encode(data).decode("ascii")
    answers = [
        allocated,
        client.request("write", buffer=handle, data=encoded, element=None).record(),
    ]
    read = client.request("read", buffer=handle).record()
    assert base64.b64decode(str(read["data"])) == data
    answers.append(read)
    if rng.random() < 0.3:
        answers.append(client.request("grant_read", buffer=handle, principal="bob").record())
    if rng.random() < 0.5:
        answers.append(client.request("free", buffer=handle).record())
    return answers


@pytest.mark.gpu
@pytest.mark.slow
def test_random_requests_never_fail_the_service(tmp_path: Path) -> None:
    path = tmp_path / "opgate.sock"
    tmp_path.chmod(0o700)
    principals = (
        Principal("alice", _token("alice"), TenantId(1)),
        Principal("bob", _token("bob"), TenantId(2)),
        Principal("carol", _token("carol"), TenantId(3)),
        Principal("admin", _token("admin"), TenantId(99), administrator=True),
    )
    rng = random.Random(23)
    kinds: dict[str, int] = {}
    with Service(
        path,
        principals,
        device=0,
        pool_bytes=1 << 27,
        target=Target(8, 6),
        schemas=SCHEMAS,
        nodes=NODES,
        limits=Limits(
            verification_workers=2,
            tenant_bytes=1 << 24,
            chunk_bytes=1 << 20,
            verification_seconds=60.0,
        ),
    ) as service:
        clients = {name: Client(path, _token(name)) for name in ("alice", "bob", "carol")}
        handles: list[str] = []
        try:
            for _ in range(REQUESTS):
                client = clients[rng.choice(tuple(clients))]
                if rng.random() < 0.15:
                    answers = _valid(rng, client, handles)
                else:
                    op, fields = _random(rng, handles)
                    answers = [client.request(op, **fields).record()]
                for answer in answers:
                    if answer.get("ok") is True:
                        kinds["ok"] = kinds.get("ok", 0) + 1
                        continue
                    error = answer["error"]
                    assert isinstance(error, dict), answer
                    kind = str(error["kind"])
                    assert kind in ("input", "refused"), answer
                    kinds[kind] = kinds.get(kind, 0) + 1
                assert not service.failure.done()
        finally:
            for client in clients.values():
                client.close()
        with Client(path, _token("admin")) as admin:
            deadline = time.monotonic() + 10
            telemetry = admin.request("telemetry").fields("telemetry")
            while telemetry.integer("live_buffers") or telemetry.integer("arena_chunks"):
                if time.monotonic() > deadline:
                    break
                time.sleep(0.01)
                telemetry = admin.request("telemetry").fields("telemetry")
            assert telemetry.integer("fatal_errors") == 0
            assert telemetry.integer("live_buffers") == 0
            assert telemetry.integer("arena_chunks") == 0
    assert kinds.get("ok", 0) > 50 and kinds.get("input", 0) > 100 and kinds.get("refused", 0) > 10
