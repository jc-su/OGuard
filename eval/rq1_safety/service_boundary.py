"""RQ1: the paired replay's archived PTX, tenant handles, and two Q6 paths through the multi-tenant service.

    make rq1-service-boundary

The experiment reads the PTX that a successful earlier run of this experiment
retained (`--archive`), and compares the text directly; missing archived text
fails before the service starts. Two spawned client processes first issue
cuDF requests to one service. Tenant A then registers the paired replay's
one-past disclosure module, wild-address fault module, and safe control under
the `read_one_i64` plan schema. The gate refuses both unsafe modules as
written, and the service applies the guard pass (design decision 38): the
gate refuses the one-past module's checked text, with zero loads and
launches, and accepts the wild module with a check of its far read, whose
launch the service answers "check_failed" to tenant A alone. The service
must run the safe bytes, keep tenant B's buffer unchanged after the wild
launch, reject cross-tenant and stale handles, and remain healthy for
tenant B. Tenant B runs TPC-H Q6 over
its catalog-owned columns through the `cudf_tpch_q6` node and through the
recorded Sonnet-written `q6_sum` PTX registered under `tpch_q6_sum` and run by
the `tpch_q6` node. Registration records keep the submitted text and the
server-derived binding, and the guard pass's text, binding, and checks,
which the experiment compares directly.
"""

import argparse
import base64
import json
import math
import multiprocessing
import os
import platform
import secrets
import struct
import subprocess
import tempfile
from datetime import date
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TypedDict

import numpy as np

from eval import RECORDS, ROOT
from eval.service_plans import NODES, SCHEMAS, VERIFICATION_WORKERS
from opgate.catalog import TenantId
from opgate.frontends import Target
from opgate.hosts.service import Client, Fields, Limits, Principal, Service

POOL_BYTES = 256 << 20
TENANT_BYTES = 80 << 20  # each of the three principals' reservation
WILD_INDEX = 1 << 32
DEFAULT_ARCHIVE = RECORDS / "e8_service_2026-09-29.json"
MODEL_Q6_PTX = ROOT / "eval" / "corpus" / "synthesized" / "sonnet" / "q6_sum" / "numba.O1.ptx"
MODEL_Q6_SOURCE = ROOT / "eval" / "corpus" / "synthesized" / "sonnet" / "q6_sum" / "numba.py"
SOURCES = (
    *(
        f"opgate/hosts/service/{name}.py"
        for name in ("protocol", "sessions", "plans", "audit", "owner", "server")
    ),
    "opgate/hosts/cudf.py",
    "opgate/hosts/rapids.py",
    "opgate/frontends/numba_cuda.py",
    "eval/service_plans.py",
)
MODULES = ("one_past", "wild", "safe")


class Parallel(TypedDict):
    principal: str
    pid: int
    response: dict[str, object]


def archived_modules(path: Path) -> dict[str, str]:
    """The PTX text of each module that a successful run submitted, by name."""
    record: object = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("success") is not True:
        raise RuntimeError(f"the service archive is not successful: {path}")
    registrations = record.get("registrations")
    rows = registrations.get("e7_audit_rows") if isinstance(registrations, dict) else None
    if not isinstance(rows, dict) or set(rows) != set(MODULES):
        raise RuntimeError(f"the service archive has no records of the modules {MODULES}: {path}")
    modules: dict[str, str] = {}
    for name, row in rows.items():
        text = row.get("ptx") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text or not text.isascii():
            raise RuntimeError(f"the archived module {name} has no ASCII PTX text")
        modules[str(name)] = text
    return modules


def _kind(response: Fields) -> str | None:
    """The kind of a failed response's error, or None when it succeeded."""
    if response.raw("ok") is True:
        return None
    return response.fields("error").text("kind")


def _parallel_client(
    path: str,
    token: str,
    base: float,
    ready: "Queue[tuple[str, int]]",
    start: Event,
    results: "Queue[Parallel]",
) -> None:
    with Client(path, token) as client:
        ready.put((client.principal, os.getpid()))
        if not start.wait(timeout=30):
            raise RuntimeError("the start event timed out")
        response = client.request(
            "cudf_linear", values=[base, base + 1, base + 2], slope=2, intercept=1
        )
        results.put(
            {"principal": client.principal, "pid": os.getpid(), "response": response.record()}
        )


def _upload(client: Client, values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    allocated = client.request("allocate", nbytes=contiguous.nbytes)
    if _kind(allocated) is not None:
        raise RuntimeError(f"an allocation failed: {allocated.record()}")
    handle = allocated.text("buffer")
    written = client.request(
        "write",
        buffer=handle,
        data=base64.b64encode(contiguous.tobytes()).decode("ascii"),
        element=None,
    )
    if _kind(written) is not None:
        raise RuntimeError(f"an upload failed: {written.record()}")
    return handle


def _environment() -> dict[str, object]:
    import cudf
    import numba
    import numba_cuda
    import rmm

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
            "--id=0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "python": platform.python_version(),
        "cudf": cudf.__version__,
        "numba": numba.__version__,
        "numba_cuda": numba_cuda.__version__,
        "rmm": rmm.__version__,
        "gpu_query": query.stdout.strip(),
        "sources": {name: (ROOT / name).read_text() for name in SOURCES},
    }


def run(archive: Path) -> dict[str, object]:
    from eval.tpch.data import day, generate

    modules = archived_modules(archive)
    model_q6 = MODEL_Q6_PTX.read_text()
    _, lineitem = generate(0.01, seed=11)
    date_lo, date_hi = day(date(1994, 1, 1)), day(date(1995, 1, 1))
    keep = (
        (lineitem.shipdate >= date_lo)
        & (lineitem.shipdate < date_hi)
        & (lineitem.discount >= 0.05)
        & (lineitem.discount <= 0.07)
        & (lineitem.quantity < 24.0)
    )
    expected = float((lineitem.extendedprice * lineitem.discount)[keep].sum())
    tokens = {name: secrets.token_hex(16) for name in ("tenant-a", "tenant-b", "observer")}
    principals = (
        Principal("tenant-a", tokens["tenant-a"], TenantId(1)),
        Principal("tenant-b", tokens["tenant-b"], TenantId(2)),
        Principal("observer", tokens["observer"], TenantId(99), administrator=True),
    )
    spawn = multiprocessing.get_context("spawn")
    ready: Queue[tuple[str, int]] = spawn.Queue()
    start = spawn.Event()
    results: Queue[Parallel] = spawn.Queue()

    with tempfile.TemporaryDirectory(prefix="opgate-service-") as temporary:
        socket_path = Path(temporary) / "service.sock"
        with Service(
            socket_path,
            principals,
            device=0,
            pool_bytes=POOL_BYTES,
            target=Target(8, 6),
            schemas=SCHEMAS,
            nodes=NODES,
            limits=Limits(verification_workers=VERIFICATION_WORKERS, tenant_bytes=TENANT_BYTES),
        ):
            children = [
                spawn.Process(
                    target=_parallel_client,
                    args=(str(socket_path), tokens[name], base, ready, start, results),
                )
                for name, base in (("tenant-a", 10.0), ("tenant-b", 100.0))
            ]
            for child in children:
                child.start()
            identities = [ready.get(timeout=30), ready.get(timeout=30)]
            start.set()
            parallel = [results.get(timeout=60), results.get(timeout=60)]
            for child in children:
                child.join(timeout=30)
                if child.exitcode != 0:
                    raise RuntimeError(f"a client exited with {child.exitcode}")

            with (
                Client(socket_path, tokens["tenant-a"]) as tenant_a,
                Client(socket_path, tokens["tenant-b"]) as tenant_b,
                Client(socket_path, tokens["observer"]) as observer,
            ):
                # tenant A's three modules in one request, checked at once
                one_past, wild, safe = tenant_a.register(
                    (modules["one_past"], {"kind": "read_one_i64", "source_index": None}),
                    (modules["wild"], {"kind": "read_one_i64", "source_index": WILD_INDEX}),
                    (modules["safe"], {"kind": "read_one_i64", "source_index": None}),
                )
                (model_registration,) = tenant_b.register((model_q6, {"kind": "tpch_q6_sum"}))
                (attacker_registration,) = tenant_a.register((model_q6, {"kind": "tpch_q6_sum"}))

                source = tenant_a.request("allocate", nbytes=24).text("buffer")
                output = tenant_a.request("allocate", nbytes=8).text("buffer")
                source_data = base64.b64encode(struct.pack("<3q", 11, 22, 33)).decode("ascii")
                tenant_a.request(
                    "write", buffer=source, data=source_data, element={"width": 8, "signed": True}
                )
                safe_launch = tenant_a.request(
                    "run_operator",
                    operator=safe.text("operator"),
                    source=source,
                    output=output,
                    n=3,
                )
                safe_read = tenant_a.request("read", buffer=output)
                (safe_value,) = struct.unpack("<q", base64.b64decode(safe_read.text("data")))

                secret = 0x77665544332211
                victim = tenant_b.request("allocate", nbytes=8).text("buffer")
                victim_data = base64.b64encode(struct.pack("<q", secret)).decode("ascii")
                tenant_b.request(
                    "write", buffer=victim, data=victim_data, element={"width": 8, "signed": True}
                )
                wild_launch = tenant_a.request(
                    "run_operator",
                    operator=wild.text("operator"),
                    source=source,
                    output=output,
                    n=3,
                )
                cross_tenant_read = tenant_a.request("read", buffer=victim)
                cross_tenant_free = tenant_a.request("free", buffer=victim)
                victim_read = tenant_b.request("read", buffer=victim)
                (victim_after,) = struct.unpack("<q", base64.b64decode(victim_read.text("data")))
                victim_query_after = tenant_b.request(
                    "cudf_linear", values=[7.0, 8.0], slope=3.0, intercept=-1.0
                )

                columns = {
                    "shipdate": _upload(tenant_b, lineitem.shipdate),
                    "discount": _upload(tenant_b, lineitem.discount),
                    "quantity": _upload(tenant_b, lineitem.quantity),
                    "extendedprice": _upload(tenant_b, lineitem.extendedprice),
                }
                q6 = {
                    **columns,
                    "rows": len(lineitem.shipdate),
                    "date_lo": date_lo,
                    "date_hi": date_hi,
                    "discount_lo": 0.05,
                    "discount_hi": 0.07,
                    "quantity_max": 24.0,
                }
                cross_tenant_cudf = tenant_a.request("cudf_tpch_q6", **q6)
                cudf_q6 = tenant_b.request("cudf_tpch_q6", **q6)
                cross_tenant_generated = tenant_a.request(
                    "tpch_q6", operator=attacker_registration.text("operator"), **q6
                )
                generated_q6 = tenant_b.request(
                    "tpch_q6", operator=model_registration.text("operator"), **q6
                )
                frees = [tenant_b.request("free", buffer=handle) for handle in columns.values()]
                audit_before_disconnect = observer.audit()
                stale_buffer, stale_operator = source, safe.text("operator")

            with Client(socket_path, tokens["tenant-a"]) as reconnected:
                stale_read = reconnected.request("read", buffer=stale_buffer)
                stale_run = reconnected.request(
                    "run_operator",
                    operator=stale_operator,
                    source=stale_buffer,
                    output=stale_buffer,
                    n=3,
                )
            with Client(socket_path, tokens["observer"]) as observer:
                final_audit = [Fields(r, "registration") for r in observer.audit()]
                telemetry = observer.request("telemetry").fields("telemetry")

    rows = {}
    for name, text in modules.items():
        matches = [r for r in final_audit if r.text("ptx") == text]
        if len(matches) != 1:
            raise RuntimeError(f"expected one registration record of {name}, found {len(matches)}")
        rows[name] = matches[0]
    cudf_rows = [r for r in final_audit if r.text("plan_kind") == "cudf_tpch_q6"]
    model_rows = [
        r for r in final_audit if r.text("ptx") == model_q6 and r.text("plan_kind") == "tpch_q6_sum"
    ]
    by_principal = {item["principal"]: item["response"] for item in parallel}
    nodes = telemetry.fields("node_requests")
    success = (
        len({pid for _, pid in identities}) == 2
        and by_principal["tenant-a"].get("values") == [21.0, 23.0, 25.0]
        and by_principal["tenant-b"].get("values") == [201.0, 203.0, 205.0]
        and _kind(one_past) == "refused"
        and _kind(wild) is None
        and wild.raw("loaded") is True
        and len(wild.values("checks")) > 0
        and _kind(wild_launch) == "check_failed"
        and _kind(safe) is None
        and _kind(model_registration) is None
        and model_registration.raw("loaded") is True
        and _kind(attacker_registration) is None
        and attacker_registration.raw("loaded") is False
        and safe_launch.record() == {"ok": True, "ran": [True]}
        and safe_value == 11
        and rows["one_past"].integer("module_load_count") == 0
        and rows["one_past"].integer("launch_count") == 0
        and all(
            rows[n].integer("module_load_count") == 1 and rows[n].integer("launch_count") == 1
            for n in ("wild", "safe")
        )
        and [rows[n].raw("guarded") is None for n in MODULES] == [False, False, True]
        and _kind(cross_tenant_read) == "refused"
        and _kind(cross_tenant_free) == "refused"
        and victim_after == secret
        and victim_query_after.raw("values") == [20.0, 23.0]
        and _kind(cross_tenant_cudf) == "refused"
        and _kind(cross_tenant_generated) == "refused"
        and math.isclose(cudf_q6.number("revenue"), expected, rel_tol=1e-12)
        and math.isclose(generated_q6.number("revenue"), expected, rel_tol=1e-9)
        and generated_q6.raw("ran") == [True]
        and all(_kind(response) is None for response in frees)
        and [(r.integer("module_load_count"), r.integer("launch_count")) for r in cudf_rows]
        == [(1, 1)]
        and sorted((r.integer("module_load_count"), r.integer("launch_count")) for r in model_rows)
        == [(0, 0), (1, 1)]
        and _kind(stale_read) == "refused"
        and _kind(stale_run) == "refused"
        and telemetry.integer("live_buffers") == 0
        and telemetry.integer("service_buffer_handles") == 0
        and telemetry.integer("service_operator_handles") == 0
        and telemetry.integer("module_loads") == 5
        and telemetry.integer("launch_submissions") == 7
        and telemetry.integer("failed_checks") == 1
        and nodes.integer("tpch_q6") == 2
        and nodes.integer("cudf_tpch_q6") == 2
        and telemetry.integer("input_errors") == 0
        and telemetry.integer("fatal_errors") == 0
    )
    return {
        "experiment": "RQ1: the paired replay's PTX, tenant handles, and two Q6 paths through the service",
        "success": success,
        "replay_archive": str(archive),
        "exact_replayed_ptx": modules,
        "server": {
            "pid": os.getpid(),
            "pool_bytes": POOL_BYTES,
            "device": 0,
            "compute_capability": [8, 6],
            "transport": "Unix domain socket, length-prefixed strict JSON",
            "plan_schemas": [s.kind for s in SCHEMAS],
            "query_nodes": [n.op for n in NODES],
        },
        "parallel_clients": {"identities": identities, "results": parallel},
        "registrations": {
            "one_past_response": one_past.record(),
            "wild_response": wild.record(),
            "safe_response": safe.record(),
            "model_q6_response": model_registration.record(),
            "attacker_q6_response": attacker_registration.record(),
            "e7_audit_rows": {name: rows[name].record() for name in MODULES},
            "all_before_disconnect": audit_before_disconnect,
        },
        "safe_control": {
            "source_data": source_data,
            "read_response": safe_read.record(),
            "launch": safe_launch.record(),
            "observed_value": safe_value,
            "expected": 11,
        },
        "wild_launch": wild_launch.record(),
        "real_query": {
            "name": "TPC-H Q6",
            "data": "deterministic TPC-H-distribution generator",
            "scale": 0.01,
            "seed": 11,
            "rows": len(lineitem.shipdate),
            "expected_revenue": expected,
            "model_operator": {
                "model": "sonnet",
                "task": "q6_sum",
                "toolchain": "numba",
                "ptx_path": str(MODEL_Q6_PTX.relative_to(ROOT)),
                "source_text": MODEL_Q6_SOURCE.read_text(),
            },
            "cudf_service_response": cudf_q6.record(),
            "generated_service_response": generated_q6.record(),
            "cross_tenant_cudf_request": cross_tenant_cudf.record(),
            "cross_tenant_generated_request": cross_tenant_generated.record(),
            "input_free_responses": [response.record() for response in frees],
            "cudf_audit_rows": [r.record() for r in cudf_rows],
            "model_audit_rows": [r.record() for r in model_rows],
        },
        "tenant_isolation": {
            "cross_tenant_read": cross_tenant_read.record(),
            "cross_tenant_free": cross_tenant_free.record(),
            "victim_expected_value": secret,
            "victim_written_data": victim_data,
            "victim_read_response": victim_read.record(),
            "victim_observed_value": victim_after,
            "victim_query_after": victim_query_after.record(),
            "stale_read": stale_read.record(),
            "stale_operator_run": stale_run.record(),
        },
        "final_audit": [r.record() for r in final_audit],
        "telemetry": telemetry.record(),
        "environment": _environment(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; a new record keeps the archived one")
    result = run(args.archive.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["success"] is not True:
        raise SystemExit(f"the service boundary did not hold; see {args.output}")


if __name__ == "__main__":
    main()
