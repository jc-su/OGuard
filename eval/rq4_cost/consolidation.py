"""RQ4: tenants on one GPU through the OpGate service, against a process or a service per tenant.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq4_cost.consolidation \\
        --output eval/records/consolidation_<date>.json

This experiment compares the same recorded Sonnet TPC-H Q6 kernel in seven
configurations on one otherwise idle RTX A6000:

* ``service``: independent client processes use the restricted Unix-socket
  service. One loader owns one lineitem table and grants its columns read-only
  to every tenant. The service has one owner thread, CUDA context, and pool.
  Each client prepares its Q6 step over the shared columns and its own result
  buffer, then runs it through its session's mailbox (design decision 34).
* ``service_per_tenant``: every tenant has its own service, with its own owner
  thread, context, pool, and table, and one client: the same request path as
  ``service``, with a process per tenant.
* ``process``: every tenant has a process, CUDA context, executor, and table,
  and runs the same prepared step directly, with no socket.
* ``mps``: the same process-per-tenant baseline under default CUDA MPS.
* ``mps_shared``: a process per tenant under default CUDA MPS that shares one
  table. One more process, the owner, allocates the five columns separately,
  each a multiple of 2 MiB, uploads them once, and exports them with legacy
  CUDA IPC (``cuIpcGetMemHandle``). Each tenant maps them
  (``cuIpcOpenMemHandle``) and launches the accepted Q6 kernel directly over
  the mapping, with a private result, on its own stream. The owner counts as
  a process, not as a tenant. IPC maps the columns writable, so after every
  tenant closed its mapping the owner compares the columns' bytes with the
  recorded ones.
* ``mps_static``: one exclusive static SM partition per tenant under CUDA MPS.
  Driver r610 adds partial containment for SM-triggered faults in this mode, so
  it is a distinct current isolation baseline rather than an MPS tuning point.
* ``guardian``: one process and one context for every tenant, each tenant's
  table and result in its own power-of-two partition of one pool, and the
  recorded kernel with Guardian's address fencing, launched directly with the
  partition's base and mask on the tenant's own stream.

Every request is checked against a NumPy reference. A service records its
registration audit and final telemetry, so a throughput point is tied to the
exact accepted PTX and binding and to a real generated launch. Each cell
(configuration, scale factor, tenants) runs `--repeats` times in fresh
processes, all cells in an order shuffled with a fixed seed.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import math
import multiprocessing as mp
import os
import platform
import random
import secrets
import shutil
import statistics
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier
from pathlib import Path
from queue import Empty
from typing import Literal, assert_never

import numpy as np

from eval import ROOT
from eval.baselines.guardian import patch_ptx
from eval.corpus.adapter import binding_of
from eval.direct import UNSIGNED, Direct, checked_kernels
from eval.service_plans import NODES, SCHEMAS, VERIFICATION_WORKERS
from eval.tpch.data import Lineitem, generate
from eval.tpch.pipelines import Q6_PARAMETERS, Columns, Session, load
from opgate.binding import ElementType
from opgate.catalog import (
    ALIGNMENT,
    Allocate,
    At,
    BufferId,
    Catalog,
    Fill,
    Kernel,
    Launch,
    Reserve,
    Step,
    StepRefused,
    TenantId,
    Value,
    Write,
)
from opgate.executor import Executor, Module, Run
from opgate.executor.driver import DriverError, address, call, driver
from opgate.frontends import Target
from opgate.gate import formal_dir
from opgate.hosts.service import Client, Fields, Limits, Principal, Service

Configuration = Literal[
    "service", "service_per_tenant", "process", "mps", "mps_shared", "mps_static", "guardian"
]
CONFIGURATIONS: tuple[Configuration, ...] = (
    "service",
    "service_per_tenant",
    "process",
    "mps",
    "mps_shared",
    "mps_static",
    "guardian",
)
REPEATS = 4
ORDER_SEED = 29
MPS_EXIT_SECONDS = 30.0  # the control daemon's end after `quit`
TENANTS = (1, 2, 4, 8, 12, 16)
SCALES = (0.1, 1.0)
WARMUP = 3.0
DURATION = 10.0
SETUP_TIMEOUT = 900.0
JOIN_TIMEOUT = 30.0
TABLE_BYTES_PER_ROW = 36
SLACK = 64 << 20
# the service's tenants each reserve this much of the pool for their results and
# status buffers; the loader reserves the table's pool
TENANT_BYTES = 1 << 20
GUARDIAN_FILLER = TenantId(1 << 20)  # holds the pool's bytes before the first partition
MAXIMUM_MESSAGE = 128 << 20
MODEL_PTX = ROOT / "eval/corpus/synthesized/sonnet/q6_sum/numba.O1.ptx"
QUERY = Q6_PARAMETERS
# the one flag that cuIpcOpenMemHandle accepts
IPC_FLAGS = int(driver.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
# CUDA's programming guide: share only allocations of a 2 MiB aligned size, so
# that no sub-allocated neighbor is exported with them
EXPORT_GRANULARITY = 2 << 20
# the Lineitem columns that the owner of mps_shared exports: Q6 reads the first
# four, and orderkey keeps the table at TABLE_BYTES_PER_ROW as in the others
SHARED_COLUMNS = ("shipdate", "discount", "quantity", "extendedprice", "orderkey")
SOURCES = (
    "eval/rq4_cost/consolidation.py",
    "eval/baselines/guardian.py",
    "eval/direct.py",
    "eval/service_plans.py",
    "eval/corpus/adapter.py",
    "eval/corpus/synthesized/sonnet/roles.tsv",
    "eval/corpus/synthesized/sonnet/q6_sum/signature.json",
    "eval/corpus/synthesized/sonnet/q6_sum/numba.O1.ptx",
    "eval/tpch/data.py",
    "eval/tpch/pipelines.py",
    "formal/Makefile",
    "uv.lock",
)


@dataclass(frozen=True)
class Stream:
    """One tenant's completed requests inside the measured window."""

    tenant: int
    q6: list[float]
    outputs: list[object]
    request: dict[str, object]


@dataclass(frozen=True)
class WorkerReport:
    """The streams returned by one outer worker and optional service, guardian,
    or shared-table evidence."""

    streams: list[Stream]
    service: dict[str, object] | None
    guardian: dict[str, object] | None = None
    shared: dict[str, object] | None = None


@dataclass(frozen=True)
class Exported:
    """The shared table as the owner of mps_shared sends it to every tenant:
    each column's IPC handle, the rows, and the reference."""

    handles: dict[str, bytes]
    rows: int
    expected: float


def _source_texts() -> dict[str, str]:
    sources = {name: ROOT / name for name in SOURCES}
    sources.update(
        (str(path.relative_to(ROOT)), path) for path in sorted((ROOT / "opgate").rglob("*.py"))
    )
    # Archive the actual selected proof build, including an explicit override.
    sources.update((f"formal/{p.name}", p) for p in sorted(formal_dir().glob("*.v")))
    return {name: path.read_bytes().decode("utf-8") for name, path in sources.items()}


def _environment() -> dict[str, object]:
    import cudf
    import rmm
    from cuda import bindings as cuda_bindings

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            _device(),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    coqc = subprocess.run(["coqc", "--version"], check=True, capture_output=True, text=True)

    return {
        "gpu": gpu,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cudf": cudf.__version__,
        "rmm": rmm.__version__,
        "cuda_bindings": cuda_bindings.__version__,
        "coqc": coqc.stdout.splitlines()[0],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "formal_library": str(formal_dir()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }


INPUT_TYPES = {
    "orderkey": "<i8",
    "quantity": "<f8",
    "extendedprice": "<f8",
    "discount": "<f8",
    "tax": "<f8",
    "shipdate": "<i4",
    "returnflag": "<i4",
    "linestatus": "<i4",
}


def _save_inputs(directory: Path, scales: tuple[float, ...], *, create: bool) -> None:
    """Retain actual table bytes once; a resume must use exactly those bytes."""
    if create:
        directory.mkdir(parents=True, exist_ok=False)
    for scale in scales:
        _, table = generate(scale, seed=5)
        folder = directory / f"sf{scale:g}"
        if create:
            folder.mkdir()
        for name, dtype in INPUT_TYPES.items():
            values = getattr(table, name)
            if values.dtype.str != dtype or values.ndim != 1:
                raise RuntimeError(f"unexpected input type for {name}")
            path = folder / f"{name}.bin"
            data = values.tobytes()
            if create:
                path.write_bytes(data)
            elif path.read_bytes() != data:
                raise RuntimeError(f"the recorded input changed: {path}")


def _read_inputs(directory: Path) -> Lineitem:
    columns = {
        name: np.frombuffer((directory / f"{name}.bin").read_bytes(), dtype=dtype)
        for name, dtype in INPUT_TYPES.items()
    }
    if len({len(values) for values in columns.values()}) != 1:
        raise RuntimeError(f"recorded columns have different lengths: {directory}")
    if not len(columns["quantity"]):
        raise RuntimeError(f"the recorded table is empty: {directory}")
    return Lineitem(**columns)


def _pool(rows: int) -> int:
    """One five-column table plus allocator and runtime slack."""
    requested = rows * TABLE_BYTES_PER_ROW + SLACK
    return -(-requested // 256) * 256


def _reference(lineitem: Lineitem) -> float:
    keep = (
        (lineitem.shipdate >= QUERY.date_lo)
        & (lineitem.shipdate < QUERY.date_hi)
        & (lineitem.discount >= QUERY.disc_lo)
        & (lineitem.discount <= QUERY.disc_hi)
        & (lineitem.quantity < QUERY.qty_max)
    )
    return float((lineitem.extendedprice * lineitem.discount)[keep].sum())


def _checked(observed: float, expected: float, tenant: int) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError(f"tenant {tenant} returned {observed}, expected {expected}")


def _direct_queries(
    session: Session,
    module: Module,
    table: Columns,
    expected: float,
    start: float,
) -> Stream:
    measured = start + WARMUP
    end = measured + DURATION
    samples: list[float] = []
    outputs: list[object] = []
    run = _direct_run(session, module, table)
    while True:
        began = time.perf_counter()
        if began >= end:
            break
        observed = _direct_q6(session, run)
        finished = time.perf_counter()
        _checked(observed, expected, int(session.tenant))
        if began >= measured and finished <= end:
            samples.append(finished - began)
            outputs.append(observed)
    if not samples:
        raise RuntimeError(f"tenant {int(session.tenant)} completed no measured request")
    return Stream(int(session.tenant), samples, outputs, {"rows": table.rows, **asdict(QUERY)})


def _double_bits(value: float) -> int:
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


def _scalars() -> tuple[int, ...]:
    """Q6's bounded parameters as the catalog's integers: dates, then float64 bits."""
    return (
        QUERY.date_lo % 2**32,
        QUERY.date_hi % 2**32,
        _double_bits(QUERY.disc_lo),
        _double_bits(QUERY.disc_hi),
        _double_bits(QUERY.qty_max),
    )


def _direct_run(session: Session, module: Module, table: Columns) -> Run:
    """Prepare the Q6 step that the service's clients prepare, over the tenant's own
    columns and result."""
    with session.executor.tenant(session.tenant):
        result = session.memory.allocate(8)
        launch = Launch(
            module.operator,
            (max(1, -(-table.rows // 256)), 1, 1),
            (table.shipdate, table.discount, table.quantity, table.price, result),
            (table.rows,),
            _scalars(),
            False,
        )
        step = Step(session.tenant, (Fill(result, ElementType(8, False), 0), launch))
        return session.executor.prepare(step, result)


def _direct_q6(session: Session, run: Run) -> float:
    """One use of the prepared Q6 step, without socket I/O."""
    executor = session.executor
    with executor.tenant(session.tenant):
        executor.submit(run, _scalars())
    while not executor.finished(run):
        pass
    (value,) = struct.unpack("<d", executor.result(run))
    return value


def _direct_worker(
    inputs: Path,
    tenant: int,
    ready: Barrier,
    results: mp.Queue[WorkerReport],
) -> None:
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    with Executor(0, _pool(rows)) as executor:
        with executor.tenant(TenantId(tenant)):
            executor.reserve(_pool(rows))
        session = Session(executor, TenantId(tenant), executor)
        text = MODEL_PTX.read_text()
        module = executor.register(text, binding_of("sonnet", "q6_sum", "numba", text))
        table = load(session, lineitem)
        ready.wait()
        stream = _direct_queries(session, module, table, expected, time.perf_counter())
    results.put(WorkerReport([stream], None))


def _require_ok(response: Fields, where: str) -> Fields:
    if response.raw("ok") is not True:
        raise RuntimeError(f"{where} failed: {response.record()}")
    return response


def _upload(client: Client, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    allocated = _require_ok(
        client.request("allocate", nbytes=int(contiguous.nbytes)),
        "service allocation",
    )
    handle = allocated.text("buffer")
    _require_ok(
        client.request(
            "write",
            buffer=handle,
            data=base64.b64encode(contiguous.tobytes()).decode("ascii"),
            element=None,
        ),
        "service upload",
    )
    return handle


def _service_queries(
    client: Client,
    tenant: int,
    operator: str,
    handles: dict[str, str],
    rows: int,
    expected: float,
    start: float,
) -> Stream:
    """Prepare the tenant's Q6 step over the shared columns and its own result, and
    run it through the session's mailbox."""
    result = _require_ok(client.request("allocate", nbytes=8), "service result").text("buffer")
    columns = ("shipdate", "discount", "quantity", "extendedprice")
    request: dict[str, object] = {
        "operator": operator,
        "grid": [max(1, -(-rows // 256)), 1, 1],
        "buffers": [*(handles[name] for name in columns), result],
        "sizes": [rows],
        "scalars": list(_scalars()),
        "fills": [{"buffer": result, "element": {"width": 8, "signed": False}, "value": 0}],
        "read": result,
    }
    prepared = client.prepare(**request)
    measured = start + WARMUP
    end = measured + DURATION
    samples: list[float] = []
    outputs: list[object] = []
    scalars = _scalars()
    with client.data_plane() as mailbox:
        while True:
            began = time.perf_counter()
            if began >= end:
                break
            (revenue,) = struct.unpack("<d", mailbox.run(prepared, scalars))
            finished = time.perf_counter()
            _checked(revenue, expected, tenant)
            if began >= measured and finished <= end:
                samples.append(finished - began)
                outputs.append(revenue)
    if not samples:
        raise RuntimeError(f"tenant {tenant} completed no measured service request")
    return Stream(tenant, samples, outputs, {"op": "prepare", **request})


def _service_client(
    socket_path: str,
    token: str,
    tenant: int,
    handles: dict[str, str],
    rows: int,
    expected: float,
    ready: Barrier,
    results: mp.Queue[dict[str, object]],
) -> None:
    text = MODEL_PTX.read_text()
    with Client(socket_path, token, message_bytes=MAXIMUM_MESSAGE) as client:
        registration = _require_ok(
            client.register((text, {"kind": "tpch_q6_sum"}))[0], "service registration"
        )
        operator = registration.text("operator")
        ready.wait()
        stream = _service_queries(
            client,
            tenant,
            operator,
            handles,
            rows,
            expected,
            time.perf_counter(),
        )
        results.put(
            {
                "stream": stream,
                "loaded": registration.raw("loaded"),
                "registration": registration.raw("registration"),
            }
        )


def _collect[T](
    queue: mp.Queue[T],
    workers: Sequence[BaseProcess],
    count: int,
    where: str,
) -> list[T]:
    deadline = time.monotonic() + SETUP_TIMEOUT
    found: list[T] = []
    while len(found) < count:
        try:
            found.append(queue.get(timeout=0.5))
            continue
        except Empty:
            pass
        failures = [worker for worker in workers if worker.exitcode not in (None, 0)]
        if failures:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            raise RuntimeError(
                f"{where}: worker exits " + ", ".join(str(worker.exitcode) for worker in failures)
            )
        if all(worker.exitcode is not None for worker in workers):
            raise RuntimeError(f"{where}: workers exited without {count} results")
        if time.monotonic() >= deadline:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            raise TimeoutError(f"{where}: did not finish within {SETUP_TIMEOUT:g} seconds")
    for worker in workers:
        worker.join(timeout=JOIN_TIMEOUT)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            raise TimeoutError(f"{where}: a worker did not exit")
        if worker.exitcode != 0:
            raise RuntimeError(f"{where}: a worker exited with {worker.exitcode}")
    return found


def _wait_for_closed_sessions(client: Client) -> Fields:
    deadline = time.monotonic() + 10.0
    while True:
        telemetry = _require_ok(client.request("telemetry"), "service telemetry").fields(
            "telemetry"
        )
        if telemetry.integer("service_operator_handles") == 0:
            return telemetry
        if time.monotonic() >= deadline:
            raise RuntimeError(f"service sessions retained operator handles: {telemetry.record()}")
        time.sleep(0.01)


def _service_worker(
    inputs: Path,
    tenant_ids: tuple[int, ...],
    ready: Barrier,
    results: mp.Queue[WorkerReport],
) -> None:
    """A service for the tenants `tenant_ids`, each with one client process; the
    clients start measuring when every client of the cell reaches `ready`."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    text = MODEL_PTX.read_text()
    tenants = len(tenant_ids)
    tokens = {
        name: secrets.token_hex(16) for name in ("loader", *(f"tenant-{t}" for t in tenant_ids))
    }
    principals = (
        Principal(
            "loader",
            tokens["loader"],
            TenantId(10_000),
            administrator=True,
            reservation=_pool(rows),
        ),
        *(
            Principal(f"tenant-{tenant}", tokens[f"tenant-{tenant}"], TenantId(tenant))
            for tenant in tenant_ids
        ),
    )
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="opgate-service-consolidation-") as temporary:
        socket_path = Path(temporary) / "service.sock"
        with Service(
            socket_path,
            principals,
            device=0,
            pool_bytes=_pool(rows) + tenants * TENANT_BYTES,
            target=Target(8, 6),
            schemas=SCHEMAS,
            nodes=NODES,
            limits=Limits(
                verification_workers=VERIFICATION_WORKERS,
                message_bytes=MAXIMUM_MESSAGE,
                tenant_bytes=TENANT_BYTES,
                chunk_bytes=TENANT_BYTES,
            ),
        ):
            with Client(socket_path, tokens["loader"], message_bytes=MAXIMUM_MESSAGE) as loader:
                handles = {
                    "shipdate": _upload(loader, lineitem.shipdate),
                    "discount": _upload(loader, lineitem.discount),
                    "quantity": _upload(loader, lineitem.quantity),
                    "extendedprice": _upload(loader, lineitem.extendedprice),
                    "orderkey": _upload(loader, lineitem.orderkey),
                }
                for tenant in tenant_ids:
                    for handle in handles.values():
                        _require_ok(
                            loader.request(
                                "grant_read",
                                buffer=handle,
                                principal=f"tenant-{tenant}",
                            ),
                            "service grant",
                        )

                client_results: mp.Queue[dict[str, object]] = context.Queue()
                clients = [
                    context.Process(
                        target=_service_client,
                        args=(
                            str(socket_path),
                            tokens[f"tenant-{tenant}"],
                            tenant,
                            handles,
                            rows,
                            expected,
                            ready,
                            client_results,
                        ),
                    )
                    for tenant in tenant_ids
                ]
                for client in clients:
                    client.start()
                raw = _collect(client_results, clients, tenants, "service clients")
                client_rows: list[dict[str, object]] = []
                streams: list[Stream] = []
                for item in raw:
                    if not isinstance(item, dict):
                        raise RuntimeError(
                            f"a service client returned malformed evidence: {item!r}"
                        )
                    stream = item.get("stream")
                    if not isinstance(stream, Stream):
                        raise RuntimeError(
                            f"a service client returned malformed evidence: {item!r}"
                        )
                    client_rows.append(item)
                    streams.append(stream)

                live = _wait_for_closed_sessions(loader)
                audit = loader.audit()
                q6_audit = [
                    row
                    for row in (Fields(value, "registration") for value in audit)
                    if row.text("plan_kind") == "tpch_q6_sum"
                ]
                if len(q6_audit) != tenants:
                    raise RuntimeError(
                        f"expected {tenants} Q6 registrations, found {len(q6_audit)}"
                    )
                if sum(row.integer("module_load_count") for row in q6_audit) != 1:
                    raise RuntimeError("the service did not load exactly one Q6 module")
                if any(row.text("ptx") != text for row in q6_audit):
                    raise RuntimeError("the service audit PTX differs from the submitted artifact")
                measured = sum(len(stream.q6) for stream in streams)
                if live.integer("launch_requests") < measured:
                    raise RuntimeError(
                        f"service telemetry is below measured completions: {live.record()}"
                    )
                if any(live.integer(field) != 0 for field in ("input_errors", "fatal_errors")):
                    raise RuntimeError(
                        f"the service recorded an input or fatal error: {live.record()}"
                    )

                for handle in handles.values():
                    _require_ok(loader.request("free", buffer=handle), "service free")
                final = _require_ok(loader.request("telemetry"), "final service telemetry").fields(
                    "telemetry"
                )
                if any(
                    final.integer(field) != 0
                    for field in (
                        "live_buffers",
                        "service_buffer_handles",
                        "service_operator_handles",
                        "input_errors",
                        "fatal_errors",
                    )
                ):
                    raise RuntimeError(f"service did not end cleanly: {final.record()}")

                service_evidence: dict[str, object] = {
                    "transport": "Unix domain socket, length-prefixed strict JSON",
                    "owner_threads": 1,
                    "contexts": 1,
                    "shared_table_copies": 1,
                    "ptx": text,
                    "client_registrations": [
                        {
                            "loaded": item["loaded"],
                            "registration": item["registration"],
                        }
                        for item in client_rows
                    ],
                    "audit": [
                        {
                            key: row.raw(key)
                            for key in (
                                "principal",
                                "plan_kind",
                                "ptx",
                                "binding",
                                "decision",
                                "module_load_count",
                                "launch_count",
                            )
                        }
                        for row in q6_audit
                    ],
                    "telemetry_before_table_free": live.record(),
                    "telemetry_after_cleanup": final.record(),
                }
    results.put(WorkerReport(streams, service_evidence))


def guardian_partition_bytes(rows: int) -> int:
    """The bytes one tenant's Q6 table and result take, each buffer aligned to the
    catalog's `ALIGNMENT`: the four read columns and the one result."""
    sizes = (rows * 4, rows * 8, rows * 8, rows * 8, 8)
    return sum(-(-max(1, n) // ALIGNMENT) * ALIGNMENT for n in sizes)


def guardian_partition_size(partition_bytes: int) -> int:
    """The smallest power of two that holds `partition_bytes`: Guardian's partition,
    which a tenant's base is a multiple of and whose mask is its size minus one."""
    if partition_bytes <= 0:
        raise ValueError("a partition holds at least one byte")
    return 1 << (partition_bytes - 1).bit_length()


def guardian_layout(pool_address: int, size: int, tenants: int) -> tuple[int, list[int]]:
    """The base of each tenant's partition in one pool: the first base is
    `pool_address` rounded up to `size`, and the bases follow by `size`, so each
    is a multiple of `size` and the partitions do not overlap."""
    first = -(-pool_address // size) * size
    return first, [first + index * size for index in range(tenants)]


def _guardian_place(
    executor: Executor,
    tenant: TenantId,
    lineitem: Lineitem,
    base_offset: int,
    size: int,
) -> tuple[dict[str, BufferId], BufferId, dict[str, int]]:
    """Place one tenant's four Q6 columns and its result in its partition, the
    tenant's reservation at the pool offsets `[base_offset, base_offset + size)`,
    each buffer aligned to the catalog's `ALIGNMENT`, and write the columns; the
    buffers, the result, and the pool offset of each. The tenant's first-fit
    suballocator places the buffers one after another from the reservation's
    start, which the caller made the partition."""
    arrays = {
        "shipdate": lineitem.shipdate,
        "discount": lineitem.discount,
        "quantity": lineitem.quantity,
        "price": lineitem.extendedprice,
    }
    offsets: dict[str, int] = {}
    placed: dict[str, BufferId] = {}
    writes: list[Write] = []
    offset = base_offset
    with executor.tenant(tenant):
        reservation = executor.catalog.reservation(tenant)
        if reservation is None or (reservation.offset, reservation.nbytes) != (base_offset, size):
            raise RuntimeError("the tenant's reservation is not its partition")
        for name, array in arrays.items():
            data = np.ascontiguousarray(array)
            offsets[name] = offset
            buffer = executor.allocate(data.nbytes)
            placed[name] = buffer
            writes.append(Write(buffer, data.tobytes(), None))
            offset = -(-(offset + data.nbytes) // ALIGNMENT) * ALIGNMENT
        offsets["result"] = offset
        result = executor.allocate(8)
        for name, buffer in (*placed.items(), ("result", result)):
            found = executor.catalog.buffer(buffer)
            if found is None or found.offset != offsets[name]:
                raise RuntimeError(f"the tenant's {name} does not lie where the layout states")
        executor.execute(Step(tenant, tuple(writes)))
    return placed, result, offsets


@dataclass(frozen=True)
class _LaunchPlan:
    """One tenant's direct Q6 launch, built once: the function, the launch
    geometry, the marshaled parameters (Guardian's with the partition's base
    and mask appended), its own stream, a pinned host copy of the result, the
    result's device address, and whether the function is Guardian's fenced one."""

    tenant: int
    function: object
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    values: tuple[int, ...]
    types: tuple[object, ...]
    stream: object
    host: int  # the pinned host copy of the result
    result: int  # the result's device address
    fenced: bool


def _guardian_worker(
    inputs: Path,
    tenant_ids: tuple[int, ...],
    ready: Barrier,
    results: mp.Queue[WorkerReport],
) -> None:
    """One process, one shared context: each tenant's table and result in its own
    power-of-two aligned partition of one pool, the recorded Sonnet Q6 kernel
    patched by Guardian's address fencing and launched with the partition's base
    and mask, the tenants on separate streams. Each tenant runs a closed loop of
    Q6 over the measured window on its own thread; the partition layout is kept
    raw."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    text = MODEL_PTX.read_text()
    binding = binding_of("sonnet", "q6_sum", "numba", text)
    fenced, report = patch_ptx(text)
    tenants = len(tenant_ids)
    size = guardian_partition_size(guardian_partition_bytes(rows))
    pool_bytes = size * (tenants + 1)
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    streams: list[Stream | None] = [None] * tenants
    errors: list[BaseException] = []
    with Executor(0, pool_bytes) as executor:
        module = executor.register(text, binding)
        first, bases = guardian_layout(executor.pool_address, size, tenants)
        # each tenant's reservation is its partition: a filler tenant reserves the
        # bytes before the first partition, and the tenants reserve theirs in order
        if first > executor.pool_address:
            with executor.tenant(GUARDIAN_FILLER):
                executor.reserve(first - executor.pool_address)
        for identifier in tenant_ids:
            with executor.tenant(TenantId(identifier)):
                executor.reserve(size)
        (fenced_handle,) = call(driver.cuModuleLoadData(fenced.encode("ascii") + b"\0"))
        (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
        created: list[object] = []
        hosts: list[int] = []
        try:
            (function,) = call(
                driver.cuModuleGetFunction(fenced_handle, binding.kernel.encode("ascii"))
            )
            direct = Direct(executor)
            count = len(binding.parameters)
            plans: list[_LaunchPlan] = []
            layout: list[dict[str, object]] = []
            for index, identifier in enumerate(tenant_ids):
                tenant = TenantId(identifier)
                placed, result, offsets = _guardian_place(
                    executor, tenant, lineitem, bases[index] - executor.pool_address, size
                )
                launch = Launch(
                    module.operator,
                    (max(1, -(-rows // 256)), 1, 1),
                    (
                        placed["shipdate"],
                        placed["discount"],
                        placed["quantity"],
                        placed["price"],
                        result,
                    ),
                    (rows,),
                    _scalars(),
                    False,
                )
                (kernel,) = checked_kernels(executor, Step(tenant, (launch,)))
                values = (*direct.values(kernel, module)[:count], bases[index], size - 1)
                widths = (*module.widths[:count], 8, 8)
                (stream,) = call(driver.cuStreamCreate(0))
                created.append(stream)
                (host,) = call(driver.cuMemAllocHost(8))
                hosts.append(address(host))
                plans.append(
                    _LaunchPlan(
                        identifier,
                        function,
                        kernel.grid,
                        module.binding.block,
                        values,
                        tuple(UNSIGNED[width] for width in widths),
                        stream,
                        address(host),
                        executor.pool_address + offsets["result"],
                        True,
                    )
                )
                layout.append(
                    {
                        "tenant": identifier,
                        "base": bases[index],
                        "mask": size - 1,
                        "columns": {name: offsets[name] for name in offsets if name != "result"},
                        "result": offsets["result"],
                    }
                )

            def tenant_loop(index: int, plan: _LaunchPlan) -> None:
                try:
                    call(driver.cuCtxSetCurrent(context))
                    ready.wait()
                    streams[index] = _launch_queries(plan, expected, time.perf_counter())
                except BaseException as error:  # noqa: BLE001 - reported to the main thread
                    errors.append(error)

            workers = [
                threading.Thread(target=tenant_loop, args=(index, plan))
                for index, plan in enumerate(plans)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            call(driver.cuCtxSetCurrent(context))
            call(driver.cuCtxSynchronize())
            if errors:
                raise errors[0]
            measured = [stream for stream in streams if stream is not None]
            if len(measured) != tenants:
                raise RuntimeError("a guardian tenant produced no stream")
            results.put(
                WorkerReport(
                    measured,
                    None,
                    {
                        "pool_bytes": pool_bytes,
                        "pool_address": executor.pool_address,
                        "first_partition": first,
                        "partition_bytes": size,
                        "fenced_accesses": report.instrumented_accesses,
                        "ptx": fenced,
                        "tenants": layout,
                    },
                )
            )
        finally:
            for stream in created:
                call(driver.cuStreamDestroy(stream))
            for host_address in hosts:
                call(driver.cuMemFreeHost(host_address))
            call(driver.cuModuleUnload(fenced_handle))
            call(driver.cuDevicePrimaryCtxRelease(device))


def _launch_queries(plan: _LaunchPlan, expected: float, start: float) -> Stream:
    """One tenant's closed loop of its direct Q6 launch on its own stream over the
    measured window: zero the result, launch, copy it back, and check it."""
    measured = start + WARMUP
    end = measured + DURATION
    samples: list[float] = []
    outputs: list[object] = []
    while True:
        began = time.perf_counter()
        if began >= end:
            break
        call(driver.cuMemsetD8Async(plan.result, 0, 8, plan.stream))
        call(
            driver.cuLaunchKernel(
                plan.function,
                *plan.grid,
                *plan.block,
                0,
                plan.stream,
                (plan.values, plan.types),
                0,
            )
        )
        call(driver.cuMemcpyDtoHAsync(plan.host, plan.result, 8, plan.stream))
        call(driver.cuStreamSynchronize(plan.stream))
        (value,) = struct.unpack("<d", ctypes.string_at(plan.host, 8))
        finished = time.perf_counter()
        _checked(value, expected, plan.tenant)
        if began >= measured and finished <= end:
            samples.append(finished - began)
            outputs.append(value)
    if not samples:
        raise RuntimeError(f"tenant {plan.tenant} completed no measured direct request")
    request: dict[str, object] = {"fenced": plan.fenced, "grid": list(plan.grid), **asdict(QUERY)}
    return Stream(plan.tenant, samples, outputs, request)


def _handle_bytes(handle: object) -> bytes:
    """The bytes of an IPC handle, which cuda-bindings exposes only through `getPtr`."""
    if not isinstance(handle, driver.CUipcMemHandle):
        raise DriverError(f"the driver returned {handle!r}, not an IPC handle")
    return ctypes.string_at(handle.getPtr(), driver.CU_IPC_HANDLE_SIZE)


def _handle_of(raw: bytes) -> object:
    """The IPC handle whose bytes are `raw`."""
    if len(raw) != driver.CU_IPC_HANDLE_SIZE:
        raise ValueError(f"an IPC handle has {driver.CU_IPC_HANDLE_SIZE} bytes, not {len(raw)}")
    handle = driver.CUipcMemHandle()
    ctypes.memmove(handle.getPtr(), raw, len(raw))
    return handle


def _release(function: Callable[..., tuple[object, ...]], *args: object) -> None:
    """Call the driver function `function` with `args` and check its status."""
    call(function(*args))


def _shared_owner(
    inputs: Path,
    tenants: int,
    inbox: mp.Queue[Exported],
    closed: Barrier,
    results: mp.Queue[WorkerReport],
) -> None:
    """The owner of mps_shared's table: allocate each recorded column apart, a
    multiple of `EXPORT_GRANULARITY` bytes, upload it, export it, and send the
    handles to each of the `tenants`. When every tenant has closed its mappings
    (`closed`), compare each column on the device with the bytes uploaded."""
    lineitem = _read_inputs(inputs)
    arrays = {name: np.ascontiguousarray(getattr(lineitem, name)) for name in SHARED_COLUMNS}
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    pointers: dict[str, object] = {}
    handles: dict[str, bytes] = {}
    allocated: dict[str, int] = {}
    try:
        call(driver.cuCtxSetCurrent(context))
        for name, array in arrays.items():
            allocated[name] = -(-array.nbytes // EXPORT_GRANULARITY) * EXPORT_GRANULARITY
            (pointers[name],) = call(driver.cuMemAlloc(allocated[name]))
            base, extent = call(driver.cuMemGetAddressRange(pointers[name]))
            if address(base) != address(pointers[name]) or extent != allocated[name]:
                raise RuntimeError(
                    f"{name}: the allocation is not one block of {allocated[name]} bytes"
                )
            call(driver.cuMemcpyHtoD(pointers[name], array.tobytes(), array.nbytes))
            (handle,) = call(driver.cuIpcGetMemHandle(pointers[name]))
            handles[name] = _handle_bytes(handle)
        # every column is on the device before any tenant maps it
        call(driver.cuCtxSynchronize())
        message = Exported(handles, len(lineitem.quantity), _reference(lineitem))
        for _ in range(tenants):
            inbox.put(message)
        closed.wait()
        compared = 0
        for name, array in arrays.items():
            read = bytearray(array.nbytes)
            call(driver.cuMemcpyDtoH(read, pointers[name], array.nbytes))
            if bytes(read) != array.tobytes():
                raise RuntimeError(f"a tenant changed the shared column {name}")
            compared += array.nbytes
        evidence: dict[str, object] = {
            "mechanism": "cuIpcGetMemHandle in the owner, cuIpcOpenMemHandle in each tenant",
            "columns": {
                name: {
                    "bytes": arrays[name].nbytes,
                    "allocated": allocated[name],
                    "address": address(pointers[name]),
                    "handle": handles[name].hex(),
                }
                for name in SHARED_COLUMNS
            },
            "compared_bytes": compared,
        }
    finally:
        for pointer in pointers.values():
            call(driver.cuMemFree(pointer))
        call(driver.cuDevicePrimaryCtxRelease(device))
    results.put(WorkerReport([], None, None, evidence))


def _shared_values(
    module: Module,
    tenant: int,
    rows: int,
    columns: tuple[int, ...],
    result: int,
) -> tuple[Kernel, tuple[int, ...]]:
    """The checked Q6 kernel over the four mapped `columns` and a private
    `result`, with its parameters: a catalog of its own checks the launch over
    a layout in which the five buffers lie one after another, and each pointer
    parameter then holds the device address of its buffer."""
    sizes = (4 * rows, 8 * rows, 8 * rows, 8 * rows, 8)
    starts: list[int] = []
    end = 0
    for nbytes in sizes:
        starts.append(end)
        end = -(-(end + nbytes) // ALIGNMENT) * ALIGNMENT
    buffers = tuple(BufferId(TenantId(tenant), index) for index in range(len(sizes)))
    launch = Launch(
        module.operator, (max(1, -(-rows // 256)), 1, 1), buffers, (rows,), _scalars(), False
    )
    catalog = Catalog(end, ((module.operator, module.binding),))
    allocations = tuple(
        Allocate(buffer, start, nbytes)
        for buffer, start, nbytes in zip(buffers, starts, sizes, strict=True)
    )
    effects = catalog.check(Step(TenantId(tenant), (Reserve(0, end), *allocations, launch)))
    if isinstance(effects, StepRefused):
        raise RuntimeError(f"the catalog refuses the shared launch: {effects.reason}")
    (kernel,) = (effect for effect in effects if isinstance(effect, Kernel))
    if kernel.checks or kernel.shared:
        raise RuntimeError(
            "a direct launch runs only kernels without checks or dynamic shared memory"
        )
    where = dict(zip(starts, (*columns, result), strict=True))
    values: list[int] = []
    for argument, width in zip(kernel.arguments, module.widths, strict=True):
        match argument:
            case At(offset=offset):
                values.append(where[offset])
            case Value(value=value):
                values.append(value % 2 ** (8 * width))
            case _ as never:
                assert_never(never)
    return kernel, tuple(values)


def _shared_tenant(
    tenant: int,
    inbox: mp.Queue[Exported],
    ready: Barrier,
    closed: Barrier,
    results: mp.Queue[WorkerReport],
) -> None:
    """A tenant of mps_shared: register the recorded kernel in an executor of
    one `ALIGNMENT` pool, map the owner's columns, and run a closed loop of the
    direct Q6 launch over them with a private result on its own stream. Close
    the mappings and release the context, then report and wait at `closed`."""
    exported = inbox.get(timeout=SETUP_TIMEOUT)
    if not isinstance(exported, Exported):
        raise RuntimeError(f"the owner sent {exported!r}, not the exported table")
    text = MODEL_PTX.read_text()
    # the executor's context is released after `cleanup` has closed every mapping
    with Executor(0, ALIGNMENT) as executor, ExitStack() as cleanup:
        module = executor.register(text, binding_of("sonnet", "q6_sum", "numba", text))
        mapped: dict[str, int] = {}
        for name in SHARED_COLUMNS:
            (pointer,) = call(
                driver.cuIpcOpenMemHandle(_handle_of(exported.handles[name]), IPC_FLAGS)
            )
            cleanup.callback(_release, driver.cuIpcCloseMemHandle, pointer)
            mapped[name] = address(pointer)
            if mapped[name] % ALIGNMENT:
                raise RuntimeError(f"{name} is mapped at {mapped[name]:#x}, not aligned")
        (result,) = call(driver.cuMemAlloc(8))
        cleanup.callback(_release, driver.cuMemFree, result)
        (host,) = call(driver.cuMemAllocHost(8))
        cleanup.callback(_release, driver.cuMemFreeHost, host)
        (stream,) = call(driver.cuStreamCreate(0))
        cleanup.callback(_release, driver.cuStreamDestroy, stream)
        # the first to run: no work remains on the mappings when they close
        cleanup.callback(_release, driver.cuCtxSynchronize)
        kernel, values = _shared_values(
            module,
            tenant,
            exported.rows,
            tuple(mapped[name] for name in SHARED_COLUMNS[:4]),
            address(result),
        )
        plan = _LaunchPlan(
            tenant,
            module.function,
            kernel.grid,
            module.binding.block,
            values,
            tuple(UNSIGNED[width] for width in module.widths),
            stream,
            address(host),
            address(result),
            False,
        )
        ready.wait()
        measured = _launch_queries(plan, exported.expected, time.perf_counter())
    mapping = {
        name: {"address": mapped[name], "handle": exported.handles[name].hex()}
        for name in SHARED_COLUMNS
    }
    results.put(
        WorkerReport([replace(measured, request={**measured.request, "mapped": mapping})], None)
    )
    closed.wait()


def _device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the index of exactly one GPU")
    return visible


def _memory(device: str) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            device,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


class Sampler:
    """Sample total device-memory use while a cell is running."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _memory(self.device))
            self._stop.wait(0.25)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


class Mps:
    """A private CUDA MPS control daemon for one measurement cell. The daemon
    sees the GPU that `CUDA_VISIBLE_DEVICES` names; its clients see that GPU
    as device 0, so the block sets `CUDA_VISIBLE_DEVICES` to 0 for the
    processes it starts and restores it afterwards."""

    def __init__(self, static_tenants: int | None = None) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="opgate-mps-"))
        self.static_tenants = static_tenants
        self.partitions: tuple[str, ...] = ()
        self.metadata: dict[str, object] = {"mode": "default"}
        self.env = {
            "CUDA_MPS_PIPE_DIRECTORY": str(self.directory / "pipe"),
            "CUDA_MPS_LOG_DIRECTORY": str(self.directory / "log"),
        }

    def _control(self, command: str) -> str:
        return subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=f"{command}\n",
            text=True,
            check=True,
            capture_output=True,
        ).stdout.strip()

    def __enter__(self) -> Mps:
        occupied = sorted(
            name for name in (*self.env, "CUDA_MPS_SM_PARTITION") if name in os.environ
        )
        if occupied:
            raise RuntimeError(f"MPS environment is already set: {occupied}")
        for path in self.env.values():
            Path(path).mkdir()
        self._gpu = _device()
        os.environ.update(self.env)
        started = False
        try:
            command = ["nvidia-cuda-mps-control", "-d"]
            if self.static_tenants is not None:
                command.append("-S")
            subprocess.run(command, check=True)
            started = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            if self.static_tenants is not None:
                self._create_static_partitions(self.static_tenants)
        except BaseException:
            if started:
                subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input="quit\n",
                    text=True,
                    check=False,
                    capture_output=True,
                )
            for name in self.env:
                os.environ.pop(name, None)
            shutil.rmtree(self.directory)
            raise
        return self

    def _create_static_partitions(self, tenants: int) -> None:
        if tenants <= 0:
            raise ValueError("static MPS requires at least one tenant")
        listing = self._control("lspart")
        device_rows = [
            line.split()
            for line in listing.splitlines()
            if line.lstrip().startswith("GPU-") and "/" not in line
        ]
        if len(device_rows) != 1 or len(device_rows[0]) != 5:
            raise RuntimeError(f"cannot parse MPS partition capacity:\n{listing}")
        try:
            total_chunks, used_chunks, total_sms, used_sms = map(int, device_rows[0][1:])
        except ValueError as error:
            raise RuntimeError(f"cannot parse MPS partition capacity:\n{listing}") from error
        if used_chunks != 0 or used_sms != 0 or total_chunks < tenants:
            raise RuntimeError(f"MPS device is not empty or is too small:\n{listing}")
        gpu_uuid = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
                "-i",
                self._gpu,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
            raise RuntimeError(f"cannot identify one GPU: {gpu_uuid!r}")
        chunks_per_partition = total_chunks // tenants
        partitions: list[str] = []
        for _ in range(tenants):
            response = self._control(f"sm_partition add {gpu_uuid} {chunks_per_partition}")
            prefix, suffix = "Partition ", " created"
            if not response.startswith(prefix) or not response.endswith(suffix):
                raise RuntimeError(f"cannot parse created MPS partition: {response!r}")
            partitions.append(response[len(prefix) : -len(suffix)])
        self.partitions = tuple(partitions)
        self.metadata = {
            "mode": "static_sm_partitioning",
            "gpu_uuid": gpu_uuid,
            "partitions": self.partitions,
            "total_chunks": total_chunks,
            "total_sms": total_sms,
            "chunks_per_partition": chunks_per_partition,
            "sms_per_partition": total_sms * chunks_per_partition // total_chunks,
            "unassigned_chunks": total_chunks - tenants * chunks_per_partition,
        }

    def __exit__(self, *_: object) -> None:
        # the control daemon removes its pipes after `quit` returns: the directory
        # is removed once the daemon's process has ended
        pid = int((self.directory / "pipe" / "nvidia-cuda-mps-control.pid").read_text())
        try:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, check=True)
            deadline = time.monotonic() + MPS_EXIT_SECONDS
            while _alive(pid):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"the MPS control daemon {pid} did not end within {MPS_EXIT_SECONDS:g} s"
                    )
                time.sleep(0.05)
        finally:
            for name in self.env:
                os.environ.pop(name, None)
            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu
        shutil.rmtree(self.directory)


def _alive(pid: int) -> bool:
    """Whether the process `pid` still runs."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _latency(sorted_seconds: list[float]) -> dict[str, float | int]:
    def percentile(q: float) -> float:
        index = min(len(sorted_seconds) - 1, int(q * len(sorted_seconds)))
        return sorted_seconds[index] * 1e6

    return {
        "count": len(sorted_seconds),
        "mean": statistics.fmean(sorted_seconds) * 1e6,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def run(
    configuration: Configuration,
    scale: float,
    tenants: int,
    device: str,
    *,
    inputs: Path,
    baseline_memory: int | None = None,
    mps: dict[str, object] | None = None,
    partitions: tuple[str, ...] = (),
) -> dict[str, object]:
    """Run one fresh-process measurement cell on the GPU `device` names."""
    context = mp.get_context("spawn")
    results: mp.Queue[WorkerReport] = context.Queue()
    baseline = _memory(device) if baseline_memory is None else baseline_memory
    match configuration:
        case "service":
            ready = context.Barrier(tenants)
            everyone = tuple(range(1, tenants + 1))
            workers = [
                context.Process(target=_service_worker, args=(inputs, everyone, ready, results))
            ]
        case "service_per_tenant":
            ready = context.Barrier(tenants)
            workers = [
                context.Process(target=_service_worker, args=(inputs, (tenant,), ready, results))
                for tenant in range(1, tenants + 1)
            ]
        case "guardian":
            ready = context.Barrier(tenants)
            everyone = tuple(range(1, tenants + 1))
            workers = [
                context.Process(target=_guardian_worker, args=(inputs, everyone, ready, results))
            ]
        case "process" | "mps" | "mps_static":
            if configuration == "mps_static" and len(partitions) != tenants:
                raise RuntimeError("static MPS needs one partition per tenant")
            if configuration != "mps_static" and partitions:
                raise RuntimeError("MPS partitions were supplied to a non-static run")
            ready = context.Barrier(tenants)
            workers = [
                context.Process(
                    target=_direct_worker,
                    args=(inputs, tenant, ready, results),
                )
                for tenant in range(1, tenants + 1)
            ]
        case "mps_shared":
            if partitions:
                raise RuntimeError("MPS partitions were supplied to a non-static run")
            ready = context.Barrier(tenants)
            closed = context.Barrier(tenants + 1)  # the tenants and the owner
            inbox: mp.Queue[Exported] = context.Queue()
            # the owner starts first; it is a process of the cell, not a tenant
            workers = [
                context.Process(
                    target=_shared_owner, args=(inputs, tenants, inbox, closed, results)
                ),
                *(
                    context.Process(
                        target=_shared_tenant, args=(tenant, inbox, ready, closed, results)
                    )
                    for tenant in range(1, tenants + 1)
                ),
            ]
        case _ as never:
            assert_never(never)
    # an MPS client names its static partition before its first CUDA call, which
    # importing the service makes as the spawned process starts: the process
    # inherits the variable from this one
    if partitions and "CUDA_MPS_SM_PARTITION" in os.environ:
        raise RuntimeError("CUDA_MPS_SM_PARTITION was already set")
    with Sampler(device) as sampler:
        for number, worker in enumerate(workers):
            if partitions:
                os.environ["CUDA_MPS_SM_PARTITION"] = partitions[number]
            try:
                worker.start()
            finally:
                os.environ.pop("CUDA_MPS_SM_PARTITION", None)
        raw = _collect(results, workers, len(workers), f"{configuration}/{scale}/{tenants}")
    reports = []
    for item in raw:
        if not isinstance(item, WorkerReport):
            raise RuntimeError(f"a worker returned malformed evidence: {item!r}")
        reports.append(item)
    streams = sorted(
        (stream for report in reports for stream in report.streams),
        key=lambda stream: stream.tenant,
    )
    if len(streams) != tenants:
        raise RuntimeError(f"expected {tenants} tenant streams, found {streams}")
    for stream in streams:
        if len(stream.outputs) != len(stream.q6):
            raise RuntimeError(f"missing query output for tenant {stream.tenant}")
    samples = sorted(value for stream in streams for value in stream.q6)
    completions = [len(stream.q6) for stream in streams]
    queries = sum(completions)
    fairness = queries**2 / (len(completions) * sum(value * value for value in completions))
    services = [report.service for report in reports if report.service is not None]
    expected_services = {"service": 1, "service_per_tenant": tenants}.get(configuration, 0)
    if len(services) != expected_services:
        raise RuntimeError(f"{len(services)} services' evidence for {configuration}")
    guardians = [report.guardian for report in reports if report.guardian is not None]
    if len(guardians) != (1 if configuration == "guardian" else 0):
        raise RuntimeError(f"{len(guardians)} guardian layouts for {configuration}")
    shared = [report.shared for report in reports if report.shared is not None]
    if len(shared) != (1 if configuration == "mps_shared" else 0):
        raise RuntimeError(f"{len(shared)} owners' evidence for {configuration}")
    return {
        "configuration": configuration,
        "scale": scale,
        "rows": (inputs / "quantity.bin").stat().st_size // 8,
        "inputs": str(inputs),
        "tenants": tenants,
        "queries": queries,
        "queries_per_second": queries / DURATION,
        "jain_fairness": fairness,
        "per_tenant": [
            {
                "tenant": stream.tenant,
                "queries": len(stream.q6),
                "request": stream.request,
                "outputs": stream.outputs,
                "latency_us_samples": [value * 1e6 for value in stream.q6],
            }
            for stream in streams
        ],
        "latency_us": _latency(samples),
        "latency_us_samples": [value * 1e6 for value in samples],
        "memory_mib": sampler.peak - baseline,
        "mps": mps,
        "services": services,
        "guardian": guardians[0] if guardians else None,
        "shared": shared[0] if shared else None,
    }


Cell = tuple[Configuration, float, int, int]  # configuration, scale factor, tenants, repeat


def _cell_key(run: dict[str, object]) -> tuple[object, object, object, object]:
    return run.get("configuration"), run.get("scale"), run.get("tenants"), run.get("repeat")


def cells(
    configurations: tuple[Configuration, ...],
    scales: tuple[float, ...],
    tenant_counts: tuple[int, ...],
    repeats: int,
) -> list[Cell]:
    """Every cell of the run, each configuration, scale factor, and tenant count
    `repeats` times, in an order shuffled with `ORDER_SEED`."""
    ordered = [
        (configuration, scale, tenants, repeat)
        for repeat in range(repeats)
        for configuration in configurations
        for scale in scales
        for tenants in tenant_counts
    ]
    random.Random(ORDER_SEED).shuffle(ordered)
    return ordered


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _record(
    *,
    complete: bool,
    started_at_utc: str,
    environment: dict[str, object],
    source_texts: dict[str, str],
    configurations: tuple[Configuration, ...],
    scales: tuple[float, ...],
    tenant_counts: tuple[int, ...],
    repeats: int,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": 4,
        "experiment": "tenants issuing TPC-H Q6 through the service and per-tenant baselines",
        "complete": complete,
        "started_at_utc": started_at_utc,
        "recorded_at_utc": datetime.now(UTC).isoformat() if complete else None,
        "environment": environment,
        "sources": source_texts,
        "workload": {
            "query": "TPC-H Q6",
            "data": "deterministic TPC-H-distribution generator",
            "seed": 5,
            "operator": "recorded Sonnet q6_sum Numba PTX",
            "ptx": MODEL_PTX.read_bytes().decode("ascii"),
            "correctness_checked_on_every_request": True,
            "parameters": asdict(QUERY),
            "input_column_types": INPUT_TYPES,
        },
        "configurations": configurations,
        "scales": scales,
        "tenant_counts": tenant_counts,
        "repeats": repeats,
        "order_seed": ORDER_SEED,
        "warmup_seconds": WARMUP,
        "duration_seconds": DURATION,
        "limitations": [
            "closed-loop Q6 only",
            "no offered-load curve",
            "no MIG baseline",
            "mps_shared launches Q6 directly over a writable CUDA IPC mapping, "
            "without the catalog's per-use check",
            f"each service's pool and the pool of each process, mps, and mps_static "
            f"tenant carry {SLACK >> 20} MiB of slack beyond the table; mps_shared "
            "tenants carry no table pool and hence no slack",
        ],
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        action="append",
        choices=CONFIGURATIONS,
        dest="configurations",
    )
    parser.add_argument("--scale", action="append", type=float, choices=SCALES, dest="scales")
    parser.add_argument("--tenants", action="append", type=int, choices=TENANTS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    configurations = tuple(args.configurations or CONFIGURATIONS)
    scales = tuple(args.scales or SCALES)
    tenant_counts = tuple(args.tenants or TENANTS)
    repeats: int = args.repeats
    if repeats < 1:
        parser.error("repeats must be positive")
    for name, values in (
        ("configurations", configurations),
        ("scales", scales),
        ("tenant counts", tenant_counts),
    ):
        if len(set(values)) != len(values):
            parser.error(f"duplicate {name}")
    environment = _environment()
    source_texts = _source_texts()
    checkpoint = args.output.with_suffix(args.output.suffix + ".partial")
    input_directory = args.output.with_suffix(args.output.suffix + ".inputs") / "tables"
    started_at_utc = datetime.now(UTC).isoformat()
    runs: list[dict[str, object]] = []
    if checkpoint.exists():
        loaded: object = json.loads(checkpoint.read_text())
        if not isinstance(loaded, dict) or loaded.get("complete") is not False:
            raise RuntimeError(f"malformed checkpoint: {checkpoint}")
        expected = {
            "schema": 4,
            "environment": environment,
            "sources": source_texts,
            "configurations": list(configurations),
            "scales": list(scales),
            "tenant_counts": list(tenant_counts),
            "repeats": repeats,
            "warmup_seconds": WARMUP,
            "duration_seconds": DURATION,
        }
        drifted = [key for key, value in expected.items() if loaded.get(key) != value]
        if drifted:
            raise RuntimeError(
                f"checkpoint does not match this run ({', '.join(drifted)}): {checkpoint}"
            )
        raw_runs = loaded.get("runs")
        if not isinstance(raw_runs, list) or not all(isinstance(run, dict) for run in raw_runs):
            raise RuntimeError(f"checkpoint has malformed runs: {checkpoint}")
        runs = raw_runs
        started = loaded.get("started_at_utc")
        if not isinstance(started, str):
            raise RuntimeError(f"checkpoint has no start time: {checkpoint}")
        started_at_utc = started
    completed = {_cell_key(run) for run in runs}
    if len(completed) != len(runs):
        raise RuntimeError(f"checkpoint has duplicate cells: {checkpoint}")
    _save_inputs(input_directory, scales, create=not checkpoint.exists())
    if not checkpoint.exists():
        _write_json(
            checkpoint,
            _record(
                complete=False,
                started_at_utc=started_at_utc,
                environment=environment,
                source_texts=source_texts,
                configurations=configurations,
                scales=scales,
                tenant_counts=tenant_counts,
                repeats=repeats,
                runs=runs,
            ),
        )
    device = _device()
    for cell in cells(configurations, scales, tenant_counts, repeats):
        configuration, scale, tenants, repeat = cell
        if cell in completed:
            print(json.dumps({"resumed": cell}), flush=True)
            continue
        if configuration in ("mps", "mps_shared", "mps_static"):
            baseline = _memory(device)
            with Mps(static_tenants=tenants if configuration == "mps_static" else None) as mps:
                result = run(
                    configuration,
                    scale,
                    tenants,
                    device,
                    inputs=input_directory / f"sf{scale:g}",
                    baseline_memory=baseline,
                    mps=mps.metadata,
                    partitions=mps.partitions,
                )
        else:
            result = run(
                configuration, scale, tenants, device, inputs=input_directory / f"sf{scale:g}"
            )
        if _source_texts() != source_texts:
            raise RuntimeError(f"an evaluation source changed during cell {cell}")
        runs.append({**result, "repeat": repeat})
        completed.add(cell)
        _write_json(
            checkpoint,
            _record(
                complete=False,
                started_at_utc=started_at_utc,
                environment=environment,
                source_texts=source_texts,
                configurations=configurations,
                scales=scales,
                tenant_counts=tenant_counts,
                repeats=repeats,
                runs=runs,
            ),
        )
        shown = {
            k: v
            for k, v in result.items()
            if k not in ("latency_us_samples", "services", "per_tenant")
        }
        print(json.dumps({**shown, "repeat": repeat}), flush=True)
    if _source_texts() != source_texts:
        raise RuntimeError("an evaluation source changed during the run")
    _save_inputs(input_directory, scales, create=False)
    record = _record(
        complete=True,
        started_at_utc=started_at_utc,
        environment=environment,
        source_texts=source_texts,
        configurations=configurations,
        scales=scales,
        tenant_counts=tenant_counts,
        repeats=repeats,
        runs=runs,
    )
    _write_json(args.output, record)
    checkpoint.unlink()


if __name__ == "__main__":
    main()
