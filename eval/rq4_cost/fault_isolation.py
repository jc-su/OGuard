"""RQ1: what one tenant's faulting kernel does to the other tenants on one GPU.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq4_cost.fault_isolation \\
        --output eval/records/fault_isolation_<date>.json

One attacker and `VICTIMS` victim tenants share one RTX A6000. Each victim
runs the recorded Sonnet TPC-H Q6 kernel in a loop over its own table and
checks every result against a NumPy reference. `FAULT_AT` seconds into the
window the attacker runs the archived wild-address kernel of the paired
replay (`eval/records/e8_service_2026-09-29.json`), which reads
`source[1 << 32]`, far outside any allocation. The window runs for
`RUNTIME` seconds, so each victim has `RUNTIME - FAULT_AT` seconds after the
fault to continue or to fail. The configurations:

* ``shared_unchecked``: the attacker and the victims share one process, one
  CUDA context, and one pool, as an engine that launches a tenant's kernel in
  its own context would. The attacker launches the wild kernel directly with
  the driver, with no gate.
* ``process``: every tenant, the attacker too, has its own process, context,
  and pool, and the attacker launches the wild kernel directly.
* ``mps``: the ``process`` baseline under one default CUDA MPS server.
* ``service``: the victims and the attacker are client processes of one
  OpGate service (design decision 34). The attacker submits the wild PTX,
  which the gate refuses as written and the service registers with the
  guard pass's check of the far read (design decision 38), and runs its own
  accepted Q6; at the fault time it launches the wild kernel, whose check
  fails, so the read does not run and the service answers "check_failed"
  to the attacker alone. The victims run Q6 throughout.

For each tenant the record keeps every completed query's time and revenue,
whether each equals the reference, the longest gap between completions, the
first raw error the tenant saw, and whether its process or context ended.
A victim that continues with correct results after the fault is isolated
from it. The timings are not a controlled throughput measurement: the point
is which tenants keep running, not how fast.

With ``--no-attacker`` the attacker does nothing, and with
``--configuration`` one configuration runs, which validate the victim and
service paths without a fault or an MPS daemon.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import platform
import secrets
import shutil
import struct
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import Literal, assert_never

import numpy as np

from eval import RECORDS, ROOT
from eval.baselines.guardian import patch_ptx
from eval.corpus.adapter import binding_of
from eval.direct import UNSIGNED, Direct, checked_kernels
from eval.rq1_safety.service_boundary import archived_modules
from eval.rq4_cost.consolidation import (
    GUARDIAN_FILLER,
    MAXIMUM_MESSAGE,
    MODEL_PTX,
    TENANT_BYTES,
    Mps,
    _collect,
    _environment,
    _guardian_place,
    _pool,
    _read_inputs,
    _reference,
    _require_ok,
    _save_inputs,
    _scalars,
    _upload,
    guardian_layout,
    guardian_partition_bytes,
    guardian_partition_size,
)
from eval.service_plans import NODES, SCHEMAS, VERIFICATION_WORKERS
from eval.tpch.pipelines import Columns, Session, load
from opgate.binding import ElementType
from opgate.catalog import Fill, Launch, Step, TenantId
from opgate.executor import Executor, Module, Run
from opgate.executor.driver import DriverError, address, call, compile_ptx, driver, integer
from opgate.frontends import Target
from opgate.gate import formal_dir
from opgate.hosts.service import Client, Limits, Principal, Service

Configuration = Literal["shared_unchecked", "process", "mps", "service", "guardian"]
CONFIGURATIONS: tuple[Configuration, ...] = (
    "shared_unchecked",
    "process",
    "mps",
    "service",
    "guardian",
)
VICTIMS = 3
ATTACKER_TENANT = 9000
SCALE = 0.1
RUNTIME = 6.0
FAULT_AT = 3.0
SETUP_TIMEOUT = 300.0
JOIN_TIMEOUT = 60.0
WILD_INDEX = 1 << 32
ARCHIVE = RECORDS / "e8_service_2026-09-29.json"
# the attacker's out-of-bounds write of the guardian configuration: a known value
# at an index far past its buffer, which Guardian's fence confines to its partition
ATTACKER_VALUE = 0x5AFEDEAD5AFEDEAD
_ATTACKER_WRITE = (
    'extern "C" __global__ void attacker_write('
    "long long* buffer, long long value, long long index) { buffer[index] = value; }\n"
)
SOURCES = (
    "eval/rq4_cost/fault_isolation.py",
    "eval/rq4_cost/consolidation.py",
    "eval/baselines/guardian.py",
    "eval/direct.py",
    "eval/rq1_safety/service_boundary.py",
    "eval/service_plans.py",
    "eval/tpch/pipelines.py",
)


@dataclass(frozen=True)
class Completion:
    """One completed query: seconds since the window's start, and its result."""

    at: float
    revenue: float


@dataclass(frozen=True)
class Outcome:
    """One tenant's record over the window."""

    tenant: int
    role: Literal["victim", "attacker"]
    completions: list[Completion]
    correct: bool
    error: str | None
    ended: bool
    note: str


def _wild_module() -> tuple[str, str]:
    """The wild-address kernel's PTX text and entry name, from the archive."""
    text = archived_modules(ARCHIVE)["wild"]
    entries = [line for line in text.splitlines() if ".entry" in line]
    if len(entries) != 1:
        raise RuntimeError(f"the wild module has {len(entries)} entry lines")
    name = entries[0].split(".entry", 1)[1].split("(", 1)[0].strip()
    if not name:
        raise RuntimeError("the wild module's entry has no name")
    return text, name


def _correct(completions: list[Completion], expected: float) -> bool:
    """Whether every completed query returned the reference revenue."""
    return all(
        abs(completion.revenue - expected) <= 1e-6 * max(1.0, abs(expected))
        for completion in completions
    )


def _max_gap(completions: list[Completion], end: float) -> float:
    """The longest gap in seconds between the window's start, each completion, and
    `end`; a tenant stalled by a fault has one long gap."""
    times = [0.0, *(completion.at for completion in completions), end]
    return max(later - earlier for earlier, later in zip(times, times[1:], strict=False))


def _after(completions: list[Completion], when: float) -> int:
    """How many queries completed after `when` seconds."""
    return sum(1 for completion in completions if completion.at >= when)


def _numba_array(pointer: int, count: int, itemsize: int) -> list[int]:
    """Numba's array ABI for a contiguous one-dimensional array: meminfo, parent,
    nitems, itemsize, data, shape, stride, as seven 64-bit values."""
    return [0, 0, count, itemsize, pointer, count, itemsize]


def _send(results: mp.Queue[list[Outcome]], outcomes: list[Outcome], dead: bool) -> None:
    """Send a worker's outcomes. A worker whose context a fault ended sends them
    and ends its process here, inside its executor's block, whose cleanup would
    fail on the dead context."""
    results.put(outcomes)
    if dead:
        results.close()
        results.join_thread()
        os._exit(0)


def _launch_wild(text: str, name: str) -> None:
    """Load `text` and launch its wild-address kernel on the current context, on a
    source of one element with the index `WILD_INDEX`. The read faults, which
    `cuCtxSynchronize` reports and which ends the context."""
    (handle,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
    try:
        (function,) = call(driver.cuModuleGetFunction(handle, name.encode("ascii")))
        (source,) = call(driver.cuMemAlloc(8))
        (output,) = call(driver.cuMemAlloc(8))
        values = [
            *_numba_array(address(source), 1, 8),
            *_numba_array(address(output), 1, 8),
            WILD_INDEX,
        ]
        types = tuple(ctypes.c_uint64 for _ in values)
        call(driver.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, (tuple(values), types), 0))
        call(driver.cuCtxSynchronize())
    finally:
        # a faulted context rejects the unload too; the process ends regardless
        driver.cuModuleUnload(handle)


def _q6_run(session: Session, module: Module, columns: Columns) -> Run:
    """The prepared Q6 step over the tenant's columns and a fresh result buffer."""
    with session.executor.tenant(session.tenant):
        result = session.memory.allocate(8)
        launch = Launch(
            module.operator,
            (max(1, -(-columns.rows // 256)), 1, 1),
            (columns.shipdate, columns.discount, columns.quantity, columns.price, result),
            (columns.rows,),
            _scalars(),
            False,
        )
        step = Step(session.tenant, (Fill(result, ElementType(8, False), 0), launch))
        return session.executor.prepare(step, result)


def _q6_once(session: Session, run: Run) -> float:
    """One use of a prepared Q6 step; the revenue it writes."""
    executor = session.executor
    with executor.tenant(session.tenant):
        executor.submit(run, _scalars())
    while not executor.finished(run):
        pass
    (value,) = struct.unpack("<d", executor.result(run))
    return value


def _victim_loop(
    session: Session, run: Run, expected: float, start: float, end: float
) -> tuple[list[Completion], str | None]:
    """Run Q6 until `end` seconds past `start`; the completions and the first raw
    error, if a query raised, as a faulted context does."""
    completions: list[Completion] = []
    while time.perf_counter() < end:
        try:
            revenue = _q6_once(session, run)
        except (DriverError, RuntimeError) as error:
            return completions, str(error)
        completions.append(Completion(time.perf_counter() - start, revenue))
    return completions, None


def _outcome(
    tenant: int,
    role: Literal["victim", "attacker"],
    completions: list[Completion],
    expected: float,
    error: str | None,
    ended: bool,
    note: str,
) -> Outcome:
    return Outcome(
        tenant=tenant,
        role=role,
        completions=completions,
        correct=_correct(completions, expected),
        error=error,
        ended=ended,
        note=note,
    )


def _shared_unchecked(
    inputs: Path, wild: tuple[str, str] | None, results: mp.Queue[list[Outcome]]
) -> None:
    """The attacker and victims in one process and one context. Victims run Q6
    until the fault, the attacker faults the context, and the victims run again
    and fail on the dead context."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    pool = _pool(rows) * (VICTIMS + 1)
    text = MODEL_PTX.read_text()
    outcomes: list[Outcome] = []
    faulted = False
    with Executor(0, pool) as executor:
        module = executor.register(text, binding_of("sonnet", "q6_sum", "numba", text))
        sessions = []
        runs = []
        for tenant in range(1, VICTIMS + 1):
            with executor.tenant(TenantId(tenant)):
                executor.reserve(_pool(rows))
            session = Session(executor, TenantId(tenant), executor)
            columns = load(session, lineitem)
            sessions.append(session)
            runs.append(_q6_run(session, module, columns))
        start = time.perf_counter()
        fault_at = start + FAULT_AT
        end = start + RUNTIME
        completions: list[list[Completion]] = [[] for _ in sessions]
        while time.perf_counter() < fault_at:
            for index, session in enumerate(sessions):
                revenue = _q6_once(session, runs[index])
                completions[index].append(Completion(time.perf_counter() - start, revenue))
        attacker_error: str | None = None
        if wild is not None:
            try:
                _launch_wild(*wild)
            except DriverError as error:
                attacker_error = str(error)
                faulted = True
        errors: list[str | None] = [None] * len(sessions)
        while time.perf_counter() < end and not all(e is not None for e in errors):
            for index, session in enumerate(sessions):
                if errors[index] is not None:
                    continue
                try:
                    revenue = _q6_once(session, runs[index])
                    completions[index].append(Completion(time.perf_counter() - start, revenue))
                except (DriverError, RuntimeError) as error:
                    errors[index] = str(error)
        for index in range(len(sessions)):
            ended = errors[index] is not None
            note = "stopped on the faulted shared context" if ended else "ran to the end"
            outcomes.append(
                _outcome(
                    index + 1, "victim", completions[index], expected, errors[index], ended, note
                )
            )
        note = "no attacker" if wild is None else "faulted the shared context"
        outcomes.append(
            _outcome(ATTACKER_TENANT, "attacker", [], expected, attacker_error, faulted, note)
        )
        _send(results, outcomes, faulted)


def _victim_worker(
    inputs: Path,
    tenant: int,
    barrier: Barrier,
    results: mp.Queue[list[Outcome]],
) -> None:
    """One victim process: its own context, running Q6 across the window."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    text = MODEL_PTX.read_text()
    with Executor(0, _pool(rows)) as executor:
        with executor.tenant(TenantId(tenant)):
            executor.reserve(_pool(rows))
        session = Session(executor, TenantId(tenant), executor)
        columns = load(session, lineitem)
        run = _q6_run(
            session,
            module=executor.register(text, binding_of("sonnet", "q6_sum", "numba", text)),
            columns=columns,
        )
        barrier.wait()
        start = time.perf_counter()
        completions, error = _victim_loop(session, run, expected, start, start + RUNTIME)
        ended = error is not None
        note = "process per tenant isolated the fault" if not ended else "stopped on an error"
        outcome = _outcome(tenant, "victim", completions, expected, error, ended, note)
        _send(results, [outcome], ended)


def _attacker_worker(
    wild: tuple[str, str], barrier: Barrier, results: mp.Queue[list[Outcome]]
) -> None:
    """One attacker process: wait the fault delay, then fault its own context."""
    lineitem_pool = _pool(1)
    error: str | None = None
    faulted = False
    with Executor(0, lineitem_pool):
        barrier.wait()
        time.sleep(FAULT_AT)
        try:
            _launch_wild(*wild)
        except DriverError as caught:
            error = str(caught)
            faulted = True
        note = "faulted its own context"
        outcome = _outcome(ATTACKER_TENANT, "attacker", [], 0.0, error, faulted, note)
        _send(results, [outcome], faulted)


def _service_request(result: str, rows: int) -> dict[str, object]:
    """The Q6 prepare request over the loader's shared columns and a result."""
    columns = ("shipdate", "discount", "quantity", "extendedprice")
    return {
        "operator": "",
        "grid": [max(1, -(-rows // 256)), 1, 1],
        "buffers": [*columns, result],
        "sizes": [rows],
        "scalars": list(_scalars()),
        "fills": [{"buffer": result, "element": {"width": 8, "signed": False}, "value": 0}],
        "read": result,
    }


def _service_victim(
    socket_path: str,
    token: str,
    tenant: int,
    handles: dict[str, str],
    rows: int,
    expected: float,
    barrier: Barrier,
    results: mp.Queue[list[Outcome]],
) -> None:
    text = MODEL_PTX.read_text()
    with Client(socket_path, token, message_bytes=MAXIMUM_MESSAGE) as client:
        operator = _require_ok(
            client.register((text, {"kind": "tpch_q6_sum"}))[0], "victim registration"
        ).text("operator")
        result = _require_ok(client.request("allocate", nbytes=8), "victim result").text("buffer")
        request = _service_request(result, rows)
        request["operator"] = operator
        request["buffers"] = [
            handles[name] for name in ("shipdate", "discount", "quantity", "extendedprice")
        ] + [result]
        prepared = client.prepare(**request)
        scalars = _scalars()
        barrier.wait()
        start = time.perf_counter()
        end = start + RUNTIME
        completions: list[Completion] = []
        error: str | None = None
        with client.data_plane() as mailbox:
            while time.perf_counter() < end:
                try:
                    (revenue,) = struct.unpack("<d", mailbox.run(prepared, scalars))
                except (RuntimeError, OSError) as caught:
                    error = str(caught)
                    break
                completions.append(Completion(time.perf_counter() - start, revenue))
    ended = error is not None
    note = "the service kept the victim running" if not ended else "stopped on an error"
    results.put([_outcome(tenant, "victim", completions, expected, error, ended, note)])


def _service_attacker(
    socket_path: str,
    token: str,
    handles: dict[str, str],
    rows: int,
    expected: float,
    wild_text: str,
    barrier: Barrier,
    results: mp.Queue[list[Outcome]],
) -> None:
    with Client(socket_path, token, message_bytes=MAXIMUM_MESSAGE) as client:
        (wild,) = client.register((wild_text, {"kind": "read_one_i64", "source_index": WILD_INDEX}))
        accepted = wild.raw("ok") is True
        registration = wild.record()
        # the wild kernel's one-element source and its output, of the attacker
        source, output = (
            _require_ok(client.request("allocate", nbytes=8), "attacker wild buffer").text("buffer")
            for _ in range(2)
        )
        text = MODEL_PTX.read_text()
        operator = _require_ok(
            client.register((text, {"kind": "tpch_q6_sum"}))[0], "attacker Q6 registration"
        ).text("operator")
        result = _require_ok(client.request("allocate", nbytes=8), "attacker result").text("buffer")
        request = _service_request(result, rows)
        request["operator"] = operator
        request["buffers"] = [
            handles[name] for name in ("shipdate", "discount", "quantity", "extendedprice")
        ] + [result]
        prepared = client.prepare(**request)
        scalars = _scalars()
        barrier.wait()
        start = time.perf_counter()
        end = start + RUNTIME
        completions: list[Completion] = []
        launched: dict[str, object] | None = None
        with client.data_plane() as mailbox:
            while time.perf_counter() < end:
                if accepted and launched is None and time.perf_counter() >= start + FAULT_AT:
                    launched = client.request(
                        "run_operator",
                        operator=wild.text("operator"),
                        source=source,
                        output=output,
                        n=1,
                    ).record()
                (revenue,) = struct.unpack("<d", mailbox.run(prepared, scalars))
                completions.append(Completion(time.perf_counter() - start, revenue))
    if not accepted:
        note = f"the service refused the wild kernel: {registration}"
    else:
        note = (
            f"the service registered the wild kernel with checks: {registration}; "
            f"its launch at the fault time answered {launched}"
        )
    error = launched.get("error") if launched is not None else None
    ended = isinstance(error, dict) and error.get("kind") == "fatal"
    results.put([_outcome(ATTACKER_TENANT, "attacker", completions, expected, None, ended, note)])


def _service_cell(
    inputs: Path, wild: tuple[str, str] | None
) -> tuple[list[Outcome], dict[str, object]]:
    """Start a service, upload and grant the shared columns, run the victims and
    the attacker as clients, and return their outcomes and the service telemetry."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    tenants = tuple(range(1, VICTIMS + 1))
    names = ("loader", *(f"tenant-{t}" for t in tenants), "attacker")
    tokens = {name: secrets.token_hex(16) for name in names}
    principals = (
        Principal(
            "loader",
            tokens["loader"],
            TenantId(10_000),
            administrator=True,
            reservation=_pool(rows),
        ),
        *(Principal(f"tenant-{t}", tokens[f"tenant-{t}"], TenantId(t)) for t in tenants),
        Principal("attacker", tokens["attacker"], TenantId(ATTACKER_TENANT)),
    )
    context = mp.get_context("spawn")
    results: mp.Queue[list[Outcome]] = context.Queue()
    pool = _pool(rows) * (VICTIMS + 1)
    wild_text = wild[0] if wild is not None else None
    participants = VICTIMS + (1 if wild_text is not None else 0)
    barrier = context.Barrier(participants)
    with tempfile.TemporaryDirectory(prefix="opgate-fault-service-") as temporary:
        socket_path = Path(temporary) / "service.sock"
        with Service(
            socket_path,
            principals,
            device=0,
            pool_bytes=pool,
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
                    name: _upload(loader, getattr(lineitem, name))
                    for name in ("shipdate", "discount", "quantity", "extendedprice", "orderkey")
                }
                for principal in (*(f"tenant-{t}" for t in tenants), "attacker"):
                    for handle in handles.values():
                        _require_ok(
                            loader.request("grant_read", buffer=handle, principal=principal),
                            "service grant",
                        )
                workers: list[BaseProcess] = [
                    context.Process(
                        target=_service_victim,
                        args=(
                            str(socket_path),
                            tokens[f"tenant-{t}"],
                            t,
                            handles,
                            rows,
                            expected,
                            barrier,
                            results,
                        ),
                    )
                    for t in tenants
                ]
                if wild_text is not None:
                    workers.append(
                        context.Process(
                            target=_service_attacker,
                            args=(
                                str(socket_path),
                                tokens["attacker"],
                                handles,
                                rows,
                                expected,
                                wild_text,
                                barrier,
                                results,
                            ),
                        )
                    )
                for worker in workers:
                    worker.start()
                gathered = _collect(results, workers, len(workers), "service clients")
                outcomes = [outcome for item in gathered for outcome in item]
                telemetry = _require_ok(loader.request("telemetry"), "service telemetry").fields(
                    "telemetry"
                )
                return outcomes, {"telemetry": telemetry.record()}


def _compute_arch(device: object) -> str:
    """The PTX target of the device, such as `compute_86`, for NVRTC."""
    attribute = driver.CUdevice_attribute
    major = integer(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    )
    minor = integer(
        driver.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
    )
    return f"compute_{major}{minor}"


def _attacker_module(arch: str) -> tuple[str, str, int]:
    """The attacker's out-of-bounds write kernel, patched by Guardian: the fenced
    PTX text, its entry name, and the accesses that the fence instruments."""
    ptx = compile_ptx(_ATTACKER_WRITE, "attacker_write", arch).decode("ascii")
    fenced, report = patch_ptx(ptx)
    return fenced, "attacker_write", report.instrumented_accesses


@dataclass(frozen=True)
class _Fenced:
    """One tenant's fenced Q6 launch on its own stream: the fenced function, the
    launch geometry, the marshaled parameters with the partition's base and mask
    appended, the stream, a pinned host copy of the result, and the result's
    device address."""

    function: object
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    values: tuple[int, ...]
    types: tuple[object, ...]
    stream: object
    host: int
    result: int


def _guardian_victim(
    plan: _Fenced, start: float, end: float
) -> tuple[list[Completion], str | None]:
    """Run the fenced Q6 until `end`; the completions and the first raw error."""
    completions: list[Completion] = []
    while time.perf_counter() < end:
        try:
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
        except DriverError as error:
            return completions, str(error)
        (value,) = struct.unpack("<d", ctypes.string_at(plan.host, 8))
        completions.append(Completion(time.perf_counter() - start, value))
    return completions, None


def _read_word(address_value: int) -> int:
    """The 8-byte word at a pool address, read synchronously."""
    data = bytearray(8)
    call(driver.cuMemcpyDtoH(data, driver.CUdeviceptr(address_value), 8))
    (word,) = struct.unpack("<Q", data)
    return word


def _guardian_cell(inputs: Path, attack: bool) -> tuple[list[Outcome], dict[str, object]]:
    """One process, one shared context: each victim's Q6 table in its own
    power-of-two aligned partition, the recorded Sonnet Q6 kernel patched by
    Guardian and launched with each partition's base and mask, the victims on
    separate streams. The attacker holds partition 0; at the fault time it
    launches a Guardian-fenced out-of-bounds write whose unfenced target is a
    victim's partition and whose fence confines it to the attacker's own
    partition. Record whether the victims continued, their stall and queries
    lost, and where the write landed."""
    lineitem = _read_inputs(inputs)
    rows = len(lineitem.quantity)
    expected = _reference(lineitem)
    q6_text = MODEL_PTX.read_text()
    q6_binding = binding_of("sonnet", "q6_sum", "numba", q6_text)
    fenced_q6, q6_report = patch_ptx(q6_text)
    tenants = VICTIMS + 1  # partition 0 is the attacker, 1..VICTIMS the victims
    size = guardian_partition_size(guardian_partition_bytes(rows))
    pool_bytes = size * (tenants + 1)
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    arch = _compute_arch(device)
    fenced_attacker, attacker_name, attacker_fences = _attacker_module(arch)
    outcomes: list[Outcome] = []
    evidence: dict[str, object] = {}
    errors: list[BaseException] = []
    with Executor(0, pool_bytes) as executor:
        q6_module = executor.register(q6_text, q6_binding)
        first, bases = guardian_layout(executor.pool_address, size, tenants)
        # each tenant's reservation is its partition: a filler tenant reserves the
        # bytes before the first partition, then the attacker and the victims in order
        if first > executor.pool_address:
            with executor.tenant(GUARDIAN_FILLER):
                executor.reserve(first - executor.pool_address)
        for tenant_id in (ATTACKER_TENANT, *range(1, VICTIMS + 1)):
            with executor.tenant(TenantId(tenant_id)):
                executor.reserve(size)
        (q6_handle,) = call(driver.cuModuleLoadData(fenced_q6.encode("ascii") + b"\0"))
        (atk_handle,) = call(driver.cuModuleLoadData(fenced_attacker.encode("ascii") + b"\0"))
        (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
        created: list[object] = []
        hosts: list[int] = []
        try:
            (q6_function,) = call(
                driver.cuModuleGetFunction(q6_handle, q6_binding.kernel.encode("ascii"))
            )
            (atk_function,) = call(
                driver.cuModuleGetFunction(atk_handle, attacker_name.encode("ascii"))
            )
            direct = Direct(executor)
            count = len(q6_binding.parameters)
            victims: list[tuple[int, _Fenced]] = []
            for position in range(VICTIMS):
                tenant = TenantId(position + 1)
                base = bases[position + 1]
                placed, result, offsets = _guardian_place(
                    executor, tenant, lineitem, base - executor.pool_address, size
                )
                launch = Launch(
                    q6_module.operator,
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
                values = (*direct.values(kernel, q6_module)[:count], base, size - 1)
                widths = (*q6_module.widths[:count], 8, 8)
                (stream,) = call(driver.cuStreamCreate(0))
                created.append(stream)
                (host,) = call(driver.cuMemAllocHost(8))
                hosts.append(address(host))
                victims.append(
                    (
                        position + 1,
                        _Fenced(
                            q6_function,
                            kernel.grid,
                            q6_module.binding.block,
                            values,
                            tuple(UNSIGNED[width] for width in widths),
                            stream,
                            address(host),
                            executor.pool_address + offsets["result"],
                        ),
                    )
                )
            attacker_base = bases[0]
            attacker_offset = attacker_base - executor.pool_address
            with executor.tenant(TenantId(ATTACKER_TENANT)):
                placed_sink = executor.catalog.buffer(executor.allocate(8))
                if placed_sink is None or placed_sink.offset != attacker_offset:
                    raise RuntimeError("the attacker's buffer does not start its partition")
            sink = executor.pool_address + attacker_offset
            index = size // 8  # buffer[index] is attacker_base + size: a victim's partition
            unfenced = sink + index * 8
            fenced_target = (unfenced & (size - 1)) | attacker_base
            victim_base = bases[1]
            (atk_stream,) = call(driver.cuStreamCreate(0))
            created.append(atk_stream)
            atk_values = (sink, ATTACKER_VALUE, index, attacker_base, size - 1)
            atk_types = tuple(UNSIGNED[8] for _ in range(5))
            call(driver.cuMemsetD8(driver.CUdeviceptr(sink), 0, 8))
            call(driver.cuCtxSynchronize())
            sink_before, victim_before = _read_word(sink), _read_word(victim_base)

            clock: dict[str, float] = {}
            parties = VICTIMS + (1 if attack else 0)
            barrier = threading.Barrier(
                parties, action=lambda: clock.__setitem__("start", time.perf_counter())
            )
            results_by_position: list[tuple[list[Completion], str | None] | None] = [None] * VICTIMS

            def victim_thread(position: int, plan: _Fenced) -> None:
                try:
                    call(driver.cuCtxSetCurrent(context))
                    barrier.wait()
                    start = clock["start"]
                    results_by_position[position] = _guardian_victim(plan, start, start + RUNTIME)
                except BaseException as error:  # noqa: BLE001 - reported to the main thread
                    errors.append(error)

            launched: dict[str, object] = {}

            def attacker_thread() -> None:
                try:
                    call(driver.cuCtxSetCurrent(context))
                    barrier.wait()
                    start = clock["start"]
                    while time.perf_counter() < start + FAULT_AT:
                        pass
                    call(
                        driver.cuLaunchKernel(
                            atk_function,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0,
                            atk_stream,
                            (atk_values, atk_types),
                            0,
                        )
                    )
                    call(driver.cuStreamSynchronize(atk_stream))
                    launched["at"] = time.perf_counter() - start
                except BaseException as error:  # noqa: BLE001 - reported to the main thread
                    errors.append(error)

            workers = [
                threading.Thread(target=victim_thread, args=(position, plan))
                for position, plan in enumerate(plan for _, plan in victims)
            ]
            if attack:
                workers.append(threading.Thread(target=attacker_thread))
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            call(driver.cuCtxSetCurrent(context))
            call(driver.cuCtxSynchronize())
            if errors:
                raise errors[0]
            sink_after, victim_after = _read_word(sink), _read_word(victim_base)
            for position, (tenant, _) in enumerate(victims):
                found = results_by_position[position]
                if found is None:
                    raise RuntimeError(f"victim {tenant} produced no outcome")
                completions, error = found
                ended = error is not None
                note = (
                    "the guardian fence confined the attacker's write"
                    if not ended
                    else "stopped on an error"
                )
                outcomes.append(
                    _outcome(tenant, "victim", completions, expected, error, ended, note)
                )
            if attack:
                note = (
                    f"the attacker's out-of-bounds write of index {index} would reach "
                    f"{unfenced:#x} in victim 1's partition; the fence sent it to {fenced_target:#x} "
                    "in the attacker's own partition"
                )
                outcomes.append(
                    _outcome(ATTACKER_TENANT, "attacker", [], expected, None, False, note)
                )
            evidence = {
                "guardian": {
                    "pool_bytes": pool_bytes,
                    "pool_address": executor.pool_address,
                    "first_partition": first,
                    "partition_bytes": size,
                    "mask": size - 1,
                    "attacker_base": attacker_base,
                    "victim_base": victim_base,
                    "write_value": ATTACKER_VALUE,
                    "write_index": index,
                    "unfenced_target": unfenced,
                    "fenced_target": fenced_target,
                    "sink_before": sink_before,
                    "sink_after": sink_after,
                    "victim_word_before": victim_before,
                    "victim_word_after": victim_after,
                    "fenced_accesses": attacker_fences,
                    "q6_fenced_accesses": q6_report.instrumented_accesses,
                    "launch": launched,
                    "ptx": fenced_attacker,
                }
            }
        finally:
            for stream in created:
                call(driver.cuStreamDestroy(stream))
            for host_address in hosts:
                call(driver.cuMemFreeHost(host_address))
            call(driver.cuModuleUnload(q6_handle))
            call(driver.cuModuleUnload(atk_handle))
            call(driver.cuDevicePrimaryCtxRelease(device))
    return outcomes, evidence


def _direct_cell(
    configuration: Literal["process", "mps"],
    inputs: Path,
    wild: tuple[str, str] | None,
    partitions: tuple[str, ...],
) -> list[Outcome]:
    """The process and MPS configurations: a victim process per tenant and an
    optional attacker process, each with its own context."""
    context = mp.get_context("spawn")
    results: mp.Queue[list[Outcome]] = context.Queue()
    participants = VICTIMS + (1 if wild is not None else 0)
    barrier = context.Barrier(participants)
    workers: list[BaseProcess] = [
        context.Process(target=_victim_worker, args=(inputs, tenant, barrier, results))
        for tenant in range(1, VICTIMS + 1)
    ]
    if wild is not None:
        workers.append(context.Process(target=_attacker_worker, args=(wild, barrier, results)))
    # an MPS client names its static partition before its first CUDA call, which
    # importing the service makes as the spawned process starts: the process
    # inherits the variable from this one
    if partitions and "CUDA_MPS_SM_PARTITION" in os.environ:
        raise RuntimeError("CUDA_MPS_SM_PARTITION was already set")
    for number, worker in enumerate(workers):
        if partitions:
            os.environ["CUDA_MPS_SM_PARTITION"] = partitions[number]
        try:
            worker.start()
        finally:
            os.environ.pop("CUDA_MPS_SM_PARTITION", None)
    gathered = _collect(results, workers, len(workers), f"{configuration} workers")
    return [outcome for item in gathered for outcome in item]


def run(
    configuration: Configuration, inputs: Path, wild: tuple[str, str] | None
) -> dict[str, object]:
    """One configuration: the tenants' outcomes and any service evidence."""
    service: dict[str, object] | None = None
    match configuration:
        case "shared_unchecked":
            spawn = mp.get_context("spawn")
            results: mp.Queue[list[Outcome]] = spawn.Queue()
            worker = spawn.Process(target=_shared_unchecked, args=(inputs, wild, results))
            worker.start()
            gathered = _collect(results, [worker], 1, "shared context")
            outcomes = [outcome for item in gathered for outcome in item]
        case "process":
            outcomes = _direct_cell("process", inputs, wild, ())
        case "mps":
            partitions_needed = VICTIMS + (1 if wild is not None else 0)
            with Mps(static_tenants=partitions_needed) as mps:
                try:
                    outcomes = _direct_cell("mps", inputs, wild, mps.partitions)
                finally:
                    # the daemon's logs, beside the record's inputs, before the
                    # daemon's directory goes, whether the cell ends or fails
                    shutil.copytree(mps.directory / "log", inputs.parent.parent / "mps_log")
                service = {"mps": mps.metadata}
        case "service":
            outcomes, service = _service_cell(inputs, wild)
        case "guardian":
            outcomes, service = _guardian_cell(inputs, wild is not None)
        case _ as never:
            assert_never(never)
    return {
        "configuration": configuration,
        "attacker": wild is not None,
        "victims": VICTIMS,
        "runtime_seconds": RUNTIME,
        "fault_at_seconds": FAULT_AT,
        "tenants": [
            {
                **{k: v for k, v in asdict(outcome).items() if k != "completions"},
                "completions": [asdict(c) for c in outcome.completions],
                "completed_after_fault": _after(outcome.completions, FAULT_AT),
                "max_gap_seconds": _max_gap(outcome.completions, RUNTIME),
            }
            for outcome in sorted(outcomes, key=lambda o: o.tenant)
        ],
        "service": service,
    }


def _source_texts() -> dict[str, str]:
    sources = {name: ROOT / name for name in SOURCES}
    sources.update((f"formal/{p.name}", p) for p in sorted(formal_dir().glob("*.v")))
    sources["model_q6.ptx"] = MODEL_PTX
    return {name: path.read_text() for name, path in sources.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configuration", action="append", choices=CONFIGURATIONS, dest="configurations"
    )
    parser.add_argument(
        "--no-attacker",
        action="store_true",
        help="run the victims without the fault, to validate the victim path",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    configurations = tuple(args.configurations or CONFIGURATIONS)
    if len(set(configurations)) != len(configurations):
        parser.error("duplicate configuration")
    wild = None if args.no_attacker else _wild_module()
    environment = _environment()
    source_texts = _source_texts()
    input_directory = args.output.with_suffix(args.output.suffix + ".inputs") / "tables"
    _save_inputs(input_directory, (SCALE,), create=True)
    inputs = input_directory / f"sf{SCALE:g}"
    started = datetime.now(UTC).isoformat()
    runs: list[dict[str, object]] = []
    for configuration in configurations:
        result = run(configuration, inputs, wild)
        runs.append(result)
        tenants = result["tenants"]
        if not isinstance(tenants, list):
            raise TypeError("a cell's tenants are a list")
        shown = {
            "configuration": configuration,
            "attacker": result["attacker"],
            "tenants": [
                {
                    "tenant": tenant["tenant"],
                    "role": tenant["role"],
                    "queries": len(tenant["completions"]),
                    "after_fault": tenant["completed_after_fault"],
                    "correct": tenant["correct"],
                    "ended": tenant["ended"],
                }
                for tenant in tenants
            ],
        }
        print(json.dumps(shown), flush=True)
    record = {
        "experiment": "fault isolation between tenants when one tenant's kernel faults",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "environment": environment,
        "started_at_utc": started,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(ARCHIVE),
        "wild_index": WILD_INDEX,
        "scale": SCALE,
        "attacker": wild is not None,
        "configurations": list(configurations),
        "sources": source_texts,
        "runs": runs,
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
