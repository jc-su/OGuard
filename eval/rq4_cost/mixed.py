"""RQ4: tenants that run a mix of model-written query operators, through one service or under MPS.

    OPGATE_FORMAL=<library> CUDA_VISIBLE_DEVICES=<gpu> taskset -c <cores of the GPU's node> \\
        uv run python -m eval.rq4_cost.mixed \\
        --registration eval/records/registration_<date>.json \\
        --guarded eval/records/guarded_<date>.json \\
        --proof-cores <cores of the other node> --output eval/records/mixed_<date>.json

Eight query templates use the Numba kernels that Claude Opus wrote for
filters, joins, aggregations, and string operators (`mix.MIXES`). A template
is one operator, or an operator whose `match` output `masked_sum` sums
`values` over. The gate refuses two of Opus's kernels, `string_contains` and
`groupby_sum`, which the templates `like_sum` and `group_by` use. Five
configurations run the mix:

* ``service``: one service, in which Sonnet's `string_contains` and
  `groupby_sum`, which the gate accepts, take the place of Opus's; every
  template runs Sonnet's or Opus's kernels.
* ``service_refusals``: one service and Opus's kernels as written. A tenant
  registers each of them; the registrations of the two refused kernels are
  answered `refused` and kept in the record with their reasons, and the two
  templates that use them are skipped.
* ``service_fallback``: one service and Opus's kernels as written, as in
  ``service_refusals``; the engine's built-in operator for the task of each
  refused kernel (`eval.rq4_cost.fallback`), a query node of the service,
  takes its place over the same inputs and outputs, so every template runs.
* ``mps``: a process per tenant under default CUDA MPS, each with its own
  copy of the inputs, which launches the operators of ``service_fallback``
  directly: the texts that the service loads, which the process's own
  executor checks and loads, and in place of each refused kernel the same
  built-in operator, which the process runs itself.
* ``mps_shared``: as ``mps``, with one copy of the inputs. One more
  process, the owner, allocates each input in whole 2 MiB units, uploads it,
  and exports it with legacy CUDA IPC (`consolidation.EXPORT_GRANULARITY`);
  each tenant maps every input. After every tenant closed its mappings, the
  owner compares the inputs' bytes with the recorded ones.

In the service, each kernel's operator is one prepared step that a tenant
runs through its session's mailbox, and a built-in operator is a request on
its session's socket; one loader owns the inputs and grants them read-only
to every tenant. Under MPS, a launch takes the parameters that the catalog
checked for the same launch over a layout of the tenant's buffers
(`mix.checked_launch`) and runs on the tenant's own stream, after fills with
the driver's memset. A refused kernel is never loaded and never launched.

The gate's decision on each kernel is computed once per service process and
once per MPS tenant, in parallel on the proof cores, before the measured
window (`decide`); later registrations reuse the gate's cached decisions.
Each decision must equal the one the registration and guarded records give
(`mix.recorded`), and the service's audit must show it for every
registration; a difference stops the run with an error that names the
kernel.

Before its measured window, each tenant runs every template once, reads all
of its outputs, and checks them with the task's reference
(`eval.corpus.workloads.check`). In the window it cycles through the
templates in a closed loop and compares each use's read-back with the
verified use's (`mix.Comparison`); after the window it reads and checks all
outputs again. A cell (configuration, tenants, repeat) runs in fresh
processes, all cells in an order shuffled with `ORDER_SEED`, and samples the
device's memory (`consolidation.Sampler`) and each process's memory
(`ProcessMemory`). Each cell records its queries per second, the median and
99th percentile of the queries' latency, overall and per template, what
served each step, and the built-in operators' share of the queries' time.
The inputs are kept in `<record>.inputs/workloads`. The record keeps the
sources and settings and lists each completed cell with the file of its run
in `<record>.runs`, which is written once, when the cell completes: a run
keeps each tenant's registrations or decisions, its requests, outputs, and
latency samples, and the service's audit and telemetry or the owner's
evidence. A cell's samples take tens of megabytes, so the checkpoint
(`<record>.partial`) lists the runs and holds none of them.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import multiprocessing as mp
import os
import random
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import assert_never

import numpy as np

from eval import ROOT
from eval.corpus.adapter import CONVENTIONS, ELEMENTS, Points, claim_of, directory_of, launch_of
from eval.corpus.workloads import Workload
from eval.direct import UNSIGNED
from eval.rq4_cost.consolidation import (
    EXPORT_GRANULARITY,
    IPC_FLAGS,
    MAXIMUM_MESSAGE,
    SETUP_TIMEOUT,
    Mps,
    Sampler,
    _collect,
    _device,
    _environment,
    _handle_bytes,
    _handle_of,
    _latency,
    _memory,
    _release,
    _require_ok,
    _source_texts,
    _wait_for_closed_sessions,
    _write_json,
)
from eval.rq4_cost.fallback import BUILTINS, NODES, Builtin, CudaArray, DeviceArray, read_back
from eval.rq4_cost.guard_cost import _cores
from eval.rq4_cost.mix import (
    BY_OP,
    CONFIGURATIONS,
    LINK,
    MIXES,
    READS,
    ROWS,
    SEED,
    TASKS,
    Argument,
    BuiltinRequest,
    Configuration,
    Kernel,
    Loop,
    Memset,
    Plan,
    PrepareRequest,
    RecordedDecision,
    Status,
    Task,
    Template,
    Use,
    buffers_of,
    builtin_request,
    builtin_share,
    calls_of,
    check_reads,
    checked_launch,
    closed_loop,
    element_of,
    encoded,
    every_kernel,
    host_of,
    kernels_of,
    label_of,
    linked,
    marshal,
    memset_of,
    placement,
    read_workload,
    recorded,
    request_of,
    runnable_templates,
    save_inputs,
    shared_inputs,
    verify,
)
from eval.service_plans import VERIFICATION_WORKERS
from opgate.binding import Binding
from opgate.catalog import ALIGNMENT, TenantId
from opgate.executor import Executor
from opgate.executor.driver import address, call, driver
from opgate.frontends import Target
from opgate.gate import Accepted, Refused
from opgate.hosts.service import Client, Fields, Limits, OperatorSchema, Principal, Service
from opgate.hosts.service.dataplane import ClientMailbox
from opgate.hosts.service.owner import Checked
from opgate.hosts.service.server import _check

TENANTS = (1, 2, 4, 8, 16)
REPEATS = 4
ORDER_SEED = 37
WARMUP = 3.0
DURATION = 10.0
SCHEMA = 2
# the first chunk of the arena in which a service tenant's built-in operators
# allocate, and the initial RMM pool of an MPS tenant's; RMM grows either pool
# when the operators need more
ARENA_BYTES = 8 << 20
# the shared inputs, about 40 MiB, 16 tenants' outputs, about 10 MiB, and 16 arenas
POOL_BYTES = 256 << 20
# the loader's reservation, which holds the shared inputs and the loader's arena
LOADER_BYTES = 64 << 20
# each tenant's reservation, which holds its outputs and its arena's chunks:
# the loader's and 16 tenants' reservations fill the pool
TENANT_BYTES = 12 << 20
LIMITS = Limits(
    verification_workers=VERIFICATION_WORKERS,
    message_bytes=MAXIMUM_MESSAGE,
    tenant_bytes=TENANT_BYTES,
    chunk_bytes=ARENA_BYTES,
)
LOADER = TenantId(10_000)


def schema_of(kernel: Kernel, text: str) -> OperatorSchema:
    """The service's schema of a corpus kernel, from its reviewed claim and launch."""
    signature, elements = claim_of(kernel.model, kernel.task)
    signature, shared, block, grid = launch_of(kernel.model, kernel.task, "numba", signature, text)
    return OperatorSchema(
        kernel.kind,
        signature,
        tuple(elements.items()),
        CONVENTIONS["numba"],
        block,
        grid,
        shared,
    )


def binding_of_schema(kernel: Kernel, text: str, schema: OperatorSchema) -> Binding:
    """The binding under which the service checks a registration of `text`."""
    return schema.plan(text, Fields({"kind": kernel.kind}, "plan")).binding


# ------------------------------------------------------------------ decisions


@dataclass(frozen=True)
class Decided:
    """The gate's decision on a kernel as a service registration makes it."""

    status: Status
    reason: str | None
    seconds: float


def status_of_check(checked: Checked) -> tuple[Status, str | None]:
    """The status and refusal reason of the service's check of a registration."""
    match checked.decision:
        case Refused(reason=reason):
            return "refused", reason
        case Accepted():
            return ("as_written" if checked.guarded is None else "with_checks"), None
        case _ as never:
            assert_never(never)


def status_of_row(row: Fields) -> Status:
    """The status of a registration in the service's audit: refused, accepted with
    the guard pass's result (`guarded`), or accepted as written."""
    match row.text("decision"):
        case "refused":
            return "refused"
        case "accepted":
            return "as_written" if row.raw("guarded") is None else "with_checks"
        case other:
            raise RuntimeError(f"the audit gives the decision {other!r}")


def decide(
    texts: Mapping[Kernel, str], schemas: Mapping[Kernel, OperatorSchema], cores: set[int]
) -> dict[Kernel, Decided]:
    """The gate's decision on each kernel, by the service's own check of a
    registration (`_check`), all at once on `cores`. The gate caches each
    decision for the process, so a later registration of the same text and
    binding, by the service or by `Executor.register`, repeats it without a
    proof. The calling thread returns to its cores; the threads that the
    checks run on, and their `coqc` processes, take `cores` from it."""

    def one(kernel: Kernel) -> Decided:
        binding = binding_of_schema(kernel, texts[kernel], schemas[kernel])
        began = time.perf_counter()
        checked = _check(texts[kernel], binding, LIMITS.verification_seconds, threading.Event())
        seconds = time.perf_counter() - began
        status, reason = status_of_check(checked)
        return Decided(status, reason, seconds)

    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cores)
    try:
        with ThreadPoolExecutor(len(texts)) as pool:
            decided = dict(zip(texts, pool.map(one, texts), strict=True))
    finally:
        os.sched_setaffinity(0, original)
    if os.sched_getaffinity(0) != original:
        raise RuntimeError("the process did not return to its cores")
    return decided


def _decision_record(decided: Decided) -> dict[str, object]:
    return {"status": decided.status, "reason": decided.reason, "seconds": decided.seconds}


def _require_recorded(decided: Mapping[Kernel, Decided], expected: Mapping[str, Status]) -> None:
    for kernel, decision in decided.items():
        if decision.status != expected[kernel.kind]:
            raise RuntimeError(
                f"{kernel.kind}: the gate decides {decision.status}, "
                f"the records {expected[kernel.kind]}"
            )


# ------------------------------------------------------------------ a service tenant


@dataclass(frozen=True)
class Prepared:
    """A kernel's prepared step in a tenant's session, and the buffer of each output."""

    kernel: Kernel
    prepared: int
    scalars: tuple[int, ...]
    request: PrepareRequest
    outputs: dict[str, str]


@dataclass(frozen=True)
class Served:
    """A built-in operator in place of a refused kernel, its request, and the
    buffer of each output."""

    kernel: Kernel
    builtin: Builtin
    request: BuiltinRequest
    outputs: dict[str, str]


Operation = Prepared | Served


def served_name(operation: Operation) -> str:
    match operation:
        case Prepared(kernel=kernel):
            return kernel.kind
        case Served(builtin=builtin):
            return builtin.op
        case _ as never:
            assert_never(never)


@dataclass(frozen=True)
class TenantReport:
    """What one tenant did and saw. `registrations` holds the service's response
    to each registration, or the decisions of an MPS tenant's own gate with the
    texts it loaded; `served`, `requests`, `verified`, and `final` hold, for each
    template that ran, what served each step, each step's request, and each
    step's outputs, as base64, after the use before the window and after the
    window; `calls` holds every use by what served it; `device_buffers` holds an
    MPS tenant's device buffers by label."""

    tenant: int
    pid: int
    registrations: list[dict[str, object]]
    refused: list[str]
    templates: list[str]
    served: dict[str, list[str]]
    requests: dict[str, list[object]]
    verified: dict[str, list[dict[str, str]]]
    final: dict[str, list[dict[str, str]]]
    loop: Loop
    calls: dict[str, int]
    device_buffers: dict[str, object]


def _operations(
    client: Client,
    template: Template,
    operators: Mapping[Kernel, str],
    schemas: Mapping[Kernel, OperatorSchema],
    workloads: Mapping[Task, Workload],
    shared: Mapping[str, Mapping[str, str]],
) -> list[Operation]:
    """Allocate the template's outputs and make one step per operator: shared
    inputs, the tenant's outputs, and for `masked_sum` the first operator's
    `match`. A kernel with an operator handle is a prepared step; one without,
    which the gate refused, is its task's built-in operator."""
    operations: list[Operation] = []
    for kernel in template.kernels:
        w = workloads[kernel.task]
        handles: dict[str, str] = {}
        outputs: dict[str, str] = {}
        for argument in buffers_of(kernel):
            if linked(kernel, argument):
                handles[argument.name] = operations[0].outputs[LINK[1]]
                continue
            match placement(argument):
                case "shared":
                    handles[argument.name] = shared[kernel.task][argument.name]
                case "output":
                    nbytes = int(w.arrays[argument.name].nbytes)
                    allocated = _require_ok(
                        client.request("allocate", nbytes=nbytes), f"{kernel.kind} output"
                    )
                    handles[argument.name] = outputs[argument.name] = allocated.text("buffer")
                case _ as never:
                    assert_never(never)
        if kernel in operators:
            request = request_of(kernel, schemas[kernel].grid, w, operators[kernel], handles)
            prepared = client.prepare(**request)
            operations.append(
                Prepared(kernel, prepared, tuple(request["scalars"]), request, outputs)
            )
        else:
            builtin = BUILTINS[kernel.task]
            served = builtin_request(builtin, w, handles)
            operations.append(Served(kernel, builtin, served, outputs))
    return operations


def _read_outputs(client: Client, operation: Operation) -> dict[str, bytes]:
    return {
        name: base64.b64decode(
            _require_ok(client.request("read", buffer=handle), f"{name} read").text("data"),
            validate=True,
        )
        for name, handle in operation.outputs.items()
    }


def _service_use(client: Client, mailbox: ClientMailbox, operation: Operation, where: str) -> Use:
    """One use of a step through the service: the prepared step through the
    session's mailbox, or the built-in operator's request on its socket."""
    match operation:
        case Prepared(prepared=prepared, scalars=scalars):
            return lambda: mailbox.run(prepared, scalars)
        case Served(request=request):

            def serve() -> bytes:
                response = _require_ok(
                    client.request(
                        request["op"], buffers=request["buffers"], sizes=request["sizes"]
                    ),
                    f"{where}, {request['op']}",
                )
                return base64.b64decode(response.text("data"), validate=True)

            return serve
        case _ as never:
            assert_never(never)


def _tenant(
    socket_path: str,
    token: str,
    tenant: int,
    configuration: Configuration,
    inputs: Path,
    shared: dict[str, dict[str, str]],
    warmup: float,
    duration: float,
    ready: Barrier,
    results: mp.Queue[TenantReport],
) -> None:
    """A tenant's client of the service: register every kernel of the mix, have
    a built-in operator serve each refused kernel or skip its templates, prepare
    and verify the templates, and run them in a closed loop over the measured
    window."""
    workloads = {task: read_workload(inputs, task) for task in TASKS}
    kernels = kernels_of(configuration)
    texts = {kernel: kernel.path.read_text() for kernel in kernels}
    schemas = {kernel: schema_of(kernel, texts[kernel]) for kernel in kernels}
    where = f"tenant {tenant}"
    with Client(socket_path, token, message_bytes=MAXIMUM_MESSAGE) as client:
        registrations: list[dict[str, object]] = []
        operators: dict[Kernel, str] = {}
        for kernel in kernels:
            (response,) = client.register((texts[kernel], {"kind": kernel.kind}))
            registrations.append({"kind": kernel.kind, "response": response.record()})
            if response.raw("ok") is True:
                operators[kernel] = response.text("operator")
            elif response.fields("error").text("kind") != "refused":
                raise RuntimeError(f"{where}: registering {kernel.kind}: {response.record()}")
        runnable, refused = runnable_templates(configuration, lambda k: k in operators)
        if not runnable:
            raise RuntimeError(f"{where}: the service refused a kernel of every template")
        operations = [
            _operations(client, t, operators, schemas, workloads, shared) for t in runnable
        ]
        with client.data_plane() as mailbox:
            plans: list[Plan] = []
            verified: list[list[dict[str, bytes]]] = []
            for t, steps in zip(runnable, operations, strict=True):

                def read_all(steps: list[Operation] = steps) -> list[dict[str, bytes]]:
                    return [_read_outputs(client, o) for o in steps]

                plan, reads = verify(
                    t.name,
                    [_service_use(client, mailbox, o, f"{where}, {t.name}") for o in steps],
                    [o.kernel for o in steps],
                    [served_name(o) for o in steps],
                    read_all,
                    workloads,
                    where,
                )
                plans.append(plan)
                verified.append(reads)
            ready.wait()
            loop = closed_loop(plans, time.perf_counter(), warmup, duration, where)
        final = []
        for t, steps in zip(runnable, operations, strict=True):
            reads = [_read_outputs(client, o) for o in steps]
            check_reads([o.kernel for o in steps], reads, workloads, f"{where}, {t.name} after")
            final.append(reads)
    results.put(
        TenantReport(
            tenant,
            os.getpid(),
            registrations,
            refused,
            [t.name for t in runnable],
            {plan.name: list(plan.served) for plan in plans},
            {
                t.name: [dict(o.request) for o in steps]
                for t, steps in zip(runnable, operations, strict=True)
            },
            {t.name: encoded(reads) for t, reads in zip(runnable, verified, strict=True)},
            {t.name: encoded(reads) for t, reads in zip(runnable, final, strict=True)},
            loop,
            calls_of(plans, loop),
            {},
        )
    )


# ------------------------------------------------------------------ an MPS tenant


@dataclass(frozen=True)
class Exported:
    """The shared inputs as the owner of mps_shared sends them to every tenant:
    each input's IPC handle, by its label (`mix.label_of`)."""

    handles: dict[str, bytes]


@dataclass(frozen=True)
class OwnerReport:
    """The owner of mps_shared's evidence: each input's allocation and handle,
    and the bytes it compared after every tenant closed its mappings."""

    pid: int
    evidence: dict[str, object]


@dataclass(frozen=True)
class DirectLaunch:
    """A kernel's step as an MPS tenant launches it: the fills, the function,
    the launch geometry and dynamic shared memory, the marshaled parameters, and
    the read-back of `nbytes` from `read` into the pinned `host`, on `stream`."""

    fills: tuple[tuple[int, Memset], ...]  # each output's address and memset
    function: object
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared: int
    values: tuple[int, ...]
    types: tuple[object, ...]
    read: int
    nbytes: int
    host: int
    stream: object

    def run(self) -> bytes:
        for pointer, memset in self.fills:
            match memset.period:
                case 1:
                    call(driver.cuMemsetD8Async(pointer, memset.value, memset.count, self.stream))
                case 2:
                    call(driver.cuMemsetD16Async(pointer, memset.value, memset.count, self.stream))
                case 4:
                    call(driver.cuMemsetD32Async(pointer, memset.value, memset.count, self.stream))
                case _ as never:
                    assert_never(never)
        call(
            driver.cuLaunchKernel(
                self.function,
                *self.grid,
                *self.block,
                self.shared,
                self.stream,
                (self.values, self.types),
                0,
            )
        )
        call(driver.cuMemcpyDtoHAsync(self.host, self.read, self.nbytes, self.stream))
        call(driver.cuStreamSynchronize(self.stream))
        return ctypes.string_at(self.host, self.nbytes)


def _builtin_use(builtin: Builtin, arrays: Mapping[str, CudaArray]) -> Use:
    def use() -> bytes:
        builtin.compute(arrays)
        return read_back(arrays, builtin.read)

    return use


def _device_bytes(pointer: int, nbytes: int) -> bytes:
    data = bytearray(nbytes)
    call(driver.cuMemcpyDtoH(data, pointer, nbytes))
    return bytes(data)


def _owner(
    inputs: Path,
    configuration: Configuration,
    tenants: int,
    inbox: mp.Queue[Exported],
    closed: Barrier,
    results: mp.Queue[TenantReport | OwnerReport],
) -> None:
    """The owner of mps_shared's inputs: allocate each input apart, in whole
    `EXPORT_GRANULARITY` units, upload it, export it, and send the handles to
    each of the `tenants`. When every tenant has closed its mappings
    (`closed`), compare each input on the device with the bytes uploaded."""
    workloads = {task: read_workload(inputs, task) for task in TASKS}
    arrays = {
        label_of(task, argument.name): np.ascontiguousarray(workloads[task].arrays[argument.name])
        for task, argument in shared_inputs(configuration)
    }
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    pointers: dict[str, object] = {}
    handles: dict[str, bytes] = {}
    allocated: dict[str, int] = {}
    try:
        call(driver.cuCtxSetCurrent(context))
        for label, array in arrays.items():
            allocated[label] = -(-max(1, array.nbytes) // EXPORT_GRANULARITY) * EXPORT_GRANULARITY
            (pointers[label],) = call(driver.cuMemAlloc(allocated[label]))
            call(driver.cuMemcpyHtoD(pointers[label], array.tobytes(), array.nbytes))
            (handle,) = call(driver.cuIpcGetMemHandle(pointers[label]))
            handles[label] = _handle_bytes(handle)
        # every input is on the device before any tenant maps it
        call(driver.cuCtxSynchronize())
        for _ in range(tenants):
            inbox.put(Exported(handles))
        closed.wait()
        compared = 0
        for label, array in arrays.items():
            if _device_bytes(address(pointers[label]), array.nbytes) != array.tobytes():
                raise RuntimeError(f"a tenant changed the shared input {label}")
            compared += array.nbytes
        evidence: dict[str, object] = {
            "mechanism": "cuIpcGetMemHandle in the owner, cuIpcOpenMemHandle in each tenant",
            "inputs": {
                label: {
                    "bytes": arrays[label].nbytes,
                    "allocated": allocated[label],
                    "address": address(pointers[label]),
                    "handle": handles[label].hex(),
                }
                for label in arrays
            },
            "compared_bytes": compared,
        }
    finally:
        for pointer in pointers.values():
            call(driver.cuMemFree(pointer))
        call(driver.cuDevicePrimaryCtxRelease(device))
    results.put(OwnerReport(os.getpid(), evidence))


def _mps_tenant(
    tenant: int,
    configuration: Configuration,
    inputs: Path,
    expected: dict[str, Status],
    proof_cores: set[int],
    inbox: mp.Queue[Exported] | None,
    warmup: float,
    duration: float,
    ready: Barrier,
    closed: Barrier | None,
    results: mp.Queue[TenantReport | OwnerReport],
) -> None:
    """A tenant process under MPS: decide and load the kernels that the records
    accept, as the service does, place the inputs (its own copy, or the owner's
    mapped with `inbox`), allocate its outputs, and launch each kernel's step
    directly on its own stream, with the built-in operator in place of each
    refused kernel; verify the templates, run them in a closed loop over the
    measured window, and check the outputs after it. With `closed`, wait
    there after the mappings are closed, so that the owner can compare the
    inputs."""
    import cupy
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    workloads = {task: read_workload(inputs, task) for task in TASKS}
    kernels = kernels_of(configuration)
    texts = {kernel: kernel.path.read_text() for kernel in kernels}
    schemas = {kernel: schema_of(kernel, texts[kernel]) for kernel in kernels}
    where = f"tenant {tenant}"
    loaded = [kernel for kernel in kernels if expected[kernel.kind] != "refused"]
    if any(expected[kernel.kind] != "as_written" for kernel in loaded):
        raise RuntimeError(f"{where}: a direct launch runs only kernels accepted as written")
    decided = decide({kernel: texts[kernel] for kernel in loaded}, schemas, proof_cores)
    _require_recorded(decided, expected)
    runnable, refused = runnable_templates(configuration, lambda k: k in decided)
    exported = None if inbox is None else inbox.get(timeout=SETUP_TIMEOUT)
    if inbox is not None and not isinstance(exported, Exported):
        raise RuntimeError(f"{where}: the owner sent {exported!r}, not the exported inputs")
    # the executor's context is released after `cleanup` has freed and closed every buffer
    with Executor(0, ALIGNMENT) as executor, ExitStack() as cleanup:
        rmm.mr.set_current_device_resource(
            rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=ARENA_BYTES)
        )
        cupy.cuda.set_allocator(rmm_cupy_allocator)
        modules = {
            kernel: executor.register(
                texts[kernel], binding_of_schema(kernel, texts[kernel], schemas[kernel])
            )
            for kernel in loaded
        }
        (stream,) = call(driver.cuStreamCreate(0))
        cleanup.callback(_release, driver.cuStreamDestroy, stream)
        pointers: dict[str, int] = {}
        nbytes: dict[str, int] = {}
        for task, argument in shared_inputs(configuration):
            label = label_of(task, argument.name)
            array = np.ascontiguousarray(workloads[task].arrays[argument.name])
            if exported is None:
                (pointer,) = call(driver.cuMemAlloc(max(1, array.nbytes)))
                cleanup.callback(_release, driver.cuMemFree, pointer)
                call(driver.cuMemcpyHtoD(pointer, array.tobytes(), array.nbytes))
            else:
                (pointer,) = call(
                    driver.cuIpcOpenMemHandle(_handle_of(exported.handles[label]), IPC_FLAGS)
                )
                cleanup.callback(_release, driver.cuIpcCloseMemHandle, pointer)
            pointers[label], nbytes[label] = address(pointer), array.nbytes
            if pointers[label] % ALIGNMENT:
                raise RuntimeError(f"{where}: {label} is at {pointers[label]:#x}, not aligned")

        def read_all(steps: list[dict[str, str]]) -> list[dict[str, bytes]]:
            call(driver.cuCtxSynchronize())
            return [
                {name: _device_bytes(pointers[label], nbytes[label]) for name, label in s.items()}
                for s in steps
            ]

        plans: list[Plan] = []
        verified: list[list[dict[str, bytes]]] = []
        requests: dict[str, list[object]] = {}
        outputs_of: list[list[dict[str, str]]] = []
        for t in runnable:
            uses: list[Use] = []
            served: list[str] = []
            steps: list[dict[str, str]] = []
            records: list[object] = []
            for kernel in t.kernels:
                w = workloads[kernel.task]
                handles: dict[str, str] = {}
                outputs: dict[str, str] = {}
                for argument in buffers_of(kernel):
                    if linked(kernel, argument):
                        handles[argument.name] = steps[0][LINK[1]]
                        continue
                    match placement(argument):
                        case "shared":
                            handles[argument.name] = label_of(kernel.task, argument.name)
                        case "output":
                            label = f"{t.name}/{kernel.kind}/{argument.name}"
                            size = int(w.arrays[argument.name].nbytes)
                            (pointer,) = call(driver.cuMemAlloc(size))
                            cleanup.callback(_release, driver.cuMemFree, pointer)
                            pointers[label], nbytes[label] = address(pointer), size
                            handles[argument.name] = outputs[argument.name] = label
                        case _ as never:
                            assert_never(never)
                steps.append(outputs)
                if kernel in modules:
                    module = modules[kernel]
                    request = request_of(kernel, schemas[kernel].grid, w, kernel.kind, handles)
                    indexes = {
                        handles[a.name]: (w.arrays[a.name].tobytes(), ELEMENTS[a.dtype])
                        for a in buffers_of(kernel)
                        if isinstance(a.role, Points)
                    }
                    checked, layout = checked_launch(module.binding, request, nbytes, indexes)
                    values = marshal(checked, module.widths, layout, pointers)
                    read = request["read"]
                    (host,) = call(driver.cuMemAllocHost(nbytes[read]))
                    cleanup.callback(_release, driver.cuMemFreeHost, host)
                    launch = DirectLaunch(
                        tuple(
                            (pointers[fill["buffer"]], memset_of(fill, nbytes[fill["buffer"]]))
                            for fill in request["fills"]
                        ),
                        module.function,
                        checked.grid,
                        module.binding.block,
                        checked.shared,
                        values,
                        tuple(UNSIGNED[width] for width in module.widths),
                        pointers[read],
                        nbytes[read],
                        address(host),
                        stream,
                    )
                    uses.append(launch.run)
                    served.append(kernel.kind)
                    records.append(
                        {
                            "prepare": dict(request),
                            "grid": list(checked.grid),
                            "block": list(module.binding.block),
                            "shared": checked.shared,
                            "values": list(values),
                        }
                    )
                else:
                    builtin = BUILTINS[kernel.task]
                    counts = builtin.counts({s: w.sizes[s] for s in builtin.sizes})
                    arrays: dict[str, CudaArray] = {
                        o.name: DeviceArray(
                            pointers[handles[o.name]], o.typestr, counts[o.name], o.access == "read"
                        )
                        for o in builtin.operands
                    }
                    uses.append(_builtin_use(builtin, arrays))
                    served.append(builtin.op)
                    records.append(dict(builtin_request(builtin, w, handles)))
            plan, reads = verify(
                t.name,
                uses,
                t.kernels,
                served,
                lambda steps=steps: read_all(steps),
                workloads,
                where,
            )
            plans.append(plan)
            verified.append(reads)
            requests[t.name] = records
            outputs_of.append(steps)
        # the first to run: no work remains on the buffers when they are freed and closed
        cleanup.callback(_release, driver.cuCtxSynchronize)
        ready.wait()
        loop = closed_loop(plans, time.perf_counter(), warmup, duration, where)
        final = []
        for t, steps in zip(runnable, outputs_of, strict=True):
            reads = read_all(steps)
            check_reads(t.kernels, reads, workloads, f"{where}, {t.name} after")
            final.append(reads)
        registrations: list[dict[str, object]] = [
            {
                "kind": kernel.kind,
                "decision": _decision_record(decided[kernel]),
                "loaded": modules[kernel].text,
            }
            for kernel in loaded
        ]
        buffers: dict[str, object] = {
            label: {
                "address": pointers[label],
                "bytes": nbytes[label],
                **(
                    {}
                    if exported is None or label not in exported.handles
                    else {"handle": exported.handles[label].hex()}
                ),
            }
            for label in pointers
        }
    results.put(
        TenantReport(
            tenant,
            os.getpid(),
            registrations,
            refused,
            [t.name for t in runnable],
            {plan.name: list(plan.served) for plan in plans},
            requests,
            {t.name: encoded(reads) for t, reads in zip(runnable, verified, strict=True)},
            {t.name: encoded(reads) for t, reads in zip(runnable, final, strict=True)},
            loop,
            calls_of(plans, loop),
            buffers,
        )
    )
    if closed is not None:
        closed.wait()


# ------------------------------------------------------------------ the service


@dataclass(frozen=True)
class CellReport:
    """The evidence of one cell: its tenants' reports, and the service's evidence
    or the owner's, with the pid of the process that gave it."""

    pid: int | None
    tenants: list[TenantReport]
    evidence: dict[str, object]


def _share(loader: Client, argument: Argument, array: np.ndarray) -> str:
    """Allocate a buffer for a shared input and write it; offsets with their
    element type, which the catalog records as their range."""
    if array.dtype != np.dtype(argument.dtype):
        raise RuntimeError(f"{argument.name}: the workload's {array.dtype} is not {argument.dtype}")
    contiguous = np.ascontiguousarray(array)
    handle = _require_ok(
        loader.request("allocate", nbytes=max(1, int(contiguous.nbytes))), "shared allocation"
    ).text("buffer")
    element = element_of(argument.dtype) if isinstance(argument.role, Points) else None
    _require_ok(
        loader.request(
            "write",
            buffer=handle,
            data=base64.b64encode(contiguous.tobytes()).decode("ascii"),
            element=element,
        ),
        "shared write",
    )
    return handle


def _gated_calls(streams: Sequence[TenantReport]) -> int:
    """The uses of prepared steps, which the service counts as launch requests."""
    return sum(n for s in streams for served, n in s.calls.items() if served not in BY_OP)


def _builtin_calls(streams: Sequence[TenantReport]) -> dict[str, int]:
    """The requests of each built-in operator's query node."""
    calls: dict[str, int] = {}
    for stream in streams:
        for served, n in stream.calls.items():
            if served in BY_OP:
                calls[served] = calls.get(served, 0) + n
    return calls


def _service_worker(
    configuration: Configuration,
    tenants: int,
    inputs: Path,
    expected: dict[str, Status],
    proof_cores: set[int],
    warmup: float,
    duration: float,
    ready: Barrier,
    results: mp.Queue[CellReport],
) -> None:
    """A service for `tenants` tenants, each with one client process."""
    kernels = kernels_of(configuration)
    texts = {kernel: kernel.path.read_text() for kernel in kernels}
    schemas = {kernel: schema_of(kernel, texts[kernel]) for kernel in kernels}
    began = time.perf_counter()
    decided = decide(texts, schemas, proof_cores)
    decision_seconds = time.perf_counter() - began
    _require_recorded(decided, expected)
    workloads = {task: read_workload(inputs, task) for task in TASKS}
    names = ("loader", *(f"tenant-{tenant}" for tenant in range(1, tenants + 1)))
    tokens = {name: secrets.token_hex(16) for name in names}
    principals = (
        Principal("loader", tokens["loader"], LOADER, administrator=True, reservation=LOADER_BYTES),
        *(
            Principal(f"tenant-{tenant}", tokens[f"tenant-{tenant}"], TenantId(tenant))
            for tenant in range(1, tenants + 1)
        ),
    )
    refused = [kernel for kernel in kernels if expected[kernel.kind] == "refused"]
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="opgate-mixed-") as temporary:
        socket_path = Path(temporary) / "service.sock"
        with Service(
            socket_path,
            principals,
            device=0,
            pool_bytes=POOL_BYTES,
            target=Target(8, 6),
            schemas=tuple(schemas.values()),
            nodes=NODES,
            limits=LIMITS,
        ):
            # the loader's registrations, in a session that ends before the tenants
            # start, so that the tenants' sessions hold every operator handle
            registered: list[dict[str, object]] = []
            with Client(socket_path, tokens["loader"], message_bytes=MAXIMUM_MESSAGE) as loader:
                for kernel in kernels:
                    started = time.perf_counter()
                    (response,) = loader.register((texts[kernel], {"kind": kernel.kind}))
                    registered.append(
                        {
                            "kind": kernel.kind,
                            "seconds": time.perf_counter() - started,
                            "response": response.record(),
                        }
                    )
                    ok = response.raw("ok") is True
                    if not ok and response.fields("error").text("kind") != "refused":
                        raise RuntimeError(f"registering {kernel.kind}: {response.record()}")
                    if ok == (expected[kernel.kind] == "refused"):
                        raise RuntimeError(
                            f"{kernel.kind}: the service answers {response.record()}, "
                            f"the records {expected[kernel.kind]}"
                        )
            with Client(socket_path, tokens["loader"], message_bytes=MAXIMUM_MESSAGE) as loader:
                for value in loader.audit():
                    row = Fields(value, "registration")
                    kind = row.text("plan_kind")
                    if status_of_row(row) != expected[kind]:
                        raise RuntimeError(
                            f"{kind}: the service decides {status_of_row(row)}, "
                            f"the records {expected[kind]}"
                        )
                shared: dict[str, dict[str, str]] = {}
                for task, argument in shared_inputs(configuration):
                    array = workloads[task].arrays[argument.name]
                    shared.setdefault(task, {})[argument.name] = _share(loader, argument, array)
                for tenant in range(1, tenants + 1):
                    for task in shared.values():
                        for handle in task.values():
                            _require_ok(
                                loader.request(
                                    "grant_read", buffer=handle, principal=f"tenant-{tenant}"
                                ),
                                "shared grant",
                            )
                reports: mp.Queue[TenantReport] = context.Queue()
                clients = [
                    context.Process(
                        target=_tenant,
                        args=(
                            str(socket_path),
                            tokens[f"tenant-{tenant}"],
                            tenant,
                            configuration,
                            inputs,
                            shared,
                            warmup,
                            duration,
                            ready,
                            reports,
                        ),
                    )
                    for tenant in range(1, tenants + 1)
                ]
                for client in clients:
                    client.start()
                found = _collect(reports, clients, tenants, f"{configuration} clients")
                streams: list[TenantReport] = []
                for item in found:
                    if not isinstance(item, TenantReport):
                        raise RuntimeError(f"a client returned malformed evidence: {item!r}")
                    streams.append(item)

                live = _wait_for_closed_sessions(loader)
                audit = loader.audit()
                _check_audit(audit, kernels, texts, expected, tenants)
                accepted = len(kernels) - len(refused)
                counts = {
                    "register_requests": len(kernels) * (tenants + 1),
                    "accepted_registrations": accepted * (tenants + 1),
                    "refused_requests": len(refused) * (tenants + 1),
                    "module_loads": accepted,
                    "launch_requests": _gated_calls(streams),
                    "failed_checks": 0,
                    "input_errors": 0,
                    "internal_errors": 0,
                    "fatal_errors": 0,
                }
                differ = {k: v for k, v in counts.items() if live.integer(k) != v}
                if differ:
                    raise RuntimeError(f"the service telemetry is not {differ}: {live.record()}")
                if live.raw("node_requests") != _builtin_calls(streams):
                    raise RuntimeError(
                        f"the service's node requests are not {_builtin_calls(streams)}: "
                        f"{live.record()}"
                    )
                for task in shared.values():
                    for handle in task.values():
                        _require_ok(loader.request("free", buffer=handle), "shared free")
                final = _require_ok(loader.request("telemetry"), "final telemetry").fields(
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
                    raise RuntimeError(f"the service did not end cleanly: {final.record()}")
    evidence: dict[str, object] = {
        "pid": os.getpid(),
        "pool_bytes": POOL_BYTES,
        "decision_seconds": decision_seconds,
        "decisions": {kernel.kind: _decision_record(d) for kernel, d in decided.items()},
        "loader_registrations": registered,
        "shared_inputs": shared,
        "audit": audit,
        "telemetry_before_free": live.record(),
        "telemetry_after_cleanup": final.record(),
    }
    results.put(CellReport(os.getpid(), streams, evidence))


def _check_audit(
    audit: list[object],
    kernels: tuple[Kernel, ...],
    texts: Mapping[Kernel, str],
    expected: Mapping[str, Status],
    tenants: int,
) -> None:
    """Each kernel's registrations: one by the loader and one by each tenant, each
    with the submitted text and the recorded status, and one module load if accepted."""
    rows = [Fields(value, "registration") for value in audit]
    for kernel in kernels:
        mine = [row for row in rows if row.text("plan_kind") == kernel.kind]
        principals = sorted(row.text("principal") for row in mine)
        wanted = sorted(("loader", *(f"tenant-{t}" for t in range(1, tenants + 1))))
        if principals != wanted:
            raise RuntimeError(f"{kernel.kind}: registrations by {principals}, not {wanted}")
        if any(row.text("ptx") != texts[kernel] for row in mine):
            raise RuntimeError(f"{kernel.kind}: the audit's PTX differs from the submitted text")
        statuses = {status_of_row(row) for row in mine}
        if statuses != {expected[kernel.kind]}:
            raise RuntimeError(
                f"{kernel.kind}: the service decides {sorted(statuses)}, "
                f"the records {expected[kernel.kind]}"
            )
        loads = sum(row.integer("module_load_count") for row in mine)
        if loads != (0 if expected[kernel.kind] == "refused" else 1):
            raise RuntimeError(f"{kernel.kind}: {loads} module loads")
    if len(rows) != len(kernels) * (tenants + 1):
        raise RuntimeError(f"the audit has {len(rows)} registrations")


# ------------------------------------------------------------------ a cell


def _compute_apps(device: str) -> list[tuple[int, int]]:
    """Each compute process on the GPU and its device memory in MiB."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            "-i",
            device,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    found: list[tuple[int, int]] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            pid, used = line.split(",")
            found.append((int(pid), int(used)))
    return found


class ProcessMemory:
    """Sample each compute process's device memory while a cell is running."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.samples: list[tuple[float, int, int]] = []  # seconds, process, MiB
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample)
        self._start = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            at = time.monotonic() - self._start
            self.samples.extend((at, pid, used) for pid, used in _compute_apps(self.device))
            self._stop.wait(0.25)

    def __enter__(self) -> ProcessMemory:
        self._start = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def _service_cell(
    configuration: Configuration,
    tenants: int,
    inputs: Path,
    expected: dict[str, Status],
    proof_cores: set[int],
    warmup: float,
    duration: float,
) -> CellReport:
    """One service cell in fresh processes: the service and its clients. The
    barrier stays referenced here until the worker ends: `Process.start` drops
    its arguments once it has sent them, and a collected barrier unlinks its
    named semaphores before the spawned worker opens them."""
    context = mp.get_context("spawn")
    results: mp.Queue[CellReport] = context.Queue()
    ready = context.Barrier(tenants)
    worker = context.Process(
        target=_service_worker,
        args=(
            configuration,
            tenants,
            inputs,
            expected,
            proof_cores,
            warmup,
            duration,
            ready,
            results,
        ),
    )
    worker.start()
    (report,) = _collect(results, [worker], 1, f"{configuration}/{tenants}")
    if not isinstance(report, CellReport):
        raise RuntimeError(f"the service returned malformed evidence: {report!r}")
    return report


def _mps_cell(
    configuration: Configuration,
    tenants: int,
    inputs: Path,
    expected: dict[str, Status],
    proof_cores: set[int],
    warmup: float,
    duration: float,
    mps: dict[str, object],
) -> CellReport:
    """One MPS cell in fresh processes: a process per tenant, and for mps_shared
    the owner of the inputs, which starts first."""
    context = mp.get_context("spawn")
    results: mp.Queue[TenantReport | OwnerReport] = context.Queue()
    ready = context.Barrier(tenants)
    shared = configuration == "mps_shared"
    inbox: mp.Queue[Exported] | None = context.Queue() if shared else None
    closed = context.Barrier(tenants + 1) if shared else None  # the tenants and the owner
    workers = []
    if inbox is not None and closed is not None:
        workers.append(
            context.Process(
                target=_owner, args=(inputs, configuration, tenants, inbox, closed, results)
            )
        )
    workers.extend(
        context.Process(
            target=_mps_tenant,
            args=(
                tenant,
                configuration,
                inputs,
                expected,
                proof_cores,
                inbox,
                warmup,
                duration,
                ready,
                closed,
                results,
            ),
        )
        for tenant in range(1, tenants + 1)
    )
    for worker in workers:
        worker.start()
    found = _collect(results, workers, len(workers), f"{configuration}/{tenants}")
    streams = [item for item in found if isinstance(item, TenantReport)]
    owners = [item for item in found if isinstance(item, OwnerReport)]
    if len(streams) != tenants or len(owners) != (1 if shared else 0):
        raise RuntimeError(f"the processes returned malformed evidence: {found!r}")
    evidence: dict[str, object] = {"mps": mps, "owner": owners[0].evidence if owners else None}
    return CellReport(owners[0].pid if owners else None, streams, evidence)


def run_cell(
    configuration: Configuration,
    tenants: int,
    device: str,
    *,
    inputs: Path,
    expected: dict[str, Status],
    proof_cores: set[int],
    warmup: float = WARMUP,
    duration: float = DURATION,
) -> dict[str, object]:
    """Run one cell in fresh processes on the GPU that `device` names; an MPS
    cell under its own MPS control daemon."""
    baseline = _memory(device)
    match host_of(configuration):
        case "service":
            with Sampler(device) as sampler, ProcessMemory(device) as processes:
                report = _service_cell(
                    configuration, tenants, inputs, expected, proof_cores, warmup, duration
                )
            pids = {} if report.pid is None else {"service": report.pid}
        case "mps":
            with Mps() as mps, Sampler(device) as sampler, ProcessMemory(device) as processes:
                report = _mps_cell(
                    configuration,
                    tenants,
                    inputs,
                    expected,
                    proof_cores,
                    warmup,
                    duration,
                    mps.metadata,
                )
            pids = {} if report.pid is None else {"owner": report.pid}
        case _ as never:
            assert_never(never)
    if len(report.tenants) != tenants:
        raise RuntimeError(f"{len(report.tenants)} tenants' evidence for {tenants} tenants")
    streams = sorted(report.tenants, key=lambda stream: stream.tenant)
    names = streams[0].templates
    if any(stream.templates != names for stream in streams):
        raise RuntimeError("the tenants ran different templates")
    refused = streams[0].refused
    if any(stream.refused != refused for stream in streams):
        raise RuntimeError("the tenants skipped different templates")
    served = streams[0].served
    if any(stream.served != served for stream in streams):
        raise RuntimeError("different operators served the tenants' templates")
    completions = [len(stream.loop.latency) for stream in streams]
    queries = sum(completions)
    calls = sum(len(step) for stream in streams for step in stream.loop.steps)
    by_template: dict[str, list[float]] = {name: [] for name in names}
    for stream in streams:
        for index, seconds in zip(stream.loop.template, stream.loop.latency, strict=True):
            by_template[names[index]].append(seconds)
    empty = [name for name, samples in by_template.items() if not samples]
    if empty:
        raise RuntimeError(f"no measured query of {empty}")
    pids |= {f"tenant-{s.tenant}": s.pid for s in streams}
    return {
        "configuration": configuration,
        "tenants": tenants,
        "templates": names,
        "refused_templates": refused,
        "refused_kernels": [
            k.kind for k in kernels_of(configuration) if expected[k.kind] == "refused"
        ],
        "served": served,
        "queries": queries,
        "queries_per_second": queries / duration,
        "operator_calls": calls,
        "operator_calls_per_second": calls / duration,
        "builtin_share": builtin_share([s.loop for s in streams], names, served),
        "jain_fairness": queries**2 / (len(completions) * sum(c * c for c in completions)),
        "latency_us": _latency(sorted(s for stream in streams for s in stream.loop.latency)),
        "latency_us_by_template": {
            name: _latency(sorted(values)) for name, values in by_template.items()
        },
        "memory_mib": sampler.peak - baseline,
        "baseline_memory_mib": baseline,
        "process_memory": {
            "pids": pids,
            # every compute process on the GPU; under MPS the server holds the contexts
            "samples": [list(sample) for sample in processes.samples],
            # the processes that held device memory: a client without a CUDA context is absent
            "peak_mib": {
                name: max(s[2] for s in processes.samples if s[1] == pid)
                for name, pid in pids.items()
                if any(s[1] == pid for s in processes.samples)
            },
        },
        "per_tenant": [
            {
                "tenant": stream.tenant,
                "pid": stream.pid,
                "queries": len(stream.loop.latency),
                "calls": stream.calls,
                "registrations": stream.registrations,
                "requests": stream.requests,
                "device_buffers": stream.device_buffers,
                "verified_outputs": stream.verified,
                "final_outputs": stream.final,
                "template": stream.loop.template,
                "latency_us_samples": [seconds * 1e6 for seconds in stream.loop.latency],
                "step_us_samples": [[s * 1e6 for s in step] for step in stream.loop.steps],
                "values": stream.loop.values,
            }
            for stream in streams
        ],
        "service": report.evidence if host_of(configuration) == "service" else None,
        "mps": report.evidence if host_of(configuration) == "mps" else None,
    }


# ------------------------------------------------------------------ the run


Cell = tuple[Configuration, int, int]  # configuration, tenants, repeat


def cells(
    configurations: tuple[Configuration, ...], tenant_counts: tuple[int, ...], repeats: int
) -> list[Cell]:
    """Every cell, each configuration and tenant count `repeats` times, in an
    order shuffled with `ORDER_SEED`."""
    ordered = [
        (configuration, tenants, repeat)
        for repeat in range(repeats)
        for configuration in configurations
        for tenants in tenant_counts
    ]
    random.Random(ORDER_SEED).shuffle(ordered)
    return ordered


def sources() -> dict[str, str]:
    """The sources of the run: the consolidation harness's, this module's, the
    mix's, the built-in operators', and the corpus code's, and each kernel's
    PTX, signature, launch, and roles."""
    names = [
        "eval/rq4_cost/mixed.py",
        "eval/rq4_cost/mix.py",
        "eval/rq4_cost/fallback.py",
        "eval/rq4_cost/guard_cost.py",
        "eval/direct.py",
        "eval/corpus/workloads.py",
        "eval/corpus/heldout_workloads.py",
        "eval/corpus/run.py",
    ]
    for kernel in every_kernel():
        directory = directory_of(kernel.model, kernel.task)
        names.append(str(kernel.path.relative_to(ROOT)))
        names.append(str((directory / "signature.json").relative_to(ROOT)))
        if (directory / "launch.json").exists():
            names.append(str((directory / "launch.json").relative_to(ROOT)))
        names.append(str((directory.parent / "roles.tsv").relative_to(ROOT)))
    texts = _source_texts()
    texts.update((name, (ROOT / name).read_bytes().decode("utf-8")) for name in names)
    return texts


def _settings(
    environment: dict[str, object],
    source_texts: dict[str, str],
    configurations: tuple[Configuration, ...],
    tenant_counts: tuple[int, ...],
    repeats: int,
    proof_cores: str,
    decisions: dict[str, RecordedDecision],
    registration: Path,
    guarded: Path,
) -> dict[str, object]:
    """What a resumed run must share with the checkpoint."""
    return {
        "schema": SCHEMA,
        "environment": environment,
        "sources": source_texts,
        "configurations": list(configurations),
        "tenant_counts": list(tenant_counts),
        "repeats": repeats,
        "order_seed": ORDER_SEED,
        "warmup_seconds": WARMUP,
        "duration_seconds": DURATION,
        "proof_cores": proof_cores,
        "registration": str(registration),
        "guarded": str(guarded),
        "recorded_decisions": decisions,
    }


def _record(
    settings: dict[str, object],
    *,
    complete: bool,
    started_at_utc: str,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        **settings,
        "experiment": "tenants running a mix of model-written query operators, "
        "through the service or under MPS",
        "complete": complete,
        "started_at_utc": started_at_utc,
        "recorded_at_utc": datetime.now(UTC).isoformat() if complete else None,
        "templates": {
            configuration: [
                {"name": t.name, "kernels": [kernel.kind for kernel in t.kernels]}
                for t in MIXES[configuration]
            ]
            for configuration in CONFIGURATIONS
        },
        "builtins": {
            task: {
                "op": builtin.op,
                "operands": [
                    {
                        "name": o.name,
                        "typestr": o.typestr,
                        "length": list(o.length),
                        "access": o.access,
                    }
                    for o in builtin.operands
                ],
                "sizes": list(builtin.sizes),
                "read": builtin.read,
            }
            for task, builtin in BUILTINS.items()
        },
        "workload": {
            "rows": ROWS,
            "seed": SEED,
            "reads": {task: [name, c.value] for task, (name, c) in READS.items()},
            "link": {"argument": LINK[0], "output": LINK[1]},
            "pool_bytes": POOL_BYTES,
            "arena_bytes": ARENA_BYTES,
            "loader_bytes": LOADER_BYTES,
            "tenant_bytes": TENANT_BYTES,
        },
        "limitations": [
            "an operator-level cycle of eight templates, not whole TPC-H queries",
            "Numba kernels of Claude Opus, with two of Claude Sonnet in the service configuration",
            "closed-loop clients on one GPU; every tenant runs the same cycle",
            "service_refusals skips a refused kernel's templates; service_fallback, mps, and "
            "mps_shared run its task's built-in libcudf operator in its place",
            "in the service a built-in operator is a request on the session's socket, which "
            "the owner thread serves, and a kernel's step goes through the session's mailbox",
            "the gate's decisions are made on the proof cores before the measured window; "
            "the service's registrations and an MPS tenant's executor repeat them from the "
            "gate's cache",
            "an MPS tenant launches each step directly with the parameters that the catalog "
            "checked over a layout of the step's buffers, without the executor's catalog",
            "a use in the window is compared with the verified use: bytes, or float64 values "
            "within the reference check's tolerance; all outputs are checked with the task's "
            "reference before and after the window",
        ],
        # each completed cell and the file of its run in `<record>.runs`
        "runs": runs,
    }


def runs_of(record: Path) -> Path:
    """The directory of a record's runs: one file per cell, written once."""
    return record.with_suffix(record.suffix + ".runs")


def run_file(cell: Cell) -> str:
    configuration, tenants, repeat = cell
    return f"{configuration}_{tenants}_{repeat}.json"


def _write_run(path: Path, run: dict[str, object]) -> None:
    """Write one cell's run once, without indentation: its samples take tens of MB."""
    if path.exists():
        raise RuntimeError(f"{path} exists; a run is written once")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(run, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--proof-cores", required=True, help="the cores of the gate's checks")
    parser.add_argument(
        "--configuration", action="append", choices=CONFIGURATIONS, dest="configurations"
    )
    parser.add_argument("--tenants", action="append", type=int, choices=TENANTS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    configurations: tuple[Configuration, ...] = tuple(args.configurations or CONFIGURATIONS)
    tenant_counts: tuple[int, ...] = tuple(args.tenants or TENANTS)
    repeats: int = args.repeats
    if repeats < 1:
        parser.error("repeats must be positive")
    if len(set(configurations)) != len(configurations) or len(set(tenant_counts)) != len(
        tenant_counts
    ):
        parser.error("duplicate configurations or tenant counts")
    proof_cores = _cores(args.proof_cores)
    if proof_cores & os.sched_getaffinity(0):
        parser.error("the proof cores overlap the cores of the measurement")
    decisions = recorded(args.registration, args.guarded)
    expected = {kind: decision["status"] for kind, decision in decisions.items()}
    source_texts = sources()
    settings = _settings(
        _environment(),
        source_texts,
        configurations,
        tenant_counts,
        repeats,
        args.proof_cores,
        decisions,
        args.registration,
        args.guarded,
    )
    checkpoint = args.output.with_suffix(args.output.suffix + ".partial")
    inputs = args.output.with_suffix(args.output.suffix + ".inputs") / "workloads"
    started_at_utc = datetime.now(UTC).isoformat()
    runs: list[dict[str, object]] = []
    resumed = checkpoint.exists()
    if resumed:
        loaded: object = json.loads(checkpoint.read_text())
        if not isinstance(loaded, dict) or loaded.get("complete") is not False:
            raise RuntimeError(f"malformed checkpoint: {checkpoint}")
        drifted = [key for key, value in settings.items() if loaded.get(key) != value]
        if drifted:
            raise RuntimeError(f"checkpoint does not match this run ({drifted}): {checkpoint}")
        raw_runs = loaded.get("runs")
        started = loaded.get("started_at_utc")
        if not isinstance(raw_runs, list) or not isinstance(started, str):
            raise RuntimeError(f"malformed checkpoint: {checkpoint}")
        runs, started_at_utc = raw_runs, started
    directory = runs_of(args.output)
    completed = {(run["configuration"], run["tenants"], run["repeat"]) for run in runs}
    if len(completed) != len(runs):
        raise RuntimeError(f"checkpoint has duplicate cells: {checkpoint}")
    written = {path.name for path in directory.glob("*.json")} if resumed else set()
    if written != {run["file"] for run in runs}:
        raise RuntimeError(f"{directory} holds other runs than the checkpoint lists")
    save_inputs(inputs, create=not resumed)
    if not resumed:
        directory.mkdir(exist_ok=False)
        _write_json(
            checkpoint, _record(settings, complete=False, started_at_utc=started_at_utc, runs=runs)
        )
    device = _device()
    for cell in cells(configurations, tenant_counts, repeats):
        configuration, tenants, repeat = cell
        if cell in completed:
            print(json.dumps({"resumed": cell}), flush=True)
            continue
        result = run_cell(
            configuration,
            tenants,
            device,
            inputs=inputs,
            expected=expected,
            proof_cores=proof_cores,
        )
        if sources() != source_texts:
            raise RuntimeError(f"an evaluation source changed during cell {cell}")
        _write_run(directory / run_file(cell), {**result, "repeat": repeat})
        runs.append(
            {
                "configuration": configuration,
                "tenants": tenants,
                "repeat": repeat,
                "file": run_file(cell),
            }
        )
        completed.add(cell)
        _write_json(
            checkpoint, _record(settings, complete=False, started_at_utc=started_at_utc, runs=runs)
        )
        shown = {
            k: v
            for k, v in result.items()
            if k not in ("per_tenant", "service", "mps", "process_memory")
        }
        print(json.dumps({**shown, "repeat": repeat}), flush=True)
    save_inputs(inputs, create=False)
    _write_json(
        args.output, _record(settings, complete=True, started_at_utc=started_at_utc, runs=runs)
    )
    checkpoint.unlink()


if __name__ == "__main__":
    main()
