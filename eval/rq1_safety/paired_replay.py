"""E7: paired unguarded and gate-protected replay of the E1 kernels.

The parent compiles three Numba kernels once and gives the exact PTX files to
three fresh child processes.  Two children launch the unsafe PTX without a
gate, reproducing E1's cross-client read and shared-context failure.  The
protected child presents those same bytes and symbolic bindings to OpGate,
submits neither refused module, then loads and launches the exact bytes of an
accepted in-bounds control through the CUDA driver.

This is an experiment around the implemented gate.  It is not the future
catalog/executor or a claim of complete mediation in Dask, whose ordinary
``Client.submit`` interface remains available in this harness.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from ctypes import addressof, c_int64, c_void_p
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, assert_never

from numba import cuda

from eval import module_of
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Binding, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Decision, Refused, verify, zeroed

if TYPE_CHECKING:
    import cudf
    from dask_cuda import LocalCUDACluster
    from distributed import Client, Future, Scheduler
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

RESULT_PREFIX = "E7_GATE_REPLAY_RESULT="
POOL_BYTES = 256 * 1024 * 1024
ROWS = 32
ELEMENT_BYTES = 8
SECRET_ONE = 0x11223344556677
SECRET_TWO = 0x77665544332211
TENANT_A_VALUE = 0x1020304050607
WILD_INDEX = 1 << 32
BLOCK = (1, 1, 1)
PTX_NAMES: tuple[str, ...] = ("one_past", "wild", "safe")
ENTRY = re.compile(r"\.entry\s+([\w$.]+)\s*\(([^)]*)\)", re.S)
PARAMETER = re.compile(r"\.param\s+(?:\.align\s+\d+\s+)?\.\w+\s+([\w$]+)")

Outcome = Literal["accepted", "refused"]


class ModuleRecord(TypedDict):
    path: str
    ptx_text: str
    entry: str
    parameters: int
    bytes: int


class DecisionRecord(TypedDict):
    decision: Outcome
    reason: str | None
    ptx_text: str
    ptx_bytes: int
    entry: str


class WorkerIdentity(TypedDict):
    worker: str
    pid: int
    python: str
    cudf: str
    dask: str
    dask_cuda: str
    distributed: str
    numba: str
    numba_cuda: str
    rmm: str
    rmm_resource: str
    gpu: str
    compute_capability: list[int]


class Observation(WorkerIdentity):
    principal: str
    pointer: int
    rows: int
    bytes: int


class WorkerTask(TypedDict):
    worker: str
    pid: int


class TaskValue(WorkerTask):
    value: int


class FillResult(WorkerTask):
    pointer: int


class LaunchResult(WorkerTask):
    value: int
    ptx_text: str
    entry: str


class SchedulerObservation(TypedDict):
    task: str
    dependencies: list[str]
    clients: list[str]


class ErrorRecord(TypedDict):
    type: str
    message: str


class ClientIds(TypedDict):
    tenant_a: str
    tenant_b: str


class RefusedLaunch(TypedDict):
    submission_attempted: bool
    launch_attempted: bool
    scheduler_task_present: bool


class UnguardedLeakCase(TypedDict):
    case: Literal["unguarded_exact_ptx_leak"]
    success: bool
    clients: ClientIds
    shared_worker: str
    same_worker: bool
    same_process: bool
    tenant_a: Observation
    tenant_b: Observation
    adjacent: bool
    first_read: LaunchResult
    second_read: LaunchResult
    tenant_b_update: FillResult
    tenant_b_sum: TaskValue
    first_schedule: SchedulerObservation
    second_schedule: SchedulerObservation
    ptx_text: str
    launch_attempted: bool


class UnguardedFaultCase(TypedDict):
    case: Literal["unguarded_exact_ptx_fault"]
    success: bool
    clients: ClientIds
    shared_worker: str
    tenant_a: Observation
    tenant_b: Observation
    tenant_b_baseline: TaskValue
    wild_index: int
    fault_schedule: SchedulerObservation
    fault_error: ErrorRecord | None
    tenant_b_after: TaskValue | None
    tenant_b_error: ErrorRecord | None
    synchronize_error: ErrorRecord | None
    ptx_text: str
    launch_attempted: bool


class ProtectedCase(TypedDict):
    case: Literal["gate_protected_exact_ptx"]
    success: bool
    clients: ClientIds
    shared_worker: str
    same_worker: bool
    tenant_a: Observation
    tenant_b: Observation
    adjacent: bool
    decisions: dict[str, DecisionRecord]
    refused_launches: dict[str, RefusedLaunch]
    tenant_b_baseline: TaskValue
    safe_control: LaunchResult
    safe_schedule: SchedulerObservation
    tenant_b_update: FillResult
    tenant_b_after: TaskValue
    tenant_b_synchronize: WorkerTask


ChildCase = UnguardedLeakCase | UnguardedFaultCase | ProtectedCase


class ChildRecord(TypedDict):
    case: str
    returncode: int
    result: dict[str, object] | None
    stderr: str
    success: bool


class CompileTarget(TypedDict):
    compute_capability: list[int]
    numba_opt: bool


class AllCases(TypedDict):
    experiment: str
    success: bool
    compile_target: CompileTarget
    modules: dict[str, dict[str, object]]
    same_ptx_bytes_in_unguarded_and_protected_cases: bool
    cases: list[ChildRecord]
    scope: str


Result = AllCases | ChildCase


def _configure_environment() -> None:
    cache = Path(os.environ.get("OPGATE_E7_CACHE", "/tmp/opgate-e7-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache / "cupy"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))


_configure_environment()


def _entry(text: str) -> tuple[str, tuple[str, ...]]:
    found = ENTRY.findall(text)
    if len(found) != 1:
        raise ValueError(f"expected one entry kernel, found {len(found)}")
    name, parameters = found[0]
    return name, tuple(PARAMETER.findall(parameters))


def _one_past_kernel(source: "DeviceNDArray", output: "DeviceNDArray") -> None:
    if cuda.grid(1) == 0:
        output[0] = source[source.size]


def _wild_kernel(source: "DeviceNDArray", output: "DeviceNDArray", source_index: int) -> None:
    if cuda.grid(1) == 0:
        output[0] = source[source_index]


def _safe_kernel(source: "DeviceNDArray", output: "DeviceNDArray") -> None:
    if cuda.grid(1) == 0:
        output[0] = source[0]


def _compile_ptx(directory: Path, cc: tuple[int, int]) -> dict[str, ModuleRecord]:
    from numba import cuda, int64

    compiled = {
        "one_past": cuda.compile_ptx(
            _one_past_kernel,
            (int64[::1], int64[::1]),
            cc=cc,
            opt=True,
        )[0],
        "wild": cuda.compile_ptx(
            _wild_kernel,
            (int64[::1], int64[::1], int64),
            cc=cc,
            opt=True,
        )[0],
        "safe": cuda.compile_ptx(
            _safe_kernel,
            (int64[::1], int64[::1]),
            cc=cc,
            opt=True,
        )[0],
    }
    records: dict[str, ModuleRecord] = {}
    for name, text in compiled.items():
        path = directory / f"{name}.ptx"
        path.write_text(text)
        kernel, parameters = _entry(text)
        records[name] = {
            "path": str(path),
            "ptx_text": text,
            "entry": kernel,
            "parameters": len(parameters),
            "bytes": len(text.encode("ascii")),
        }
    return records


def _read_ptx(directory: Path, name: str) -> str:
    if name not in PTX_NAMES:
        raise ValueError(f"unknown PTX module {name!r}")
    return (directory / f"{name}.ptx").read_text()


def _signature(*, with_index: bool) -> sig.Signature:
    arguments: list[sig.Argument] = [
        sig.Span("source", "source", sig.Direction.READ, sig.Affine.symbol(0)),
        sig.Span("output", "output", sig.Direction.WRITE, sig.Affine(1)),
    ]
    if with_index:
        arguments.append(sig.Scalar("source_index", sig.Between(WILD_INDEX, WILD_INDEX)))
    return sig.Signature((sig.SizeSymbol("n", 1),), tuple(arguments))


def _binding(text: str, *, with_index: bool) -> Binding:
    kernel, parameters = _entry(text)
    elements = {
        "source": ElementType(8, True),
        "output": ElementType(8, True),
    }
    return derive_binding(
        _signature(with_index=with_index),
        elements,
        Convention.NUMBA,
        kernel,
        parameters,
        BLOCK,
        GRID_LIMIT,
        sig.Affine(0),
    )


def _decision_record(text: str, decision: Decision) -> DecisionRecord:
    match decision:
        case Accepted():
            outcome: Outcome = "accepted"
            reason = None
        case Refused(reason=refused):
            outcome = "refused"
            reason = refused
        case _ as never:
            assert_never(never)
    return {
        "decision": outcome,
        "reason": reason,
        "ptx_text": text,
        "ptx_bytes": len(text.encode("ascii")),
        "entry": _entry(text)[0],
    }


def _worker_identity() -> WorkerIdentity:
    import cudf
    import dask
    import dask_cuda
    import distributed
    import numba
    import numba_cuda
    import rmm
    from distributed import get_worker
    from numba import cuda

    worker = get_worker()
    device = cuda.get_current_device()
    resource = rmm.mr.get_current_device_resource()
    return {
        "worker": worker.address,
        "pid": os.getpid(),
        "python": platform.python_version(),
        "cudf": cudf.__version__,
        "dask": dask.__version__,
        "dask_cuda": dask_cuda.__version__,
        "distributed": distributed.__version__,
        "numba": numba.__version__,
        "numba_cuda": numba_cuda.__version__,
        "rmm": rmm.__version__,
        "rmm_resource": type(resource).__name__,
        "gpu": device.name,
        "compute_capability": list(device.compute_capability),
    }


def _make_column(value: int, rows: int) -> "cudf.Series":
    import cudf
    import cupy as cp

    values = cp.full(rows, value, dtype=cp.int64)
    result = cudf.Series(values)
    if int(result._column.data.ptr) != int(values.data.ptr):
        raise RuntimeError("cuDF copied the CuPy allocation instead of adopting it")
    return result


def _column_observation(column: "cudf.Series", principal: str) -> Observation:
    identity = _worker_identity()
    return {
        **identity,
        "principal": principal,
        "pointer": int(column._column.data.ptr),
        "rows": int(column.size),
        "bytes": int(column.size) * ELEMENT_BYTES,
    }


def _array_pointer(array: "DeviceNDArray") -> int:
    data = getattr(array, "data", None)
    pointer = getattr(data, "ptr", None)
    if pointer is not None:
        return int(pointer)
    return int(array.device_ctypes_pointer.value)


def _array_abi(array: "DeviceNDArray") -> list[c_void_p | c_int64]:
    return [
        c_void_p(0),
        c_void_p(0),
        c_int64(array.size),
        c_int64(array.dtype.itemsize),
        c_void_p(_array_pointer(array)),
        c_int64(array.shape[0]),
        c_int64(array.strides[0]),
    ]


def _launch_exact_ptx(
    column: "cudf.Series", text: str, source_index: int | None = None
) -> LaunchResult:
    """Load and launch ``text`` itself using Numba's documented array ABI."""
    import importlib

    import numpy as np
    from numba import cuda

    driver = importlib.import_module("cuda.bindings.driver")

    source = column.values
    output = cuda.device_array(1, dtype=np.int64)
    arguments = _array_abi(source) + _array_abi(output)
    if source_index is not None:
        arguments.append(c_int64(source_index))
    parameter_pointers = (c_void_p * len(arguments))(
        *(addressof(argument) for argument in arguments)
    )

    kernel, _ = _entry(text)
    module = cuda.current_context().create_module_ptx(text.encode("ascii"))
    function = module.get_function(kernel)
    launched = driver.cuLaunchKernel(
        function.handle,
        1,
        1,
        1,
        *BLOCK,
        0,
        0,
        parameter_pointers,
        0,
    )
    if len(launched) != 1 or int(launched[0]) != 0:
        raise RuntimeError(f"cuLaunchKernel failed: {launched}")
    value = int(output.copy_to_host()[0])
    cuda.synchronize()
    identity = _worker_identity()
    return {
        "value": value,
        "worker": identity["worker"],
        "pid": identity["pid"],
        "ptx_text": text,
        "entry": kernel,
    }


def _fill_column(column: "cudf.Series", value: int) -> FillResult:
    import numpy as np
    from numba import cuda

    column.values.fill(np.int64(value))
    cuda.synchronize()
    identity = _worker_identity()
    return {
        "pointer": int(column._column.data.ptr),
        "worker": identity["worker"],
        "pid": identity["pid"],
    }


def _sum_column(column: "cudf.Series") -> TaskValue:
    from numba import cuda

    value = int(column.sum())
    cuda.synchronize()
    identity = _worker_identity()
    return {"value": value, "worker": identity["worker"], "pid": identity["pid"]}


def _synchronize_worker() -> WorkerTask:
    from numba import cuda

    cuda.synchronize()
    identity = _worker_identity()
    return {"worker": identity["worker"], "pid": identity["pid"]}


def _scheduler_observation(dask_scheduler: "Scheduler", task_key: str) -> SchedulerObservation:
    task = dask_scheduler.tasks[task_key]
    dependencies: list[str] = []
    for dependency in task.dependencies:
        key = dependency.key
        if not isinstance(key, str):
            raise TypeError(f"task key is not a string: {key!r}")
        dependencies.append(key)
    return {
        "task": task_key,
        "dependencies": sorted(dependencies),
        "clients": sorted(dask_scheduler.clients),
    }


def _scheduler_has_task(dask_scheduler: "Scheduler", task_key: str) -> bool:
    return task_key in dask_scheduler.tasks


def _start_deployment() -> tuple["LocalCUDACluster", "Client", "Client", str]:
    import logging

    from dask_cuda import LocalCUDACluster
    from distributed import Client

    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0",
        n_workers=1,
        threads_per_worker=1,
        rmm_pool_size=POOL_BYTES,
        rmm_maximum_pool_size=POOL_BYTES,
        device_memory_limit=0,
        dashboard_address=None,
        silence_logs=logging.ERROR,
    )
    client_a = Client(cluster, name="tenant-a", set_as_default=False)
    client_b = Client(cluster, name="tenant-b", set_as_default=False)
    client_a.wait_for_workers(1)
    workers = sorted(client_a.scheduler_info()["workers"])
    if len(workers) != 1:
        raise RuntimeError(f"expected one shared worker, found {workers}")
    return cluster, client_a, client_b, workers[0]


def _wait_without_gathering(client: "Client", future: "Future[cudf.Series]") -> None:
    from distributed.client import _wait

    client.sync(_wait, [future])
    if future.status == "error":
        error = future.exception()
        if error is None:
            raise RuntimeError(f"task {future.key} failed without an exception")
        raise error


def _error_record(error: BaseException | None) -> ErrorRecord | None:
    if error is None:
        return None
    return {"type": type(error).__name__, "message": str(error)}


def _close_deployment(cluster: "LocalCUDACluster", client_a: "Client", client_b: "Client") -> None:
    errors: list[Exception] = []
    for resource in (client_a, client_b, cluster):
        try:
            resource.close(timeout=10)
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            "deployment cleanup failed: " + "; ".join(str(error) for error in errors)
        )


def _make_adjacent_columns(
    client_a: "Client", client_b: "Client"
) -> tuple[str, str, "Future[cudf.Series]", "Future[cudf.Series]", Observation, Observation]:
    a_key = "tenant-a-live-column"
    b_key = "tenant-b-secret-column"
    tenant_a = client_a.submit(
        _make_column,
        TENANT_A_VALUE,
        ROWS,
        key=a_key,
        pure=False,
    )
    _wait_without_gathering(client_a, tenant_a)
    tenant_b = client_b.submit(
        _make_column,
        SECRET_ONE,
        ROWS,
        key=b_key,
        pure=False,
    )
    _wait_without_gathering(client_b, tenant_b)
    a_observation = client_a.submit(
        _column_observation,
        tenant_a,
        "tenant-a",
        key="observe-tenant-a-column",
        pure=False,
    ).result()
    b_observation = client_b.submit(
        _column_observation,
        tenant_b,
        "tenant-b",
        key="observe-tenant-b-column",
        pure=False,
    ).result()
    expected_b_pointer = a_observation["pointer"] + a_observation["bytes"]
    if b_observation["pointer"] != expected_b_pointer:
        raise RuntimeError(
            "the fixed RMM pool did not place B immediately after A: "
            f"expected {expected_b_pointer}, found {b_observation['pointer']}"
        )
    return a_key, b_key, tenant_a, tenant_b, a_observation, b_observation


def unguarded_leak_case(ptx_directory: Path) -> UnguardedLeakCase:
    text = _read_ptx(ptx_directory, "one_past")
    cluster, client_a, client_b, worker = _start_deployment()
    try:
        a_key, b_key, tenant_a, tenant_b, a_observation, b_observation = _make_adjacent_columns(
            client_a, client_b
        )
        first_key = "tenant-a-unguarded-one-past-first"
        second_key = "tenant-a-unguarded-one-past-second"
        first = client_a.submit(
            _launch_exact_ptx,
            tenant_a,
            text,
            key=first_key,
            pure=False,
        ).result()
        first_schedule: SchedulerObservation = client_a.run_on_scheduler(
            _scheduler_observation,
            task_key=first_key,
        )
        update = client_b.submit(
            _fill_column,
            tenant_b,
            SECRET_TWO,
            key="tenant-b-update-secret",
            pure=False,
        ).result()
        second = client_a.submit(
            _launch_exact_ptx,
            tenant_a,
            text,
            key=second_key,
            pure=False,
        ).result()
        second_schedule: SchedulerObservation = client_a.run_on_scheduler(
            _scheduler_observation,
            task_key=second_key,
        )
        b_sum = client_b.submit(
            _sum_column,
            tenant_b,
            key="tenant-b-check-secret",
            pure=False,
        ).result()
        same_worker = {
            a_observation["worker"],
            b_observation["worker"],
            first["worker"],
            second["worker"],
            update["worker"],
            b_sum["worker"],
        } == {worker}
        same_process = (
            len(
                {
                    a_observation["pid"],
                    b_observation["pid"],
                    first["pid"],
                    second["pid"],
                    update["pid"],
                    b_sum["pid"],
                }
            )
            == 1
        )
        success = (
            client_a.id != client_b.id
            and same_worker
            and same_process
            and first["value"] == SECRET_ONE
            and second["value"] == SECRET_TWO
            and b_sum["value"] == SECRET_TWO * ROWS
            and first_schedule["dependencies"] == [a_key]
            and second_schedule["dependencies"] == [a_key]
            and b_key not in first_schedule["dependencies"]
            and b_key not in second_schedule["dependencies"]
        )
        return {
            "case": "unguarded_exact_ptx_leak",
            "success": success,
            "clients": {"tenant_a": client_a.id, "tenant_b": client_b.id},
            "shared_worker": worker,
            "same_worker": same_worker,
            "same_process": same_process,
            "tenant_a": a_observation,
            "tenant_b": b_observation,
            "adjacent": True,
            "first_read": first,
            "second_read": second,
            "tenant_b_update": update,
            "tenant_b_sum": b_sum,
            "first_schedule": first_schedule,
            "second_schedule": second_schedule,
            "ptx_text": text,
            "launch_attempted": True,
        }
    finally:
        _close_deployment(cluster, client_a, client_b)


def unguarded_fault_case(ptx_directory: Path) -> UnguardedFaultCase:
    text = _read_ptx(ptx_directory, "wild")
    cluster, client_a, client_b, worker = _start_deployment()
    try:
        a_key, b_key, tenant_a, tenant_b, a_observation, b_observation = _make_adjacent_columns(
            client_a, client_b
        )
        baseline = client_b.submit(
            _sum_column,
            tenant_b,
            key="tenant-b-baseline-sum",
            pure=False,
        ).result()
        fault_key = "tenant-a-unguarded-wild-read"
        fault_future = client_a.submit(
            _launch_exact_ptx,
            tenant_a,
            text,
            WILD_INDEX,
            key=fault_key,
            pure=False,
        )
        fault_error: BaseException | None = None
        try:
            fault_future.result()
        except BaseException as error:
            fault_error = error
        fault_schedule: SchedulerObservation = client_a.run_on_scheduler(
            _scheduler_observation,
            task_key=fault_key,
        )
        tenant_b_after: TaskValue | None = None
        tenant_b_error: BaseException | None = None
        try:
            tenant_b_after = client_b.submit(
                _sum_column,
                tenant_b,
                key="tenant-b-sum-after-unguarded-fault",
                pure=False,
            ).result()
        except BaseException as error:
            tenant_b_error = error
        synchronize_error: BaseException | None = None
        try:
            client_b.submit(
                _synchronize_worker,
                key="tenant-b-sync-after-unguarded-fault",
                pure=False,
            ).result()
        except BaseException as error:
            synchronize_error = error
        success = (
            client_a.id != client_b.id
            and a_observation["worker"] == worker
            and b_observation["worker"] == worker
            and baseline["value"] == SECRET_ONE * ROWS
            and fault_error is not None
            and tenant_b_after is None
            and tenant_b_error is not None
            and synchronize_error is not None
            and fault_schedule["dependencies"] == [a_key]
            and b_key not in fault_schedule["dependencies"]
        )
        return {
            "case": "unguarded_exact_ptx_fault",
            "success": success,
            "clients": {"tenant_a": client_a.id, "tenant_b": client_b.id},
            "shared_worker": worker,
            "tenant_a": a_observation,
            "tenant_b": b_observation,
            "tenant_b_baseline": baseline,
            "wild_index": WILD_INDEX,
            "fault_schedule": fault_schedule,
            "fault_error": _error_record(fault_error),
            "tenant_b_after": tenant_b_after,
            "tenant_b_error": _error_record(tenant_b_error),
            "synchronize_error": _error_record(synchronize_error),
            "ptx_text": text,
            "launch_attempted": True,
        }
    finally:
        _close_deployment(cluster, client_a, client_b)


def protected_case(ptx_directory: Path) -> ProtectedCase:
    texts = {name: _read_ptx(ptx_directory, name) for name in PTX_NAMES}
    bindings = {
        name: _binding(texts[name], with_index=name == "wild")
        for name in ("one_past", "wild", "safe")
    }
    decisions = {name: verify(zeroed(texts[name], b), [b]) for name, b in bindings.items()}
    if not isinstance(decisions["one_past"], Refused):
        raise RuntimeError("OpGate accepted the one-past module")
    if not isinstance(decisions["wild"], Refused):
        raise RuntimeError("OpGate accepted the wild-access module")
    if not isinstance(decisions["safe"], Accepted):
        raise RuntimeError("OpGate refused the in-bounds control")

    records = {name: _decision_record(texts[name], decisions[name]) for name in PTX_NAMES}
    refused_keys = {
        "one_past": "tenant-a-protected-one-past",
        "wild": "tenant-a-protected-wild-read",
    }
    cluster, client_a, client_b, worker = _start_deployment()
    try:
        a_key, _, tenant_a, tenant_b, a_observation, b_observation = _make_adjacent_columns(
            client_a, client_b
        )
        baseline = client_b.submit(
            _sum_column,
            tenant_b,
            key="tenant-b-protected-baseline-sum",
            pure=False,
        ).result()

        # The gate branch is deliberately explicit: refused bytes never
        # become a Dask task or reach the CUDA loader.
        refused_task_presence: dict[str, bool] = {
            name: client_a.run_on_scheduler(_scheduler_has_task, task_key=task_key)
            for name, task_key in refused_keys.items()
        }

        safe_key = "tenant-a-protected-safe-control"
        safe = client_a.submit(
            _launch_exact_ptx,
            tenant_a,
            texts["safe"],
            key=safe_key,
            pure=False,
        ).result()
        safe_schedule: SchedulerObservation = client_a.run_on_scheduler(
            _scheduler_observation,
            task_key=safe_key,
        )
        update = client_b.submit(
            _fill_column,
            tenant_b,
            SECRET_TWO,
            key="tenant-b-protected-update-secret",
            pure=False,
        ).result()
        after = client_b.submit(
            _sum_column,
            tenant_b,
            key="tenant-b-protected-sum-after-refusals",
            pure=False,
        ).result()
        synchronized = client_b.submit(
            _synchronize_worker,
            key="tenant-b-protected-synchronize",
            pure=False,
        ).result()

        same_worker = {
            a_observation["worker"],
            b_observation["worker"],
            baseline["worker"],
            safe["worker"],
            update["worker"],
            after["worker"],
            synchronized["worker"],
        } == {worker}
        success = (
            client_a.id != client_b.id
            and same_worker
            and baseline["value"] == SECRET_ONE * ROWS
            and safe["value"] == TENANT_A_VALUE
            and safe["ptx_text"] == records["safe"]["ptx_text"]
            and safe_schedule["dependencies"] == [a_key]
            and after["value"] == SECRET_TWO * ROWS
            and update["pointer"] == b_observation["pointer"]
            and not any(refused_task_presence.values())
        )
        return {
            "case": "gate_protected_exact_ptx",
            "success": success,
            "clients": {"tenant_a": client_a.id, "tenant_b": client_b.id},
            "shared_worker": worker,
            "same_worker": same_worker,
            "tenant_a": a_observation,
            "tenant_b": b_observation,
            "adjacent": True,
            "decisions": records,
            "refused_launches": {
                name: {
                    "submission_attempted": False,
                    "launch_attempted": False,
                    "scheduler_task_present": refused_task_presence[name],
                }
                for name in refused_keys
            },
            "tenant_b_baseline": baseline,
            "safe_control": safe,
            "safe_schedule": safe_schedule,
            "tenant_b_update": update,
            "tenant_b_after": after,
            "tenant_b_synchronize": synchronized,
        }
    finally:
        _close_deployment(cluster, client_a, client_b)


def _emit(result: Result) -> None:
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def _child_succeeded(returncode: int, parsed: dict[str, object] | None) -> bool:
    if returncode != 0 or parsed is None:
        return False
    success = parsed["success"]
    if not isinstance(success, bool):
        raise TypeError(f"the child's success is not a boolean: {success!r}")
    return success


def _run_child(case: str, ptx_directory: Path, timeout: int) -> ChildRecord:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module_of(__file__),
            f"{case}-child",
            "--ptx-directory",
            str(ptx_directory),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )
    parsed: dict[str, object] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            loaded: object = json.loads(line.removeprefix(RESULT_PREFIX))
            if not isinstance(loaded, dict):
                raise TypeError(f"the child result is not an object: {loaded!r}")
            parsed = loaded
    return {
        "case": case,
        "returncode": completed.returncode,
        "result": parsed,
        "stderr": completed.stderr,
        "success": _child_succeeded(completed.returncode, parsed),
    }


def _child_text(record: dict[str, object], key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise TypeError(f"child field {key} is not a string: {value!r}")
    return value


def _child_object(record: dict[str, object], key: str) -> dict[str, object]:
    value = record[key]
    if not isinstance(value, dict):
        raise TypeError(f"child field {key} is not an object: {value!r}")
    return value


def all_cases(timeout: int, cc: tuple[int, int]) -> AllCases:
    with tempfile.TemporaryDirectory(prefix="opgate-e7-") as temporary:
        directory = Path(temporary)
        modules = _compile_ptx(directory, cc)
        leak = _run_child("unguarded-leak", directory, timeout)
        fault = _run_child("unguarded-fault", directory, timeout)
        protected = _run_child("protected", directory, timeout)

    leak_result = leak["result"]
    fault_result = fault["result"]
    protected_result = protected["result"]
    same_bytes = False
    if leak_result is not None and fault_result is not None and protected_result is not None:
        protected_decisions = _child_object(protected_result, "decisions")
        same_bytes = (
            _child_text(leak_result, "ptx_text")
            == _child_text(_child_object(protected_decisions, "one_past"), "ptx_text")
            and _child_text(fault_result, "ptx_text")
            == _child_text(_child_object(protected_decisions, "wild"), "ptx_text")
            and modules["safe"]["ptx_text"]
            == _child_text(_child_object(protected_decisions, "safe"), "ptx_text")
        )
    return {
        "experiment": "E7: paired unguarded and OpGate-protected exact-PTX replay",
        "success": leak["success"] and fault["success"] and protected["success"] and same_bytes,
        "compile_target": {"compute_capability": list(cc), "numba_opt": True},
        "modules": {
            name: {key: value for key, value in record.items() if key != "path"}
            for name, record in modules.items()
        },
        "same_ptx_bytes_in_unguarded_and_protected_cases": same_bytes,
        "cases": [leak, fault, protected],
        "scope": (
            "This evaluates the implemented OpGate decision and an exact-PTX "
            "loader. The full catalog, executor, allocator mediation, and Dask host boundary "
            "are not implemented, so it does not establish complete mediation."
        ),
    }


def _parse_cc(text: str) -> tuple[int, int]:
    pieces = text.split(".")
    if len(pieces) != 2 or any(not piece.isdigit() for piece in pieces):
        raise argparse.ArgumentTypeError("compute capability must look like 8.6")
    return int(pieces[0]), int(pieces[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "all",
            "unguarded-leak-child",
            "unguarded-fault-child",
            "protected-child",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--ptx-directory", type=Path)
    parser.add_argument("--cc", type=_parse_cc, default=(8, 6))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: Result
    if args.case == "all":
        result = all_cases(args.timeout, args.cc)
    else:
        if args.ptx_directory is None:
            parser.error("child cases require --ptx-directory")
        if args.case == "unguarded-leak-child":
            result = unguarded_leak_case(args.ptx_directory)
        elif args.case == "unguarded-fault-child":
            result = unguarded_fault_case(args.ptx_directory)
        else:
            result = protected_case(args.ptx_directory)

    _emit(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
