"""E1 real-case motivation: two Dask clients sharing one cuDF GPU worker.

The confidentiality case gives client A a future for only A's cuDF column.
Client B then persists a secret cuDF column on the same dask-cuda worker.  An
ordinary Numba kernel submitted by A reads one element past A's column.  The
fixed RMM pool places B's column at that address, so A receives B's value even
though B's future is not a dependency of A's task.

The availability case uses the same two-client deployment.  A submitted
kernel performs a far out-of-bounds load; B's reduction succeeds before that
task and fails afterward because both clients use the same CUDA context.

Each case runs in a fresh process because an illegal CUDA access permanently
poisons the process's context.  This experiment preserves ``e1_cudf.py`` as
the lower-level cuDF ``Series.apply`` experiment.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from eval import ROOT, module_of

if TYPE_CHECKING:
    import cudf
    from dask_cuda import LocalCUDACluster
    from distributed import Client, Future, Scheduler
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

RESULT_PREFIX = "E1_DASK_RESULT="
POOL_BYTES = 256 * 1024 * 1024
ROWS = 32
ELEMENT_BYTES = 8
SECRET_ONE = 0x11223344556677
SECRET_TWO = 0x77665544332211
WILD_INDEX = 1 << 32


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


class LeakRecord(TypedDict):
    case: Literal["leak"]
    success: bool
    meaning: str
    clients: ClientIds
    shared_worker: str
    same_worker: bool
    same_process: bool
    tenant_a: Observation
    tenant_b: Observation
    adjacent: bool
    first_read: TaskValue
    second_read: TaskValue
    tenant_b_update: FillResult
    tenant_b_sum: TaskValue
    first_schedule: SchedulerObservation
    second_schedule: SchedulerObservation
    secret_one: int
    secret_two: int
    kernel: str


class CrashRecord(TypedDict):
    case: Literal["crash"]
    success: bool
    meaning: str
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
    kernel: str


class ChildRecord(TypedDict):
    case: str
    returncode: int
    result: dict[str, object] | None
    stderr: str
    success: bool


class AllCases(TypedDict):
    experiment: str
    success: bool
    cases: list[ChildRecord]


Result = LeakRecord | CrashRecord | AllCases


def _configure_environment() -> None:
    cache = Path(os.environ.get("OPGATE_E1_DASK_CACHE", "/tmp/opgate-e1-dask-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache / "cupy"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))


_configure_environment()


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


def _read_one_past_a(column: "cudf.Series") -> TaskValue:
    import warnings

    import numpy as np
    from numba import cuda

    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")

    @cuda.jit
    def read_one_past(source: "DeviceNDArray", output: "DeviceNDArray") -> None:
        if cuda.grid(1) == 0:
            output[0] = source[source.size]

    source = column.values
    output = cuda.device_array(1, dtype=np.int64)
    read_one_past[1, 1](source, output)
    value = int(output.copy_to_host()[0])
    cuda.synchronize()
    identity = _worker_identity()
    return {"value": value, "worker": identity["worker"], "pid": identity["pid"]}


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


def _far_out_of_bounds_read(column: "cudf.Series", index: int) -> None:
    import warnings

    import numpy as np
    from numba import cuda

    warnings.filterwarnings("ignore", message="Grid size .* under-utilization")

    @cuda.jit
    def fault(source: "DeviceNDArray", output: "DeviceNDArray", source_index: int) -> None:
        if cuda.grid(1) == 0:
            output[0] = source[source_index]

    source = column.values
    output = cuda.device_array(1, dtype=np.int64)
    fault[1, 1](source, output, index)
    output.copy_to_host()
    cuda.synchronize()


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
        except Exception as error:  # cleanup follows a deliberately poisoned context
            errors.append(error)
    if errors:
        raise RuntimeError(
            "deployment cleanup failed: " + "; ".join(str(error) for error in errors)
        )


def leak_case() -> LeakRecord:
    cluster, client_a, client_b, worker = _start_deployment()
    try:
        a_key = "tenant-a-live-column"
        b_key = "tenant-b-secret-column"
        first_key = "tenant-a-one-past-read-first"
        second_key = "tenant-a-one-past-read-second"

        tenant_a = client_a.submit(_make_column, 0, ROWS, key=a_key, pure=False)
        _wait_without_gathering(client_a, tenant_a)
        tenant_b = client_b.submit(_make_column, SECRET_ONE, ROWS, key=b_key, pure=False)
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

        first_future = client_a.submit(
            _read_one_past_a,
            tenant_a,
            key=first_key,
            pure=False,
        )
        first = first_future.result()
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
        second_future = client_a.submit(
            _read_one_past_a,
            tenant_a,
            key=second_key,
            pure=False,
        )
        second = second_future.result()
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
        first_dependencies = first_schedule["dependencies"]
        second_dependencies = second_schedule["dependencies"]
        success = (
            client_a.id != client_b.id
            and same_worker
            and same_process
            and b_observation["pointer"] == expected_b_pointer
            and update["pointer"] == b_observation["pointer"]
            and first["value"] == SECRET_ONE
            and second["value"] == SECRET_TWO
            and b_sum["value"] == SECRET_TWO * ROWS
            and first_dependencies == [a_key]
            and second_dependencies == [a_key]
            and b_key not in first_dependencies
            and b_key not in second_dependencies
        )
        return {
            "case": "leak",
            "success": success,
            "meaning": (
                "A task whose only data dependency was A's cuDF column returned B's "
                "secret from the same Dask GPU worker and tracked B's in-place update."
            ),
            "clients": {"tenant_a": client_a.id, "tenant_b": client_b.id},
            "shared_worker": worker,
            "same_worker": same_worker,
            "same_process": same_process,
            "tenant_a": a_observation,
            "tenant_b": b_observation,
            "adjacent": b_observation["pointer"] == expected_b_pointer,
            "first_read": first,
            "second_read": second,
            "tenant_b_update": update,
            "tenant_b_sum": b_sum,
            "first_schedule": first_schedule,
            "second_schedule": second_schedule,
            "secret_one": SECRET_ONE,
            "secret_two": SECRET_TWO,
            "kernel": "ordinary Numba source[source.size] load",
        }
    finally:
        _close_deployment(cluster, client_a, client_b)


def crash_case() -> CrashRecord:
    cluster, client_a, client_b, worker = _start_deployment()
    try:
        a_key = "tenant-a-crash-input"
        b_key = "tenant-b-live-column"
        fault_key = "tenant-a-far-out-of-bounds-read"

        tenant_a = client_a.submit(_make_column, 0, ROWS, key=a_key, pure=False)
        _wait_without_gathering(client_a, tenant_a)
        tenant_b = client_b.submit(_make_column, 1, ROWS, key=b_key, pure=False)
        _wait_without_gathering(client_b, tenant_b)

        a_observation = client_a.submit(
            _column_observation,
            tenant_a,
            "tenant-a",
            key="observe-crash-tenant-a",
            pure=False,
        ).result()
        b_observation = client_b.submit(
            _column_observation,
            tenant_b,
            "tenant-b",
            key="observe-crash-tenant-b",
            pure=False,
        ).result()
        baseline = client_b.submit(
            _sum_column,
            tenant_b,
            key="tenant-b-baseline-sum",
            pure=False,
        ).result()

        fault_future = client_a.submit(
            _far_out_of_bounds_read,
            tenant_a,
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

        tenant_b_error: BaseException | None = None
        tenant_b_after: TaskValue | None = None
        try:
            tenant_b_after = client_b.submit(
                _sum_column,
                tenant_b,
                key="tenant-b-sum-after-a-fault",
                pure=False,
            ).result()
        except BaseException as error:
            tenant_b_error = error

        synchronize_error: BaseException | None = None
        try:
            client_b.submit(
                _synchronize_worker,
                key="tenant-b-synchronize-after-a-fault",
                pure=False,
            ).result()
        except BaseException as error:
            synchronize_error = error

        dependencies = fault_schedule["dependencies"]
        success = (
            client_a.id != client_b.id
            and a_observation["worker"] == worker
            and b_observation["worker"] == worker
            and a_observation["pid"] == b_observation["pid"] == baseline["pid"]
            and baseline["value"] == ROWS
            and fault_error is not None
            and tenant_b_after is None
            and tenant_b_error is not None
            and synchronize_error is not None
            and dependencies == [a_key]
            and b_key not in dependencies
        )
        return {
            "case": "crash",
            "success": success,
            "meaning": (
                "A's faulting task poisoned the shared worker's CUDA context; B's "
                "previously valid cuDF reduction and direct synchronization then failed."
            ),
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
            "kernel": "ordinary Numba source[2**32] load",
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


def _run_child(case: str, timeout: int) -> ChildRecord:
    completed = subprocess.run(
        [sys.executable, "-m", module_of(__file__), f"{case}-child"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
        cwd=ROOT,
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


def all_cases(timeout: int) -> AllCases:
    leak = _run_child("leak", timeout)
    crash = _run_child("crash", timeout)
    return {
        "experiment": "E1: two Dask clients on one cuDF GPU worker",
        "success": leak["success"] and crash["success"],
        "cases": [leak, crash],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=("all", "leak-child", "crash-child"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: Result
    if args.case == "leak-child":
        result = leak_case()
    elif args.case == "crash-child":
        result = crash_case()
    else:
        result = all_cases(args.timeout)

    _emit(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
