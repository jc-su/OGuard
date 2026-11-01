"""RQ1: containment of corpus defects under MPS and Guardian.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq1_safety.defect_systems \\
        --output eval/records/defect_systems_<date>.json

The guarded-defects experiment found 27 corpus kernels whose triggers drive an
access outside a buffer.  That experiment records the outcome under OpGate's
guard pass and under a direct (unchecked) launch.  This experiment runs the
same kernels, with the same triggers, under MPS (separate address spaces) and
Guardian (address-fenced PTX), and records whether each system keeps the
defective kernel's effects inside its tenant.

Tenant B runs the defective kernel; tenant A holds the data that B's
out-of-bounds accesses would reach in a shared pool: A's memory holds the
fill byte of the secret, and after each of B's input buffers, at the same
offset in A's memory, the trigger's neighbour values for the secret, of
which there are two (`secret` 0 and 1).  B's memory holds one fill byte
under both secrets.  Each defect runs once with each secret.

Under MPS, A and B are separate client processes of one private CUDA MPS
daemon, each with its own pool.  B launches its kernel; after B ends, A
reads its pool.  Under Guardian, the kernel's PTX is patched with address
fencing (``eval.baselines.guardian.patch_ptx``); A's and B's buffers lie in
their power-of-two partitions of one pool, and B's patched kernel runs with
B's partition base and mask.

Both systems launch the kernel as written with the parameters that its
binding lays out (``eval.baselines.launch``): a Numba array is Numba's
seven-field array structure, an nvcc argument its pointer or scalar.  A
launch that the driver refuses, a process that ends without its result, and
a client of A or B that fails before B's kernel runs end the experiment.

Each cell records, per secret, the launch's parameter values, the bytes of
A's memory that changed, with their offsets and values before and after, and
B's outputs.  The cell's outcome is:

- ``escaped``: a byte of A's memory changed, or, for a trigger with neighbour
  values, B's outputs differ between the two secrets;
- ``faulted``: B's kernel faulted a context that A uses (Guardian's shared
  context, or A's MPS client);
- ``timeout``: B's kernel did not end within ``CELL_SECONDS``;
- ``unsupported``: Guardian's patcher refuses the kernel's PTX;
- ``contained``: B's kernel ran under both secrets, ending or faulting its
  own MPS client, A's memory and A's client stayed intact, and B's outputs
  are the same bytes under both secrets.

The three far triggers (32-bit wrapping indices, 16--20 GiB pools) are
excluded: Guardian's partition size would exceed device memory, and MPS with
those pool sizes is impractical.  Each Guardian cell runs in a subprocess,
so that a faulted context ends one cell, not the experiment.
"""

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Literal

import numpy as np

from eval.baselines.guardian import UnsupportedPtxError, patch_ptx
from eval.baselines.launch import launch, parameters
from eval.corpus.adapter import (
    Role,
    binding_of,
    directory_of,
    read_raw,
    read_roles,
)
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import encode
from eval.rq1_safety.guarded_defects import (
    FILL,
    TRIGGERS,
    Trigger,
    _inputs,
    _Layout,
    _layout,
    _output,
)
from eval.rq4_cost.consolidation import Mps, guardian_layout, guardian_partition_size
from opgate.binding import Binding
from opgate.executor.driver import DriverError, address, call, driver

__all__ = ["main"]

System = Literal["mps", "guardian"]

# a Guardian cell's process, and each MPS client of B, may run for this long
CELL_SECONDS = 300
SECRETS = (0, 1)
B_FILL = FILL[1]  # B's memory, under both secrets


@dataclass(frozen=True)
class _Kernel:
    """A defect's kernel as written, on its trigger."""

    text: str
    binding: Binding
    names: tuple[str, ...]  # the array arguments, in order
    arrays: dict[str, bytes]  # each array argument's bytes
    outputs: tuple[str, ...]  # the arrays that the kernel may write
    layout: _Layout
    sizes: tuple[int, ...]  # one value per size symbol of the binding
    scalars: tuple[int, ...]  # each scalar argument, as its bits spell it


def _kernel(key: str, trigger: Trigger) -> _Kernel:
    model, task, toolchain = key.split("/")
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    raws = read_raw(directory_of(model, task) / "signature.json")
    roles = read_roles(model, task)
    arrays = [r for r in raws if roles.arguments[r.name] not in (Role.SIZE, Role.SCALAR)]
    names = tuple(r.name for r in arrays)
    values = {
        r.name: np.ascontiguousarray(trigger.arrays[r.name], dtype=np.dtype(r.dtype))
        for r in arrays
    }
    return _Kernel(
        text,
        binding,
        names,
        {name: values[name].tobytes() for name in names},
        tuple(name for name in names if _output(roles.arguments[name])),
        _layout(values, names),
        tuple(trigger.sizes[symbol.name] for symbol in binding.sizes),
        tuple(
            encode(r.dtype, trigger.scalars[r.name])
            for r in raws
            if roles.arguments[r.name] is Role.SCALAR
        ),
    )


def _write(at: int, data: bytes) -> None:
    call(driver.cuMemcpyHtoD(driver.CUdeviceptr(at), data, len(data)))


def _read(at: int, nbytes: int) -> bytes:
    data = bytearray(nbytes)
    call(driver.cuMemcpyDtoH(data, driver.CUdeviceptr(at), nbytes))
    return bytes(data)


def _changes(before: bytes, after: bytes) -> list[dict[str, int]]:
    """Each byte that differs, with its offset and its values before and after."""
    old = np.frombuffer(before, dtype=np.uint8)
    new = np.frombuffer(after, dtype=np.uint8)
    return [
        {"offset": int(i), "before": int(old[i]), "after": int(new[i])}
        for i in np.nonzero(old != new)[0]
    ]


def _place_secret(base: int, kernel: _Kernel, trigger: Trigger, secret: int) -> None:
    """A's neighbour values for `secret`, after each of B's input buffers, at
    the same offset from A's `base`."""
    for name, secrets in trigger.after.items():
        _write(base + kernel.layout.offsets[name] + kernel.layout.sizes[name], secrets[secret])


def _place_buffers(base: int, kernel: _Kernel) -> tuple[int, ...]:
    """B's buffers from `base`, at the layout's offsets; their addresses."""
    for name in kernel.names:
        _write(base + kernel.layout.offsets[name], kernel.arrays[name])
    return tuple(base + kernel.layout.offsets[name] for name in kernel.names)


def _outputs(base: int, kernel: _Kernel) -> dict[str, str]:
    return {
        name: base64.b64encode(
            _read(base + kernel.layout.offsets[name], kernel.layout.sizes[name])
        ).decode("ascii")
        for name in kernel.outputs
    }


def _outcome(trigger: Trigger, runs: list[dict[str, object]]) -> str:
    """The cell's outcome from its runs, as the module states it."""
    if any(run.get("a_changes") for run in runs):
        return "escaped"
    if any(run.get("timeout") for run in runs):
        return "timeout"
    if any(run["a_faulted"] for run in runs):
        return "faulted"
    if trigger.after and runs[0]["b_outputs"] != runs[1]["b_outputs"]:
        return "escaped"
    return "contained"


def _cell(key: str, system: System, trigger: Trigger, kernel: _Kernel) -> dict[str, object]:
    """The cell's inputs: the trigger's arrays, neighbour values, and grid, and the block."""
    return {
        "id": key,
        "system": system,
        "inputs": _inputs(trigger),
        "after": {
            name: [base64.b64encode(b).decode("ascii") for b in secrets]
            for name, secrets in trigger.after.items()
        },
        "grid": [trigger.grid, 1, 1],
        "block": list(kernel.binding.block),
    }


# ---------------------------------------------------------------- Guardian


def _guardian_cell(key: str, trigger: Trigger) -> dict[str, object]:
    """Run one defect's trigger under Guardian, with each secret, and return its cell."""
    kernel = _kernel(key, trigger)
    cell = _cell(key, "guardian", trigger, kernel)
    try:
        fenced, report = patch_ptx(kernel.text)
    except UnsupportedPtxError as error:
        return cell | {"outcome": "unsupported", "reason": str(error)}
    # the partition holds the tenant's buffers and their neighbours
    size = guardian_partition_size(kernel.layout.end)
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    call(driver.cuCtxSetCurrent(context))
    (pool,) = call(driver.cuMemAlloc(max(size * 4, 1 << 20)))
    _, (a_base, b_base) = guardian_layout(address(pool), size, 2)
    (handle,) = call(driver.cuModuleLoadData(fenced.encode("ascii") + b"\0"))
    (function,) = call(driver.cuModuleGetFunction(handle, kernel.binding.kernel.encode("ascii")))
    cell["instrumented_accesses"] = report.instrumented_accesses
    runs: list[dict[str, object]] = []
    for secret in SECRETS:
        call(driver.cuMemsetD8(driver.CUdeviceptr(a_base), FILL[secret], size))
        call(driver.cuMemsetD8(driver.CUdeviceptr(b_base), B_FILL, size))
        addresses = _place_buffers(b_base, kernel)
        _place_secret(a_base, kernel, trigger, secret)
        before = _read(a_base, size)
        # the kernel's own parameters, then B's partition base and mask
        own = parameters(kernel.binding, addresses, kernel.sizes, kernel.scalars)
        values = (*own, b_base, size - 1)
        launch(function, (trigger.grid, 1, 1), kernel.binding.block, values)
        run: dict[str, object] = {"secret": secret, "parameters": list(values)}
        try:
            call(driver.cuCtxSynchronize())
        except DriverError as fault:
            # the context that A shares is lost; the process ends with it
            runs.append(run | {"error": str(fault), "a_faulted": True})
            return cell | {"outcome": _outcome(trigger, runs), "runs": runs}
        run |= {
            "error": None,
            "a_faulted": False,
            "a_changes": _changes(before, _read(a_base, size)),
            "b_outputs": _outputs(b_base, kernel),
        }
        runs.append(run)
    call(driver.cuModuleUnload(handle))
    call(driver.cuMemFree(pool))
    call(driver.cuDevicePrimaryCtxRelease(device))
    return cell | {"outcome": _outcome(trigger, runs), "runs": runs}


def _guardian_worker(key: str, result_path: str) -> None:
    Path(result_path).write_text(json.dumps(_guardian_cell(key, TRIGGERS[key]())))


def _run_guardian(key: str) -> dict[str, object]:
    """Run one Guardian cell in a process of its own, so that a faulted
    context ends the cell; a kernel that runs past `CELL_SECONDS` ends it."""
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "cell.json"
        process = ctx.Process(target=_guardian_worker, args=(key, str(path)))
        process.start()
        process.join(timeout=CELL_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
            trigger = TRIGGERS[key]()
            return _cell(key, "guardian", trigger, _kernel(key, trigger)) | {"outcome": "timeout"}
        if process.exitcode != 0 or not path.exists():
            raise RuntimeError(
                f"{key} under Guardian: the cell's process ended with {process.exitcode} "
                "and no result"
            )
        loaded: dict[str, object] = json.loads(path.read_text())
        return loaded


# ---------------------------------------------------------------- MPS


def _mps_b_worker(key: str, result_path: str) -> None:
    """B under MPS: run the defective kernel in B's own address space.  B has
    its own context and pool; its out-of-bounds access reaches B's own pool
    or faults B's own context.  A launch that the driver refuses, or any
    other failure, ends the process without a result."""
    trigger = TRIGGERS[key]()
    kernel = _kernel(key, trigger)
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    call(driver.cuCtxSetCurrent(context))
    pool_bytes = max(kernel.layout.end + (1 << 16), 1 << 20)
    (pool,) = call(driver.cuMemAlloc(pool_bytes))
    call(driver.cuMemsetD8(pool, B_FILL, pool_bytes))
    addresses = _place_buffers(address(pool), kernel)
    (handle,) = call(driver.cuModuleLoadData(kernel.text.encode("ascii") + b"\0"))
    (function,) = call(driver.cuModuleGetFunction(handle, kernel.binding.kernel.encode("ascii")))
    values = parameters(kernel.binding, addresses, kernel.sizes, kernel.scalars)
    launch(function, (trigger.grid, 1, 1), kernel.binding.block, values)
    result: dict[str, object] = {"parameters": list(values)}
    try:
        call(driver.cuCtxSynchronize())
    except DriverError as fault:
        # B's own context is lost; the process ends with it
        result |= {"error": str(fault), "b_outputs": None}
    else:
        result |= {"error": None, "b_outputs": _outputs(address(pool), kernel)}
        call(driver.cuModuleUnload(handle))
        call(driver.cuMemFree(pool))
        call(driver.cuDevicePrimaryCtxRelease(device))
    Path(result_path).write_text(json.dumps(result))


def _mps_a_worker(key: str, secret: int, ready: Event, done: Event, result_path: str) -> None:
    """A under MPS: fill A's own pool for `secret`, signal `ready`, wait for B
    to end, and read A's pool.  A failure before B's kernel runs ends the
    process without a result; a failure after it is recorded, since B's fault
    may end A's client."""
    trigger = TRIGGERS[key]()
    kernel = _kernel(key, trigger)
    call(driver.cuInit(0))
    (device,) = call(driver.cuDeviceGet(0))
    (context,) = call(driver.cuDevicePrimaryCtxRetain(device))
    call(driver.cuCtxSetCurrent(context))
    pool_bytes = max(kernel.layout.end + (1 << 16), 1 << 20)
    (pool,) = call(driver.cuMemAlloc(pool_bytes))
    call(driver.cuMemsetD8(pool, FILL[secret], pool_bytes))
    _place_secret(address(pool), kernel, trigger, secret)
    before = _read(address(pool), pool_bytes)
    ready.set()
    if not done.wait(timeout=2 * CELL_SECONDS):
        raise TimeoutError("B's end was not signalled")
    try:
        after = _read(address(pool), pool_bytes)
    except DriverError as fault:
        result: dict[str, object] = {"a_faulted": True, "a_error": str(fault)}
    else:
        result = {"a_faulted": False, "a_changes": _changes(before, after)}
        call(driver.cuMemFree(pool))
        call(driver.cuDevicePrimaryCtxRelease(device))
    Path(result_path).write_text(json.dumps(result))


def _mps_run(key: str, secret: int, partitions: tuple[str, ...]) -> dict[str, object]:
    """One run of a defect under MPS: A and B as separate client processes."""
    ctx = mp.get_context("spawn")
    ready, done = ctx.Event(), ctx.Event()
    with tempfile.TemporaryDirectory() as scratch:
        a_path, b_path = Path(scratch) / "a.json", Path(scratch) / "b.json"
        a = ctx.Process(target=_mps_a_worker, args=(key, secret, ready, done, str(a_path)))
        b = ctx.Process(target=_mps_b_worker, args=(key, str(b_path)))
        if "CUDA_MPS_SM_PARTITION" in os.environ:
            raise RuntimeError("CUDA_MPS_SM_PARTITION was already set")
        for number, worker in enumerate((a, b)):
            if partitions:
                os.environ["CUDA_MPS_SM_PARTITION"] = partitions[number]
            try:
                worker.start()
            finally:
                os.environ.pop("CUDA_MPS_SM_PARTITION", None)
            if worker is a and not ready.wait(timeout=CELL_SECONDS):
                a.kill()
                b.kill()
                raise RuntimeError(f"{key} under MPS: A's client did not fill its pool")
        b.join(timeout=CELL_SECONDS)
        timeout = b.is_alive()
        if timeout:
            b.kill()
            b.join()
        done.set()
        a.join(timeout=CELL_SECONDS)
        if a.is_alive():
            a.kill()
            a.join()
            raise RuntimeError(f"{key} under MPS: A's client did not end")
        if a.exitcode != 0 or not a_path.exists():
            raise RuntimeError(f"{key} under MPS: A's client ended with {a.exitcode} and no result")
        run: dict[str, object] = {"secret": secret, "timeout": timeout}
        run |= json.loads(a_path.read_text())
        if timeout:
            return run
        if b.exitcode != 0 or not b_path.exists():
            raise RuntimeError(f"{key} under MPS: B's client ended with {b.exitcode} and no result")
        return run | json.loads(b_path.read_text())


def _mps_cell(key: str, partitions: tuple[str, ...]) -> dict[str, object]:
    """Run one defect under MPS, with each secret, and return its cell."""
    trigger = TRIGGERS[key]()
    runs = [_mps_run(key, secret, partitions) for secret in SECRETS]
    cell = _cell(key, "mps", trigger, _kernel(key, trigger))
    return cell | {"outcome": _outcome(trigger, runs), "runs": runs}


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--system",
        choices=("mps", "guardian", "both"),
        default="both",
        help="which containment system to evaluate (default: both)",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record")
    systems: list[System] = []
    if args.system in ("guardian", "both"):
        systems.append("guardian")
    if args.system in ("mps", "both"):
        systems.append("mps")
    # the far triggers (large pools) are excluded
    keys = [k for k in sorted(TRIGGERS) if not TRIGGERS[k]().large]
    started = datetime.now(UTC).isoformat()
    cells: list[dict[str, object]] = []

    # Guardian cells run without MPS
    if "guardian" in systems:
        print(f"--- Guardian: {len(keys)} defects", flush=True)
        for key in keys:
            result = _run_guardian(key)
            cells.append(result)
            print(f"  {key}: {result['outcome']}", flush=True)

    # MPS cells run directly (each already spawns A and B as subprocesses,
    # so a second subprocess wrapper would create grandchild processes whose
    # import chain fails under the MPS daemon's remapped device)
    if "mps" in systems:
        print(f"--- MPS: {len(keys)} defects", flush=True)
        with Mps(static_tenants=2) as mps:
            for key in keys:
                result = _mps_cell(key, mps.partitions)
                cells.append(result)
                print(f"  {key}: {result['outcome']}", flush=True)

    args.output.write_text(
        json.dumps(
            {
                "experiment": "containment of corpus defects under MPS and Guardian",
                "started_at_utc": started,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "systems": systems,
                "defects": len(keys),
                "cells": cells,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
