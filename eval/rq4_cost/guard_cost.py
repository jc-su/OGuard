"""RQ4: what the guard pass's checks cost when a kernel runs, and that they keep its results.

    taskset -c <core> uv run python -m eval.timing eval.rq4_cost.guard_cost \\
        eval/records/guard_cost_<date> --registration eval/records/registration_<date>.json \\
        --proof-cores 32-63

Each run first has the gate check every text, `WORKERS` at a time on the
proof cores, and then returns to its one core for the timing; the gate
keeps its decisions for the process, so the registrations of the timing
part do not check again.

For each safe kernel of the synthesized corpus whose task has a workload,
it takes four texts:

- `as_written`: the kernel as the model wrote it, loaded directly;
- `opgate`: the text that the service loads for the kernel: the text that
  the gate accepts, the guard pass's (`opgate.frontends.guard.guard`), which
  checks the accesses that the gate does not prove when they run, rewrites
  barrier structure (`opgate.frontends.participation`), and limits the loops
  that it does not prove to end (design decision 43), with the zeroing
  prologue of its shared and local memory (`opgate.gate.zeroed`) that
  `Executor.register` adds before it loads the text;
- `every`: the kernel with every access that the pass can check checked
  (`guard_every`), as a design that checks accesses when they run instead
  of proving them would, registered and loaded with its prologue as the
  service loads a text; the gate checks it too;
- `guardian`: the kernel with Guardian's fencing of each global access,
  `(address & mask) | base` (`eval.baselines.guardian`), launched directly
  with the base and mask of the smallest aligned power-of-two partition
  that holds the executor's pool, so that each access inside the pool keeps
  its address; Guardian bounds a tenant's partition, not a buffer, and the
  record states the accesses it fences, or why it cannot fence a kernel.

For each kernel that the gate accepts, as written or guarded, at `ROWS`
rows of its task's workload, the accepted text and the `every` text are
registered with the executor, which loads each with its prologue, and the
catalog checks each one's launch on the workload, with the workload's grid
within the largest grid that the kernel's claim states; the kernel as
written is loaded with the driver,
without the gate, and launched directly with the accepted text's checked
parameters on the same buffers (`eval.direct`), which it reads as its own,
since the appended extent and status parameters come last. The `opgate` and
`every` texts are timed whenever the bytes that the executor loads differ
from the kernel as written; when they are equal, the run of the kernel as
written is the run of that text. The record keeps the text that the
executor loads for each (`loaded`), so that the bytes show which texts equal
the kernel as written, and Guardian's text. A text with the
pass's checks takes a status buffer of the tenant, to which a failed check
stores 1; it is set to zeros with the outputs before each launch. For each
text, from the same inputs, it records each output's bytes, whether a check
failed (the status buffer's first word after the launch; None for a text
without the pass's checks), and the task's check of the outputs against the
reference (`eval.corpus.workloads.check`), the device time, as
`eval/timing.py` measures with the outputs' contents on entry restored
before each launch, outside the timed events, and the registers and local memory per thread
that the driver gives for the loaded function, on which the blocks that a
multiprocessor holds at once depend. The record keeps the loaded texts and
each output's bytes beside the times.
"""

import argparse
import base64
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from eval import ROOT
from eval.baselines.guardian import UnsupportedPtxError, patch_ptx
from eval.checkpoint import _publish, read_artifact, snapshot_inputs
from eval.corpus.run import tasks_of
from eval.corpus.workloads import Loaded, Prepared, check, prepare, workload
from eval.direct import Direct, checked_kernels
from eval.timing import device_time, environment, pinned_core
from opgate.binding import Binding
from opgate.catalog import BufferId, Kernel, Step, TenantId, Write
from opgate.executor import Executor, Module
from opgate.executor.driver import call, driver, integer
from opgate.frontends.guard import STATUS_BYTES, Guarded, guard, guard_every, reports
from opgate.gate import (
    REGISTRATION_SECONDS,
    Accepted,
    Refused,
    VerificationBudget,
    export_terms,
    formal_dir,
    verify,
    zeroed,
)
from opgate.hosts.service.audit import binding_record, parse_binding

__all__ = ["main"]

ROWS = 1_000_000
POOL = 4 << 30
TENANT = TenantId(1)
MODELS = ("fable", "haiku", "opus", "sonnet")
WORKERS = 16  # the gate's checks at a time, before the timing


def _has_workload(task: str) -> bool:
    try:
        workload(task, 1, np.random.default_rng(0))
    except ValueError:
        return False
    return True


def _candidates(registration: Path) -> list[tuple[tuple[str, str, str], bool]]:
    record = json.loads(registration.read_text())
    if record.get("complete") is not True:
        raise ValueError("the registration record is not complete")
    keys = []
    for entry in record["results"]:
        model, task, toolchain = str(entry["id"]).split("/")
        if entry["verdict"] == "safe" and model in MODELS and _has_workload(task):
            keys.append(((model, task, toolchain), entry["accepted"] is True))
    return sorted(keys)


def _resources(function: object) -> dict[str, int]:
    """The registers and the local memory per thread of a loaded function."""
    attribute = driver.CUfunction_attribute
    return {
        "registers": integer(
            driver.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function)
        ),
        "local_bytes": integer(
            driver.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function)
        ),
    }


def _outputs(direct: Direct, prepared: Prepared) -> dict[str, np.ndarray]:
    return {
        name: np.frombuffer(
            direct.read(buffer, np.dtype(dtype).itemsize * count), dtype=dtype, count=count
        ).copy()
        for name, (buffer, dtype, count) in prepared.outputs.items()
    }


def _launch(
    executor: Executor,
    module: Module,
    prepared: Prepared,
    grid: tuple[int, int, int],
    status: BufferId,
) -> Kernel:
    """The checked launch of `module` on the prepared buffers, sizes, and scalars,
    and `status` as its status buffer if it has the guard pass's checks."""
    buffers = (*prepared.launch.buffers, *((status,) if reports(module.binding) else ()))
    launch = replace(prepared.launch, operator=module.operator, grid=grid, buffers=buffers)
    (kernel,) = checked_kernels(executor, Step(TENANT, (launch,)))
    return kernel


def _partition(pool_address: int) -> tuple[int, int]:
    """Guardian's partition for the pool: the base and mask of the smallest
    power-of-two region, aligned to its size, that holds the whole pool, so
    that each access inside the pool keeps its address."""
    size = 1 << (POOL - 1).bit_length()
    while (pool_address // size) * size + size < pool_address + POOL:
        size *= 2
    return (pool_address // size) * size, size - 1


@dataclass(frozen=True)
class _Texts:
    """A kernel's texts and the gate's decisions on them."""

    text: str
    binding: Binding
    guarded: Guarded
    every_text: str
    every_binding: Binding
    every: Accepted | Refused
    fenced: str | None
    fences: int | None
    unsupported: str | None


def source_files() -> dict[str, Path]:
    files = {f"formal/{p.name}": p for p in formal_dir().glob("*.v")}
    for p in (ROOT / "opgate").rglob("*.py"):
        files[str(p.relative_to(ROOT))] = p
    for name in (
        "eval/rq4_cost/guard_cost.py",
        "eval/corpus/workloads.py",
        "eval/corpus/heldout_workloads.py",
        "eval/corpus/adapter.py",
        "eval/corpus/run.py",
        "eval/direct.py",
        "eval/timing.py",
        "eval/checkpoint.py",
        "eval/baselines/guardian.py",
    ):
        files[name] = ROOT / name
    return files


def _texts(key: tuple[str, str, str], registration: Path, entry: dict) -> _Texts:
    """Check the frozen registration's raw input, and the text with every access
    checked, each under the service verification budget."""
    identifier = "/".join(key)
    text = read_artifact(registration, identifier, "ptx")
    binding = parse_binding(entry["binding"])
    if read_artifact(registration, identifier, "v") != export_terms(
        zeroed(text, binding), (binding,)
    ):
        raise ValueError(f"{identifier}: archived query differs from PTX and binding")
    guarded = guard(text, binding, VerificationBudget(checker=verify))
    every_text, every_binding = guard_every(text, binding)
    every = VerificationBudget(checker=verify)(zeroed(every_text, every_binding), every_binding)
    try:
        fenced, report = patch_ptx(text)
    except UnsupportedPtxError as error:
        return _Texts(
            text, binding, guarded, every_text, every_binding, every, None, None, str(error)
        )
    fences = report.instrumented_accesses
    return _Texts(text, binding, guarded, every_text, every_binding, every, fenced, fences, None)


def _labels(text: str, loaded: dict[str, str]) -> tuple[str, ...]:
    """The texts that run for a kernel that the gate accepts, of the texts that
    the executor loads for it (`loaded`): the kernel as written; the text that
    the service loads and the kernel with every access checked, each when its
    bytes differ from the kernel as written; and Guardian's when it can fence
    the kernel."""
    labels = ["as_written"]
    labels += [v for v in ("opgate", "every") if v in loaded and loaded[v] != text]
    if "guardian" in loaded:
        labels.append("guardian")
    return tuple(labels)


def _measure(
    executor: Executor,
    key: tuple[str, str, str],
    recorded: bool,
    texts: _Texts,
    rng: np.random.Generator,
    rows: int = ROWS,
) -> dict[str, object]:
    model, task, toolchain = key
    text, binding, guarded = texts.text, texts.binding, texts.guarded
    every_text, every_binding, every = texts.every_text, texts.every_binding, texts.every
    fenced = texts.fenced
    row: dict[str, object] = {
        "kernel": "/".join(key),
        "tasks": tasks_of(model, task),
        "original": {"ptx": text, "binding": binding_record(binding)},
        "accepted_input": {"ptx": guarded.text, "binding": binding_record(guarded.binding)},
        "recorded": "accepted" if recorded else "refused",
        "decision": "accepted" if isinstance(guarded.decision, Accepted) else "refused",
        "guarded_accesses": list(guarded.guarded),
        "limited_loops": list(guarded.limited),
        "restructured": guarded.restructured,
        "every_decision": "accepted" if isinstance(every, Accepted) else every.reason,
        # each check predicates its access and the store of a failure to the status
        "every_checks": every_text.count("@%ogp") // 2,
        "guardian_fences": texts.fences,
        "guardian_unsupported": texts.unsupported,
    }
    if not isinstance(guarded.decision, Accepted):
        return row
    # the text that the gate accepts, as written or with its unproved accesses
    # checked, which the executor loads with its prologue
    accepted = executor.register(guarded.text, guarded.binding)
    checked = executor.register(every_text, every_binding) if isinstance(every, Accepted) else None
    # the text that the executor loads for each, and Guardian's
    loaded = {"opgate": accepted.text}
    if checked is not None:
        loaded["every"] = checked.text
    if fenced is not None:
        loaded["guardian"] = fenced
    row["loaded"] = loaded
    w = workload(task, rows, rng)
    row["inputs"] = {
        "arrays": {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "bytes": base64.b64encode(array.tobytes()).decode("ascii"),
            }
            for name, array in w.arrays.items()
        },
        "scalars": w.scalars,
        "sizes": w.sizes,
    }
    prepared = prepare(executor, TENANT, Loaded(model, task, toolchain, accepted), w)
    # the workload's grid, within the largest grid that the kernel's claim states
    most = tuple(min(g, m) for g, m in zip(prepared.launch.grid, binding.grid, strict=True))
    grid = (most[0], most[1], most[2])
    (handle,) = call(driver.cuModuleLoadData(text.encode("ascii") + b"\0"))
    (function,) = call(driver.cuModuleGetFunction(handle, binding.kernel.encode("ascii")))
    direct = Direct(executor)
    with executor.tenant(TENANT):
        status = executor.allocate(STATUS_BYTES)
    kernel = _launch(executor, accepted, prepared, grid, status)
    # each label's launch, loaded function, and whether the text has the guard
    # pass's checks: the kernel as written runs with the parameters of the
    # accepted text, whose appended extents and status buffer come last
    runs: dict[str, tuple[Callable[[], None], object, bool]] = {
        "as_written": (partial(direct.launch, kernel, accepted, function), function, False)
    }
    labels = _labels(text, loaded)
    if "opgate" in labels:
        runs["opgate"] = (
            partial(direct.launch, kernel, accepted),
            accepted.function,
            reports(accepted.binding),
        )
    if "every" in labels:
        assert checked is not None
        every_kernel = _launch(executor, checked, prepared, grid, status)
        runs["every"] = (
            partial(direct.launch, every_kernel, checked),
            checked.function,
            reports(checked.binding),
        )
    if fenced is not None:
        (fenced_handle,) = call(driver.cuModuleLoadData(fenced.encode("ascii") + b"\0"))
        (fenced_function,) = call(
            driver.cuModuleGetFunction(fenced_handle, binding.kernel.encode("ascii"))
        )
        # the kernel's parameters, then the partition's base and mask
        count = len(binding.parameters)
        base, mask = _partition(executor.pool_address)
        values = [*direct.values(kernel, accepted)[:count], base, mask]
        widths = (*accepted.widths[:count], 8, 8)
        launch = partial(direct.launch_values, kernel, accepted, fenced_function, values, widths)
        runs["guardian"] = (launch, fenced_function, False)
    initial = Step(
        TENANT,
        (
            *(
                Write(buffer, np.ascontiguousarray(w.arrays[name]).tobytes(), None)
                for name, (buffer, _, _) in prepared.outputs.items()
            ),
            Write(status, bytes(STATUS_BYTES), None),
        ),
    )

    def reset() -> None:
        """The outputs' contents on entry, which a launch changes: a hash table
        that fills, or a sum that grows; and the status buffer's zeros."""
        with executor.tenant(TENANT):
            executor.execute(initial)

    results: dict[str, dict[str, object]] = {}
    for label, (launch, loaded_function, checks) in runs.items():
        reset()
        launch()
        call(driver.cuCtxSynchronize())
        out = _outputs(direct, prepared)
        failed = any(direct.read(status, STATUS_BYTES)) if checks else None
        results[label] = {
            "check": check(task, w, out),
            "check_failed": failed,
            "outputs": {n: base64.b64encode(a.tobytes()).decode("ascii") for n, a in out.items()},
            "time": device_time(launch, reset),
            "resources": _resources(loaded_function),
        }
    row["same_outputs"] = {
        label: results[label]["outputs"] == results["as_written"]["outputs"] for label in results
    }
    row["results"] = results
    call(driver.cuModuleUnload(handle))
    if fenced is not None:
        call(driver.cuModuleUnload(fenced_handle))
    with executor.tenant(TENANT):
        for buffer in (*prepared.buffers, status):
            executor.free(buffer)
    return row


def _cores(spec: str) -> set[int]:
    """The cores of a `taskset -c` list, such as `32-63` or `32,33`."""
    cores: set[int] = set()
    for part in spec.split(","):
        low, _, high = part.partition("-")
        cores.update(range(int(low), int(high or low) + 1))
    return cores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument(
        "--only", nargs="+", help="explicit diagnostic cohort; not a full-corpus run"
    )
    parser.add_argument(
        "--proof-cores", required=True, help="the cores of the gate's checks, before the timing"
    )
    args = parser.parse_args()
    if args.output.exists() or args.rows <= 0:
        parser.error("use a new output record and a positive row count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_inputs(args.output, "sources", source_files(), create=True)
    snapshot_inputs(
        args.output, "registration", {"registration.json": args.registration}, create=True
    )
    rng = np.random.default_rng(7)
    core = pinned_core()
    record: dict[str, object] = {
        "environment": environment(),
        "core": core,
        "proof_cores": args.proof_cores,
        "rows": args.rows,
        "complete": False,
        "registration_seconds": REGISTRATION_SECONDS,
        "registration": str(args.registration),
    }
    candidates = _candidates(args.registration)
    original = json.loads(args.registration.read_text())
    entries = {e["id"]: e for e in original["results"]}
    if args.only is not None:
        if len(set(args.only)) != len(args.only) or set(args.only) - {
            "/".join(k) for k, _ in candidates
        }:
            raise ValueError("diagnostic cohort has duplicate or ineligible kernels")
        candidates = [(k, accepted) for k, accepted in candidates if "/".join(k) in args.only]
    chosen = {"/".join(k) for k, _ in candidates}
    record["selection"] = {
        e["id"]: (
            "selected"
            if e["id"] in chosen
            else "explicit diagnostic subset"
            if args.only
            else "no execution workload"
        )
        for e in original["results"]
        if e["verdict"] == "safe"
    }
    _publish(args.output, (json.dumps(record, indent=1) + "\n").encode())
    # the gate's checks run on the proof cores, before any timing, and the process
    # returns to its one core for the timing
    os.sched_setaffinity(0, _cores(args.proof_cores))
    with ThreadPoolExecutor(WORKERS) as pool:
        keys = [k for k, _ in candidates]
        texts = dict(
            zip(
                keys,
                pool.map(lambda k: _texts(k, args.registration, entries["/".join(k)]), keys),
                strict=True,
            )
        )
    os.sched_setaffinity(0, {core})
    if pinned_core() != core:
        raise RuntimeError("the process did not return to its core")
    rows = []
    with Executor(0, POOL) as executor:
        with executor.tenant(TENANT):
            executor.reserve(POOL)
        for key, recorded in candidates:
            rows.append(_measure(executor, key, recorded, texts[key], rng, args.rows))
            shown = {k: v for k, v in rows[-1].items() if k != "results"}
            print(json.dumps(shown, default=str)[:300], flush=True)
    record["kernels"] = rows
    snapshot_inputs(args.output, "sources", source_files(), create=False)
    snapshot_inputs(
        args.output, "registration", {"registration.json": args.registration}, create=False
    )
    record["complete"] = True
    _publish(args.output, (json.dumps(record, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
