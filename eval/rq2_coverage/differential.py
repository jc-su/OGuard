"""RQ2: a safe kernel the gate accepts only after the guard pass changed it, as written and as changed.

    CUDA_VISIBLE_DEVICES=<gpu> uv run python -m eval.rq2_coverage.differential \\
        --guarded eval/records/guarded_2026-10-07b.json \\
        --output eval/records/differential_<date>.json

The guard pass may add runtime checks, rewrite barrier participation, or limit
a loop (design decisions 38, 39, 43). For a safe kernel on inputs that meet its
contract, the change must not alter what the kernel computes, and no check must
fail. This experiment runs each safe kernel that the gate accepts only after the
guard pass changed it (`eval/records/guarded_2026-10-07b.json`, a row whose
label is `safe`, whose decision is `accepted`, and that the pass gave a check, a
limit, or a rewrite) both as the model wrote it and as the gate accepts it, on
its task's workloads (`eval/corpus/workloads.py`), and compares their output
bytes.

For each such kernel whose task has a workload, and for each size in `SIZES`
(0, 1, 1000, 1000003, with the edge values the workload module injects), it
builds inputs that meet the contract, writes the outputs' contents on entry,
and launches:

- the kernel as written, loaded with the driver, with no gate, on the executor's
  pool (`eval.direct`);
- the text that the gate accepts, the guard pass's, which carries its status
  buffer when it added checks or limits.

Both launches read the same leading parameters, since the pass appends its
extents and status buffer last. From the same inputs it reads each output's
bytes, compares the two byte for byte, runs the task's reference check of each
(`eval.corpus.workloads.check`), and, for the changed text, reads the status
buffer, which must report no failed check on these in-bounds inputs. A kernel
whose task has no workload (every DataKernelBench kernel, and some held-out
tasks) is recorded with no runs.

The raw output bytes are retained for sizes up to `RETAIN`; a larger size is
compared in process and its byte lengths and comparison are recorded, since
retaining a million rows for every kernel is not useful. Original PTX, the
pass's PTX, and the bindings come from the guarded record's artifacts; no kernel
is registered from a freshly derived corpus binding. The gate checks the
pass's text under the guarded record's registration budget
(`registration_seconds` in its provenance), the budget under which the record
accepted it, and the executor then loads the accepted text.
"""

import argparse
import base64
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import assert_never

import numpy as np

from eval import ROOT
from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from eval.corpus.adapter import Points, directory_of, read_raw, read_roles
from eval.corpus.workloads import Loaded, Prepared, check, prepare, workload
from eval.direct import Direct, checked_kernels
from eval.timing import environment
from opgate.binding import Base, Binding
from opgate.catalog import BufferId, Kernel, Step, TenantId, Write, evaluate
from opgate.executor import Executor, Module, RefusedError
from opgate.executor.driver import call, driver
from opgate.frontends.guard import STATUS_BYTES, reports
from opgate.gate import (
    Accepted,
    Refused,
    VerificationBudget,
    export_terms,
    formal_dir,
    zeroed,
)
from opgate.hosts.service.audit import parse_binding

__all__ = ["main", "measure", "validate_evidence"]

GUARDED = ROOT / "eval/records/guarded_2026-10-07b.json"
SIZES = (0, 1, 1000, 1_000_003)
RETAIN = 1000  # retain output bytes for sizes up to this; a larger size is compared in process
POOL = 4 << 30
TENANT = TenantId(1)
MODELS = ("fable", "haiku", "opus", "sonnet")  # the synthesized corpus, whose tasks have workloads
KIND = "safe guard-changed kernels, as written and as the gate accepts them, on their workloads"


def source_files() -> dict[str, Path]:
    files = {f"formal/{p.name}": p for p in formal_dir().glob("*.v")}
    for p in (ROOT / "opgate").rglob("*.py"):
        files[str(p.relative_to(ROOT))] = p
    for name in (
        "eval/rq2_coverage/differential.py",
        "eval/corpus/workloads.py",
        "eval/corpus/heldout_workloads.py",
        "eval/corpus/adapter.py",
        "eval/corpus/run.py",
        "eval/direct.py",
        "eval/checkpoint.py",
    ):
        files[name] = ROOT / name
    return files


@dataclass(frozen=True)
class Candidate:
    """A row of the guarded record: the kernel, the pass's checks, limits, and
    rewrite, and the binding of the text that the gate accepts."""

    id: str
    label: str
    decision: str
    guarded: tuple[str, ...]
    limited: tuple[str, ...]
    restructured: bool
    binding: dict[str, object]

    @property
    def changed(self) -> bool:
        """Whether the guard pass changed this kernel: a check, a limit, or a rewrite."""
        return bool(self.guarded or self.limited or self.restructured)


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where}: not a list of strings")
    return tuple(str(v) for v in value)


def parse_row(value: object) -> Candidate:
    """A guarded record's row, with the type of each field this experiment reads."""
    if not isinstance(value, dict):
        raise ValueError("a guarded row is not an object")
    row = {str(k): v for k, v in value.items()}
    identifier, label, decision = row["id"], row["label"], row["decision"]
    restructured, binding = row["restructured"], row["binding"]
    if not (isinstance(identifier, str) and isinstance(label, str) and isinstance(decision, str)):
        raise ValueError(f"{identifier}: the id, label, or decision is not a string")
    if type(restructured) is not bool or not isinstance(binding, dict):
        raise ValueError(f"{identifier}: the rewrite flag or the binding has another type")
    return Candidate(
        identifier,
        label,
        decision,
        _strings(row["guarded"], f"{identifier}.guarded"),
        _strings(row["limited"], f"{identifier}.limited"),
        restructured,
        {str(k): v for k, v in binding.items()},
    )


def candidates(record: dict[str, object]) -> list[Candidate]:
    """The safe kernels that the gate accepts only after the guard pass changed them."""
    results = record["results"]
    if not isinstance(results, list):
        raise ValueError("the guarded record's results are not a list")
    rows = [parse_row(value) for value in results]
    return sorted(
        (row for row in rows if row.label == "safe" and row.decision == "accepted" and row.changed),
        key=lambda row: row.id,
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_launch(
    executor: Executor,
    module: Module,
    prepared: Prepared,
    grid: tuple[int, int, int],
    status: BufferId,
) -> Kernel:
    """The checked launch of `module` on the prepared buffers, with `status` as its
    status buffer when the pass's checks report through it."""
    buffers = (*prepared.launch.buffers, *((status,) if reports(module.binding) else ()))
    launch = replace(prepared.launch, operator=module.operator, grid=grid, buffers=buffers)
    (kernel,) = checked_kernels(executor, Step(TENANT, (launch,)))
    return kernel


def _outputs(direct: Direct, prepared: Prepared) -> dict[str, np.ndarray]:
    return {
        name: np.frombuffer(
            direct.read(buffer, np.dtype(dtype).itemsize * count), dtype=dtype, count=count
        ).copy()
        for name, (buffer, dtype, count) in prepared.outputs.items()
    }


def _output_record(name_to_bytes: dict[str, bytes], *, retain: bool) -> dict[str, object]:
    """Each output's byte length and digest, and the bytes themselves when retained."""
    return {
        name: {
            "nbytes": len(data),
            "sha256": _digest(data),
            "bytes": base64.b64encode(data).decode("ascii") if retain else None,
        }
        for name, data in name_to_bytes.items()
    }


def _register(executor: Executor, text: str, binding: Binding, seconds: float) -> Module:
    """Load the pass's `text` once the gate accepts it within `seconds`, the
    guarded record's budget of one registration; the executor's check of the
    loaded text is the gate's decision for this text and binding."""
    match VerificationBudget(seconds)(zeroed(text, binding), binding):
        case Accepted():
            return executor.register(text, binding)
        case Refused(reason=reason):
            raise RefusedError(reason)
        case _ as never:
            assert_never(never)


def _run_size(
    executor: Executor,
    key: tuple[str, str, str],
    original_text: str,
    final_text: str,
    final_binding: Binding,
    seconds: float,
    n: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Run the kernel as written and as the gate accepts it at `n` rows; the two
    outputs' bytes, their byte-for-byte comparison, each run's reference check, and
    the changed run's reported failed check."""
    model, task, toolchain = key
    try:
        w = workload(task, n, rng)
    except ValueError as error:
        return {"size": n, "supported": False, "reason": f"no workload: {error}"}
    roles = read_roles(model, task)
    for raw in read_raw(directory_of(model, task) / "signature.json"):
        if isinstance(roles.arguments[raw.name], Points) and len(w.arrays[raw.name]) == 0:
            # the catalog checks an index input inside its range, which an empty one lacks
            return {
                "size": n,
                "supported": False,
                "reason": f"no inputs: {raw.name} is an empty index buffer",
            }
    accepted = _register(executor, final_text, final_binding, seconds)
    prepared = prepare(executor, TENANT, Loaded(model, task, toolchain, accepted), w)
    entry = {
        name: np.ascontiguousarray(w.arrays[name], dtype=dtype)
        for name, (_, dtype, _) in prepared.outputs.items()
    }
    try:
        # the reference is a function of the inputs: an empty input has no minimum
        check(task, w, entry)
    except ValueError as error:
        with executor.tenant(TENANT):
            for buffer in prepared.buffers:
                executor.free(buffer)
        return {"size": n, "supported": False, "reason": f"no reference: {error}"}
    grid = tuple(min(g, m) for g, m in zip(prepared.launch.grid, final_binding.grid, strict=True))
    direct = Direct(executor)
    with executor.tenant(TENANT):
        status = executor.allocate(STATUS_BYTES)
    handle: object | None = None
    try:
        (handle,) = call(driver.cuModuleLoadData(original_text.encode("ascii") + b"\0"))
        (function,) = call(driver.cuModuleGetFunction(handle, final_binding.kernel.encode("ascii")))
        # the binding's index range on an output is a precondition of the launch: the
        # initial contents are written with the element type so that the catalog
        # records the range of their values
        bases = [p for _, p in final_binding.parameters if isinstance(p, Base)]
        if reports(final_binding):
            bases.pop()  # the status buffer, which ends the arguments
        elements = {
            buffer: parameter.span.index.element
            for buffer, parameter in zip(prepared.launch.buffers, bases, strict=True)
            if parameter.span.index is not None
        }
        for parameter in bases:
            index = parameter.span.index
            if index is not None and evaluate(index.hi, prepared.launch.sizes) < evaluate(
                index.lo, prepared.launch.sizes
            ):
                # no initial contents of the output lie in an empty range
                return {"size": n, "supported": False, "reason": "an index range is empty"}
        initial = Step(
            TENANT,
            (
                *(
                    Write(
                        buffer,
                        np.ascontiguousarray(w.arrays[name]).tobytes(),
                        elements.get(buffer),
                    )
                    for name, (buffer, _, _) in prepared.outputs.items()
                ),
                Write(status, bytes(STATUS_BYTES), None),
            ),
        )
        with executor.tenant(TENANT):
            executor.execute(initial)
        kernel = _checked_launch(executor, accepted, prepared, (grid[0], grid[1], grid[2]), status)

        def reset() -> None:
            with executor.tenant(TENANT):
                executor.execute(initial)

        reset()
        direct.launch(kernel, accepted)  # the text that the gate accepts
        call(driver.cuCtxSynchronize())
        changed_out = _outputs(direct, prepared)
        check_failed = any(direct.read(status, STATUS_BYTES)) if reports(accepted.binding) else None

        reset()
        direct.launch(kernel, accepted, function)  # the kernel as written
        call(driver.cuCtxSynchronize())
        written_out = _outputs(direct, prepared)
    finally:
        if handle is not None:
            call(driver.cuModuleUnload(handle))
        with executor.tenant(TENANT):
            for buffer in (*prepared.buffers, status):
                executor.free(buffer)
    retain = n <= RETAIN
    changed_bytes = {name: array.tobytes() for name, array in changed_out.items()}
    written_bytes = {name: array.tobytes() for name, array in written_out.items()}
    same = {name: changed_bytes[name] == written_bytes[name] for name in changed_bytes}
    return {
        "size": n,
        "supported": True,
        "rows": len(next(iter(w.arrays.values()))) if w.arrays else 0,
        "changed": {
            "outputs": _output_record(changed_bytes, retain=retain),
            "check": check(task, w, changed_out),
            "check_failed": check_failed,
        },
        "as_written": {
            "outputs": _output_record(written_bytes, retain=retain),
            "check": check(task, w, written_out),
        },
        "same_bytes": same,
        "same_bytes_all": all(same.values()),
    }


def measure(guarded: Path, output: Path, *, only: list[str] | None, sizes: tuple[int, ...]) -> None:
    record = json.loads(guarded.read_text())
    if record.get("complete") is not True:
        raise ValueError("the guarded record is not complete")
    seconds = record["provenance"]["registration_seconds"]
    if not isinstance(seconds, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"the guarded record's registration budget is {seconds!r}")
    rows = {row.id: row for row in candidates(record)}
    if only is not None:
        missing = set(only) - set(rows)
        if len(set(only)) != len(only) or missing:
            raise ValueError(f"selection has duplicate or non-candidate kernels: {sorted(missing)}")
        rows = {identifier: rows[identifier] for identifier in only}
    provenance = {
        "kind": KIND,
        "guarded_record": str(guarded),
        "registration_seconds": seconds,
        "library": str(formal_dir()),
        "environment": environment(),
        "sizes": list(sizes),
        "retain_bytes_up_to": RETAIN,
        "kernels": sorted(rows),
    }
    rng = np.random.default_rng(13)
    with Checkpoint(output, provenance, resume=False) as checkpoint:
        snapshot_inputs(output, "sources", source_files(), create=True)
        with Executor(0, POOL) as executor:
            with executor.tenant(TENANT):
                executor.reserve(POOL)
            for identifier, row in rows.items():
                model, task, toolchain = identifier.split("/")
                final_text = read_artifact(guarded, identifier + "/final", "ptx")
                final_binding = parse_binding(row.binding)
                if read_artifact(guarded, identifier + "/final", "v") != export_terms(
                    zeroed(final_text, final_binding), (final_binding,)
                ):
                    raise ValueError(f"{identifier}: the pass's query differs from its PTX")
                original_text = read_artifact(guarded, identifier + "/original", "ptx")
                has_workload = _supported(task)
                runs: list[dict[str, object]] = []
                if has_workload and model in MODELS:
                    for n in sizes:
                        runs.append(
                            _run_size(
                                executor,
                                (model, task, toolchain),
                                original_text,
                                final_text,
                                final_binding,
                                seconds,
                                n,
                                rng,
                            )
                        )
                checkpoint.add(
                    {
                        "id": identifier,
                        "guarded_accesses": list(row.guarded),
                        "limited_loops": list(row.limited),
                        "restructured": row.restructured,
                        "workload": has_workload and model in MODELS,
                        "runs": runs,
                    }
                )
                print(
                    f"{identifier}: {'runs' if runs else 'no workload'} "
                    f"({sum(1 for r in runs if r.get('supported') and not r['same_bytes_all'])} mismatch)",
                    flush=True,
                )
        snapshot_inputs(output, "sources", source_files(), create=False)
        checkpoint.finish()
    validate_evidence(output)


def _supported(task: str) -> bool:
    try:
        workload(task, 1, np.random.default_rng(0))
    except ValueError:
        return False
    return True


def validate_evidence(path: Path) -> dict[str, object]:
    """A complete record whose every run's recorded comparison agrees with its
    retained output bytes, and whose changed runs report no failed check on these
    in-bounds inputs."""
    record = json.loads(path.read_text())
    if record.get("schema") != 2 or record.get("complete") is not True:
        raise ValueError(f"{path}: the differential record is incomplete")
    if record["provenance"]["kind"] != KIND:
        raise ValueError(f"{path}: another experiment")
    snapshot_inputs(path, "sources", source_files(), create=False)
    for row in record["results"]:
        if row["workload"] != bool(row["runs"]):
            raise ValueError(f"{row['id']}: the workload flag disagrees with its runs")
        for run in row["runs"]:
            if not run["supported"]:
                continue
            changed_out, written = run["changed"]["outputs"], run["as_written"]["outputs"]
            if set(changed_out) != set(written):
                raise ValueError(
                    f"{row['id']} size {run['size']}: the two runs wrote other outputs"
                )
            for name, same in run["same_bytes"].items():
                left, right = changed_out[name], written[name]
                recorded = left["sha256"] == right["sha256"]
                if left["bytes"] is not None and right["bytes"] is not None:
                    recorded = base64.b64decode(left["bytes"]) == base64.b64decode(right["bytes"])
                if same != recorded:
                    raise ValueError(
                        f"{row['id']} size {run['size']} {name}: the comparison differs from the bytes"
                    )
            if run["changed"]["check_failed"]:
                raise ValueError(
                    f"{row['id']} size {run['size']}: the changed text reported a failed check on correct input"
                )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--guarded", type=Path, default=GUARDED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new record to preserve archived evidence")
    measure(args.guarded, args.output, only=args.only, sizes=tuple(args.sizes))


if __name__ == "__main__":
    main()
