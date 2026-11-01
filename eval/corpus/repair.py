"""The refusal-guided repair loop (design section 9.1, RQ2).

A refused corpus kernel goes back to the model that wrote it with the gate's
reason, in the words of `prompt`, and the model writes its revision in
`<model>/<task>/repair<k>/` as `numba.py`, `kernel.cu`, or `triton.py`,
tested against its reference as the first kernel was. Round k revises the kernels that round
k - 1 left refused, with that round's reason. A revision that no prompt of
its round asks for is an error. A later gate can accept a kernel that a
prompt asked for, as decisions 24 and 25 did for nine of Opus's kernels; the
revision is then superseded, and the loop reports it and leaves it out.

Run from the repository root:

    uv run python -m eval.corpus.repair --output eval/records/repairs_<date>.json

recompiles originals and revisions from source, verifies them with one explicit
per-kernel timeout, and checks accepted kernels on inputs of 1, 1000, and
100003 rows. Each outcome is written atomically as it completes. ``--resume``
requires identical sources, proofs, toolchains, selected kernels, and settings.
The record retains timeouts, incorrect results, superseded revisions, missing
revision sources, timings, and exact source/PTX files. Known-defect originals are
never launched; acceptance stops the run. Historical TSV tables stay frozen.

    uv run python -m eval.corpus.repair --prompts

writes, for each set of tasks and each model, the prompt of its next round,
`prompts/<tasks>/round<k>/<model>.<toolchains>.txt`: each kernel that its
last round left refused or invalid, with the reason, for at most `ROUNDS`
rounds. The prompts of each round are kept as the models received them, and
a corpus that joins later, such as Triton's or the held-out tasks', gets
prompts of its own.
"""

import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, assert_never

import numpy as np

from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from eval.corpus.adapter import SYNTHESIZED, claim_of
from eval.corpus.run import SPECS, TOOLCHAINS, WORKERS, Tasks, read_table, tasks_of
from eval.corpus.sources import SOURCES, interface_of, operator_of, stated_interface
from eval.corpus.workloads import Loaded, check, run, workload_of
from opgate.catalog import TenantId
from opgate.executor import Executor
from opgate.frontends import Operator
from opgate.gate import (
    TIMEOUT,
    TIMEOUT_REASON,
    Accepted,
    Refused,
    export_terms,
    formal_dir,
    verify,
    zeroed,
)

__all__ = ["Revision", "environment", "prompt", "revisions"]


def environment() -> str:
    """The environment that the prompt names: this Python and the `nvcc` on the path."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("no nvcc on the path")
    return f"""The Python environment is {sys.executable}, which
has numba-cuda 0.30.4 (numba 0.64: `from numba import cuda`), Triton 3.8.0
with PyTorch 2.14, numpy, and cupy. nvcc is {nvcc}
(CUDA 13.0). Use GPU 1 only: prefix every command that uses the GPU with
`CUDA_VISIBLE_DEVICES=1`. A file named `numba.py` or `triton.py` in the
working directory shadows its package: run Python tests from the model's
directory, not a task directory, or with `python -P`, and load a task's
source by path with importlib."""


# the loop asks the models of the synthesized collection, which wrote its kernels
CORPUS = SYNTHESIZED.root
TABLE = CORPUS / "repairs.tsv"
COLUMNS = ("model", "task", "toolchain", "round", "decision", "reason", "result")
ROWS = (1, 1000, 100003)
TENANT = TenantId(1)
ROUNDS = 2  # the rounds of the loop: a kernel refused after round 2 is refused

Key = tuple[str, str, str]  # a model, a task, and a toolchain
# a set of tasks, a round, a model, and the toolchains of its kernels
Ask = tuple[Tasks, int, str, tuple[str, ...]]
Decision = Literal["accepted", "refused", "invalid"]


@dataclass(frozen=True)
class Revision:
    model: str
    task: str
    toolchain: str
    round: int

    @property
    def key(self) -> Key:
        return (self.model, self.task, self.toolchain)

    @property
    def directory(self) -> Path:
        return source_of(*self.key, self.round).parent


@dataclass(frozen=True)
class Outcome:
    """A revision's decision, the reason of a refusal or an invalid revision,
    and the result of an accepted one."""

    decision: Decision
    reason: str
    result: str  # correct, wrong, or - for a revision the gate did not accept


def source_of(model: str, task: str, toolchain: str, round: int) -> Path:
    """The source of a kernel's round: the first kernel for round 0, and the
    round's revision after it."""
    return CORPUS / model / task / (f"repair{round}" if round else "") / SOURCES[toolchain]


def opening(toolchain: str, source: Path, task: str) -> str:
    """The first line of a prompt, which names the source it asks to revise."""
    return f"You wrote the {toolchain} kernel {source} for the task `{task}` of"


def prompt(model: str, task: str, toolchain: str, source: Path, reason: str, round: int) -> str:
    """What the model that wrote a refused kernel receives."""
    target = CORPUS / model / task / f"repair{round}"
    spec = SPECS[tasks_of(model, task)]
    return f"""{opening(toolchain, source, task)}
{CORPUS / spec}. A static checker of GPU memory accesses refused it
with this reason:

    {reason}

The checker accepts a kernel only if it can prove that every memory access
stays inside its buffer for every input that meets the task's contract,
whatever the output buffers hold on entry, and for any grid. Revise the
kernel so that the checker can prove it, keeping the task's semantics and
the arguments that your signature.json states. Write the revision as {target / SOURCES[toolchain]},
test it on GPU 1 (`CUDA_VISIBLE_DEVICES=1`) against the reference as you
tested the first kernel, and do not change any other file. Read only
{spec} and your own files of this task."""


def revisions(round: int) -> list[Revision]:
    """Every revision of round `round` in the corpus, in order."""
    found = []
    for path in sorted(CORPUS.glob(f"*/*/repair{round}")):
        model, task = path.parent.parent.name, path.parent.name
        for toolchain, source in SOURCES.items():
            if (path / source).is_file():
                found.append(Revision(model, task, toolchain, round))
    return found


def asked(revision: Revision) -> bool:
    """Whether a prompt of the revision's round asks for it: the prompt names
    the source that the revision revises."""
    model, task, toolchain = revision.key
    line = opening(toolchain, source_of(model, task, toolchain, revision.round - 1), task)
    directory = CORPUS / "prompts" / tasks_of(model, task) / f"round{revision.round}"
    prompts = directory.glob(f"{model}.*.txt")
    return any(line in path.read_text() for path in prompts)


def compiled(revision: Revision) -> Operator | Outcome:
    """The revision's operator, or why it is invalid: its arguments are not the
    task's, which `signature.json` states, or its source does not compile."""
    model, task, toolchain = revision.key
    stated = stated_interface(model, task)
    # an untrusted source can fail to load or compile in any way; the failure
    # is the revision's recorded outcome
    try:
        revised = interface_of(model, task, toolchain, revision.directory)
    except Exception as error:
        return Outcome("invalid", f"the source does not compile: {str(error).splitlines()[0]}", "-")
    if revised != stated:
        return Outcome("invalid", f"the arguments {revised} are not the task's {stated}", "-")
    try:
        return operator_of(model, task, toolchain, revision.directory)
    except Exception as error:
        return Outcome("invalid", f"the source does not compile: {str(error).splitlines()[0]}", "-")


def result(executor: Executor, revision: Revision, operator: Operator) -> str:
    """Whether the accepted revision gives its task's result at each size."""
    model, task, toolchain = revision.key
    module = executor.register(operator.text, operator.binding(claim_of(model, task)[1]))
    kernel = Loaded(model, task, toolchain, module)
    rng = np.random.default_rng(1)
    for rows in ROWS:
        w = workload_of(executor, TENANT, model, task, rows, rng)
        if check(task, w, run(executor, TENANT, kernel, w)) is not None:
            return "wrong"
    return "correct"


def gate(
    revision: Revision, operator: Operator | Outcome, *, timeout: float = TIMEOUT
) -> Operator | Outcome:
    """The operator of a compiled revision that the gate accepts, or the outcome
    of one it refuses or that is invalid."""
    if isinstance(operator, Outcome):
        return operator
    binding = operator.binding(claim_of(revision.model, revision.task)[1])
    match verify(zeroed(operator.text, binding), [binding], timeout=timeout):
        case Accepted():
            return operator
        case Refused(reason=reason):
            return Outcome("refused", reason, "-")
        case _ as never:
            assert_never(never)


def next_prompts() -> dict[Ask, str]:
    """For each round up to `ROUNDS` and each model, the prompt of the model's
    kernels that their last round left refused or invalid, with its reasons:
    round 0 is the decision table's, and the later rounds are those of
    `repairs.tsv`. The prompt's key names the set of its tasks and the
    toolchains of its kernels."""
    last: dict[Key, tuple[int, str, str]] = {
        key: (0, row.expected, row.reason)
        for key, row in read_table().items()
        if row.model in SYNTHESIZED.models
    }
    with TABLE.open(newline="") as handle:
        for line in csv.DictReader(handle, delimiter="\t"):
            key = (line["model"], line["task"], line["toolchain"])
            round = int(line["round"])
            if round > last[key][0]:
                last[key] = (round, line["decision"], line["reason"])
    asks: dict[tuple[Tasks, int, str], list[tuple[str, str]]] = {}
    for (model, task, toolchain), (round, decision, reason) in sorted(last.items()):
        if decision == "accepted" or round == ROUNDS:
            continue
        text = prompt(
            model, task, toolchain, source_of(model, task, toolchain, round), reason, round + 1
        )
        asks.setdefault((tasks_of(model, task), round + 1, model), []).append((toolchain, text))
    return {
        (
            tasks,
            round,
            model,
            tuple(t for t in SOURCES if t in {toolchain for toolchain, _ in items}),
        ): (
            f"{environment()}\n\nRevise each of these kernels:\n\n"
            + "\n\n---\n\n".join(text for _, text in items)
            + "\n\nWhen done, reply with the revisions you wrote and whether each passed its test."
        )
        for (tasks, round, model), items in asks.items()
    }


def prompt_path(ask: Ask) -> Path:
    """`prompts/<tasks>/round<k>/<model>.<toolchains>.txt`, such as
    `development/round1/fable.numba-nvcc.txt`."""
    tasks, round, model, toolchains = ask
    name = f"{model}.{'-'.join(toolchains)}.txt"
    return CORPUS / "prompts" / tasks / f"round{round}" / name


def write_prompts() -> None:
    """Write each prompt of `next_prompts` that is not written yet.

    A prompt that is written already stays as the model received it, even
    when a later gate gives other reasons or accepts one of its kernels, and
    no other prompt of its round and model may ask for a kernel of its
    toolchains, so no kernel is asked twice in one round."""
    for ask, text in next_prompts().items():
        path = prompt_path(ask)
        _, round, model, toolchains = ask
        for other in path.parent.glob(f"{model}.*.txt"):
            asked = other.name.split(".")[1].split("-")
            if other != path and set(asked) & set(toolchains):
                raise ValueError(f"{other} already asks round {round} of {model}'s {asked} kernels")
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(path)


def _id(revision: Revision) -> str:
    return f"{'/'.join(revision.key)}/{revision.round}"


def input_files(selected: list[Key]) -> dict[str, Path]:
    """The actual sources and declarations retained with a measurement."""
    root = Path(__file__).resolve().parents[2]
    paths = set((root / "opgate").rglob("*.py"))
    # Archived inputs contain copies of these Python files. Traversing records
    # would copy every earlier snapshot into each new one, growing recursively.
    excluded = {root / "eval" / "records", CORPUS}
    for directory, children, filenames in (root / "eval").walk():
        children[:] = [
            name for name in children if directory / name not in excluded and name != "__pycache__"
        ]
        paths.update(directory / name for name in filenames if name.endswith(".py"))
    paths.update(
        (
            root / "pyproject.toml",
            root / "uv.lock",
            root / "formal" / "Makefile",
            root / "formal" / "__main__.py",
            CORPUS / "decisions.tsv",
        )
    )
    paths.update(CORPUS / name for name in SPECS.values())
    paths.update((CORPUS / "prompts").rglob("*.txt"))
    for model, task, toolchain in selected:
        paths.add(CORPUS / model / "roles.tsv")
        paths.add(CORPUS / model / task / "signature.json")
        # The probe workload uses a recorded build operator as its fixture.
        if task == "hash_probe":
            paths.add(CORPUS / model / "hash_build" / "signature.json")
            paths.add(CORPUS / model / "hash_build" / TOOLCHAINS["numba"])
        for round in range(ROUNDS + 1):
            source = source_of(model, task, toolchain, round)
            if source.is_file():
                paths.add(source)
                launch = source.parent / "launch.json"
                if launch.exists():
                    paths.add(launch)
    return {str(p.relative_to(root)): p for p in sorted(paths)}


def proof_files(library: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(library.iterdir()) if p.suffix in (".v", ".vo")}


def provenance(selected: list[Key], timeout: float, workers: int) -> dict:
    """Record toolchain versions and settings; inputs are ordinary archived files."""
    versions = {
        name: version(name)
        for name in ("numba", "numba-cuda", "triton", "torch", "numpy", "cuda-bindings")
    }
    for name in ("coqc", "nvcc"):
        versions[name] = subprocess.run(
            [name, "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "library": str(formal_dir().resolve()),
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpu,
        "selected": [list(k) for k in selected],
        "timeout": timeout,
        "workers": workers,
        "rows": list(ROWS),
        "rounds": ROUNDS,
        "baseline": "recompiled from source; historical classifications retained",
        "artifacts": "exact PTX and exported Rocq terms, named by kernel/revision",
    }


def measure(
    revision: Revision, operator: Operator | Outcome, timeout: float
) -> tuple[Operator | Outcome, float]:
    started = time.monotonic()
    return gate(revision, operator, timeout=timeout), time.monotonic() - started


def evaluate(
    found: list[Revision], executor: Executor, record: Checkpoint, *, timeout: float, workers: int
) -> None:
    """Compile on the owner thread, verify concurrently, save every completion.

    GPU validation stays on the executor's owner thread. Completed refusals
    are saved even if another proof is still running or a later check fails.
    """
    table = read_table()

    def save(revision, outcome, compile_seconds, verify_seconds, result_seconds, ptx, query):
        row = {
            "id": _id(revision),
            "model": revision.model,
            "task": revision.task,
            "toolchain": revision.toolchain,
            "round": revision.round,
            "verdict": table[revision.key].verdict,
            "decision": outcome.decision,
            "reason": outcome.reason,
            "result": outcome.result,
            "timeout": outcome.reason.startswith(TIMEOUT_REASON),
            "compile_seconds": compile_seconds,
            "verify_seconds": verify_seconds,
            "result_seconds": result_seconds,
            "ptx_artifact": ptx,
            "query_artifact": query,
        }
        record.add(row)
        print(
            f"{row['id']}: {outcome.decision}, {outcome.result}, {verify_seconds:.2f}s", flush=True
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def collect(future):
            revision, compile_seconds, ptx, query = pending.pop(future)
            checked, verify_seconds = future.result()
            if isinstance(checked, Outcome):
                save(revision, checked, compile_seconds, verify_seconds, 0.0, ptx, query)
                return
            if revision.round == 0 and table[revision.key].verdict != "safe":
                save(
                    revision,
                    Outcome("accepted", "", "not_run"),
                    compile_seconds,
                    verify_seconds,
                    0.0,
                    ptx,
                    query,
                )
                raise RuntimeError(
                    f"{_id(revision)}: a known defect was accepted; device execution stopped"
                )
            started = time.monotonic()
            outcome = Outcome("accepted", "", result(executor, revision, checked))
            save(
                revision,
                outcome,
                compile_seconds,
                verify_seconds,
                time.monotonic() - started,
                ptx,
                query,
            )

        for revision in found:
            if (row := record.get(_id(revision))) is not None:
                for key, kind in (("ptx_artifact", "ptx"), ("query_artifact", "v")):
                    if row[key] is not None:
                        read_artifact(record.path, row[key], kind)
                continue
            started = time.monotonic()
            operator = compiled(revision)
            compile_seconds = time.monotonic() - started
            if isinstance(operator, Outcome):
                save(revision, operator, compile_seconds, 0.0, 0.0, None, None)
            else:
                ptx = record.artifact(_id(revision), operator.text, "ptx")
                binding = operator.binding(claim_of(revision.model, revision.task)[1])
                query = record.artifact(
                    _id(revision), export_terms(zeroed(operator.text, binding), [binding]), "v"
                )
                pending[pool.submit(measure, revision, operator, timeout)] = (
                    revision,
                    compile_seconds,
                    ptx,
                    query,
                )
            # Bound queued operators as well as running proof processes.
            if len(pending) >= workers:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    collect(future)
        for future in as_completed(pending):
            collect(future)


def main(argv: list[str]) -> int:
    if argv == ["--prompts"]:
        write_prompts()
        return 0
    parser = argparse.ArgumentParser(
        description="Recompile, verify, and validate the repair corpus with durable results."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("kernels", nargs="*", help="optional model/task[/toolchain] selection")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.workers <= 0:
        parser.error("timeout and workers must be positive; timeout must be finite")
    table = read_table()
    selected = sorted(
        k
        for k in table
        if not args.kernels or "/".join(k) in args.kernels or "/".join(k[:2]) in args.kernels
    )
    for named in args.kernels:
        if not any(named in ("/".join(k), "/".join(k[:2])) for k in selected):
            parser.error(f"unknown kernel selection: {named}")
    planned = [Revision(*key, 0) for key in selected]
    for round in range(1, ROUNDS + 1):
        planned.extend(r for r in revisions(round) if r.key in selected)
    if revisions(ROUNDS + 1):
        raise ValueError(f"revisions exceed the declared {ROUNDS}-round budget")
    unasked = [r for r in planned if r.round and not asked(r)]
    if unasked:
        raise ValueError(f"revisions without frozen prompts: {unasked}")
    frozen = provenance(selected, args.timeout, args.workers)
    with Checkpoint(args.output, frozen, resume=args.resume) as record:
        sources = input_files(selected)
        proofs = proof_files(formal_dir())
        snapshot_inputs(args.output, "sources", sources, create=not args.resume)
        snapshot_inputs(args.output, "proofs", proofs, create=not args.resume)
        if record.data["complete"]:
            validate_record(record.data)
            validate_artifacts(args.output, record.data)
            print(f"{args.output}: already complete")
            return 0
        active = set(selected)
        missing = []
        with Executor(0, 1 << 28) as executor:
            with executor.tenant(TENANT):
                executor.reserve(1 << 28)
            for round in range(ROUNDS + 1):
                found = [r for r in planned if r.round == round]
                for r in found:
                    if r.key not in active and record.get(_id(r)) is None:
                        if not any(
                            row["decision"] == "accepted"
                            and (row["model"], row["task"], row["toolchain"]) == r.key
                            for row in record.data["results"]
                        ):
                            raise ValueError(f"{_id(r)} has no completed predecessor revision")
                        record.add(
                            {
                                "id": _id(r),
                                "model": r.model,
                                "task": r.task,
                                "toolchain": r.toolchain,
                                "round": round,
                                "decision": "superseded",
                                "reason": "an earlier round was accepted",
                                "result": "-",
                                "verdict": table[r.key].verdict,
                            }
                        )
                missing.extend(
                    f"{'/'.join(k)}/{round}" for k in sorted(active - {r.key for r in found})
                )
                evaluate(
                    [r for r in found if r.key in active],
                    executor,
                    record,
                    timeout=args.timeout,
                    workers=args.workers,
                )
                active = set()
                for r in found:
                    row = record.get(_id(r))
                    if row is None:
                        raise RuntimeError(f"missing completed outcome: {_id(r)}")
                    if row["decision"] == "accepted" and row["result"] not in ("correct", "wrong"):
                        raise RuntimeError(
                            f"accepted kernel has no completed correctness check: {_id(r)}"
                        )
                    if row["decision"] in ("refused", "invalid"):
                        active.add(r.key)
        if provenance(selected, args.timeout, args.workers) != frozen:
            raise RuntimeError(
                "experiment inputs changed during the run; record remains incomplete"
            )
        snapshot_inputs(args.output, "sources", sources, create=False)
        snapshot_inputs(args.output, "proofs", proofs, create=False)
        expected = sorted(_id(r) for r in planned)
        if sorted(r["id"] for r in record.data["results"]) != expected:
            raise RuntimeError("checkpoint does not match the planned revisions")
        validate_artifacts(args.output, record.data)
        record.finish(planned=expected, missing_revisions=missing)
        print(f"{args.output}: {len(expected)} recorded outcomes; {len(missing)} missing revisions")
    return 0


def validate_artifacts(path: Path, record: dict) -> None:
    for row in record["results"]:
        if row["decision"] in ("accepted", "refused"):
            for key, kind in (("ptx_artifact", "ptx"), ("query_artifact", "v")):
                read_artifact(path, row[key], kind)


def validate_record(record: dict) -> None:
    """Do not publish an aggregate from partial, duplicate, or unvalidated data."""
    if record["schema"] != 2 or not record["complete"]:
        raise ValueError("the repair evaluation is incomplete")
    ids = [r["id"] for r in record["results"]]
    if len(ids) != len(set(ids)) or sorted(ids) != record["planned"]:
        raise ValueError("repair outcomes do not match the planned revisions")
    originals = {
        (r["model"], r["task"], r["toolchain"]) for r in record["results"] if r["round"] == 0
    }
    if originals != {tuple(k) for k in record["provenance"]["selected"]}:
        raise ValueError("repair record does not cover its selected original kernels")
    for row in record["results"]:
        if row["decision"] not in ("accepted", "refused", "invalid", "superseded"):
            raise ValueError(f"invalid outcome: {row['id']}")
        if row["decision"] == "accepted" and row["result"] not in ("correct", "wrong"):
            raise ValueError(f"accepted kernel has no correctness result: {row['id']}")
        if row["round"] == 0 and row["verdict"] != "safe" and row["decision"] == "accepted":
            raise ValueError(f"known defect accepted: {row['id']}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
