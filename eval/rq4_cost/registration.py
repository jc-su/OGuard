"""Measure one frozen corpus with raw PTX, bindings, reviews and proof sources.

Each kernel's text as written, with its zeroing prologue, is checked under the
service's budget of one registration (`VerificationBudget` of
`REGISTRATION_SECONDS`, or `--seconds`), which the record states. Use a new
--output record, or --resume with byte-identical inputs. Every kernel is
reviewed before checking; a complete record includes every selected PTX,
including kernels not yet entered into a historical decision table.
"""

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from eval import ROOT
from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.corpus.adapter import COLLECTIONS, binding_of, directory_of
from eval.corpus.run import TOOLCHAINS, Key, kernels, read_table, read_verdicts, tasks_of
from opgate.binding import Binding
from opgate.gate import (
    CHECK_QUERY,
    REGISTRATION_SECONDS,
    Accepted,
    Refused,
    VerificationBudget,
    export_terms,
    formal_dir,
    verify,
    zeroed,
)
from opgate.hosts.service.audit import binding_record

WORKERS = 8
KIND = "registration with raw verification inputs and the service's budget"


@dataclass(frozen=True)
class Input:
    key: Key
    ptx: str
    binding: Binding
    verdict: str
    baseline: str | None


def inputs() -> list[Input]:
    """Capture the reviewed inputs once, before any verification starts."""
    table, reviews = read_table(), read_verdicts()
    result = []
    for key in kernels(set()):
        model, task, toolchain = key
        previous = table.get(key)
        if key not in reviews and previous is None:
            raise ValueError(f"{key}: no reviewed classification")
        if key in reviews:
            verdict = reviews[key]
        else:
            assert previous is not None
            verdict = previous.verdict
        ptx = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_bytes().decode("ascii")
        result.append(
            Input(key, ptx, binding_of(*key, ptx), verdict, previous.expected if previous else None)
        )
    return result


def source_files(library: Path) -> dict[str, Path]:
    files = {f"formal/{p.name}": p for p in library.glob("*.v")}
    for name in (
        "opgate/signature.py",
        "opgate/binding/__init__.py",
        "opgate/gate/__init__.py",
        "opgate/frontends/__init__.py",
        "opgate/hosts/service/audit.py",
        "eval/checkpoint.py",
        "eval/corpus/adapter.py",
        "eval/corpus/run.py",
    ):
        files[name] = ROOT / name
    for collection in COLLECTIONS:
        files[str(collection.decisions.relative_to(ROOT))] = collection.decisions
        for model in collection.models:
            for name in ("roles.tsv", "verdicts.tsv"):
                path = collection.root / model / name
                if path.exists():
                    files[str(path.relative_to(ROOT))] = path
    files["eval/rq4_cost/registration.py"] = Path(__file__).resolve()
    return files


def _measured(item: Input, budget: float) -> dict:
    start = time.perf_counter()
    check = VerificationBudget(budget, checker=verify)
    decision = check(zeroed(item.ptx, item.binding), item.binding)
    seconds = time.perf_counter() - start
    key = "/".join(item.key)
    print(f"{key}: {seconds:.1f} s", flush=True)
    return {
        "id": key,
        "accepted": isinstance(decision, Accepted),
        "reason": decision.reason if isinstance(decision, Refused) else "",
        "seconds": seconds,
        "ptx_bytes": len(item.ptx.encode("ascii")),
        "binding": binding_record(item.binding),
        "verdict": item.verdict,
        "tasks": tasks_of(*item.key[:2]),
        "baseline": item.baseline,
        "ptx_artifact": key,
        "query_artifact": key,
    }


def _archive(record: Checkpoint, item: Input) -> None:
    """Publish both raw inputs durably before any verification starts: the PTX as
    the corpus holds it, and the query that the gate decides, of the text with
    its zeroing prologue, which registration loads."""
    key = "/".join(item.key)
    record.artifact(key, item.ptx, "ptx")
    record.artifact(key, export_terms(zeroed(item.ptx, item.binding), [item.binding]), "v")


def measure(
    output: Path,
    *,
    resume: bool = False,
    workers: int = WORKERS,
    seconds: float = REGISTRATION_SECONDS,
) -> None:
    if workers < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("workers and the registration budget must be positive and finite")
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < workers:
        raise ValueError(f"{workers} workers need at least {workers} cores")
    library = formal_dir()
    os.environ["OPGATE_FORMAL"] = str(library)
    captured = inputs()
    provenance = {
        "kind": KIND,
        "library": str(library),
        "cores": cores,
        "workers": workers,
        "registration_seconds": seconds,
        "query_suffix": CHECK_QUERY,
        "processes_per_verification": 1,
        "kernels": ["/".join(i.key) for i in captured],
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        sources = source_files(library)
        snapshot_inputs(output, "sources", sources, create=not resume)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepared = [pool.submit(_archive, record, item) for item in captured]
            try:
                for future in as_completed(prepared):
                    future.result()
            except BaseException:
                for future in prepared:
                    future.cancel()
                raise
            pending = [i for i in captured if record.get("/".join(i.key)) is None]
            for future in as_completed([pool.submit(_measured, item, seconds) for item in pending]):
                record.add(future.result())
        # A concurrent edit cannot silently become the input of a completed run.
        snapshot_inputs(output, "sources", source_files(library), create=False)
        if inputs() != captured:
            raise ValueError("corpus inputs changed during measurement; use a new record")
        record.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seconds", type=float, default=REGISTRATION_SECONDS)
    args = parser.parse_args()
    measure(args.output, resume=args.resume, workers=args.workers, seconds=args.seconds)


if __name__ == "__main__":
    main()
