"""RQ2: the gate's decisions and check time with the zeroing prologue (design decision 46).

The gate refuses a kernel that does not begin with the prologue that clears
its shared and local memory (`opgate.gate.zeroed`), or that may read a
register before writing it. This experiment takes every kernel that a
complete registration record (`eval.rq4_cost.registration`) accepts and
decides it twice, in turn on one worker: the registration's archived Rocq
query with the baseline library, which must be the library that made the
registration, and the query of the kernel's text with its prologue with the
current library. Both run the registration's proof query (`CHECK_QUERY`) in
one `coqc` process with the same time limit. The binding is the corpus
adapter's. The record keeps, for each kernel, both outcomes, the reason of a
refusal, both times, and the lines of the prologue; the Rocq terms that the
current library decides are retained as an artifact of every kernel, and the
text with its prologue when the prologue changes the text.

Run from the repository root, on the proof cores and without a GPU:

    CUDA_VISIBLE_DEVICES= taskset -c 32-47 uv run python -m eval.rq2_coverage.zeroing \\
        --registration eval/records/registration_<date>.json \\
        --baseline formal/build/library.<registration's> --library formal/build/current \\
        --output eval/records/zeroing_<date>.json --workers 14
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, TypedDict

from eval import ROOT
from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from eval.corpus.adapter import binding_of
from eval.rq2_coverage.rule_ablation import REGISTRATION_KIND, same_library
from opgate.gate import CHECK_QUERY, TIMEOUT, export_terms, zeroed

__all__ = ["main", "measure"]

KIND = "zeroing prologue decisions of a registration's accepted kernels"
WORKERS = 14
# the reason that the proof query prints for a refusal (opgate.gate.verify)
_REFUSAL = re.compile(r'OPGATE_REFUSAL\s*\(?Some\s*"((?:[^"]|"")*)"')

Outcome = Literal["accepted", "refused", "timeout"]


class Row(TypedDict):
    id: str
    tasks: str
    verdict: str
    prologue: int  # the lines that the prologue adds, declarations included
    baseline: Outcome
    baseline_reason: str
    baseline_seconds: float
    zeroed: Outcome
    zeroed_reason: str
    zeroed_seconds: float


class Kernel(TypedDict):
    id: str
    tasks: str
    verdict: str
    ptx: str
    terms: str


def _decided(library: Path, query: str, timeout: float) -> tuple[Outcome, str, float]:
    """The outcome, the refusal's reason, and the seconds of one proof query."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Query.v"
        path.write_text(query)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                ["coqc", "-q", "-Q", str(library), "OpGate", str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "", time.perf_counter() - started
        seconds = time.perf_counter() - started
    if result.returncode == 0:
        return "accepted", "", seconds
    found = _REFUSAL.search(result.stdout)
    if found is None:
        raise RuntimeError(f"Rocq refused without a reason: {result.stderr[-2000:]}")
    return "refused", found.group(1).replace('""', '"'), seconds


def kernels(registration: Path, baseline: Path) -> list[Kernel]:
    """The registration's accepted kernels with their archived PTX and terms."""
    record = json.loads(registration.read_text())
    provenance = record["provenance"]
    if (
        record["schema"] != 2
        or record["complete"] is not True
        or provenance["kind"] != REGISTRATION_KIND
        or provenance["query_suffix"] != CHECK_QUERY
    ):
        raise ValueError(
            f"{registration}: a complete registration record of the current query is needed"
        )
    same_library(registration, baseline)
    chosen: list[Kernel] = []
    for row in sorted(record["results"], key=lambda r: r["id"]):
        if row["accepted"] is not True:
            continue
        ptx = read_artifact(registration, row["ptx_artifact"], "ptx")
        terms = read_artifact(registration, row["query_artifact"], "v")
        chosen.append(
            Kernel(id=row["id"], tasks=row["tasks"], verdict=row["verdict"], ptx=ptx, terms=terms)
        )
    return chosen


def _row(kernel: Kernel, record: Checkpoint, baseline: Path, library: Path, timeout: float) -> Row:
    model, task, toolchain = kernel["id"].split("/")
    binding = binding_of(model, task, toolchain, kernel["ptx"])
    text = zeroed(kernel["ptx"], binding)
    terms = export_terms(text, [binding])
    if text != kernel["ptx"]:
        record.artifact(kernel["id"] + "/zeroed", text, "ptx")
    record.artifact(kernel["id"] + "/zeroed", terms, "v")
    before, before_reason, before_seconds = _decided(
        baseline, kernel["terms"] + CHECK_QUERY, timeout
    )
    after, after_reason, after_seconds = _decided(library, terms + CHECK_QUERY, timeout)
    print(
        f"{kernel['id']}: {before} {before_seconds:.1f} s, zeroed {after} {after_seconds:.1f} s",
        flush=True,
    )
    return Row(
        id=kernel["id"],
        tasks=kernel["tasks"],
        verdict=kernel["verdict"],
        prologue=len(text.split("\n")) - len(kernel["ptx"].split("\n")),
        baseline=before,
        baseline_reason=before_reason,
        baseline_seconds=before_seconds,
        zeroed=after,
        zeroed_reason=after_reason,
        zeroed_seconds=after_seconds,
    )


def _sources(baseline: Path, library: Path) -> dict[str, Path]:
    files = {f"baseline/{p.name}": p for p in baseline.glob("*.v")}
    files.update({f"library/{p.name}": p for p in library.glob("*.v")})
    for name in (
        "opgate/gate/__init__.py",
        "opgate/gate/prologue.py",
        "eval/checkpoint.py",
        "eval/corpus/adapter.py",
        "eval/rq2_coverage/rule_ablation.py",
        "eval/rq2_coverage/zeroing.py",
    ):
        files[name] = ROOT / name
    return files


def summary(rows: list[Row]) -> dict[str, object]:
    """The counts of each pair of outcomes, and the median times of the kernels
    that both decisions accept."""
    pairs: dict[str, int] = {}
    for row in rows:
        key = f"{row['baseline']}/{row['zeroed']}"
        pairs[key] = pairs.get(key, 0) + 1
    both = [r for r in rows if r["baseline"] == "accepted" and r["zeroed"] == "accepted"]
    changed = [r for r in both if r["prologue"] > 0]
    return {
        "outcomes": pairs,
        "median_baseline_seconds": statistics.median(r["baseline_seconds"] for r in both),
        "median_zeroed_seconds": statistics.median(r["zeroed_seconds"] for r in both),
        "median_ratio_with_prologue": statistics.median(
            r["zeroed_seconds"] / r["baseline_seconds"] for r in changed
        ),
        "median_ratio_without_prologue": statistics.median(
            r["zeroed_seconds"] / r["baseline_seconds"] for r in both if r["prologue"] == 0
        ),
    }


def measure(
    registration: Path, baseline: Path, library: Path, output: Path, *, workers: int, resume: bool
) -> None:
    chosen = kernels(registration, baseline.resolve())
    provenance = {
        "kind": KIND,
        "registration": str(registration),
        "baseline": str(baseline.resolve()),
        "library": str(library.resolve()),
        "cores": sorted(os.sched_getaffinity(0)),
        "workers": workers,
        "timeout": TIMEOUT,
        "query_suffix": CHECK_QUERY,
        "kernels": [k["id"] for k in chosen],
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        sources = _sources(baseline.resolve(), library.resolve())
        snapshot_inputs(output, "sources", sources, create=not resume)
        pending = [k for k in chosen if record.get(k["id"]) is None]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_row, k, record, baseline.resolve(), library.resolve(), TIMEOUT)
                for k in pending
            ]
            for future in as_completed(futures):
                record.add(dict(future.result()))
        snapshot_inputs(
            output, "sources", _sources(baseline.resolve(), library.resolve()), create=False
        )
        rows: list[Row] = [Row(**r) for r in record.data["results"]]
        record.finish(summary=summary(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    measure(
        args.registration,
        args.baseline,
        args.library,
        args.output,
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
