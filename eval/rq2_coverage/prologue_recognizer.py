"""RQ2: the launch model's recognition of the zeroing prologue (design decision 50).

`prologue_of` in `formal/GatePrologue.v` reads the lowered zeroing prologue of
a kernel with shared variables off its program, and `check_entry_certificate`
requires it: the gate accepts such a kernel only with a recognized prologue.
This experiment takes every kernel that a complete registration record
(`eval.rq4_cost.registration`) accepts and evaluates, with the selected
library, `prologue_ok` on what `front` gives each entry kernel of the
registration's archived Rocq query, the text with its prologue and its
bindings as the gate decided them. The record keeps, for each kernel,
whether an entry kernel has shared variables, whether every entry kernel's
prologue is recognized, whether an entry kernel has local variables, the
printed result, and the time of the evaluation. A kernel with shared
variables whose prologue the model does not recognize would be refused by
the gate of the library. The kernels with local variables are those whose
local memory the launch model keeps at zero on the fact of decision 46.

Run from the repository root, on the proof cores and without a GPU:

    CUDA_VISIBLE_DEVICES= taskset -c 32-63 uv run python -m eval.rq2_coverage.prologue_recognizer \\
        --registration eval/records/registration_<date>.json \\
        --library formal/build/library.<the library> \\
        --output eval/records/prologue_recognizer_<date>.json --workers 16
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from eval import ROOT
from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from eval.rq2_coverage.rule_ablation import REGISTRATION_KIND
from opgate.gate import CHECK_QUERY, TIMEOUT

__all__ = ["main", "measure"]

KIND = "zeroing prologue recognition of a registration's accepted kernels"
WORKERS = 16

# For each entry kernel of the module: whether its launch facts list a shared
# variable, whether the launch model recognizes its prologue, and whether its
# launch facts list a local variable.
REPORT = """
Require Import OpGate.GatePrologue.
Definition report := Eval vm_compute in
  match parse_text text size with
  | Ok ks =>
      Some (map (fun e =>
                   match lookup mfs (fst e) with
                   | Some mf =>
                       match front (snd e) mf with
                       | Ok (prog, F) =>
                           (negb (Nat.eqb (List.length (lf_shared F)) 0), prologue_ok F mf prog,
                            negb (Nat.eqb (List.length (lf_locals F)) 0))
                       | Err _ _ => (false, false, false)
                       end
                   | None => (false, false, false)
                   end) (ks_entries ks))
  | Err _ _ => None
  end.
Print report.
"""
_REPORT = re.compile(r"report = (.*?)\n\s*: ", re.S)
_ENTRY = re.compile(r"\((true|false), (true|false), (true|false)\)")


class Row(TypedDict):
    id: str
    tasks: str
    verdict: str
    shared: bool  # an entry kernel has a shared variable
    recognized: bool  # every entry kernel's prologue is recognized
    local: bool  # an entry kernel has a local variable
    report: str  # what Rocq printed of `report`
    seconds: float


class Kernel(TypedDict):
    id: str
    tasks: str
    verdict: str
    terms: str


def _evaluated(library: Path, query: str, timeout: float) -> tuple[str, float]:
    """What Rocq prints of `report`, and the seconds of the evaluation."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Recognizer.v"
        path.write_text(query)
        started = time.perf_counter()
        result = subprocess.run(
            ["coqc", "-q", "-Q", str(library), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        seconds = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"Rocq did not evaluate the report: {result.stderr[-2000:]}")
    found = _REPORT.search(result.stdout)
    if found is None:
        raise RuntimeError(f"Rocq printed no report: {result.stdout[-2000:]}")
    return re.sub(r"\s+", " ", found.group(1)).strip(), seconds


def kernels(registration: Path) -> list[Kernel]:
    """The registration's accepted kernels with their archived terms."""
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
    chosen: list[Kernel] = []
    for row in sorted(record["results"], key=lambda r: r["id"]):
        if row["accepted"] is not True:
            continue
        terms = read_artifact(registration, row["query_artifact"], "v")
        if "Lemma accepted" in terms:
            raise ValueError(f"{registration}: the archived query of {row['id']} is not its terms")
        chosen.append(Kernel(id=row["id"], tasks=row["tasks"], verdict=row["verdict"], terms=terms))
    return chosen


def _row(kernel: Kernel, library: Path, timeout: float) -> Row:
    report, seconds = _evaluated(library, kernel["terms"] + REPORT, timeout)
    entries = _ENTRY.findall(report)
    if not entries or not report.startswith("Some"):
        raise RuntimeError(
            f"{kernel['id']}: the module's entry kernels were not evaluated: {report}"
        )
    shared = any(s == "true" for s, _, _ in entries)
    recognized = all(r == "true" for _, r, _ in entries)
    local = any(v == "true" for _, _, v in entries)
    print(
        f"{kernel['id']}: shared {shared}, recognized {recognized}, local {local}, {seconds:.1f} s",
        flush=True,
    )
    return Row(
        id=kernel["id"],
        tasks=kernel["tasks"],
        verdict=kernel["verdict"],
        shared=shared,
        recognized=recognized,
        local=local,
        report=report,
        seconds=seconds,
    )


def _sources(library: Path) -> dict[str, Path]:
    files = {f"library/{p.name}": p for p in library.glob("*.v")}
    for name in (
        "opgate/gate/__init__.py",
        "eval/checkpoint.py",
        "eval/rq2_coverage/rule_ablation.py",
        "eval/rq2_coverage/prologue_recognizer.py",
    ):
        files[name] = ROOT / name
    return files


def summary(rows: list[Row]) -> dict[str, object]:
    """The counts of kernels with shared variables, of recognized prologues, and of
    kernels with local variables."""
    with_shared = [r for r in rows if r["shared"]]
    return {
        "kernels": len(rows),
        "with_shared_variables": len(with_shared),
        "recognized": sum(1 for r in with_shared if r["recognized"]),
        "unrecognized": sorted(r["id"] for r in with_shared if not r["recognized"]),
        "with_local_variables": sum(1 for r in rows if r["local"]),
    }


def measure(registration: Path, library: Path, output: Path, *, workers: int, resume: bool) -> None:
    chosen = kernels(registration)
    provenance = {
        "kind": KIND,
        "registration": str(registration),
        "library": str(library.resolve()),
        "cores": sorted(os.sched_getaffinity(0)),
        "workers": workers,
        "timeout": TIMEOUT,
        "query_suffix": REPORT,
        "kernels": [k["id"] for k in chosen],
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        snapshot_inputs(output, "sources", _sources(library.resolve()), create=not resume)
        pending = [k for k in chosen if record.get(k["id"]) is None]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_row, k, library.resolve(), TIMEOUT) for k in pending]
            for future in as_completed(futures):
                record.add(dict(future.result()))
        snapshot_inputs(output, "sources", _sources(library.resolve()), create=False)
        rows: list[Row] = [Row(**r) for r in record.data["results"]]
        record.finish(summary=summary(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    measure(args.registration, args.library, args.output, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
