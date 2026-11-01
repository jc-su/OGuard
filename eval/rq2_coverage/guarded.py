"""Measure registration as the service decides it, and retain every checked input.

Each entry is decided as the service registers it: the gate checks the text
as written with its zeroing prologue, and, if the gate refuses it for a
reason that the guard pass addresses (`guard.applies`), the pass's rounds;
all the checks of one entry share one budget of `REGISTRATION_SECONDS`
(`VerificationBudget`), which the record states. The record is resumable and
includes unchanged, rewrite-only, inlined and instrumented kernels. Original
PTX and bindings come from the frozen registration, never from a newly
derived corpus binding. No kernel is launched by this experiment.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval import ROOT
from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from opgate.binding import Binding
from opgate.frontends.guard import Guarded, applies, guard
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

__all__ = ["main", "measure", "validate_evidence"]

WORKERS = 16
KIND = "registration as the service decides it, with raw inputs and one budget"


def source_files() -> dict[str, Path]:
    files = {f"formal/{p.name}": p for p in formal_dir().glob("*.v")}
    for p in (ROOT / "opgate").rglob("*.py"):
        files[str(p.relative_to(ROOT))] = p
    for name in ("eval/checkpoint.py", "eval/rq2_coverage/guarded.py"):
        files[name] = ROOT / name
    return files


def _archive(record: Checkpoint, key: str, text: str, binding: Binding) -> None:
    """The text, and the query that the gate decides: of the text with its
    zeroing prologue, which registration loads."""
    record.artifact(key, text, "ptx")
    record.artifact(key, export_terms(zeroed(text, binding), (binding,)), "v")


def _one(entry: dict, text: str, record: Checkpoint, seconds: float) -> dict:
    identifier = entry["id"]
    binding = parse_binding(entry["binding"])
    _archive(record, identifier + "/original", text, binding)
    decisions: list[Accepted | Refused] = []
    started = time.monotonic()
    budget = VerificationBudget(seconds, checker=verify)

    def check(candidate: str, bound: Binding) -> Accepted | Refused:
        decision = budget(candidate, bound)
        decisions.append(decision)
        return decision

    first = check(zeroed(text, binding), binding)
    if isinstance(first, Refused) and applies(first.reason):
        guarded = guard(text, binding, check)
        # The pass's first check is the check of the text as written again,
        # which the gate's cache answers unless the budget ended with the first.
        if decisions[1] == first:
            del decisions[1]
    else:
        guarded = Guarded(text, binding, (), (), False, first)
    elapsed = time.monotonic() - started
    # Always retain the exact final input, even when only participation or
    # device-function inlining changed it, or when no change was necessary.
    _archive(record, identifier + "/final", guarded.text, guarded.binding)
    last = guarded.decision
    return {
        "id": identifier,
        "label": entry["verdict"],
        "recorded": "accepted" if entry["accepted"] else "refused",
        "first": "accepted" if isinstance(first, Accepted) else "refused",
        "first_reason": first.reason if isinstance(first, Refused) else None,
        "decision": "accepted" if isinstance(last, Accepted) else "refused",
        "reason": last.reason if isinstance(last, Refused) else None,
        "guarded": list(guarded.guarded),
        "limited": list(guarded.limited),
        "restructured": guarded.restructured,
        "binding": binding_record(guarded.binding),
        "checks": len(decisions),
        "seconds": elapsed,
    }


def measure(
    registration: Path,
    output: Path,
    *,
    every: bool = False,
    only: list[str] | None = None,
    workers: int = WORKERS,
    seconds: float = REGISTRATION_SECONDS,
    resume: bool = False,
) -> None:
    if workers < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("workers and registration seconds must be positive and finite")
    original = json.loads(registration.read_text())
    if original.get("complete") is not True:
        raise ValueError("the registration record is not complete")
    all_entries = original["results"]
    if len({e["id"] for e in all_entries}) != len(all_entries):
        raise ValueError("the registration has duplicate entries")
    entries = [e for e in all_entries if every or not e["accepted"]]
    if only is not None:
        if len(set(only)) != len(only) or set(only) - {e["id"] for e in entries}:
            raise ValueError("selection has duplicate, missing or excluded entries")
        entries = [e for e in entries if e["id"] in only]
    if not entries:
        raise ValueError("the selection is empty")
    entries.sort(key=lambda e: e["id"])
    inputs = {"registration.json": registration}
    texts = {}
    for entry in entries:
        key = entry["id"]
        if entry["ptx_artifact"] != key or entry["query_artifact"] != key:
            raise ValueError(f"{key}: registration artifact names differ")
        text = read_artifact(registration, key, "ptx")
        bound = parse_binding(entry["binding"])
        if read_artifact(registration, key, "v") != export_terms(zeroed(text, bound), (bound,)):
            raise ValueError(f"{key}: registration query differs from its PTX and binding")
        texts[key] = text
        inputs[key + ".ptx"] = registration.with_suffix(registration.suffix + ".artifacts") / (
            key + ".ptx"
        )
    provenance = {
        "kind": KIND,
        "registration": str(registration),
        "kernels": [e["id"] for e in entries],
        "workers": workers,
        "registration_seconds": seconds,
    }
    with Checkpoint(output, provenance, resume=resume) as record:
        snapshot_inputs(output, "sources", source_files(), create=not resume)
        snapshot_inputs(output, "registration", inputs, create=not resume)
        if record.data["complete"]:
            validate_evidence(output, registration=registration)
            return
        pending = [e for e in entries if record.get(e["id"]) is None]
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(_one, e, texts[e["id"]], record, seconds) for e in pending]
            try:
                for future in as_completed(futures):
                    row = future.result()
                    record.add(row)
                    print(
                        f"{len(record.data['results'])}/{len(entries)} {row['id']} {row['decision']} ({row['seconds']:.1f} s)",
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        snapshot_inputs(output, "sources", source_files(), create=False)
        snapshot_inputs(output, "registration", inputs, create=False)
        record.data["results"].sort(key=lambda r: r["id"])
        record.finish()
    validate_evidence(output, registration=registration)


def validate_evidence(path: Path, *, registration: Path | None = None) -> dict:
    """Require a complete cohort, current raw sources, and both actual checker inputs."""
    record = json.loads(path.read_text())
    if record.get("schema") != 2 or record.get("complete") is not True:
        raise ValueError(f"{path}: guarded evidence is incomplete or lacks raw inputs")
    provenance = record["provenance"]
    if provenance["kind"] != KIND:
        raise ValueError(f"{path}: another experiment")
    snapshot_inputs(path, "sources", source_files(), create=False)
    archived = path.with_suffix(path.suffix + ".inputs") / "registration"
    original_bytes = (archived / "registration.json").read_bytes()
    if registration is not None and original_bytes != registration.read_bytes():
        raise ValueError(f"{path}: registration bytes differ")
    original = json.loads(original_bytes)
    if original.get("complete") is not True:
        raise ValueError(f"{path}: incomplete input registration")
    entries = {e["id"]: e for e in original["results"]}
    planned = provenance["kernels"]
    if len(set(planned)) != len(planned) or sorted(r["id"] for r in record["results"]) != sorted(
        planned
    ):
        raise ValueError(f"{path}: incomplete or duplicate guarded cohort")
    for row in record["results"]:
        key = row["id"]
        entry = entries[key]
        raw = (archived / (key + ".ptx")).read_bytes().decode("ascii")
        if read_artifact(path, key + "/original", "ptx") != raw:
            raise ValueError(f"{key}: original PTX differs from the registration")
        for phase, binding in (("original", entry["binding"]), ("final", row["binding"])):
            text = read_artifact(path, key + "/" + phase, "ptx")
            bound = parse_binding(binding)
            query = export_terms(zeroed(text, bound), (bound,))
            if read_artifact(path, key + "/" + phase, "v") != query:
                raise ValueError(f"{key}: {phase} query differs from its raw PTX and binding")
        if row["first"] == "accepted" and (
            read_artifact(path, key + "/final", "ptx") != raw or row["binding"] != entry["binding"]
        ):
            raise ValueError(f"{key}: an unchanged acceptance changed its input")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seconds", type=float, default=REGISTRATION_SECONDS)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--every", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    measure(
        args.registration,
        args.output,
        every=args.every,
        only=args.only,
        workers=args.workers,
        seconds=args.seconds,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
