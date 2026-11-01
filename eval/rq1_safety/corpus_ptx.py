"""Run OpGate over PTX artifacts compiled from the 27 sources exercised by E3.

E3 imported each ``numba.py`` and JIT-compiled it, or compiled each
``kernel.cu`` to a temporary cubin. Its archive did not retain those executed
binaries. The corpus separately retains the level-3 ``numba.ptx`` and
``kernel.ptx`` artifacts generated from the same sources. This experiment
gates those corresponding artifacts and records their paths. It deliberately
does not claim byte identity with E3's execution.
"""

import argparse
import json
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, TypedDict, assert_never

from eval import RECORDS
from eval.corpus.adapter import SYNTHESIZED, binding_of, directory_of
from opgate.gate import Accepted, Refused, verify, zeroed

E3_ARCHIVE = RECORDS / "e3_2026-09-26.tar.gz"

Toolchain = Literal["numba", "nvcc"]
Effect = Literal["context_fault", "silent_cross_buffer_read", "silent_cross_buffer_write"]
Outcome = Literal["accepted", "refused"]

DEFAULT_PTX: dict[Toolchain, str] = {"numba": "numba.ptx", "nvcc": "kernel.ptx"}


class E3Record(TypedDict):
    """One line of E3's ``results.jsonl``: a kernel, its trigger, and what E3 observed."""

    model: str
    task: str
    toolchain: Toolchain
    trigger: str
    pool_write: str
    pool_read: str
    pool_fault: bool
    sep_fault: bool
    san_pool: str
    san_sep: str


class ArchiveContents(TypedDict):
    files: list[str]
    compiled_artifacts: list[str]


class E3Observation(TypedDict):
    pool_write: str
    pool_read: str
    pool_fault: bool
    separate_allocation_fault: bool
    sanitizer_pool: str
    sanitizer_separate: str


class GateRecord(TypedDict):
    decision: Outcome
    reason: str | None
    seconds: float


class KernelRecord(TypedDict):
    model: str
    task: str
    toolchain: Toolchain
    effect: Effect
    trigger: str
    e3: E3Observation
    gate: GateRecord
    ptx: str


class Provenance(TypedDict):
    e3_archive: str
    archive_contents: ArchiveContents


class Counts(TypedDict):
    kernels: int
    effects: dict[Effect, int]
    gate_decisions: dict[Outcome, int]
    accepted: int


class Result(TypedDict):
    experiment: str
    success: bool
    provenance: Provenance
    counts: Counts
    accepted: list[list[str]]
    wall_seconds: float
    kernels: list[KernelRecord]
    scope: str


def _text(record: dict[str, object], key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise TypeError(f"E3 record field {key} is not a string: {value!r}")
    return value


def _flag(record: dict[str, object], key: str) -> bool:
    value = record[key]
    if not isinstance(value, bool):
        raise TypeError(f"E3 record field {key} is not a boolean: {value!r}")
    return value


def _toolchain(value: str) -> Toolchain:
    match value:
        case "numba" | "nvcc":
            return value
        case _:
            raise ValueError(f"E3 record has an unknown toolchain: {value!r}")


def _e3_record(line: bytes) -> E3Record:
    loaded: object = json.loads(line)
    if not isinstance(loaded, dict):
        raise TypeError(f"E3 record is not an object: {loaded!r}")
    record: dict[str, object] = loaded
    return {
        "model": _text(record, "model"),
        "task": _text(record, "task"),
        "toolchain": _toolchain(_text(record, "toolchain")),
        "trigger": _text(record, "trigger"),
        "pool_write": _text(record, "pool_write"),
        "pool_read": _text(record, "pool_read"),
        "pool_fault": _flag(record, "pool_fault"),
        "sep_fault": _flag(record, "sep_fault"),
        "san_pool": _text(record, "san_pool"),
        "san_sep": _text(record, "san_sep"),
    }


def _e3_records(archive: Path) -> tuple[list[E3Record], ArchiveContents]:
    with tarfile.open(archive, "r:gz") as bundle:
        results_file = bundle.extractfile("e3/results.jsonl")
        harness_file = bundle.extractfile("e3/harness.py")
        if results_file is None or harness_file is None:
            raise RuntimeError(f"{archive} does not contain e3/results.jsonl")
        results_bytes = results_file.read()
        harness_bytes = harness_file.read()
        records = [_e3_record(line) for line in results_bytes.splitlines()]
        archived_files = sorted(member.name for member in bundle.getmembers() if member.isfile())
    if len(records) != 27:
        raise RuntimeError(f"expected 27 E3 records, found {len(records)}")
    keys = [(r["model"], r["task"], r["toolchain"]) for r in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("E3 contains a duplicate kernel key")
    harness = harness_bytes.decode()
    required_harness_fragments = (
        "cuda.jit(load_numba_kernel(model, task))",
        '["nvcc", "-cubin", "-arch=sm_86", "-O3"',
        "os.remove(cubin)",
    )
    missing = [fragment for fragment in required_harness_fragments if fragment not in harness]
    if missing:
        raise RuntimeError(f"E3 harness lacks expected compiler path: {missing}")
    compiled_suffixes = (".ptx", ".cubin", ".fatbin", ".so")
    compiled_files = [name for name in archived_files if name.endswith(compiled_suffixes)]
    if compiled_files:
        raise RuntimeError(f"E3 unexpectedly contains compiled artifacts: {compiled_files}")
    return records, {
        "files": archived_files,
        "compiled_artifacts": compiled_files,
    }


def _effect(record: E3Record) -> Effect:
    if record["pool_fault"]:
        return "context_fault"
    if record["pool_read"].startswith("cross-buffer read"):
        return "silent_cross_buffer_read"
    if record["pool_write"].startswith("cross-buffer write"):
        return "silent_cross_buffer_write"
    raise RuntimeError(f"E3 record has no effect: {record}")


def _gate_one(record: E3Record) -> KernelRecord:
    model, task, toolchain = record["model"], record["task"], record["toolchain"]
    directory = directory_of(model, task)
    ptx_path = directory / DEFAULT_PTX[toolchain]
    text = ptx_path.read_text()
    binding = binding_of(model, task, toolchain, text)
    started = time.perf_counter()
    decision = verify(zeroed(text, binding), [binding])
    elapsed = time.perf_counter() - started
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
        "model": model,
        "task": task,
        "toolchain": toolchain,
        "effect": _effect(record),
        "trigger": record["trigger"],
        "e3": {
            "pool_write": record["pool_write"],
            "pool_read": record["pool_read"],
            "pool_fault": record["pool_fault"],
            "separate_allocation_fault": record["sep_fault"],
            "sanitizer_pool": record["san_pool"],
            "sanitizer_separate": record["san_sep"],
        },
        "gate": {"decision": outcome, "reason": reason, "seconds": elapsed},
        "ptx": str(ptx_path.relative_to(SYNTHESIZED.root)),
    }


def run(archive: Path) -> Result:
    records, archive_contents = _e3_records(archive)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions = list(pool.map(_gate_one, records))
    wall_seconds = time.perf_counter() - started
    effects = Counter(record["effect"] for record in decisions)
    gate_decisions = Counter(record["gate"]["decision"] for record in decisions)
    accepted: list[list[str]] = [
        [record["model"], record["task"], record["toolchain"]]
        for record in decisions
        if record["gate"]["decision"] == "accepted"
    ]
    expected_effects: dict[Effect, int] = {
        "context_fault": 3,
        "silent_cross_buffer_read": 5,
        "silent_cross_buffer_write": 19,
    }
    return {
        "experiment": "E3 gate coverage on source-corresponding PTX artifacts",
        "success": not accepted and dict(sorted(effects.items())) == expected_effects,
        "provenance": {
            "e3_archive": str(archive),
            "archive_contents": archive_contents,
        },
        "counts": {
            "kernels": len(decisions),
            "effects": dict(sorted(effects.items())),
            "gate_decisions": dict(sorted(gate_decisions.items())),
            "accepted": len(accepted),
        },
        "accepted": accepted,
        "wall_seconds": wall_seconds,
        "kernels": sorted(
            decisions,
            key=lambda record: (record["model"], record["task"], record["toolchain"]),
        ),
        "scope": (
            "E3 JIT-compiled Numba source or compiled CUDA source to a temporary cubin, "
            "and its archive has no executed binary. These decisions cover the "
            "retained level-3 PTX artifacts generated from the same source files, not "
            "provably identical execution bytes. This also does not test the "
            "unimplemented catalog/executor."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=E3_ARCHIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.archive.resolve())
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
