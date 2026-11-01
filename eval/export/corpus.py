"""The corpus kernels with their decisions, repairs, final decisions, and registration times.

RQ1, RQ2, and RQ4 read the same list: the decision tables (`decisions.tsv`
of each collection), which keep the gate's decision at the time each kernel
entered them; the repair table; the reviewed features of each refused safe
kernel of the first two task sets (`limits.tsv`); and a complete registration
record whose raw inputs match the current corpus. Historical decisions and
repairs are annotations, not a before/after comparison under current claims.

The reports preserve the recorded development/test split (`group_of`). The test
set names the task sets `held-out-4` and `held-out-5`; that classification does
not establish that the final checker was never evaluated on them.
The selected registration record holds one decision per kernel. These task
sets have historical replays under multiple proof builds; their chronology
is documented in docs/READINESS.md and is distinct from a one-shot blind test.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from eval.checkpoint import read_artifact, snapshot_inputs
from eval.corpus.adapter import DATAKERNELBENCH, SYNTHESIZED, directory_of
from eval.corpus.repair import TABLE as REPAIRS
from eval.corpus.run import TOOLCHAINS, Tasks, tasks_of
from eval.export.records import RECORDS, items, load, number, text
from eval.rq4_cost.registration import KIND as REGISTRATION_KIND
from eval.rq4_cost.registration import Input, inputs, source_files
from opgate.gate import (
    CHECK_QUERY,
    REGISTRATION_SECONDS,
    TIMEOUT_REASON,
    export_terms,
    formal_dir,
    zeroed,
)
from opgate.hosts.service.audit import binding_record

REGISTRATION = RECORDS / "registration_2026-10-24c.json"  # rq4_cost/registration.py

LIMITS = SYNTHESIZED.root / "limits.tsv"
# the checks of termination, participation, the zeroing prologue, and registers
# written before they are read (design decision 46), which the gate of the
# decision tables did not make: a kernel that a table accepts is refused for
# these alone
LATER = (
    "the kernel is not proved to end",
    "CTA barrier participation is not proved",
    "warp collective participation is not proved",
    "the zeroing prologue",
    "an instruction may read the register",
)

Key = tuple[str, str, str]  # a model, a task, and a toolchain
Class = Literal["safe", "C", "E"]
Final = Literal["accepted", "refused", "timeout"]
Split = Literal["development", "test"]
CollectionName = Literal["synthesized", "DataKernelBench"]
SPLIT_NAMES: dict[Split, str] = {"development": "development set", "test": "test set"}
CONFIGURATION = re.compile(r"^(.+)_([2-9]|[1-9]\d+)$")  # a DataKernelBench configuration's suffix

TOOLCHAIN_NAMES = {"numba": "Numba", "nvcc": "CUDA C++", "triton": "Triton"}
FEATURES = {
    "BS": "binary search midpoint",
    "LR": "relation of loop registers",
    "DL": "length check before a loop",
    "SZ": "bound through a loaded offset",
    "DS": "dynamic shared memory",
    "FL": "flag carried through a loop",
    "RD": "reduction across lanes",
    "PL": "masked load",
    "KS": "k-ary search",
    "MM": "bound through min and max",
    "MC": "merge cursors",
    "SQ": "shared work queue",
    "LA": "local array at a loop index",
    "BL": "binary lifting",
    "CD": "count-down loop",
    "BM": "bitonic merge in registers",
    "SH": "shuffle across lanes",
    "BI": "ballot across lanes",
}


def group_of(tasks: Tasks) -> tuple[Split, CollectionName]:
    """The set and the collection of a task set. The held-out sets 1 to 3 were
    held out when added and later used to tune rules, so they are development
    data. Sets 4 and 5 retain their recorded test classification; the historical
    replay chronology must accompany any generalization claim."""
    match tasks:
        case "development" | "held-out" | "held-out-2" | "held-out-3":
            return "development", "synthesized"
        case "datakernelbench" | "datakernelbench-core":
            return "development", "DataKernelBench"
        case "held-out-4" | "held-out-5":
            return "test", "synthesized"
        case _ as unreachable:
            assert_never(unreachable)


@dataclass(frozen=True)
class Kernel:
    """One corpus kernel with its decisions and measurements."""

    key: Key
    tasks: Tasks
    cls: Class
    baseline: Literal["accepted", "refused"] | None  # historical decision, if recorded
    check: str  # current refusal reason, empty when accepted
    repaired: int | None  # the round that accepted it: 0 as written, 1 or 2 a revision
    final: Final  # the frozen checker's decision
    seconds: float  # the frozen checker's time
    bytes: int
    instructions: int
    features: tuple[str, ...]
    source: str  # the plan's kernel that a DataKernelBench kernel configures; the task otherwise

    @property
    def split(self) -> Split:
        return group_of(self.tasks)[0]

    @property
    def collection(self) -> CollectionName:
        return group_of(self.tasks)[1]


def _class(value: str) -> Class:
    match value:
        case "safe":
            return "safe"
        case "C":
            return "C"
        case "E":
            return "E"
        case _:
            raise ValueError(f"a class {value!r} of no known kind")


def _baseline(value: str | None) -> Literal["accepted", "refused"] | None:
    match value:
        case "accepted" | "refused" | None:
            return value
        case _:
            raise ValueError(f"an invalid historical decision {value!r}")


def instructions(ptx: str) -> int:
    """The instructions of a PTX text: the statements that are not directives."""
    count = 0
    for line in ptx.splitlines():
        statement = line.split("//")[0].strip()
        if statement.endswith(";") and not statement.startswith("."):
            count += 1
    return count


def _repaired() -> dict[Key, int]:
    """The first round of the repair loop that accepts each kernel."""
    accepted: dict[Key, int] = {}
    with REPAIRS.open(newline="") as handle:
        for line in csv.DictReader(handle, delimiter="\t"):
            key = (line["model"], line["task"], line["toolchain"])
            if line["decision"] == "accepted":
                round = int(line["round"])
                accepted[key] = min(round, accepted.get(key, round))
    return accepted


Decision = tuple[Final, float, int, str]  # a decision, its time, the PTX size, and the reason


def _record(path: Path) -> dict[str, object]:
    """A complete registration record of the current query and process model,
    each check under the service's budget of one registration."""
    record = load(path)
    if record.get("complete") is not True:
        raise ValueError(f"{path}: a complete record with raw inputs is required")
    provenance = record.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != REGISTRATION_KIND
        or provenance.get("query_suffix") != CHECK_QUERY
        or type(provenance.get("processes_per_verification")) is not int
        or provenance["processes_per_verification"] != 1
    ):
        raise ValueError(f"{path}: the recorded verification query or process model changed")
    if provenance.get("registration_seconds") != REGISTRATION_SECONDS:
        raise ValueError(f"{path}: the checks did not have the service's registration budget")
    return record


def _final(path: Path) -> dict[Key, Decision]:
    """Each kernel's decision, time, and PTX size in the registration record of
    the current gate, whose inputs are the current sources and library."""
    record = _record(path)
    snapshot_inputs(path, "sources", source_files(formal_dir()), create=False)
    return _decisions(path, record, {item.key: item for item in inputs()})


def _decisions(
    path: Path, record: dict[str, object], captured: dict[Key, Input]
) -> dict[Key, Decision]:
    """Each captured kernel's decision in the record's results, whose raw inputs
    must be the captured ones."""
    found: dict[Key, Decision] = {}
    for row in items(record["results"], "results"):
        model, task, toolchain = text(row["id"], "id").split("/")
        key = (model, task, toolchain)
        if key in found or key not in captured:
            raise ValueError(f"{path}: duplicate or unexpected kernel {key}")
        item = captured[key]
        if (
            read_artifact(path, text(row["ptx_artifact"], "ptx_artifact"), "ptx").encode("ascii")
            != item.ptx.encode("ascii")
            or row["binding"] != binding_record(item.binding)
            or read_artifact(path, text(row["query_artifact"], "query_artifact"), "v")
            != export_terms(zeroed(item.ptx, item.binding), [item.binding])
            or row["verdict"] != item.verdict
            or row["baseline"] != item.baseline
            or row["ptx_bytes"] != len(item.ptx.encode("ascii"))
        ):
            raise ValueError(f"{path}: raw inputs or review changed for {key}")
        if type(row["accepted"]) is not bool:
            raise ValueError(f"{path}: invalid decision for {key}")
        final: Final
        if row["accepted"] is True:
            final = "accepted"
        elif TIMEOUT_REASON in text(row["reason"], "reason"):
            final = "timeout"
        else:
            final = "refused"
        found[key] = (
            final,
            number(row["seconds"], "seconds"),
            int(number(row["ptx_bytes"], "ptx_bytes")),
            text(row["reason"], "reason"),
        )
    if set(found) != set(captured):
        raise ValueError(
            f"{path}: missing measurements for {len(set(captured) - set(found))} kernels"
        )
    return found


def source_of(model: str, task: str) -> str:
    """The plan's kernel of a DataKernelBench kernel: its task without the suffix
    of a second or later configuration, when the kernel without it comes from the
    same plan file; any other kernel is its own."""
    matched = CONFIGURATION.match(task)
    if model not in DATAKERNELBENCH.models or matched is None:
        return task
    first = directory_of(model, matched.group(1))
    plan = (directory_of(model, task) / "plan.py").read_bytes()
    if first.is_dir() and (first / "plan.py").read_bytes() == plan:
        return matched.group(1)
    return task


def _features() -> dict[Key, tuple[str, ...]]:
    found: dict[Key, tuple[str, ...]] = {}
    with LIMITS.open(newline="") as handle:
        for line in csv.DictReader(handle, delimiter="\t"):
            codes = tuple(line["features"].split())
            unknown = [c for c in codes if c not in FEATURES]
            if unknown:
                raise ValueError(f"{LIMITS}: codes {unknown} have no name")
            found[(line["model"], line["task"], line["toolchain"])] = codes
    return found


def corpus(registration: Path) -> list[Kernel]:
    """Every reviewed PTX input, including those outside historical tables, with
    the decisions of the current gate's registration record."""
    return _kernels(_final(registration), inputs())


def splits(kernels: list[Kernel]) -> dict[Split, list[Kernel]]:
    """The kernels of each set."""
    return {split: [k for k in kernels if k.split == split] for split in SPLIT_NAMES}


def _kernels(final: dict[Key, Decision], captured: list[Input]) -> list[Kernel]:
    """The captured inputs with their decisions, repairs, and features."""
    repaired, features = _repaired(), _features()
    kernels: list[Kernel] = []
    for item in captured:
        key = item.key
        model, task, toolchain = key
        ptx = item.ptx
        cls = _class(item.verdict)
        tasks = tasks_of(model, task)
        decision, seconds, size, reason = final[key]
        accepted = item.baseline == "accepted"
        if accepted and not (
            decision == "accepted" or (decision == "refused" and any(c in reason for c in LATER))
        ):
            raise ValueError(f"{key}: the frozen checker refuses a kernel its table accepts")
        if cls != "safe" and decision == "accepted":
            raise ValueError(f"{key}: the frozen checker accepts a defect")
        kernels.append(
            Kernel(
                key=key,
                tasks=tasks,
                cls=cls,
                baseline=_baseline(item.baseline),
                check=reason,
                repaired=0 if accepted else repaired.get(key),
                final=decision,
                seconds=seconds,
                bytes=size,
                instructions=instructions(ptx),
                features=features.get(key, ()),
                source=source_of(model, task),
            )
        )
    if len(kernels) != len(captured):
        raise ValueError("a kernel of the table is missing")
    order = list(TOOLCHAINS)
    kernels.sort(key=lambda k: (k.key[0], k.key[1], order.index(k.key[2])))
    return kernels
