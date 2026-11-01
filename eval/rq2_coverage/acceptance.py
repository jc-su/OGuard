"""RQ2: the gate's coverage of the corpora and the repair loop.

Run after `eval.corpus.repair` has recorded the frozen experiment inputs:

    uv run python -m eval.rq2_coverage.acceptance --repairs eval/records/repairs_<date>.json \\
        --output eval/records/coverage_<date>.json

reads the complete versioned repair record, and writes: for each set of
kernels (the development tasks of `TASKS.md`, the held-out tasks of `HELDOUT.md` and `HELDOUT2.md`, and the DataKernelBench
plans), and each model and toolchain, its kernels, the kernels that
are safe under their task's contract and those the gate accepts, the
defects and those accepted (none), and the safe kernels it refuses, which
are limits of the checker; the refusals by the check that failed; and each
round of the repair loop for each model and toolchain (design section 9.1).
The rules that the acceptances need are measured by
`eval.rq2_coverage.rule_ablation` on the registration record.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TypedDict, get_args

from eval.corpus.repair import validate_record
from eval.corpus.run import Tasks, read_table, tasks_of
from opgate.gate import TIMEOUT_REASON


class Corpus(TypedDict):
    """The decisions of one model and toolchain on one set of kernels."""

    tasks: Tasks
    model: str
    toolchain: str
    kernels: int
    safe: int
    safe_accepted: int
    defects: int
    defects_accepted: int
    limits: int


class Round(TypedDict):
    """One round of the repair loop for one model and toolchain on one set of tasks."""

    tasks: Tasks
    model: str
    toolchain: str
    round: int
    revisions: int
    accepted: int
    correct: int
    wrong: int
    refused: int
    invalid: int


# the check that a refusal names, before the instruction
CHECKS = (
    "the access is not proved inside its buffer, a shared variable, or a local variable",
    "the access is not proved aligned to its width",
    "the stored value is not proved to be one element in the index range",
    "the load does not read one whole element of the index buffer",
    "the write to the flag buffer is not a store of the number 1 into one whole word",
    "the shared memory access is not proved inside a shared variable",
    "the local load is not proved inside a local variable",
    "the local store is not proved inside a live local variable",
    "the instruction is not proved unreachable",
    "the analysis's invariant does not follow after this instruction",
    "has bounds its register cannot hold",
    "holds a constant its register cannot hold",
    "the analysis reaches no fixpoint",
    "CTA barrier participation is not proved by the finite phase checker",
    "warp collective participation is not proved by the finite phase checker",
    "the kernel is not proved to end",
    "memory instruction the translation does not model",
    "the arguments",
    "the source does not compile",
    TIMEOUT_REASON,
)


def check_of(reason: str) -> str:
    """The check that a refusal's reason names."""
    for check in CHECKS:
        if check in reason:
            return check
    raise ValueError(f"a refusal of no known check: {reason[-200:]}")


def corpora(record: dict) -> list[Corpus]:
    """The decisions of each model and toolchain, on each set of kernels."""
    groups: dict[tuple[Tasks, str, str], list[tuple[str, str]]] = {}
    for row in record["results"]:
        if row["round"] != 0:
            continue
        key = (tasks_of(row["model"], row["task"]), row["model"], row["toolchain"])
        groups.setdefault(key, []).append((row["decision"], row["verdict"]))
    records: list[Corpus] = []
    for (tasks, model, toolchain), rows in sorted(groups.items()):
        safe = [expected for expected, verdict in rows if verdict == "safe"]
        defects = [expected for expected, verdict in rows if verdict in ("C", "E")]
        records.append(
            Corpus(
                tasks=tasks,
                model=model,
                toolchain=toolchain,
                kernels=len(rows),
                safe=len(safe),
                safe_accepted=safe.count("accepted"),
                defects=len(defects),
                defects_accepted=defects.count("accepted"),
                limits=safe.count("refused") + safe.count("invalid"),
            )
        )
    return records


def refusals(record: dict) -> dict[Tasks, dict[str, int]]:
    """The refusals of the decision table, for each set of kernels, by the check
    that failed."""
    counted: dict[Tasks, Counter[str]] = {tasks: Counter() for tasks in get_args(Tasks)}
    for row in record["results"]:
        if row["round"] == 0 and row["decision"] in ("refused", "invalid"):
            counted[tasks_of(row["model"], row["task"])][check_of(row["reason"])] += 1
    return {tasks: dict(checks.most_common()) for tasks, checks in counted.items()}


def repair(record: dict) -> list[Round]:
    """Each round of the repair loop for each model and toolchain, on each set of
    tasks: its revisions and their outcomes."""
    rounds: dict[tuple[Tasks, str, str, int], list[tuple[str, str]]] = {}
    for line in record["results"]:
        if line["round"] != 0 and line["decision"] != "superseded":
            tasks = tasks_of(line["model"], line["task"])
            key = (tasks, line["model"], line["toolchain"], line["round"])
            rounds.setdefault(key, []).append((line["decision"], line["result"]))
    return [
        Round(
            tasks=tasks,
            model=model,
            toolchain=toolchain,
            round=k,
            revisions=len(outcomes),
            accepted=sum(1 for d, _ in outcomes if d == "accepted"),
            correct=sum(1 for _, r in outcomes if r == "correct"),
            wrong=sum(1 for _, r in outcomes if r == "wrong"),
            refused=sum(1 for d, _ in outcomes if d == "refused"),
            invalid=sum(1 for d, _ in outcomes if d == "invalid"),
        )
        for (tasks, model, toolchain, k), outcomes in sorted(rounds.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.repairs.read_text())
    validate_record(measured)
    if {tuple(k) for k in measured["provenance"]["selected"]} != set(read_table()):
        raise ValueError("aggregate coverage requires the complete corpus selection")
    found, refused, rounds = corpora(measured), refusals(measured), repair(measured)
    record = {
        "corpora": found,
        "refusals": refused,
        "repair": rounds,
        "provenance": measured["provenance"],
        "missing_revisions": measured["missing_revisions"],
        "timeouts": [r["id"] for r in measured["results"] if r.get("timeout")],
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    for corpus in found:
        print(corpus)
    print(refused)
    for round in rounds:
        print(round)


if __name__ == "__main__":
    main()
