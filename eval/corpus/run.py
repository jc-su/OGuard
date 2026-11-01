"""Run the gate over the corpus and compare the decisions with its tables.

Each collection's decision table `decisions.tsv` records, for each kernel,
the decision the gate is expected to make and the classification of the
kernel (a defect, or safe under its contract). It is the measure of the
gate's precision (architecture, engineering method): a rule flips named
rows, and a defect row that flips to accepted is a failure.

    uv run python -m eval.corpus.run                  compare the corpus with its tables
    uv run python -m eval.corpus.run haiku/gather ... only these kernels
    ... --from-source                                 compile each kernel from its source
                                                      and register it with the executor
    ... --write                                       add the kernels that the tables do
                                                      not have, with their reviewed verdicts,
                                                      and take a changed review's verdict

The review of a kernel that entered after its collection's table was first
written, or whose verdict changed when the claim language changed (a grid
claim), is in its model's `verdicts.tsv` (columns model, task, toolchain,
verdict: C, E, or safe, and evidence), made before the gate decided it.
"""

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, assert_never

from eval.corpus.adapter import (
    COLLECTIONS,
    DATAKERNELBENCH,
    binding_of,
    claim_of,
    collection_of,
    directory_of,
)
from opgate.gate import Accepted, Decision, Refused, verify, zeroed

COLUMNS = ("model", "task", "toolchain", "expected", "verdict", "reason")
TOOLCHAINS = {"numba": "numba.O1.ptx", "nvcc": "kernel.O1.ptx", "triton": "triton.ptx"}
Tasks = Literal[
    "development",
    "held-out",
    "held-out-2",
    "held-out-3",
    "held-out-4",
    "held-out-5",
    "datakernelbench",
    "datakernelbench-core",
]
# the task list of each set of the synthesized collection: the rules were
# developed on the tasks of TASKS.md; the models wrote the tasks of
# HELDOUT.md after the rule change of 2026-09-28, the tasks of HELDOUT2.md
# after the checker froze at 0df8e4b, and the tasks of HELDOUT3.md after the
# list was committed at 05ea0e4, for the checker that followed the coverage work,
# and the tasks of HELDOUT4.md after the list was committed at 044bfbf, before
# the gate froze, and the tasks of HELDOUT5.md after the list was committed at
# 949f8bb, after the final gate froze; their kernels were reviewed away from the repository that the
# gate's development used, so no rule was written with them in view
SPECS: dict[Tasks, str] = {
    "development": "TASKS.md",
    "held-out": "HELDOUT.md",
    "held-out-2": "HELDOUT2.md",
    "held-out-3": "HELDOUT3.md",
    "held-out-4": "HELDOUT4.md",
    "held-out-5": "HELDOUT5.md",
}
TASK_SETS: dict[Tasks, frozenset[str]] = {
    "development": frozenset(
        {
            "filter_select",
            "gather",
            "projection",
            "q6_sum",
            "hash_build",
            "hash_probe",
            "groupby_sum",
            "scatter_by_offsets",
            "string_length",
            "starts_with",
            "histogram",
            "segmented_sum",
        }
    ),
    "held-out": frozenset(
        {
            "valid_rows",
            "dictionary_decode",
            "run_end_decode",
            "string_contains",
            "min_max",
            "rolling_sum",
            "radix_scatter",
            "days_to_year",
            "merge_sorted",
            "sorted_contains",
        }
    ),
    "held-out-2": frozenset(
        {
            "coalesce",
            "gather_nullable",
            "string_equal",
            "ascii_upper",
            "lag",
            "bloom_build",
            "bloom_probe",
            "tile_scan",
            "block_sort",
            "bucketize",
        }
    ),
    "held-out-3": frozenset(
        {
            "run_starts",
            "segment_max",
            "lower_bound",
            "group_max",
            "string_reverse",
            "masked_sum",
            "bucket_histogram",
            "diff",
            "scatter_add",
            "tile_reverse",
        }
    ),
    "held-out-4": frozenset(
        {
            "stream_compact",
            "string_prefix",
            "join_count",
            "null_count",
            "prefix_offsets",
            "transpose",
            "hash_partition",
            "month_start",
            "window_max",
            "pair_sum",
        }
    ),
    "held-out-5": frozenset(
        {
            "between_bitmap",
            "gather_strings",
            "group_sum_valid",
            "asof_join",
            "decimal_rescale",
            "string_hash",
            "bitmap_slice",
            "join_materialize",
            "add_months",
            "argmin",
            "run_end_encode",
            "segmented_cummax",
            "string_concat",
            "hash_distinct",
            "range_window_sum",
            "bit_unpack",
            "group_argmax",
            "fill_forward",
            "timestamp_floor",
            "string_to_int",
        }
    ),
}
# the kernels decided at a time; each decision runs up to two coqc processes
WORKERS = 16


def tasks_of(model: str, task: str) -> Tasks:
    """The set of the kernel that `model` wrote for `task`: the DataKernelBench
    collection's full plans or its core plans, the held-out set, or one of the
    synthesized collection's task lists."""
    if collection_of(model) == DATAKERNELBENCH:
        return "datakernelbench-core" if "_core_" in task else "datakernelbench"
    (tasks,) = (name for name, members in TASK_SETS.items() if task in members)
    return tasks


@dataclass(frozen=True)
class Row:
    model: str
    task: str
    toolchain: str
    expected: str  # accepted, refused
    verdict: str  # C, E, safe: from classification.md
    reason: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model, self.task, self.toolchain)


def read_table() -> dict[tuple[str, str, str], Row]:
    """The rows of every collection's decision table."""
    rows: list[Row] = []
    for collection in COLLECTIONS:
        with collection.decisions.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows.extend(Row(**{column: line[column] for column in COLUMNS}) for line in reader)
    table = {row.key: row for row in rows}
    if len(table) != len(rows):
        raise ValueError("a kernel occurs twice in the decision tables")
    return table


def write_table(rows: list[Row]) -> None:
    """Write each collection's decision table from its models' rows."""
    for collection in COLLECTIONS:
        kept = sorted((row for row in rows if row.model in collection.models), key=lambda r: r.key)
        with collection.decisions.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(COLUMNS)
            for row in kept:
                writer.writerow(
                    (row.model, row.task, row.toolchain, row.expected, row.verdict, row.reason)
                )


def decide(model: str, task: str, toolchain: str) -> Decision:
    """The gate's decision on a kernel's recorded PTX with its zeroing prologue,
    the text that registration loads."""
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    return verify(zeroed(text, binding), [binding])


def kernels(only: set[str]) -> list[tuple[str, str, str]]:
    """The corpus kernels with PTX, or those that `only` names as `model/task`
    or `model/task/toolchain`."""
    found: list[tuple[str, str, str]] = []
    paths = sorted(
        path
        for collection in COLLECTIONS
        for model in collection.models
        for path in (collection.root / model).glob("*/signature.json")
    )
    for path in paths:
        model, task = path.parent.parent.name, path.parent.name
        for toolchain, ptx in TOOLCHAINS.items():
            named = f"{model}/{task}" in only or f"{model}/{task}/{toolchain}" in only
            if (path.parent / ptx).is_file() and (named or not only):
                found.append((model, task, toolchain))
    return found


Key = tuple[str, str, str]  # a model, a task, and a toolchain
VERDICTS = ("C", "E", "safe")


def read_verdicts() -> dict[Key, str]:
    """The reviewed verdicts in the models' `verdicts.tsv` files."""
    verdicts: dict[Key, str] = {}
    paths = [c.root / m / "verdicts.tsv" for c in COLLECTIONS for m in c.models]
    for path in (p for p in paths if p.exists()):
        with path.open(newline="") as handle:
            for line in csv.DictReader(handle, delimiter="\t"):
                key = (line["model"], line["task"], line["toolchain"])
                if line["verdict"] not in VERDICTS or key[0] != path.parent.name:
                    raise ValueError(f"{path}: a row {key} {line['verdict']!r} of another model")
                if key in verdicts:
                    raise ValueError(f"{path}: a second verdict of {key}")
                verdicts[key] = line["verdict"]
    return verdicts


def _recorded(found: list[Key]) -> list[Decision]:
    """The gate's decision on each kernel's recorded PTX."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(lambda k: decide(*k), found))


def compiled(found: list[Key]) -> list[Decision]:
    """Each kernel compiled from its source by its front end and registered with
    the executor, which verifies it and loads it when the gate accepts it; the
    gate verifies the kernels on `WORKERS` threads first, and registration then
    reads the cached decisions."""
    from eval.corpus.sources import operator_of
    from opgate.executor import Executor, RefusedError

    operators = [operator_of(*key) for key in found]
    bindings = [
        operator.binding(claim_of(model, task)[1])
        for operator, (model, task, _) in zip(operators, found, strict=True)
    ]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda o, b: verify(zeroed(o.text, b), [b]), operators, bindings))
    decisions: list[Decision] = []
    with Executor(0, 1 << 20) as executor:
        for operator, binding in zip(operators, bindings, strict=True):
            try:
                executor.register(operator.text, binding)
            except RefusedError as error:
                decisions.append(Refused(error.reason, ()))
                continue
            decisions.append(Accepted())
    return decisions


def run(only: set[str], write: bool, verdicts: dict[Key, str], from_source: bool) -> int:
    """Decide each kernel, compare the decisions with the tables, and return the
    count of failures: a defect that the gate accepts, and a kernel that its
    table records as accepted and the gate refuses. A safe kernel that its table
    records as refused and the gate accepts is reported, since the tables keep
    the decisions of the gate at the time each kernel entered them. A kernel
    that the tables do not have enters them, with `write`, with its reviewed
    verdict from `verdicts` and the gate's decision; the tables' rows stay as
    they are. A kernel whose review differs from its table's verdict is a
    failure, and with `write` the table takes the review's verdict."""
    table = read_table()
    found = kernels(only)
    decisions = compiled(found) if from_source else _recorded(found)
    failures = 0
    added: list[Row] = []
    reviewed: dict[Key, Row] = {}  # the rows whose verdict the review changed
    decided: list[tuple[str, str]] = []  # each kernel's decision and verdict
    for key, decision in zip(found, decisions, strict=True):
        match decision:
            case Accepted():
                actual, reason = "accepted", ""
            case Refused(reason=reason):
                actual = "refused"
            case _ as never:
                assert_never(never)
        expected = table.get(key)
        if expected is None:
            # a kernel the table does not classify yet: a defect or safe, by review
            if key not in verdicts:
                failures += 1
                print(f"{key}: not in the table and not reviewed, {actual} {reason}")
                continue
            verdict = verdicts[key]
            added.append(Row(*key, actual, verdict, reason))
        else:
            verdict = expected.verdict
            if verdicts.get(key, verdict) != verdict:
                print(f"{key}: the review says {verdicts[key]}, the table {verdict}")
                if write:
                    verdict = verdicts[key]
                    reviewed[key] = Row(*key, expected.expected, verdict, expected.reason)
                else:
                    failures += 1
            if actual == "refused" and expected.expected == "accepted":
                failures += 1
                print(f"{key}: the table records it as accepted, refused: {reason}")
            elif actual == "accepted" and expected.expected == "refused" and verdict == "safe":
                print(f"{key}: accepted, the table records it as refused")
        if actual == "accepted" and verdict != "safe":
            failures += 1
            print(f"{key}: ALARM: a defect ({verdict}) is accepted")
        decided.append((actual, verdict))
    safe = [actual for actual, verdict in decided if verdict == "safe"]
    defects = [actual for actual, verdict in decided if verdict != "safe"]
    print(
        f"accepted: safe {safe.count('accepted')} of {len(safe)}, "
        f"defects {defects.count('accepted')} of {len(defects)}; {failures} failures"
    )
    if write:
        write_table([*(reviewed.get(key, row) for key, row in table.items()), *added])
    return failures


def sample(table: dict[Key, Row]) -> list[Key]:
    """The kernels that the test suite decides, since the whole corpus takes
    hours: for each set of kernels and each toolchain, the first kernel that
    its table records as accepted and the first defect."""
    chosen: dict[tuple[Tasks, str, bool], Key] = {}
    for key, row in sorted(table.items()):
        model, task, toolchain = key
        group = (tasks_of(model, task), toolchain, row.verdict == "safe")
        if (row.verdict != "safe" or row.expected == "accepted") and group not in chosen:
            chosen[group] = key
    return sorted(chosen.values())


def main(argv: list[str]) -> int:
    write = "--write" in argv
    from_source = "--from-source" in argv
    only = {arg for arg in argv if not arg.startswith("--")}
    return 1 if run(only, write, read_verdicts(), from_source) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
