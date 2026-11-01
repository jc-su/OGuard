"""Coverage: acceptance of safe kernels by toolchain and model family, the
corpus's kernels and entries, refusal causes, registration with the guard
pass, the refused kernels' features, repair, cuDF's UDFs, and the rule ablation."""

import csv
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Literal

from eval.checkpoint import read_artifact, snapshot_inputs
from eval.corpus.repair import TABLE as REPAIRS
from eval.corpus.run import TOOLCHAINS, Tasks, tasks_of
from eval.export.corpus import (
    FEATURES,
    REGISTRATION,
    SPLIT_NAMES,
    TOOLCHAIN_NAMES,
    Final,
    Kernel,
    Split,
    group_of,
    splits,
)
from eval.export.records import (
    RECORDS,
    Numbers,
    field,
    items,
    load,
    number,
    obj,
    text,
    values,
    write,
)
from eval.rq2_coverage.guarded import validate_evidence
from eval.rq2_coverage.rule_ablation import GROUPS, RULES, library_files
from eval.rq2_coverage.rule_ablation import KIND as ABLATION_KIND
from eval.rq4_cost.registration import inputs as registration_inputs
from opgate.gate import REGISTRATION_SECONDS, TIMEOUT_REASON, formal_dir

UDFS = RECORDS / "cudf_udfs_2026-10-24.json"  # rq2_coverage/cudf_udfs.py
# the guard pass on every entry, with the final gate of `--registration`
GUARDED = RECORDS / "guarded_2026-10-24c.json"  # rq2_coverage/guarded.py
# each safe guard-changed kernel as written and as the gate accepts it, on its workloads
DIFFERENTIAL = RECORDS / "differential_2026-10-24b.json"  # rq2_coverage/differential.py
# the final gate with each group of its rules turned off, on its accepted entries
ABLATION = RECORDS / "rule_ablation_2026-10-24c.json"  # rq2_coverage/rule_ablation.py
# each task set of the test set: the first registration record that decided it,
# with the gate frozen at that time, and that registration's guarded record
FIRST: dict[Tasks, tuple[Path, Path]] = {
    "held-out-4": (RECORDS / "registration_2026-10-02.json", RECORDS / "guarded_2026-10-02.json"),
    "held-out-5": (
        RECORDS / "registration_2026-10-03c.json",
        RECORDS / "guarded_2026-10-03c.json",
    ),
}
# the fields of a row of the guarded record
GUARDED_FIELDS = frozenset(
    {
        "id",
        "label",
        "recorded",
        "first",
        "first_reason",
        "decision",
        "reason",
        "guarded",
        "limited",
        "restructured",
        "binding",
        "checks",
        "seconds",
    }
)

# the gate's reasons, by the words that each begins with after the kernel's name
CAUSES = (
    ("the check did not finish", "timeout"),
    ("CTA barrier participation is not proved", "barrier participation"),
    ("warp collective participation is not proved", "warp collective participation"),
    ("the kernel is not proved to end", "termination"),
    ("the access is not proved inside its buffer", "access outside a checked region"),
    ("the shared memory access is not proved inside", "shared access outside a checked region"),
    ("the access is not proved aligned", "alignment"),
    ("the stored value is not proved to be one element", "index output value"),
    ("the load does not read one whole element", "index element width"),
    ("the write to the flag buffer is not a store", "flag write"),
    ("the analysis's invariant does not follow", "loop invariant"),
    ("the analysis reaches no fixpoint", "no fixpoint"),
    ("the local load is not proved", "local memory"),
    ("memory instruction the translation does not model", "instruction not modeled"),
    ("the buffer's direction does not allow", "argument direction"),
    ("the instruction is not proved unreachable", "trap"),
    ("an aligned barrier requires a supported target", "barrier form"),
    ("the zeroing prologue", "zeroing prologue"),
    ("an instruction may read the register", "register read before a write"),
)


def _listed(names: list[str]) -> str:
    """Names joined for prose: "", "a", "a and b", or "a, b, and c"."""
    if len(names) < 3:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def export_corpus(kernels: list[Kernel]) -> None:
    write(
        "corpus.csv",
        (
            "model",
            "task",
            "toolchain",
            "set",
            "class",
            "historical_decision",
            "check",
            "historical_repair_round",
            "final",
            "seconds",
            "bytes",
            "instructions",
            "features",
            "source",
        ),
        (
            (
                *k.key,
                SPLIT_NAMES[split],
                k.cls,
                "" if k.baseline is None else k.baseline,
                k.check,
                "" if k.repaired is None else k.repaired,
                k.final,
                f"{k.seconds:.3f}",
                k.bytes,
                k.instructions,
                " ".join(k.features),
                k.source,
            )
            for split, members in splits(kernels).items()
            for k in members
        ),
    )


# the final gate's decision on a safe kernel: accepted as written, accepted with
# the guard pass's checks and limits, or refused or timed out with them
Outcome = Literal["as_written", "checked", "refused", "timeout"]
OUTCOMES: tuple[Outcome, ...] = ("as_written", "checked", "refused", "timeout")


def _first(row: dict[str, object]) -> Final:
    """The first actual check in this replay, including its deadline outcome."""
    if row["first"] == "accepted" and row["first_reason"] is None:
        return "accepted"
    if row["first"] != "refused" or not text(row["first_reason"], "first_reason"):
        raise ValueError(f"{row['id']}: invalid first verification outcome")
    return "timeout" if TIMEOUT_REASON in text(row["first_reason"], "first_reason") else "refused"


def _outcome(k: Kernel, checked: dict[str, dict[str, object]]) -> Outcome:
    """A safe kernel's outcome, from its row of the guarded record."""
    name = "/".join(k.key)
    if name not in checked:
        raise ValueError(f"{k.key}: no row of the guarded record")
    row = checked[name]
    if _first(row) == "accepted":
        if row["decision"] != "accepted":
            raise ValueError(f"{k.key}: accepted as written and refused with checks")
        return "as_written"
    match row["decision"]:
        case "accepted":
            return "checked"
        case "refused" if TIMEOUT_REASON in text(row["reason"], "reason"):
            return "timeout"
        case "refused":
            return "refused"
        case _:
            raise ValueError(f"{k.key}: no decision with checks")


def export_acceptance(kernels: list[Kernel], guarded: list[dict[str, object]]) -> Numbers:
    """The final gate's decisions on the safe kernels of each set and toolchain,
    as written and with the guard pass's checks and limits (`guarded`, rows of
    the guarded record), with each kernel counted once. A timeout is counted
    apart from a refusal. Both stages use the actual checks in the same guarded
    replay. A separate registration's acceptance is not substituted for a
    timeout in this replay, nor vice versa.

    Historical repairs and earlier classifications have different inputs and
    denominators; they must not become stacked portions of this population.
    """
    checked = {text(r["id"], "id"): r for r in guarded}
    rows = []
    numbers: Numbers = {}
    for split, members in splits(kernels).items():
        safe = [k for k in members if k.cls == "safe"]
        for toolchain in TOOLCHAINS:
            outcomes = Counter(_outcome(k, checked) for k in safe if k.key[2] == toolchain)
            rows.append(
                (
                    f"{SPLIT_NAMES[split]} {TOOLCHAIN_NAMES[toolchain]}",
                    outcomes.total(),
                    *(outcomes[o] for o in OUTCOMES),
                )
            )
        name = split.capitalize()
        first = Counter(_first(checked["/".join(k.key)]) for k in safe)
        numbers |= {
            f"valKernels{name}": f"{len(members):,}",
            f"valSafe{name}": f"{len(safe):,}",
            f"valSafe{name}Accepted": f"{first['accepted']:,}",
            f"valSafe{name}Refused": f"{first['refused']:,}",
            f"valSafe{name}Timeout": f"{first['timeout']:,}",
        }
    write("acceptance.csv", ("group", "safe", *OUTCOMES), rows)
    return numbers


Family = Literal["Claude", "DeepSeek", "Devstral", "Gemini", "GPT", "MiniMax", "Qwen"]
# each model of the corpus, by its identifier in the records, and its family; the
# order of the families is the figure's
FAMILIES: dict[str, Family] = {
    "haiku": "Claude",
    "sonnet": "Claude",
    "opus": "Claude",
    "fable": "Claude",
    "deepseek-v4-flash": "DeepSeek",
    "devstral-2-123b": "Devstral",
    "gemini-3.1-pro": "Gemini",
    "gpt-5.5": "GPT",
    "gpt-oss-120b": "GPT",
    "minimax-m2.5": "MiniMax",
    "qwen3.5-397b": "Qwen",
}
FAMILY_ORDER: tuple[Family, ...] = tuple(dict.fromkeys(FAMILIES.values()))


def family_of(model: str) -> Family:
    """The family of a model of the corpus; a model of no known family fails."""
    if model not in FAMILIES:
        raise ValueError(f"{model}: a model of no known family")
    return FAMILIES[model]


def export_families(kernels: list[Kernel], guarded: list[dict[str, object]]) -> Numbers:
    """The final gate's decisions on the safe entries of each model family and
    set, as written and with the guard pass's checks and limits (`guarded`,
    rows of the guarded record), as `_outcome` counts them. A family appears
    in a set when the set has its entries; for each, the macros give the
    percentage of its safe entries accepted as written and with checks. For a
    family in both sets, they also give the development set's percentage less
    the test set's, of the printed percentages, in points."""
    checked = {text(r["id"], "id"): r for r in guarded}
    models = {k.key[0] for k in kernels}
    families = {family_of(model) for model in models}
    rows = []
    numbers: Numbers = {"valModels": f"{len(models)}", "valFamilies": f"{len(families)}"}
    for family in FAMILY_ORDER:
        printed: dict[tuple[Split, str], str] = {}
        for split, members in splits(kernels).items():
            safe = [k for k in members if k.cls == "safe" and family_of(k.key[0]) == family]
            if not safe:
                continue
            outcomes = Counter(_outcome(k, checked) for k in safe)
            rows.append((family, SPLIT_NAMES[split], len(safe), *(outcomes[o] for o in OUTCOMES)))
            accepted = outcomes["as_written"] + outcomes["checked"]
            printed[(split, "Written")] = f"{100 * outcomes['as_written'] / len(safe):.1f}"
            printed[(split, "Checked")] = f"{100 * accepted / len(safe):.1f}"
        for (split, stage), value in printed.items():
            numbers[f"valFamily{family}{split.capitalize()}{stage}Percent"] = value
        for stage in ("Written", "Checked"):
            if ("development", stage) in printed and ("test", stage) in printed:
                gap = float(printed[("development", stage)]) - float(printed[("test", stage)])
                numbers[f"valFamily{family}{stage}GapPoints"] = f"{gap:.1f}"
    write("families.csv", ("family", "set", "safe", *OUTCOMES), rows)
    return numbers


def export_collections(kernels: list[Kernel]) -> Numbers:
    """The entries and the kernels of each set and collection. An entry is a
    kernel in one launch configuration; the entries of one model, task set,
    toolchain, and plan kernel (`Kernel.source`) are one kernel, so a
    DataKernelBench plan kernel launched in several configurations counts once
    per plan, and every other kernel is one entry."""

    def unique(members: list[Kernel]) -> str:
        return f"{len({(k.key[0], k.tasks, k.key[2], k.source) for k in members}):,}"

    sets = splits(kernels)
    synthesized = [k for k in kernels if k.collection == "synthesized"]
    dkb = [k for k in kernels if k.collection == "DataKernelBench"]
    return {
        "valKernelsUnique": unique(kernels),
        "valKernelsDevelopmentUnique": unique(sets["development"]),
        "valKernelsTestUnique": unique(sets["test"]),
        "valSynthesizedEntries": f"{len(synthesized):,}",
        "valSynthesizedKernels": unique(synthesized),
        "valSynthesizedModels": f"{len({k.key[0] for k in synthesized})}",
        "valDkbEntries": f"{len(dkb):,}",
        "valDkbKernels": unique(dkb),
        "valDkbModels": f"{len({k.key[0] for k in dkb})}",
    }


def export_limits(kernels: list[Kernel]) -> Numbers:
    """Historical feature annotations and the current decisions of those inputs.

    These annotations do not enumerate the limits of newly reviewed inputs.
    """
    counts: Counter[str] = Counter()
    repaired: Counter[str] = Counter()
    final: Counter[str] = Counter()
    for k in kernels:
        for code in k.features:
            counts[code] += 1
            repaired[code] += k.repaired in (1, 2)
            final[code] += k.final == "accepted"
    ordered = sorted(counts, key=lambda c: (counts[c], c))
    write(
        "limits.csv",
        ("code", "feature", "kernels", "repaired", "final"),
        ((c, FEATURES[c], counts[c], repaired[c], final[c]) for c in ordered),
    )
    return {"valLimitKernels": f"{sum(1 for k in kernels if k.features)}"}


def export_repair() -> Numbers:
    """The historical repair rounds, all of development set kernels."""
    counted: Counter[tuple[int, str]] = Counter()
    with REPAIRS.open(newline="") as handle:
        for line in csv.DictReader(handle, delimiter="\t"):
            round = int(line["round"])
            if round == 0:
                continue
            split, _ = group_of(tasks_of(line["model"], line["task"]))
            if split != "development":
                raise ValueError(f"{REPAIRS}: a repair of a {SPLIT_NAMES[split]} kernel")
            counted[(round, "revisions")] += 1
            counted[(round, line["decision"])] += 1
            counted[(round, "correct")] += line["result"] == "correct"
            counted[(round, "timeout")] += TIMEOUT_REASON in line["reason"]
    columns = ("revisions", "accepted", "correct", "refused", "invalid", "timeout")
    write(
        "repair.csv",
        ("set", "round", *columns),
        (
            (SPLIT_NAMES["development"], round, *(counted[(round, c)] for c in columns))
            for round in (1, 2)
        ),
    )
    accepted = sum(v for (_, c), v in counted.items() if c == "accepted")
    correct = sum(v for (_, c), v in counted.items() if c == "correct")
    return {"valRepairAccepted": f"{accepted}", "valRepairCorrect": f"{correct}"}


def export_udfs() -> Numbers:
    record = load(UDFS)
    if record.get("registration_seconds") != REGISTRATION_SECONDS:
        raise ValueError(f"{UDFS}: the checks did not have the service's registration budget")
    examples = items(record["examples"], "examples")
    write(
        "udfs.csv",
        ("name", "path", "outcome", "reason"),
        (
            (
                text(e["name"], "name"),
                text(e["path"], "path"),
                text(e["outcome"], "outcome"),
                text(e["reason"], "reason"),
            )
            for e in examples
        ),
    )
    outcomes = Counter(text(e["outcome"], "outcome") for e in examples)
    if set(outcomes) - {"accepted", "refused"}:
        raise ValueError(f"{UDFS}: outcomes {sorted(outcomes)} of no known kind")
    return {
        "valUdfExamples": f"{len(examples)}",
        "valUdfAccepted": f"{outcomes['accepted']}",
        "valUdfRefused": f"{outcomes['refused']}",
    }


def _cause(reason: str) -> str:
    """The cause of a refusal, from the gate's reason; a reason that no cause names
    fails the export."""
    for words, cause in CAUSES:
        if words in reason:
            return cause
    raise ValueError(f"a refusal of no known cause: {reason}")


def guarded_rows(path: Path, kernels: list[Kernel]) -> tuple[list[dict[str, object]], int]:
    """Validate every replay row and count the first checks that differ from
    registration by the budget.

    The replay must have decided each entry under the service's budget of one
    registration, as the registration record did. Labels and recorded decisions
    must match the original registration. A first check may differ only when
    exactly one of the two checks did not finish in the budget; both
    observations remain intact. Coverage uses this replay's own first and final
    outcomes. Conclusive acceptance/refusal disagreements fail the export.
    """
    record = validate_evidence(path)
    if obj(record["provenance"], "provenance")["registration_seconds"] != REGISTRATION_SECONDS:
        raise ValueError(f"{path}: the replay did not have the service's registration budget")
    decided = {"/".join(k.key): k for k in kernels}
    rows = [r for r in items(record["results"], "results") if r["id"] in decided]
    if sorted(text(r["id"], "id") for r in rows) != sorted(decided):
        raise ValueError(f"{path}: a kernel has no row or more than one")
    deadline = 0
    for r in rows:
        k = decided[text(r["id"], "id")]
        if set(r) != GUARDED_FIELDS:
            raise ValueError(f"{path}: {k.key} has the fields {sorted(r)}")
        written = "accepted" if k.final == "accepted" else "refused"
        if r["label"] != k.cls or r["recorded"] != written:
            raise ValueError(f"{path}: {k.key} differs from its registration record")
        first = _first(r)
        expected = None if written == "accepted" else k.check
        if r["first"] != written or r["first_reason"] != expected:
            if (first == "timeout") == (expected is not None and TIMEOUT_REASON in expected):
                raise ValueError(f"{path}: {k.key} differs from its registration record")
            deadline += 1
        if type(r["checks"]) is not int or r["checks"] < 1 or number(r["seconds"], "seconds") < 0:
            raise ValueError(f"{path}: {k.key} has invalid check counts or elapsed time")
        changes = [values(r[f], f) for f in ("guarded", "limited")]
        if type(r["restructured"]) is not bool or any(
            type(c) is not str for change in changes for c in change
        ):
            raise ValueError(f"{path}: {k.key} has invalid guarded accesses or limited loops")
        if r["decision"] == "accepted":
            if r["reason"] is not None:
                raise ValueError(f"{path}: {k.key} is accepted with a reason")
        elif r["decision"] != "refused" or not text(r["reason"], "reason"):
            raise ValueError(f"{path}: {k.key} has no decision or no reason")
        if first == "accepted" and (
            r["decision"] != "accepted" or any(changes) or r["restructured"] is not False
        ):
            raise ValueError(f"{path}: {k.key} is accepted as written but changed")
    return rows, deadline


def export_refusal_causes(guarded: dict[Split, tuple[list[dict[str, object]], int]]) -> Numbers:
    """Why the gate refuses the safe kernels it refuses: as written, and with the
    guard pass's checks of the accesses that it does not prove and limits of
    the loops that it does not prove to end, from the guarded record's rows of
    each set and their reasons that differ by the deadline. The two sets are
    kept apart. Over both sets, the macros also give the defects that the gate
    accepts with the guard pass's checks and limits."""
    populations = {split: rows for split, (rows, _) in guarded.items()}
    safe = {s: [r for r in rows if r["label"] == "safe"] for s, rows in populations.items()}
    first: dict[Split, Counter[str]] = {s: Counter() for s in safe}
    last: dict[Split, Counter[str]] = {s: Counter() for s in safe}
    for split, rows in safe.items():
        for r in rows:
            if r["first"] == "refused":
                first[split][_cause(text(r["first_reason"], "first_reason"))] += 1
            if r["decision"] == "refused":
                last[split][_cause(text(r["reason"], "reason"))] += 1
    causes = [cause for _, cause in CAUSES]
    write(
        "refusal_causes.csv",
        (
            "cause",
            "development_as_written",
            "development_checked",
            "test_as_written",
            "test_checked",
        ),
        (
            (
                c,
                first["development"][c],
                last["development"][c],
                first["test"][c],
                last["test"][c],
            )
            for c in causes
        ),
    )
    numbers: Numbers = {}
    for split, rows in safe.items():
        name = split.capitalize()
        accepted = sum(r["first"] == "accepted" for r in rows)
        checked = sum(r["decision"] == "accepted" for r in rows)
        limited = sum(r["decision"] == "accepted" and bool(r["limited"]) for r in rows)
        numbers |= {
            f"valGuarded{name}Safe": f"{len(rows):,}",
            f"valGuarded{name}SafeAsWritten": f"{accepted:,}",
            f"valGuarded{name}SafeAsWrittenPercent": f"{100 * accepted / len(rows):.1f}",
            f"valGuarded{name}SafeChecked": f"{checked:,}",
            f"valGuarded{name}SafeCheckedPercent": f"{100 * checked / len(rows):.1f}",
            f"valGuarded{name}SafeLimited": f"{limited:,}",
            f"valGuarded{name}SafeUnaccepted": f"{len(rows) - checked:,}",
        }
        for stage, counted in (("Written", first[split]), ("Checked", last[split])):
            for cause in causes:
                macro = "".join(word.capitalize() for word in cause.split())
                numbers[f"valRefused{name}{stage}{macro}"] = f"{counted[cause]:,}"
    # the whole corpus: both sets together
    every = [r for rows in safe.values() for r in rows]
    accepted = sum(r["first"] == "accepted" for r in every)
    checked = sum(r["decision"] == "accepted" for r in every)
    numbers |= {
        "valGuardedAllSafe": f"{len(every):,}",
        "valGuardedAllSafeAsWritten": f"{accepted:,}",
        "valGuardedAllSafeAsWrittenPercent": f"{100 * accepted / len(every):.1f}",
        "valGuardedAllSafeChecked": f"{checked:,}",
        "valGuardedAllSafeCheckedPercent": f"{100 * checked / len(every):.1f}",
    }
    defects = [r for rows in populations.values() for r in rows if r["label"] != "safe"]
    if any(r["first"] == "accepted" for r in defects):
        raise ValueError(f"{GUARDED}: a defect is accepted as written")
    defects_checked = sum(r["decision"] == "accepted" for r in defects)
    numbers |= {
        "valGuardedAllDefects": f"{len(defects):,}",
        "valGuardedAllDefectsChecked": f"{defects_checked:,}",
        "valGuardedAllDefectsCheckedPercent": f"{100 * defects_checked / len(defects):.1f}",
    }
    deadline = sum(count for _, count in guarded.values())
    return numbers | {"valDeadlineReasonDiffers": f"{deadline}"}


def export_registration_rounds(guarded: list[dict[str, object]]) -> Numbers:
    """The time to register each safe entry that the gate accepts only with the
    guard pass's changes, from the first check of the text as written through
    every rewrite round to the accepting check, each check a fresh Rocq
    process, under the record's one budget for the whole registration; these
    times, and the checks that each took."""
    provenance = obj(load(GUARDED)["provenance"], "provenance")
    changed = [
        r
        for r in guarded
        if r["label"] == "safe" and _first(r) != "accepted" and r["decision"] == "accepted"
    ]
    seconds = sorted(number(r["seconds"], "seconds") for r in changed)
    checks = [int(number(r["checks"], "checks")) for r in changed]
    return {
        "valRoundsEntries": f"{len(changed):,}",
        "valRoundsMedianSeconds": f"{statistics.median(seconds):.1f}",
        "valRoundsMaxSeconds": f"{max(seconds):.0f}",
        "valRoundsMedianChecks": f"{statistics.median(checks):g}",
        "valRoundsMaxChecks": f"{max(checks)}",
        "valRoundsBudgetSeconds": f"{number(provenance['registration_seconds'], 'budget'):.0f}",
        "valRoundsWorkers": f"{int(number(provenance['workers'], 'workers'))}",
    }


def export_first_decisions(kernels: list[Kernel]) -> Numbers:
    """The first frozen gate's decisions on the test set's safe entries, as
    written and with the guard pass: for each task set of the test set, the
    first registration record that decided it and the guarded record of that
    registration (`FIRST`). Each record must hold one row of each of the task
    set's entries, with the entry's current PTX bytes and label; the guarded
    record must name the registration and its library and agree with its
    decisions as written. The binding records of these days have another
    format, so the bindings are not compared."""
    captured = {item.key: item for item in registration_inputs()}
    test = [k for k in kernels if k.split == "test"]
    if {k.tasks for k in test} != set(FIRST):
        raise ValueError("the test set's task sets are not those of the first decisions")
    written = checked = 0
    for tasks, (registration, guarded) in FIRST.items():
        members = {"/".join(k.key): k for k in test if k.tasks == tasks}
        record = load(registration)
        if record.get("complete") is not True:
            raise ValueError(f"{registration}: an incomplete record")
        library = text(field(record, "provenance", "library"), "library")
        decided: dict[str, bool] = {}
        for row in items(record["results"], "results"):
            name = text(row["id"], "id")
            if name not in members:
                continue
            k = members[name]
            ptx = read_artifact(registration, text(row["ptx_artifact"], "ptx_artifact"), "ptx")
            if name in decided or ptx.encode("ascii") != captured[k.key].ptx.encode("ascii"):
                raise ValueError(f"{registration}: {name} is repeated or has other PTX")
            if row["verdict"] != k.cls or type(row["accepted"]) is not bool:
                raise ValueError(f"{registration}: {name} has another label or no decision")
            decided[name] = row["accepted"]
        if set(decided) != set(members):
            raise ValueError(f"{registration}: the record lacks entries of {tasks}")
        replay = load(guarded)
        if Path(text(replay["registration"], "registration")).name != registration.name or (
            text(replay["library"], "library") != library
        ):
            raise ValueError(f"{guarded}: the replay of another registration or library")
        replayed: dict[str, str] = {}
        for row in items(replay["results"], "results"):
            name = text(row["id"], "id")
            if name not in members:
                continue
            recorded = "accepted" if decided[name] else "refused"
            if name in replayed or row["label"] != members[name].cls or row["recorded"] != recorded:
                raise ValueError(f"{guarded}: {name} differs from its registration")
            replayed[name] = text(row["decision"], "decision")
        if set(replayed) != set(members):
            raise ValueError(f"{guarded}: the replay lacks entries of {tasks}")
        safe = [name for name, k in members.items() if k.cls == "safe"]
        written += sum(decided[name] for name in safe)
        checked += sum(replayed[name] == "accepted" for name in safe)
    total = sum(1 for k in test if k.cls == "safe")
    return {
        "valFirstTestSafeAsWritten": f"{written:,}",
        "valFirstTestSafeAsWrittenPercent": f"{100 * written / total:.1f}",
        "valFirstTestSafeChecked": f"{checked:,}",
        "valFirstTestSafeCheckedPercent": f"{100 * checked / total:.1f}",
    }


def _ablation_decisions(row: dict[str, object], planned: list[str]) -> dict[str, Final]:
    """A variant's decision on each planned entry; each has a time."""
    where = text(row["id"], "id")
    decisions, seconds = obj(row["decisions"], where), obj(row["seconds"], where)
    if set(decisions) != set(planned) or set(seconds) != set(planned):
        raise ValueError(f"{where}: the decisions are not those of the planned entries")
    for value in seconds.values():
        if number(value, where) < 0:
            raise ValueError(f"{where}: a negative time")
    found: dict[str, Final] = {}
    for entry, value in decisions.items():
        match value:
            case "accepted":
                found[entry] = "accepted"
            case "refused":
                found[entry] = "refused"
            case "timeout":
                found[entry] = "timeout"
            case _:
                raise ValueError(f"{where}: an invalid decision {value!r} of {entry}")
    return found


def export_ablation(
    kernels: list[Kernel], path: Path = ABLATION, registration: Path = REGISTRATION
) -> Numbers:
    """The safe entries that the final gate accepts as written and that it loses
    with each group of its rules turned off, refused or not decided in the
    record's time limit. The record must have decided every such entry and its
    defect sample with the unchanged checker and every group of
    `rule_ablation.GROUPS`, from the registration's own bytes and library; the
    unchanged checker must accept every such entry, and no variant a defect."""
    record = load(path)
    if record.get("schema") != 2 or record.get("complete") is not True:
        raise ValueError(f"{path}: the ablation record is incomplete")
    provenance = obj(record["provenance"], "provenance")
    groups = [
        {
            "name": g.name,
            "description": g.description,
            "rules": list(g.rules),
            "subsumed": list(g.subsumed),
        }
        for g in GROUPS
    ]
    if provenance["kind"] != ABLATION_KIND or provenance["groups"] != groups:
        raise ValueError(f"{path}: the ablation did not turn off every group of the rules")
    snapshot_inputs(path, "registration", {"registration.json": registration}, create=False)
    library = formal_dir()
    snapshot_inputs(path, "library", library_files(library), create=False)
    safe = [text(v, "safe") for v in values(provenance["safe"], "safe")]
    defects = [text(v, "defects") for v in values(provenance["defects"], "defects")]
    accepted = sorted("/".join(k.key) for k in kernels if k.cls == "safe" and k.final == "accepted")
    if safe != accepted:
        raise ValueError(f"{path}: the entries are not the safe entries that the gate accepts")
    wrong = {"/".join(k.key) for k in kernels if k.cls != "safe"}
    if not defects or len(set(defects)) != len(defects) or not set(defects) <= wrong:
        raise ValueError(f"{path}: the defect sample is not a sample of the corpus's defects")
    timeout = number(provenance["timeout"], "timeout")
    rows = items(record["results"], "results")
    names = [text(r["id"], "id") for r in rows]
    if len(set(names)) != len(names) or set(names) != {"baseline", *(g.name for g in GROUPS)}:
        raise ValueError(f"{path}: the variants are not the baseline and the groups")
    planned = [*safe, *defects]
    decided = {name: _ablation_decisions(r, planned) for name, r in zip(names, rows, strict=True)}
    seconds = sum(
        number(s, "seconds") for r in rows for s in obj(r["seconds"], "seconds").values()
    ) + sum(number(r["build_seconds"], "build_seconds") for r in rows)
    if any(decided["baseline"][e] != "accepted" for e in safe):
        raise ValueError(f"{path}: the unchanged checker does not accept every safe entry")
    for name, found in decided.items():
        if any(found[e] == "accepted" for e in defects):
            raise ValueError(f"{path}: {name} accepts a defect")
    split = {"/".join(k.key): k.split for k in kernels}
    lost = {g.name: [e for e in safe if decided[g.name][e] != "accepted"] for g in GROUPS}
    write(
        "ablation.csv",
        ("group", "rules", "refused", "timeout", "lost", "development", "test"),
        (
            (
                g.name,
                len(g.rules) + len(g.subsumed),
                sum(decided[g.name][e] == "refused" for e in safe),
                sum(decided[g.name][e] == "timeout" for e in safe),
                len(lost[g.name]),
                sum(split[e] == "development" for e in lost[g.name]),
                sum(split[e] == "test" for e in lost[g.name]),
            )
            for g in GROUPS
        ),
    )
    needed = {e for entries in lost.values() for e in entries}
    numbers: Numbers = {
        "valAblationGroups": f"{len(GROUPS)}",
        "valAblationRules": f"{len(RULES)}",
        "valAblationSafe": f"{len(safe):,}",
        "valAblationDefects": f"{len(defects):,}",
        "valAblationTimeout": f"{timeout:.0f}",
        "valAblationNeeded": f"{len(needed):,}",
        "valAblationNeededPercent": f"{100 * len(needed) / len(safe):.1f}",
        "valAblationNeededNone": f"{len(safe) - len(needed):,}",
        "valAblationProofHours": f"{seconds / 3600:.1f}",
    }
    for g in GROUPS:
        macro = "".join(word.capitalize() for word in re.split(r"[^a-z]+", g.name) if word)
        numbers[f"valAblationLost{macro}"] = f"{len(lost[g.name]):,}"
    return numbers


def export_differential() -> Numbers:
    """The safe kernels the gate accepts only after the guard pass changed them,
    run as written and as changed on their workloads: how many have a workload,
    how many runs agree byte for byte, the mismatches and the tasks they occur in,
    the runs whose reference check differs between the two texts, and the failed
    checks on correct inputs, which must be none."""
    from eval.rq2_coverage.differential import SIZES, validate_evidence

    record = validate_evidence(DIFFERENTIAL)
    results = items(record["results"], "results")
    with_workload = [row for row in results if row["workload"]]
    rows = []
    runs = mismatch_runs = correct_mismatch_runs = check_failures = reference_changed = 0
    mismatched_kernels: list[str] = []
    empty_skipped = 0
    for row in results:
        unsupported = [run for run in items(row["runs"], "runs") if not run["supported"]]
        # the text states that a kernel with a workload skips only the empty input
        if row["workload"] and any(run["size"] != 0 for run in unsupported):
            raise ValueError(f"{row['id']}: a run above the empty input is unsupported")
        empty_skipped += bool(row["workload"] and unsupported)
        supported = [run for run in items(row["runs"], "runs") if run["supported"]]
        mismatches = [run for run in supported if not run["same_bytes_all"]]
        # a mismatch whose kernel as written gives the reference's result is the
        # change's; one whose kernel as written does not is the kernel's own
        correct = [run for run in mismatches if field(run, "as_written", "check") is None]
        failures = [run for run in supported if field(run, "changed", "check_failed")]
        reference_changed += sum(
            (field(run, "as_written", "check") is None) != (field(run, "changed", "check") is None)
            for run in supported
        )
        runs += len(supported)
        mismatch_runs += len(mismatches)
        correct_mismatch_runs += len(correct)
        check_failures += len(failures)
        if mismatches:
            mismatched_kernels.append(text(row["id"], "id"))
        rows.append(
            (
                text(row["id"], "id"),
                len(values(row["guarded_accesses"], "guarded_accesses")),
                len(values(row["limited_loops"], "limited_loops")),
                row["restructured"],
                bool(row["workload"]),
                len(supported),
                len(mismatches),
                len(correct),
                len(failures),
            )
        )
    write(
        "differential.csv",
        (
            "kernel",
            "checked_accesses",
            "limited_loops",
            "restructured",
            "workload",
            "sizes_run",
            "byte_mismatches",
            "byte_mismatches_written_correct",
            "check_failures",
        ),
        rows,
    )
    return {
        "valDiffSizes": f"{len(values(field(record, 'provenance', 'sizes'), 'sizes'))}",
        "valDiffKernels": f"{len(results)}",
        "valDiffWithWorkload": f"{len(with_workload)}",
        "valDiffNoWorkload": f"{len(results) - len(with_workload)}",
        "valDiffRuns": f"{runs}",
        "valDiffEmptySkipped": f"{empty_skipped}",
        "valDiffByteMismatchRuns": f"{mismatch_runs}",
        "valDiffMismatchRunsWrittenCorrect": f"{correct_mismatch_runs}",
        "valDiffMismatchedKernels": f"{len(mismatched_kernels)}",
        "valDiffMismatchedTasks": _listed(
            sorted({name.split("/")[1].replace("_", r"\_") for name in mismatched_kernels})
        ),
        "valDiffReferenceChanged": f"{reference_changed}",
        "valDiffCheckFailures": f"{check_failures}",
        "valDiffMaxRows": f"{max(SIZES):,}",
    }


def export(kernels: list[Kernel]) -> Numbers:
    """RQ2 from the final gate's registration."""
    export_corpus(kernels)
    guarded = {split: guarded_rows(GUARDED, members) for split, members in splits(kernels).items()}
    rows = [r for found, _ in guarded.values() for r in found]
    return (
        export_acceptance(kernels, rows)
        | export_families(kernels, rows)
        | export_collections(kernels)
        | export_limits(kernels)
        | export_repair()
        | export_udfs()
        | export_refusal_causes(guarded)
        | export_registration_rounds(rows)
        | export_first_decisions(kernels)
        | export_ablation(kernels)
        | export_differential()
    )
