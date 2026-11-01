"""RQ1, safety: the kernels with a memory defect and the measured effects of unsafe operators."""

import math
from collections import Counter
from typing import Literal

from eval.export.corpus import Kernel, splits
from eval.export.records import (
    RECORDS,
    Numbers,
    _output,
    field,
    items,
    load,
    number,
    obj,
    text,
    values,
    write,
)
from opgate.gate import TIMEOUT_REASON, formal_dir

CORPUS_PTX = RECORDS / "corpus_ptx_2026-10-24.json"  # rq1_safety/corpus_ptx.py
GENERATED_PTX = RECORDS / "generated_ptx_2026-10-24.json"  # rq1_safety/generated_ptx.py
SERVICE_BOUNDARY = RECORDS / "service_boundary_2026-10-24.json"  # rq1_safety/service_boundary.py
GUARDED_DEFECTS = RECORDS / "guarded_defects_2026-10-24.json"  # rq1_safety/guarded_defects.py
LEAKAGE = RECORDS / "leakage_2026-10-24.json"  # rq1_safety/leakage.py
DEFECT_SYSTEMS = RECORDS / "defect_systems_2026-10-24.json"  # rq1_safety/defect_systems.py
LEAKAGE_NAMES = {  # each system in macro names
    "shared_unchecked": "Shared",
    "mps": "Mps",
    "guardian": "Guardian",
    "opgate": "Opgate",
}
LEAKAGE_HEADERS = {  # each system in the table's header
    "shared_unchecked": "\\shortstack{Shared context\\\\without checks}",
    "mps": "MPS",
    "guardian": "Guardian",
    "opgate": "\\sys",
}
LEAKAGE_ATTACKS = {  # each attack in the table's rows
    "read_oob": "Out-of-bounds read",
    "write_oob": "Out-of-bounds write",
    "shared_leftover": "Leftover shared memory",
    "local_leftover": "Leftover local memory",
    "freed_buffer": "Freed buffer",
    "context_fault": "Context fault",
    "index_race": "Out-of-range index",
    "revoked_grant": "Revoked grant",
    "infinite_loop": "Non-terminating kernel",
}
# attacks judged by whether the attacker affected other work, not by a leak
LEAKAGE_AVAILABILITY = ("context_fault", "infinite_loop")
CHECK_FAILED = "accepted_with_checks:check_failed"  # a launch whose check failed
LEAKAGE_DECISIONS = {  # OpGate's decision on B's kernel, as the table states it
    "refused": "refused",
    "accepted": "accepted",
    "accepted_with_checks": "accepted with checks",
    CHECK_FAILED: "\\code{check\\_failed}",
}
Result = Literal["leaked", "failed", "blocked"]  # what an attack did to A


# each module that the service experiment registers: its decision, whether the
# service applied the guard pass to it, and its loads and launches
SERVICE_MODULES: dict[str, tuple[str, bool, int]] = {
    "one_past": ("refused", True, 0),
    "wild": ("accepted", True, 1),
    "safe": ("accepted", False, 1),
}


def _error(response: dict[str, object]) -> str | None:
    """The kind of a failed response's error, or None when it succeeded."""
    if response["ok"] is True:
        return None
    return text(field(response, "error", "kind"), "kind")


def validate_service(record: dict[str, object]) -> None:
    """Validate the recorded inputs and outcomes before making safety claims.

    The gate refuses the one-past module with its checks, the service accepts
    the wild module with checks of its far read and answers its launch
    `check_failed`, and the safe control runs as submitted. Each module's
    audit row has the submitted PTX bytes, the binding, the registration of
    its response, and its loads and launches.
    """
    if record.get("success") is not True:
        raise ValueError("the service record is not successful")
    submitted = obj(record["exact_replayed_ptx"], "exact_replayed_ptx")
    if set(submitted) != set(SERVICE_MODULES):
        raise ValueError("the service record has an incomplete replay")
    audit = items(record["final_audit"], "final_audit")
    registrations = obj(record["registrations"], "registrations")
    registered = obj(registrations["e7_audit_rows"], "e7_audit_rows")
    for name, (decision, guarded, count) in SERVICE_MODULES.items():
        ptx = text(submitted[name], "submitted.ptx").encode("ascii")
        matches = [r for r in audit if text(r["ptx"], "audit.ptx").encode("ascii") == ptx]
        if len(matches) != 1:
            raise ValueError(f"{name}: the audit does not identify exactly one submitted PTX")
        row = matches[0]
        before = obj(registered[name], "registered")
        response = obj(registrations[f"{name}_response"], "response")
        if (
            text(before["ptx"], "registered.ptx").encode("ascii") != ptx
            or row["binding"] != before["binding"]
            or row["ptx_bytes"] != len(ptx)
            or row["decision"] != decision
            or (row["guarded"] is not None) != guarded
            or row["module_load_count"] != count
            or row["launch_count"] != count
            or _error(response) != (None if decision == "accepted" else "refused")
            or response["registration"] != row["registration"]
        ):
            raise ValueError(f"{name}: inconsistent registration, load or launch evidence")
        if decision == "accepted" and response["checks"] != (
            [] if row["guarded"] is None else field(row, "guarded", "checks")
        ):
            raise ValueError(f"{name}: the response's checks differ from its audit row")
    checks = values(field(registrations, "wild_response", "checks"), "checks")
    if not checks or _error(obj(record["wild_launch"], "wild_launch")) != "check_failed":
        raise ValueError("the wild module has no checks, or its launch is not check_failed")
    if field(record, "telemetry", "failed_checks") != 1:
        raise ValueError("the service did not count one failed check")
    control = obj(record["safe_control"], "safe_control")
    if (
        control["launch"] != {"ok": True, "ran": [True]}
        or control["observed_value"] != control["expected"]
    ):
        raise ValueError("the safe control did not run with the expected result")
    isolation = obj(record["tenant_isolation"], "tenant_isolation")
    if isolation["victim_observed_value"] != isolation["victim_expected_value"]:
        raise ValueError("the other tenant's buffer changed")
    query = obj(record["real_query"], "real_query")
    for responses, names in (
        (isolation, ("cross_tenant_read", "cross_tenant_free", "stale_read", "stale_operator_run")),
        (query, ("cross_tenant_cudf_request", "cross_tenant_generated_request")),
    ):
        for name in names:
            response = obj(responses[name], name)
            if _error(response) != "refused":
                raise ValueError(f"{name}: not a service refusal")
    expected = number(query["expected_revenue"], "expected_revenue")
    for name, tolerance in (("cudf_service_response", 1e-12), ("generated_service_response", 1e-9)):
        response = obj(query[name], name)
        if response["ok"] is not True or not math.isclose(
            number(response["revenue"], name), expected, rel_tol=tolerance
        ):
            raise ValueError(f"{name}: incorrect query result")
    if field(query, "generated_service_response", "ran") != [True]:
        raise ValueError("the generated query did not run")


def export_safety(kernels: list[Kernel]) -> Numbers:
    """The entries and the defects of class C and E, of each set and of both, and
    the defects that the final gate accepts as written."""
    numbers: Numbers = {}
    for split, members in splits(kernels).items():
        _, c, e, _ = _counts(members)
        name = split.capitalize()
        numbers |= {
            f"valDefects{name}Contract": f"{c:,}",
            f"valDefects{name}Precondition": f"{e:,}",
        }
    entries, c, e, accepted = _counts(kernels)
    return numbers | {
        "valKernels": f"{entries:,}",
        "valDefects": f"{c + e:,}",
        "valDefectsContract": f"{c:,}",
        "valDefectsPrecondition": f"{e:,}",
        "valDefectsAccepted": f"{accepted:,}",
    }


def _counts(group: list[Kernel]) -> tuple[int, int, int, int]:
    """The entries of a group, its defects of class C and of class E, and the
    defects that the gate accepts."""
    c = sum(1 for k in group if k.cls == "C")
    e = sum(1 for k in group if k.cls == "E")
    accepted = sum(1 for k in group if k.cls != "safe" and k.final == "accepted")
    return len(group), c, e, accepted


def validate_triggered(record: dict[str, object]) -> tuple[Counter[str], int]:
    """The summary must agree with every recorded kernel outcome."""
    kernels = items(record["kernels"], "kernels")
    if any(type(field(row, "e3", "pool_fault")) is not bool for row in kernels):
        raise ValueError("a triggered kernel has no recorded fault outcome in the pool")
    keys = [(row["model"], row["task"], row["toolchain"]) for row in kernels]
    if not kernels or len(set(keys)) != len(keys):
        raise ValueError("the triggered-kernel record has missing or duplicate cases")
    effects = Counter(text(row["effect"], "effect") for row in kernels)
    if any(
        field(row, "gate", "decision") != "refused"
        or TIMEOUT_REASON in text(field(row, "gate", "reason"), "reason")
        for row in kernels
    ):
        raise ValueError("a triggered kernel has no conclusive refusal")
    if (
        record.get("success") is not True
        or record["accepted"] != []
        or record["counts"]
        != {
            "accepted": 0,
            "effects": dict(effects),
            "gate_decisions": {"refused": len(kernels)},
            "kernels": len(kernels),
        }
    ):
        raise ValueError("the triggered-kernel summary differs from its recorded outcomes")
    return effects, len(kernels)


TRIGGERED_EFFECTS = {  # each effect of a triggered defect in macro names
    "silent_cross_buffer_write": "Writes",
    "silent_cross_buffer_read": "Reads",
    "context_fault": "Faults",
}


def export_security() -> Numbers:
    """The measured effects of unsafe operators, unchecked and through OpGate:
    the triggered defects by effect, the models and tasks they come from, those
    that run without a fault in one pool and that compute-sanitizer does not
    report there, the generated Q3 module's refusal, and the service's
    refusals of other tenants' and stale handles."""
    corpus_ptx = load(CORPUS_PTX)
    effects, refused = validate_triggered(corpus_ptx)
    if set(effects) != set(TRIGGERED_EFFECTS):
        raise ValueError(f"{CORPUS_PTX}: effects {sorted(effects)} of no known kind")
    triggered = items(corpus_ptx["kernels"], "kernels")
    silent = [row for row in triggered if field(row, "e3", "pool_fault") is False]
    unreported = [
        row
        for row in silent
        if text(field(row, "e3", "sanitizer_pool"), "sanitizer_pool").startswith("0 errors")
    ]
    generated_record = load(GENERATED_PTX)
    generated = text(field(generated_record, "gate", "decision"), "decision")
    if (
        generated_record.get("success") is not True
        or generated != "refused"
        or TIMEOUT_REASON in text(field(generated_record, "gate", "reason"), "reason")
    ):
        raise ValueError(f"{GENERATED_PTX}: the generated module has no conclusive refusal")
    service = load(SERVICE_BOUNDARY)
    validate_service(service)
    isolation = obj(service["tenant_isolation"], "tenant_isolation")
    rejected = sum(
        1
        for name in ("cross_tenant_read", "cross_tenant_free", "stale_read", "stale_operator_run")
        if obj(isolation[name], name)["ok"] is False
    )
    query = obj(service["real_query"], "real_query")
    rejected += sum(
        1
        for name in ("cross_tenant_cudf_request", "cross_tenant_generated_request")
        if obj(query[name], name)["ok"] is False
    )
    return {
        "valTriggeredKernels": f"{refused}",
        "valTriggeredModels": f"{len({text(row['model'], 'model') for row in triggered})}",
        "valTriggeredTasks": f"{len({text(row['task'], 'task') for row in triggered})}",
        "valTriggeredPoolSilent": f"{len(silent)}",
        "valTriggeredSanitizerSilent": f"{len(unreported)}",
        "valServiceRejected": f"{rejected}",
    } | {f"valTriggered{name}": f"{effects[effect]}" for effect, name in TRIGGERED_EFFECTS.items()}


def export_guarded_defects() -> Numbers:
    """The triggered defects with the guard pass's checks, checked and as written,
    decided by the current gate.

    Each checked run that ends must change no byte outside its outputs and
    report a failed check, since its trigger drives an access outside its
    region; a run as written has no status buffer, so it reports none. A read
    test's checked outputs must not depend on the neighbour's values.
    """
    record = load(GUARDED_DEFECTS)
    if text(record["library"], "library") != str(formal_dir()):
        raise ValueError(f"{GUARDED_DEFECTS}: the record's library is not the current gate's")
    defects = items(record["defects"], "defects")
    rows = []
    for d in defects:
        accepted = d["decision"] == "accepted"
        if d["decision"] not in ("accepted", "refused") or accepted != ("runs" in d):
            raise ValueError(f"{d['id']}: an accepted defect has runs, and a refused one none")
        runs = items(d["runs"], "runs") if accepted else []
        checked = [r for r in runs if r["mode"] == "guarded"]
        written = [r for r in runs if r["mode"] == "as_written"]
        if len(checked) + len(written) != len(runs):
            raise ValueError(f"{d['id']}: a run of no known mode")
        read_test = accepted and bool(obj(d["after"], "after"))
        for r in checked:
            if r["launched"] is not True or items(r["changed"], "changed"):
                raise ValueError(f"{d['id']}: a checked run did not launch or changed a byte")
            if r["check_failed"] is not True:
                raise ValueError(
                    f"{d['id']}: a checked run on its trigger reported no failed check"
                )
        if any(r["check_failed"] is not None for r in written):
            raise ValueError(f"{d['id']}: a run as written reported a check")
        if read_test and checked and checked[0]["outputs"] != checked[1]["outputs"]:
            raise ValueError(f"{d['id']}: the checked outputs depend on the neighbour's values")
        rows.append(
            (
                text(d["id"], "id"),
                text(d["decision"], "decision"),
                len(values(d["guarded"], "guarded")) if accepted else "",
                "unfinished" if "unfinished" in d else len(checked),
                max((len(items(r["changed"], "changed")) for r in written), default=""),
                ""
                if not (read_test and written)
                else written[0]["outputs"] != written[1]["outputs"],
            )
        )
    write(
        "guarded_defects.csv",
        (
            "kernel",
            "decision",
            "checked_accesses",
            "checked_runs",
            "as_written_bytes_outside",
            "as_written_leak",
        ),
        rows,
    )
    accepted = [d for d in defects if d["decision"] == "accepted"]
    ended = [d for d in accepted if "unfinished" not in d]
    refused = [d for d in defects if d["decision"] == "refused"]
    far = set(_far())
    non_far = [d for d in defects if text(d["id"], "id") not in far]
    non_far_refused = [d for d in non_far if d["decision"] == "refused"]
    far_refused = [d for d in defects if d["id"] in far and d["decision"] == "refused"]
    return {
        "valGuardedDefectsFarRefused": f"{len(far_refused)}",
        "valGuardedDefects": f"{len(defects)}",
        "valGuardedDefectsAccepted": f"{len(accepted)}",
        "valGuardedDefectsEnded": f"{len(ended)}",
        "valGuardedDefectsUnfinished": f"{len(accepted) - len(ended)}",
        "valGuardedDefectsRefused": f"{len(refused)}",
        "valDefectOpgateContained": f"{len(non_far)}",
        "valGuardedDefectsNonFarRefused": f"{len(non_far_refused)}",
    }


def _result(attack: str, entry: dict[str, object]) -> Result:
    """What an attack did to A: B read or changed A's data, held the GPU or ended
    A's context, or neither. An availability attack (a fault or a non-terminating
    kernel) is not a leak, so it is judged by whether A failed and every other
    attack by the cell's leak flag."""
    flag = entry["a_failed"] if attack in LEAKAGE_AVAILABILITY else entry["leaked"]
    if type(flag) is not bool:
        raise ValueError(f"{attack}: the outcome of {entry['system']} is not recorded")
    if not flag:
        return "blocked"
    return "failed" if attack in LEAKAGE_AVAILABILITY else "leaked"


# OpGate's decision on each attack and the component that blocks it: a run-time
# check or an iteration limit that fails, the catalog's refusal of the step, the
# catalog's clearing of a freed region, or the zeroing prologue
OPGATE_MECHANISM: dict[str, tuple[str, str]] = {
    "read_oob": (CHECK_FAILED, "check"),
    "write_oob": (CHECK_FAILED, "check"),
    "shared_leftover": ("accepted", "prologue"),
    "local_leftover": ("accepted", "prologue"),
    "freed_buffer": ("accepted", "clearing"),
    "context_fault": (CHECK_FAILED, "check"),
    "index_race": (CHECK_FAILED, "check"),
    "revoked_grant": ("refused", "catalog"),
    "infinite_loop": (CHECK_FAILED, "limit"),
}


def _leakage_cell(result: Result, decision: object, attack: str) -> str:
    """A cell of the security table: Blocked, Leaked, or Disrupted, and for
    OpGate, whose cells have a decision, the component that blocked the attack,
    which must be the one that the record's decision names."""
    label = {"leaked": "Leaked", "failed": "Disrupted", "blocked": "Blocked"}[result]
    if decision is None:
        return label
    tag = text(decision, "decision")
    if tag not in LEAKAGE_DECISIONS:
        raise ValueError(f"{LEAKAGE}: a decision {tag!r} of no known kind")
    expected, mechanism = OPGATE_MECHANISM[attack]
    if tag != expected or result != "blocked":
        raise ValueError(f"{LEAKAGE}: OpGate's {attack} is {tag} and {result}")
    return f"Blocked {{\\scriptsize({mechanism})}}"


def export_leakage() -> Numbers:
    """The leakage matrix: for each system and attack, whether A's data reached
    B's kernel, A's buffer stayed its secret, A failed, and OpGate's decision;
    the table of the matrix and the counts of each system's results.

    Each cell's summary must agree with its raw bytes (`leakage.validate_cell`),
    the record must cover every system and attack it declares, and its gate must
    be the current one.
    """
    from eval.rq1_safety.leakage import ATTACKS, SYSTEMS, validate_cell

    record = load(LEAKAGE)
    if field(record, "environment", "formal_library") != str(formal_dir()):
        raise ValueError(f"{LEAKAGE}: the record's library is not the current gate's")
    systems = tuple(text(v, "systems") for v in values(record["systems"], "systems"))
    attacks = tuple(text(v, "attacks") for v in values(record["attacks"], "attacks"))
    if systems != SYSTEMS or attacks != ATTACKS:
        raise ValueError(f"{LEAKAGE}: other systems or attacks than the experiment's")
    cells = items(record["cells"], "cells")
    by_key: dict[tuple[str, str], dict[str, object]] = {}
    for entry in cells:
        validate_cell(entry)
        key = (text(entry["system"], "system"), text(entry["attack"], "attack"))
        if key in by_key:
            raise ValueError(f"{LEAKAGE}: a repeated cell {key}")
        by_key[key] = entry
    if set(by_key) != {(s, a) for s in systems for a in attacks}:
        raise ValueError(f"{LEAKAGE}: the record does not cover every system and attack")
    rows = []
    for system in systems:
        for attack in attacks:
            entry = by_key[(system, attack)]
            rows.append(
                (
                    system,
                    attack,
                    entry["leaked"],
                    "" if entry["matches_secret"] is None else entry["matches_secret"],
                    "" if entry["victim_unchanged"] is None else entry["victim_unchanged"],
                    "" if entry["a_failed"] is None else entry["a_failed"],
                    "" if entry["decision"] is None else text(entry["decision"], "decision"),
                )
            )
    write(
        "leakage.csv",
        (
            "system",
            "attack",
            "leaked",
            "matches_secret",
            "victim_unchanged",
            "a_failed",
            "decision",
        ),
        rows,
    )
    for (system, attack), entry in by_key.items():
        if (entry["decision"] is not None) != (system == "opgate"):
            raise ValueError(f"{LEAKAGE}: {system} {attack}: a decision belongs to OpGate alone")
    _combined_table(by_key, systems, attacks)
    numbers: Numbers = {
        "valLeakSystems": f"{len(systems)}",
        "valLeakAttacks": f"{len(attacks)}",
    }
    for system in systems:
        results = [_result(a, by_key[(system, a)]) for a in attacks]
        name = LEAKAGE_NAMES[system]
        numbers[f"valLeak{name}Leaks"] = f"{results.count('leaked')}"
        numbers[f"valLeak{name}Faults"] = f"{results.count('failed')}"
        numbers[f"valLeak{name}Blocked"] = f"{results.count('blocked')}"
    failed = sum(1 for a in attacks if by_key[("opgate", a)]["decision"] == CHECK_FAILED)
    numbers["valLeakOpgateCheckFailed"] = f"{failed}"
    return numbers


# ---------------------------------------------------------------- defect containment under MPS and Guardian

DEFECT_MECHANISMS: dict[str, tuple[str, ...]] = {
    "Counter overflow": ("filter_select", "hash_probe"),
    "Missing bounds check": ("groupby_sum", "histogram", "q6_sum"),
    "Offset past output": ("scatter_by_offsets", "hash_build"),
    "Type/size mismatch": ("string_length",),
    "Sequence inversion": ("starts_with",),
}

# the outcomes of `eval.rq1_safety.defect_systems`, which its module states: B's
# kernel changed A's bytes or leaked A's secret into B's outputs, faulted a
# context that A uses, ran past the cell's limit, was rejected by Guardian's
# patcher, or ran under both secrets with A intact and the same outputs
DefectOutcome = Literal["contained", "escaped", "faulted", "timeout", "unsupported"]
# a defect's outcome under a system as the security table counts it: contained,
# run and not contained, refused by the gate at registration, or not run
TableOutcome = Literal["contained", "escaped", "refused", "unrun"]
# the outcomes under which B's kernel ran
RAN: tuple[DefectOutcome, ...] = ("contained", "escaped", "faulted", "timeout")
DEFECT_SYSTEMS_NAMES: tuple[str, ...] = ("guardian", "mps")
# the patcher's reasons for an unsupported kernel, by their words, in macro names
GUARDIAN_UNSUPPORTED = {
    "no global or generic memory accesses were found to instrument": "NoAccess",
    ".func bodies with global or generic memory accesses are not patched": "Call",
    "a generic access in a function that makes generic addresses of": "Converted",
}


def _far() -> list[str]:
    """The triggered defects whose triggers index pools of many GiB."""
    from eval.rq1_safety.guarded_defects import TRIGGERS

    return sorted(k for k, v in TRIGGERS.items() if v().large)


def _near() -> list[str]:
    """The triggered defects that MPS and Guardian run: all but the far ones."""
    from eval.rq1_safety.guarded_defects import TRIGGERS

    return sorted(k for k, v in TRIGGERS.items() if not v().large)


def _runs_outcome(cell: dict[str, object], where: str) -> DefectOutcome:
    """The outcome that a cell's runs show, by the rules of the experiment's
    module: A's changed bytes, a run past the limit, a fault of a context that A
    uses, then B's outputs under the two secrets for a trigger with neighbour
    values, compared as bytes."""
    runs = items(cell["runs"], where)
    secrets = [int(number(run["secret"], where)) for run in runs]
    ended_early = bool(runs) and (runs[-1].get("timeout") is True or runs[-1]["a_faulted"] is True)
    if secrets != [0, 1] and not (ended_early and secrets == [0]):
        raise ValueError(f"{where}: runs with the secrets {secrets}")
    if any(values(run.get("a_changes", []), where) for run in runs):
        return "escaped"
    if any(run.get("timeout") is True for run in runs):
        return "timeout"
    if any(run["a_faulted"] is True for run in runs):
        return "faulted"
    if obj(cell["after"], where) and runs[0]["b_outputs"] != runs[1]["b_outputs"]:
        return "escaped"
    return "contained"


def _defect_outcome(cell: dict[str, object]) -> DefectOutcome:
    """A cell's outcome, which must be the one that its raw fields show: a
    rejected kernel has the patcher's reason and no runs, a Guardian cell whose
    process ran past the limit has no runs, and every other cell has runs."""
    where = f"{DEFECT_SYSTEMS} {cell['system']} {cell['id']}"
    outcome = text(cell["outcome"], where)
    system = text(cell["system"], where)
    if outcome == "unsupported":
        if system != "guardian" or "runs" in cell:
            raise ValueError(f"{where}: an unsupported cell of {system} with runs")
        text(cell["reason"], where)
        return "unsupported"
    if "runs" not in cell:
        if (system, outcome) != ("guardian", "timeout"):
            raise ValueError(f"{where}: a {outcome} cell without runs")
        return "timeout"
    shown = _runs_outcome(cell, where)
    if outcome != shown:
        raise ValueError(f"{where}: the outcome {outcome} differs from the runs' {shown}")
    return shown


def defect_outcomes() -> dict[tuple[str, str], DefectOutcome]:
    """Each system's outcome on each triggered defect that MPS and Guardian run;
    the record must have one cell of each."""
    record = load(DEFECT_SYSTEMS)
    found: dict[tuple[str, str], DefectOutcome] = {}
    for cell in items(record["cells"], "cells"):
        key = (text(cell["system"], "system"), text(cell["id"], "id"))
        if key in found:
            raise ValueError(f"{DEFECT_SYSTEMS}: repeated cell {key}")
        found[key] = _defect_outcome(cell)
    if set(found) != {(s, k) for s in DEFECT_SYSTEMS_NAMES for k in _near()}:
        raise ValueError(f"{DEFECT_SYSTEMS}: the cells are not those of the near defects")
    return found


def export_defect_systems() -> Numbers:
    """Containment of the near triggered defects under Guardian and MPS: the
    defects that each runs and contains, that escape, fault a context that A
    uses, or run past the limit, and those that Guardian's patcher rejects (by
    reason); and the far defects, which neither runs."""
    record = load(DEFECT_SYSTEMS)
    outcomes = defect_outcomes()
    rows: list[tuple[str, ...]] = []
    unsupported: Counter[str] = Counter()
    for cell in items(record["cells"], "cells"):
        key = (text(cell["system"], "system"), text(cell["id"], "id"))
        changed = sum(
            len(values(run.get("a_changes", []), "a_changes"))
            for run in items(cell.get("runs", []), "runs")
        )
        rows.append((*key, outcomes[key], str(changed)))
        if outcomes[key] == "unsupported":
            reason = text(cell["reason"], "reason")
            names = [name for words, name in GUARDIAN_UNSUPPORTED.items() if words in reason]
            if len(names) != 1:
                raise ValueError(f"{DEFECT_SYSTEMS}: {key}: a reason of no known kind")
            unsupported[names[0]] += 1
    write(
        "defect_systems.csv",
        ("system", "defect", "outcome", "a_changed_bytes"),
        rows,
    )
    numbers: Numbers = {"valDefectFar": f"{len(_far())}"}
    for system in DEFECT_SYSTEMS_NAMES:
        counted = Counter(v for (s, _), v in outcomes.items() if s == system)
        name = LEAKAGE_NAMES[system]
        numbers |= {
            f"valDefect{name}Total": f"{counted.total()}",
            f"valDefect{name}Ran": f"{sum(counted[o] for o in RAN)}",
            f"valDefect{name}Contained": f"{counted['contained']}",
            f"valDefect{name}Escaped": f"{counted['escaped']}",
            f"valDefect{name}Faulted": f"{counted['faulted']}",
            f"valDefect{name}Timeout": f"{counted['timeout']}",
            f"valDefect{name}Unsupported": f"{counted['unsupported']}",
        }
    for name in GUARDIAN_UNSUPPORTED.values():
        numbers[f"valDefectGuardianUnsupported{name}"] = f"{unsupported[name]}"
    return numbers


def _combined_table(
    leakage_by_key: dict[tuple[str, str], dict[str, object]],
    systems: tuple[str, ...],
    attacks: tuple[str, ...],
) -> None:
    """Write tab_leakage.tex as a combined security evaluation table: the
    record's leakage attacks individually under its systems, then the near
    triggered defects grouped by mechanism, then a summary row. A defect cell
    counts the defects contained over those that ran: in one pool without
    checks, every near defect runs and reaches another buffer
    (`CORPUS_PTX`); under MPS and Guardian, the defect-systems record says
    which ran; under OpGate, a defect that the gate accepts with checks runs,
    and its runs, which `export_guarded_defects` validates, contain it, and the
    cell adds the defects that the gate refuses at registration, which never
    run."""
    attack_rows: list[str] = []
    for attack in attacks:
        cells = [LEAKAGE_ATTACKS[attack]]
        for system in systems:
            entry = leakage_by_key[(system, attack)]
            result = _result(attack, entry)
            cells.append(_leakage_cell(result, entry["decision"], attack))
        attack_rows.append(" & ".join(cells) + " \\\\")

    near = _near()
    outcomes = defect_outcomes()
    effects = {
        "/".join(text(row[f], f) for f in ("model", "task", "toolchain")): text(
            row["effect"], "effect"
        )
        for row in items(load(CORPUS_PTX)["kernels"], "kernels")
    }
    guarded = {text(d["id"], "id"): d for d in items(load(GUARDED_DEFECTS)["defects"], "defects")}
    if set(effects) != set(guarded) or not set(near) <= set(guarded):
        raise ValueError("the triggered defects differ between their records")

    def outcome(system: str, key: str) -> TableOutcome:
        """What a defect did under a system, as the table counts it."""
        match system:
            case "shared_unchecked":
                if effects[key] not in ("silent_cross_buffer_write", "silent_cross_buffer_read"):
                    raise ValueError(f"{CORPUS_PTX}: {key} reaches no other buffer")
                return "escaped"
            case "opgate":
                d = guarded[key]
                match d["decision"]:
                    case "accepted" if "unfinished" not in d:
                        return "contained"
                    case "refused":
                        return "refused"
                    case _:
                        raise ValueError(f"{GUARDED_DEFECTS}: {key} has no decision or run")
            case _:
                found = outcomes[(system, key)]
                if found not in RAN:
                    return "unrun"
                return "contained" if found == "contained" else "escaped"

    defect_rows: list[str] = []
    totals: dict[str, Counter[TableOutcome]] = {
        s: Counter[TableOutcome](
            "contained" if _result(a, leakage_by_key[(s, a)]) == "blocked" else "escaped"
            for a in attacks
        )
        for s in systems
    }

    def cell(found: Counter[TableOutcome]) -> str:
        """Contained over run, and the defects that the gate refused, which never ran."""
        ran = found["contained"] + found["escaped"]
        shown = f"{found['contained']}/{ran}" if ran else "--"
        if found["refused"]:
            shown += f" {{\\scriptsize(+{found['refused']} refused)}}"
        return shown

    grouped = [k for tasks in DEFECT_MECHANISMS.values() for k in near if k.split("/")[1] in tasks]
    if sorted(grouped) != near:
        raise ValueError("a near defect is in no mechanism or in two")
    for mechanism, tasks in DEFECT_MECHANISMS.items():
        group = [k for k in near if k.split("/")[1] in tasks]
        cells = [f"{mechanism} ({len(group)})"]
        for system in systems:
            found = Counter[TableOutcome](outcome(system, k) for k in group)
            totals[system] += found
            cells.append(cell(found))
        defect_rows.append(" & ".join(cells) + " \\\\")

    summary_cells = [f"\\textbf{{Total ({len(attacks)} + {len(near)})}}"]
    for system in systems:
        summary_cells.append(f"\\textbf{{{cell(totals[system])}}}")
    summary_row = " & ".join(summary_cells) + " \\\\"

    header = " & ".join(["\\textbf{Attack / defect}", *(LEAKAGE_HEADERS[s] for s in systems)])
    lines = [
        "% generated by `python -m eval.export` from the records; do not edit",
        "\\begin{table}[t]",
        "\\caption{\\sys blocks all \\valLeakAttacks{} attacks and refuses or contains all "
        "\\valGuardedDefects{} triggered defects; only MPS, with separate contexts, matches. "
        "Leakage cells give Blocked, Leaked, or Disrupted, and for \\sys the mechanism that "
        "blocks; defect cells give the defects contained over those that ran (--: none ran), "
        "and Guardian's patcher and \\sys's gate reject the others.}",
        "\\label{tab:leakage}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        "\\multicolumn{5}{@{}l}{\\textit{Leakage attacks}} \\\\[2pt]",
        *attack_rows,
        "\\midrule",
        "\\multicolumn{5}{@{}l}{\\textit{Corpus defects by violation mechanism}} \\\\[2pt]",
        *defect_rows,
        "\\midrule",
        summary_row,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    directory = _output.get()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tab_leakage.tex").write_text("\n".join(lines))


def export(kernels: list[Kernel]) -> Numbers:
    """RQ1 from the final gate's registration."""
    return (
        export_safety(kernels)
        | export_security()
        | export_guarded_defects()
        | export_leakage()
        | export_defect_systems()
    )
