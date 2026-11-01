"""The leakage experiment's pure parts: the leak rule, the record cell shape, the
guardian partition layout, the compiled local-memory kernels, and the exporter, all
without a GPU."""

import json
import struct
from typing import cast

import pytest

from eval.export import rq1
from eval.export.records import output_to
from eval.rq1_safety import leakage
from eval.rq1_safety.leakage import (
    ATTACKS,
    SECRET,
    SYSTEMS,
    Attack,
    System,
    cell,
    determine_leak,
    matches_secret,
    validate_cell,
)
from eval.rq4_cost.consolidation import (
    ALIGNMENT,
    guardian_layout,
    guardian_partition_bytes,
    guardian_partition_size,
)


def _words(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}Q", *values)


def test_matches_secret_finds_the_secret_in_any_word() -> None:
    assert matches_secret(_words(SECRET))
    assert matches_secret(_words(leakage.B_PATTERN, SECRET, 7))
    assert not matches_secret(_words(leakage.B_PATTERN, 0, 7))
    with pytest.raises(ValueError, match="whole 64-bit words"):
        matches_secret(b"\x00\x01\x02")


def test_determine_leak_reads_the_secret_and_detects_corruption() -> None:
    assert determine_leak("read_oob", read=_words(SECRET), victim_after=None)
    assert not determine_leak("read_oob", read=_words(leakage.B_PATTERN), victim_after=None)
    assert determine_leak("shared_leftover", read=_words(SECRET), victim_after=None)
    assert determine_leak("local_leftover", read=_words(SECRET), victim_after=None)
    assert not determine_leak("local_leftover", read=_words(0), victim_after=None)
    assert determine_leak("freed_buffer", read=_words(SECRET), victim_after=None)
    secret = _words(*([SECRET] * leakage.WORDS))
    changed = _words(*([leakage.ATTACKER_VALUE] + [SECRET] * (leakage.WORDS - 1)))
    assert not determine_leak("write_oob", read=None, victim_after=secret)
    assert determine_leak("write_oob", read=None, victim_after=changed)
    assert not determine_leak("context_fault", read=None, victim_after=None)


def test_the_local_kernels_keep_their_arrays_in_local_memory() -> None:
    # A's fill stores the secret to local memory, and B's read loads local memory
    # that its kernel never wrote, so that the attack reads what A's threads left there
    fill = leakage._compiled("fill_local", "compute_86")
    read = leakage._compiled("read_local", "compute_86")
    assert ".local .align" in fill and "st.local" in fill
    assert ".local .align" in read and "ld.local" in read and "st.local" not in read


def test_cell_round_trips_through_validate() -> None:
    leaked = cell("shared_unchecked", "read_oob", read=_words(SECRET), note="B read A's secret")
    assert leaked["leaked"] is True and leaked["matches_secret"] is True
    validate_cell(leaked)
    safe = cell("mps", "read_oob", read=_words(leakage.B_PATTERN), note="own address space")
    assert safe["leaked"] is False
    validate_cell(safe)
    fault = cell("opgate", "context_fault", a_failed=False, decision="refused", note="refused")
    assert fault["leaked"] is False and fault["a_failed"] is False
    validate_cell(fault)
    with_mps = {**safe, "mps": {"mode": "static_sm_partitioning"}}
    validate_cell(with_mps)  # the optional MPS metadata is allowed


@pytest.mark.parametrize(
    "corruption",
    ["leaked", "matches", "unchanged", "note", "system", "attack", "field"],
)
def test_validate_cell_rejects_inconsistent_evidence(corruption) -> None:
    secret = _words(*([SECRET] * leakage.WORDS))
    entry = cell("shared_unchecked", "write_oob", victim_after=secret, note="unchanged")
    if corruption == "leaked":
        entry["leaked"] = True
    elif corruption == "matches":
        entry = cell("shared_unchecked", "read_oob", read=_words(SECRET), note="x")
        entry["matches_secret"] = False
    elif corruption == "unchanged":
        entry["victim_unchanged"] = False
    elif corruption == "note":
        entry["note"] = ""
    elif corruption == "system":
        entry["system"] = "unknown"
    elif corruption == "attack":
        entry["attack"] = "unknown"
    elif corruption == "field":
        entry["surprise"] = 1
    with pytest.raises(ValueError):
        validate_cell(entry)


def test_guardian_partition_size_is_the_next_power_of_two() -> None:
    assert guardian_partition_size(1) == 1
    assert guardian_partition_size(ALIGNMENT) == ALIGNMENT
    assert guardian_partition_size(4 * ALIGNMENT) == 4 * ALIGNMENT
    assert guardian_partition_size(4 * ALIGNMENT + 1) == 8 * ALIGNMENT
    size = guardian_partition_size(guardian_partition_bytes(600000))
    assert size & (size - 1) == 0 and size >= guardian_partition_bytes(600000)
    with pytest.raises(ValueError):
        guardian_partition_size(0)


def test_guardian_layout_places_aligned_non_overlapping_partitions() -> None:
    size = 1 << 20
    first, bases = guardian_layout(300, size, 3)
    assert first % size == 0 and 0 <= first - 300 < size
    assert bases == [first, first + size, first + 2 * size]
    assert all(base % size == 0 for base in bases)


def test_guardian_partition_bytes_sums_the_four_columns_and_the_result() -> None:
    rows = 1000

    def aligned(nbytes: int) -> int:
        return -(-nbytes // ALIGNMENT) * ALIGNMENT

    expected = aligned(rows * 4) + 3 * aligned(rows * 8) + aligned(8)
    assert guardian_partition_bytes(rows) == expected


def _cell_for(system: System, attack: Attack) -> dict[str, object]:
    """One honest synthetic cell: the shared context leaks every memory attack,
    faults, and lets a non-terminating kernel hold the GPU; MPS, Guardian, and
    OpGate do not reach A for the memory attacks; leftover shared and local
    memory still leak under Guardian, which also cannot stop a non-terminating
    kernel, and OpGate's zeroing prologue clears both leftovers and its gate
    bounds the loop."""
    full = leakage.WORDS
    secret = _words(*([SECRET] * full))
    corrupt = _words(*([leakage.ATTACKER_VALUE] * full))
    leftover_leaks = system in ("shared_unchecked", "guardian")
    not_isolated = system in ("shared_unchecked", "guardian")
    decision = (
        {
            "read_oob": "accepted_with_checks:check_failed",
            "write_oob": "accepted_with_checks:check_failed",
            "shared_leftover": "accepted",
            "local_leftover": "accepted",
            "freed_buffer": "accepted",
            "context_fault": "accepted_with_checks:check_failed",
            "index_race": "accepted_with_checks:check_failed",
            "revoked_grant": "refused",
            "infinite_loop": "accepted_with_checks:check_failed",
        }[attack]
        if system == "opgate"
        else None
    )
    match attack:
        case "read_oob":
            read = _words(SECRET) if system == "shared_unchecked" else _words(leakage.B_PATTERN)
            return cell(system, attack, read=read, decision=decision, note="read")
        case "write_oob":
            victim = corrupt if system == "shared_unchecked" else secret
            return cell(system, attack, victim_after=victim, decision=decision, note="write")
        case "shared_leftover":
            read = _words(SECRET) if leftover_leaks else _words(0)
            return cell(system, attack, read=read, decision=decision, note="shared")
        case "local_leftover":
            read = _words(SECRET) if leftover_leaks else _words(0)
            return cell(system, attack, read=read, decision=decision, note="local")
        case "freed_buffer":
            read = _words(SECRET) if system == "shared_unchecked" else _words(0)
            return cell(system, attack, read=read, decision=decision, note="freed")
        case "index_race":
            read = _words(SECRET) if system == "shared_unchecked" else _words(leakage.B_PATTERN)
            return cell(system, attack, read=read, decision=decision, note="index")
        case "revoked_grant":
            read = _words(SECRET) if system == "shared_unchecked" else _words(0)
            return cell(system, attack, read=read, decision=decision, note="grant")
        case "infinite_loop":
            return cell(system, attack, a_failed=not_isolated, decision=decision, note="loop")
        case _:  # context_fault
            return cell(
                system,
                attack,
                a_failed=(system == "shared_unchecked"),
                decision=decision,
                note="fault",
            )


def _full_record() -> dict[str, object]:
    cells = [_cell_for(system, attack) for system in SYSTEMS for attack in ATTACKS]
    return {
        "environment": {"formal_library": str(rq1.formal_dir())},
        "secret": f"{SECRET:016x}",
        "systems": list(SYSTEMS),
        "attacks": list(ATTACKS),
        "cells": cells,
    }


def test_export_leakage_counts_each_systems_results_and_writes_the_table(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "leakage.json"
    path.write_text(json.dumps(_full_record()))
    monkeypatch.setattr(rq1, "LEAKAGE", path)
    with output_to(tmp_path):
        numbers = rq1.export_leakage()
    assert numbers["valLeakSystems"] == "4" and numbers["valLeakAttacks"] == "9"
    counts = {
        "Shared": ("7", "2", "0"),
        "Mps": ("0", "0", "9"),
        "Guardian": ("2", "1", "6"),
        "Opgate": ("0", "0", "9"),
    }
    for name, (leaks, faults, blocked) in counts.items():
        assert numbers[f"valLeak{name}Leaks"] == leaks
        assert numbers[f"valLeak{name}Faults"] == faults
        assert numbers[f"valLeak{name}Blocked"] == blocked
    assert numbers["valLeakOpgateCheckFailed"] == "5"
    assert (tmp_path / "data/leakage.csv").exists()
    table = (tmp_path / "tab_leakage.tex").read_text()
    lines = table.splitlines()
    start = lines.index("\\multicolumn{5}{@{}l}{\\textit{Leakage attacks}} \\\\[2pt]") + 1
    rows = [
        line.removesuffix(" \\\\").split(" & ")
        for line in lines[start : lines.index("\\midrule", start)]
    ]
    assert [row[0] for row in rows] == [
        "Out-of-bounds read",
        "Out-of-bounds write",
        "Leftover shared memory",
        "Leftover local memory",
        "Freed buffer",
        "Context fault",
        "Out-of-range index",
        "Revoked grant",
        "Non-terminating kernel",
    ]
    # the shared context leaks seven attacks and is disrupted by the fault and the
    # non-terminating kernel; Guardian leaks both leftovers and is disrupted by the loop;
    # MPS blocks every attack
    others = [entry for row in rows for entry in row[1:4]]
    assert others.count("Leaked") == 9 and others.count("Disrupted") == 3
    assert others.count("Blocked") == 3 * 9 - 9 - 3
    # each OpGate cell names the component that its decision names
    mechanisms = (
        "check",
        "check",
        "prologue",
        "prologue",
        "clearing",
        "check",
        "check",
        "catalog",
        "limit",
    )
    assert [row[4] for row in rows] == [
        f"Blocked {{\\scriptsize({mechanism})}}" for mechanism in mechanisms
    ]
    # the caption and the summary row count the record's nine attacks
    assert "\\caption{Security evaluation: 9 leakage attacks and " in table
    assert "\\textbf{Total (9 + " in table
    # OpGate's defect cells count the near defects that the gate accepts with checks
    # as run and contained, and add those that it refuses, which never ran
    near = set(rq1._near())
    decisions = [
        d["decision"]
        for d in json.loads(rq1.GUARDED_DEFECTS.read_text())["defects"]
        if d["id"] in near
    ]
    accepted, refused = decisions.count("accepted"), decisions.count("refused")
    assert accepted + refused == len(near) and refused > 0
    summary = lines[lines.index("\\bottomrule") - 1].removesuffix(" \\\\").split(" & ")
    ran = len(ATTACKS) + accepted
    assert summary[4] == f"\\textbf{{{ran}/{ran} {{\\scriptsize(+{refused} refused)}}}}"
    assert "\\label{tab:leakage}" in table


@pytest.mark.parametrize(
    "flaw",
    [
        "inconsistent",
        "incomplete",
        "library",
        "decision",
        "unknown_decision",
        "mechanism",
        "attacks",
    ],
)
def test_export_leakage_rejects_bad_records(tmp_path, monkeypatch, flaw) -> None:
    record = _full_record()
    cells = cast(list[dict[str, object]], record["cells"])
    if flaw == "inconsistent":
        cells[0]["leaked"] = not cells[0]["leaked"]
    elif flaw == "incomplete":
        record["cells"] = cells[:-1]
    elif flaw == "library":
        record["environment"] = {"formal_library": "/another/library"}
    elif flaw == "decision":
        cells[0]["decision"] = "accepted"  # a decision of a system without a gate
    elif flaw == "unknown_decision":
        cells[-1]["decision"] = "unproved"
    elif flaw == "mechanism":
        # a refusal of the out-of-bounds read, which the table names a failed check
        opgate = next(c for c in cells if c["system"] == "opgate")
        opgate["decision"] = "refused"
    else:
        record["attacks"] = list(reversed(ATTACKS))
    path = tmp_path / "leakage.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(rq1, "LEAKAGE", path)
    with output_to(tmp_path), pytest.raises(ValueError):
        rq1.export_leakage()


ILLEGAL_ADDRESS = "the driver returned <CUresult.CUDA_ERROR_ILLEGAL_ADDRESS: 700>"
CHANGE = {"offset": 0, "before": 1, "after": 2}


def _run(secret: int, **fields: object) -> dict[str, object]:
    """A run of B's kernel that ended with A intact, with `fields` replaced."""
    run: dict[str, object] = {
        "secret": secret,
        "error": None,
        "a_faulted": False,
        "a_changes": [],
        "b_outputs": {"out": "AAAA"},
    }
    return run | fields


def _cell(system: str, outcome: str, runs: list[dict[str, object]], after: bool = False):
    return {
        "id": "m/t/numba",
        "system": system,
        "outcome": outcome,
        "after": {"x": ["AA==", "AQ=="]} if after else {},
        "runs": runs,
    }


@pytest.mark.parametrize(
    ("cell", "outcome"),
    [
        (_cell("guardian", "contained", [_run(0), _run(1)]), "contained"),
        (_cell("guardian", "contained", [_run(0), _run(1)], after=True), "contained"),
        (
            _cell("guardian", "escaped", [_run(0), _run(1, b_outputs={"out": "AQAA"})], True),
            "escaped",
        ),
        (_cell("guardian", "escaped", [_run(0), _run(1, a_changes=[CHANGE])]), "escaped"),
        (
            _cell(
                "guardian",
                "faulted",
                [{"secret": 0, "error": ILLEGAL_ADDRESS, "a_faulted": True}],
            ),
            "faulted",
        ),
        (
            {"id": "m/t/numba", "system": "guardian", "outcome": "unsupported", "reason": "r"},
            "unsupported",
        ),
        ({"id": "m/t/numba", "system": "guardian", "outcome": "timeout"}, "timeout"),
        (
            # B faulted its own MPS client, and A stayed intact
            _cell(
                "mps",
                "contained",
                [_run(s, timeout=False, error=ILLEGAL_ADDRESS, b_outputs=None) for s in (0, 1)],
                after=True,
            ),
            "contained",
        ),
        (
            _cell(
                "mps",
                "faulted",
                [_run(s, timeout=False, a_faulted=True, a_error=ILLEGAL_ADDRESS) for s in (0, 1)],
            ),
            "faulted",
        ),
        (
            _cell("mps", "timeout", [_run(s, timeout=s == 1) for s in (0, 1)]),
            "timeout",
        ),
    ],
)
def test_defect_outcome_is_the_one_that_the_runs_show(cell, outcome) -> None:
    assert rq1._defect_outcome(cell) == outcome


@pytest.mark.parametrize(
    "cell",
    [
        _cell("guardian", "contained", [_run(0), _run(1, a_changes=[CHANGE])]),
        _cell("guardian", "contained", [_run(0), _run(1, b_outputs={"out": "AQAA"})], True),
        _cell("guardian", "contained", [_run(0)]),
        _cell("mps", "contained", [_run(0), _run(1)], after=True) | {"runs": []},
        {"id": "m/t/numba", "system": "mps", "outcome": "unsupported", "reason": "r"},
        {"id": "m/t/numba", "system": "guardian", "outcome": "contained"},
        _cell("guardian", "skipped", [_run(0), _run(1)]),
    ],
)
def test_defect_outcome_rejects_cells_that_contradict_their_raw_fields(cell) -> None:
    with pytest.raises(ValueError):
        rq1._defect_outcome(cell)
