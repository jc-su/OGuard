"""Reports require actual evidence and leave published files intact on failure."""

import copy
import csv
import json
from dataclasses import replace

import pytest

from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.export import __main__ as exporter
from eval.export import corpus as corpus_module
from eval.export import prologue, rq1, rq2, rq3, rq4
from eval.export.corpus import Kernel
from eval.export.records import output_to, write
from eval.rq1_safety.sass_prologue import KIND as SASS_KIND
from eval.rq4_cost.registration import KIND as REGISTRATION_KIND
from eval.rq4_cost.registration import Input
from eval.timing import RUNS
from opgate.binding import GRID_LIMIT, Binding
from opgate.gate import CHECK_QUERY, REGISTRATION_SECONDS, TIMEOUT_REASON, export_terms
from opgate.hosts.service.audit import binding_record
from opgate.signature import Affine


@pytest.fixture
def service_record(raw_record):
    return raw_record("service_boundary_2026-10-09.json")


def test_recorded_service_evidence_is_consistent(service_record):
    rq1.validate_service(service_record)


@pytest.mark.parametrize("change", ["none", "decision", "count", "duplicate", "timeout", "fault"])
def test_triggered_kernel_report_checks_each_actual_outcome(change, raw_record):
    record = raw_record("corpus_ptx_2026-10-09.json")
    if change == "fault":
        record["kernels"][0]["e3"]["pool_fault"] = "unknown"
    elif change == "decision":
        record["kernels"][0]["gate"]["decision"] = "accepted"
    elif change == "count":
        record["counts"]["effects"]["context_fault"] += 1
    elif change == "duplicate":
        record["kernels"][1] = copy.deepcopy(record["kernels"][0])
    elif change == "timeout":
        record["kernels"][0]["gate"]["reason"] = "the check did not finish in 180 seconds"
    if change == "none":
        assert rq1.validate_triggered(record)[1] == 27
    else:
        with pytest.raises(ValueError):
            rq1.validate_triggered(record)


@pytest.mark.parametrize(
    "change",
    ["output", "ptx", "binding", "load", "refusal", "success", "launch", "checks", "victim"],
)
def test_service_report_rejects_inconsistent_raw_evidence(service_record, change):
    record = copy.deepcopy(service_record)
    row = next(
        r
        for r in record["final_audit"]
        if r["plan_kind"] == "read_one_i64" and r["decision"] == "accepted"
    )
    if change == "output":
        record["safe_control"]["observed_value"] += 1
    elif change == "ptx":
        row["ptx"] += "\n"
    elif change == "binding":
        row["binding"]["block"][0] += 1
    elif change == "load":
        row["module_load_count"] = 0
    elif change == "refusal":
        record["tenant_isolation"]["cross_tenant_read"]["error"]["kind"] = "fatal"
    elif change == "launch":
        record["wild_launch"]["error"]["kind"] = "refused"
    elif change == "checks":
        record["registrations"]["wild_response"]["checks"] = []
    elif change == "victim":
        record["tenant_isolation"]["victim_observed_value"] += 1
    else:
        record["success"] = False
    with pytest.raises(ValueError):
        rq1.validate_service(record)


def test_late_export_failure_preserves_every_published_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "old.csv").write_bytes(b"old result\r\n")
    (tmp_path / "numbers.tex").write_bytes(b"old numbers\n")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(exporter, "FIGURES", tmp_path)
    monkeypatch.setattr(exporter, "DATA", data)
    monkeypatch.setattr(exporter, "corpus", lambda path: [])

    def first(kernels):
        write("new.csv", ("value",), [(1,)])
        return {}

    def failed(kernels):
        raise ValueError("incomplete evidence")

    monkeypatch.setattr(rq1, "export", first)
    monkeypatch.setattr(rq2, "export", failed)
    with pytest.raises(ValueError, match="incomplete"):
        exporter.generate(tmp_path / "registration.json")
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def kernel(**changes):
    original = Kernel(
        key=("model", "task", "nvcc"),
        tasks="held-out-3",
        cls="safe",
        baseline="accepted",
        check="",
        repaired=0,
        final="accepted",
        seconds=1.0,
        bytes=10,
        instructions=1,
        features=(),
        source="same_kernel",
    )
    return replace(original, **changes)


def _checked(kernels, decision="refused", reason="not proved"):
    """Rows of the guarded record: a kernel accepted as written stays accepted,
    and each other kernel has `decision`, with `reason` when refused."""
    return [
        {
            "id": "/".join(k.key),
            "first": "accepted" if k.final == "accepted" else "refused",
            "first_reason": (
                None
                if k.final == "accepted"
                else f"{TIMEOUT_REASON} 180 seconds"
                if k.final == "timeout"
                else k.check or "not proved"
            ),
            "decision": "accepted" if k.final == "accepted" else decision,
            "reason": None if k.final == "accepted" or decision == "accepted" else reason,
        }
        for k in kernels
    ]


def _acceptance(path):
    with (path / "data/acceptance.csv").open() as handle:
        return {r["group"]: r for r in csv.DictReader(handle)}


def test_development_set_sums_its_task_sets_in_both_reports(tmp_path):
    kernels = [
        kernel(key=("model", "a", "nvcc")),
        kernel(key=("model", "b", "nvcc"), tasks="development"),
        kernel(key=("model", "c", "nvcc"), tasks="datakernelbench", cls="C", final="refused"),
    ]
    with output_to(tmp_path):
        numbers = rq2.export_acceptance(kernels, _checked(kernels)) | rq1.export_safety(kernels)
    assert numbers["valKernelsDevelopment"] == "3"
    assert numbers["valSafeDevelopmentAccepted"] == "2"
    assert numbers["valDefectsDevelopmentContract"] == "1"
    assert numbers["valDefectsTestContract"] == "0"
    row = _acceptance(tmp_path)["development set CUDA C++"]
    assert row["safe"] == "2" and row["as_written"] == "2"


def test_headline_coverage_counts_all_populations_without_defects(tmp_path):
    cases = [
        kernel(key=("model", "a", "nvcc"), tasks="development"),
        kernel(key=("model", "b", "nvcc"), tasks="datakernelbench", final="refused"),
        kernel(key=("model", "c", "nvcc"), tasks="datakernelbench-core", final="timeout"),
        kernel(key=("model", "d", "nvcc"), tasks="held-out-3", cls="C", final="refused"),
    ]
    guarded = _checked(cases[:2], decision="accepted") + _checked(
        cases[2:], reason=f"{TIMEOUT_REASON} 180 seconds"
    )
    with output_to(tmp_path):
        numbers = rq2.export_acceptance(cases, guarded)
    assert numbers["valSafeDevelopment"] == "3"
    assert numbers["valSafeDevelopmentAccepted"] == "1"
    assert numbers["valSafeDevelopmentRefused"] == "1"
    assert numbers["valSafeDevelopmentTimeout"] == "1"
    row = _acceptance(tmp_path)["development set CUDA C++"]
    assert [row[c] for c in ("safe", "as_written", "checked", "refused", "timeout")] == [
        "3",
        "1",
        "1",
        "0",
        "1",
    ]


def test_test_set_sums_held_out_4_and_5_apart_from_the_development_set(tmp_path):
    kernels = [
        kernel(key=("model", "a", "nvcc"), tasks="held-out-4", final="accepted"),
        kernel(key=("model", "b", "nvcc"), tasks="held-out-5", final="refused"),
        kernel(key=("model", "c", "nvcc"), tasks="held-out-3"),
    ]
    with output_to(tmp_path):
        numbers = rq2.export_acceptance(kernels, _checked(kernels)) | rq1.export_safety(kernels)
    assert numbers["valSafeTest"] == "2"
    assert numbers["valSafeTestAccepted"] == "1"
    assert numbers["valSafeTestRefused"] == "1"
    assert numbers["valSafeDevelopment"] == "1"
    assert numbers["valKernels"] == "3"
    rows = _acceptance(tmp_path)
    assert rows["test set CUDA C++"]["safe"] == "2"
    assert rows["test set CUDA C++"]["refused"] == "1"
    assert rows["development set CUDA C++"]["safe"] == "1"


@pytest.mark.parametrize("change", ["missing", "refused"])
def test_acceptance_requires_each_safe_kernel_with_its_decision_with_checks(tmp_path, change):
    kernels = [kernel(key=("model", "a", "nvcc")), kernel(key=("model", "b", "nvcc"))]
    guarded = _checked(kernels)
    if change == "missing":
        del guarded[1]
    else:
        guarded[0]["decision"] = "refused"
    with output_to(tmp_path), pytest.raises(ValueError):
        rq2.export_acceptance(kernels, guarded)


def _decided(row):
    """The kernel of a guarded record's row, with the decision as written that it states."""
    model, task, toolchain = row["id"].split("/")
    return kernel(
        key=(model, task, toolchain),
        cls=row["label"],
        final=row["first"],
        check=row["first_reason"] or "",
    )


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "deadline",
        "field",
        "reason",
        "label",
        "decision",
        "changed",
        "limit",
        "duplicate",
        "elapsed",
        "count",
    ],
)
@pytest.mark.parametrize("library", [True, False])
def test_guarded_report_checks_each_row_against_its_registration(
    tmp_path, monkeypatch, change, library
):
    from tests.eval.test_guarded_record import measured

    path = measured(tmp_path, monkeypatch)
    record = json.loads(path.read_text())
    first = next(r for r in record["results"] if r["first"] == "accepted")
    checked = next(
        r
        for r in record["results"]
        if r["first"] != r["decision"] and TIMEOUT_REASON not in r["first_reason"]
    )
    rows = [copy.deepcopy(first), copy.deepcopy(checked)]
    kernels = [_decided(r) for r in rows]
    if change == "deadline":
        rows[1]["first_reason"] = f"{TIMEOUT_REASON} 180 seconds"
    elif change == "field":
        rows[0]["note"] = ""
    elif change == "reason":
        rows[1]["first_reason"] += " "
    elif change == "label":
        rows[1]["label"] = "C" if rows[1]["label"] == "safe" else "safe"
    elif change == "decision":
        rows[1]["reason"] = "accepted with a reason"
    elif change == "changed":
        rows[0]["guarded"] = rows[1]["guarded"] or ["ld.global.u32 %r1, [%rd1]"]
    elif change == "limit":
        rows[0]["restructured"] = True
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "elapsed":
        rows[0]["seconds"] = -1
    elif change == "count":
        rows[0]["checks"] = 0
    record["results"] = rows
    if not library:
        (tmp_path / "source.v").write_text("a different proof library")
    copied = tmp_path / path.name
    copied.write_text(json.dumps(record))
    if change in ("none", "deadline") and library:
        rows, deadline = rq2.guarded_rows(copied, kernels)
        assert len(rows) == 2 and deadline == (change == "deadline")
    else:
        with pytest.raises(ValueError):
            rq2.guarded_rows(copied, kernels)


@pytest.mark.parametrize("direction", ["lost", "recovered", "contradiction"])
def test_coverage_keeps_actual_replay_outcomes_when_a_deadline_differs(
    tmp_path, monkeypatch, direction
):
    from tests.eval.test_guarded_record import measured

    path = measured(tmp_path, monkeypatch)
    record = json.loads(path.read_text())
    kernels = [_decided(r) for r in record["results"]]
    row = record["results"][0]
    timed_out = f"{TIMEOUT_REASON} 180 seconds"
    if direction in ("lost", "contradiction"):
        reason = timed_out if direction == "lost" else "not proved"
        row.update(first="refused", first_reason=reason, decision="refused", reason=reason)
    else:
        row["recorded"] = "refused"
        kernels[0] = replace(kernels[0], final="timeout", check=timed_out)
        archived = path.with_suffix(".json.inputs") / "registration/registration.json"
        original = json.loads(archived.read_text())
        original["results"][0].update(accepted=False, reason=timed_out)
        archived.write_text(json.dumps(original))
    path.write_text(json.dumps(record))
    if direction == "contradiction":
        with pytest.raises(ValueError, match="differs from its registration"):
            rq2.guarded_rows(path, kernels)
        return
    rows, differences = rq2.guarded_rows(path, kernels)
    assert differences == 1
    with output_to(tmp_path):
        numbers = rq2.export_acceptance(kernels, rows)
    assert numbers["valSafeDevelopmentAccepted"] == ("0" if direction == "lost" else "1")
    assert numbers["valSafeDevelopmentTimeout"] == ("1" if direction == "lost" else "0")
    exported = _acceptance(tmp_path)["development set CUDA C++"]
    assert exported["checked"] == "1"
    assert exported["as_written"] == numbers["valSafeDevelopmentAccepted"]


@pytest.mark.parametrize("change", ["none", "unreported", "written", "escaped", "runs"])
def test_guarded_defect_report_requires_contained_runs_that_report_a_failed_check(
    tmp_path, monkeypatch, change, raw_record
):
    record = raw_record("guarded_defects_2026-10-09.json")
    # The fixture omits its machine-local proof path; this test checks raw outcomes.
    record["library"] = str(rq1.formal_dir())
    accepted = next(d for d in record["defects"] if d["decision"] == "accepted")
    checked = next(r for r in accepted["runs"] if r["mode"] == "guarded")
    written = next(r for r in accepted["runs"] if r["mode"] == "as_written")
    if change == "unreported":
        checked["check_failed"] = False
    elif change == "written":
        written["check_failed"] = True
    elif change == "escaped":
        checked["changed"] = copy.deepcopy(written["changed"]) or [{"offset": 0}]
    elif change == "runs":
        next(d for d in record["defects"] if d["decision"] == "refused")["runs"] = []
    copied = tmp_path / rq1.GUARDED_DEFECTS.name
    copied.write_text(json.dumps(record))
    monkeypatch.setattr(rq1, "GUARDED_DEFECTS", copied)
    with output_to(tmp_path):
        if change == "none":
            numbers = rq1.export_guarded_defects()
            assert numbers["valGuardedDefectsAccepted"] == "19"
        else:
            with pytest.raises(ValueError):
                rq1.export_guarded_defects()


def test_configurations_of_one_plan_kernel_are_one_kernel_and_several_entries():
    first = kernel(
        key=("qwen3.5-397b", "q6_sum", "nvcc"), tasks="datakernelbench-core", source="q6_sum"
    )
    second = replace(first, key=("qwen3.5-397b", "q6_sum_2", "nvcc"))
    full = replace(first, key=("qwen3.5-397b", "q6_full", "nvcc"), tasks="datakernelbench")
    synthesized = [
        kernel(key=("opus", "gather", "nvcc"), tasks="held-out-4", source="gather"),
        kernel(key=("opus", "gather", "triton"), tasks="held-out-4", source="gather"),
    ]
    numbers = rq2.export_collections([first, second, full, *synthesized])
    # the two configurations of the core plan's kernel are one kernel; the full
    # plan's kernel and each synthesized entry are one kernel each
    assert numbers["valDkbEntries"] == "3" and numbers["valDkbKernels"] == "2"
    assert numbers["valSynthesizedEntries"] == numbers["valSynthesizedKernels"] == "2"
    assert numbers["valKernelsUnique"] == "4"
    assert numbers["valKernelsDevelopmentUnique"] == numbers["valKernelsTestUnique"] == "2"
    assert numbers["valDkbModels"] == numbers["valSynthesizedModels"] == "1"


def _families(path):
    with (path / "data/families.csv").open() as handle:
        return {(r["family"], r["set"]): r for r in csv.DictReader(handle)}


def test_families_count_each_set_apart_and_their_models_together(tmp_path):
    kernels = [
        kernel(key=("haiku", "a", "nvcc"), tasks="development"),
        kernel(key=("opus", "b", "nvcc"), tasks="held-out-4", final="refused"),
        kernel(key=("sonnet", "c", "nvcc"), tasks="held-out-5"),
        kernel(key=("gpt-5.5", "q1", "nvcc"), tasks="datakernelbench", final="refused"),
        kernel(key=("gpt-oss-120b", "q6", "triton"), tasks="datakernelbench-core"),
        kernel(key=("gpt-oss-120b", "q3", "triton"), tasks="datakernelbench-core", cls="C"),
    ]
    guarded = _checked(kernels[:3]) + _checked(kernels[3:5], decision="accepted")
    with output_to(tmp_path):
        numbers = rq2.export_families(kernels, guarded)
    rows = _families(tmp_path)
    # the families in the figure's order, each set apart; a defect is not counted
    assert list(rows) == [
        ("Claude", "development set"),
        ("Claude", "test set"),
        ("GPT", "development set"),
    ]
    test = rows[("Claude", "test set")]
    assert [test[c] for c in ("safe", "as_written", "checked", "refused")] == ["2", "1", "0", "1"]
    gpt = rows[("GPT", "development set")]
    assert [gpt[c] for c in ("safe", "as_written", "checked", "refused")] == ["2", "1", "1", "0"]
    assert numbers["valModels"] == "5" and numbers["valFamilies"] == "2"
    assert numbers["valFamilyClaudeTestWrittenPercent"] == "50.0"
    assert numbers["valFamilyClaudeTestCheckedPercent"] == "50.0"
    assert numbers["valFamilyGPTDevelopmentCheckedPercent"] == "100.0"
    assert "valFamilyGPTTestWrittenPercent" not in numbers


def test_families_refuse_a_model_of_no_known_family(tmp_path):
    kernels = [kernel(key=("unknown-model", "a", "nvcc"))]
    with output_to(tmp_path), pytest.raises(ValueError, match="no known family"):
        rq2.export_families(kernels, _checked(kernels))


def test_current_coverage_does_not_count_historical_repairs_as_original_acceptances(tmp_path):
    kernels = [
        kernel(key=("model", "a", "nvcc"), final="accepted", baseline=None, repaired=None),
        kernel(key=("model", "b", "nvcc"), final="refused", baseline="refused", repaired=1),
        kernel(key=("model", "c", "nvcc"), final="timeout", baseline="refused", repaired=2),
        kernel(key=("model", "d", "nvcc"), cls="C", final="refused", repaired=1),
        kernel(key=("model", "e", "nvcc"), cls="E", final="timeout", repaired=None),
    ]
    guarded = _checked(kernels[:2]) + _checked(kernels[2:3], reason=f"{TIMEOUT_REASON} 180 seconds")
    with output_to(tmp_path):
        rq2.export_acceptance(kernels, guarded)
    row = _acceptance(tmp_path)["development set CUDA C++"]
    assert row["safe"] == "3" and row["checked"] == "0"
    assert all(row[c] == "1" for c in ("as_written", "refused", "timeout"))
    assert "repaired_safe" not in row and "accepted" not in row


@pytest.mark.parametrize(
    "change",
    ["none", "ptx", "binding", "missing", "unfinished", "query", "processes", "budget", "kind"],
)
def test_registration_matches_actual_inputs(tmp_path, monkeypatch, change):
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    text = ".visible .entry kernel()\r\n{\r\nret;\r\n}\r\n// A\r\n"
    item = Input(("model", "task", "nvcc"), text, binding, "safe", "refused")
    key = "/".join(item.key)
    output = tmp_path / "registration.json"
    monkeypatch.setattr(corpus_module, "inputs", lambda: [item])
    monkeypatch.setattr(corpus_module, "source_files", lambda library: {})
    (tmp_path / "registration.json.inputs/sources").mkdir(parents=True)
    provenance: dict[str, object] = {
        "kind": REGISTRATION_KIND,
        "query_suffix": CHECK_QUERY,
        "processes_per_verification": 1,
        "registration_seconds": REGISTRATION_SECONDS,
    }
    if change == "query":
        provenance["query_suffix"] = "Goal True. exact I. Qed.\n"
    elif change == "processes":
        provenance["processes_per_verification"] = 2
    elif change == "budget":
        provenance["registration_seconds"] = 180.0
    elif change == "kind":
        provenance["kind"] = "registration with raw verification inputs"
    with Checkpoint(output, provenance) as record:
        record.artifact(key, item.ptx, "ptx")
        record.artifact(key, export_terms(item.ptx, [binding]), "v")
        if change != "missing":
            row = {
                "id": key,
                "ptx_artifact": key,
                "query_artifact": key,
                "accepted": False,
                "reason": "not proved",
                "seconds": 1.0,
                "ptx_bytes": len(item.ptx),
                "binding": binding_record(binding),
                "verdict": "safe",
                "baseline": "refused",
            }
            if change == "binding":
                row["binding"] = binding_record(replace(binding, grid=(1, 1, 1)))
            record.add(row)
        if change != "unfinished":
            record.finish()
    if change == "ptx":
        (tmp_path / "registration.json.artifacts/model/task/nvcc.ptx").write_bytes(
            text.replace("// A", "// B").encode("ascii")
        )
    if change == "none":
        assert corpus_module._final(output)[item.key][0] == "refused"
    else:
        with pytest.raises(ValueError):
            corpus_module._final(output)


def _result(microseconds, failed):
    return {"check": None, "check_failed": failed, "time": {"median_us": microseconds}}


WRITTEN = "// the kernel as written\n"


def _cost_kernel(name, results, **fields):
    """A kernel of the cost record whose texts that the executor loads, and
    Guardian's, differ from the kernel as written."""
    entry = {
        "kernel": name,
        "original": {"ptx": WRITTEN, "binding": {}},
        "recorded": "refused",
        "decision": "accepted",
        "guarded_accesses": [],
        "limited_loops": [],
        "restructured": False,
        "every_decision": "accepted",
        "every_checks": 2,
        "guardian_fences": 3,
        "guardian_unsupported": None,
        "loaded": {
            "opgate": WRITTEN + "// checked\n",
            "every": WRITTEN + "// every access checked\n",
            "guardian": WRITTEN + "// fenced\n",
        },
        "results": results,
    }
    return entry | fields


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "untimed",
        "unreported",
        "written",
        "failed",
        "later",
        "incomplete",
        "stale",
        "no_common",
        "prologue",
        "prologue_untimed",
        "timed_as_written",
        "unloaded",
        "loaded_differs",
    ],
)
def test_guard_cost_times_every_loaded_text_that_differs_from_the_kernel(
    tmp_path, monkeypatch, change
):
    limited = {
        "as_written": _result(10.0, None),
        "opgate": _result(11.0, False),
        "every": _result(12.0, False),
        "guardian": _result(13.0, None),
    }
    if change == "untimed":
        del limited["opgate"]
    elif change == "unreported":
        limited["opgate"]["check_failed"] = None
    elif change == "written":
        limited["as_written"]["check_failed"] = False
    elif change == "failed":
        limited["every"]["check_failed"] = True
    elif change == "no_common":
        del limited["every"]
    # accepted as written, with no shared or local memory: the text that the
    # service loads is the kernel as written, and the gate refuses every access
    # checked
    written = _cost_kernel(
        "sonnet/q6_sum/numba",
        {"as_written": _result(10.0, None), "guardian": _result(11.0, None)},
        recorded="accepted",
        every_decision="not proved",
        loaded={"opgate": WRITTEN, "guardian": WRITTEN + "// fenced\n"},
    )
    if change in ("prologue", "prologue_untimed"):
        # accepted as written, with the zeroing prologue of its shared memory
        written["loaded"]["opgate"] = WRITTEN + "// the zeroing prologue\n"
    if change == "prologue":
        written["results"]["opgate"] = _result(12.0, None)
    elif change == "timed_as_written":
        written["results"]["opgate"] = _result(10.0, None)
    elif change == "unloaded":
        del written["loaded"]["opgate"]
    kernels = [
        written,
        _cost_kernel("fable/groupby_sum/numba", limited, limited_loops=["@%p1 bra $L__BB0_1"]),
        _cost_kernel("fable/hash_probe/nvcc", {}, decision="refused"),
        _cost_kernel(
            "model/incorrect_answer/nvcc",
            {
                "as_written": _result(1.0, None) | {"check": "wrong answer"},
                "opgate": _result(1000.0, False) | {"check": "wrong answer"},
                "every": _result(1000.0, False) | {"check": "wrong answer"},
                "guardian": _result(1000.0, None) | {"check": "wrong answer"},
            },
            limited_loops=["ordinary loop"],
        ),
    ]
    del kernels[2]["results"], kernels[2]["loaded"]
    directory = tmp_path / "guard_cost"
    directory.mkdir()
    source = tmp_path / "source.py"
    source.write_text("original implementation\n")
    monkeypatch.setattr(rq4, "guard_cost_sources", lambda: {"source.py": source})
    for run in range(1, rq4.GUARD_COST_RUNS + 1):
        record = {
            "complete": True,
            "core": 40,
            "environment": {"gpu": "none"},
            "kernels": copy.deepcopy(kernels),
        }
        if run == 2 and change == "later":
            record["kernels"][1]["results"]["every"]["check_failed"] = True
        if run == 2 and change == "loaded_differs":
            record["kernels"][1]["loaded"]["opgate"] += "// another text\n"
        if run == rq4.GUARD_COST_RUNS and change == "incomplete":
            record["complete"] = False
        (directory / f"run{run}.json").write_text(json.dumps(record))
        snapshot_inputs(
            directory / f"run{run}.json", "sources", rq4.guard_cost_sources(), create=True
        )
    if change == "stale":
        source.write_text("different implementation\n")
    monkeypatch.setattr(rq4, "GUARD_COST", directory)
    with output_to(tmp_path):
        if change == "none":
            numbers = rq4.export_guard_cost()
            assert numbers["valCostKernels"] == "3"
            assert numbers["valCostUsefulKernels"] == "2"
            assert numbers["valCostWrongAsWritten"] == "1"
            assert numbers["valCostOpgateMax"] == "10"
            assert numbers["valCostCommonKernels"] == "1"
            assert numbers["valCostCommonOpgateGeomean"] == "10.0"
            assert numbers["valCostCommonEveryGeomean"] == "20.0"
            assert numbers["valCostCommonGuardianGeomean"] == "30.0"
            # the cost set holds the refused kernel; the gate runs the three others, the
            # every-access text two of them, and Guardian's three
            assert numbers["valCostSet"] == "4"
            assert numbers["valCostOpgateCovered"] == "3"
            assert numbers["valCostEveryCovered"] == "2"
            assert numbers["valCostGuardianCovered"] == "3"
            # the useful kernels: one that the pass left as written, whose ratio is 1, and
            # one that it changed, which alone gives the changed overhead
            assert numbers["valCostChangedKernels"] == "1"
            assert numbers["valCostChangedGeomean"] == "10.0"
            assert numbers["valCostChangedMedian"] == "10.0"
            assert numbers["valCostChangedDoubled"] == "0"
            assert numbers["valCostOpgateMaxRatio"] == "1.10"
            assert numbers["valCostOpgateGeomean"] == "4.9"
            with (tmp_path / "data/guard_cost_common.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert [r["kernel"] for r in rows] == ["fable/groupby_sum/numba"]
            assert [rows[0][v] for v in rq4.CHECKED_VERSIONS] == ["1.1000", "1.2000", "1.3000"]
        elif change == "prologue":
            numbers = rq4.export_guard_cost()
            # the prologue is timed, and the kernel is not one that the pass changed
            assert numbers["valCostOpgateMaxRatio"] == "1.20"
            assert numbers["valCostChangedKernels"] == "1"
            assert numbers["valCostOpgateCovered"] == "3"
        else:
            with pytest.raises(ValueError):
                rq4.export_guard_cost()


SHARED_PTX = """.version 8.8
.target sm_86
.address_size 64
.shared .align 4 .b8 s[128];
.visible .entry k(.param .u64 out)
{
	ret;
}
"""
JIT = "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.610.57.04"


def _sass_row(key, group, compiler=JIT, passed=True):
    return {"id": key, "group": group, "compiler": compiler, "passed": passed}


def test_the_sass_check_lists_the_guarded_texts_that_zeroed_gives_a_prologue(tmp_path, monkeypatch):
    binding = binding_record(Binding("k", (), (), (32, 1, 1), (1, 1, 1), Affine(0)))
    rows = []
    for key, ptx, first in (
        ("m/shared/nvcc", SHARED_PTX, "refused"),
        ("m/plain/nvcc", SHARED_PTX.replace(".shared .align 4 .b8 s[128];\n", ""), "refused"),
        ("m/written/nvcc", SHARED_PTX, "accepted"),
    ):
        artifact = tmp_path / "guarded.json.artifacts" / key / "final.ptx"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(ptx)
        rows.append({"id": key, "first": first, "decision": "accepted", "binding": binding})
    (tmp_path / "guarded.json").write_text(json.dumps({"results": rows}))
    monkeypatch.setattr(prologue, "GUARDED", tmp_path / "guarded.json")
    assert prologue._with_checks() == ["m/shared/nvcc"]


@pytest.mark.parametrize(
    "change", ["none", "failed", "compilers", "with_checks", "as_written", "summary", "guarded"]
)
def test_the_sass_check_counts_each_group_and_names_the_driver(tmp_path, monkeypatch, change):
    rows = [
        _sass_row("a/one/nvcc", "as_written"),
        _sass_row("a/two/triton", "as_written"),
        _sass_row("c/three/numba", "with_checks"),
    ]
    if change == "failed":
        rows[2]["passed"] = False
    elif change == "compilers":
        rows[1]["compiler"] = JIT.replace("610.57.04", "610.43.02")
    guarded = prologue.GUARDED.name if change != "guarded" else "guarded_2026-10-23.json"
    record = {
        "complete": True,
        "provenance": {
            "kind": SASS_KIND,
            "registration": f"eval/records/{prologue.REGISTRATION.name}",
            "guarded": f"eval/records/{guarded}",
        },
        "results": rows,
        "summary": {
            "kernels": 3,
            "as_written": 2 if change != "summary" else 3,
            "with_checks": 1,
            "passed": 3,
            "failed": {},
        },
    }
    path = tmp_path / "sass.json"
    path.write_text(json.dumps(record))
    source = tmp_path / "source.py"
    source.write_text("implementation\n")
    monkeypatch.setattr(prologue, "sass_sources", lambda: {"source.py": source})
    snapshot_inputs(path, "sources", prologue.sass_sources(), create=True)
    monkeypatch.setattr(prologue, "SASS", path)
    checked = ["c/three/numba"] if change != "with_checks" else ["c/three/numba", "c/four/numba"]
    monkeypatch.setattr(prologue, "_with_checks", lambda: checked)
    listed = ["a/one/nvcc", "a/two/triton"] if change != "as_written" else ["a/one/nvcc"]
    if change == "none":
        assert prologue.export_sass(listed) == {
            "valSassKernels": "3",
            "valSassWritten": "2",
            "valSassChecked": "1",
            "valSassPassed": "3",
            "valSassDriverVersion": "610.57.04",
        }
    else:
        with pytest.raises(ValueError):
            prologue.export_sass(listed)


@pytest.fixture
def isolation_record(raw_record):
    """The record that the experiment writes for every configuration, from the
    archived runs of its earlier records: the shared context, process, and
    service runs, the MPS run, and a guardian run."""
    record = raw_record("fault_isolation_2026-10-01.json")
    (mps,) = raw_record("fault_isolation_mps_2026-10-01.json")["runs"]
    shared, process, service = record["runs"]
    record["runs"] = [shared, process, mps, service, _guardian_fault_run(process)]
    record["configurations"] = list(rq4.ISOLATION_ORDER)
    return record


def _guardian_fault_run(process):
    """A guardian fault-isolation run from the process run's contained victims:
    the attacker's fenced write lands in its own partition, not the victim's."""
    victims = [copy.deepcopy(t) for t in process["tenants"] if t["role"] == "victim"]
    runtime, fault = process["runtime_seconds"], process["fault_at_seconds"]
    size, base = 1 << 20, 2 << 20
    attacker = {
        "tenant": 9000,
        "role": "attacker",
        "correct": True,
        "error": None,
        "ended": False,
        "note": "the fence confined the write",
        "completions": [],
        "completed_after_fault": 0,
        "max_gap_seconds": runtime,
    }
    return {
        "configuration": "guardian",
        "attacker": True,
        "victims": process["victims"],
        "runtime_seconds": runtime,
        "fault_at_seconds": fault,
        "tenants": [*victims, attacker],
        "service": {
            "guardian": {
                "partition_bytes": size,
                "attacker_base": base,
                "unfenced_target": base + size,
                "fenced_target": base,
                "sink_before": 0,
                "sink_after": 0x5AFEDEAD5AFEDEAD,
                "write_value": 0x5AFEDEAD5AFEDEAD,
                "victim_word_before": 123,
                "victim_word_after": 123,
                "fenced_accesses": 1,
            }
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "victim",
        "summary",
        "order",
        "attacker",
        "check",
        "shared",
        "window",
        "partition",
        "guardian_leak",
        "guardian_fence",
        "guardian_unfenced",
        "configurations",
        "missing",
    ],
)
def test_fault_isolation_report_requires_the_recorded_outcome_of_each_configuration(
    tmp_path, monkeypatch, isolation_record, change
):
    shared, process, mps, service, guardian = isolation_record["runs"]
    if change == "victim":
        process["tenants"][0]["ended"] = True
    elif change == "summary":
        process["tenants"][0]["completed_after_fault"] += 1
    elif change == "order":
        completions = process["tenants"][1]["completions"]
        completions[0], completions[1] = completions[1], completions[0]
    elif change == "attacker":
        service["tenants"][-1]["ended"] = True
    elif change == "check":
        service["service"]["telemetry"]["failed_checks"] = 0
    elif change == "shared":
        shared["tenants"][0] |= {"ended": False, "error": None}
    elif change == "window":
        mps["fault_at_seconds"] = 2.0
    elif change == "partition":
        mps["service"]["mps"]["mode"] = "default"
    elif change == "guardian_leak":
        guardian["service"]["guardian"]["victim_word_after"] = 456
    elif change == "guardian_fence":
        guardian["service"]["guardian"]["sink_after"] = 0
    elif change == "guardian_unfenced":
        guardian["service"]["guardian"]["fenced_accesses"] = 0
    elif change == "configurations":
        isolation_record["configurations"].reverse()
    elif change == "missing":
        del isolation_record["runs"][-1]
    copied = tmp_path / "fault_isolation.json"
    copied.write_text(json.dumps(isolation_record))
    monkeypatch.setattr(rq4, "FAULT_ISOLATION", copied)
    with output_to(tmp_path):
        if change == "none":
            numbers = rq4.export_fault_isolation()
            assert numbers["valIsoProcessStallMs"] == "135"
            assert numbers["valIsoServiceAfterFault"] == "25,711"
            assert {"valIsoGuardianStallMs", "valIsoGuardianAfterFault"} <= numbers.keys()
            with (tmp_path / "data/fault_isolation.csv").open() as handle:
                assert next(csv.reader(handle)) == ["seconds", *rq4.ISOLATION_ORDER]
        else:
            with pytest.raises(ValueError):
                rq4.export_fault_isolation()


PTX = ".visible .entry q6()\n{\nret;\n}\n"


SHARED_BYTES = {"shipdate": 4, "discount": 8, "quantity": 8, "extendedprice": 8, "orderkey": 8}


def _consolidation_run(configuration, scale, tenants):
    """One run of the consolidation record: 100 queries per tenant, the service
    configurations' evidence, Guardian's partition layout, and the shared
    table's owner and mappings. A service holds 500 MiB however many tenants
    use it, the shared table 800 MiB and 265 MiB for each tenant, and every
    other configuration 500 MiB for each tenant and 10 MiB of overhead."""
    queries = 100 * tenants
    rows = 1 << 20 if scale == 1.0 else 1 << 17
    memory = {"service": 500.0, "mps_shared": 800.0 + 265.0 * tenants}
    services = {"service": 1, "service_per_tenant": tenants}.get(configuration, 0)
    counters = ("live_buffers", "service_buffer_handles", "service_operator_handles")
    run = {
        "configuration": configuration,
        "scale": scale,
        "tenants": tenants,
        "repeat": 0,
        "queries": queries,
        "latency_us_samples": [100.0] * queries,
        "latency_us": {"count": queries, "p50": 100.0, "p99": 100.0 * tenants},
        "queries_per_second": queries / 10.0,
        "memory_mib": memory.get(configuration, 500.0 * tenants + 10.0),
        "rows": rows,
        "mps": None,
        "services": [
            {
                "ptx": PTX,
                "audit": [{"module_load_count": 1, "ptx": PTX}],
                "telemetry_after_cleanup": dict.fromkeys(
                    (*counters, "input_errors", "fatal_errors"), 0
                ),
            }
            for _ in range(services)
        ],
        "guardian": None,
        "shared": None,
    }
    if configuration == "mps_shared":
        handles = {name: f"{index:02x}" * 64 for index, name in enumerate(SHARED_BYTES)}
        run["mps"] = {"mode": "default"}
        run["shared"] = {
            "columns": {
                name: {
                    "bytes": width * rows,
                    "allocated": max(2 << 20, width * rows),
                    "address": 0,
                    "handle": handles[name],
                }
                for name, width in SHARED_BYTES.items()
            },
            "compared_bytes": 36 * rows,
        }
        run["per_tenant"] = [
            {
                "tenant": tenant,
                "request": {
                    "fenced": False,
                    "mapped": {
                        name: {"address": tenant << 30, "handle": handles[name]}
                        for name in SHARED_BYTES
                    },
                },
            }
            for tenant in range(1, tenants + 1)
        ]
    if configuration == "mps_static":
        # 21 chunks of 4 SMs, an equal share of whole chunks for each tenant
        each = 21 // tenants
        run["mps"] = {
            "mode": "static_sm_partitioning",
            "total_chunks": 21,
            "total_sms": 84,
            "chunks_per_partition": each,
            "sms_per_partition": 4 * each,
            "unassigned_chunks": 21 - tenants * each,
        }
    if configuration == "guardian":
        size = 1 << 20
        run["guardian"] = {
            "partition_bytes": size,
            "pool_bytes": size * (tenants + 1),
            "pool_address": 0,
            "first_partition": 0,
            "tenants": [{"base": i * size, "mask": size - 1} for i in range(tenants)],
            "fenced_accesses": 4,
            "ptx": "fenced",
        }
    return run


def _consolidation_record(tenant_counts=(1, 8, 16)):
    scales = (0.1, 1.0)
    return {
        "complete": True,
        "sources": {},
        "configurations": list(rq4.CONFIGURATIONS),
        "scales": list(scales),
        "tenant_counts": list(tenant_counts),
        "repeats": 1,
        "duration_seconds": 10.0,
        "workload": {"ptx": PTX},
        "runs": [
            _consolidation_run(c, s, n)
            for c in rq4.CONFIGURATIONS
            for s in scales
            for n in tenant_counts
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "eight",
        "sixteen",
        "guardian",
        "services",
        "memory",
        "partition",
        "count",
        "mapping",
        "handle",
        "compared",
        "allocated",
        "copied",
        "shared",
    ],
)
def test_consolidation_gives_each_configurations_memory_and_its_ratio_to_the_service(
    tmp_path, monkeypatch, change
):
    counts = {"eight": (1, 16), "sixteen": (1, 8), "count": (1, 8, 16, 32)}
    record = _consolidation_record(counts.get(change, (1, 8, 16)))
    runs = record["runs"]
    if change == "partition":
        static = [r for r in runs if r["configuration"] == "mps_static" and r["tenants"] == 8]
        static[0]["mps"]["unassigned_chunks"] = 0
    elif change == "guardian":
        next(r for r in runs if r["configuration"] == "guardian")["guardian"]["tenants"].pop()
    elif change == "services":
        eight = [
            r for r in runs if r["configuration"] == "service_per_tenant" and r["tenants"] == 8
        ]
        eight[0]["services"].pop()
    elif change == "memory":
        next(r for r in runs if r["configuration"] == "service" and r["tenants"] == 8)[
            "memory_mib"
        ] = "500"
    shared = next(r for r in runs if r["configuration"] == "mps_shared" and r["tenants"] == 1)
    if change == "mapping":
        shared["per_tenant"].pop()
    elif change == "handle":
        shared["per_tenant"][0]["request"]["mapped"]["discount"]["handle"] = "00" * 64
    elif change == "compared":
        shared["shared"]["compared_bytes"] -= 1
    elif change == "allocated":
        shared["shared"]["columns"]["orderkey"]["allocated"] += 256
    elif change == "copied":
        sixteen = [r for r in runs if r["configuration"] == "mps_shared" and r["tenants"] == 16]
        sixteen[0]["memory_mib"] = 9000.0
    elif change == "shared":
        next(r for r in runs if r["configuration"] == "mps")["shared"] = shared["shared"]
    path = tmp_path / "consolidation.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(rq4, "CONSOLIDATION", path)
    monkeypatch.setattr(rq4, "consolidation_sources", lambda: {})
    with output_to(tmp_path):
        if change != "none":
            with pytest.raises(ValueError):
                rq4.export_consolidation()
            return
        numbers = rq4.export_consolidation()
    assert numbers["valConsTenants"] == "16"
    assert numbers["valConsServiceMibEight"] == numbers["valConsServiceMibSixteen"] == "500"
    assert numbers["valConsProcessMibEight"] == "4,010"
    assert numbers["valConsGuardianMibSixteen"] == "8,010"
    assert numbers["valConsProcessMibRatioEight"] == "8.0"
    assert numbers["valConsMpsStaticMibRatioSixteen"] == "16.0"
    assert numbers["valConsServiceEachMibRatioEight"] == "8.0"
    # the shared table: 800 MiB and 265 MiB for each tenant, 36 MiB of table
    assert numbers["valConsMpsSharedMibSixteen"] == "5,040"
    assert numbers["valConsMpsSharedMibRatioSixteen"] == "10.1"
    assert numbers["valConsMpsSharedTenantMib"] == "265"
    assert numbers["valConsTableMib"] == "36.0"
    assert not {"valConsServiceMibRatioEight", "valConsServiceMibRatioSixteen"} & numbers.keys()
    # throughput grows with the tenants, so the most tenants give the highest
    assert numbers["valConsServiceQpsOne"] == "10" and numbers["valConsServiceQps"] == "160"
    assert numbers["valConsServicePeakTenants"] == "16"
    assert numbers["valConsMpsSmallTailUs"] == "1,600"
    assert numbers["valConsServiceMedianUsOne"] == "100"
    # static MPS: 21 chunks of 4 SMs; 8 tenants take 2 chunks each and leave 5
    assert numbers["valConsSms"] == "84" and numbers["valConsChunkSms"] == "4"
    assert numbers["valConsStaticSmsEight"] == "8" and numbers["valConsStaticIdleSmsEight"] == "20"
    assert numbers["valConsStaticSmsSixteen"] == "4" and numbers["valConsStaticIdleSmsOne"] == "0"
    with (tmp_path / "data/consolidation_one.csv").open() as handle:
        header = next(csv.reader(handle))
    assert header[0] == "tenants" and "memory_mib_guardian" in header
    assert header.index("qps_mps_shared") == header.index("memory_mib_mps") + 1


def _torch_run(scale, opgate_us, equal):
    """One run of the PyTorch queries: Q6 in every configuration, compared with
    pandas at scale factor 1 alone."""
    times = {"torch": 300.0, "direct": 100.0, "opgate": opgate_us}
    return {
        "core": 6,
        "environment": {"gpu": "none"},
        "scale": scale,
        "queries": [
            {
                "query": "Q6",
                "mismatches": [],
                "equals_torch": dict.fromkeys(rq3.TORCH_CONFIGURATIONS, True),
                "equals_reference": (
                    dict.fromkeys(rq3.TORCH_CONFIGURATIONS, equal) if scale == 1.0 else None
                ),
                "measured": {c: {"time": {"median_us": t}} for c, t in times.items()},
            }
        ],
    }


@pytest.mark.parametrize("change", ["none", "pandas", "scale"])
def test_torch_queries_require_every_result_to_equal_its_references(tmp_path, monkeypatch, change):
    directories = []
    for scale, opgate in ((1.0, 150.0), (10.0, 400.0)):
        directory = tmp_path / f"torch_{scale:g}"
        directory.mkdir()
        for run in range(1, RUNS + 1):
            record = _torch_run(scale, opgate, equal=not (change == "pandas" and run == 2))
            if change == "scale" and run == 3:
                record["scale"] = 2.0
            (directory / f"run{run}.json").write_text(json.dumps(record))
        directories.append(directory)
    monkeypatch.setattr(rq3, "TORCH_QUERIES", tuple(directories))
    with output_to(tmp_path):
        if change != "none":
            with pytest.raises(ValueError):
                rq3.export_torch_queries()
            return
        numbers = rq3.export_torch_queries()
    # the executor adds 50 and 300 microseconds to the direct launches' 100, the
    # most at scale factor 10 in both time and share
    assert numbers == {
        "valTorchAddLowMs": "0.05",
        "valTorchAddHighMs": "0.30",
        "valTorchAddLowPercent": "50.0",
        "valTorchAddHighPercent": "300",
        "valTorchAddHighMsQuery": "Q6",
        "valTorchAddHighMsScale": "10",
        "valTorchAddHighMsPercent": "300.0",
        "valTorchAddHighPercentQuery": "Q6",
        "valTorchAddHighPercentScale": "10",
        "valTorchAddHighPercentMs": "0.30",
        "valTorchAddHighPercentDirectMs": "0.10",
    }


def _differential_output(data: bytes, *, retain: bool) -> dict[str, object]:
    import base64
    import hashlib

    return {
        "out": {
            "nbytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": base64.b64encode(data).decode("ascii") if retain else None,
        }
    }


def _differential_run(size: int, changed: bytes, written: bytes, check_failed: bool) -> dict:
    retain = size <= 1000
    return {
        "size": size,
        "supported": True,
        "rows": size,
        "changed": {
            "outputs": _differential_output(changed, retain=retain),
            "check": None,
            "check_failed": check_failed,
        },
        "as_written": {"outputs": _differential_output(written, retain=retain), "check": None},
        "same_bytes": {"out": changed == written},
        "same_bytes_all": changed == written,
    }


def _differential_record(tmp_path, monkeypatch):
    from eval.rq2_coverage import differential

    source = tmp_path / "source.v"
    source.write_text("raw proof source\n")
    monkeypatch.setattr(differential, "source_files", lambda: {"source.v": source})
    path = tmp_path / "differential.json"
    provenance = {
        "kind": differential.KIND,
        "guarded_record": "guarded.json",
        "library": "library",
        "environment": {},
        "sizes": list(differential.SIZES),
        "retain_bytes_up_to": differential.RETAIN,
        "kernels": ["model/gather/nvcc", "model/q01/nvcc"],
    }
    with Checkpoint(path, provenance) as record:
        snapshot_inputs(path, "sources", {"source.v": source}, create=True)
        record.add(
            {
                "id": "model/gather/nvcc",
                "guarded_accesses": ["ld.global.u64 %rd1, [%rd2]"],
                "limited_loops": [],
                "restructured": False,
                "workload": True,
                "runs": [
                    _differential_run(0, b"", b"", False),
                    _differential_run(1, b"\x01" * 8, b"\x01" * 8, False),
                    _differential_run(1000, b"\x02" * 8000, b"\x02" * 8000, False),
                    _differential_run(1_000_003, b"\x03" * 16, b"\x03" * 16, False),
                ],
            }
        )
        record.add(
            {
                "id": "model/q01/nvcc",
                "guarded_accesses": [],
                "limited_loops": ["$L__BB0_3"],
                "restructured": True,
                "workload": False,
                "runs": [],
            }
        )
        record.finish()
    return path


@pytest.mark.parametrize("change", ["none", "mismatch", "check_failed", "comparison"])
def test_differential_report_checks_each_run_against_its_bytes(tmp_path, monkeypatch, change):
    path = _differential_record(tmp_path, monkeypatch)
    record = json.loads(path.read_text())
    gather = record["results"][0]
    if change == "mismatch":
        gather["runs"][1] = _differential_run(1, b"\x01" * 8, b"\x09" * 8, False)
    elif change == "check_failed":
        gather["runs"][1]["changed"]["check_failed"] = True
    elif change == "comparison":
        gather["runs"][1]["same_bytes"] = {"out": False}  # disagrees with the retained bytes
        gather["runs"][1]["same_bytes_all"] = False
    path.write_text(json.dumps(record))
    monkeypatch.setattr(rq2, "DIFFERENTIAL", path)
    with output_to(tmp_path):
        if change in ("check_failed", "comparison"):
            with pytest.raises(ValueError):
                rq2.export_differential()
            return
        numbers = rq2.export_differential()
    assert numbers["valDiffKernels"] == "2"
    assert numbers["valDiffWithWorkload"] == "1"
    assert numbers["valDiffNoWorkload"] == "1"
    assert numbers["valDiffRuns"] == "4"
    assert numbers["valDiffCheckFailures"] == "0"
    assert numbers["valDiffByteMismatchRuns"] == ("1" if change == "mismatch" else "0")
    # the kernel as written passes its reference, so the mismatch is the change's
    assert numbers["valDiffMismatchRunsWrittenCorrect"] == ("1" if change == "mismatch" else "0")
    assert numbers["valDiffMismatchedKernels"] == ("1" if change == "mismatch" else "0")
    rows = list(csv.DictReader((tmp_path / "data/differential.csv").open()))
    assert [row["kernel"] for row in rows] == ["model/gather/nvcc", "model/q01/nvcc"]
    assert rows[1]["workload"] == "False" and rows[1]["sizes_run"] == "0"
