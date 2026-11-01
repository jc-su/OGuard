"""The ablation report requires a complete record of every group on the final gate's entries."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval.checkpoint import Checkpoint, snapshot_inputs
from eval.export import rq2
from eval.export.corpus import Kernel
from eval.export.records import output_to
from eval.rq2_coverage.rule_ablation import GROUPS, KIND

ORIGINAL = Kernel(
    key=("model", "task", "nvcc"),
    tasks="development",
    cls="safe",
    baseline="accepted",
    check="",
    repaired=0,
    final="accepted",
    seconds=1.0,
    bytes=10,
    instructions=1,
    features=(),
    source="task",
)
KERNELS = [
    replace(ORIGINAL, key=("model", "a", "nvcc")),
    replace(ORIGINAL, key=("model", "b", "nvcc"), tasks="held-out-5"),
    replace(ORIGINAL, key=("model", "c", "nvcc"), final="refused"),
    replace(ORIGINAL, key=("model", "d", "nvcc"), cls="E", final="refused"),
    replace(ORIGINAL, key=("model", "e", "nvcc"), cls="C", final="timeout"),
]
SAFE = ["model/a/nvcc", "model/b/nvcc"]
DEFECTS = ["model/d/nvcc"]


def _groups() -> list[dict[str, object]]:
    return [
        {
            "name": g.name,
            "description": g.description,
            "rules": list(g.rules),
            "subsumed": list(g.subsumed),
        }
        for g in GROUPS
    ]


def _row(name: str, **changes: str) -> dict[str, object]:
    decisions = {e: "accepted" for e in SAFE} | {e: "refused" for e in DEFECTS} | changes
    return {
        "id": name,
        "decisions": decisions,
        "seconds": {e: 2.0 for e in decisions},
        "build_seconds": 60.0,
    }


def _record(tmp_path: Path, rows: list[dict[str, object]], **provenance: object) -> Path:
    registration = tmp_path / "registration.json"
    registration.write_text("registration bytes")
    library = tmp_path / "library"
    library.mkdir(exist_ok=True)
    (library / "GateParse.v").write_text("proof source")
    (library / "Makefile").write_text("FILES := GateParse.v\n\n")
    path = tmp_path / "ablation.json"
    frozen: dict[str, object] = {
        "kind": KIND,
        "groups": _groups(),
        "safe": SAFE,
        "defects": DEFECTS,
        "timeout": 360.0,
    } | provenance
    snapshot_inputs(path, "registration", {"registration.json": registration}, create=True)
    snapshot_inputs(path, "library", {"GateParse.v": library / "GateParse.v"}, create=True)
    with Checkpoint(path, frozen) as record:
        for row in rows:
            record.add(row)
        record.finish()
    return path


def _rows() -> list[dict[str, object]]:
    rows = [_row("baseline")] + [_row(g.name) for g in GROUPS]
    rows[1] = _row(GROUPS[0].name, **{"model/a/nvcc": "refused"})
    rows[2] = _row(GROUPS[1].name, **{"model/a/nvcc": "timeout", "model/b/nvcc": "refused"})
    return rows


def _export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: Path) -> dict[str, str]:
    monkeypatch.setattr(rq2, "formal_dir", lambda: tmp_path / "library")
    with output_to(tmp_path / "out"):
        return rq2.export_ablation(KERNELS, path, tmp_path / "registration.json")


def test_ablation_counts_the_acceptances_that_each_group_loses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    numbers = _export(tmp_path, monkeypatch, _record(tmp_path, _rows()))
    with (tmp_path / "out/data/ablation.csv").open() as handle:
        rows = {r["group"]: r for r in csv.DictReader(handle)}
    assert list(rows) == [g.name for g in GROUPS]
    first, second = rows[GROUPS[0].name], rows[GROUPS[1].name]
    assert [first[c] for c in ("refused", "timeout", "lost", "development", "test")] == [
        "1",
        "0",
        "1",
        "1",
        "0",
    ]
    assert [second[c] for c in ("refused", "timeout", "lost", "development", "test")] == [
        "1",
        "1",
        "2",
        "1",
        "1",
    ]
    assert rows[GROUPS[2].name]["lost"] == "0"
    assert int(first["rules"]) == len(GROUPS[0].rules) + len(GROUPS[0].subsumed)
    assert numbers["valAblationSafe"] == "2"
    assert numbers["valAblationDefects"] == "1"
    assert numbers["valAblationNeeded"] == "2"
    assert numbers["valAblationNeededNone"] == "0"
    assert numbers["valAblationLostTermination"] == "1"
    assert numbers["valAblationTimeout"] == "360"
    assert all(name.isalpha() for name in numbers)


@pytest.mark.parametrize(
    "change",
    ["incomplete", "defect", "baseline", "group", "entries", "decision", "registration", "library"],
)
def test_ablation_report_rejects_an_inconsistent_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    rows = _rows()
    provenance: dict[str, object] = {}
    match change:
        case "defect":
            rows[3] = _row(GROUPS[2].name, **{"model/d/nvcc": "accepted"})
        case "baseline":
            rows[0] = _row("baseline", **{"model/b/nvcc": "timeout"})
        case "group":
            rows.pop()
        case "entries":
            provenance["safe"] = SAFE[:1]
        case "decision":
            rows[3] = _row(GROUPS[2].name, **{"model/b/nvcc": "unknown"})
    path = _record(tmp_path, rows, **provenance)
    match change:
        case "incomplete":
            data = json.loads(path.read_text())
            path.write_text(json.dumps(data | {"complete": False}))
        case "registration":
            (tmp_path / "registration.json").write_text("other registration")
        case "library":
            (tmp_path / "library" / "GateParse.v").write_text("other proof source")
    with pytest.raises(ValueError):
        _export(tmp_path, monkeypatch, path)
