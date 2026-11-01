"""The service-boundary experiment replays archived PTX text and fails on incomplete archives."""

import json
from pathlib import Path

import pytest

from eval.rq1_safety.service_boundary import DEFAULT_ARCHIVE, MODULES, archived_modules


def archive(rows: dict[str, object]) -> dict[str, object]:
    return {"success": True, "registrations": {"e7_audit_rows": rows}}


def test_metadata_only_replay_fails_before_starting_the_service(tmp_path: Path) -> None:
    record = tmp_path / "historical.json"
    record.write_text(json.dumps(archive({name: {} for name in MODULES})))
    with pytest.raises(RuntimeError, match="has no ASCII PTX text"):
        archived_modules(record)


def test_replay_preserves_the_exact_archived_text(tmp_path: Path) -> None:
    texts = {name: f"// archived {name}\r\n\t\n" for name in MODULES}
    record = tmp_path / "archived.json"
    record.write_text(json.dumps(archive({name: {"ptx": text} for name, text in texts.items()})))
    assert archived_modules(record) == texts


def test_archived_fixture_contains_the_submitted_text(tmp_path, raw_record) -> None:
    record = raw_record(DEFAULT_ARCHIVE.name)
    rows = record["registrations"]["e7_audit_rows"]
    path = tmp_path / DEFAULT_ARCHIVE.name
    path.write_text(json.dumps(record))
    assert archived_modules(path) == {name: row["ptx"] for name, row in rows.items()}


@pytest.mark.parametrize("ptx", [None, "", "// non-ASCII \N{SNOWMAN}"])
def test_invalid_archived_text_fails(ptx: str | None, tmp_path: Path) -> None:
    record = tmp_path / "invalid.json"
    record.write_text(json.dumps(archive({name: {"ptx": ptx} for name in MODULES})))
    with pytest.raises(RuntimeError, match="has no ASCII PTX text"):
        archived_modules(record)


@pytest.mark.parametrize(
    "data,reason",
    [
        ({"success": False}, "not successful"),
        ({"success": True}, "has no records of the modules"),
        (archive({"safe": {"ptx": "// safe"}}), "has no records of the modules"),
    ],
)
def test_incomplete_archive_fails(data: dict[str, object], reason: str, tmp_path: Path) -> None:
    record = tmp_path / "incomplete.json"
    record.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match=reason):
        archived_modules(record)
