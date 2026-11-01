"""Guarded coverage preserves exact checker inputs and fails closed on stale evidence."""

import json
from unittest.mock import Mock

import pytest

from eval.checkpoint import Checkpoint, read_artifact
from eval.rq2_coverage import guarded
from opgate.binding import GRID_LIMIT, Binding
from opgate.frontends.guard import UNPROVED as UNPROVED_PREFIXES
from opgate.frontends.guard import Guarded
from opgate.gate import REGISTRATION_SECONDS, Accepted, Refused, export_terms
from opgate.hosts.service.audit import binding_record
from opgate.signature import Affine

BINDING = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
TEXT = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry kernel() { ret; }\n"


def registration(path, keys=("model/a/nvcc", "model/b/nvcc")):
    with Checkpoint(path, {}) as record:
        for index, key in enumerate(keys):
            record.artifact(key, TEXT, "ptx")
            record.artifact(key, export_terms(TEXT, (BINDING,)), "v")
            record.add(
                {
                    "id": key,
                    "accepted": index == 0,
                    "verdict": "safe",
                    "reason": "" if index == 0 else "unproved",
                    "binding": binding_record(BINDING),
                    "ptx_artifact": key,
                    "query_artifact": key,
                }
            )
        record.finish()


UNPROVED = Refused(f"kernel: {UNPROVED_PREFIXES[0]}ld.global.u32 %r1, [%rd1]", ())


def measured(tmp_path, monkeypatch, first=UNPROVED):
    """A replay of two entries: the first accepted as written, the second refused
    as written with `first` and, when the pass addresses that refusal,
    accepted after one rewrite."""
    source = tmp_path / "source.v"
    source.write_text("raw proof source\n")
    monkeypatch.setattr(guarded, "source_files", lambda: {"source.v": source})
    original = tmp_path / "registration.json"
    registration(original)

    def transform(text, binding, check):
        # the pass's first check repeats the check of the text as written
        assert check(text, binding) == first
        rewritten = text + "// retained rewrite\r\n"
        return Guarded(rewritten, binding, (), (), True, check(rewritten, binding))

    monkeypatch.setattr(guarded, "verify", Mock(side_effect=[Accepted(), first, first, Accepted()]))
    monkeypatch.setattr(guarded, "guard", transform)
    path = tmp_path / "guarded.json"
    guarded.measure(original, path, every=True, workers=1)
    return path


def test_the_pass_runs_only_for_a_refusal_that_it_addresses(tmp_path, monkeypatch):
    path = measured(tmp_path, monkeypatch)
    record = guarded.validate_evidence(path)
    rows = {row["id"]: row for row in record["results"]}
    checked = rows["model/b/nvcc"]
    assert checked["first_reason"] == UNPROVED.reason and checked["decision"] == "accepted"
    assert checked["checks"] == 2  # the repeat of the first check is the cache's answer
    assert record["provenance"]["registration_seconds"] == REGISTRATION_SECONDS
    other = tmp_path / "other"
    other.mkdir()
    path = measured(other, monkeypatch, first=Refused("kernel: unmodeled instruction effects", ()))
    refused = {row["id"]: row for row in guarded.validate_evidence(path)["results"]}
    assert refused["model/b/nvcc"]["decision"] == "refused"
    assert refused["model/b/nvcc"]["reason"] == "kernel: unmodeled instruction effects"
    assert refused["model/b/nvcc"]["checks"] == 1
    assert read_artifact(path, "model/b/nvcc/final", "ptx") == TEXT


def test_unchanged_and_rewrite_only_inputs_are_retained(tmp_path, monkeypatch):
    path = measured(tmp_path, monkeypatch)
    record = guarded.validate_evidence(path)
    assert len(record["results"]) == 2
    assert read_artifact(path, "model/a/nvcc/final", "ptx") == TEXT
    assert read_artifact(path, "model/b/nvcc/final", "ptx") == TEXT + "// retained rewrite\r\n"
    assert all(row["binding"] == binding_record(BINDING) for row in record["results"])


@pytest.mark.parametrize("change", ["source", "missing", "text", "binding", "cohort", "incomplete"])
def test_changed_or_incomplete_evidence_is_rejected(tmp_path, monkeypatch, change):
    path = measured(tmp_path, monkeypatch)
    record = json.loads(path.read_text())
    final = path.with_suffix(".json.artifacts") / "model/b/nvcc/final.ptx"
    if change == "source":
        (tmp_path / "source.v").write_text("changed proof")
    elif change == "missing":
        final.unlink()
    elif change == "text":
        final.write_text(TEXT)
    elif change == "binding":
        record["results"][1]["binding"]["block"] = [2, 1, 1]
    elif change == "cohort":
        record["results"].pop()
    else:
        record["complete"] = False
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        guarded.validate_evidence(path)


def test_completed_resume_performs_no_new_checks(tmp_path, monkeypatch):
    path = measured(tmp_path, monkeypatch)
    verify = Mock(side_effect=AssertionError("already complete"))
    monkeypatch.setattr(guarded, "verify", verify)
    guarded.measure(tmp_path / "registration.json", path, every=True, workers=1, resume=True)
    verify.assert_not_called()


def test_binding_is_taken_from_the_archived_registration(tmp_path, monkeypatch):
    path = measured(tmp_path, monkeypatch)
    original = json.loads((tmp_path / "registration.json").read_text())
    original["results"][0]["binding"]["block"] = [2, 1, 1]
    (tmp_path / "registration.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="query differs"):
        guarded.measure(tmp_path / "registration.json", path, every=True, workers=1, resume=True)
