"""Interrupted measurements cannot silently become complete evaluation tables."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eval.checkpoint import Checkpoint, read_artifact, snapshot_inputs
from eval.corpus import repair


def test_resume_preserves_results_and_requires_exact_inputs(tmp_path):
    path = tmp_path / "run.json"
    with Checkpoint(path, {"source": "v1"}) as record:
        record.add({"id": "first", "decision": "refused"})
    with Checkpoint(path, {"source": "v1"}, resume=True) as record:
        assert record.get("first")["decision"] == "refused"
        assert not record.data["complete"]
        with pytest.raises(ValueError, match="duplicate"):
            record.add({"id": "first"})
        record.finish()
    with pytest.raises(ValueError, match="inputs or settings changed"):
        Checkpoint(path, {"source": "v2"}, resume=True)
    with pytest.raises(FileExistsError):
        Checkpoint(path, {"source": "v1"})


def test_only_one_writer_can_open_a_record(tmp_path):
    path = tmp_path / "run.json"
    with Checkpoint(path, {}) as record:
        with pytest.raises(BlockingIOError):
            Checkpoint(path, {}, resume=True)
        record.add({"id": "first"})
    with Checkpoint(path, {}, resume=True) as record:
        assert record.get("first") is not None


def test_failed_publication_leaves_the_previous_json_intact(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    with Checkpoint(path, {}) as record:
        record.add({"id": "first"})
        monkeypatch.setattr("eval.checkpoint.os.replace", Mock(side_effect=OSError("disk")))
        with pytest.raises(OSError, match="disk"):
            record.add({"id": "second"})
    assert [r["id"] for r in json.loads(path.read_text())["results"]] == ["first"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json", "run.json.lock"]


def test_completed_results_survive_another_worker_failure_and_resume_skips_them(
    tmp_path, monkeypatch
):
    found = [repair.Revision("m", task, "nvcc", 0) for task in ("slow", "fast")]
    table = {r.key: SimpleNamespace(verdict="safe") for r in found}
    monkeypatch.setattr(repair, "read_table", lambda: table)
    compile_kernel = Mock(
        side_effect=lambda r: SimpleNamespace(text=r.task, binding=lambda _: None)
    )
    monkeypatch.setattr(repair, "compiled", compile_kernel)
    monkeypatch.setattr(repair, "claim_of", lambda *args: (None, None))
    monkeypatch.setattr(repair, "export_terms", lambda text, bindings: text)
    monkeypatch.setattr(repair, "zeroed", lambda text, binding: text)
    saved = Event()

    def measure(r, operator, timeout):
        if r.task == "slow":
            assert saved.wait(5)
            raise RuntimeError("worker failed")
        return repair.Outcome("refused", "unproved", "-"), 0.1

    monkeypatch.setattr(repair, "measure", measure)
    path = tmp_path / "run.json"
    with Checkpoint(path, {}) as record:
        add = record.add

        def save(row):
            add(row)
            saved.set()

        monkeypatch.setattr(record, "add", save)
        with pytest.raises(RuntimeError, match="worker failed"):
            repair.evaluate(found, Mock(), record, timeout=1, workers=2)
    assert [r["task"] for r in json.loads(path.read_text())["results"]] == ["fast"]
    compile_kernel.reset_mock()
    monkeypatch.setattr(
        repair, "measure", lambda *args: (repair.Outcome("refused", "unproved", "-"), 0.1)
    )
    with Checkpoint(path, {}, resume=True) as record:
        repair.evaluate(found, Mock(), record, timeout=1, workers=2)
        assert len(record.data["results"]) == 2
    compile_kernel.assert_called_once_with(found[0])


def test_known_defect_acceptance_stops_before_device_execution(tmp_path, monkeypatch):
    revision = repair.Revision("m", "task", "nvcc", 0)
    operator = SimpleNamespace(text="test", binding=lambda _: None)
    monkeypatch.setattr(repair, "claim_of", lambda *args: (None, None))
    monkeypatch.setattr(repair, "export_terms", lambda text, bindings: text)
    monkeypatch.setattr(repair, "zeroed", lambda text, binding: text)
    monkeypatch.setattr(repair, "read_table", lambda: {revision.key: SimpleNamespace(verdict="C")})
    monkeypatch.setattr(repair, "compiled", lambda _: operator)
    monkeypatch.setattr(repair, "measure", lambda *args: (operator, 0.1))
    execute = Mock()
    monkeypatch.setattr(repair, "result", execute)
    with Checkpoint(tmp_path / "run.json", {}) as record:
        with pytest.raises(RuntimeError, match="known defect was accepted"):
            repair.evaluate([revision], Mock(), record, timeout=1, workers=1)
        assert record.data["results"][0]["result"] == "not_run"
        assert not record.data["complete"]
    execute.assert_not_called()


def test_aggregate_rejects_partial_records():
    with pytest.raises(ValueError, match="incomplete"):
        repair.validate_record({"schema": 2, "complete": False})


def test_round_zero_uses_the_original_source_directory():
    revision = repair.Revision("fable", "sorted_contains", "nvcc", 0)
    assert revision.directory == repair.CORPUS / "fable" / "sorted_contains"


@pytest.mark.parametrize("model", ["model", "gpt-5.5"])
def test_archived_artifacts_are_named_and_never_silently_overwritten(tmp_path, model):
    path = tmp_path / "run.json"
    key = f"{model}/task/nvcc/0"
    with Checkpoint(path, {}) as record:
        record.artifact(key, "exact terms", "v")
        record.add({"id": key, "query_artifact": key})
    assert read_artifact(path, key, "v") == "exact terms"
    with Checkpoint(path, {}, resume=True) as record:
        assert record.artifact(key, "exact terms", "v") == key
        with pytest.raises(ValueError, match="different contents"):
            record.artifact(key, "different terms", "v")
    with pytest.raises(ValueError, match="invalid artifact"):
        read_artifact(path, "../outside", "v")


def test_missing_artifact_cannot_be_reported_as_a_valid_completed_run(tmp_path):
    path = tmp_path / "run.json"
    with Checkpoint(path, {}) as record:
        ptx = record.artifact("first", "ptx", "ptx")
        query = record.artifact("first", "query", "v")
        record.add(
            {"id": "first", "decision": "accepted", "ptx_artifact": ptx, "query_artifact": query}
        )
        record.finish()
    repair.validate_artifacts(path, record.data)
    (path.with_suffix(".json.artifacts") / f"{query}.v").unlink()
    with pytest.raises(FileNotFoundError):
        repair.validate_artifacts(path, record.data)


def test_resume_compares_archived_input_bytes_directly(tmp_path):
    source = tmp_path / "roles.tsv"
    source.write_text("original roles")
    record = tmp_path / "run.json"
    inputs = {"roles.tsv": source}
    snapshot_inputs(record, "sources", inputs, create=True)
    snapshot_inputs(record, "sources", inputs, create=False)
    source.write_text("changed roles")
    with pytest.raises(ValueError, match="input changed"):
        snapshot_inputs(record, "sources", inputs, create=False)
    with pytest.raises(ValueError, match="inputs changed"):
        snapshot_inputs(record, "sources", {}, create=False)


def test_probe_fixture_and_roles_are_archived_as_inputs():
    files = repair.input_files([("haiku", "hash_probe", "numba")])
    assert "eval/corpus/synthesized/haiku/roles.tsv" in files
    assert "eval/corpus/synthesized/haiku/hash_build/numba.O1.ptx" in files


def test_measurement_inputs_do_not_recursively_include_previous_snapshots(tmp_path, monkeypatch):
    root = tmp_path
    module = root / "eval" / "corpus" / "repair.py"
    module.parent.mkdir(parents=True)
    module.write_text("current measurement code")
    archived = root / "eval" / "records" / "old.json.inputs" / "sources" / "stale.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("previous measurement code")
    monkeypatch.setattr(repair, "__file__", str(module))
    monkeypatch.setattr(repair, "CORPUS", root / "eval" / "corpus" / "synthesized")
    files = repair.input_files([])
    assert "eval/corpus/repair.py" in files
    assert not any(name.startswith("eval/records/") for name in files)
