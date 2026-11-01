"""Proof-worker bounds and durable input publication before verification."""

import json
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest

from eval.rq4_cost import registration
from opgate.binding import Binding
from opgate.gate import CHECK_QUERY, REGISTRATION_SECONDS, export_query, export_terms, zeroed
from opgate.signature import Affine


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_fails_before_creating_a_record(tmp_path, workers):
    output = tmp_path / "record.json"
    with pytest.raises(ValueError, match="positive"):
        registration.measure(output, workers=workers)
    assert not output.exists()


def test_insufficient_cores_fail_before_creating_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(registration.os, "sched_getaffinity", lambda _: {0})
    output = tmp_path / "record.json"
    with pytest.raises(ValueError, match="2 workers need at least 2 cores"):
        registration.measure(output, workers=2)
    assert not output.exists()


@pytest.fixture
def archive_inputs(monkeypatch):
    binding = Binding("kernel", (), (), (1, 1, 1), (1, 1, 1), Affine(0))
    items = [
        registration.Input(
            ("model", f"task{i}", "nvcc"),
            f".version 8.0\n//{i}\n.visible .entry kernel()\n{{\nret;\n}}\n",
            binding,
            "safe",
            None,
        )
        for i in range(4)
    ]
    monkeypatch.setattr(registration.os, "sched_getaffinity", lambda _: set(range(4)))
    monkeypatch.setattr(registration, "inputs", lambda: items)
    monkeypatch.setattr(
        registration, "source_files", lambda _: {"registration.py": Path(registration.__file__)}
    )
    return items


def test_parallel_publication_is_complete_before_any_verification(
    tmp_path, monkeypatch, archive_inputs
):
    path = tmp_path / "run.json"
    barrier = Barrier(2)
    archive = registration._archive

    def archive_pair(record, item):
        barrier.wait(timeout=5)
        archive(record, item)

    monkeypatch.setattr(registration, "_archive", archive_pair)

    def measured(item, budget):
        assert budget == REGISTRATION_SECONDS
        for expected in archive_inputs:
            key = "/".join(expected.key)
            folder = path.with_suffix(".json.artifacts")
            assert (folder / (key + ".ptx")).read_bytes() == expected.ptx.encode("ascii")
            assert (folder / (key + ".v")).read_bytes() == export_terms(
                zeroed(expected.ptx, expected.binding), [expected.binding]
            ).encode("ascii")
        return {"id": "/".join(item.key), "accepted": True}

    monkeypatch.setattr(registration, "_measured", measured)
    registration.measure(path, workers=2)
    record = json.loads(path.read_text())
    assert record["complete"] and len(record["results"]) == 4
    assert record["provenance"]["query_suffix"] == CHECK_QUERY
    assert record["provenance"]["processes_per_verification"] == 1
    assert record["provenance"]["registration_seconds"] == REGISTRATION_SECONDS
    for item in archive_inputs:
        assert export_terms(item.ptx, [item.binding]) + record["provenance"][
            "query_suffix"
        ] == export_query(item.ptx, [item.binding])


def test_publication_failure_starts_no_verification_and_can_resume(
    tmp_path, monkeypatch, archive_inputs
):
    path = tmp_path / "run.json"
    archive = registration._archive
    measured = Mock()

    def fail(record, item):
        if item == archive_inputs[1]:
            raise OSError("publication failed")
        archive(record, item)

    monkeypatch.setattr(registration, "_archive", fail)
    monkeypatch.setattr(registration, "_measured", measured)
    with pytest.raises(OSError, match="publication failed"):
        registration.measure(path, workers=2)
    measured.assert_not_called()
    assert not json.loads(path.read_text())["complete"]
    monkeypatch.setattr(registration, "_archive", archive)
    monkeypatch.setattr(
        registration, "_measured", lambda item, budget: {"id": "/".join(item.key), "accepted": True}
    )
    registration.measure(path, workers=2, resume=True)
    assert json.loads(path.read_text())["complete"]


def test_changed_raw_query_fails_before_resume_verification(tmp_path, monkeypatch, archive_inputs):
    path = tmp_path / "run.json"
    monkeypatch.setattr(
        registration, "_measured", lambda item, budget: {"id": "/".join(item.key), "accepted": True}
    )
    registration.measure(path, workers=2)
    artifact = path.with_suffix(".json.artifacts") / "model/task2/nvcc.v"
    artifact.write_bytes(b"changed query")
    measured = Mock()
    monkeypatch.setattr(registration, "_measured", measured)
    with pytest.raises(ValueError, match="different contents"):
        registration.measure(path, workers=2, resume=True)
    measured.assert_not_called()
    assert artifact.read_bytes() == b"changed query"
