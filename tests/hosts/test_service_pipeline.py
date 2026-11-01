"""A safe multi-operator service pipeline also feeds a real cuDF operation."""

from pathlib import Path

import pytest

from eval.rq3_pipelines.service_pipeline import run_pipeline


@pytest.mark.gpu
def test_declarative_index_pipeline_feeds_cudf(tmp_path: Path):
    result = run_pipeline(tmp_path / "service.sock", rows=257)
    assert result["rows"] == 257
    audit = result["audit"]
    assert isinstance(audit, list) and len(audit) == 2
