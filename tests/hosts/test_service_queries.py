"""Strict table serialization and a complete mixed generated/engine query."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.corpus.adapter import binding_of
from eval.rq3_pipelines.service_q3 import PROJECTION, cases, decode_result, run_queries
from eval.rq3_pipelines.service_queries import Q3_OUTPUT, REVENUE, result_columns, utf8_values
from opgate.hosts.service import Fields, RequestError


def test_projection_contract_is_the_unchanged_archived_contract():
    ptx = PROJECTION.read_bytes().decode("ascii")
    plan = REVENUE.plan(ptx, Fields({"kind": REVENUE.kind}, "plan"))
    assert plan.binding == binding_of("sonnet", "projection", "numba", ptx)


@pytest.mark.parametrize("values,rows", [([None], 1), ([12], 1), (["a"], 2), (["\ud800"], 1)])
def test_string_column_rejects_invalid_values(values, rows):
    with pytest.raises(RequestError):
        utf8_values(Fields({"segment": values}, "columns"), "segment", rows)


def test_string_limit_counts_actual_utf8_bytes(monkeypatch):
    monkeypatch.setattr("eval.rq3_pipelines.service_queries.MAX_STRING_BYTES", 3)
    assert utf8_values(Fields({"s": ["é", "a"]}, "columns"), "s", 2) == ["é", "a"]
    with pytest.raises(RequestError, match="byte limit"):
        utf8_values(Fields({"s": ["é", "é"]}, "columns"), "s", 2)


def test_binary_result_preserves_numeric_bytes():
    frame = pd.DataFrame(
        {
            name: np.array([value], dtype=dtype)
            for (name, dtype), value in zip(Q3_OUTPUT, [123, -0.0, 9200, 0], strict=True)
        }
    )
    output = result_columns(frame, Q3_OUTPUT)
    decoded = decode_result(Fields(output, "result"))
    for name, _ in Q3_OUTPUT:
        assert decoded[name].to_numpy().tobytes() == frame[name].to_numpy().tobytes()
    output["rows"] = 2
    with pytest.raises(AssertionError, match="length"):
        decode_result(Fields(output, "result"))


def test_fixture_covers_join_multiplicity_date_edges_and_top10():
    from eval.tpch.queries import host, q3

    inputs = cases(include_generated=False)
    actual = host(q3(pd, inputs["date_boundaries_duplicates_ties_top10"]))
    assert actual["l_orderkey"].tolist() == list(range(1, 11))
    assert actual["revenue"].tolist() == [64.0] * 10
    assert host(q3(pd, inputs["empty_join"])).empty
    assert host(q3(pd, inputs["empty_customer_table"])).empty


@pytest.mark.gpu
def test_complete_q3_uses_generated_projection_and_trusted_engine(tmp_path: Path):
    result = run_queries(tmp_path / "q3.sock", include_generated=False, repeats=1)
    assert result["generated_launches"] == 2
    assert len(result["cases"]) == 3
    assert all(
        sample["reference_bytes_equal"] for case in result["cases"] for sample in case["samples"]
    )
