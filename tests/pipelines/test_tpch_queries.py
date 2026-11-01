"""The 22 TPC-H queries give pandas' results on cuDF with every allocation through the catalog.

At scale factor 0.01, each query of `eval/tpch/queries.py` runs on cuDF inside
`install_cudf`, in a session of one tenant, with `CatalogResource` as RMM's
current resource and the tables loaded into the executor's pool in that
session (`eval.rq3_pipelines.tpch_queries.in_catalog`). Each result equals the pandas
reference and has rows, every live buffer is a chunk of the tenant's arena,
the queries after the first take few steps, and after the tables are freed
no catalog buffer is live, which `in_catalog` checks.
"""

import pandas as pd
import pytest

cudf = pytest.importorskip("cudf")
pytest.importorskip("rmm")

from eval.rq3_pipelines.tpch_queries import (  # noqa: E402
    SEED,
    TENANT,
    CountingExecutor,
    in_catalog,
)
from eval.tpch.database import Tables, generate  # noqa: E402
from eval.tpch.queries import QUERIES, Frame, host, mismatch  # noqa: E402
from opgate.hosts.rapids import CatalogResource  # noqa: E402

SCALE = 0.01


@pytest.mark.gpu
@pytest.mark.slow
def test_the_22_queries_through_the_catalog_give_pandas_results() -> None:
    database = generate(SCALE, SEED)
    expected = {name: host(query(pd, database)) for name, query in QUERIES.items()}
    assert [name for name, result in expected.items() if result.empty] == []

    def run(
        executor: CountingExecutor, resource: CatalogResource, tables: Tables[Frame]
    ) -> dict[str, str | None]:
        differences: dict[str, str | None] = {}
        steps = []
        for name, query in QUERIES.items():
            with executor.counting() as counts:
                actual = host(query(cudf, tables))
            steps.append(counts.steps)
            chunks = set(resource.chunks(TENANT))
            assert {buffer.id for buffer in executor.catalog.buffers} == chunks, name
            assert all(buffer.owner == TENANT for buffer in executor.catalog.buffers), name
            differences[name] = mismatch(expected[name], actual)
        # the arena grows by few chunks in all; the tables' chunks came before
        assert sum(steps) <= 8, steps
        return differences

    differences = in_catalog(database, run)
    assert {name: d for name, d in differences.items() if d is not None} == {}
