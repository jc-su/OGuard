"""TPC-H pipelines run through the executor and give NumPy's results.

On TPC-H columns at a small scale, Q6 as one corpus launch, the revenue of
the small-quantity lines on the static path and on the guarded path, each
eagerly and as a graph, and the revenue of the lines of early orders through
pylibcudf's inner join and guarded gathers each equal the NumPy reference.
Every launch is one that the catalog checked.
"""

from collections.abc import Iterator
from datetime import date

import numpy as np
import pytest

rmm = pytest.importorskip("rmm")
pytest.importorskip("pylibcudf")

from eval.tpch.data import Lineitem, Orders, day, generate  # noqa: E402
from eval.tpch.operators import Operators, register  # noqa: E402
from eval.tpch.pipelines import (  # noqa: E402
    Q6,
    THRESHOLD,
    Path,
    Session,
    join_revenue,
    load,
    q6,
    revenue,
    unload,
)
from opgate.catalog import TenantId  # noqa: E402
from opgate.executor import Executor  # noqa: E402
from opgate.hosts.rapids import CatalogResource  # noqa: E402

ALICE = TenantId(1)
SCALE = 0.002
QUERY = Q6(day(date(1994, 1, 1)), day(date(1995, 1, 1)), 0.05, 0.07, 24.0)


@pytest.fixture(scope="module")
def tables() -> tuple[Orders, Lineitem]:
    return generate(SCALE, seed=11)


@pytest.fixture
def executor() -> Iterator[Executor]:
    with Executor(0, 1 << 26) as executor:
        with executor.tenant(ALICE):
            executor.reserve(1 << 26)
        yield executor


def _revenue(lineitem: Lineitem, mask: np.ndarray) -> float:
    return float((lineitem.extendedprice * (1 - lineitem.discount))[mask].sum())


@pytest.mark.gpu
def test_q6_and_the_revenue_pipelines_give_numpys_results(
    executor: Executor, tables: tuple[Orders, Lineitem]
) -> None:
    _, lineitem = tables
    s = Session(executor, ALICE, executor)
    ops: Operators = register(executor)
    columns = load(s, lineitem)
    selected = (
        (lineitem.shipdate >= QUERY.date_lo)
        & (lineitem.shipdate < QUERY.date_hi)
        & (lineitem.discount >= QUERY.disc_lo)
        & (lineitem.discount <= QUERY.disc_hi)
        & (lineitem.quantity < QUERY.qty_max)
    )
    expected = float((lineitem.extendedprice * lineitem.discount)[selected].sum())
    assert np.isclose(q6(s, ops, columns, QUERY), expected, rtol=1e-9)
    small = _revenue(lineitem, lineitem.quantity < THRESHOLD)
    for path in Path:
        for graph in (False, True):
            got = revenue(s, ops, columns, path, THRESHOLD, graph)
            assert np.isclose(got, small, rtol=1e-9), (path, graph, got, small)
    unload(s, columns)
    assert executor.catalog.buffers == ()


@pytest.mark.gpu
def test_the_join_pipeline_gives_numpys_result(tables: tuple[Orders, Lineitem]) -> None:
    orders, lineitem = tables
    before = day(date(1995, 3, 15))
    early = orders.orderkey[orders.orderdate < before]
    expected = _revenue(lineitem, np.isin(lineitem.orderkey, early))
    with (
        Executor(0, 1 << 26) as executor,
        CatalogResource(executor, chunk_bytes=1 << 22) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 26)
        with resource.current():
            s = Session(executor, ALICE, resource)
            ops = register(executor)
            columns = load(s, lineitem)
            got = join_revenue(s, ops, columns, orders, before, resource)
            unload(s, columns)
    assert np.isclose(got, expected, rtol=1e-9)
