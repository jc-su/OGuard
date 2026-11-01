"""TPC-H Q1, Q6, and Q14 on PyTorch give pandas' results with each kind of operator.

At scale factor 0.01, each query of `eval.tpch.torch_engine` runs with
PyTorch's own operators, with the Triton kernels that the engine's `MODELS`
names launched directly, and with the same kernels through the gate and the
executor, on tensors in one tenant's segments; each result equals the pandas
query of `eval.tpch.queries`. Q1 sums with Opus's `scatter_add`, and Q6 and
Q14 use Fable's kernels. Through the executor, every catalog buffer is a
segment of the tenant or the buffer of an index input, the gather's row ids
or the `scatter_add`'s group numbers, and after the session no buffer is
live.
"""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cudf")

from eval.rq3_pipelines.tpch_queries import SEED  # noqa: E402
from eval.tpch.database import generate  # noqa: E402
from eval.tpch.queries import QUERIES as PANDAS  # noqa: E402
from eval.tpch.queries import host, mismatch  # noqa: E402
from eval.tpch.torch_engine import (  # noqa: E402
    QUERIES,
    DirectOperators,
    GatedOperators,
    TorchOperators,
    load,
)
from opgate.catalog import TenantId  # noqa: E402
from opgate.executor import Executor  # noqa: E402
from opgate.hosts.torch import TorchHost  # noqa: E402

SCALE = 0.01
TENANT = TenantId(1)


@pytest.mark.gpu
@pytest.mark.slow
def test_the_queries_on_pytorch_give_pandas_results() -> None:
    database = generate(SCALE, SEED)
    expected = {name: host(PANDAS[name](pd, database)) for name in QUERIES}
    tables = load(database)
    for ops in (TorchOperators(), DirectOperators()):
        for name, query in QUERIES.items():
            assert mismatch(expected[name], query(ops, tables)) is None, (type(ops), name)
    del tables
    with Executor(0, 1 << 28) as executor, TorchHost(executor) as torch_host:
        with executor.tenant(TENANT):
            executor.reserve(1 << 28)
            modules = GatedOperators.register(executor)
        with torch_host.session(TENANT):
            tables = load(database)
            ops = GatedOperators(torch_host, modules)
            for name, query in QUERIES.items():
                assert mismatch(expected[name], query(ops, tables)) is None, name
                assert {b.id for b in executor.catalog.buffers} == {
                    *torch_host.segments(TENANT),
                    *torch_host.buffers(TENANT),
                }
            del tables, ops
        torch_host.release(TENANT)
        assert not executor.catalog.buffers
