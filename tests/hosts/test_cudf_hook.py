"""cuDF's `apply` UDFs run through the gate and the executor.

Inside `install_cudf`, with the catalog's RMM resource current, a
`Series.apply` UDF on a column and on a slice of it, and a `DataFrame.apply`
UDF, give cuDF's results, and each UDF's kernel is an operator of the
catalog, which the executor verified before its first launch. A TPC-H
revenue query, whose filter and sum are cuDF's own kernels and whose revenue
per row is a UDF, gives NumPy's result. A UDF over a column with a null mask
or a string column is refused, and so are a group-by UDF, whose kernel calls
cuDF's shim, and a rolling UDF, an aggregation that libcudf compiles into a
kernel of its own.
"""

import gc

import numpy as np
import pytest

cudf = pytest.importorskip("cudf")
rmm = pytest.importorskip("rmm")

from eval.tpch.data import generate  # noqa: E402
from opgate.catalog import TenantId  # noqa: E402
from opgate.executor import Executor, RefusedError  # noqa: E402
from opgate.frontends import Target  # noqa: E402
from opgate.hosts.cudf import install_cudf  # noqa: E402
from opgate.hosts.rapids import CatalogResource  # noqa: E402

ALICE = TenantId(1)


@pytest.mark.gpu
@pytest.mark.parametrize("extra", [np.int32(-7), np.uint32(7), np.float32(1.5), np.float64(-2.5)])
def test_extra_scalar_arguments_preserve_their_compiled_type(extra) -> None:
    with (
        Executor(0, 1 << 26) as executor,
        CatalogResource(executor, chunk_bytes=1 << 22) as resource,
    ):
        with resource.current():
            with executor.tenant(ALICE), install_cudf(executor, resource, Target(8, 6)):
                executor.reserve(1 << 26)
                frame = cudf.DataFrame({"value": np.arange(32, dtype=np.float64)})
                result = frame.apply(
                    lambda row, amount: row["value"] + amount, axis=1, args=(extra,)
                )
                assert np.allclose(result.to_numpy(), np.arange(32) + extra)
                del result, frame
                gc.collect()


@pytest.mark.gpu
def test_apply_udfs_run_through_the_executor() -> None:
    with Executor(0, 1 << 28) as executor, CatalogResource(executor) as resource:
        with resource.current():
            with executor.tenant(ALICE), install_cudf(executor, resource, Target(8, 6)):
                executor.reserve(1 << 28)
                values = cudf.Series(np.arange(1000, dtype=np.int64))
                doubled = values.apply(lambda x: 2 * x + 1)
                assert np.array_equal(doubled.to_numpy(), 2 * np.arange(1000) + 1)
                tail = values.iloc[100:].apply(lambda x: x - 1)
                assert np.array_equal(tail.to_numpy(), np.arange(100, 1000) - 1)
                assert len(executor.catalog.operators) == 2

                _, lineitem = generate(0.002, seed=7)
                frame = cudf.DataFrame(
                    {
                        "quantity": lineitem.quantity,
                        "price": lineitem.extendedprice,
                        "discount": lineitem.discount,
                    }
                )
                small = frame[frame["quantity"] < 24]
                revenue = small.apply(lambda row: row["price"] * (1 - row["discount"]), axis=1)
                selected = lineitem.quantity < 24
                expected = lineitem.extendedprice[selected] * (1 - lineitem.discount[selected])
                assert np.allclose(revenue.to_numpy(), expected, rtol=1e-12)
                assert np.isclose(float(revenue.sum()), float(expected.sum()), rtol=1e-9)
                assert len(executor.catalog.operators) == 3

                nullable = cudf.Series([1, None, 3], dtype="int64")
                with pytest.raises(ValueError) as caught:
                    nullable.apply(lambda x: x + 1)
                assert isinstance(caught.value.__cause__, RefusedError)
                assert "null mask" in caught.value.__cause__.reason

                strings = cudf.Series(["", "abc", "some_example"])
                with pytest.raises(ValueError) as caught:
                    strings.apply(lambda s: len(s))
                assert isinstance(caught.value.__cause__, RefusedError)
                assert "does not bind" in caught.value.__cause__.reason

                groups = cudf.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]}).groupby("a")
                with pytest.raises(RefusedError) as refused:
                    groups.apply(lambda group: group["b"].max(), engine="jit")
                assert "groupby_apply UDF" in refused.value.reason

                def peak(window: np.ndarray) -> float:
                    return max(window)

                with pytest.raises(RefusedError) as refused:
                    values.rolling(3).apply(peak)
                assert "UDF aggregation" in refused.value.reason
                # a refusal's traceback holds cuDF's frames and their columns;
                # free them while the tenant's session lasts
                del caught, refused
                del values, doubled, tail, frame, small, revenue, nullable, strings, groups
                gc.collect()
            assert all(b.owner == ALICE for b in executor.catalog.buffers)
