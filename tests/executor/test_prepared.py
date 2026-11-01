"""Prepared steps on the tenants' streams give the eager path's results, and every
use is decided by the catalog."""

import struct
from pathlib import Path

import pytest

from eval.corpus.adapter import binding_of
from eval.tpch.data import generate
from eval.tpch.pipelines import Q6_PARAMETERS, Columns, Session, load
from opgate.binding import ElementType
from opgate.catalog import BufferId, Fill, Launch, Step, TenantId
from opgate.executor import Executor, Module, RefusedError, Run

ROOT = Path(__file__).resolve().parents[2]
PTX = ROOT / "eval/corpus/synthesized/sonnet/q6_sum/numba.O1.ptx"
DOUBLE = ElementType(8, False)
ALICE, BOB = TenantId(1), TenantId(2)


def _bits(value: float) -> int:
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


def _scalars(date_hi: int) -> tuple[int, ...]:
    q = Q6_PARAMETERS
    return (q.date_lo, date_hi, _bits(q.disc_lo), _bits(q.disc_hi), _bits(q.qty_max))


def _step(
    tenant: TenantId, module: Module, columns: Columns, result: BufferId, date_hi: int
) -> Step:
    buffers = (columns.shipdate, columns.discount, columns.quantity, columns.price, result)
    grid = (max(1, -(-columns.rows // 256)), 1, 1)
    launch = Launch(module.operator, grid, buffers, (columns.rows,), _scalars(date_hi), False)
    return Step(tenant, (Fill(result, DOUBLE, 0), launch))


def _value(data: bytes) -> float:
    (value,) = struct.unpack("<d", data)
    return value


@pytest.fixture(scope="module")
def lineitem():
    _, items = generate(0.01, 5)
    return items


def _reference(items, date_hi: int) -> float:
    q = Q6_PARAMETERS
    keep = (
        (items.shipdate >= q.date_lo)
        & (items.shipdate < date_hi)
        & (items.discount >= q.disc_lo)
        & (items.discount <= q.disc_hi)
        & (items.quantity < q.qty_max)
    )
    return float((items.extendedprice * items.discount)[keep].sum())


def test_prepared_uses_match_the_eager_path(lineitem):
    text = PTX.read_text()
    with Executor(0, 64 * 2**20) as executor:
        module = executor.register(text, binding_of("sonnet", "q6_sum", "numba", text))
        runs: dict[TenantId, Run] = {}
        eager: dict[TenantId, tuple[Columns, BufferId]] = {}
        for tenant in (ALICE, BOB):
            with executor.tenant(tenant):
                executor.reserve(32 * 2**20)
            columns = load(Session(executor, tenant, executor), lineitem)
            with executor.tenant(tenant):
                result = executor.allocate(8)
                runs[tenant] = executor.prepare(
                    _step(tenant, module, columns, result, Q6_PARAMETERS.date_hi), result
                )
                eager[tenant] = (columns, executor.allocate(8))
        for date_hi in (Q6_PARAMETERS.date_hi, Q6_PARAMETERS.date_lo + 90, Q6_PARAMETERS.date_lo):
            for tenant, run in runs.items():
                with executor.tenant(tenant):
                    executor.submit(run, _scalars(date_hi))
            for tenant, run in runs.items():
                while not executor.finished(run):
                    pass
                observed = _value(executor.result(run))
                columns, result = eager[tenant]
                with executor.tenant(tenant):
                    executor.execute(_step(tenant, module, columns, result, date_hi))
                    # the kernel adds with atomics, so the order of its additions varies
                    assert observed == pytest.approx(
                        _value(executor.read(tenant, result)), rel=1e-12
                    )
                assert observed == pytest.approx(_reference(lineitem, date_hi), rel=1e-9)


def test_each_use_is_decided_by_the_catalog(lineitem):
    text = PTX.read_text()
    with Executor(0, 64 * 2**20) as executor:
        module = executor.register(text, binding_of("sonnet", "q6_sum", "numba", text))
        with executor.tenant(ALICE):
            executor.reserve(32 * 2**20)
        columns = load(Session(executor, ALICE, executor), lineitem)
        with executor.tenant(ALICE):
            result = executor.allocate(8)
            run = executor.prepare(_step(ALICE, module, columns, result, 9000), result)
            with pytest.raises(RefusedError, match="a scalar lies outside its bounds"):
                executor.submit(run, (*_scalars(9000)[:4], 2**64))
            assert not run.running
            with pytest.raises(RefusedError, match="reads back a buffer it names"):
                executor.prepare(_step(ALICE, module, columns, result, 9000), columns.orderkey)
        with executor.tenant(BOB), pytest.raises(RefusedError, match="not the session's tenant"):
            executor.submit(run, _scalars(9000))
        with executor.tenant(ALICE):
            executor.submit(run, _scalars(9000))
            # a free waits for the use in progress, then revokes the step
            executor.free(columns.discount)
            assert _value(executor.result(run)) == pytest.approx(
                _reference(lineitem, 9000), rel=1e-9
            )
            with pytest.raises(RefusedError, match="may not read"):
                executor.submit(run, _scalars(9000))
            executor.unprepare(run)
