"""Rocq decides every step that the executor applied in a run as Python did.

The executor runs a small workload of two tenants and records the catalog's
decisions (`Executor.trace`): allocations, writes with and without an element
type, a grant, frees, kept launches of a Triton projection and a Q6 sum with
new scalars each time, a gather whose row ids the catalog checked, a guarded
gather whose row ids it checks when it runs, and refused launches. Replayed
through Rocq from the catalog at the trace's start, every decision equals
Python's, and Python's equals the decision that the run applied, a kept
launch's included.
"""

import struct

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.binding import ElementType
from opgate.catalog import Grant, Launch, Step, TenantId, View, Write
from opgate.catalog.rocq import as_sets, replay
from opgate.executor import Executor, Module, RefusedError

ALICE, BOB = TenantId(1), TenantId(2)
INDEX = ElementType(4, True)


def _register(executor: Executor, model: str, task: str) -> Module:
    text = (directory_of(model, task) / "triton.ptx").read_text()
    return executor.register(text, binding_of(model, task, "triton", text))


def _bits(value: float) -> int:
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


@pytest.mark.gpu
@pytest.mark.slow
def test_rocq_replays_every_decision_of_a_run() -> None:
    n = 3000
    rng = np.random.default_rng(1)
    price, discount = rng.uniform(1, 100, n), rng.choice([0.04, 0.05, 0.06, 0.07], n)
    shipdate, quantity = rng.integers(0, 50, n, dtype=np.int32), rng.uniform(1, 50, n)
    rows = rng.integers(0, n, n, dtype=np.int32)
    with Executor(0, 1 << 24) as executor, executor.trace() as (start, recorded):
        with executor.tenant(ALICE):
            executor.reserve(1 << 23)
            projection = _register(executor, "fable", "projection")
            gather = _register(executor, "fable", "gather")
            q6 = _register(executor, "opus", "q6_sum")
            # one buffer holds the projection's columns as views
            columns = executor.allocate(3 * 8 * n + 512)
            data = price.tobytes() + bytes(256 - (8 * n) % 256) + discount.tobytes()
            executor.execute(Step(ALICE, (Write(columns, data, None),)))
            second = 8 * n + 256 - (8 * n) % 256
            out = View(columns, 2 * second)
            for count in (n, n // 2, 1):
                launch = Launch(
                    projection.operator,
                    (-(-count // 1024), 1, 1),
                    (View(columns, 0), View(columns, second), out),
                    (count,),
                    (),
                    False,
                )
                assert executor.launch(launch)
                assert executor.launch(launch)
            inputs = [executor.allocate(a.nbytes) for a in (shipdate, discount, quantity, price)]
            result = executor.allocate(8)
            writes = (shipdate, discount, quantity, price)
            executor.execute(
                Step(
                    ALICE,
                    tuple(Write(b, a.tobytes(), None) for b, a in zip(inputs, writes, strict=True)),
                )
            )
            for hi in (10, 20, 30):
                scalars = (0, hi, _bits(0.05), _bits(0.07), _bits(24.0))
                launch = Launch(q6.operator, (4, 1, 1), (*inputs, result), (n,), scalars, False)
                executor.execute(Step(ALICE, (Write(result, bytes(8), None),)))
                assert executor.launch(launch)
            outside = Launch(
                q6.operator, (4, 1, 1), (*inputs, result), (n,), (0, 2**32, 0, 0, 0), False
            )
            with pytest.raises(RefusedError):
                executor.launch(outside)
            # a gather with row ids that the catalog checked, and with raw ones,
            # which a guarded launch checks
            column, index, gathered = (executor.allocate(k * n) for k in (8, 4, 8))
            executor.execute(
                Step(
                    ALICE,
                    (Write(column, price.tobytes(), None), Write(index, rows.tobytes(), INDEX)),
                )
            )
            gathering = Launch(
                gather.operator, (3, 1, 1), (column, index, gathered), (n, n), (), False
            )
            assert executor.launch(gathering)
            executor.execute(Step(ALICE, (Write(index, rows.tobytes(), None),)))
            with pytest.raises(RefusedError):
                executor.launch(gathering)
            guarded = Launch(
                gather.operator, (3, 1, 1), (column, index, gathered), (n, n), (), True
            )
            assert executor.launch(guarded)
            executor.execute(Step(ALICE, (Grant(column, BOB),)))
        with executor.tenant(BOB):
            executor.reserve(1 << 23)
            theirs = executor.allocate(8 * n)
            # Bob reads the granted column, and may not write Alice's buffer
            reading = Launch(gather.operator, (3, 1, 1), (column, index, theirs), (n, n), (), True)
            with pytest.raises(RefusedError):
                executor.launch(reading)
            writing = Launch(
                gather.operator, (3, 1, 1), (column, index, gathered), (n, n), (), True
            )
            with pytest.raises(RefusedError):
                executor.launch(writing)
            executor.free(theirs)
        with executor.tenant(ALICE):
            for buffer in (columns, *inputs, result, column, index, gathered):
                executor.free(buffer)
        replayed = replay(start, recorded)
    assert len(replayed) > 30
    assert [r for r in replayed if not r.as_run] == []
    assert [r for r in replayed if as_sets(r.rocq) != as_sets(r.python)] == []
    assert sum(r.python.startswith("refused") for r in replayed) >= 4
