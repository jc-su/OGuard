"""`Executor.launch` decides each launch as `execute` decides the step of the launch.

A kept launch takes one full check while the records of its buffers stay
those of that check: later uses with new scalars check only the scalars and
give the kernel's results. A free or a grant of a buffer it names, or a
scalar outside its bounds, is refused with the reason that `execute` gives.
"""

import struct
from dataclasses import replace

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.catalog import Catalog, Free, Grant, Launch, Pending, Step, StepRefused, TenantId, Write
from opgate.executor import Executor, RefusedError

ALICE, BOB = TenantId(1), TenantId(2)
ROWS = 5000


def _bits(value: float) -> int:
    (bits,) = struct.unpack("<Q", struct.pack("<d", value))
    return bits


def _scalars(lo: int, hi: int) -> tuple[int, ...]:
    return (lo % 2**32, hi % 2**32, _bits(0.05), _bits(0.07), _bits(24.0))


@pytest.mark.gpu
def test_a_kept_launch_decides_as_execute_does(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    shipdate = rng.integers(0, 100, ROWS, dtype=np.int32)
    discount = rng.choice([0.04, 0.05, 0.06, 0.07, 0.08], ROWS)
    quantity = rng.uniform(1, 50, ROWS)
    price = rng.uniform(1, 1000, ROWS)
    checks = 0
    begin = Catalog.begin

    def counted(catalog: Catalog, step: Step) -> Pending | StepRefused:
        """`Catalog.begin`, counting the full checks of a step with a launch."""
        nonlocal checks
        checks += any(isinstance(action, Launch) for action in step.actions)
        return begin(catalog, step)

    monkeypatch.setattr(Catalog, "begin", counted)
    with Executor(0, 1 << 24) as executor, executor.tenant(ALICE):
        executor.reserve(1 << 24)
        text = (directory_of("opus", "q6_sum") / "triton.ptx").read_text()
        module = executor.register(text, binding_of("opus", "q6_sum", "triton", text))
        columns = [executor.allocate(a.nbytes) for a in (shipdate, discount, quantity, price)]
        result = executor.allocate(8)
        executor.execute(
            Step(
                ALICE,
                tuple(
                    Write(b, a.tobytes(), None)
                    for b, a in zip(columns, (shipdate, discount, quantity, price), strict=True)
                ),
            )
        )
        launch = Launch(module.operator, (8, 1, 1), (*columns, result), (ROWS,), (), False)
        first = None
        for lo, hi in ((0, 100), (10, 20), (50, 51), (30, 90)):
            executor.execute(Step(ALICE, (Write(result, bytes(8), None),)))
            assert executor.launch(replace(launch, scalars=_scalars(lo, hi)))
            (revenue,) = struct.unpack("<d", executor.read(ALICE, result))
            keep = (
                (shipdate >= lo)
                & (shipdate < hi)
                & (discount >= 0.05)
                & (discount <= 0.07)
                & (quantity < 24)
            )
            assert revenue == pytest.approx(float((price * discount)[keep].sum()), rel=1e-9)
            if first is None:
                first = checks
        # the prepare and the first use checked the launch in full, and no later use
        assert first == checks
        outside = replace(launch, scalars=(0, 2**32, *_scalars(0, 0)[2:]))
        with pytest.raises(RefusedError, match="a scalar lies outside its bounds"):
            executor.launch(outside)
        executor.execute(Step(ALICE, (Grant(result, BOB),)))
        granted = replace(launch, scalars=_scalars(0, 10))
        with pytest.raises(RefusedError) as kept:
            executor.launch(granted)
        with pytest.raises(RefusedError) as stepped:
            executor.execute(Step(ALICE, (granted,)))
        assert str(kept.value) == str(stepped.value)
        assert "may not write" in str(kept.value)
        executor.execute(Step(ALICE, (Free(result),)))
        with pytest.raises(RefusedError, match="may not read"):
            executor.launch(granted)
