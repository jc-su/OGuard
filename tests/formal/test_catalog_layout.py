"""Registration-time ABI structure never retains dynamic launch authority."""

from dataclasses import replace

import pytest

from opgate.binding import Base, Binding, Bounded, Constant, Size, Span, Zero
from opgate.catalog import (
    Allocate,
    At,
    BufferId,
    Catalog,
    Free,
    Grant,
    Kernel,
    Launch,
    OperatorId,
    Reserve,
    Step,
    StepRefused,
    TenantId,
    Value,
    View,
)
from opgate.signature import Affine, Direction, SizeSymbol

ALICE, BOB = TenantId(1), TenantId(2)
SOURCE, OUTPUT = BufferId(ALICE, 1), BufferId(ALICE, 2)
OPERATOR = OperatorId(1)
N = Affine.symbol(0)
BINDING = Binding(
    "mixed",
    (SizeSymbol("n", 1, 8),),
    (
        ("zero", Zero()),
        ("source", Base(Span(Direction.READ, N.times(4), 8, None, False))),
        ("constant", Constant(-17)),
        ("size", Size(N)),
        ("scalar", Bounded(-4, 7)),
        ("output", Base(Span(Direction.WRITE, N.times(8), 8, None, False))),
        ("end", Size(N.plus(2))),
        ("fixed", Size(Affine(3))),
    ),
    (32, 1, 1),
    (2, 1, 1),
    N.times(4),
)


def catalog() -> Catalog:
    result = Catalog(4096, ((OPERATOR, BINDING),))
    assert not isinstance(
        result.apply(
            Step(ALICE, (Reserve(0, 4096), Allocate(SOURCE, 0, 32), Allocate(OUTPUT, 256, 64)))
        ),
        StepRefused,
    )
    return result


def launch(n: int = 2, scalar: int = -3) -> Launch:
    return Launch(OPERATOR, (1, 1, 1), (SOURCE, OUTPUT), (n,), (scalar,), False)


def test_repeated_launches_rebuild_every_dynamic_argument():
    state = catalog()
    for n, scalar in ((2, -3), (8, 7), (1, -4)):
        assert state.apply(Step(ALICE, (launch(n, scalar),))) == (
            Kernel(
                OPERATOR,
                (1, 1, 1),
                4 * n,
                (
                    Value(0),
                    At(0),
                    Value(-17),
                    Value(n),
                    Value(scalar),
                    At(256),
                    Value(n + 2),
                    Value(3),
                ),
                (),
            ),
        )
    assert not isinstance(
        state.apply(Step(ALICE, (Free(OUTPUT), Allocate(OUTPUT, 512, 64)))), StepRefused
    )
    effects = state.apply(Step(ALICE, (launch(),)))
    assert isinstance(effects, tuple)
    assert isinstance(effects[0], Kernel)
    assert effects[0].arguments[5] == At(512)


@pytest.mark.parametrize(
    ("changed", "reason"),
    (
        (replace(launch(), sizes=(9,)), "a size lies outside its declared bounds"),
        (replace(launch(), scalars=(8,)), "a scalar lies outside its bounds"),
        (replace(launch(), scalars=()), "a scalar is missing or extra"),
        (replace(launch(), grid=(3, 1, 1)), "the grid is larger than the binding's maxima"),
        (
            replace(launch(), buffers=(SOURCE, View(OUTPUT, 128))),
            "a view is not at a multiple of 256 bytes, or an index span's view does not start "
            "its buffer",
        ),
        (
            replace(launch(n=8), buffers=(View(OUTPUT, 0), View(OUTPUT, 256))),
            "a buffer is smaller than its extent",
        ),
    ),
)
def test_success_never_caches_dynamic_launch_checks(changed, reason):
    state = catalog()
    assert not isinstance(state.apply(Step(ALICE, (launch(),))), StepRefused)
    before = state.buffers
    assert state.apply(Step(ALICE, (changed,))) == StepRefused(f"action 0: {reason}")
    assert state.buffers == before


def test_current_permissions_and_extent_are_checked_after_success():
    state = catalog()
    assert not isinstance(state.apply(Step(ALICE, (launch(),))), StepRefused)
    fork = state.copy()
    assert fork.binding(OPERATOR) == BINDING
    assert fork.operators == state.operators
    assert not isinstance(fork.apply(Step(ALICE, (Grant(OUTPUT, BOB),))), StepRefused)
    assert fork.apply(Step(ALICE, (launch(),))) == StepRefused(
        "action 0: the tenant may not write a buffer argument, or passes a written buffer "
        "twice with an index range or a flag"
    )
    assert not isinstance(state.apply(Step(ALICE, (launch(),))), StepRefused)
    assert not isinstance(
        state.apply(Step(ALICE, (Free(OUTPUT), Allocate(OUTPUT, 512, 8)))), StepRefused
    )
    assert state.apply(Step(ALICE, (launch(),))) == StepRefused(
        "action 0: a buffer is smaller than its extent"
    )


def test_an_operator_added_after_copy_has_its_own_layout():
    state = catalog()
    fork = state.copy()
    other = OperatorId(2)
    binding = replace(BINDING, parameters=(("constant", Constant(42)),))
    fork.add_operator(other, binding)
    proposed = Launch(other, (1, 1, 1), (), (2,), (), False)
    assert fork.apply(Step(ALICE, (proposed,))) == (Kernel(other, (1, 1, 1), 8, (Value(42),), ()),)
    assert state.apply(Step(ALICE, (proposed,))) == StepRefused(
        "action 0: the operator is not verified"
    )
    assert state.binding(other) is None


def test_a_view_is_the_buffer_from_its_offset():
    """A view's argument is its buffer's address plus the view's offset, and a
    buffer that spans without an index range read and write may be passed twice."""
    state = catalog()
    assert not isinstance(
        state.apply(Step(ALICE, (Free(OUTPUT), Allocate(OUTPUT, 256, 1024)))), StepRefused
    )
    for buffers, source, output in (
        ((SOURCE, View(OUTPUT, 512)), 0, 768),
        ((View(OUTPUT, 0), View(OUTPUT, 768)), 256, 1024),
        ((OUTPUT, OUTPUT), 256, 256),
    ):
        effects = state.apply(Step(ALICE, (replace(launch(), buffers=buffers),)))
        assert isinstance(effects, tuple) and isinstance(effects[0], Kernel)
        assert (effects[0].arguments[1], effects[0].arguments[5]) == (At(source), At(output))
