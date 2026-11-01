"""A prepared step's use decides exactly as the full check of the same step."""

import random

from opgate.binding import Base, Binding, Bounded, ElementType, Span
from opgate.catalog import (
    Allocate,
    BufferId,
    Catalog,
    Fill,
    Free,
    Grant,
    Launch,
    OperatorId,
    Pending,
    Prepared,
    Reserve,
    Step,
    StepRefused,
    TenantId,
    Use,
    Write,
)
from opgate.signature import Affine, Direction, SizeSymbol

ALICE, BOB = TenantId(1), TenantId(2)
SOURCE, OUTPUT = BufferId(ALICE, 1), BufferId(ALICE, 2)
OPERATOR = OperatorId(1)
N = Affine.symbol(0)
DOUBLE = ElementType(8, False)
BINDING = Binding(
    "sum",
    (SizeSymbol("n", 1, 8),),
    (
        ("source", Base(Span(Direction.READ, N.times(4), 4, None, False))),
        ("low", Bounded(-4, 7)),
        ("high", Bounded(0, 2**64 - 1)),
        ("output", Base(Span(Direction.READ_WRITE, Affine(8), 8, None, False))),
    ),
    (32, 1, 1),
    (2, 1, 1),
    Affine(0),
)


def step(scalars: tuple[int, ...]) -> Step:
    launch = Launch(OPERATOR, (1, 1, 1), (SOURCE, OUTPUT), (8,), scalars, False)
    return Step(ALICE, (Fill(OUTPUT, DOUBLE, 0), launch))


def fresh() -> Catalog:
    catalog = Catalog(8192, ((OPERATOR, BINDING),))
    allocated = catalog.apply(
        Step(ALICE, (Reserve(0, 8192), Allocate(SOURCE, 0, 32), Allocate(OUTPUT, 256, 8)))
    )
    assert not isinstance(allocated, StepRefused)
    return catalog


def state(catalog: Catalog) -> tuple[object, ...]:
    return (set(catalog.buffers), set(catalog.grants), catalog.freed)


def change(catalog: Catalog, rng: random.Random) -> None:
    """A random change to the prepared step's buffers, which the catalog may refuse."""
    target = rng.choice((SOURCE, OUTPUT))
    tenant = rng.choice((ALICE, ALICE, BOB))
    found = catalog.buffer(target)
    match rng.randrange(4):
        case 0 if found is not None:
            data = bytes(rng.randrange(256) for _ in range(found.nbytes))
            element = rng.choice((None, ElementType(4, True), DOUBLE))
            catalog.apply(Step(tenant, (Write(target, data, element),)))
        case 1:
            catalog.apply(Step(tenant, (Grant(target, BOB),)))
        case 2:
            catalog.apply(Step(tenant, (Free(target),)))
        case _:
            offset, nbytes = (0, 32) if target == SOURCE else (256, rng.choice((8, 16)))
            catalog.apply(Step(tenant, (Allocate(target, offset, nbytes),)))


def test_uses_decide_as_the_full_check():
    rng = random.Random(7)
    accepted = 0
    for _ in range(40):
        catalog = fresh()
        prepared = catalog.prepare(step((0, 0)))
        assert isinstance(prepared, Prepared)
        for _ in range(60):
            if rng.random() < 0.2:
                change(catalog, rng)
            scalars = (rng.randrange(-6, 9), rng.choice((0, 5, 2**64 - 1, 2**64, -1)))
            expected = catalog.copy()
            full = expected.begin(step(scalars))
            use = catalog.begin_prepared(prepared, scalars)
            match full, use:
                case StepRefused(reason=a), StepRefused(reason=b):
                    assert a == b
                case Pending(), Use():
                    assert full.effects == use.checked()
                    full.commit()
                    use.commit()
                    assert state(expected) == state(catalog)
                    accepted += 1
                case _:
                    raise AssertionError(f"full check {full!r}, prepared use {use!r}")
    assert accepted > 300


def test_a_use_after_a_free_is_refused():
    catalog = fresh()
    prepared = catalog.prepare(step((1, 1)))
    assert isinstance(prepared, Prepared)
    first = catalog.begin_prepared(prepared, (1, 1))
    assert isinstance(first, Use)
    first.commit()
    assert not isinstance(catalog.apply(Step(ALICE, (Free(SOURCE),))), StepRefused)
    refused = catalog.begin_prepared(prepared, (1, 1))
    assert refused == StepRefused("action 1: the tenant may not read a buffer argument")


def test_a_grant_revokes_the_write():
    catalog = fresh()
    prepared = catalog.prepare(step((1, 1)))
    assert isinstance(prepared, Prepared)
    used = catalog.begin_prepared(prepared, (1, 1))
    assert isinstance(used, Use)
    used.commit()
    assert not isinstance(catalog.apply(Step(ALICE, (Grant(OUTPUT, BOB),))), StepRefused)
    assert isinstance(catalog.begin_prepared(prepared, (1, 1)), StepRefused)


def test_only_fills_and_one_launch_are_prepared():
    catalog = fresh()
    assert catalog.prepare(Step(ALICE, (Fill(OUTPUT, DOUBLE, 0),))) == StepRefused(
        "a prepared step is fills followed by one launch"
    )
    assert isinstance(catalog.prepare(Step(BOB, step((1, 1)).actions)), StepRefused)
