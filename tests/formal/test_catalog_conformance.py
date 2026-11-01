"""The Python `check_step` agrees with `check_step` of `formal/CatalogStep.v`.

A seeded generator runs 400 steps of random tenants over a small pool:
reservations, which each tenant makes first and may end and make again,
allocations at aligned and unaligned offsets, in the tenant's reservation
or not and with the tenant's ids or another's, frees, grants, writes with
and without an element type, fills, and launches of five operators, which
cover every kind of parameter and every role, a flag buffer included,
guarded or not, one of them with grid maxima below the generated grids; a
guarded
launch of an unchecked index input carries a check. Buffer arguments are
whole buffers or views at aligned and unaligned offsets, and a launch may
pass one buffer several times. About half the launches
are built from the current state to be accepted, the rest at random. Frees
of adjacent buffers of one tenant merge their regions, and allocations
across them keep their bytes. The catalog after each accepted step is the
next step's. Rocq evaluates `check_step` on every (catalog, step) pair, and
its printed decision, `show_decision`, equals the one printed here from the
Python decision: the same refusal reason, or the same next catalog and
effects. The seed 7 run, with hand-written edge cases, must reach every
refusal reason; eight more seeds add 3,200 steps.
"""

import random
import re

import pytest

from opgate import signature as sig
from opgate.binding import (
    BASE_ALIGN,
    GRID_LIMIT,
    Base,
    Binding,
    Convention,
    Direction,
    ElementType,
    Span,
    derive_binding,
)
from opgate.catalog import (
    Action,
    Allocate,
    BufferId,
    Catalog,
    Fill,
    Free,
    Grant,
    Launch,
    OperatorId,
    Region,
    Reserve,
    Step,
    StepAccepted,
    StepRefused,
    TenantId,
    Unreserve,
    View,
    Write,
    check_step,
)
from opgate.catalog.rocq import as_sets, decide, show
from opgate.signature import Affine

POOL = 4096
STEPS = 400
TENANTS = (TenantId(1), TenantId(2), TenantId(3))
SLOT = 1280
"""Tenant t's reservation is the SLOT bytes from (t - 1) * SLOT; the pool's last
bytes lie outside every reservation."""
F64, I32, U8 = ElementType(8, False), ElementType(4, True), ElementType(1, False)
N, M = Affine.symbol(0), Affine.symbol(1)


def _operators() -> tuple[tuple[OperatorId, Binding], ...]:
    scale = sig.Signature(
        (sig.SizeSymbol("n"),),
        (
            sig.Span("values", "values", Direction.READ, N),
            sig.Span("out", "out", Direction.WRITE, N),
            sig.Scalar("n", sig.SizeOf(N)),
        ),
    )
    gather = sig.Signature(
        (sig.SizeSymbol("n"), sig.SizeSymbol("m", 1)),
        (
            sig.Span("col", "col", Direction.READ, N),
            sig.IndexIn("sel", "sel", M, "col"),
            sig.Span("out", "out", Direction.READ_WRITE, M),
            sig.Scalar("shift", sig.Between(0, 24)),
        ),
    )
    select = sig.Signature(
        (sig.SizeSymbol("n"),),
        (
            sig.Span("rows", "rows", Direction.READ, N),
            sig.IndexOut("sel", "sel", N, "rows"),
            sig.Scalar("n", sig.SizeOf(N.plus(1))),
        ),
    )
    block = (256, 1, 1)
    return (
        (
            OperatorId(0),
            derive_binding(
                scale,
                {"values": F64, "out": F64},
                Convention.C,
                "a",
                ("a0", "a1", "a2"),
                block,
                GRID_LIMIT,
                Affine(0),
            ),
        ),
        (
            OperatorId(1),
            derive_binding(
                gather,
                {"col": F64, "sel": I32, "out": F64},
                Convention.C,
                "b",
                ("b0", "b1", "b2", "b3"),
                block,
                (2, 1, 2),
                Affine(0),
            ),
        ),
        (
            OperatorId(2),
            derive_binding(
                select,
                {"rows": U8, "sel": I32},
                Convention.C,
                "c",
                ("c0", "c1", "c2"),
                block,
                GRID_LIMIT,
                Affine(0),
            ),
        ),
        (
            OperatorId(3),
            derive_binding(
                scale,
                {"values": F64, "out": F64},
                Convention.NUMBA,
                "d",
                tuple(f"d{i}" for i in range(15)),
                block,
                GRID_LIMIT,
                Affine(0),
            ),
        ),
        # a written buffer and a flag buffer, as the guard pass's status buffer
        (
            OperatorId(4),
            Binding(
                "e",
                (),
                (
                    ("e0", Base(Span(Direction.WRITE, Affine(4), BASE_ALIGN, None, False))),
                    ("e1", Base(Span(Direction.WRITE, Affine(4), BASE_ALIGN, None, True))),
                ),
                block,
                GRID_LIMIT,
                Affine(0),
            ),
        ),
    )


# ------------------------------------------------------------------
# Generated steps.


def _data(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def _free_offset(c: Catalog, tenant: TenantId, nbytes: int) -> int | None:
    """An offset in the tenant's reservation, a multiple of 256, where nbytes
    overlap no live buffer."""
    reservation = c.reservation(tenant)
    if reservation is None:
        return None
    start, end = reservation.offset, reservation.offset + reservation.nbytes
    for offset in range(start, end - nbytes + 1, 256):
        if all(offset + nbytes <= b.offset or b.offset + b.nbytes <= offset for b in c.buffers):
            return offset
    return None


def _coin(rng: random.Random) -> bool:
    return rng.random() < 0.5


def _likely_step(rng: random.Random, c: Catalog, tenant: TenantId) -> tuple[Action, ...]:
    """Actions built from the state, which the catalog mostly accepts."""
    if c.reservation(tenant) is None:
        return (Reserve((tenant - 1) * SLOT, SLOT),)
    own = [b for b in c.buffers if b.owner == tenant and not c.granted(b.id)]
    if not own and rng.random() < 0.2:
        return (Unreserve(),)
    used = {b.id for b in c.buffers}
    fresh = [BufferId(tenant, i) for i in range(1, 13) if BufferId(tenant, i) not in used]
    # two adjacent buffers of the tenant, freed together, become one region;
    # half the time an allocation across both takes it again and keeps its bytes
    pairs = [(x, y) for x in own for y in own if x.offset + x.nbytes == y.offset]
    frees: tuple[Action, ...] = ()
    if pairs:
        x, y = rng.choice(pairs)
        frees = (Free(x.id), Free(y.id))
        if _coin(rng):
            frees += (Allocate(x.id, x.offset, x.nbytes + y.nbytes),)
    kind = rng.randrange(6)
    if kind <= 1 or len(own) < 3:
        # a whole freed region of the tenant, which may join several frees, or
        # the first place where the bytes fit; buffers of 256 and 512 bytes
        # lie next to each other
        mine = [r for r in c.freed if r.owner == tenant and r.offset % 256 == 0]
        if mine and _coin(rng):
            region = rng.choice(mine)
            offset, nbytes = region.offset, region.nbytes
        else:
            nbytes = rng.choice((256, 512, rng.randrange(8, 520, 8)))
            offset = _free_offset(c, tenant, nbytes)
        if offset is None or not fresh:
            return frees or ((Free(rng.choice(own).id),) if own else ())
        return (Allocate(rng.choice(fresh), offset, nbytes),)
    a, b, d = rng.sample(own, 3)
    match kind:
        case 2:
            # half the time, views at multiples of 256 bytes into the buffers, or
            # two views of one buffer, one read and one written
            if _coin(rng):
                n = min(a.nbytes, b.nbytes) // 8
                return (
                    Write(a.id, _data(rng, 8 * n), None),
                    Launch(
                        OperatorId(0), (rng.randint(1, 4), 1, 1), (a.id, b.id), (n,), (), _coin(rng)
                    ),
                )
            x, y = (a, b) if _coin(rng) else (a, a)
            ox, oy = (rng.randrange(0, v.nbytes, 256) for v in (x, y))
            n = min(x.nbytes - ox, y.nbytes - oy) // 8
            views = (View(x.id, ox), View(y.id, oy))
            return (Launch(OperatorId(0), (rng.randint(1, 4), 1, 1), views, (n,), (), _coin(rng)),)
        case 3:
            # an index input written unchecked is checked by a guarded launch when it runs
            n, m = a.nbytes // 8, min(b.nbytes // 4, d.nbytes // 8)
            rows = b"".join(
                rng.randrange(max(n, 1)).to_bytes(4, "little") for _ in range(b.nbytes // 4)
            )
            checked = _coin(rng)
            return (
                Write(b.id, rows, I32 if checked else None),
                Launch(
                    OperatorId(1),
                    (1, 1, 1),
                    (a.id, b.id, d.id),
                    (n, max(m, 1)),
                    (rng.randint(0, 30),),
                    not checked or _coin(rng),
                ),
            )
        case 4:
            n = min(a.nbytes, b.nbytes // 4)
            launch = Launch(OperatorId(2), (2, 1, 1), (a.id, b.id), (n,), (), _coin(rng))
            return (Fill(b.id, I32, 0), launch)
        case _:
            return rng.choice(
                (
                    frees or (Free(a.id),),
                    (Grant(a.id, TenantId(tenant % 3 + 1)),),
                    (Fill(d.id, U8, 7),),
                )
            )


def _random_action(rng: random.Random, c: Catalog, tenant: TenantId) -> Action:
    # mostly the tenant's own ids, sometimes another tenant's
    ids = [BufferId(tenant if rng.random() < 0.8 else rng.choice(TENANTS), i) for i in range(1, 9)]
    element = rng.choice((F64, I32, U8, ElementType(2, True)))
    match rng.randrange(8):
        case 0 | 1:
            offset = rng.randrange(0, POOL + 1, rng.choice((128, 256)))
            return Allocate(rng.choice(ids), offset, rng.randrange(0, 1025))
        case 7:
            if _coin(rng):
                return Unreserve()
            return Reserve(rng.randrange(-256, POOL + 1, 256), rng.randrange(0, 2049, 256))
        case 2:
            return Free(rng.choice(ids))
        case 3:
            return Grant(rng.choice(ids), TenantId(rng.randint(1, 3)))
        case 4:
            buffer = rng.choice(ids)
            found = c.buffer(buffer)
            if found is not None and rng.random() < 0.6:
                n = found.nbytes - found.nbytes % element.width
                return Write(buffer, _data(rng, n), element)
            return Write(buffer, _data(rng, rng.randrange(0, 300)), rng.choice((None, element)))
        case 5:
            value = rng.randint(-(2 ** (8 * element.width)), 2 ** (8 * element.width))
            return Fill(rng.choice(ids), element, rng.choice((value, 0, 3)))
        case _:
            op = OperatorId(rng.randrange(6))
            grid = (rng.randint(0, 3), rng.randint(1, 2), rng.randint(1, 2))
            buffers = tuple(
                View(rng.choice(ids), rng.choice((0, 128, 256, 512)))
                if _coin(rng)
                else rng.choice(ids)
                for _ in range(rng.randint(1, 3))
            )
            sizes = tuple(rng.randint(-1, 40) for _ in range(rng.randint(1, 2)))
            scalars = tuple(rng.randint(-1, 30) for _ in range(rng.randint(0, 1)))
            return Launch(op, grid, buffers, sizes, scalars, _coin(rng))


SEEDS = (11, 13, 17, 19, 23, 29, 31, 37)


def _cases(seed: int = 7) -> list[tuple[Catalog, Step]]:
    rng = random.Random(seed)
    catalog = Catalog(POOL, operators=_operators())
    cases = []
    for _ in range(STEPS):
        tenant = TenantId(rng.randint(1, 3))
        if rng.random() < 0.7:
            actions = _likely_step(rng, catalog, tenant)
        else:
            actions = tuple(_random_action(rng, catalog, tenant) for _ in range(rng.randint(1, 3)))
        step = Step(tenant, actions)
        cases.append((catalog, step))
        decision = check_step(catalog, step)
        if isinstance(decision, StepAccepted):
            catalog = decision.next
    return cases


REASONS = {
    "the buffer id is not the tenant's",
    "the buffer id is in use",
    "the region is empty or outside the pool",
    "the region is not aligned to 256 bytes",
    "the region lies outside the tenant's reservation",
    "the region overlaps a live buffer",
    "the tenant holds a reservation",
    "the reservation is empty or outside the pool",
    "the reservation overlaps another",
    "the tenant holds no reservation",
    "the tenant holds a live buffer",
    "the tenant does not own the buffer",
    "the tenant does not own the buffer, or grants it to itself",
    "the grant exists",
    "the tenant may not write the buffer",
    "the data is larger than the buffer",
    "checked data must fill the buffer with whole elements",
    "the value is not an element, or the buffer is not whole elements",
    "the operator is not verified",
    "the grid is outside the limits the gate assumes",
    "the grid is larger than the binding's maxima",
    "a size is missing or extra",
    "a size lies outside its declared bounds",
    "a buffer argument is missing or extra",
    "a scalar is missing or extra",
    "a scalar lies outside its bounds",
    "a view is not at a multiple of 256 bytes, or an index span's view does not start its buffer",
    "a buffer is smaller than its extent",
    "the tenant may not read a buffer argument",
    "the tenant may not write a buffer argument, or passes a written buffer twice with an index range or a flag",
    "an index buffer is not checked inside its declared range",
}
"""Every reason `check_step` gives; the cases reach each of them."""


def _edge_cases() -> list[tuple[Catalog, Step]]:
    """One case for each refusal that random steps rarely reach, frees of
    adjacent buffers of one tenant and of two tenants, and the end of a
    reservation, which drops the tenant's freed regions."""
    t, u, w = TenantId(1), TenantId(2), TenantId(3)
    b1, b2, b3, b6, b7, b8 = (BufferId(t, i) for i in (1, 2, 3, 6, 7, 8))
    b4, u6 = BufferId(u, 4), BufferId(u, 6)
    setup = check_step(
        Catalog(POOL, operators=_operators()),
        Step(
            t,
            (Reserve(0, 2048), Allocate(b1, 0, 64), Allocate(b2, 256, 64), Allocate(b3, 512, 64)),
        ),
    )
    assert isinstance(setup, StepAccepted)
    theirs = check_step(setup.next, Step(u, (Reserve(2048, 1280), Allocate(b4, 2304, 64))))
    assert isinstance(theirs, StepAccepted)
    c = theirs.next
    # u's freed region at 2048, the start of its reservation, lies next to the
    # bytes that t frees at 1792
    parted = check_step(c, Step(u, (Allocate(u6, 2048, 256), Free(u6))))
    assert isinstance(parted, StepAccepted)
    scale, gather = OperatorId(0), OperatorId(1)
    rows = b"".join(v.to_bytes(4, "little") for v in range(8, 24))
    steps = [
        (c, (Allocate(b6, 1024, 0),)),
        (c, (Allocate(b6, 4096, 256),)),
        (c, (Allocate(BufferId(u, 9), 768, 256),)),
        (c, (Allocate(b6, 2048, 256),)),
        (c, (Reserve(3584, 256),)),
        (c, (Unreserve(),)),
        (c, (Grant(b1, u), Grant(b1, u))),
        (c, (Write(b1, bytes(65), None),)),
        (c, (Write(b1, bytes(60), I32),)),
        (c, (Launch(gather, (3, 1, 1), (b1, b2, b3), (8, 8), (0,), True),)),
        (c, (Fill(b1, I32, 2**31),)),
        (c, (Launch(scale, (1, 1, 1), (b1, b2), (9,), (), False),)),
        (c, (Launch(scale, (1, 1, 1), (b1, b1), (8,), (), False),)),
        (c, (Launch(scale, (1, 1, 1), (b1, b4), (8,), (), False),)),
        (c, (Launch(scale, (1, 1, 1), (b1, b2), (-1,), (), False),)),
        (c, (Launch(scale, (1, 1, 1), (b1, b6), (8,), (), False),)),
        (c, (Launch(scale, (1, 1, 1), (b4, b2), (8,), (), False),)),
        (c, (Launch(scale, (1, 1, 1), (b1, b2), (8,), (3,), False),)),
        (c, (Launch(gather, (1, 1, 1), (b1, b2, b3), (8, 8), (25,), False),)),
        (c, (Write(b2, rows, I32), Launch(gather, (1, 1, 1), (b1, b2, b3), (8, 8), (0,), False))),
        # a guarded launch checks an unchecked index input when it runs, but
        # not an unchecked index output, which it must write in range
        (c, (Write(b2, rows, None), Launch(gather, (1, 1, 1), (b1, b2, b3), (8, 8), (0,), True))),
        (c, (Launch(OperatorId(2), (1, 1, 1), (b1, b2), (8,), (), True),)),
        # views: off a multiple of 256, of an index span, past the end of the buffer
        (c, (Launch(scale, (1, 1, 1), (View(b1, 128), b2), (8,), (), False),)),
        (
            c,
            (
                Write(b2, rows, I32),
                Launch(gather, (1, 1, 1), (b1, View(b2, 256), b3), (8, 8), (0,), False),
            ),
        ),
        (c, (Launch(scale, (1, 1, 1), (b1, View(b2, 256)), (0,), (), False),)),
        # a written buffer passed twice: through spans without an index range, or
        # once as an index input
        (c, (Launch(scale, (1, 1, 1), (b2, b2), (8,), (), False),)),
        (
            c,
            (
                Write(b2, bytes(64), I32),
                Launch(gather, (1, 1, 1), (b1, b2, b2), (8, 8), (0,), False),
            ),
        ),
        # a flag buffer is passed once, and a written buffer is not also a flag
        # buffer
        (c, (Launch(OperatorId(4), (1, 1, 1), (b1, b2), (), (), False),)),
        (c, (Launch(OperatorId(4), (1, 1, 1), (b1, b1), (), (), False),)),
        # the middle one of three adjacent buffers of t, freed last, merges with
        # the regions on both sides, and an allocation across all three keeps its bytes
        (
            c,
            (
                *(Allocate(b, o, 256) for b, o in ((b6, 1024), (b7, 1280), (b8, 1536))),
                *(Free(b) for b in (b6, b8, b7)),
                Allocate(b6, 1024, 768),
            ),
        ),
        # t's region is not merged with u's, and an allocation across both lies
        # outside t's reservation
        (parted.next, (Allocate(b7, 1792, 256), Free(b7), Allocate(b7, 1792, 512))),
    ]
    reservations = [
        (c, w, (Reserve(3584, 0),)),
        (c, w, (Reserve(3072, 512),)),
        (c, w, (Unreserve(),)),
        # u frees its buffers and ends its reservation; its freed regions go
        (parted.next, u, (Free(b4), Unreserve())),
    ]
    return [(catalog, Step(t, actions)) for catalog, actions in steps] + [
        (catalog, Step(tenant, actions)) for catalog, tenant, actions in reservations
    ]


def _merged_and_spanned(cases: list[tuple[Catalog, Step]]) -> tuple[int, int]:
    """In a run of generated steps: the frees whose region merges with a freed region,
    and the allocations that keep their bytes although no single earlier free of their
    tenant holds them, which only a merged region allows."""
    frees: list[Region] = []
    merged = spanned = 0
    for catalog, step in cases:
        if isinstance(check_step(catalog, step), StepRefused):
            continue
        for action in step.actions:
            decision = check_step(catalog, Step(step.tenant, (action,)))
            assert isinstance(decision, StepAccepted)
            if isinstance(action, Free):
                buffer = catalog.buffer(action.buffer)
                assert buffer is not None
                frees.append(Region(buffer.offset, buffer.nbytes, step.tenant))
                if len(decision.next.freed) <= len(catalog.freed):
                    merged += 1
            elif isinstance(action, Allocate) and not decision.effects:
                end = action.offset + action.nbytes
                if not any(
                    r.owner == step.tenant
                    and r.offset <= action.offset
                    and end <= r.offset + r.nbytes
                    for r in frees
                ):
                    spanned += 1
            catalog = decision.next
    return merged, spanned


@pytest.mark.parametrize("divisor", [1, 3, 32, 1024])
def test_exact_derived_size_launches_agree_withdecide(divisor):
    binding = Binding(
        "kernel",
        (sig.SizeSymbol("n", 0, 100003), sig.SizeSymbol("q", 0, 100003)),
        (),
        (32, 1, 1),
        GRID_LIMIT,
        Affine(0),
        (sig.DerivedSize(1, N.plus(divisor - 1), divisor),),
    )
    catalog = Catalog(4096, operators=((OperatorId(0), binding),))
    cases = []
    for n in (0, 1, divisor - 1, divisor, divisor + 1, 100003):
        exact = (n + divisor - 1) // divisor
        for supplied in sorted({exact, max(0, exact - 1), min(100003, exact + 1)}):
            launch = Launch(OperatorId(0), (1, 1, 1), (), (n, supplied), (), False)
            step = Step(TenantId(1), (launch,))
            decision = check_step(catalog, step)
            assert isinstance(decision, StepAccepted) is (supplied == exact)
            if supplied != exact:
                assert isinstance(decision, StepRefused)
                assert (
                    decision.reason
                    == "action 0: a derived size does not equal its declared quotient"
                )
            cases.append((catalog, step))
    assert decide(cases) == [show(check_step(c, step)) for c, step in cases]


def test_negative_derived_numerator_uses_floor_in_python_anddecide():
    binding = Binding(
        "kernel",
        (sig.SizeSymbol("n"), sig.SizeSymbol("q")),
        (),
        (32, 1, 1),
        GRID_LIMIT,
        Affine(0),
        (sig.DerivedSize(1, Affine(-1), 4),),
    )
    catalog = Catalog(4096, operators=((OperatorId(0), binding),))
    step = Step(TenantId(1), (Launch(OperatorId(0), (1, 1, 1), (), (0, 0), (), False),))
    actual = decide([(catalog, step)])
    assert actual == ["refused action 0: a derived size does not equal its declared quotient"]
    assert show(check_step(catalog, step)) == actual[0]


def _state(catalog: Catalog) -> str:
    return as_sets(show(StepAccepted(catalog, ())))


def test_a_checked_or_refused_step_leaves_the_catalog_as_it_was() -> None:
    """`check` undoes every change of an accepted step, and `apply` every change of
    a refused one, including the actions it accepted before the refused one."""
    refused = 0
    for catalog, step in _cases() + _edge_cases():
        before = _state(catalog)
        decision = catalog.check(step)
        assert _state(catalog) == before
        if isinstance(decision, StepRefused):
            assert catalog.apply(step) == decision
            assert _state(catalog) == before
            refused += len(step.actions) > 1
    assert refused > 20, refused


@pytest.mark.slow
def test_python_and_rocq_check_step_agree() -> None:
    generated = _cases()
    cases = generated + _edge_cases()
    python = [show(check_step(c, s)) for c, s in cases]
    accepted = sum(1 for shown in python if shown.startswith("accepted"))
    launched = sum(1 for shown in python if "kernel(" in shown)
    checked = sum(1 for shown in python if re.search(r"kernel\([^]]*\],\[\(", shown))
    assert accepted > STEPS // 4 and launched > 20 and checked > 5, (accepted, launched, checked)
    merged, spanned = _merged_and_spanned(generated)
    assert merged > 10 and spanned > 2, (merged, spanned)
    # an allocation across adjacent frees of one tenant keeps its bytes, one
    # across frees of two tenants lies outside the reservation, and the end of
    # a reservation drops its tenant's freed regions
    assert python[-6].endswith("effects=[clear(1024,256);clear(1280,256);clear(1536,256)]")
    assert python[-5] == "refused action 2: the region lies outside the tenant's reservation"
    assert python[-1].startswith("accepted") and "freed=[]" in python[-1], python[-1]
    refused = {shown.split(": ", 1)[1] for shown in python if shown.startswith("refused")}
    assert refused == REASONS
    rocq = decide(cases)
    assert len(rocq) == len(cases)
    different = [
        (i, p, r)
        for i, (p, r) in enumerate(zip(python, rocq, strict=True))
        if as_sets(p) != as_sets(r)
    ]
    assert different == []


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_python_and_rocq_agree_on_more_seeds(seed: int) -> None:
    cases = _cases(seed)
    python = [as_sets(show(check_step(c, s))) for c, s in cases]
    assert sum(1 for shown in python if shown.startswith("accepted")) > STEPS // 4
    rocq = [as_sets(shown) for shown in decide(cases)]
    assert len(rocq) == len(cases)
    assert [(i, p, r) for i, (p, r) in enumerate(zip(python, rocq, strict=True)) if p != r] == []


@pytest.mark.parametrize("decreasing", [False, True])
def test_symbolic_shared_launch_effect_agrees_withdecide(decreasing):
    extent = Affine(49152, ((0, -1),)) if decreasing else Affine.symbol(0)
    binding = Binding(
        "kernel", (sig.SizeSymbol("k", 0, 49152),), (), (32, 1, 1), GRID_LIMIT, extent
    )
    catalog = Catalog(4096, operators=((OperatorId(0), binding),))
    cases = [
        (catalog, Step(TenantId(1), (Launch(OperatorId(0), (1, 1, 1), (), (size,), (), False),)))
        for size in (-1, 0, 1, 256, 49152, 49153)
    ]
    assert decide(cases) == [show(check_step(c, step)) for c, step in cases]
