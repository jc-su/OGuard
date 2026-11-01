"""Exact dependent sizes are derived by schemas and independently enforced by the catalog."""

from dataclasses import replace

import pytest

from opgate import signature as sig
from opgate.binding import Binding, Convention, ElementType, derive_binding
from opgate.catalog import (
    Allocate,
    BufferId,
    Catalog,
    Launch,
    OperatorId,
    Reserve,
    Step,
    StepAccepted,
    TenantId,
    check_launch,
    check_step,
)
from opgate.hosts.service import Fields, OperatorSchema, RequestError
from opgate.hosts.service.audit import binding_record, parse_binding

N = sig.Affine.symbol(0)
R = sig.DerivedSize(1, N.plus(1023), 1024)
SIGNATURE = sig.Signature(
    (sig.SizeSymbol("n", 0, 100000), sig.SizeSymbol("blocks", 0, 98)),
    (
        sig.Span("blocks", "blocks", sig.Direction.READ, sig.Affine.symbol(1)),
        sig.Scalar("n", sig.SizeOf(N)),
    ),
    (R,),
)
PTX = ".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry kernel(.param .u64 p0, .param .u64 p1) {ret;}"


def plan(signature=SIGNATURE):
    return OperatorSchema(
        "quotient", signature, (("blocks", ElementType(8, True)),), Convention.C, (32, 1, 1)
    ).plan(PTX, Fields({"kind": "quotient"}, "plan"))


def request(sizes):
    return Fields(
        {
            "op": "run_operator",
            "operator": "verified",
            "grid": [1, 1, 1],
            "buffers": {"blocks": "buffer"},
            "sizes": sizes,
            "scalars": {},
            "guarded": False,
        },
        "request",
    )


@pytest.mark.parametrize(
    "n,blocks", [(0, 0), (1, 1), (1023, 1), (1024, 1), (1025, 2), (100000, 98)]
)
def test_schema_derives_ceil_and_preserves_raw_audit(n, blocks):
    p = plan()
    assert p.launch(request({"n": n})).sizes == (n, blocks)
    raw = binding_record(p.binding)
    assert raw["derived_sizes"] == [
        {"target": 1, "numerator": {"constant": 1023, "terms": [[0, 1]]}, "divisor": 1024}
    ]
    assert parse_binding(raw) == p.binding


@pytest.mark.parametrize(
    "sizes",
    [{"n": 1025, "blocks": 2}, {"n": 1025, "blocks": 1}, {"n": True}, {"n": -1}, {"n": 100001}, {}],
)
def test_client_cannot_supply_or_override_derived_size(sizes):
    with pytest.raises(RequestError):
        plan().launch(request(sizes))


BUFFER = BufferId(TenantId(1), 1)


def catalog(binding: Binding):
    c = Catalog(4096, ((OperatorId(1), binding),))
    accepted = check_step(c, Step(TenantId(1), (Reserve(0, 4096), Allocate(BUFFER, 0, 1024))))
    assert isinstance(accepted, StepAccepted)
    return accepted.next


@pytest.mark.parametrize("n", [0, 1, 1023, 1024, 1025, 100000])
def test_direct_catalog_refuses_inconsistent_tuple_even_when_all_buffers_fit(n):
    b = plan().binding
    c = catalog(b)
    exact = (n + 1023) // 1024
    launch = Launch(OperatorId(1), (1, 1, 1), (BUFFER,), (n, exact), (), False)
    assert not isinstance(check_launch(c, TenantId(1), launch), str)
    for wrong in {max(0, exact - 1), min(98, exact + 1)} - {exact}:
        bad = replace(launch, sizes=(n, wrong))
        assert (
            check_launch(c, TenantId(1), bad)
            == "a derived size does not equal its declared quotient"
        )


def test_chained_general_affine_relations_and_derived_bounds():
    signature = replace(
        SIGNATURE,
        sizes=SIGNATURE.sizes + (sig.SizeSymbol("groups", 0, 33),),
        derived_sizes=(R, sig.DerivedSize(2, sig.Affine(2, ((1, 1),)), 3)),
    )
    assert plan(signature).launch(request({"n": 100000})).sizes == (100000, 98, 33)
    limited = replace(SIGNATURE, sizes=(SIGNATURE.sizes[0], sig.SizeSymbol("blocks", 0, 1)))
    with pytest.raises(RequestError, match="derived size"):
        plan(limited).launch(request({"n": 1025}))


@pytest.mark.parametrize(
    "target,numerator,divisor",
    [
        (1, N, 0),
        (1, N, -1),
        (1, N, True),
        (True, N, 1),
        (-1, N, 1),
        (1, sig.Affine.symbol(1), 1),
        (1, sig.Affine.symbol(-1), 1),
        (1, sig.Affine(0, [(0, 1)]), 1),  # ty: ignore[invalid-argument-type]
        (1, sig.Affine(False), 1),
        (1, sig.Affine(0, ((False, 1),)), 1),
    ],
)
def test_invalid_relation_configuration_fails(target, numerator, divisor):
    with pytest.raises((TypeError, ValueError)):
        sig.DerivedSize(target, numerator, divisor)


@pytest.mark.parametrize("relations", [[R], (R, R), (sig.DerivedSize(2, N, 1),)])
def test_relation_targets_and_immutable_sequence_checked(relations):
    with pytest.raises((TypeError, ValueError)):
        replace(SIGNATURE, derived_sizes=relations)


def test_negative_affine_uses_floor_not_truncation_and_cannot_escape_bounds():
    negative = replace(SIGNATURE, derived_sizes=(sig.DerivedSize(1, sig.Affine(-1), 4),))
    with pytest.raises(RequestError, match="derived size"):
        plan(negative).launch(request({"n": 0}))
    c = catalog(plan(negative).binding)
    launch = Launch(OperatorId(1), (1, 1, 1), (BUFFER,), (0, 0), (), False)
    assert (
        check_launch(c, TenantId(1), launch)
        == "a derived size does not equal its declared quotient"
    )


def test_derive_binding_propagates_relation_and_old_wire_shape_fails():
    b = derive_binding(
        SIGNATURE,
        {"blocks": ElementType(8, True)},
        Convention.C,
        "kernel",
        ("p0", "p1"),
        (32, 1, 1),
        (1, 1, 1),
        sig.Affine(0),
    )
    assert b.derived_sizes == (R,)
    raw = binding_record(b)
    del raw["derived_sizes"]
    with pytest.raises(RequestError):
        parse_binding(raw)
