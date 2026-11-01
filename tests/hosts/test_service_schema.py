"""Deployment contracts stay fixed while named launch values vary."""

import pickle
from dataclasses import FrozenInstanceError, replace

import pytest

from opgate import signature as sig
from opgate.binding import Base, Bounded, Convention, Direction, ElementType
from opgate.hosts.service import Fields, OperatorSchema, RequestError

N = sig.Affine.symbol(0)
SIGNATURE = sig.Signature(
    (sig.SizeSymbol("n", 1, 1024),),
    (
        sig.Span("source", "column", Direction.READ, N),
        sig.Span("output", "result", Direction.WRITE, N),
        sig.Scalar("n", sig.SizeOf(N)),
        sig.Scalar("delta", sig.Between(0, 100)),
    ),
)


def _schema(convention: Convention = Convention.C) -> OperatorSchema:
    return OperatorSchema(
        "transform",
        SIGNATURE,
        (("source", ElementType(8, True)), ("output", ElementType(8, True))),
        convention,
        (128, 1, 1),
        (8, 1, 1),
    )


def _ptx(parameters: int) -> str:
    arguments = ", ".join(f".param .u64 p{i}" for i in range(parameters))
    return f".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry kernel({arguments}) {{ret;}}"


def _request(**changes: object) -> Fields:
    return Fields(
        {
            "op": "run_operator",
            "operator": "registered-handle",
            "grid": [2, 1, 1],
            "buffers": {"output": "output-handle", "source": "source-handle"},
            "sizes": {"n": 200},
            "scalars": {"delta": 7},
            "guarded": False,
            **changes,
        },
        "request",
    )


@pytest.mark.parametrize(
    ("convention", "count", "bases"),
    [(Convention.C, 4, (0, 1)), (Convention.NUMBA, 16, (4, 11)), (Convention.TRITON, 6, (0, 1))],
)
def test_named_launch_follows_the_compiler_abi(convention, count, bases):
    schema = _schema(convention)
    plan = schema.plan(_ptx(count), Fields({"kind": "transform"}, "plan"))
    assert (
        tuple(i for i, (_, p) in enumerate(plan.binding.parameters) if isinstance(p, Base)) == bases
    )
    assert sum(isinstance(p, Bounded) for _, p in plan.binding.parameters) == 1
    launch = plan.launch(_request())
    assert launch.grid == (2, 1, 1)
    assert launch.buffers == ("source-handle", "output-handle")
    assert launch.sizes == (200,)
    assert launch.scalars == (7,)
    assert launch.guarded is False
    assert pickle.loads(pickle.dumps(plan)) == plan
    with pytest.raises(FrozenInstanceError):
        schema.kind = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "changes",
    [
        {"grid": [9, 1, 1]},
        {"grid": [1, 0, 1]},
        {"grid": [True, 1, 1]},
        {"grid": [1.0, 1, 1]},
        {"grid": [1, 1]},
        {"grid": [1, 1, 1, 1]},
        {"grid": "1,1,1"},
        {"sizes": {"n": 0}},
        {"sizes": {"n": 1025}},
        {"sizes": {"n": True}},
        {"sizes": {"n": 200, "capacity": 300}},
        {"scalars": {"delta": -1}},
        {"scalars": {"delta": 101}},
        {"scalars": {"delta": 1.0}},
        {"scalars": {}},
        {"scalars": {"delta": 7, "n": 200}},
        {"buffers": {"source": "s"}},
        {"buffers": {"source": 123, "output": "o"}},
        {"guarded": 1},
        {"op": "other"},
        {"operator": 123},
        {"block": [1, 1, 1]},
    ],
)
def test_launch_rejects_invalid_or_extra_values(changes):
    plan = _schema().plan(_ptx(4), Fields({"kind": "transform"}, "plan"))
    with pytest.raises(RequestError):
        plan.launch(_request(**changes))


@pytest.mark.parametrize("field", ["grid", "buffers", "sizes", "scalars", "guarded"])
def test_launch_requires_every_field(field):
    values = _request().record()
    del values[field]
    plan = _schema().plan(_ptx(4), Fields({"kind": "transform"}, "plan"))
    with pytest.raises(RequestError):
        plan.launch(Fields(values, "request"))


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "other"},
        {"kind": "transform", "binding": {}},
        {"kind": "transform", "block": [1, 1, 1]},
    ],
)
def test_client_cannot_override_deployment_contract(fields):
    with pytest.raises(RequestError):
        _schema().plan(_ptx(4), Fields(fields, "plan"))


@pytest.mark.parametrize("text", ["", _ptx(4) + _ptx(4)])
def test_schema_requires_exactly_one_entry(text):
    with pytest.raises(RequestError):
        _schema().plan(text, Fields({"kind": "transform"}, "plan"))


def test_bad_layout_is_refused_before_registration():
    with pytest.raises(ValueError, match="signature lays out"):
        _schema().plan(_ptx(5), Fields({"kind": "transform"}, "plan"))


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": ""},
        {"block": (True, 1, 1)},
        {"block": (1, 0, 1)},
        {"grid": (2**31, 1, 1)},
        {"element_types": ()},
        {"element_types": (("source", ElementType(8, True)),) * 2},
        {"convention": "c"},
        {"shared": sig.Affine(-1)},
    ],
)
def test_bad_deployment_configuration_fails_at_construction(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(_schema(), **changes)


def test_signature_configuration_cannot_alias_mutable_containers():
    sizes = list(SIGNATURE.sizes)
    signature = sig.Signature(sizes, SIGNATURE.arguments)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="immutable"):
        replace(_schema(), signature=signature)
    terms = [(0, 1)]
    affine = sig.Affine(0, terms)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="immutable"):
        replace(_schema(), shared=affine)


@pytest.mark.parametrize("bound", [sig.Between(-1, 0), sig.Between(0, 2**64)])
def test_schema_rejects_unrepresentable_parameter_bits(bound):
    signature = replace(SIGNATURE, arguments=(*SIGNATURE.arguments[:3], sig.Scalar("delta", bound)))
    with pytest.raises(ValueError, match="parameter bits"):
        replace(_schema(), signature=signature)


@pytest.mark.parametrize("argument", [sig.IndexIn, sig.IndexOut, sig.Offsets])
def test_index_contracts_and_guard_request_survive_generic_planning(argument):
    m = sig.Affine.symbol(1)
    signature = sig.Signature(
        (sig.SizeSymbol("n", 1, 1024), sig.SizeSymbol("m", 0, 1024)),
        (sig.Span("source", "source", Direction.READ, N), argument("index", "index", m, "source")),
    )
    schema = OperatorSchema(
        "indexed",
        signature,
        (("source", ElementType(8, False)), ("index", ElementType(4, True))),
        Convention.C,
        (128, 1, 1),
    )
    plan = schema.plan(_ptx(2), Fields({"kind": "indexed"}, "plan"))
    launch = plan.launch(
        _request(
            buffers={"source": "s", "index": "i"},
            sizes={"n": 200, "m": 7},
            scalars={},
            guarded=True,
        )
    )
    assert launch.guarded is True
    assert launch.sizes == (200, 7)
    assert launch.buffers == ("s", "i")
    span = plan.binding.bases[1][1]
    assert span.index is not None
    assert span.index.hi == (N if argument is sig.Offsets else N.plus(-1))
