"""The export prints bindings as terms that Rocq accepts at the type of `verify_text`."""

import subprocess
import tempfile
from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

import pytest

from opgate import signature as sig
from opgate.binding import (
    GRID_LIMIT,
    Base,
    Binding,
    Bounded,
    Constant,
    Convention,
    ElementType,
    IndexRange,
    Size,
    Span,
    derive_binding,
)
from opgate.gate import binding_term, export_query, export_terms, formal_dir
from opgate.signature import Affine, Direction

SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"), sig.SizeSymbol("chars", 1)),
    arguments=(
        sig.Offsets("offsets", "offsets", Affine.symbol(0).plus(1), "chars"),
        sig.Span("chars", "chars", Direction.READ, Affine.symbol(1)),
        sig.IndexOut("lengths", "lengths", Affine.symbol(0), "chars"),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
        sig.Scalar("k", sig.Between(0, 24)),
    ),
)
ELEMENTS = {
    "offsets": ElementType(4, True),
    "chars": ElementType(1, False),
    "lengths": ElementType(4, True),
}


def test_a_printed_binding_has_the_rocq_type() -> None:
    names = tuple(f"kernel_param_{i}" for i in range(5))
    binding = derive_binding(
        SIGNATURE, ELEMENTS, Convention.C, "kernel", names, (256, 1, 1), GRID_LIMIT, Affine(0)
    )
    source = export_terms(".version 9.0\n", [binding]) + "\nCheck mfs : list (string * binding).\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Export.v"
        path.write_text(source)
        result = subprocess.run(
            ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
        )
    assert result.returncode == 0, result.stderr


class IntegerSubclass(int):
    pass


class IntegerEnum(IntEnum):
    ONE = 1


@pytest.mark.parametrize("value", [True, 1.0, "1", b"1", None, IntegerSubclass(1), IntegerEnum.ONE])
def test_numeric_terms_require_builtin_integers(value):
    binding = Binding("kernel", (), (("p", Constant(value)),), (1, 1, 1), GRID_LIMIT, Affine(0))
    with pytest.raises(TypeError, match="Rocq integer fields must be integers"):
        export_terms("", [binding])


@pytest.mark.parametrize(
    "parameter",
    [
        Constant(cast(Any, "1")),
        Bounded(cast(Any, 0.5), 2),
        Bounded(0, cast(Any, 2.5)),
        Size(Affine(cast(Any, "1"))),
        Size(Affine(0, cast(Any, ((0, 1.5),)))),
        Size(Affine(0, cast(Any, ((0.5, 1),)))),
        Base(Span(Direction.READ, Affine(cast(Any, "4")), 2, None, False)),
        Base(Span(Direction.READ, Affine(4), cast(Any, 2.0), None, False)),
        Base(
            Span(
                Direction.READ,
                Affine(4),
                2,
                IndexRange(ElementType(cast(Any, 4.0), True), Affine(0), Affine(1)),
                False,
            )
        ),
        Base(
            Span(
                Direction.READ,
                Affine(4),
                2,
                IndexRange(ElementType(4, True), Affine(cast(Any, "0")), Affine(1)),
                False,
            )
        ),
        Base(
            Span(
                Direction.READ,
                Affine(4),
                2,
                IndexRange(ElementType(4, True), Affine(0), Affine(cast(Any, "1"))),
                False,
            )
        ),
    ],
)
def test_nested_parameter_numbers_are_validated(parameter):
    binding = Binding(
        "kernel",
        (sig.SizeSymbol("n", 1, 16),),
        (("p", parameter),),
        (1, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    with pytest.raises(TypeError, match="Rocq integer fields must be integers"):
        binding_term(binding)


@pytest.mark.parametrize(
    "changes",
    [{"block": (1.0, 1, 1)}, {"grid": (1.0, 1, 1)}, {"shared": Affine(cast(Any, 0.5))}],
)
def test_launch_numbers_are_validated(changes):
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    with pytest.raises(TypeError, match="Rocq integer fields must be integers"):
        export_query("", [replace(binding, **changes)])


@pytest.mark.parametrize("signed", [0, 1, "false", None])
def test_index_signedness_is_not_coerced(signed):
    index = IndexRange(ElementType(4, signed), Affine(0), Affine(1))
    parameter = Base(Span(Direction.READ, Affine(4), 2, index, False))
    binding = Binding("kernel", (), (("p", parameter),), (1, 1, 1), GRID_LIMIT, Affine(0))
    with pytest.raises(TypeError, match="index signedness must be a boolean"):
        binding_term(binding)


def test_integer_terms_preserve_sign_and_arbitrary_precision():
    values = (0, -1, 2**200, -(2**200))
    parameters = tuple((f"p{i}", Constant(value)) for i, value in enumerate(values))
    binding = Binding("kernel", (), parameters, (1, 1, 1), GRID_LIMIT, Affine(0))
    term = binding_term(binding)
    for value in values:
        assert f"PConst ({value})%Z" in term
