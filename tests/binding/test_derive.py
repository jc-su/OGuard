"""The derivation of a binding from a signature and a calling convention."""

import pytest

from opgate import signature as sig
from opgate.binding import (
    BASE_ALIGN,
    GRID_LIMIT,
    Base,
    Bounded,
    Constant,
    Convention,
    Direction,
    ElementType,
    Size,
    Zero,
    derive_binding,
)
from opgate.signature import Affine

GATHER = sig.Signature(
    sizes=(sig.SizeSymbol("n"), sig.SizeSymbol("m")),
    arguments=(
        sig.Span("col", "col", Direction.READ, Affine.symbol(0)),
        sig.IndexIn("sel", "sel", Affine.symbol(1), "col"),
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(1)),
        sig.Scalar("m", sig.SizeOf(Affine.symbol(1))),
    ),
)
ELEMENTS = {"col": ElementType(8, False), "sel": ElementType(4, True), "out": ElementType(8, False)}


def test_c_convention_is_one_parameter_per_argument() -> None:
    binding = derive_binding(
        GATHER,
        ELEMENTS,
        Convention.C,
        "k",
        ("p0", "p1", "p2", "p3"),
        (256, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    names = [name for name, _ in binding.parameters]
    assert names == ["p0", "p1", "p2", "p3"]
    col, sel, out = (span for _, span in binding.bases)
    assert col.extent == Affine(0, ((0, 8),))
    assert col.index is None and col.align == BASE_ALIGN
    assert sel.index is not None
    assert sel.index.lo == Affine(0) and sel.index.hi == Affine(-1, ((0, 1),))
    assert sel.direction is Direction.READ and out.direction is Direction.WRITE
    assert binding.parameters[3][1] == Size(Affine.symbol(1))


def test_numba_convention_is_seven_parameters_per_array() -> None:
    names = tuple(f"p{i}" for i in range(3 * 7 + 1))
    binding = derive_binding(
        GATHER, ELEMENTS, Convention.NUMBA, "k", names, (256, 1, 1), GRID_LIMIT, Affine(0)
    )
    first = [p for _, p in binding.parameters[:7]]
    assert first[0] == Zero() and first[1] == Zero()
    assert first[2] == Size(Affine.symbol(0)) and first[5] == Size(Affine.symbol(0))
    assert first[3] == Constant(8) and first[6] == Constant(8)
    assert isinstance(first[4], Base)


def test_triton_convention_appends_two_zero_pointers() -> None:
    names = ("p0", "p1", "p2", "p3", "global_scratch", "profile_scratch")
    binding = derive_binding(
        GATHER, ELEMENTS, Convention.TRITON, "k", names, (128, 1, 1), GRID_LIMIT, Affine(4096)
    )
    assert [p for _, p in binding.parameters[4:]] == [Zero(), Zero()]
    assert binding.shared == Affine(4096)
    assert binding.parameters[3][1] == Size(Affine.symbol(1))


def test_offsets_reach_the_end_of_their_target() -> None:
    signature = sig.Signature(
        sizes=(sig.SizeSymbol("n"), sig.SizeSymbol("chars")),
        arguments=(
            sig.Offsets("offsets", "offsets", Affine.symbol(0).plus(1), "chars"),
            sig.Span("chars", "chars", Direction.READ, Affine.symbol(1)),
        ),
    )
    elements = {"offsets": ElementType(4, True), "chars": ElementType(1, False)}
    binding = derive_binding(
        signature, elements, Convention.C, "k", ("a", "b"), (256, 1, 1), GRID_LIMIT, Affine(0)
    )
    offsets = binding.bases[0][1]
    assert offsets.extent == Affine(4, ((0, 4),))
    assert offsets.index is not None and offsets.index.hi == Affine.symbol(1)


def test_a_bounded_scalar_and_a_parameter_count_mismatch() -> None:
    signature = sig.Signature((sig.SizeSymbol("n"),), (sig.Scalar("shift", sig.Between(0, 24)),))
    binding = derive_binding(
        signature, {}, Convention.C, "k", ("s",), (256, 1, 1), GRID_LIMIT, Affine(0)
    )
    assert binding.parameters == (("s", Bounded(0, 24)),)
    with pytest.raises(ValueError, match="PTX parameters"):
        derive_binding(
            signature, {}, Convention.C, "k", ("s", "t"), (256, 1, 1), GRID_LIMIT, Affine(0)
        )


def test_a_signature_checks_its_symbols_and_targets() -> None:
    n = (sig.SizeSymbol("n"),)
    with pytest.raises(ValueError, match="size symbol"):
        sig.Signature(n, (sig.Span("a", "a", Direction.READ, Affine.symbol(1)),))
    with pytest.raises(ValueError, match="targets"):
        sig.Signature(n, (sig.IndexIn("a", "a", Affine.symbol(0), "b"),))
    with pytest.raises(ValueError, match="duplicate size"):
        sig.Signature(n + n, ())
    with pytest.raises(ValueError, match="bounds"):
        sig.SizeSymbol("n", -1)


def test_a_size_minimum_reaches_the_binding() -> None:
    signature = sig.Signature(
        (sig.SizeSymbol("capacity", 1),),
        (sig.Span("table", "table", Direction.READ_WRITE, Affine.symbol(0)),),
    )
    binding = derive_binding(
        signature,
        {"table": ElementType(8, True)},
        Convention.C,
        "k",
        ("t",),
        (256, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    assert binding.sizes == (sig.SizeSymbol("capacity", 1),)


def test_an_element_type_and_a_bounded_parameter_check_their_values() -> None:
    for width in (0, 3, 16):
        with pytest.raises(ValueError, match=f"an element of {width} bytes"):
            # a width outside the declared type, as untyped callers such as JSON parsers pass
            ElementType(width, False)  # ty: ignore[invalid-argument-type]
    assert Bounded(5, 5) == Bounded(5, 5)
    with pytest.raises(ValueError, match="empty interval"):
        Bounded(1, 0)
