"""The corpus adapter reads the lengths of `signature.json` into affine sizes."""

import pytest

from eval.corpus.adapter import (
    Assigned,
    RawArgument,
    Role,
    Roles,
    claim_of,
    collection_of,
    signature_of,
)
from opgate import signature as sig
from opgate.signature import Affine, Direction


def test_a_length_term_can_have_a_constant_factor():
    raws = (
        RawArgument("pairs", "input", "int64", "2 * n + 1"),
        RawArgument("out", "output", "int64", "n * 1 + 3 * m"),
        RawArgument("n", "scalar", "int64", "rows"),
    )
    roles = Roles(
        {"pairs": Role.READ, "out": Role.WRITE, "n": Role.SIZE},
        {"n": 0, "m": 0},
    )
    signature, _ = signature_of(raws, roles)
    assert signature.arguments[0] == sig.Span(
        "pairs", "pairs", Direction.READ, Affine(1, ((0, 2),))
    )
    assert signature.arguments[1] == sig.Span(
        "out", "out", Direction.WRITE, Affine(0, ((0, 1), (1, 3)))
    )


def test_a_length_the_adapter_does_not_read_is_an_error():
    raws = (RawArgument("pairs", "input", "int64", "n * n"),)
    with pytest.raises(ValueError, match="does not read"):
        signature_of(raws, Roles({"pairs": Role.READ}, {"n": 0}))


def test_each_model_belongs_to_one_collection():
    assert collection_of("sonnet").block == (256, 1, 1)
    assert collection_of("gpt-5.5").block is None
    with pytest.raises(ValueError):
        collection_of("nobody")


def test_a_derived_symbol_is_the_quotient_its_buffer_length_states():
    raws = (
        RawArgument("offsets", "inout", "int64", "n"),
        RawArgument("block_offsets", "input", "int64", "(n + 1023) / 1024"),
        RawArgument("n", "scalar", "int64", "rows"),
    )
    arguments: dict[str, Assigned] = {
        "offsets": Role.READ_WRITE,
        "block_offsets": Role.READ,
        "n": Role.SIZE,
    }
    symbols = {"n": 1, "block_offsets": 1}
    independent, _ = signature_of(raws, Roles(arguments, symbols))
    assert independent.derived_sizes == ()
    derived, _ = signature_of(raws, Roles(arguments, symbols, frozenset({"block_offsets"})))
    assert derived.derived_sizes == (sig.DerivedSize(1, Affine(1023, ((0, 1),)), 1024),)


def test_a_derived_symbol_needs_a_quotient_length():
    raws = (
        RawArgument("bitmap", "input", "uint8", "valid[n]"),
        RawArgument("n", "scalar", "int64", "rows"),
    )
    roles = Roles(
        {"bitmap": Role.READ, "n": Role.SIZE}, {"n": 0, "bitmap": 0}, frozenset({"bitmap"})
    )
    with pytest.raises(ValueError, match="not a quotient"):
        signature_of(raws, roles)


def test_the_sort_reads_back_the_identifiers_it_writes():
    # the DataKernelBench contract correction of 2026-10-03: the insertion sort
    # reads `sorted_ids[i]` and `sorted_ids[j]` after the copy loop writes them
    signature, _ = claim_of("gpt-5.5", "q05_core_cuda_q5_sort_kernel")
    spans = {a.name: a for a in signature.arguments if isinstance(a, sig.Span)}
    assert spans["sorted_ids"].direction is Direction.READ_WRITE
