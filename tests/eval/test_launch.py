"""Reviewed launch allocations reach both archived and recompiled operators."""

import json

import pytest

from eval.corpus.adapter import SYNTHESIZED, binding_of, claim_of, declared_launch, directory_of
from eval.corpus.sources import operator_of
from opgate.gate import Refused, verify, zeroed
from opgate.signature import Affine, Between, Scalar, Signature


def test_minmax_shared_allocation_reaches_both_compilation_paths():
    text = (directory_of("haiku", "min_max") / "kernel.O1.ptx").read_text()
    archived = binding_of("haiku", "min_max", "nvcc", text)
    operator = operator_of("haiku", "min_max", "nvcc")
    binding = operator.binding(claim_of("haiku", "min_max")[1])
    assert archived.shared == operator.shared == binding.shared == Affine(2 * 256 * 8)
    decision = verify(zeroed(operator.text, binding), [binding], timeout=30)
    # The safety proof holds; the termination check names the block reduction,
    # which halves a signed count as (s + (s >> 31)) >> 1, and the kernel's
    # atomicMin and atomicMax on doubles retry a compare-and-swap that the model
    # lets fail on every try.
    assert isinstance(decision, Refused), decision
    assert "the kernel is not proved to end" in decision.reason, decision


@pytest.mark.parametrize("toolchain", ["numba", "nvcc"])
def test_rolling_allocation_covers_every_declared_window(toolchain):
    signature, _ = claim_of("sonnet", "rolling_sum")
    window = next(a for a in signature.arguments if a.name == "window")
    assert isinstance(window, Scalar) and isinstance(window.bound, Between)
    assert window.bound.upper == 1024
    directory = directory_of("sonnet", "rolling_sum")
    _, shared, block, grid = declared_launch(directory, toolchain, signature, SYNTHESIZED.block)
    assert (shared, block) == (Affine((256 + 1024 - 1) * 8), (256, 1, 1))


@pytest.mark.parametrize("value", [-1, True, "4096", None])
def test_malformed_launch_allocation_fails_immediately(tmp_path, value):
    (tmp_path / "launch.json").write_text(
        json.dumps({"nvcc": {"shared": {"constant": value, "terms": {}}, "size_maxima": {}}})
    )
    with pytest.raises(ValueError, match="shared"):
        declared_launch(tmp_path, "nvcc", Signature((), ()), SYNTHESIZED.block)


def test_a_collection_without_a_fixed_block_needs_a_declared_one(tmp_path):
    with pytest.raises(ValueError, match="block"):
        declared_launch(tmp_path, "nvcc", Signature((), ()), None)
    shared = {"constant": 0, "terms": {}}
    (tmp_path / "launch.json").write_text(
        json.dumps({"nvcc": {"block": [128, 1, 1], "shared": shared, "size_maxima": {}}})
    )
    assert declared_launch(tmp_path, "nvcc", Signature((), ()), None)[2] == (128, 1, 1)
    with pytest.raises(ValueError, match="fields"):
        declared_launch(tmp_path, "nvcc", Signature((), ()), SYNTHESIZED.block)
    (tmp_path / "launch.json").write_text(
        json.dumps({"nvcc": {"block": [2048, 1, 1], "shared": shared, "size_maxima": {}}})
    )
    with pytest.raises(ValueError, match="not a block"):
        declared_launch(tmp_path, "nvcc", Signature((), ()), None)
