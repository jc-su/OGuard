"""A launch-sized shared allocation must cover the original pattern accesses.

The gate decides each text with the zeroing prologue that `Executor.register`
puts there (design decision 46), so the texts are decided as the executor
would load them.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed
from opgate.signature import Affine


@pytest.mark.parametrize("toolchain", ["numba", "nvcc"])
@pytest.mark.parametrize("allocation", ["exact", "one_short", "zero"])
@pytest.mark.slow
def test_original_pattern_accesses_require_the_declared_allocation(toolchain, allocation):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29x.json"
    text = read_artifact(archive, f"sonnet/string_contains/{toolchain}/0", "ptx")
    binding = binding_of("sonnet", "string_contains", toolchain, text)
    assert binding.shared == Affine.symbol(2)
    assert binding.sizes[2].maximum == 49152
    if allocation == "one_short":
        sizes = (*binding.sizes[:2], replace(binding.sizes[2], minimum=1))
        binding = replace(binding, sizes=sizes, shared=binding.shared.plus(-1))
    elif allocation == "zero":
        binding = replace(binding, shared=Affine(0))
    # Underallocated bindings are static controls and never launch on GPU.
    result = verify(zeroed(text, binding), [binding], timeout=180)
    if allocation == "exact":
        assert result == Accepted()
    else:
        assert isinstance(result, Refused)
        assert not result.reason.startswith(TIMEOUT_REASON)
