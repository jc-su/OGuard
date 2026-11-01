"""Descending counters keep each original rolling window inside its allocations."""

from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed


@pytest.mark.parametrize("change", [None, "tile", "suffix", "carry", "input_end", "output_end"])
@pytest.mark.slow
def test_descending_rolling_checks_all_boundaries(change):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29y.json"
    text = read_artifact(archive, "opus/rolling_sum/numba/0", "ptx")
    if change is not None:
        # Boundary controls are static checks only; never execute modified PTX.
        before, after = {
            "tile": ("_cudapy_smem_37[16384]", "_cudapy_smem_37[16368]"),
            "suffix": ("_cudapy_smem_38[8192]", "_cudapy_smem_38[8184]"),
            "carry": ("_cudapy_smem_39[2048]", "_cudapy_smem_39[2040]"),
            "input_end": ("ld.f64 \t%fd55, [%rd103];", "ld.f64 %fd55, [%rd103+8];"),
            "output_end": ("st.f64 \t[%rd329], %fd67;", "st.f64 [%rd329+8], %fd67;"),
        }[change]
        assert text.count(before) == 1
        text = text.replace(before, after)
    binding = binding_of("opus", "rolling_sum", "numba", text)
    result = verify(zeroed(text, binding), [binding], timeout=180)
    if change is None:
        assert result == Accepted()
    else:
        assert isinstance(result, Refused)
        assert not result.reason.startswith(TIMEOUT_REASON)
