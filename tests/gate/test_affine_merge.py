"""Affine launch bounds survive midpoint, clamp, selection and exit joins."""

from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify


@pytest.mark.parametrize(
    "past_end",
    [
        None,
        "loop_a",
        "loop_b",
        "compare_a",
        "compare_b",
        "exit_a",
        "exit_b",
        "output_a",
        "output_b",
    ],
)
@pytest.mark.slow
def test_affine_merge_checks_each_memory_access(past_end):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29v.json"
    text = read_artifact(archive, "sonnet/merge_sorted/numba/0", "ptx")
    if past_end is not None:
        # These boundary controls are static only; they never execute on GPU.
        operands = {
            "loop_a": ("%rd46", "%rd39"),
            "loop_b": ("%rd45", "%rd44"),
            "compare_a": ("%rd58", "%rd51"),
            "compare_b": ("%rd57", "%rd56"),
            "exit_a": ("%rd64", "%rd63"),
            "exit_b": ("%rd75", "%rd74"),
            "output_a": ("%rd64", "%rd69"),
            "output_b": ("%rd75", "%rd80"),
        }
        value, address = operands[past_end]
        if past_end.startswith("output"):
            old = f"st.u64 \t[{address}], {value};"
            new = f"st.u64 [{address}+8], {value};"
        else:
            old = f"ld.u64 \t{value}, [{address}];"
            new = f"ld.u64 {value}, [{address}+8];"
        assert text.count(old) == 1
        text = text.replace(old, new)
    result = verify(text, [binding_of("sonnet", "merge_sorted", "numba", text)], timeout=180)
    if past_end is None:
        assert result == Accepted()
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result
