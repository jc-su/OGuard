"""Equivalent guard orientations retain a nested search's projected bounds."""

from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify


@pytest.mark.parametrize("reverse_guards", [False, True])
@pytest.mark.parametrize("past_character_end", [False, True])
@pytest.mark.slow
def test_projection_keeps_nested_bounds_in_either_orientation(reverse_guards, past_character_end):
    # Preserve the actual measured compiler output, including its entry name.
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29i.json"
    text = read_artifact(archive, "haiku/string_contains/numba/0", "ptx")
    if reverse_guards:
        for before, after in (
            ("setp.gt.s64 \t%p15, %rd79, %rd6;", "setp.lt.s64 \t%p15, %rd6, %rd79;"),
            ("setp.le.s64 \t%p20, %rd7, %rd4;", "setp.ge.s64 \t%p20, %rd4, %rd7;"),
        ):
            assert text.count(before) == 1
            text = text.replace(before, after)
    if past_character_end:
        # This is a static negative control; never launch the modified PTX.
        before = "ld.u8 \t%rs5, [%rd53];"
        assert text.count(before) == 1
        text = text.replace(before, "ld.u8 \t%rs5, [%rd53+1];")
    binding = binding_of("haiku", "string_contains", "numba", text)
    decision = verify(text, [binding], timeout=90)
    if past_character_end:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
    else:
        assert decision == Accepted()
