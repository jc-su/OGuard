"""Nested search bounds survive loop recovery and 32-to-64-bit extension.

The gate proves both cursors' bounds and that both loops end: the outer
counter, written in the block before the latch that holds only the back
edge's `bra.uni`, is a counter incremented before the latch (DESIGN
decision 40). A read one byte past the string is refused for its bounds.
"""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify


@pytest.mark.parametrize("signed_extension", [True, False])
@pytest.mark.parametrize("past_character_end", [False, True])
@pytest.mark.slow
def test_nested_search_preserves_both_cursor_bounds(signed_extension, past_character_end):
    text = (directory_of("haiku", "string_contains") / "kernel.O1.ptx").read_text()
    conversion = "cvt.s64.s32 \t%rd17, %r15;"
    assert text.count(conversion) == 1
    if not signed_extension:
        text = text.replace(conversion, "cvt.u64.u32 \t%rd17, %r15;")
    if past_character_end:
        # Static negative control only: the loop's last legal byte is no
        # longer sufficient for the shifted access. Never execute this PTX.
        access = "ld.global.u8 \t%rs5, [%rd18];"
        assert text.count(access) == 1
        text = text.replace(access, "ld.global.u8 \t%rs5, [%rd18+1];")
    binding = binding_of("haiku", "string_contains", "nvcc", text)
    decision = verify(text, [binding], timeout=90)
    if past_character_end:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
        assert "the kernel is not proved to end" not in decision.reason, decision
    else:
        assert decision == Accepted(), decision
