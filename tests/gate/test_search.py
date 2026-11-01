"""A complete search loop keeps its midpoint and cursor bounds.

All variants compute the same binary search. Signed/unsigned right shift
agree on its nonnegative dividend; reversing the comparison and both select
operands preserves the cursor updates. The proof must tolerate either form.
"""

from dataclasses import replace

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.binding import Bounded
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed
from opgate.signature import Affine


@pytest.mark.parametrize(
    "toolchain,filename", [("numba", "numba.O1.ptx"), ("nvcc", "kernel.O1.ptx")]
)
@pytest.mark.slow
def test_string_search_keeps_the_predicate_when_its_cursor_advances(toolchain, filename):
    text = (directory_of("fable", "string_contains") / filename).read_text()
    binding = binding_of("fable", "string_contains", toolchain, text)
    assert verify(zeroed(text, binding), [binding], timeout=90) == Accepted()


@pytest.mark.parametrize("reverse_select", [False, True])
@pytest.mark.slow
def test_binary_search_preserves_a_copied_midpoint_bound(reverse_select):
    text = (directory_of("haiku", "sorted_contains") / "numba.O1.ptx").read_text()
    if reverse_select:
        assert "setp.lt.s64 \t%p4" in text
        assert "%rd46, %rd40, %rd46, %p4" in text
        assert "%rd47, %rd47, %rd5, %p4" in text
        text = text.replace("setp.lt.s64 \t%p4", "setp.ge.s64 \t%p4")
        text = text.replace("%rd46, %rd40, %rd46, %p4", "%rd46, %rd46, %rd40, %p4")
        text = text.replace("%rd47, %rd47, %rd5, %p4", "%rd47, %rd5, %rd47, %p4")
    binding = binding_of("haiku", "sorted_contains", "numba", text)
    assert verify(zeroed(text, binding), [binding], timeout=90) == Accepted()


@pytest.mark.parametrize("reverse_updates", [False, True])
def test_rolling_cursor_crosses_unsigned_zero(reverse_updates):
    text = (directory_of("haiku", "rolling_sum") / "kernel.O1.ptx").read_text()
    if reverse_updates:
        updates = "add.s64 \t%rd20, %rd20, 8;\n\tadd.s64 \t%rd21, %rd21, 1;"
        assert updates in text
        text = text.replace(updates, "add.s64 \t%rd21, %rd21, 1;\n\tadd.s64 \t%rd20, %rd20, 8;")
    binding = binding_of("haiku", "rolling_sum", "nvcc", text)
    assert verify(zeroed(text, binding), [binding], timeout=90) == Accepted()


@pytest.mark.parametrize("early_cursor", [False, True])
def test_shared_tile_bounds_use_the_simpler_loop_cursor(early_cursor):
    text = (directory_of("sonnet", "rolling_sum") / "kernel.O1.ptx").read_text()
    if early_cursor:
        initialize = "\tmov.u64 \t%rd32, %rd4;\n"
        before = "\tsub.s64 \t%rd30, %rd23, %rd20;\n"
        assert text.count(initialize) == 1 and text.count(before) == 1
        text = text.replace(initialize, "").replace(before, initialize + before)
    binding = binding_of("sonnet", "rolling_sum", "nvcc", text)
    assert verify(zeroed(text, binding), [binding], timeout=90) == Accepted()


@pytest.mark.parametrize("window_limit", [37, 513, 1024])
@pytest.mark.parametrize("short_allocation", [False, True])
def test_numba_shared_window_keeps_its_launch_bound_after_widening(window_limit, short_allocation):
    text = (directory_of("sonnet", "rolling_sum") / "numba.O1.ptx").read_text()
    binding = binding_of("sonnet", "rolling_sum", "numba", text)
    assert sum(p == Bounded(1, 1024) for _, p in binding.parameters) == 1
    binding = replace(
        binding,
        parameters=tuple(
            (name, Bounded(1, window_limit) if p == Bounded(1, 1024) else p)
            for name, p in binding.parameters
        ),
        shared=Affine((256 + window_limit - 1) * 8 - (8 if short_allocation else 0)),
    )
    decision = verify(zeroed(text, binding), [binding], timeout=90)
    if short_allocation:
        assert isinstance(decision, Refused)
        assert not decision.reason.startswith(TIMEOUT_REASON)
    else:
        assert decision == Accepted()


@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("reverse_select", [False, True])
def test_binary_search_with_either_shift_and_select_order(signed, reverse_select):
    text = (directory_of("fable", "sorted_contains") / "kernel.O1.ptx").read_text()
    if not signed:
        text = text.replace("shr.s64", "shr.u64")
    if reverse_select:
        text = text.replace("setp.lt.s64 \t%p3", "setp.ge.s64 \t%p3")
        text = text.replace("%rd34, %rd26, %rd34, %p3", "%rd34, %rd34, %rd26, %p3")
        text = text.replace("%rd32, %rd32, %rd22, %p3", "%rd32, %rd22, %rd32, %p3")
    binding = binding_of("fable", "sorted_contains", "nvcc", text)
    decision = verify(zeroed(text, binding), [binding], timeout=60)
    assert isinstance(decision, Accepted), decision
