"""Relations between loop registers that the gate proves (design decision 41).

A binary search keeps 0 <= lo <= mid < hi <= n, or lo <= mid <= hi <= n - 1
for a closed one, through the selections that update its bounds; a search's
result that is -1 or a position keeps its bound by n across the merge of
the two; a counter that leaves its loop when it equals K is widened to K - 1;
a counter and a pointer that a loop steps in proportion keep their relation.
Each variant that breaks the fact the access needs is refused.
"""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify


def _decide(
    model: str, task: str, toolchain: str, edits: tuple[tuple[str, str], ...] = ()
) -> Accepted | Refused:
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    for before, after in edits:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    return verify(text, [binding_of(model, task, toolchain, text)], timeout=170)


def _refused(decision: Accepted | Refused) -> None:
    assert isinstance(decision, Refused), decision
    assert not decision.reason.startswith(TIMEOUT_REASON), decision


HALF_OPEN = ("gemini-3.1-pro", "q21_core_cuda_q21_kernel", "nvcc")
CLOSED = ("gpt-oss-120b", "q22_core_cuda_agg_kernel", "nvcc")
SENTINEL = ("deepseek-v4-flash", "q17_cuda_compute_part_stats_kernel", "nvcc")
EQUALITY_EXIT = ("minimax-m2.5", "q16_core_triton__count_unique_kernel", "triton")
PROPORTION = ("haiku", "dictionary_decode", "triton")


@pytest.mark.slow
def test_a_half_open_search_keeps_its_midpoint_below_its_upper_end() -> None:
    assert _decide(*HALF_OPEN) == Accepted()


@pytest.mark.slow
def test_a_half_open_search_that_continues_at_lo_equal_hi_is_refused() -> None:
    # with lo = hi the midpoint is hi, which can be n
    _refused(
        _decide(
            *HALF_OPEN, (("setp.gt.s64 \t%p6, %rd47, %rd49;", "setp.ge.s64 \t%p6, %rd47, %rd49;"),)
        )
    )


@pytest.mark.slow
def test_a_closed_search_keeps_its_midpoint_inside_its_bounds() -> None:
    assert _decide(*CLOSED) == Accepted()


@pytest.mark.slow
def test_a_closed_search_from_hi_equal_n_is_refused() -> None:
    # hi starts at n rather than n - 1, so the midpoint can be n
    _refused(_decide(*CLOSED, (("add.s64 \t%rd40, %rd16, -1;", "add.s64 \t%rd40, %rd16, 0;"),)))


@pytest.mark.slow
def test_a_search_result_keeps_its_bound_across_the_sentinel_merge() -> None:
    assert _decide(*SENTINEL) == Accepted()


@pytest.mark.slow
def test_a_sentinel_that_passes_the_test_is_refused() -> None:
    # found = -1 passes found >= -1, and the atomic reaches sum_qty[-1]
    _refused(
        _decide(*SENTINEL, (("setp.lt.s64 \t%p11, %rd56, 0;", "setp.lt.s64 \t%p11, %rd56, -1;"),))
    )


def test_a_counter_that_leaves_at_a_constant_is_widened_below_it() -> None:
    assert _decide(*EQUALITY_EXIT) == Accepted()


def test_a_counter_without_the_row_guard_is_refused() -> None:
    # the counter stays below the block size, but the row pid * BLOCK + i can pass n
    _refused(_decide(*EQUALITY_EXIT, (("@%p1 bra \t$L__BB0_3;", "@%p2 bra \t$L__BB0_3;"),)))


def test_a_pointer_stepped_in_proportion_to_its_counter_is_accepted() -> None:
    assert _decide(*PROPORTION) == Accepted()


def test_a_pointer_stepped_out_of_proportion_is_refused() -> None:
    # the pointer moves two elements for each row that the guard lets through
    _refused(_decide(*PROPORTION, (("add.s64 \t%rd12, %rd12, 4;", "add.s64 \t%rd12, %rd12, 8;"),)))
