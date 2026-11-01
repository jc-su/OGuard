"""The check's time on large kernels, and joins in the analysis (design decision 45).

The shared-value participation check finds the waiting sites of a position
in maps; the cached range layers stop at the first layer whose cached ranges
equal those of the layer below. Gemini's Triton Q1 kernel, with 575 waiting
sites in a body of 7,462 instructions, is decided inside a registration's
time limit, and a variant whose first shuffle names half of the warp is
refused. The analysis
allows a block's entry state LIMIT changes for each edge into it, so the loop
of gpt-5.5's CUDA C++ Q1 kernel, which joins the six cases of a switch,
reaches its fixpoint; a variant that adds to the sixth group when there are
five groups is refused.
"""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.gate import REGISTRATION_SECONDS, TIMEOUT_REASON, Accepted, Refused, verify, zeroed


def _decide(
    model: str, task: str, toolchain: str, edits: tuple[tuple[str, str], ...] = ()
) -> Accepted | Refused:
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    for before, after in edits:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    binding = binding_of(model, task, toolchain, text)
    return verify(zeroed(text, binding), [binding], timeout=REGISTRATION_SECONDS)


def _refused(decision: Accepted | Refused) -> None:
    assert isinstance(decision, Refused), decision
    assert not decision.reason.startswith(TIMEOUT_REASON), decision


WAITING_SITES = ("gemini-3.1-pro", "q01_triton_q1_kernel_5", "triton")
SWITCH_JOIN = ("gpt-5.5", "q01_cuda_q1_kernel6", "nvcc")


@pytest.mark.slow
def test_a_kernel_with_many_waiting_sites_is_decided_inside_the_limit() -> None:
    assert _decide(*WAITING_SITES) == Accepted()


@pytest.mark.slow
def test_a_shuffle_that_names_half_of_the_warp_is_refused() -> None:
    _refused(
        _decide(
            *WAITING_SITES,
            (
                (
                    "shfl.sync.bfly.b32 \t%r366, %r365, 16, 31, -1;",
                    "shfl.sync.bfly.b32 \t%r366, %r365, 16, 31, 65535;",
                ),
            ),
        )
    )


@pytest.mark.slow
def test_a_loop_that_joins_six_cases_reaches_its_fixpoint() -> None:
    assert _decide(*SWITCH_JOIN) == Accepted()


@pytest.mark.slow
def test_adding_to_the_sixth_group_of_five_is_refused() -> None:
    # the sixth group's atomics run when n_groups >= 5 instead of >= 6
    _refused(
        _decide(*SWITCH_JOIN, (("setp.lt.s32 \t%p42, %r22, 6;", "setp.lt.s32 \t%p42, %r22, 5;"),))
    )
