"""Triton's counted loop over a reduction, and the sign extension of a 64-bit sum.

`for j in range(0, longest)` with `longest = tl.max(...)` exits by
`setp.ne` on the counter, and the byte load `chars + (begin + j)` sign-extends
the low 32 bits of the 64-bit sum `zext(begin) + j`. The counter keeps
`longest - j >= 1` at the loop head, and the check reads the sign extension
as the sum where the access's mask `j < length` keeps the sum below the end
offset. Variants that break either fact are refused.
"""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed

MODEL = "fable"
TASK = "string_equal"
TOOLCHAIN = "triton"

LOAD = "@%p4 ld.global.b8 { %rs1 }, [ %rd19 + 0 ];"
STEP = "add.s64 \t%rd26, %rd26, 1;"
MASK = "and.pred \t%p4, %p8, %p5;"


def _text() -> str:
    return (directory_of(MODEL, TASK) / f"{TOOLCHAIN}.ptx").read_text()


def _decide(text: str) -> Accepted | Refused:
    binding = binding_of(MODEL, TASK, TOOLCHAIN, text)
    return verify(zeroed(text, binding), [binding], timeout=120)


@pytest.mark.slow
def test_the_counted_loop_over_a_reduction_is_accepted() -> None:
    assert _decide(_text()) == Accepted()


@pytest.mark.parametrize(
    ("before", "after"),
    [
        # one byte past the end offset, which the mask j < length allows
        (LOAD, LOAD.replace("+ 0", "+ 1")),
        # a counter that steps by 2 can pass the bound and wrap
        (STEP, STEP.replace(", 1;", ", 2;")),
        # a load whose mask no longer bounds j by the string's length
        (MASK, MASK.replace("%p5", "%p8")),
    ],
)
@pytest.mark.slow
def test_a_broken_bound_or_counter_is_refused(before: str, after: str) -> None:
    # Static negative controls; never launch the modified PTX.
    text = _text()
    assert text.count(before) == 1
    decision = _decide(text.replace(before, after))
    assert isinstance(decision, Refused), decision
    assert not decision.reason.startswith(TIMEOUT_REASON), decision


@pytest.mark.slow
@pytest.mark.parametrize("variant", [5, 8])
def test_unrolled_key_comparisons_keep_their_acceptance(variant: int) -> None:
    # Projecting every 64-bit key comparison made these previously accepted
    # inputs exceed the common deadline. They have no counted-loop obligation.
    model = "qwen3.5-397b"
    task = f"q16_triton__count_distinct_per_group_kernel_{variant}"
    text = (directory_of(model, task) / "triton.ptx").read_text()
    binding = binding_of(model, task, "triton", text)
    assert verify(zeroed(text, binding), [binding], timeout=180) == Accepted()
