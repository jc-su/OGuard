"""Explicit signed guards constrain arbitrary 64-bit loaded indices."""

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction

BINDING = Binding(
    "kernel",
    (),
    (
        ("input", Base(Span(Direction.READ, Affine(8), 8, None, False))),
        ("output", Base(Span(Direction.WRITE, Affine(17), 8, None, False))),
    ),
    (1, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def kernel(lower, upper, conjunction=False):
    pred = "and" if conjunction else "or"
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 input, .param .u64 output) {{
.reg .pred %p<4>;
.reg .b64 %rd<5>;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.global.u64 %rd3, [%rd1];
setp.lt.s64 %p1, %rd3, {lower};
setp.gt.s64 %p2, %rd3, {upper};
{pred}.pred %p3, %p1, %p2;
@%p3 bra End;
add.u64 %rd4, %rd2, %rd3;
st.global.u8 [%rd4], 1;
End:
ret;
}}
"""


@pytest.mark.parametrize(
    "lower,upper,conjunction,accepted",
    [(0, 16, False, True), (-1, 16, False, False), (0, 17, False, False), (0, 16, True, False)],
)
def test_signed_guards_are_needed_on_both_sides(lower, upper, conjunction, accepted):
    result = verify(kernel(lower, upper, conjunction), [BINDING], timeout=180)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result


@pytest.mark.parametrize(
    "variant", ["original", "negative-one-key", "one-past-max", "missing-lower-guard"]
)
def test_current_flag_kernel_requires_both_index_guards(variant):
    from eval.corpus.adapter import binding_of, directory_of

    model, task = "gpt-5.5", "q21_cuda_q21_set_order_flags_kernel"
    text = (directory_of(model, task) / "kernel.O1.ptx").read_text()
    binding = binding_of(model, task, "nvcc", text)
    changes = {
        "negative-one-key": ("setp.lt.s64 \t%p2, %rd2, 0;", "setp.lt.s64 \t%p2, %rd2, -1;"),
        "one-past-max": (
            "setp.gt.s64 \t%p3, %rd2, %rd4;",
            "add.s64 %rd4, %rd4, 1;\nsetp.gt.s64 \t%p3, %rd2, %rd4;",
        ),
        "missing-lower-guard": ("or.pred  \t%p4, %p2, %p3;", "mov.pred %p4, %p3;"),
    }
    if variant != "original":
        # Static controls only; modified kernels are never executed.
        before, after = changes[variant]
        assert text.count(before) == 1
        text = text.replace(before, after)
    result = verify(text, [binding], timeout=180)
    if variant == "original":
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result
