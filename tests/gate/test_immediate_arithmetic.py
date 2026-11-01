"""Immediate arithmetic preserves modular bounds and shared-address alignment."""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.binding import Base, Binding, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction


def _kernel(body: str) -> str:
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .b64 %rd<5>;
.shared .align 8 .b8 tile[16384];
ld.param.u64 %rd1, [output];
mov.u32 %r1, %tid.x;
{body}
ret;
}}
"""


def _binding(extent: int) -> Binding:
    return Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(extent), 8, None, False))),),
        (256, 1, 1),
        (1, 1, 1),
        Affine(0),
    )


def _assert_decision(text: str, binding: Binding, accepted: bool) -> None:
    decision = verify(zeroed(text, binding), [binding], timeout=180)
    if accepted:
        assert decision == Accepted(), decision
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("constant,accepted", [(255, True), (254, False)])
def test_immediate_subtraction_respects_modular_underflow(width, alias, constant, accepted):
    # 255-tid is in [0,255]; 254-tid wraps at tid=255. Reusing the
    # source as the destination must read its old value.
    if width == 32:
        dst = "%r1" if alias else "%r2"
        body = f"sub.u32 {dst}, {constant}, %r1;\nmul.wide.u32 %rd2, {dst}, 8;"
    else:
        dst = "%rd2" if alias else "%rd3"
        body = f"cvt.u64.u32 %rd2, %r1;\nsub.u64 {dst}, {constant}, %rd2;\nshl.b64 %rd2, {dst}, 3;"
    body += "\nadd.u64 %rd4, %rd1, %rd2;\nst.global.u64 [%rd4], 0;"
    _assert_decision(_kernel(body), _binding(2048), accepted)


@pytest.mark.parametrize("immediate,accepted", [(14448, True), (14449, False), (16400, False)])
def test_immediate_xor_keeps_both_bounds_and_alignment(immediate, accepted):
    # 8*tid has low three bits zero. XOR 14448 stays below 16384 and
    # preserves those bits; 14449 breaks alignment and 16400 sets bit14.
    body = f"""shl.b32 %r2, %r1, 3;
xor.b32 %r3, %r2, {immediate};
mov.u32 %r4, tile;
add.u32 %r3, %r4, %r3;
st.shared.u64 [%r3], 0;"""
    _assert_decision(_kernel(body), _binding(16384), accepted)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["original", "bounds", "alignment", "inclusive-guard"])
def test_current_triton_rolling_arithmetic(variant):
    text = (directory_of("opus", "rolling_sum") / "triton.ptx").read_text()
    binding = binding_of("opus", "rolling_sum", "triton", text)
    changes = {
        "bounds": ("%r651, %r13, 14448;", "%r651, %r13, 16400;"),
        "alignment": ("%r651, %r13, 14448;", "%r651, %r13, 14449;"),
        "inclusive-guard": (
            "setp.lt.s64 \t%p24, %rd10, %rd13;",
            "setp.le.s64 \t%p24, %rd10, %rd13;",
        ),
    }
    if variant != "original":
        # Static controls only; modified kernels are never executed.
        before, after = changes[variant]
        assert text.count(before) == 1
        text = text.replace(before, after)
    _assert_decision(text, binding, variant == "original")
