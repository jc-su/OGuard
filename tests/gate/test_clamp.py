"""Safe clamping retains a relation to a symbolic buffer length."""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("selection,reverse", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("past_end", [False, True])
def test_clamp_bound(signed, selection, reverse, in_place, past_end):
    signature = sig.Signature(
        sizes=(sig.SizeSymbol("n", 1),),
        arguments=(
            sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
            sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
        ),
    )
    binding = derive_binding(
        signature,
        {"out": ElementType(4, True)},
        Convention.C,
        "kernel",
        ("out", "n"),
        (256, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    kind = "s64" if signed else "u64"
    comparison = (
        f"setp.gt.{kind} %p1, %rd2, %rd3;" if reverse else f"setp.lt.{kind} %p1, %rd3, %rd2;"
    )
    clamp = (
        f"{comparison} selp.b64 %rd5, %rd3, %rd4, %p1;"
        if selection
        else f"min.{kind} %rd5, %rd3, %rd4;"
    )
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 out, .param .u64 n) {{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<9>;
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [n];
    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd3, %r1;
    add.s64 %rd4, %rd2, {0 if past_end else -1};
    {clamp}
    shl.b64 %rd6, %rd5, 2;
    add.s64 %rd7, %rd1, %rd6;
    st.global.u32 [%rd7], %r1;
    ret;
}}"""
    if in_place:
        text = text.replace("%rd5", "%rd3")
    decision = verify(text, [binding], timeout=30)
    if past_end:
        assert isinstance(decision, Refused), decision
    else:
        assert decision == Accepted()
