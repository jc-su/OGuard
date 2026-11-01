"""Clamped window offsets retain their relation to the launch position."""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine


@pytest.mark.parametrize("minimum", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("short_allocation", [False, True])
def test_clamped_window_keeps_its_bound_relative_to_start(minimum, in_place, short_allocation):
    signature = sig.Signature(
        sizes=(),
        arguments=(
            sig.Span("out", "out", sig.Direction.WRITE, sig.Affine(63 if short_allocation else 64)),
            sig.Scalar("start", sig.Between(0, 2**30)),
            sig.Scalar("window", sig.Between(1, 64)),
        ),
    )
    binding = derive_binding(
        signature,
        {"out": ElementType(8, True)},
        Convention.C,
        "kernel",
        ("out", "start", "window"),
        (1, 1, 1),
        GRID_LIMIT,
        Affine(0),
    )
    clamp = (
        "add.s64 %rd4, %rd2, %rd3; add.s64 %rd4, %rd4, -1; "
        "min.s64 %rd5, %rd4, 1073741824; sub.s64 %rd6, %rd5, %rd2;"
        if minimum
        else "sub.s64 %rd4, %rd2, %rd3; add.s64 %rd4, %rd4, 1; "
        "max.s64 %rd5, %rd4, 0; sub.s64 %rd6, %rd2, %rd5;"
    )
    if in_place:
        clamp = clamp.replace("%rd5", "%rd4")
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 out, .param .u64 start, .param .u64 window) {{
    .reg .b64 %rd<9>;
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [start];
    ld.param.u64 %rd3, [window];
    {clamp}
    shl.b64 %rd7, %rd6, 3;
    add.s64 %rd8, %rd1, %rd7;
    st.global.u64 [%rd8], %rd6;
    ret;
}}"""
    decision = verify(text, [binding], timeout=30)
    if short_allocation:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON)
    else:
        assert decision == Accepted()
