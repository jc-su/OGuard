"""Selections retain load guards while independent masked data stays unrelated."""

import pytest

from opgate.binding import Base, Binding, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

B = Binding(
    "kernel",
    (SizeSymbol("n", 1, 128),),
    (
        ("flags", Base(Span(Direction.READ, Affine(0, ((0, 8),)), 8, None, False))),
        ("values", Base(Span(Direction.READ, Affine(0, ((0, 4),)), 4, None, False))),
        ("n", Size(Affine.symbol(0))),
    ),
    (128, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def ptx(width, default_allowed=False, past_end=False):
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 flags,.param .u64 values,.param .u32 n) {{
.reg .b64 %rd<7>;
.reg .b32 %r<6>;
.reg .b{width} %h;
.reg .pred %p<5>;
ld.param.u64 %rd1,[flags];
ld.param.u64 %rd2,[values];
ld.param.u32 %r1,[n];
mov.u32 %r2,%tid.x;
setp.lt.u32 %p1,%r2,%r1;
mad.wide.u32 %rd3,%r2,8,%rd1;
mov.b{width} %h,0;
@%p1 ld.global.b{width} %h,[%rd3];
setp.gt.u32 %p3,%r1,1;
setp.gt.u32 %p4,%r1,2;
mov.b64 %rd5,0;
mov.b64 %rd6,0;
@%p3 ld.global.b64 %rd5,[%rd1];
@%p4 ld.global.b64 %rd6,[%rd1];
setp.{"eq" if default_allowed else "gt"}.u{width} %p2,%h,0;
selp.u32 %r4,%r2,0,%p2;
mad.wide.u32 %rd4,%r4,4,%rd2;
ld.global.u32 %r3,[%rd4+{4 if past_end else 0}];
ret;
}}"""


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("case", ["valid", "default_allowed", "past_end"])
def test_selection_guard_ignores_independent_masked_values(width, case):
    # Refusals are static controls, never GPU launches.
    decision = verify(ptx(width, case == "default_allowed", case == "past_end"), [B], timeout=30)
    if case == "valid":
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
