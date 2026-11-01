"""Signed selections retain only branch-proved orderings at each full width."""

import pytest

from opgate.binding import Base, Binding, ElementType, IndexRange, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

_BINDING64 = Binding(
    "kernel",
    (SizeSymbol("m", 1, 2**31 - 1),),
    (
        (
            "offsets",
            Base(
                Span(
                    Direction.READ,
                    Affine(8),
                    8,
                    IndexRange(ElementType(4, True), Affine(0), Affine.symbol(0)),
                    False,
                )
            ),
        ),
        ("chars", Base(Span(Direction.READ, Affine.symbol(0), 8, None, False))),
        ("output", Base(Span(Direction.WRITE, Affine(1), 1, None, False))),
        ("m", Size(Affine.symbol(0))),
    ),
    (1, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def _ptx64(cut=256, constant=0, extra=0, swapped=False):
    compare = "le" if swapped else "gt"
    alternatives = "%rd6, " + str(constant) if swapped else str(constant) + ", %rd6"
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 offsets,.param .u64 chars,.param .u64 output,.param .u64 m) {{
.reg .b64 %rd<11>;
.reg .b32 %r<4>;
.reg .pred %p<3>;
ld.param.u64 %rd1,[offsets];
ld.param.u64 %rd2,[chars];
ld.param.u64 %rd3,[output];
ld.global.u32 %r1,[%rd1];
ld.global.u32 %r2,[%rd1+4];
cvt.s64.s32 %rd4,%r1;
cvt.s64.s32 %rd5,%r2;
sub.s64 %rd6,%rd5,%rd4;
setp.{compare}.s64 %p1,%rd6,{cut};
selp.b64 %rd7,{alternatives},%p1;
mov.u64 %rd8,0;
setp.lt.s64 %p2,%rd8,%rd7;
add.s64 %rd9,%rd2,%rd4;
add.s64 %rd10,%rd9,%rd8;
mov.u32 %r3,0;
@%p2 ld.global.u8 %r3,[%rd10+{extra}];
st.global.u8 [%rd3],%r3;
ret;
}}"""


def _binding(width):
    if width == 64:
        return _BINDING64
    bits = width
    size = bits // 8
    return Binding(
        "kernel",
        (SizeSymbol("m", 1, 2 ** (bits - 1) - 1),),
        (
            (
                "offsets",
                Base(
                    Span(
                        Direction.READ,
                        Affine(2 * size),
                        size,
                        IndexRange(ElementType(size, True), Affine(0), Affine.symbol(0)),
                        False,
                    )
                ),
            ),
            ("chars", Base(Span(Direction.READ, Affine.symbol(0), 8, None, False))),
            ("output", Base(Span(Direction.WRITE, Affine(1), 1, None, False))),
            ("m", Size(Affine.symbol(0))),
        ),
        (1, 1, 1),
        (1, 1, 1),
        Affine(0),
    )


def _ptx(width, cut=256, constant=0, extra=0, swapped=False):
    if width == 64:
        return _ptx64(cut, constant, extra, swapped)
    compare = "le" if swapped else "gt"
    alternatives = "%s3, " + str(constant) if swapped else str(constant) + ", %s3"
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 offsets,.param .u64 chars,.param .u64 output,.param .u64 m) {{
.reg .b64 %rd<7>;
.reg .b{width} %s<6>;
.reg .b32 %r<2>;
.reg .pred %p<3>;
ld.param.u64 %rd1,[offsets];
ld.param.u64 %rd2,[chars];
ld.param.u64 %rd3,[output];
ld.global.u{width} %s1,[%rd1];
ld.global.u{width} %s2,[%rd1+{width // 8}];
sub.s{width} %s3,%s2,%s1;
setp.{compare}.s{width} %p1,%s3,{cut};
selp.b{width} %s4,{alternatives},%p1;
mov.u{width} %s5,0;
setp.lt.s{width} %p2,%s5,%s4;
cvt.s64.s{width} %rd4,%s1;
add.s64 %rd5,%rd2,%rd4;
mov.u32 %r1,0;
@%p2 ld.global.u8 %r1,[%rd5+{extra}];
st.global.u8 [%rd3],%r1;
ret;
}}"""


_CASES = [
    ("positive", {}, True),
    ("swapped", {"swapped": True}, True),
    ("zero_cut", {"cut": 0}, True),
    ("past_end", {"extra": 1}, False),
    ("constant_not_bounded", {"cut": -1, "constant": 1}, False),
    ("swapped_not_bounded", {"cut": -1, "constant": 1, "swapped": True}, False),
]


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("label,options,accepted", _CASES, ids=[case[0] for case in _CASES])
def test_signed_selection_preserves_only_proved_ordering(width, label, options, accepted):
    # Invalid boundary variants are static proof controls; never launch them.
    decision = verify(_ptx(width, **options), [_binding(width)], timeout=30)
    if accepted:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
