"""Packed integer predicate bits retain their original bounds comparisons."""

import itertools

import pytest

from opgate.binding import Base, Binding, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

_BINDING = Binding(
    "kernel",
    (SizeSymbol("n", 0, 256),),
    (
        ("input", Base(Span(Direction.READ, Affine(0, ((0, 4),)), 4, None, False))),
        ("n", Size(Affine.symbol(0))),
    ),
    (32, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def _ptx(
    width,
    position=1,
    encoding=-1,
    inverted=False,
    wrong_bit=False,
    past_end=False,
    linear_mask=False,
):
    compare = "eq" if inverted else "ne"
    guard = "!%p3" if inverted else "%p3"
    bit = 0 if wrong_bit else position
    text = f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 input,.param .u32 n) {{
.reg .b64 %rd<4>;
.reg .b32 %r<9>;
.reg .b{width} %s<7>;
.reg .pred %p<4>;
ld.param.u64 %rd1,[input];
ld.param.u32 %r1,[n];
mov.u32 %r2,%tid.x;
add.u32 %r3,%r2,32;
setp.lt.u32 %p1,%r2,%r1;
setp.lt.u32 %p2,%r3,%r1;
selp.b{width} %s1,1,0,%p1;
selp.b{width} %s2,{encoding},0,%p2;
shl.b{width} %s3,%s2,{position};
or.b{width} %s4,%s1,%s3;
and.b{width} %s5,%s4,{2 ** (position + 1) - 1};
cvt.u32.u{width} %r4,%s5;
shr.u32 %r5,%r4,{bit};
and.b32 %r6,%r5,1;
setp.{compare}.u32 %p3,%r6,0;
mad.wide.u32 %rd2,%r3,4,%rd1;
@{guard} ld.global.u32 %r7,[%rd2+{4 if past_end else 0}];
ret;
}}"""
    if linear_mask:
        text = text.replace(f"selp.b{width} %s1,1,0,%p1;", f"selp.b{width} %s1,3,0,%p2;")
        text = text.replace(f"selp.b{width} %s2,{encoding},0,%p2;", f"selp.b{width} %s2,1,0,%p1;")
        text = text.replace(
            f"and.b{width} %s5,%s4,{2 ** (position + 1) - 1};", f"and.b{width} %s5,%s4,3;"
        )
        text = text.replace(f"shr.u32 %r5,%r4,{bit};", "shr.u32 %r5,%r4,0;")
    return text


def _cases():
    cases = []
    for width, position, encoding, inverted in itertools.product(
        (16, 32, 64), (1, 7), (-1, 1), (False, True)
    ):
        cases.append(
            (
                f"w{width}_p{position}_e{encoding}_inv{int(inverted)}",
                dict(width=width, position=position, encoding=encoding, inverted=inverted),
                True,
            )
        )
    for width in (16, 32, 64):
        for bad in ("wrong_bit", "past_end"):
            cases.append((f"w{width}_{bad}", dict(width=width, **{bad: True}), False))
    for width in (16, 32, 64):
        cases.append(
            (f"w{width}_linear_mask", dict(width=width, position=2, linear_mask=True), True)
        )
    return cases


@pytest.mark.parametrize("label,options,accepted", _cases(), ids=[case[0] for case in _cases()])
def test_packed_predicate_guards_keep_only_the_selected_bit(label, options, accepted):
    # Invalid extraction/address variants are static controls only.
    decision = verify(_ptx(**options), [_BINDING], timeout=30)
    if accepted:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
