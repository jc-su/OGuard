"""Nondefault guarded loads preserve their original bounds predicate."""

import pytest

from opgate.binding import Base, Binding, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

B = Binding(
    "kernel",
    (SizeSymbol("n", 0, 128),),
    (
        ("flags", Base(Span(Direction.READ, Affine(0, ((0, 8),)), 8, None, False))),
        ("values", Base(Span(Direction.READ, Affine(0, ((0, 4),)), 4, None, False))),
        ("n", Size(Affine.symbol(0))),
    ),
    (128, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def ptx(
    width=16,
    default=0,
    compare="ne",
    threshold=0,
    mask=False,
    negated=False,
    overwrite=None,
    past_end=False,
):
    cmp = "ge" if negated else "lt"
    loadguard = "!%p1" if negated else "%p1"
    extra = {
        "flag": "mov.b" + str(width) + " %h1, 1;",
        "predicate": "mov.pred %p1, 0;",
        "row": "add.u32 %r2, %r2, 1;",
    }.get(overwrite, "")
    maskline = f"and.b{width} %h2, %h1, 255;" if mask else f"mov.b{width} %h2, %h1;"
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 flags,.param .u64 values,.param .u32 n) {{
.reg .b64 %rd<5>;
.reg .b32 %r<5>;
.reg .b{width} %h<3>;
.reg .pred %p<3>;
ld.param.u64 %rd1,[flags];
ld.param.u64 %rd2,[values];
ld.param.u32 %r1,[n];
mov.u32 %r2,%tid.x;
setp.{cmp}.u32 %p1,%r2,%r1;
mad.wide.u32 %rd3,%r2,8,%rd1;
mov.b{width} %h1,{default};
@{loadguard} ld.global.b{width} %h1,[%rd3];
{extra}
{maskline}
setp.{compare}.u{width} %p2,%h2,{threshold};
mad.wide.u32 %rd4,%r2,4,%rd2;
@%p2 ld.global.u32 %r3,[%rd4+{4 if past_end else 0}];
ret;
}}"""


CASES = [
    ("w16_neg0_mask0", {"width": 16, "negated": False, "mask": False}, True),
    ("w16_neg0_mask1", {"width": 16, "negated": False, "mask": True}, True),
    ("w16_neg1_mask0", {"width": 16, "negated": True, "mask": False}, True),
    ("w16_neg1_mask1", {"width": 16, "negated": True, "mask": True}, True),
    ("w16_default7_gt", {"width": 16, "default": 7, "compare": "gt", "threshold": 7}, True),
    ("w16_default7_lt", {"width": 16, "default": 7, "compare": "lt", "threshold": 7}, True),
    (
        "w16_default65535_ne",
        {"width": 16, "default": 65535, "compare": "ne", "threshold": 65535},
        True,
    ),
    ("w16_default_allowed", {"width": 16, "compare": "eq"}, False),
    ("w16_flag_overwritten", {"width": 16, "overwrite": "flag"}, False),
    ("w16_row_overwritten", {"width": 16, "overwrite": "row"}, False),
    ("w16_past_end", {"width": 16, "past_end": True}, False),
    ("w32_neg0_mask0", {"width": 32, "negated": False, "mask": False}, True),
    ("w32_neg0_mask1", {"width": 32, "negated": False, "mask": True}, True),
    ("w32_neg1_mask0", {"width": 32, "negated": True, "mask": False}, True),
    ("w32_neg1_mask1", {"width": 32, "negated": True, "mask": True}, True),
    ("w32_default7_gt", {"width": 32, "default": 7, "compare": "gt", "threshold": 7}, True),
    ("w32_default7_lt", {"width": 32, "default": 7, "compare": "lt", "threshold": 7}, True),
    (
        "w32_default4294967295_ne",
        {"width": 32, "default": 4294967295, "compare": "ne", "threshold": 4294967295},
        True,
    ),
    ("w32_default_allowed", {"width": 32, "compare": "eq"}, False),
    ("w32_flag_overwritten", {"width": 32, "overwrite": "flag"}, False),
    ("w32_row_overwritten", {"width": 32, "overwrite": "row"}, False),
    ("w32_past_end", {"width": 32, "past_end": True}, False),
    ("w64_neg0_mask0", {"width": 64, "negated": False, "mask": False}, True),
    ("w64_neg0_mask1", {"width": 64, "negated": False, "mask": True}, True),
    ("w64_neg1_mask0", {"width": 64, "negated": True, "mask": False}, True),
    ("w64_neg1_mask1", {"width": 64, "negated": True, "mask": True}, True),
    ("w64_default7_gt", {"width": 64, "default": 7, "compare": "gt", "threshold": 7}, True),
    ("w64_default7_lt", {"width": 64, "default": 7, "compare": "lt", "threshold": 7}, True),
    ("w64_default_allowed", {"width": 64, "compare": "eq"}, False),
    ("w64_flag_overwritten", {"width": 64, "overwrite": "flag"}, False),
    ("w64_row_overwritten", {"width": 64, "overwrite": "row"}, False),
    ("w64_past_end", {"width": 64, "past_end": True}, False),
    ("predicate_overwritten_past_end", {"overwrite": "predicate", "past_end": True}, False),
]


@pytest.mark.parametrize("label,options,accepted", CASES, ids=[c[0] for c in CASES])
def test_conditional_load_provenance(label, options, accepted):
    # Refused variants are static proof checks; none execute on a GPU.
    decision = verify(ptx(**options), [B], timeout=30)
    if accepted:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
