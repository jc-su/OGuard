"""Selections preserve signed index bounds after checking both alternatives."""

import pytest

from opgate.binding import Base, Binding, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

B = Binding(
    "kernel",
    (SizeSymbol("n", 1, 128),),
    (
        ("input", Base(Span(Direction.READ, Affine(8), 8, None, False))),
        ("output", Base(Span(Direction.WRITE, Affine.symbol(0), 1, None, False))),
        ("n", Size(Affine.symbol(0))),
    ),
    (1, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def ptx(width=64, inverted=False, default=0, missing=None, extra=0, overwrite=None):
    args = f"{default},%h1" if inverted else f"%h1,{default}"
    conj = "or.pred %p3,%p1,%p2;" if inverted else "and.pred %p3,%p1,%p2;"
    lo = "lt" if inverted else "ge"
    hi = "ge" if inverted else "lt"
    if missing == "lower":
        lower = "mov.pred %p1, " + ("0" if inverted else "1") + ";"
    else:
        lower = f"setp.{lo}.s{width} %p1,%h1,0;"
    if missing == "upper":
        upper = "mov.pred %p2, " + ("0" if inverted else "1") + ";"
    else:
        upper = f"setp.{hi}.s{width} %p2,%h1,%h2;"
    change = {"selected": f"mov.b{width} %h3,128;", "limit": "mov.u64 %rd3,128;"}.get(overwrite, "")
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 input,.param .u64 output,.param .u64 n) {{
.reg .b64 %rd<6>;
.reg .b{width} %h<4>;
.reg .b32 %r;
.reg .pred %p<4>;
ld.param.u64 %rd1,[input];
ld.param.u64 %rd2,[output];
ld.param.u64 %rd3,[n];
cvt.u{width}.u64 %h2,%rd3;
ld.global.b{width} %h1,[%rd1];
{lower}
{upper}
{conj}
selp.b{width} %h3,{args},%p3;
{change}
cvt.u64.u{width} %rd4,%h3;
add.u64 %rd5,%rd2,%rd4;
mov.u32 %r,1;
st.global.u8 [%rd5+{extra}],%r;
ret;
}}"""


CASES = [
    ("w16_inverse0", {"width": 16, "inverted": False}, True),
    ("w16_inverse1", {"width": 16, "inverted": True}, True),
    ("w16_default_outside", {"width": 16, "default": 128}, False),
    ("w16_missing_lower", {"width": 16, "missing": "lower"}, False),
    ("w16_missing_upper", {"width": 16, "missing": "upper"}, False),
    ("w16_past_end", {"width": 16, "extra": 1}, False),
    ("w16_overwrite", {"width": 16, "overwrite": "selected"}, False),
    ("w32_inverse0", {"width": 32, "inverted": False}, True),
    ("w32_inverse1", {"width": 32, "inverted": True}, True),
    ("w32_default_outside", {"width": 32, "default": 128}, False),
    ("w32_missing_lower", {"width": 32, "missing": "lower"}, False),
    ("w32_missing_upper", {"width": 32, "missing": "upper"}, False),
    ("w32_past_end", {"width": 32, "extra": 1}, False),
    ("w32_overwrite", {"width": 32, "overwrite": "selected"}, False),
    ("w64_inverse0", {"width": 64, "inverted": False}, True),
    ("w64_inverse1", {"width": 64, "inverted": True}, True),
    ("w64_default_outside", {"width": 64, "default": 128}, False),
    ("w64_missing_lower", {"width": 64, "missing": "lower"}, False),
    ("w64_missing_upper", {"width": 64, "missing": "upper"}, False),
    ("w64_past_end", {"width": 64, "extra": 1}, False),
    ("w64_overwrite", {"width": 64, "overwrite": "selected"}, False),
]


@pytest.mark.parametrize("label,options,accepted", CASES, ids=[case[0] for case in CASES])
def test_constant_selection_preserves_checked_index_bounds(label, options, accepted):
    # Refused variants are static controls; no invalid memory access is run.
    result = verify(ptx(**options), [B], timeout=30)
    if accepted:
        assert result == Accepted()
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result
