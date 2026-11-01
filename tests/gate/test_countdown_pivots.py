"""A bounded countdown anchors a moving pointer in either direction."""

import pytest

from opgate.binding import Base, Binding, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

B = Binding(
    "kernel",
    (SizeSymbol("n", 0, 128),),
    (
        ("output", Base(Span(Direction.WRITE, Affine.symbol(0), 1, None, False))),
        ("n", Size(Affine.symbol(0))),
    ),
    (1, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def ptx(descending=False, step=1, past_end=False, precheck=True, overwrite=False):
    setup = "add.u64 %rd1,%rd1,%rd2; add.u64 %rd1,%rd1,-1;" if descending else ""
    change = -step if descending else step
    extra = (-1 if descending else 1) if past_end else 0
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 output,.param .u64 n) {{
.reg .b64 %rd<5>;
.reg .b32 %r;
.reg .pred %p<3>;
ld.param.u64 %rd1,[output];
ld.param.u64 %rd2,[n];
{"setp.eq.u64 %p1,%rd2,0; @%p1 bra DONE;" if precheck else ""}
{setup}
mov.u64 %rd3,%rd2;
mov.u32 %r,1;
LOOP:
st.global.u8 [%rd1+{extra}],%r;
add.u64 %rd1,%rd1,{change};
add.u64 %rd3,%rd3,-{step};
{"mov.u64 %rd3,128;" if overwrite else ""}
setp.ne.u64 %p2,%rd3,0;
@%p2 bra LOOP;
DONE:
ret;
}}"""


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("case", ["valid", "step", "past_end", "missing_entry", "overwrite"])
def test_countdown_keeps_pointer_bounds_in_both_directions(descending, case):
    decision = verify(
        ptx(
            descending,
            2 if case == "step" else 1,
            case == "past_end",
            case != "missing_entry",
            case == "overwrite",
        ),
        [B],
        timeout=30,
    )
    if case == "valid":
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
