"""Checked counter bounds survive symbolic starts and reordered loop blocks."""

import pytest

from opgate.binding import Base, Binding, ElementType, IndexRange, Size, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction, SizeSymbol

B = Binding(
    "kernel",
    (SizeSymbol("n", 1, 256), SizeSymbol("start", 0, 255)),
    (
        (
            "limit",
            Base(
                Span(
                    Direction.READ,
                    Affine(4),
                    4,
                    IndexRange(ElementType(4, True), Affine(0), Affine.symbol(0)),
                    False,
                )
            ),
        ),
        ("output", Base(Span(Direction.WRITE, Affine.symbol(0), 1, None, False))),
        ("n", Size(Affine.symbol(0))),
        ("start", Size(Affine.symbol(1))),
    ),
    (1, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def ptx(step=1, extra=0, exit_eq=True, precheck=True, overwrite=False, reordered=False):
    body = f"""LOOP:
add.u64 %rd7,%rd2,%rd6;
mov.u32 %r2,1;
st.global.u8 [%rd7+{extra}],%r2;
"""
    step_body = f"""add.u64 %rd6,%rd6,{step};
{"mov.u64 %rd5,256;" if overwrite else ""}
setp.{"eq" if exit_eq else "ne"}.u64 %p2,%rd6,%rd5;
"""
    if reordered:
        loop = (
            "bra LOOP;\nSTEP:\n"
            + step_body
            + f"@{'' if exit_eq else '!'}%p2 bra DONE;\n"
            + body
            + "bra STEP;\n"
        )
    else:
        loop = body + step_body + f"@{'!' if exit_eq else ''}%p2 bra LOOP;\n"
    return f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 limit,.param .u64 output,.param .u64 n,.param .u64 start) {{
.reg .b64 %rd<8>;
.reg .b32 %r<3>;
.reg .pred %p<3>;
ld.param.u64 %rd1,[limit];
ld.param.u64 %rd2,[output];
ld.param.u64 %rd3,[n];
ld.param.u64 %rd4,[start];
ld.global.u32 %r1,[%rd1];
cvt.s64.s32 %rd5,%r1;
mov.u64 %rd6,%rd4;
{"setp.ge.s64 %p1,%rd6,%rd5; @%p1 bra DONE;" if precheck else ""}
{loop}
DONE:
ret;
}}"""


@pytest.mark.parametrize("exit_eq", [True, False])
@pytest.mark.parametrize("reordered", [True, False])
def test_symbolic_counter_is_independent_of_block_order(exit_eq, reordered):
    assert verify(ptx(exit_eq=exit_eq, reordered=reordered), [B], timeout=30) == Accepted()


@pytest.mark.parametrize(
    "options", [{"step": 2}, {"extra": 1}, {"precheck": False}, {"overwrite": True}]
)
@pytest.mark.parametrize("reordered", [True, False])
def test_counter_still_requires_entry_step_limit_and_access_bounds(options, reordered):
    # Static controls only: modified programs are never executed on a GPU.
    result = verify(ptx(reordered=reordered, **options), [B], timeout=30)
    assert isinstance(result, Refused), result
    assert not result.reason.startswith(TIMEOUT_REASON), result
