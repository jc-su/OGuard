"""A load from a buffer with an index range reads one whole element (DESIGN
decision 47).

`sel` holds eight 4-byte elements in `[0, 7]`. Thread `i < 8` loads
`sel[i]` and, when the kernel writes `sel`, stores `i` to it. A load of the
element's width at the element's alignment is accepted. A load of two bytes
of an element, of one byte inside it, or of two elements at once is
refused: of an index output that the launch also writes, such a load would
race with the stores of whole elements in a way that PTX's memory model
does not describe, and an index input is held to the same rule.
"""

import pytest

from opgate.binding import Base, Binding, ElementType, IndexRange, Span
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction

WHOLE = "ld.global.u32 %r2, [%rd4];"
HALF = "ld.global.u16 %rs1, [%rd4];"
BYTE = "ld.global.u8 %rs1, [%rd4+1];"
# two elements at once, at an address that is a multiple of 8 inside the buffer
PAIR = """and.b32 %r3, %r1, 6;
mul.wide.u32 %rd5, %r3, 4;
add.s64 %rd6, %rd2, %rd5;
ld.global.v2.u32 {%r2, %r3}, [%rd6];"""
STORE = "st.global.u32 [%rd4], %r1;"
REASON = "the load does not read one whole element of the index buffer"


def kernel(load: str, store: str) -> str:
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 sel) {{
.reg .pred %p<2>;
.reg .b16 %rs<2>;
.reg .b32 %r<4>;
.reg .b64 %rd<7>;
ld.param.u64 %rd1, [sel];
cvta.to.global.u64 %rd2, %rd1;
mov.u32 %r1, %tid.x;
setp.lt.u32 %p1, %r1, 8;
@!%p1 bra DONE;
mul.wide.u32 %rd3, %r1, 4;
add.s64 %rd4, %rd2, %rd3;
{load}
{store}
DONE:
ret;
}}
"""


def binding(direction: Direction) -> Binding:
    index = IndexRange(ElementType(4, True), Affine(0), Affine(7))
    return Binding(
        "kernel",
        (),
        (("sel", Base(Span(direction, Affine(32), 8, index, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )


@pytest.mark.parametrize(
    ("direction", "store"),
    [(Direction.READ_WRITE, STORE), (Direction.READ, "")],
    ids=["index output", "index input"],
)
def test_a_load_of_one_whole_element_is_accepted(direction: Direction, store: str) -> None:
    assert verify(kernel(WHOLE, store), [binding(direction)]) == Accepted()


@pytest.mark.parametrize("load", [HALF, BYTE, PAIR], ids=["two bytes", "one byte", "two elements"])
@pytest.mark.parametrize(
    ("direction", "store"),
    [(Direction.READ_WRITE, STORE), (Direction.READ, "")],
    ids=["index output", "index input"],
)
def test_a_load_of_another_width_is_refused(load: str, direction: Direction, store: str) -> None:
    decision = verify(kernel(load, store), [binding(direction)])
    assert isinstance(decision, Refused), decision
    assert REASON in decision.reason, decision.reason
