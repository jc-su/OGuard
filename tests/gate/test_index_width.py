"""Index-value facts require a complete element of the declared width."""

import pytest

from opgate.binding import Base, Binding, ElementType, IndexRange, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction


def kernel(bits, *, offset=0, indexed_store=True):
    address = "%rd4" if indexed_store else "%rd2"
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 input, .param .u64 output) {{
.reg .pred %p<2>;
.reg .b32 %r<3>;
.reg .b64 %rd<5>;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
mov.u32 %r1, %tid.x;
setp.lt.u32 %p1, %r1, 1;
mov.u32 %r2, 0;
@%p1 ld.global.u{bits} %r2, [%rd1+{offset}];
cvt.u64.u32 %rd3, %r2;
add.u64 %rd4, %rd2, %rd3;
@%p1 st.global.u8 [{address}], 1;
ret;
}}
"""


def binding(element, upper=7):
    index = (
        None
        if element is None
        else IndexRange(ElementType(element, False), Affine(0), Affine(upper))
    )
    return Binding(
        "kernel",
        (),
        (
            ("input", Base(Span(Direction.READ, Affine(element or 1), 8, index, False))),
            ("output", Base(Span(Direction.WRITE, Affine(8), 8, None, False))),
        ),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )


@pytest.mark.parametrize(
    "bits,element,upper,accepted",
    [(32, 4, 7, True), (8, 1, 7, True), (8, 4, 1023, False), (8, None, 7, False)],
)
def test_loaded_index_facts_require_matching_element_width(bits, element, upper, accepted):
    # Static controls only. A byte of an arbitrary value or of a 32-bit
    # value in [0, 1023] may exceed the eight-byte output's valid indices.
    result = verify(kernel(bits), [binding(element, upper)], timeout=180)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result


@pytest.mark.parametrize("offset,accepted", [(0, True), (1, False)])
def test_plain_guarded_byte_load_keeps_its_access_check(offset, accepted):
    result = verify(kernel(8, offset=offset, indexed_store=False), [binding(None)], timeout=180)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result
