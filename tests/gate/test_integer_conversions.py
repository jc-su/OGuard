"""Narrowing and signed extension retain exact integer-width boundaries."""

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction


@pytest.mark.parametrize(
    "source,destination,value,signed",
    [
        (16, 32, 7, False),
        (16, 32, 7, True),
        (16, 64, 7, False),
        (16, 64, 7, True),
        (32, 16, 65539, False),
        (32, 16, 65539, True),
        (64, 16, 65539, False),
        (64, 16, 65539, True),
        (16, 32, 65535, False),
        (16, 64, 65535, False),
        (64, 16, -1, True),
        (32, 16, -1, True),
    ],
)
@pytest.mark.parametrize("inside", [True, False])
def test_integer_conversion_exact_extent(source, destination, value, signed, inside):
    source_bits = value % (1 << source)
    read = (
        source_bits - (1 << source)
        if signed and source_bits >= (1 << (source - 1))
        else source_bits
    )
    converted = read % (1 << destination)
    extent = converted + int(inside)
    kind = "s" if signed else "u"
    ptx = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {{
.reg .b{source} %source;
.reg .b{destination} %converted;
.reg .b64 %base, %offset, %address;
ld.param.u64 %base, [output];
mov.b{source} %source, {value};
cvt.u{destination}.{kind}{source} %converted, %source;
cvt.u64.u{destination} %offset, %converted;
add.u64 %address, %base, %offset;
st.global.u8 [%address], 1;
ret;
}}
"""
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(extent), 8, None, False))),),
        (1, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    result = verify(ptx, [binding], timeout=60)
    if inside:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result


@pytest.mark.parametrize("destination", [32, 64])
def test_negative_signed_extension_is_not_zero_extension(destination):
    ptx = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {{
.reg .b16 %small;
.reg .b{destination} %extended;
.reg .b64 %base,%offset,%address;
ld.param.u64 %base,[output];
mov.b16 %small,65535;
cvt.s{destination}.s16 %extended,%small;
cvt.u64.u{destination} %offset,%extended;
add.u64 %address,%base,%offset;
st.global.u8 [%address],1;
ret;
}}
"""
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(65536), 8, None, False))),),
        (1, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    result = verify(ptx, [binding], timeout=60)
    assert isinstance(result, Refused), result
    assert not result.reason.startswith(TIMEOUT_REASON), result
