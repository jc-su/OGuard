"""OR with bits already present preserves the original value and its relations."""

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, verify
from opgate.signature import Affine, Direction


def kernel(width, mask):
    if width == 32:
        body = f"""not.b32 %r2, %r1;
or.b32 %r3, %r2, {mask};
add.u32 %r3, %r3, %r1;
add.u32 %r3, %r3, 1;
cvt.u64.u32 %rd2, %r3;"""
    else:
        body = f"""cvt.u64.u32 %rd2, %r1;
not.b64 %rd3, %rd2;
or.b64 %rd3, %rd3, {mask};
add.u64 %rd3, %rd3, %rd2;
add.u64 %rd2, %rd3, 1;"""
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {{
.reg .b32 %r<4>;
.reg .b64 %rd<5>;
ld.param.u64 %rd1, [output];
mov.u32 %r1, %tid.x;
{body}
shl.b64 %rd3, %rd2, 2;
add.u64 %rd4, %rd1, %rd3;
st.global.u32 [%rd4], 0;
ret;
}}
"""


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("mask_kind", ["zero", "already-set", "two-set", "may-be-clear"])
def test_or_identity_requires_every_mask_bit_already_set(width, mask_kind):
    mask = {
        "zero": 0,
        "already-set": 1 << (width - 2),
        "two-set": (1 << (width - 2)) | 256,
        "may-be-clear": 128,
    }[mask_kind]
    # With i=tid in [0,255], all complement bits above bit7 are set.
    # ((~i | mask) + i + 1) is zero for the first three masks, modulo
    # the width. For mask128, i=128 produces128 and an out-of-bounds store.
    # Every modified kernel is a static control and is never executed.
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(4), 4, None, False))),),
        (256, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    decision = verify(kernel(width, mask), [binding], timeout=180)
    if mask_kind == "may-be-clear":
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
    else:
        assert decision == Accepted(), decision
