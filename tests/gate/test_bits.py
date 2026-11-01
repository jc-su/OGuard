"""Bits of the thread index: high bits, bit-field guards, masks, and xor (design decision 24).

`out` has `n >= 256` rows and `small` has `m >= 8`, for a block of 256
threads, `%r1` the thread index. Each accepted kernel stores to a row that
its bits bound; each refused variant differs in one instruction that lets the
row leave its buffer.

- A warp's start `tid - (tid & 31)` is `32 (tid >> 5)`, never negative; the
  lane less the thread index, `(tid & 31) - tid`, is.
- A guard `(tid & 31) <= 7` bounds the bit field that indexes `small`; the
  guard `<= 8` does not.
- Numba's `tid % 32` subtracts `tid & -32`, which keeps bits 5 to 31 of the
  32-bit `tid`, so `tid - lane` is the warp's start again.
- `(tid & 4) ^ (tid & 64)` lies below 128; with `2 tid` in place of `tid &
  64`, the xor reaches 1023.
"""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction

SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n", 256), sig.SizeSymbol("m", 8)),
    arguments=(
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Span("small", "small", Direction.WRITE, Affine.symbol(1)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
        sig.Scalar("m", sig.SizeOf(Affine.symbol(1))),
    ),
)
BINDING = derive_binding(
    SIGNATURE,
    {"out": ElementType(4, True), "small": ElementType(4, True)},
    Convention.C,
    "kernel",
    tuple(f"kernel_param_{k}" for k in range(4)),
    (256, 1, 1),
    GRID_LIMIT,
    Affine(0),
)


def kernel(body: str) -> str:
    """The kernel with `%rd1` the address of `out`, `%rd2` of `small`, and `%r1` the thread index."""
    return f""".version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1,
	.param .u64 kernel_param_2,
	.param .u64 kernel_param_3
)
{{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<12>;

	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	mov.u32 	%r1, %tid.x;
	and.b32 	%r2, %r1, 31;
{body}
$L__end:
	ret;
}}
"""


# out[i] for the 32-bit row index %r3
STORE_OUT = """	mul.wide.u32 	%rd4, %r3, 4;
	add.s64 	%rd5, %rd1, %rd4;
	st.global.u32 	[%rd5], %r1;"""
WARP_START = "\tsub.s32 \t%r3, %r1, %r2;\n" + STORE_OUT
LANE_LESS_TID = "\tsub.s32 \t%r3, %r2, %r1;\n" + STORE_OUT

# small[tid & 31] under a guard on the lane
GUARD_TAIL = """	@%p1 bra 	$L__end;
	mul.wide.u32 	%rd4, %r2, 4;
	add.s64 	%rd5, %rd2, %rd4;
	st.global.u32 	[%rd5], %r1;"""
GUARD_7 = "\tsetp.gt.u32 \t%p1, %r2, 7;\n" + GUARD_TAIL
GUARD_8 = "\tsetp.gt.u32 \t%p1, %r2, 8;\n" + GUARD_TAIL

# Numba's tid % 32: lane = tid - (tid & -32); out[tid - lane]
MOD_HEAD = """	cvt.u64.u32 	%rd3, %r1;
	and.b64 	%rd6, %rd3, -32;
	sub.s64 	%rd7, %rd3, %rd6;
	sub.s64 	%rd8, %rd3, %rd7;
"""
MOD_TAIL = """	shl.b64 	%rd9, %rd8, 2;
	add.s64 	%rd10, %rd1, %rd9;
	st.global.u32 	[%rd10], %r1;"""
MOD_START = MOD_HEAD + MOD_TAIL
MOD_START_LESS_ONE = MOD_HEAD + "\tadd.s64 \t%rd8, %rd8, -1;\n" + MOD_TAIL

# out[(tid & 4) ^ x]
XOR_HEAD = "\tand.b32 \t%r4, %r1, 4;\n"
XOR_TAIL = "\txor.b32 \t%r3, %r4, %r5;\n" + STORE_OUT
XOR_OF_BITS = XOR_HEAD + "\tand.b32 \t%r5, %r1, 64;\n" + XOR_TAIL
XOR_OF_DOUBLE = XOR_HEAD + "\tshl.b32 \t%r5, %r1, 1;\n" + XOR_TAIL

UNPROVED = "the access is not proved inside its buffer, a shared variable, or a local variable"


@pytest.mark.parametrize(
    "body",
    [WARP_START, GUARD_7, MOD_START, XOR_OF_BITS],
    ids=["warp start", "guarded lane", "numba modulo", "xor of bits"],
)
def test_a_row_that_its_bits_bound_is_accepted(body: str) -> None:
    assert verify(kernel(body), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    "body",
    [LANE_LESS_TID, GUARD_8, MOD_START_LESS_ONE, XOR_OF_DOUBLE],
    ids=["lane less tid", "guard past small", "numba modulo less one", "xor of a double"],
)
def test_a_row_that_leaves_its_buffer_is_refused(body: str) -> None:
    decision = verify(kernel(body), [BINDING])
    assert isinstance(decision, Refused)
    assert f"{UNPROVED}: st.global.u32" in decision.reason
