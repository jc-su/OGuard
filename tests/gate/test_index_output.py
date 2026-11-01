"""Index outputs: every value a kernel stores to one lies in the range its consumer gathers by.

`sel` is an index output into `rows`, a buffer of `n` elements, so each value
stored to `sel` must lie in `[0, n - 1]`. Each kernel stores to `sel[i]` for
`i < n`. The accepted kernels store `i`, from a 32-bit register, as the low
bits of a 64-bit one, or as a vector of one register, as Triton writes a
store. The refused variants store `i + 1`, which reaches `n`, or store `i`
with an atomic, whose written value the checker does not know.
"""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction

SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("rows", "rows", Direction.READ, Affine.symbol(0)),
        sig.IndexOut("sel", "sel", Affine.symbol(0), "rows"),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
BINDING = derive_binding(
    SIGNATURE,
    {"rows": ElementType(4, True), "sel": ElementType(4, True)},
    Convention.C,
    "kernel",
    ("kernel_param_0", "kernel_param_1", "kernel_param_2"),
    (256, 1, 1),
    GRID_LIMIT,
    Affine(0),
)


def kernel(body: str) -> str:
    """The kernel with `i = tid` in `%r1` and `%rd3`, `%rd7` the address of `sel[i]`, and `i < n` checked."""
    return f""".version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1,
	.param .u64 kernel_param_2
)
{{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<9>;

	ld.param.u64 	%rd1, [kernel_param_1];
	ld.param.u64 	%rd2, [kernel_param_2];
	mov.u32 	%r1, %tid.x;
	cvt.u64.u32 	%rd3, %r1;
	setp.ge.u64 	%p1, %rd3, %rd2;
	@%p1 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd5, %rd1;
	shl.b64 	%rd6, %rd3, 2;
	add.s64 	%rd7, %rd5, %rd6;
{body}
$L__BB0_2:
	ret;
}}
"""


IN_RANGE = "\tst.global.u32 \t[%rd7], %r1;"
LOW_BITS = "\tst.global.u32 \t[%rd7], %rd3;"
# Triton writes every store's value as a vector of one register
VECTOR_OF_ONE = "\tst.global.b32 \t[ %rd7 + 0 ], { %r1 };"
PAST_THE_RANGE = """	add.s32 	%r2, %r1, 1;
	st.global.u32 	[%rd7], %r2;"""
VECTOR_PAST_THE_RANGE = """	add.s32 	%r2, %r1, 1;
	st.global.b32 	[ %rd7 + 0 ], { %r2 };"""
ATOMIC = "\tatom.global.exch.b32 \t%r3, [%rd7], %r1;"


@pytest.mark.parametrize(
    "body",
    [IN_RANGE, LOW_BITS, VECTOR_OF_ONE],
    ids=["32-bit value", "low bits of a 64-bit value", "vector of one register"],
)
def test_a_store_of_a_value_in_the_range_is_accepted(body: str) -> None:
    assert verify(kernel(body), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            PAST_THE_RANGE,
            "the stored value is not proved to be one element in the index range: st.global.u32",
        ),
        (
            VECTOR_PAST_THE_RANGE,
            "the stored value is not proved to be one element in the index range: st.global.b32",
        ),
        (ATOMIC, "the buffer's direction does not allow the access: atom.global.exch.b32"),
    ],
    ids=["past the range", "vector past the range", "atomic"],
)
def test_a_store_of_a_value_not_proved_in_the_range_is_refused(body: str, reason: str) -> None:
    decision = verify(kernel(body), [BINDING])
    assert isinstance(decision, Refused)
    assert reason in decision.reason
