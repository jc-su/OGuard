"""Fault freedom: every access is aligned, and no trap runs.

Each kernel writes to `out`, a buffer of `n` 4-byte elements, at `4 * i` for
`i < n`. The accepted kernels store two bytes at an even offset, or trap
under a predicate that never holds there. The refused variants differ in one
instruction: two bytes at an odd offset, which lies in the buffer but faults,
or a trap under a predicate that holds for some thread.
"""

import pytest

from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction

SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
BINDING = derive_binding(
    SIGNATURE,
    {"out": ElementType(4, True)},
    Convention.C,
    "kernel",
    ("kernel_param_0", "kernel_param_1"),
    (256, 1, 1),
    GRID_LIMIT,
    Affine(0),
)


def kernel(body: str) -> str:
    """The kernel with `i = tid`, `%rd7` the address of `out[i]`, and `i < n` checked."""
    return f""".version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
{{
	.reg .pred 	%p<3>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<8>;

	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	mov.u32 	%r1, %tid.x;
	cvt.u64.u32 	%rd3, %r1;
	setp.ge.u64 	%p1, %rd3, %rd2;
	@%p1 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd5, %rd1;
	shl.b64 	%rd6, %rd3, 2;
	add.s64 	%rd7, %rd5, %rd6;
	cvt.u16.u32 	%rs1, %r1;
{body}
$L__BB0_2:
	ret;
}}
"""


EVEN = "\tst.global.u16 \t[%rd7+2], %rs1;"
ODD = "\tst.global.u16 \t[%rd7+1], %rs1;"
TRAP_NEVER = """	setp.ge.u64 	%p2, %rd3, %rd2;
	@%p2 trap;
	st.global.u16 	[%rd7], %rs1;"""
TRAP_SOMETIMES = """	setp.eq.u64 	%p2, %rd3, 7;
	@%p2 trap;
	st.global.u16 	[%rd7], %rs1;"""


@pytest.mark.parametrize("body", [EVEN, TRAP_NEVER], ids=["aligned", "trap that never runs"])
def test_an_aligned_kernel_without_a_reachable_trap_is_accepted(body: str) -> None:
    assert verify(kernel(body), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (ODD, "not proved aligned to its width: st.global.u16 [%rd7+1]"),
        (
            TRAP_SOMETIMES,
            "not proved unreachable: it traps, or PTX leaves its behavior undefined here: @%p2 trap",
        ),
    ],
    ids=["misaligned", "trap that runs"],
)
def test_a_misaligned_access_or_a_reachable_trap_is_refused(body: str, reason: str) -> None:
    decision = verify(kernel(body), [BINDING])
    assert isinstance(decision, Refused)
    assert reason in decision.reason
