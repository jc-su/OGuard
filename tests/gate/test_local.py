"""Local memory: a value through a local cell keeps its facts, and a store stays in a live variable.

Each kernel writes `out[i]` for `i < n`, with `i` taken from a local cell. The
accepted kernel stores the guarded index into its local variable and loads it
back. The refused variants store the unguarded size, store past the end of the
variable, or store through the address of a call's frame after the call
returns, where the next call's frame, with its spilled registers, may lie. In
the last two, the index itself is bounded, so only the store is refused.
"""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.signature import Affine, Direction

SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n"),),
    arguments=(
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
ELEMENTS = {"out": ElementType(4, True)}
BINDING = derive_binding(
    SIGNATURE,
    ELEMENTS,
    Convention.C,
    "kernel",
    ("kernel_param_0", "kernel_param_1"),
    (256, 1, 1),
    GRID_LIMIT,
    Affine(0),
)

HEADER = """.version 8.0
.target sm_80
.address_size 64
"""

FRAME_OF = """
.func (.param .b64 func_retval0) frame_of()
{
	.local .align 8 .b8 	__local_depot1[8];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;

	mov.u64 	%SPL, __local_depot1;
	cvta.local.u64 	%SP, %SPL;
	st.param.b64 	[func_retval0+0], %SP;
	ret;
}
"""


def kernel(body: str, size: int = 8) -> str:
    """The kernel with local variable `__local_depot0` of `size` bytes, `i = tid`, and `n`."""
    return f"""
.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
{{
	.local .align 8 .b8 	__local_depot0[{size}];
	.reg .b64 	%SPL;
	.reg .pred 	%p<2>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<10>;

	mov.u64 	%SPL, __local_depot0;
	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	mov.u32 	%r1, %tid.x;
	cvt.u64.u32 	%rd3, %r1;
	setp.ge.u64 	%p1, %rd3, %rd2;
	@%p1 bra 	$L__BB0_2;
{body}
	cvta.to.global.u64 	%rd5, %rd1;
	shl.b64 	%rd6, %rd4, 2;
	add.s64 	%rd7, %rd5, %rd6;
	st.global.u32 	[%rd7], %r1;
$L__BB0_2:
	ret;
}}
"""


THROUGH_CELL = """
	st.local.u64 	[%SPL], %rd3;
	ld.local.u64 	%rd4, [%SPL];"""

UNGUARDED = """
	st.local.u64 	[%SPL], %rd2;
	ld.local.u64 	%rd4, [%SPL];"""

PAST_THE_VARIABLE = """
	st.local.u64 	[%SPL], %rd3;
	st.local.u32 	[%SPL+8], %r1;
	ld.local.u64 	%rd4, [%SPL];"""

INTO_A_RETURNED_FRAME = """
	{
	.param .b64 retval0;
	call.uni (retval0), frame_of;
	ld.param.b64 	%rd8, [retval0+0];
	}
	st.u64 	[%rd8], %rd3;
	mov.u64 	%rd4, %rd3;"""


# A store at a register index that the compiler did not unroll: `%r2` in
# [0, 1] indexes a local array of two 8-byte elements.
AT_A_REGISTER_INDEX = """
	and.b32 	%r2, %r1, 1;
	cvt.u64.u32 	%rd8, %r2;
	shl.b64 	%rd9, %rd8, 3;
	add.s64 	%rd8, %SPL, %rd9;
	st.local.u64 	[%rd8], %rd3;
	mov.u64 	%rd4, %rd3;"""

# The same store at an index in [0, 3], past the two elements.
PAST_THE_ARRAY = """
	and.b32 	%r2, %r1, 3;
	cvt.u64.u32 	%rd8, %r2;
	shl.b64 	%rd9, %rd8, 3;
	add.s64 	%rd8, %SPL, %rd9;
	st.local.u64 	[%rd8], %rd3;
	mov.u64 	%rd4, %rd3;"""

# The guarded index through cell 0, then a store of the unguarded size at
# an index in [1, 2] of three elements: the span misses cell 0, which keeps
# its bound.
BESIDE_THE_CELL = """
	st.local.u64 	[%SPL], %rd3;
	and.b32 	%r2, %r1, 1;
	add.s32 	%r2, %r2, 1;
	cvt.u64.u32 	%rd8, %r2;
	shl.b64 	%rd9, %rd8, 3;
	add.s64 	%rd8, %SPL, %rd9;
	st.local.u64 	[%rd8], %rd2;
	ld.local.u64 	%rd4, [%SPL];"""

# The same store at an index in [0, 1]: the span covers cell 0, which the
# store may overwrite with the unguarded size, so the cell is forgotten.
OVER_THE_CELL = """
	st.local.u64 	[%SPL], %rd3;
	and.b32 	%r2, %r1, 1;
	cvt.u64.u32 	%rd8, %r2;
	shl.b64 	%rd9, %rd8, 3;
	add.s64 	%rd8, %SPL, %rd9;
	st.local.u64 	[%rd8], %rd2;
	ld.local.u64 	%rd4, [%SPL];"""


def test_a_value_through_a_local_cell_keeps_its_bound() -> None:
    assert verify(zeroed(HEADER + kernel(THROUGH_CELL), BINDING), [BINDING]) == Accepted()


def test_a_local_store_at_a_register_index_inside_the_variable_is_accepted() -> None:
    assert (
        verify(zeroed(HEADER + kernel(AT_A_REGISTER_INDEX, 16), BINDING), [BINDING]) == Accepted()
    )


def test_a_local_store_at_a_register_index_beside_a_cell_keeps_the_cell() -> None:
    assert verify(zeroed(HEADER + kernel(BESIDE_THE_CELL, 24), BINDING), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            HEADER + kernel(UNGUARDED),
            "not proved inside its buffer, a shared variable, or a local variable: st.global.u32",
        ),
        (
            HEADER + kernel(PAST_THE_VARIABLE),
            "the local store is not proved inside a live local variable: st.local.u32",
        ),
        (
            HEADER + FRAME_OF + kernel(INTO_A_RETURNED_FRAME),
            "not proved inside its buffer, a shared variable, or a local variable: st.u64 [%rd8+0]",
        ),
        (
            HEADER + kernel(PAST_THE_ARRAY, 16),
            "the local store is not proved inside a live local variable: st.local.u64 [%rd8+0]",
        ),
        (
            HEADER + kernel(OVER_THE_CELL, 16),
            "not proved inside its buffer, a shared variable, or a local variable: st.global.u32",
        ),
    ],
    ids=[
        "unguarded value",
        "past the variable",
        "into a returned frame",
        "past the array at a register index",
        "over the cell at a register index",
    ],
)
def test_a_local_store_that_is_not_proved_is_refused(text: str, reason: str) -> None:
    decision = verify(zeroed(text, BINDING), [BINDING])
    assert isinstance(decision, Refused)
    assert reason in decision.reason


@pytest.mark.slow
def test_current_register_indexed_local_accumulators_are_proved() -> None:
    model, task = "gemini-3.1-pro", "q12_cuda_q12_kernel"
    text = (directory_of(model, task) / "kernel.O1.ptx").read_text()
    binding = binding_of(model, task, "nvcc", text)
    # its 32-bit grid stride is 0 for 2^32 threads, so a thread may never reach the barrier;
    # participation holds for a kernel that ends, and the termination check refuses this one
    decision = verify(zeroed(text, binding), [binding], timeout=180)
    assert isinstance(decision, Refused) and "not proved to end" in decision.reason, decision
    # The accumulators h[sm] and l[sm], with sm in [0, 2) by the signature, are proved
    # in a variant that steps by the stride in 64 bits and has no barrier.
    variant = text.replace("\tbar.sync \t0;\n", "")
    for old, new in (
        ("\tcvt.u64.u32 \t%rd60, %r22;", "\tmul.wide.u32 \t%rd60, %r11, %r21;"),
        ("\tmul.wide.u32 \t%rd61, %r22, 4;", "\tshl.b64 \t%rd61, %rd60, 2;"),
    ):
        assert variant.count(old) == 1
        variant = variant.replace(old, new)
    assert verify(zeroed(variant, binding), [binding], timeout=180) == Accepted()
