"""The gate proves that every thread of an accepted kernel ends (decision 40).

A grid-stride loop ends because its counter grows by the grid's width until it
reaches `n`. With a step of 0 every access stays in bounds, but the loop does
not end, and the gate refuses the kernel at the loop's back edge. A binary
search ends because each iteration moves one bound strictly inside: Opus's
`lower_bound` is accepted, and the same search with `lo = mid` in place of
`lo = mid + 1`, whose accesses stay in bounds but which does not end once
`hi = lo + 1`, is refused. A warp collective in a kernel without a CTA barrier
needs the proof that every lane that its member mask names reaches it: a
shuffle that every lane reaches is accepted, and the same shuffle that one
lane branches around is refused.

The compilers' other loop forms end by the same ranks, and each control below
changes the loop so that the accesses stay in bounds but the loop does not
end: Numba's 32-bit range loop adds a `selp` on the back edge's guard,
Numba's 64-bit range loop copies a successor that the head computes, Triton's
masked loop has two back edges, and a merge-path search changes its bounds on
several paths through the loop. Triton's masked loop with every access checked
increments its counter before the head and leaves when the counter equals the
loop's length; it ends however large the counter is on entry.

A counter incremented by 1 ends when the test after the increment keeps it from
one value while the loop continues, as `c <= n` does for a 32-bit counter `c`
read in 64 bits and `n` below 2^32 - 1; the same loop with a bound that reaches
2^32 - 1 does not end, since the counter wraps, and is refused. Haiku's CUDA C++
`string_contains` increments its counter in a block between the loop's head and
its latch, and DeepSeek's TPC-H Q22 group-by reads its counter through a
conversion to 64 bits and back; each control stops the increment.

With every access checked, MiniMax's Triton TPC-H Q17 loop steps its counter by
8 from 0 and leaves when it equals 8192, so the counter stays a multiple of 8;
Qwen's TPC-H Q17 binary search keeps its lower bound at least 0 and its upper
bound at most `n` from its entry, which bounds the midpoint. Each control stops
the counter or lets the lower bound stay at the midpoint.
"""

import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.frontends.guard import guard_every
from opgate.gate import Accepted, Refused, verify
from opgate.signature import Affine, Direction

ENDS = "the kernel is not proved to end"
COLLECTIVE = "warp collective participation is not proved"

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

STEP = "add.s64 \t%rd6, %rd6, %rd7;"
GRID_STRIDE = f""".version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
{{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;

	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	cvta.to.global.u64 	%rd3, %rd1;
	mov.u32 	%r1, %tid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %nctaid.x;
	mul.wide.u32 	%rd4, %r2, %r3;
	cvt.u64.u32 	%rd5, %r1;
	add.s64 	%rd6, %rd4, %rd5;
	setp.ge.u64 	%p1, %rd6, %rd2;
	@%p1 bra 	$L__BB0_2;
	mul.wide.u32 	%rd7, %r3, %r4;
$L__BB0_1:
	shl.b64 	%rd8, %rd6, 2;
	add.s64 	%rd9, %rd3, %rd8;
	st.global.u32 	[%rd9], %r1;
	{STEP}
	setp.lt.u64 	%p2, %rd6, %rd2;
	@%p2 bra 	$L__BB0_1;
$L__BB0_2:
	ret;
}}
"""

SKIP = "setp.eq.u32 \t%p1, %r1, 5;\n\t@%p1 bra \t$L__BB0_2;"
SHUFFLE = """.version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<7>;

	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	mov.u32 	%r1, %tid.x;
	SKIP
	shfl.sync.down.b32 	%r2, %r1, 1, 31, -1;
	cvt.u64.u32 	%rd3, %r1;
	setp.ge.u64 	%p2, %rd3, %rd2;
	@%p2 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd4, %rd1;
	shl.b64 	%rd5, %rd3, 2;
	add.s64 	%rd6, %rd4, %rd5;
	st.global.u32 	[%rd6], %r2;
$L__BB0_2:
	ret;
}
"""

# A block between the head and the latch increments the counter %r2, and the
# test that leaves the loop reads it in 64 bits; the latch holds only a jump.
TEST = "setp.lt.u64 \t%p2, %rd2, %rd7;"
KEPT = f""".version 8.0
.target sm_80
.address_size 64

.visible .entry kernel(
	.param .u64 kernel_param_0,
	.param .u64 kernel_param_1
)
{{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<9>;

	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u64 	%rd2, [kernel_param_1];
	cvta.to.global.u64 	%rd3, %rd1;
	mov.u32 	%r1, %tid.x;
	cvt.u64.u32 	%rd4, %r1;
	setp.ge.u64 	%p1, %rd4, %rd2;
	@%p1 bra 	$L__BB0_3;
	shl.b64 	%rd5, %rd4, 2;
	add.s64 	%rd6, %rd3, %rd5;
	add.s64 	%rd8, %rd2, 4294967296;
	mov.u32 	%r2, 0;
$L__BB0_1:
	st.global.u32 	[%rd6], %r2;
	setp.eq.u32 	%p1, %r1, 5;
	@%p1 bra 	$L__BB0_3;
	add.s32 	%r2, %r2, 1;
	cvt.u64.u32 	%rd7, %r2;
	{TEST}
	@%p2 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_1;
$L__BB0_3:
	ret;
}}
"""

SEARCH_STEP = "add.s64 \t%rd24, %rd20, 1;"


def test_a_grid_stride_loop_ends() -> None:
    assert verify(GRID_STRIDE, [BINDING]) == Accepted()


def test_a_loop_whose_counter_does_not_move_is_refused() -> None:
    assert GRID_STRIDE.count(STEP) == 1
    decision = verify(GRID_STRIDE.replace(STEP, "add.s64 \t%rd6, %rd6, 0;"), [BINDING])
    assert isinstance(decision, Refused), decision
    assert ENDS in decision.reason and "@%p2 bra $L__BB0_1" in decision.reason, decision


def test_a_counter_that_its_test_keeps_from_one_value_ends() -> None:
    assert verify(KEPT, [BINDING]) == Accepted()


def test_a_counter_that_its_test_never_stops_is_refused() -> None:
    assert KEPT.count(TEST) == 1
    # The bound n + 2^32 lets the counter wrap; never launch this PTX.
    decision = verify(KEPT.replace(TEST, "setp.lt.u64 \t%p2, %rd8, %rd7;"), [BINDING])
    assert isinstance(decision, Refused), decision
    assert ENDS in decision.reason and "bra.uni $L__BB0_1" in decision.reason, decision


def _search(text: str) -> Accepted | Refused:
    return verify(text, [binding_of("opus", "lower_bound", "nvcc", text)], timeout=120)


@pytest.mark.slow
def test_a_binary_search_ends() -> None:
    text = (directory_of("opus", "lower_bound") / "kernel.O1.ptx").read_text()
    assert _search(text) == Accepted()


@pytest.mark.slow
def test_a_search_whose_lower_bound_can_stay_is_refused() -> None:
    # Static negative control; never launch the modified PTX.
    text = (directory_of("opus", "lower_bound") / "kernel.O1.ptx").read_text()
    assert text.count(SEARCH_STEP) == 1
    decision = _search(text.replace(SEARCH_STEP, "add.s64 \t%rd24, %rd20, 0;"))
    assert isinstance(decision, Refused), decision
    assert ENDS in decision.reason, decision


def test_a_shuffle_that_every_lane_reaches_is_accepted() -> None:
    assert verify(SHUFFLE.replace("SKIP", ""), [BINDING]) == Accepted()


def test_a_shuffle_that_a_lane_branches_around_is_refused() -> None:
    decision = verify(SHUFFLE.replace("SKIP", SKIP), [BINDING])
    assert isinstance(decision, Refused), decision
    assert COLLECTIVE in decision.reason, decision


# (model, task, toolchain), and the control's replacements of instructions
FORMS = {
    # the update's step is a `selp` on the back edge's guard
    "numba_range_32": ("haiku", "segmented_sum", "numba"),
    "numba_range_64": ("haiku", "string_equal", "numba"),
    "triton_two_latches": ("minimax-m2.5", "q16_core_triton__count_unique_kernel_2", "triton"),
    "merge_path": ("sonnet", "merge_sorted", "nvcc"),
    "update_before_latch": ("haiku", "string_contains", "nvcc"),
    "counter_through_conversions": (
        "deepseek-v4-flash",
        "q22_core_cuda_q22_groupby_kernel",
        "nvcc",
    ),
}
# Numba's 32-bit range loop has no control: the safety proof recognizes the
# loop only with its exact update.
CONTROLS = {
    "numba_range_64": (("mov.u64 \t%rd54, 1;", "mov.u64 \t%rd54, 0;"),),
    # the control stops both the counter and the address
    "triton_two_latches": (
        ("add.s64 \t%rd12, %rd12, 1;", "add.s64 \t%rd12, %rd12, 0;"),
        ("add.s64 \t%rd11, %rd11, 8;", "add.s64 \t%rd11, %rd11, 0;"),
    ),
    # the control's new upper bound is the midpoint, which can equal it
    "merge_path": (("add.s64 \t%rd37, %rd53, -1;", "add.s64 \t%rd37, %rd53, 0;"),),
    "update_before_latch": (("add.s32 \t%r16, %r16, 1;", "add.s32 \t%r16, %r16, 0;"),),
    "counter_through_conversions": (("add.s32 \t%r10, %r9, 1;", "add.s32 \t%r10, %r9, 0;"),),
}


def _form(name: str, control: bool) -> Accepted | Refused:
    model, task, toolchain = FORMS[name]
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    if control:
        for update, stuck in CONTROLS[name]:
            assert text.count(update) == 1
            # Static negative control; never launch the modified PTX.
            text = text.replace(update, stuck)
    return verify(text, [binding_of(model, task, toolchain, text)], timeout=300)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(FORMS))
def test_a_compiled_loop_form_ends(name: str) -> None:
    assert _form(name, False) == Accepted()


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_the_same_loop_without_progress_is_refused(name: str) -> None:
    decision = _form(name, True)
    assert isinstance(decision, Refused), decision
    assert ENDS in decision.reason, decision


ROLLING_STEP = "add.s64 \t%rd20, %rd20, 1;"


def _rolling(step: str) -> Accepted | Refused:
    text = (directory_of("haiku", "rolling_sum") / TOOLCHAINS["triton"]).read_text()
    every, binding = guard_every(text, binding_of("haiku", "rolling_sum", "triton", text))
    assert every.count(ROLLING_STEP) == 1
    # Static negative control when the step changes; never launch the modified PTX.
    return verify(every.replace(ROLLING_STEP, step), [binding], timeout=120)


@pytest.mark.slow
def test_a_loop_that_leaves_at_its_length_ends() -> None:
    assert _rolling(ROLLING_STEP) == Accepted()


@pytest.mark.slow
def test_the_same_loop_without_its_increment_is_refused() -> None:
    decision = _rolling("add.s64 \t%rd20, %rd20, 0;")
    assert isinstance(decision, Refused), decision
    assert ENDS in decision.reason, decision


# (model, task, toolchain), the instruction that makes progress, and the
# control's replacement; every access of the kernel is checked
CHECKED = {
    "triton_step_8": (
        ("minimax-m2.5", "q17_core_triton__q17_phase1_kernel", "triton"),
        ("add.s64 \t%rd22, %rd22, 8;", "add.s64 \t%rd22, %rd22, 0;"),
    ),
    "search_from_entry": (
        ("qwen3.5-397b", "q17_cuda_kernel_agg", "nvcc"),
        ("add.s64 \t%rd29, %rd25, 1;", "add.s64 \t%rd29, %rd25, 0;"),
    ),
}


def _checked(name: str, control: bool) -> Accepted | Refused:
    (model, task, toolchain), (update, stuck) = CHECKED[name]
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    every, binding = guard_every(text, binding_of(model, task, toolchain, text))
    assert every.count(update) == 1
    # Static negative control when the update changes; never launch the modified PTX.
    return verify(every.replace(update, stuck) if control else every, [binding], timeout=300)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CHECKED))
def test_a_checked_loop_ends(name: str) -> None:
    assert _checked(name, False) == Accepted()


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CHECKED))
def test_the_same_checked_loop_without_progress_is_refused(name: str) -> None:
    decision = _checked(name, True)
    assert isinstance(decision, Refused), decision
    assert ENDS in decision.reason, decision
