"""Barrier participation that a block reduction controls, around loops that
threads leave at different iterations (formal/GateShared.v); CPU only."""

import subprocess
from pathlib import Path

import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.binding import Base, Binding, Size, Span
from opgate.gate import Accepted, Refused, formal_dir, verify, zeroed
from opgate.signature import Affine, Direction, SizeSymbol

PARTICIPATION = "CTA barrier participation is not proved"

# Each thread votes; warp maxima go to shared memory; after a barrier each
# lane reads the maximum of warp (lane & 3) and a butterfly over lane masks 2
# and 1 combines the four warps. The branch on the block maximum skips a
# second barrier. Placeholders: VOTE, FIRST (the store of the warp maxima),
# STORE (between the barriers), LOAD (the load address), STAGES (the
# combine), TAIL (the code after the second barrier).
KERNEL = """.version 8.0
.target sm_80
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry kernel(.param .u64 out, .param .u64 n)
.reqntid 128
{{
.reg .pred %p<4>;
.reg .b32 %r<24>;
.reg .b64 %rd<8>;
ld.param.u64 %rd1, [out];
ld.param.u64 %rd2, [n];
mov.u32 %r1, %tid.x;
cvt.u64.u32 %rd3, %r1;
setp.lt.u64 %p1, %rd3, %rd2;
selp.b32 %r2, 1, 0, %p1;
{vote}
shr.u32 %r4, %r1, 3;
and.b32 %r5, %r4, 12;
mov.b32 %r6, global_smem;
add.s32 %r7, %r6, %r5;
{first}
bar.sync 0;
{store}
{load}
add.s32 %r10, %r6, %r9;
ld.shared.b32 %r11, [%r10];
{stages}
setp.ne.s32 %p2, %r15, 1;
@%p2 bra DONE;
bar.sync 0;
{tail}
DONE:
ret;
}}
"""

VOTE = "redux.sync.max.s32 %r3, %r2, -1;"
# Every lane of a warp stores the warp maximum to the warp's word.
FIRST = "st.shared.b32 [%r7], %r3;"
TAIL = """mul.wide.u32 %rd4, %r1, 4;
add.s64 %rd5, %rd1, %rd4;
st.global.u32 [%rd5], %r1;"""
LOAD = "shl.b32 %r8, %r1, 2;\nand.b32 %r9, %r8, 12;"
STAGES = """shfl.sync.bfly.b32 %r12, %r11, 2, 31, -1;
max.s32 %r13, %r11, %r12;
shfl.sync.bfly.b32 %r14, %r13, 1, 31, -1;
max.s32 %r15, %r13, %r14;"""


def kernel(
    vote: str = VOTE,
    first: str = FIRST,
    store: str = "",
    load: str = LOAD,
    stages: str = STAGES,
    tail: str = TAIL,
) -> str:
    return KERNEL.format(vote=vote, first=first, store=store, load=load, stages=stages, tail=tail)


BINDING = Binding(
    "kernel",
    (SizeSymbol("n", 0, 2147483647),),
    (
        ("out", Base(Span(Direction.WRITE, Affine(512), 4, None, False))),
        ("n", Size(Affine.symbol(0))),
    ),
    (128, 1, 1),
    (16, 1, 1),
    Affine(16),
)


def test_block_maximum_branch_accepted() -> None:
    assert verify(zeroed(kernel(), BINDING), [BINDING]) == Accepted()


def test_register_member_mask_accepted() -> None:
    vote = "mov.b32 %r16, -1;\nredux.sync.max.s32 %r3, %r2, %r16;"
    assert verify(zeroed(kernel(vote=vote), BINDING), [BINDING]) == Accepted()


def test_block_sum_branch_accepted() -> None:
    stages = STAGES.replace("max.s32", "add.s32")
    assert verify(zeroed(kernel(stages=stages), BINDING), [BINDING]) == Accepted()


# A 64-bit block sum shuffles its two halves, as Triton does.
WIDE = """cvt.u64.u32 %rd6, %r11;
mov.b64 {{_, %r17}}, %rd6;
cvt.u32.u64 %r18, %rd6;
shfl.sync.bfly.b32 %r19, %r18, {mask}, 31, -1;
shfl.sync.bfly.b32 %r20, %r17, {mask}, 31, -1;
cvt.u64.u32 %rd4, %r19;
cvt.u64.u32 %rd5, %r20;
shl.b64 %rd5, %rd5, 32;
or.b64 %rd7, %rd4, %rd5;
add.s64 %rd6, %rd6, %rd7;"""


def wide_stages(masks: list[int]) -> str:
    body = "\n".join(WIDE.format(mask=m).replace("cvt.u64.u32 %rd6, %r11;\n", "") for m in masks)
    return (
        "cvt.u64.u32 %rd6, %r11;\n"
        + body
        + "\nsetp.ne.s64 %p3, %rd6, 1;\nselp.b32 %r15, 0, 1, %p3;"
    )


def test_wide_block_sum_branch_accepted() -> None:
    assert verify(zeroed(kernel(stages=wide_stages([2, 1])), BINDING), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    "change",
    [
        # A store between the barriers races with the loads of that phase.
        {"store": "st.shared.b32 [%r7], %r2;"},
        # One butterfly stage leaves the value of lane bit 0.
        {"stages": "shfl.sync.bfly.b32 %r12, %r11, 2, 31, -1;\nmax.s32 %r15, %r11, %r12;"},
        # The butterflies combine lanes 0-3, the load reads 8 words.
        {"load": "shl.b32 %r8, %r1, 2;\nand.b32 %r9, %r8, 28;"},
        # A float addition is not modeled: its result may differ by thread.
        {"stages": STAGES.replace("max.s32", "add.f32")},
        # A 64-bit sum of one stage keeps lane bit 0.
        {"stages": wide_stages([2])},
        # Without the reduction each thread branches on its own vote.
        {"vote": "mov.b32 %r3, %r2;", "load": "shr.u32 %r8, %r1, 3;\nand.b32 %r9, %r8, 12;"},
    ],
)
def test_thread_dependent_branch_refused(change: dict[str, str]) -> None:
    result = verify(zeroed(kernel(**change), BINDING), [BINDING])
    assert isinstance(result, Refused)
    assert PARTICIPATION in result.reason


@pytest.mark.parametrize(
    "first",
    [
        # Two stores of the number 0 in an earlier phase than the maxima.
        "st.shared.b32 [%r7], 0;\nmov.b32 %r16, 0;\nst.shared.b32 [%r7], %r16;\nbar.sync 0;\n"
        + FIRST,
        # The store in a later block than the address it uses.
        "bra.uni STORE;\nSTORE:\n" + FIRST,
    ],
)
def test_ordered_earlier_writes_accepted(first: str) -> None:
    assert verify(zeroed(kernel(first=first), BINDING), [BINDING]) == Accepted()


@pytest.mark.parametrize(
    "first",
    [
        # The lanes of a warp store their own votes at one word.
        "st.shared.b32 [%r7], %r2;",
        # A second store of another value at the same words in the same phase.
        FIRST + "\nst.shared.b32 [%r7], 7;",
        # A store of 0 and a store of the maxima at the same words in one phase.
        "st.shared.b32 [%r7], 0;\n" + FIRST,
    ],
)
def test_racing_earlier_writes_refused(first: str) -> None:
    result = verify(zeroed(kernel(first=first), BINDING), [BINDING])
    assert isinstance(result, Refused)
    assert PARTICIPATION in result.reason


# Threads that branch on their own index around a store meet again before
# the next barrier.
def divergent(region: str) -> str:
    return f"""setp.lt.u32 %p3, %r1, 64;
@!%p3 bra SKIP;
{region}
SKIP:
bar.sync 0;"""


def test_divergent_region_accepted() -> None:
    assert verify(zeroed(kernel(tail=divergent(TAIL)), BINDING), [BINDING]) == Accepted()


# A loop over the thread's words inside the region: each thread below 64
# leaves it after its own number of iterations, and every thread meets the
# others again before the next barrier because its run ends (kernel_ends).
STRIDED = """mov.u32 %r16, %r1;
STRIDE:
mul.wide.u32 %rd4, %r16, 4;
add.s64 %rd5, %rd1, %rd4;
st.global.u32 [%rd5], %r1;
add.s32 %r16, %r16, 64;
setp.lt.u32 %p3, %r16, 128;
{barrier}@%p3 bra STRIDE;"""


def test_loop_in_divergent_region_accepted() -> None:
    assert (
        verify(zeroed(kernel(tail=divergent(STRIDED.format(barrier=""))), BINDING), [BINDING])
        == Accepted()
    )


@pytest.mark.parametrize(
    "region",
    [
        # A barrier that only the threads below 64 reach.
        TAIL + "\nbar.sync 0;",
        # A shuffle that only the lanes of the first two warps reach.
        TAIL + "\nshfl.sync.bfly.b32 %r16, %r1, 1, 31, -1;",
        # A barrier in the loop, which only the threads below 64 reach.
        STRIDED.format(barrier="bar.sync 0;\n"),
    ],
)
def test_waiting_site_in_divergent_region_refused(region: str) -> None:
    result = verify(zeroed(kernel(tail=divergent(region)), BINDING), [BINDING])
    assert isinstance(result, Refused)
    assert PARTICIPATION in result.reason


def _corpus(
    model: str, task: str, toolchain: str, edits: tuple[tuple[str, str], ...] = ()
) -> Accepted | Refused:
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    for before, after in edits:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    binding = binding_of(model, task, toolchain, text)
    return verify(zeroed(text, binding), [binding])


def test_corpus_strided_loops_between_barriers_accepted() -> None:
    # Opus's CUDA C++ bucket_histogram clears its shared bins, counts into
    # them, and adds them to the output in loops over the thread's bins and
    # rows, which each thread leaves after its own number of iterations.
    assert _corpus("opus", "bucket_histogram", "nvcc") == Accepted()


def test_corpus_barrier_in_a_strided_loop_refused() -> None:
    # A barrier in the loop that clears the bins, which a thread reaches once
    # for each of its bins.
    store = "\tst.shared.u32 \t[%r7], %r8;"
    result = _corpus("opus", "bucket_histogram", "nvcc", ((store, store + "\n\tbar.sync \t0;"),))
    assert isinstance(result, Refused)
    assert PARTICIPATION in result.reason


def test_two_warp_redux_branch_refused() -> None:
    # A reduction is the same in the lanes of one warp, not across warps.
    text = kernel(
        load="mov.b32 %r9, 0;",
        stages="mov.b32 %r15, %r3;",
    )
    result = verify(zeroed(text, BINDING), [BINDING])
    assert isinstance(result, Refused)
    assert PARTICIPATION in result.reason


META = r"""From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateDomain GateBarrier GateShared.
Import ListNotations. Local Open Scope Z_scope.
"""


def compile_query(tmp_path: Path, query: str) -> None:
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "obligation",
    [
        "single_lane 16 = true",
        "single_lane 3 = false",
        "single_lane 32 = false",
        # Agreeing on bits 2 and 3 of the index: threads 1 and 3 do, 1 and 5 do not.
        "Z.land 12 (Z.lxor 1 3) = 0",
        "Z.land 12 (Z.lxor 1 5) <> 0",
        "phase_next [(0%nat,1%nat)] (0%nat,1%nat) 1 = 2",
        "shared_write (IShA None KStore 1%positive 0 4 None) = true",
        "shared_write (IShA None KLoad 1%positive 0 4 None) = false",
        "compatible (WZero 4) (WZero 4) = true",
        "compatible (WAtomic 4) (WAtomic 8) = false",
        "compatible (WStore 4) (WStore 4) = false",
        "shared_write (IMem None KLoad true 1%positive 0 4 None [] []) = false",
    ],
)
def test_metadata(tmp_path: Path, obligation: str) -> None:
    compile_query(
        tmp_path,
        META
        + f"Goal {obligation}.\nProof. vm_compute. first [reflexivity|intro H; discriminate H]. Qed.\n",
    )
