"""Barrier participation through a counter that leaves at its limit, a loop that the
termination check proves ends, and acyclic segments; static CPU checks only."""

import subprocess

import pytest

from opgate.binding import Base, Binding, Size, Span
from opgate.gate import Accepted, Refused, formal_dir, verify
from opgate.signature import Affine, Direction, SizeSymbol

PARTICIPATION = "CTA barrier participation is not proved"


def compile_query(tmp_path, query):
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def counted(test: str, step: int = 1, gap: str = "") -> str:
    # Triton's `for i in range(n)` around barriers and a warp collective: the
    # counter starts at 0 after a guard on n, steps by `step`, and the back
    # edge tests it against n.
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<3>;
.reg .b32 %r<3>;
.reg .b64 %rd<6>;
ld.param.u64 %rd1,[n];
setp.lt.s64 %p1,%rd1,1;
@%p1 bra DONE;
mov.u64 %rd2,0;
mov.u64 %rd3,0;
LOOP:
bar.sync 0;
mov.u32 %r1,%tid.x;
shfl.sync.idx.b32 %r2,%r1,0,31,-1;
bar.sync 0;
add.s64 %rd2,%rd2,{step};
{gap}
{test}
@%p2 bra LOOP;
DONE:
ret;
}}
"""


def binding_n(block: int = 64) -> Binding:
    return Binding(
        "kernel",
        (SizeSymbol("n", 0, 2147483647),),
        (("n", Size(Affine.symbol(0))),),
        (block, 1, 1),
        (16, 1, 1),
        Affine(0),
    )


@pytest.mark.parametrize(
    "test,gap",
    [
        ("setp.lt.s64 %p2,%rd2,%rd1;", ""),
        ("setp.ne.s64 %p2,%rd2,%rd1;", ""),
        ("setp.ne.b64 %p2,%rd1,%rd2;", ""),
        ("setp.ne.b64 %p2,%rd1,%rd2;", "add.s64 %rd3,%rd3,8;"),
    ],
)
def test_counted_loop_accepted(test, gap):
    assert verify(counted(test, gap=gap), [binding_n()]) == Accepted()


ENDS = "the kernel is not proved to end"


@pytest.mark.parametrize(
    "test,step,gap,reason",
    [
        # A step of 2 can pass the limit, and the loop does not leave at it.
        # The counter is the same in every thread, so the shared value route
        # (decision 41) proves participation and the termination check refuses.
        ("setp.ne.s64 %p2,%rd2,%rd1;", 2, "", ENDS),
        # The counter changes between its update and its test: the same.
        ("setp.ne.s64 %p2,%rd2,%rd1;", 1, "add.s64 %rd2,%rd2,1;", ENDS),
        # The thread index is not the same for every thread of the block.
        ("setp.ne.b64 %p2,%rd1,%rd2;", 1, "cvt.u64.u32 %rd2,%r1;", PARTICIPATION),
    ],
)
def test_counted_loop_refused(test, step, gap, reason):
    result = verify(counted(test, step, gap), [binding_n()])
    assert isinstance(result, Refused)
    assert reason in result.reason


def search(hi: str) -> str:
    # A binary search over 8 loaded words before a reduction whose loop holds
    # two barriers: the termination check proves that the search ends.
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 data) {{
.reg .pred %p<4>;
.reg .b32 %r<6>;
.reg .b64 %rd<8>;
ld.param.u64 %rd1,[data];
mov.u64 %rd2,0;
mov.u64 %rd3,8;
SEARCH:
add.s64 %rd4,%rd2,%rd3;
shr.u64 %rd4,%rd4,1;
shl.b64 %rd5,%rd4,2;
add.s64 %rd6,%rd1,%rd5;
ld.global.u32 %r2,[%rd6];
setp.lt.u32 %p1,%r2,7;
add.s64 %rd7,%rd4,1;
selp.b64 %rd2,%rd7,%rd2,%p1;
{hi}
setp.lt.s64 %p3,%rd2,%rd3;
@%p3 bra SEARCH;
mov.u32 %r5,4;
RED:
bar.sync 0;
bar.sync 0;
shr.u32 %r5,%r5,1;
setp.ne.u32 %p2,%r5,0;
@%p2 bra RED;
ret;
}}
"""


BINDING_DATA = Binding(
    "kernel",
    (),
    (("data", Base(Span(Direction.READ, Affine(32), 4, None, False))),),
    (64, 1, 1),
    (16, 1, 1),
    Affine(0),
)


def test_search_before_a_barrier_loop_accepted():
    assert verify(search("selp.b64 %rd3,%rd3,%rd4,%p1;"), [BINDING_DATA]) == Accepted()


def test_loop_that_may_not_end_refused_for_termination():
    # An upper bound that keeps its value can keep the search going:
    # participation holds for a kernel that ends, and the termination check
    # refuses this one.
    result = verify(search("selp.b64 %rd3,%rd3,%rd3,%p1;"), [BINDING_DATA])
    assert isinstance(result, Refused)
    assert "not proved to end" in result.reason


META = """From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateDomain GateBarrier GateLoop GateContext GateEnds.
Import ListNotations. Local Open Scope Z_scope.
Definition P:program:=[
 [IImm None 1%positive 5];
 [ISetp None CEq 32 false 2%positive 1%positive(OImm 0);IBra(Some(false,2%positive))3%nat];
 [IBin None BShr 32 1%positive 1%positive(OImm 1);IBra None 1%nat];
 [INop;IExit None]].
Definition broken:program:=[
 [IImm None 1%positive 5];
 [ISetp None CEq 32 false 2%positive 1%positive(OImm 0);IBra(Some(false,2%positive))3%nat];
 [IBin None BShr 32 1%positive 1%positive(OImm 1);IBra(Some(false,2%positive))4%nat;IBra None 1%nat];
 [INop;IExit None];[INop;IExit None]].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "ends_region P[(3%nat,0%nat)](1%nat,0%nat)=Some((3%nat,0%nat),[(1%nat,0%nat);(1%nat,1%nat);(2%nat,0%nat);(2%nat,1%nat)])",
        "ends_region P[(1%nat,0%nat)](1%nat,0%nat)=None",
        "ends_region P[(3%nat,0%nat)](2%nat,0%nat)=None",
        "ends_region broken[(3%nat,0%nat);(4%nat,0%nat)](1%nat,0%nat)=None",
        "counted_test_of 64 2%positive(ISetp None CNe 64 false 3%positive 1%positive(OReg 2%positive))=Some(Counted_ne true false,3%positive,1%positive)",
        "counted_test_of 64 2%positive(ISetp None CGt 64 true 3%positive 1%positive(OReg 2%positive))=None",
    ],
)
def test_route_metadata(tmp_path, obligation):
    compile_query(
        tmp_path, META + f"Example checked:{obligation}. Proof. vm_compute. reflexivity. Qed.\n"
    )


def test_route_theorems_closed(tmp_path):
    names = [
        "counted_test_sound",
        "counted_gap_sound",
        "occurrence_segment_total_protected_at",
        "ends_word_sound",
        "ends_summary_ok",
        "ends_progress_check_sound",
        "fixed_progress_check_sound",
        "front_event_check_sound",
        "accepted_front_events",
        "accepted_front_linked_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateContext OpGate.GateEnds OpGate.GateFront OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
