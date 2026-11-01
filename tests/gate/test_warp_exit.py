"""An entry exit that every lane of a warp takes together; static CPU checks only."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import Accepted, Refused, formal_dir, verify
from opgate.signature import Affine, SizeSymbol

PARTICIPATION = "warp collective participation is not proved"


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


def kernel(index: str) -> str:
    # One warp per segment: the warp of global thread g handles segment g >> 5,
    # and two collectives follow the exit.
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .b32 %r<6>;
.reg .b64 %rd<7>;
.reg .pred %p<2>;
ld.param.u64 %rd6,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ctaid.x;
mov.u32 %r3,%ntid.x;
mul.wide.u32 %rd1,%r2,%r3;
cvt.u64.u32 %rd2,%r1;
add.s64 %rd3,%rd1,%rd2;
{index}
setp.ge.s64 %p1,%rd4,%rd6;
@%p1 bra DONE;
shfl.sync.idx.b32 %r4,%r1,0,31,-1;
shfl.sync.idx.b32 %r5,%r4,1,31,-1;
DONE:
ret;
}}
"""


SEGMENT = "shr.u64 %rd4,%rd3,5;"


def binding(block: int) -> Binding:
    return Binding(
        "kernel",
        (SizeSymbol("n", 0, 2147483647),),
        (("n", Size(Affine.symbol(0))),),
        (block, 1, 1),
        (16, 1, 1),
        Affine(0),
    )


@pytest.mark.parametrize("block", [32, 64, 256, 1024])
def test_warp_uniform_exit_accepted(block):
    assert verify(kernel(SEGMENT), [binding(block)]) == Accepted()


@pytest.mark.parametrize(
    "index,block",
    [
        # Half a warp leaves: lanes 0-15 and 16-31 of one warp differ.
        ("shr.u64 %rd4,%rd3,4;", 64),
        # Each lane decides alone.
        ("mov.u64 %rd4,%rd3;", 64),
        # The low bits of the global index differ inside a warp.
        ("and.b64 %rd4,%rd3,31;", 64),
        # A block of 48 threads splits its second warp between two blocks.
        (SEGMENT, 48),
    ],
)
def test_lane_dependent_exit_refused(index, block):
    result = verify(kernel(index), [binding(block)])
    assert isinstance(result, Refused)
    assert PARTICIPATION in result.reason


META = """From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateDomain GateBarrier GateLoop GateContext.
Import ListNotations. Local Open Scope Z_scope.
Definition entry:list instr:=[IMov None 2%positive 1%positive;
 IBin None BShr 32 2%positive 2%positive(OImm 5);
 ISetp None CGe 32 false 3%positive 2%positive(OReg 4%positive)].
Definition P:program:=[entry++[IBra(Some(false,3%positive))2%nat];
 [IDef None[5%positive];INop];[IExit None]].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "warp_exit_split P=Some(entry,false,3%positive,2%nat,[[IDef None[5%positive];INop];[IExit None]])",
        "warp_exit_split[[IBra None 1%nat];[IExit None]]=None",
        "warp_entry_closed P=true",
        "warp_entry_closed(P++[[IBra None 0%nat]])=false",
        "warp_exit_selected P[(1%nat,0%nat)][]=true",
        "warp_exit_selected P[(0%nat,0%nat)][]=false",
        "warp_exit_selected P[(2%nat,0%nat)][]=false",
        "warp_exit_selected P[(1%nat,0%nat)][(1%nat,0%nat)]=false",
    ],
)
def test_warp_exit_metadata(tmp_path, obligation):
    compile_query(
        tmp_path,
        META + f"Example checked:{obligation}. Proof. vm_compute. reflexivity. Qed.\n",
    )


def test_warp_exit_theorems_closed(tmp_path):
    names = [
        "warp_level_step_sound",
        "warp_exit_reach",
        "warp_trace_agree",
        "warp_exit_trace",
        "warp_exit_check_sound",
        "front_fixed_event_check_sound",
        "front_event_check_sound",
        "accepted_front_events",
        "linked_warp_exit_schedule_sound",
        "accepted_front_linked_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateContext OpGate.GateFront OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
