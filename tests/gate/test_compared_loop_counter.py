"""Choose a checked compared counter before loop analysis; CPU/static only."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol


def compile_query(tmp_path, query):
    path = tmp_path / "Query.v"
    path.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("block", [16, 64, 256])
@pytest.mark.parametrize("address_after", [False, True])
def test_compared_loop_with_independent_address_counter(tmp_path, block, address_after):
    increments = ["add.u64 %rd1,%rd1,%rd2;", "add.u64 %rd3,%rd3,%rd4;"]
    if not address_after:
        increments.reverse()
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .b32 %r<3>;
.reg .b64 %rd<6>;
.reg .pred %p<2>;
ld.param.u64 %rd5,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd1,%r1;
cvt.u64.u32 %rd2,%r2;
shl.b64 %rd3,%rd1,2;
shl.b64 %rd4,%rd2,2;
setp.ge.u64 %p1,%rd1,%rd5;
@%p1 bra DONE;
LOOP:
{chr(10).join(increments)}
setp.lt.u64 %p1,%rd1,%rd5;
@%p1 bra LOOP;
DONE:
bar.sync 0;
bar.sync 0;
ret;
}}
"""
    binding = Binding(
        "kernel",
        (SizeSymbol("n", 0, 2147483647),),
        (("n", Size(Affine.symbol(0))),),
        (block, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


META = """From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateDomain GateBarrier GateLoop.
Import ListNotations.
Definition counter:=IBin None BAdd 64 1%positive 1%positive(OReg 2%positive).
Definition address:=IBin None BAdd 64 4%positive 4%positive(OReg 5%positive).
Definition P(after:bool):program:=[
 (if after then[counter;address]else[address;counter])++
 [ISetp None CLt 64 false 3%positive 1%positive(OImm 8);
  IBra(Some(false,3%positive))0%nat];[INop;IExit None]].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "match loop_at(P true)(0%nat,0%nat)with Some d=>loop_counter_compared(P true)(0%nat,0%nat)d|None=>true end=false",
        "match loop_at(P false)(0%nat,0%nat)with Some d=>loop_counter_compared(P false)(0%nat,0%nat)d|None=>false end=true",
        "loop_summary_at(P true)(0%nat,0%nat)=option_map Loop_dag_descriptor(loop_dag_at(P true)(0%nat,0%nat))",
        "loop_summary_at(P false)(0%nat,0%nat)=option_map Loop_straight_descriptor(loop_at(P false)(0%nat,0%nat))",
        "loop_dag_selected(P true)=true",
        "loop_dag_selected(P false)=false",
    ],
)
def test_structural_selection(tmp_path, obligation):
    compile_query(
        tmp_path, META + f"Example checked:{obligation}. Proof. vm_compute. reflexivity. Qed.\n"
    )


def test_selection_soundness_interfaces_remain_closed(tmp_path):
    names = [
        "loop_summary_straight_preserved",
        "loop_summary_compared_selected",
        "loop_summary_validate_progress",
        "ranked_summary_all_progress",
        "event_progress_general_sound",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
