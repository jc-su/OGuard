"""Per-warp words through Numba range loops walked one iteration at a time; static CPU checks only."""

import subprocess

import pytest

from opgate.gate import formal_dir


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


# Register 1 is the thread index, 2 the counter, 3 the comparison's copy of
# the counter, 4 the back edge's predicate, and 5 the step: Numba's
# `for j in range(7, -1, -1)` counts from -6 while the copy is below 1 and
# steps by `selp 5, 1, 0, 4`. The barrier is the INop of block 2.
META = """From Coq Require Import List ZArith String.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.
Import ListNotations. Local Open Scope Z_scope. Local Open Scope string_scope.
Definition named:lfacts:={|
 lf_widths:=PM.add 1%positive 32%nat(PM.add 4%positive 1%nat(PM.empty nat));
 lf_fixed:=[1%positive;10%positive;11%positive];lf_bounds:=[(1%positive,(0,63))];
 lf_aligns:=[];lf_roles:=[];lf_shared_window:=10%positive;lf_local_window:=11%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];
 lf_names:=[("%tid.x",1%positive)]|}.
Definition loop(start:instr)(bound:Z):program:=[
 [start];
 [IMov None 3%positive 2%positive;ISetp None CLt 64 true 4%positive 3%positive(OImm bound);
  ISelp None 64 5%positive(OImm 1)(OImm 0)4%positive;IBin None BAdd 64 2%positive 2%positive(OReg 5%positive);
  IBra(Some(false,4%positive))1%nat];
 [INop;IExit None]].
Definition counted:=loop(IImm None 2%positive(-6))1.
Definition by_thread:=loop(IMov None 2%positive 1%positive)1.
Definition events:list pc:=[(2%nat,0%nat)].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "unrolled_loop_at counted events(1%nat,0%nat)<>None",
        "unrolled_loop_at counted[(1%nat,0%nat)](1%nat,0%nat)=None",
        "occurrence_compound_segments_supported counted events(2%nat,0%nat)=false",
        "occurrence_unrolled_segments_supported counted events(2%nat,0%nat)=true",
        "scoped_complete_word(warp_facts named 1%positive 0)counted events=Some events",
        "scoped_complete_word(warp_facts named 1%positive 1)counted events=Some events",
        # A start that depends on the thread leaves the back edge undecided.
        "scoped_complete_word(warp_facts named 1%positive 0)by_thread events=None",
        # A constant loop longer than the round budget is refused.
        "scoped_complete_word(warp_facts named 1%positive 0)(loop(IImm None 2%positive(-6))100)events=None",
    ],
)
def test_unrolled_obligations(tmp_path, obligation):
    compile_query(
        tmp_path, META + f"Example checked:{obligation}. Proof. vm_compute. congruence. Qed.\n"
    )


def test_unrolled_theorems_closed(tmp_path):
    names = [
        "unroll_iterate_sound",
        "unrolled_occurrence_summary_sound",
        "scoped_summary_select_sound",
        "scoped_complete_word_sound",
        "warp_scoped_check_sound",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
