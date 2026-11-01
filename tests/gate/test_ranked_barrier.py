"""CPU-only progress and ordering controls; no PTX executes on a GPU."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol

BINDING = Binding(
    "kernel",
    (SizeSymbol("n", 0, 2147483647),),
    (("n", Size(Affine.symbol(0))),),
    (256, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def _compile(tmp_path, query, *, timeout=30):
    path = tmp_path / "Query.v"
    path.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("split_prefix", [False, True])
def test_input_sized_loop_between_ordered_events(tmp_path, split_prefix):
    # The maximum input needs millions of concrete iterations. Verification
    # checks a decreasing rank instead of enumerating those iterations.
    prefix = "bra BODY;\nBODY:\n" if split_prefix else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<3>;
.reg .b32 %r<4>;
.reg .b64 %rd<5>;
ld.param.u64 %rd1,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd2,%r1;
cvt.u64.u32 %rd3,%r2;
bar.sync 0;
setp.ge.s64 %p1,%rd2,%rd1;
@%p1 bra DONE;
LOOP:
{prefix}add.u32 %r3,%r1,1;
add.s64 %rd2,%rd2,%rd3;
setp.lt.s64 %p2,%rd2,%rd1;
@%p2 bra LOOP;
DONE:
bar.sync 0;
ret;
}}
"""
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(BINDING))
    _compile(tmp_path, export_terms(text, [BINDING]) + CHECK_QUERY)


def test_rank_and_event_theorems_are_closed(tmp_path):
    names = [
        "loop_step_preserves",
        "loop_linear_sound",
        "loop_alias_sound",
        "loop_increment_sound",
        "loop_branch_sound",
        "loop_iteration_sound",
        "loop_validate_progress",
        "ranked_phase_check_sound",
        "loop_events_check_sound",
        "loop_events_in_order",
        "accepted_front_events",
    ]
    query = "Require Import OpGate.GateLoop OpGate.GateFront.\n"
    query += "".join(f"Print Assumptions {name}.\n" for name in names)
    result = _compile(tmp_path, query)
    assert result.stdout.count("Closed under the global context") == len(names)


def test_loop_metadata_rejects_missing_premises(tmp_path):
    # These are direct checker metadata controls, not executable kernels.
    _compile(
        tmp_path,
        """
From Coq Require Import List ZArith NArith.
Require Import OpGate.GateIsa OpGate.GateLoop OpGate.GateBarrier.
Import ListNotations.
Example protected_counter_write :
 loop_preserves 1%positive (IImm None 1%positive 5)=false.
Proof. vm_compute. reflexivity. Qed.
Example protected_multidestination_write :
 loop_preserves 1%positive (IDef None [2%positive;1%positive])=false.
Proof. vm_compute. reflexivity. Qed.
Example unrelated_register_write :
 loop_preserves 1%positive (IImm None 2%positive 5)=true.
Proof. vm_compute. reflexivity. Qed.
Example prefix_conditional_side_edge :
 loop_next [[INop]] (0%nat,0%nat) (IBra (Some (false,1%positive)) 0)=None.
Proof. vm_compute. reflexivity. Qed.
Example prefix_exit : loop_next [[INop]] (0%nat,0%nat) (IExit None)=None.
Proof. vm_compute. reflexivity. Qed.
Example repeated_event : loop_events_ordered [(0%nat,0%nat);(0%nat,0%nat)]=false.
Proof. vm_compute. reflexivity. Qed.
Example reverse_event_order : loop_events_ordered [(2%nat,0%nat);(1%nat,0%nat)]=false.
Proof. vm_compute. reflexivity. Qed.
Example forward_event_order : loop_events_ordered [(1%nat,0%nat);(2%nat,0%nat)]=true.
Proof. vm_compute. reflexivity. Qed.
""",
    )


@pytest.mark.parametrize("negated", [False, True])
@pytest.mark.parametrize("split_prefix", [False, True])
def test_copied_cursor_and_preincrement_guard(tmp_path, negated, split_prefix):
    prefix = "bra BODY;\nBODY:\n" if split_prefix else ""
    comparison = "ge" if negated else "lt"
    polarity = "!" if negated else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<3>;
.reg .b32 %r<4>;
.reg .b64 %rd<6>;
ld.param.u64 %rd1,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd2,%r1;
cvt.u64.u32 %rd3,%r2;
bar.sync 0;
setp.ge.s64 %p1,%rd2,%rd1;
@%p1 bra DONE;
LOOP:
{prefix}mov.b64 %rd4,%rd2;
setp.{comparison}.s64 %p2,%rd4,%rd1;
add.s64 %rd2,%rd4,%rd3;
@{polarity}%p2 bra LOOP;
DONE:
bar.sync 0;
ret;
}}
"""
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(BINDING))
    _compile(tmp_path, export_terms(text, [BINDING]) + CHECK_QUERY)


def test_generic_increment_requires_alias_progress_and_no_wrap(tmp_path):
    # Direct checker metadata only; no executable defective kernel is created.
    _compile(
        tmp_path,
        """
From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain GateLoop.
Import ListNotations.
Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 4%positive 1%nat (PM.empty nat);
 lf_fixed := []; lf_bounds := []; lf_aligns := []; lf_roles := [];
 lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit;
 lf_derived := []; lf_names := []
|}.
Definition d := {|li_prefix:=0;li_suffix:=0;li_counter:=1%positive;
 li_source:=3%positive;li_stride:=2%positive;li_predicate:=4%positive;li_negated:=false|}.
Definition interval (lo hi : Z) := FForm (range_form 64 lo hi) 0 None.
Definition before (hi stride : Z) (alias : fact) : astate := {|
 amap := PM.add 1%positive (interval 0 hi)
  (PM.add 2%positive (interval stride stride) (PM.add 3%positive alias (PM.empty fact)));
 acons := []|}.
Definition body := [[loop_counter_step d]].
Example valid_alias : loop_increment_ok facts body (0%nat,0%nat)
 (before 10 1 (loop_alias d)) d=true.
Proof. vm_compute. reflexivity. Qed.
Example unrelated_source : loop_increment_ok facts body (0%nat,0%nat)
 (before 10 1 (interval 0 10)) d=false.
Proof. vm_compute. reflexivity. Qed.
Example zero_stride : loop_increment_ok facts body (0%nat,0%nat)
 (before 10 0 (loop_alias d)) d=false.
Proof. vm_compute. reflexivity. Qed.
Example wrapping_increment : loop_increment_ok facts body (0%nat,0%nat)
 (before (2^64-1) 1 (loop_alias d)) d=false.
Proof. vm_compute. reflexivity. Qed.
Example suffix_counter_write : loop_linear 1 facts [[IImm None 1%positive 0]]
 1%positive 2%positive (0%nat,0%nat)
 (before 10 1 (loop_alias d))=None.
Proof. vm_compute. reflexivity. Qed.
Example suffix_stride_write : loop_linear 1 facts [[IImm None 2%positive 0]]
 1%positive 2%positive (0%nat,0%nat)
 (before 10 1 (loop_alias d))=None.
Proof. vm_compute. reflexivity. Qed.
""",
    )
