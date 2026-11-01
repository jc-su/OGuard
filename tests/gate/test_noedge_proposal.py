"""Exact worklist result when the program has no internal CFG edge."""

import subprocess

from opgate.gate import formal_dir

SOURCE = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain.
Import ListNotations.
Example empty_program_has_no_edges : no_internal_edges []=true.
Proof. reflexivity. Qed.
Example out_of_range_branch_has_no_internal_edge : no_internal_edges [[IBra None 8]]=true.
Proof. reflexivity. Qed.
Example self_loop_requires_original_route : no_internal_edges [[IBra None 0]]=false.
Proof. reflexivity. Qed.
Example guarded_self_loop_requires_original_route : no_internal_edges [[IBra(Some(true,1%positive))0]]=false.
Proof. reflexivity. Qed.
Example fallthrough_requires_original_route : no_internal_edges [[INop];[IExit None]]=false.
Proof. reflexivity. Qed.
Example independent_exits_have_no_internal_edge : no_internal_edges [[IExit None];[IExit None]]=true.
Proof. reflexivity. Qed.
Example no_edge_fuel_boundary : forall F entry visits,
 iterate F 1 [[IExit None]] entry visits [0%nat]=None /\
 iterate F 2 [[IExit None]] entry visits [0%nat]=Some entry.
Proof. intros. split;reflexivity. Qed.
Example no_edge_invalid_work_preserved : forall F entry visits,
 iterate F 3 [[IExit None]] entry visits [42%nat;2%nat]=Some entry.
Proof. intros. reflexivity. Qed.

Example duplicate_work_consumes_fuel : forall F entry visits,
 iterate F 2 [[IExit None]] entry visits [0%nat;0%nat]=None /\
 iterate F 3 [[IExit None]] entry visits [0%nat;0%nat]=Some entry.
Proof. intros. split;reflexivity. Qed.
Example zero_fuel_empty_work_remains_none : forall F entry visits,
 iterate F 0 [[IExit None]] entry visits []=None.
Proof. reflexivity. Qed.
Example missing_certificate_still_refused : forall F,
 check_thread F [[IExit None]] []=false.
Proof. intros. unfold check_thread,check_cases. cbn.
 destruct (lf_ok F);reflexivity. Qed.
Example exact_original_proposal : forall F fuel prog entry visits work,
 iterate F fuel prog entry visits work =
 iterate_with_boundary F
 (trim_join_live_state F (live_program F prog) (live_incoming_controls prog))
 fuel prog (revisit_limits prog) entry visits work.
Proof. apply iterate_no_internal_edges_same. Qed.
Print Assumptions no_internal_edges_outside.
Print Assumptions no_internal_edges_iteration.
Print Assumptions iterate_no_internal_edges_same.
"""


def test_noedge_proposal_preserves_worklist_and_checker(tmp_path):
    query = tmp_path / "NoEdgeControls.v"
    query.write_text(SOURCE)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 3, result.stdout
