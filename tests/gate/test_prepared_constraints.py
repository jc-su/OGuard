"""Exact constraint preparation retains complete transport results."""

import subprocess

from opgate.gate import formal_dir


def test_prepared_constraints_exact_and_selective(tmp_path):
    source = r"""
From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain GateSoundJoin GateSoundTransfer.
Import ListNotations.
Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.empty nat; lf_fixed := []; lf_bounds := []; lf_aligns := [];
 lf_roles := []; lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit;
 lf_derived := []; lf_names := []
|}.
Definition interval (lo hi : Z) : fact := FForm {|fm:=64;ft:=[];flo:=lo;fhi:=hi|} 0 None.
Definition empty : astate := {|amap:=PM.empty fact;acons:=[]|}.
Definition ranges : astate := {|
 amap := PM.add 1%positive (interval 0 0)
         (PM.add 2%positive (interval 0 8) (PM.empty fact));acons:=[]|}.
Definition relation : constraint := ([((1%positive,EX),1);((2%positive,EX),-1)],0,0).
Definition pending := cons_prepare facts ranges relation.
Example pending_keeps_full_query : pending = inr relation.
Proof. vm_compute. reflexivity. Qed.
Example unknown_register_refuses :
 cons_prepare facts empty ([((1%positive,EX),1)],0,8) = inl false.
Proof. vm_compute. reflexivity. Qed.
Example near_decides_true :
 cons_prepare facts ranges ([((2%positive,EX),1)],0,8) = inl true.
Proof. vm_compute. reflexivity. Qed.
Example near_decides_false :
 cons_prepare facts ranges ([((2%positive,EX),1)],9,10) = inl false.
Proof. vm_compute. reflexivity. Qed.
Example empty_batch_needs_no_deep_query : prepared_pending [] = false.
Proof. reflexivity. Qed.
Example terminal_batch_needs_no_deep_query :
 prepared_pending [(relation,inl true);(relation,inl false)] = false.
Proof. reflexivity. Qed.
Example mixed_batch_keeps_pending :
 prepared_pending [(relation,inl false);(relation,pending);(relation,inl true)] = true.
Proof. vm_compute. reflexivity. Qed.
Example imm_source_has_no_jobs : branch_work facts ranges 64 3%positive (OImm 0) (OReg 2%positive) = [].
Proof. reflexivity. Qed.
Example width_mismatch_has_no_jobs : branch_work facts ranges 32 3%positive (OReg 1%positive) (OReg 2%positive) = [].
Proof. vm_compute. reflexivity. Qed.
Example duplicate_terminal_is_added_once :
 prepared_apply facts ranges (fun _=>None)
 [(([],1,1),inl true);(([],2,2),inl false);(([],3,3),inl true);(([],1,1),inl true)] empty =
 {|amap:=PM.empty fact;acons:=[([],3,3);([],1,1)]|}.
Proof. reflexivity. Qed.
Print Assumptions cons_holds_exact.
Print Assumptions cons_prepare_exact.
Print Assumptions branch_cons1_exact.
Print Assumptions branch_cons_exact.
Print Assumptions sel_cons_exact.
Print Assumptions branch_cons1_prepared_exact.
Print Assumptions branch_cons_prepared_exact.
Print Assumptions sel_cons_prepared_exact.
Print Assumptions sel_refined_cons_prepared_exact.
Print Assumptions selected_step_cached_exact.
"""
    path = tmp_path / "PreparedControls.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 10, result.stdout
