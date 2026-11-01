"""Prepared joins preserve constraint order and predicate recovery exactly."""

import subprocess

from opgate.gate import formal_dir


def test_prepared_join_exact_and_eligibility(tmp_path):
    source = r"""
From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain GateSoundTransfer.
Import ListNotations.
Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 1%positive 1%nat (PM.empty nat); lf_fixed := []; lf_bounds := []; lf_aligns := [];
 lf_roles := []; lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit;
 lf_derived := []; lf_names := []
|}.
Definition empty : astate := {|amap:=PM.empty fact;acons:=[]|}.
Definition cmp0:cmpf:={|cc:=CEq;cw:=64;csg:=false;ca:=2%positive;cb:=OImm 0|}.
Definition exact_pred:astate:={|amap:=PM.add 1%positive(FPred [cmp0] true)(PM.empty fact);acons:=[]|}.
Definition inexact_pred:astate:={|amap:=PM.add 1%positive(FPred [cmp0] false)(PM.empty fact);acons:=[]|}.
Definition inexact_neg:astate:={|amap:=PM.add 1%positive(FNeg [cmp0] false)(PM.empty fact);acons:=[]|}.
Definition exact_neg:astate:={|amap:=PM.add 1%positive(FNeg [cmp0] true)(PM.empty fact);acons:=[]|}.
Definition bit(v:Z):astate:={|amap:=PM.add 1%positive(FForm(fconst 1 v)0 None)(PM.empty fact);acons:=[]|}.
Example missing_is_impossible:join_bit_possible facts empty 1%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example inexact_pred_is_impossible:join_bit_possible facts inexact_pred 1%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example inexact_neg_is_impossible:join_bit_possible facts inexact_neg 1%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example exact_pred_is_eligible:join_bit_possible facts exact_pred 1%positive=true.
Proof. vm_compute. reflexivity. Qed.
Example exact_neg_is_eligible:join_bit_possible facts exact_neg 1%positive=true.
Proof. vm_compute. reflexivity. Qed.
Example constant_zero_eligible:join_bit_possible facts(bit 0)1%positive=true.
Proof. vm_compute. reflexivity. Qed.
Example constant_one_eligible:join_bit_possible facts(bit 1)1%positive=true.
Proof. vm_compute. reflexivity. Qed.
Example both_equal:join_predicate_bits_ready facts(bit 1)(bit 1)1%positive None=Some(FForm(fconst 1 1)0 None).
Proof. vm_compute. reflexivity. Qed.
Example constants_different:join_predicate_bits_ready facts(bit 0)(bit 1)1%positive None=None.
Proof. vm_compute. reflexivity. Qed.
Example impossible_right_preserves:join_predicate_bits_ready facts exact_pred inexact_neg 1%positive (Some(FPred [] false))=Some(FPred [] false).
Proof. vm_compute. reflexivity. Qed.
Example impossible_left_preserves:join_predicate_bits_ready facts inexact_neg exact_pred 1%positive (Some(FPred [] false))=Some(FPred [] false).
Proof. vm_compute. reflexivity. Qed.
Definition k:constraint:=([((2%positive,EX),1)],0,5).
Definition narrow:astate:={|amap:=PM.empty fact;acons:=[k]|}.
Example kept_duplicates_order:prepared_keep_two facts false narrow [k;k][] =([k;k],[]).
Proof. vm_compute. reflexivity. Qed.
Example widening_rejects_larger:prepared_keep_two facts true narrow [([((2%positive,EX),1)],1,4)][] =([],[]).
Proof. vm_compute. reflexivity. Qed.
Example no_pending_empty:keep_pending []=false.
Proof. reflexivity. Qed.
Print Assumptions join_bit_impossible.
Print Assumptions join_predicate_bits_ready_exact.
Print Assumptions prepared_keep_two_exact.
Print Assumptions join_kept_prepared_exact.
"""
    path = tmp_path / "JoinControls.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 4, result.stdout
