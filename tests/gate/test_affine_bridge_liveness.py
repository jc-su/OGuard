"""Preserve justified bounds carried by dead interval temporaries."""

import subprocess

from opgate.gate import formal_dir

SOURCE = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning.
Import ListNotations. Local Open Scope Z_scope.
Definition facts (bridge_width:nat):lfacts:={|
 lf_widths:=PM.add 9%positive bridge_width
   (PM.add 2%positive 32%nat (PM.add 3%positive 32%nat (PM.empty nat)));
 lf_fixed:=[1%positive;2%positive;3%positive];
 lf_bounds:=[(1%positive,(1,2147483647));(2%positive,(0,8388607));(3%positive,(0,255))];
 lf_aligns:=[];lf_roles:=[];lf_shared_window:=100%positive;
 lf_local_window:=101%positive;lf_shared:=[];lf_locals:=[];lf_cells:=[];
 lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition F:=facts 64.
Definition interval:form:={|fm:=64;ft:=[];flo:=0;fhi:=256|}.
Definition bridge:fact:=FForm interval 8 None.
Definition anchor:constraint:=
 (sorted_terms [((1%positive,EX),-1);((2%positive,EZ),256);((9%positive,EX),1)],
  -2147483647,0).
Definition guard:constraint:=
 (sorted_terms [((9%positive,EX),1);((3%positive,EZ),-1)],1,256).
Definition unrelated:constraint:=([((10%positive,EX),1);((11%positive,EX),-1)],0,9).
Definition empty:astate:={|amap:=PM.empty fact;acons:=[]|}.
Definition state (value:fact) (cs:list constraint):astate:={|
 amap:=PM.add 9%positive value
   (PM.add 10%positive (FAlign 16) (PM.add 11%positive (FAlign 32) (PM.empty fact)));
 acons:=cs|}.
Definition original:=state bridge [anchor;guard;unrelated].
Definition retained:=retain_affine_bridges F original empty.
Example interval_affine_bridge_selected:affine_bridge_candidate F original empty 9%positive=true.
Proof. vm_compute. reflexivity. Qed.
Example fixed_launch_input_not_selected:affine_bridge_candidate F original empty 1%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example already_retained_not_selected:affine_bridge_candidate F original original 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example absent_not_selected:affine_bridge_candidate F original empty 99%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example wrong_width_not_selected:affine_bridge_candidate (facts 32) original empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example point_value_not_selected:
 affine_bridge_candidate F (state (FForm (fconst 64 256) 8 None) [anchor;guard]) empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example symbolic_value_not_selected:
 affine_bridge_candidate F (state (FForm (fatom 64 (1%positive,EX) 1) 8 None) [anchor;guard]) empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example separate_value_range_not_selected:
 affine_bridge_candidate F (state (FForm interval 8 (Some (0,256))) [anchor;guard]) empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example alignment_only_not_selected:
 affine_bridge_candidate F (state (FAlign 8) [anchor;guard]) empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example unanchored_guard_not_selected:
 affine_bridge_candidate F (state bridge [guard]) empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example extra_dynamic_dependency_not_selected:
 affine_bridge_candidate F (state bridge
 [([((1%positive,EX),-1);((2%positive,EZ),256);((9%positive,EX),1);((10%positive,EX),1)],0,100)])
 empty 9%positive=false.
Proof. vm_compute. reflexivity. Qed.
Example whole_fact_and_alignment_retained:get retained 9%positive=Some bridge.
Proof. vm_compute. reflexivity. Qed.
Example original_constraints_retained:acons retained=[anchor;guard].
Proof. vm_compute. reflexivity. Qed.
Example unrelated_facts_still_dropped:map (get retained) [10%positive;11%positive]=[None;None].
Proof. vm_compute. reflexivity. Qed.
Example empty_bridge_is_exact:
 retain_affine_bridges F (state (FAlign 8) [guard]) original=original.
Proof. vm_compute. reflexivity. Qed.
Definition goal:=form_of_terms(sorted_terms
 [((1%positive,EX),1);((2%positive,EZ),-256);((3%positive,EZ),-1)]).
Example whole_bridge_projects_positive_output_bound:
 project_range F retained [] goal=Some(1,2147483647).
Proof. vm_compute. reflexivity. Qed.
Example missing_guard_does_not_imply_positive_output_bound:
 project_range F (state bridge [anchor]) [] goal=Some(-255,2147483647).
Proof. vm_compute. reflexivity. Qed.
Example two_distinct_fixed_anchors_seed_bridge:
 projection_fixed_seeds F (ft goal) [anchor]=[(9%positive,EX)].
Proof. vm_compute. reflexivity. Qed.
Example repeated_fixed_atom_does_not_seed:
 projection_fixed_seeds F [((1%positive,EX),1);((1%positive,EX),2)] [anchor]=[].
Proof. vm_compute. reflexivity. Qed.
Example single_fixed_hub_does_not_seed:
 projection_fixed_seeds F (ft goal) [guard]=[].
Proof. vm_compute. reflexivity. Qed.
Definition literal_old_slice Fc ts cs:=
 let xs:=projection_component Fc (length cs) cs (projection_moving Fc ts) in
 filter (fun k=>let '(ks,_,_):=k in
   if null (projection_moving Fc ks) then true else projection_touches Fc xs ks) cs.
Theorem moving_goal_keeps_previous_selection:forall Fc ts cs,
 projection_moving Fc ts<>[] -> projection_slice Fc ts cs=literal_old_slice Fc ts cs.
Proof.
 intros Fc ts cs H. unfold projection_slice,literal_old_slice.
 destruct (projection_moving Fc ts);[contradiction|reflexivity].
Qed.
Theorem selected_constraints_are_original:forall Fc ts cs k,
 In k (projection_slice Fc ts cs)->In k cs.
Proof. intros. unfold projection_slice in H. apply filter_In in H. tauto. Qed.
Theorem selected_constraints_are_justified:forall Fc rs ts cs,
 Forall (cons_ok Fc rs) cs->Forall (cons_ok Fc rs) (projection_slice Fc ts cs).
Proof.
 intros. rewrite Forall_forall in *. intros k Hk. apply H.
 eapply selected_constraints_are_original;exact Hk.
Qed.
Print Assumptions moving_goal_keeps_previous_selection.
Print Assumptions selected_constraints_are_original.
Print Assumptions selected_constraints_are_justified.
Print Assumptions retain_affine_bridges_sound.
Print Assumptions trim_join_live_state_sound.
"""


def test_affine_bridge_bounds_and_component_isolation(tmp_path):
    query = tmp_path / "AffineBridgeControls.v"
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
    assert result.stdout.count("Closed under the global context") == 5, result.stdout
