"""Retain only current incoming facts consumed by exit-bound proposals."""

import subprocess

from opgate.gate import formal_dir

SOURCE = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning.
Import ListNotations.
Open Scope Z_scope.
Definition exit_bound_candidates_before (F:lfacts) (st other : astate) : list constraint :=
 let counter := fun (j b : reg) (sign : Z) =>
   if is_fixed F j || negb (Z.eqb (wd F j) 64) || negb (Z.eqb (wd F b) 64) then [] else
   match get st j, get other j with
   | Some (FForm a _ _), Some (FForm a' _ _) =>
       if Z.eqb (fm a) 64 && Z.eqb (fm a') 64 && terms_eqb (ft a) (ft a') &&
          Z.eqb (flo a' - flo a) 1 && Z.eqb (fhi a' - fhi a) 1 then
         let fb := cmp_form F other (OReg b) 64 false in
         let ts := sorted_terms (tscale sign (ft fb) ++ [((j, EX), - sign)]) in
         let c := if Z.eqb sign 1 then (ts, 1 - fhi fb, 2 ^ 64 - 1 - flo fb)
                  else (ts, flo fb - (2 ^ 64 - 1), fhi fb - 1) in
         if cons_holds F st c then if cons_holds F other c then [c] else [] else []
       else []
   | _, _ => []
   end in
 flat_map (fun pair => let '(_, x) := pair in
   match x with
   | FPred [k] _ =>
       match cc k, Nat.eqb (cw k) 64, cb k with
       | (CEq | CNe), true, OReg rb => counter rb (ca k) 1 ++ counter (ca k) rb (-1)
       | _, _, _ => []
       end
   | _ => []
   end) (PM.elements (amap other)).

Theorem exit_bound_candidates_before_same : forall F st other,
 exit_bound_candidates F st other=exit_bound_candidates_before F st other.
Proof.
 intros. unfold exit_bound_candidates,exit_bound_candidates_before.
 apply flat_map_ext. intros [r x].
 destruct x as [f al vr|al|cs exact|cs exact];try reflexivity.
 destruct cs as [|k cs];[reflexivity|]. destruct cs;[|reflexivity].
 unfold exit_bound_witness. destruct (cc k),(Nat.eqb (cw k) 64),(cb k);reflexivity.
Qed.
Definition trim_live_state_before F (table:list(PM.t unit)) b st:=
 let keep:=live_dependency_closure F(S(PM.cardinal(amap st)+length(acons st)))st
 (nth b table(PM.empty unit))in
 whole_state_filter keep(fun k=>forallb(fun r=>is_fixed F r||PM.mem r keep)(constraint_inputs k))st.
Theorem existing_trim_wrapper_same : forall F table b st,
 trim_live_state F table b st=trim_live_state_before F table b st.
Proof. reflexivity. Qed.
Definition facts:lfacts:={|lf_widths:=PM.empty nat;lf_fixed:=[];
 lf_bounds:=[];lf_aligns:=[];lf_roles:=[];lf_shared_window:=10%positive;
 lf_local_window:=11%positive;lf_shared:=[];lf_locals:=[];lf_cells:=[];
 lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition marks(rs:list reg):=mark_regs rs(PM.empty unit).
Definition comparison c w rhs:cmpf:={|cc:=c;cw:=w;csg:=false;ca:=1%positive;cb:=rhs|}.
Definition eq64:=comparison CEq 64(OReg 2%positive).
Definition ne64:=comparison CNe 64(OReg 2%positive).
Definition state(f:fact):astate:={|amap:=PM.add 4%positive f
 (PM.add 1%positive(FAlign 1)(PM.add 2%positive(FAlign 2)
 (PM.add 77%positive(FAlign 7)(PM.empty fact))));acons:=[]|}.
Definition seed f:=live_join_seed [marks []][marks [4%positive]]0(state f).
Example exact_equality_selected:PM.mem 4%positive(seed(FPred[eq64]true))=true.
Proof. vm_compute. reflexivity. Qed.
Example nonexact_equality_selected:PM.mem 4%positive(seed(FPred[eq64]false))=true.
Proof. vm_compute. reflexivity. Qed.
Example exact_disequality_selected:PM.mem 4%positive(seed(FPred[ne64]true))=true.
Proof. vm_compute. reflexivity. Qed.
Example nonexact_disequality_selected:PM.mem 4%positive(seed(FPred[ne64]false))=true.
Proof. vm_compute. reflexivity. Qed.
Example signed_flag_does_not_change_original_eligibility:
 exit_bound_witness(FPred[{|cc:=CEq;cw:=64;csg:=true;ca:=1%positive;cb:=OReg 2%positive|}]true)
 =Some(2%positive,1%positive).
Proof. reflexivity. Qed.
Example wrong_width_not_added:PM.mem 4%positive(seed(FPred[comparison CEq 32(OReg 2%positive)]true))=false.
Proof. vm_compute. reflexivity. Qed.
Example immediate_not_added:PM.mem 4%positive(seed(FPred[comparison CEq 64(OImm 0)]true))=false.
Proof. vm_compute. reflexivity. Qed.
Example ordering_not_added:PM.mem 4%positive(seed(FPred[comparison CLt 64(OReg 2%positive)]true))=false.
Proof. vm_compute. reflexivity. Qed.
Example compound_not_added:PM.mem 4%positive(seed(FPred[eq64;ne64]true))=false.
Proof. vm_compute. reflexivity. Qed.
Example negative_fact_not_added:PM.mem 4%positive(seed(FNeg[eq64]true))=false.
Proof. vm_compute. reflexivity. Qed.
Example empty_predicate_not_added:PM.mem 4%positive(seed(FPred[]true))=false.
Proof. vm_compute. reflexivity. Qed.
Example normalized_bit_not_added:PM.mem 4%positive(seed(FForm(fconst 1 1)0 None))=false.
Proof. vm_compute. reflexivity. Qed.
Example nonincoming_predicate_not_added:
 PM.mem 4%positive(live_join_seed [marks []][marks []]0(state(FPred[eq64]true)))=false.
Proof. vm_compute. reflexivity. Qed.
Example execution_live_ineligible_fact_stays_whole:
 get(trim_join_live_state facts [marks[4%positive]][marks[4%positive]]0(state(FPred[eq64;ne64]true)))4%positive
 =Some(FPred[eq64;ne64]true).
Proof. vm_compute. reflexivity. Qed.
Example actual_operands_retained_by_dynamic_closure:
 map(get(trim_join_live_state facts [marks []][marks[4%positive]]0(state(FPred[eq64]true))))
 [1%positive;2%positive;77%positive]=[Some(FAlign 1);Some(FAlign 2);None].
Proof. vm_compute. reflexivity. Qed.
Example exact_noedge_current_boundary : forall F fuel prog entry visits work,
 iterate F fuel prog entry visits work = iterate_with_boundary F
 (trim_join_live_state F(live_program F prog)(live_incoming_controls prog))fuel prog(revisit_limits prog)entry visits work.
Proof. apply iterate_no_internal_edges_same. Qed.
Print Assumptions exit_bound_candidates_before_same.
Print Assumptions existing_trim_wrapper_same.
Print Assumptions trim_live_seed_sound.
Print Assumptions trim_join_live_state_sound.
Print Assumptions exact_noedge_current_boundary.
"""


def test_join_witness_eligibility_and_whole_fact_weakening(tmp_path):
    query = tmp_path / "JoinWitnessControls.v"
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
