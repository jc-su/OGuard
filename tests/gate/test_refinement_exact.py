"""Comparison refinement skips only computations that cannot affect its result."""

import subprocess

from opgate.gate import formal_dir


def test_refinement_preserves_every_scalar_and_predicate_result(tmp_path):
    source = r"""From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain.
Import ListNotations.
Open Scope Z_scope.
Definition refine_cmp_old (F:lfacts) (st : astate) (k : cmpf) : option astate :=
  let w := Z.of_nat (cw k) in
  let sg := csg k in
  let a := ca k in
  let b := cb k in
  if negb (cmp_wf F k) then Some st else
  let fa := cmp_form F st (OReg a) w sg in
  let fb := cmp_form F st b w sg in
  (* both compared values lie in [lo_ok, hi_ok], so their difference lies
     within 2^w - 1 of zero *)
  let far := 2 ^ w - 1 in
  match cc k with
  | CGt | CGe =>
      let d := match cc k with CGt => 1 | _ => 0 end in
      match b with
      | OReg rb =>
          (* a > b is b < a *)
          let ra := cmp_range F st (OReg a) w sg in
          let rbr := cmp_range F st b w sg in
          let st1 := match ra with Some (_, y) => constrain F st b None (Some (y - d)) w sg | None => Some st end in
          match st1 with
          | None => None
          | Some s1 =>
              let s2 := match rbr with Some (x, _) => constrain F s1 (OReg a) (Some (x + d)) None w sg | None => Some s1 end in
              match s2, ra, rbr with
              | Some s2, Some _, Some _ => settle F s2 (relate_cmp F s2 a fa fb w sg d far)
              | _, _, _ => s2
              end
          end
      | OImm z =>
          match constrain F st (OReg a) (Some (extend (cw k) sg z + d)) None w sg with
          | Some s1 => relate_atom F s1 a fb w sg d far
          | None => None
          end
      end
  | CLt | CLe =>
      let d := match cc k with CLt => 1 | _ => 0 end in
      let ra := cmp_range F st (OReg a) w sg in
      let rbr := cmp_range F st b w sg in
      let st1 := match rbr with Some (_, y) => constrain F st (OReg a) None (Some (y - d)) w sg | None => Some st end in
      match st1 with
      | None => None
      | Some s1 =>
          let s2 := match ra, b with Some (x, _), OReg _ => constrain F s1 b (Some (x + d)) None w sg | _, _ => Some s1 end in
          match s2, ra, rbr, b with
          | Some s2, Some _, Some _, OReg _ => settle F s2 (relate_cmp F s2 a fa fb w sg (- far) (- d))
          | Some s2, _, _, OImm _ => relate_atom F s2 a fb w sg (- far) (- d)
          | _, _, _, _ => s2
          end
      end
  | CEq | CNe =>
      (* a = b exactly when D = a - b is 0 modulo 2^w; when D's range lies
         inside (-2^w, 2^w), exactly when D = 0 *)
      let D := normalize F st (fadd (cmp_form F st (OReg a) w sg) (fscale (cmp_form F st b w sg) (-1) 64)) in
      if fm D <? w then Some st else
      if null (ft D) && Z.eqb (flo D) (fhi D) then
        let zero := Z.eqb (flo D mod 2 ^ w) 0 in
        if Bool.eqb zero (match cc k with CEq => true | _ => false end) then Some st else None
      else
      match form_range F st D with
      | None => Some st
      | Some (x, y) =>
          if (x <=? - 2 ^ w) || (2 ^ w <=? y) then Some st else
          let al := Z.min (Z.min (reg_align F st a) (opnd_align F st b)) (w - 1) in
          let g := round al (x, y) in
          match cc k with
          | CEq => constrain_form F st st D 0 0 (Some g)
          | _ => if Z.eqb (fst g) 0 then constrain_form F st st D (2 ^ al) (snd g) (Some g)
                 else if Z.eqb (snd g) 0 then constrain_form F st st D (fst g) (- 2 ^ al) (Some g)
                 else Some st
          end
      end
  end.
Theorem refine_cmp_raw_exact F st k : refine_cmp_raw F st k = refine_cmp_old F st k.
Proof.
 unfold refine_cmp_raw,refine_cmp_old.
 destruct(cmp_wf F k);cbn beta iota zeta delta [negb];[|reflexivity].
 destruct(cc k);try reflexivity;destruct(cb k);try reflexivity;
 destruct(cmp_range F st(OReg(ca k))(Z.of_nat(cw k))(csg k))as[[x y]|];
 destruct(cmp_range F st(OImm z)(Z.of_nat(cw k))(csg k))as[[u v]|];
 repeat match goal with
 | |- context[match constrain F ?s ?o ?l ?h ?w ?sg with _ => _ end] =>
     destruct(constrain F s o l h w sg)
 end;reflexivity.
Qed.
Print Assumptions refine_cmp_raw_exact.

Lemma refine_cmp_old_same F st k : refine_cmp_old F st k=refine_cmp_raw F st k.
Proof. symmetry. apply refine_cmp_raw_exact. Qed.
Fixpoint refine_all_old F st cs : option astate :=
 match cs with [] => Some st | k::rest =>
 match refine_cmp_old F st (unsigned_equality k) with None => None | Some s => refine_all_old F s rest end end.
Definition refine_cmps_old F st cs (exact all:bool) :=
 if all then refine_all_old F st cs
 else if negb exact then Some st
 else join_alternatives st (flat_map (fun k => match refine_cmp_old F st (unsigned_equality (negate k)) with Some s => [s] | None => [] end) cs).
Definition refine_pred_old F st p holds :=
 match get st p with
 | Some(FPred cs exact) => refine_cmps_old F st cs exact holds
 | Some(FNeg cs exact) => refine_cmps_old F st cs exact (negb holds)
 | Some(FForm f _ _) =>
 if null(ft f) && Z.eqb(flo f)(fhi f) && (1 <=? fm f) && Z.eqb(wd F p)1 then
 if Bool.eqb(negb(Z.eqb(flo f mod 2)0))holds then Some st else None else Some st
 | _ => Some st end.
Definition refine_old F st p holds :=
 match refine_pred_old F st p holds with
 | Some s => Some(if Z.eqb(wd F p)1 then subst_pred s p(if holds then 1 else 0) else s)
 | None => None end.
Lemma refine_all_old_same F st cs : refine_all_old F st cs = refine_all F st cs.
Proof. revert st; induction cs as[|k cs IH];intros st;cbn [refine_all_old refine_all];auto.
 rewrite refine_cmp_old_same; unfold refine_cmp;destruct(refine_cmp_raw F st(unsigned_equality k));auto. Qed.
Lemma refine_cmps_old_same F st cs exact all : refine_cmps_old F st cs exact all = refine_cmps F st cs exact all.
Proof. unfold refine_cmps_old,refine_cmps;destruct all;[apply refine_all_old_same|].
 destruct exact;cbn;[|reflexivity]. f_equal. apply flat_map_ext. intros k. rewrite refine_cmp_old_same. reflexivity. Qed.
Lemma refine_pred_old_same F st p holds : refine_pred_old F st p holds = refine_pred F st p holds.
Proof. unfold refine_pred_old,refine_pred;destruct(get st p)as[f|];[destruct f|];try reflexivity;apply refine_cmps_old_same. Qed.
Theorem refine_old_same F st p holds : refine_old F st p holds = refine F st p holds.
Proof. unfold refine_old,refine. rewrite refine_pred_old_same. reflexivity. Qed.
Print Assumptions refine_old_same.

From OpGate Require Import GateSoundRefine.
Definition facts (width:nat) : lfacts := {|
 lf_widths:=PM.add 1%positive width(PM.add 2%positive width(PM.empty nat));
 lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;lf_shared:=[];
 lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition interval_state : astate :=
 {|amap:=PM.add 1%positive(FForm(range_form 64 0 3)0 None)
   (PM.add 2%positive(FForm(fconst 64 4)0 None)(PM.empty fact));acons:=[]|}.
Definition comparison c w sg rhs : cmpf := {|cc:=c;cw:=w;csg:=sg;ca:=1%positive;cb:=rhs|}.
Definition feasible (s:option astate):bool:=match s with Some _=>true|None=>false end.
Example less_immediate_impossible:
 refine_cmp_raw(facts 64)interval_state(comparison CLt 64 false(OImm 0))=None.
Proof. vm_compute. reflexivity. Qed.
Example greater_equal_immediate_impossible:
 refine_cmp_raw(facts 64)interval_state(comparison CGe 64 false(OImm 4))=None.
Proof. vm_compute. reflexivity. Qed.
Example less_register_possible:
 feasible(refine_cmp_raw(facts 64)interval_state(comparison CLt 64 false(OReg 2%positive)))=true.
Proof. vm_compute. reflexivity. Qed.
Example less_equal_immediate_possible:
 feasible(refine_cmp_raw(facts 64)interval_state(comparison CLe 64 false(OImm 3)))=true.
Proof. vm_compute. reflexivity. Qed.
Example mismatched_width_still_unchanged:
 refine_cmp_raw(facts 32)interval_state(comparison CLt 64 true(OImm 0))=Some interval_state.
Proof. vm_compute. reflexivity. Qed.
Definition minus_one (w:Z):astate:=
 {|amap:=PM.add 1%positive(FForm(fconst w(2^w-1))0 None)(PM.empty fact);acons:=[]|}.
Example signed32_negative_retained:
 feasible(refine_cmp_raw(facts 32)(minus_one 32)(comparison CLt 32 true(OImm 0)))=true.
Proof. vm_compute. reflexivity. Qed.
Example signed64_negative_retained:
 feasible(refine_cmp_raw(facts 64)(minus_one 64)(comparison CLt 64 true(OImm 0)))=true.
Proof. vm_compute. reflexivity. Qed.
Example unsigned32_negative_is_large:
 refine_cmp_raw(facts 32)(minus_one 32)(comparison CLt 32 false(OImm 0))=None.
Proof. vm_compute. reflexivity. Qed.
Example unsigned64_negative_is_large:
 refine_cmp_raw(facts 64)(minus_one 64)(comparison CLt 64 false(OImm 0))=None.
Proof. vm_compute. reflexivity. Qed.
Example equality_impossible:
 refine_cmp_raw(facts 64)interval_state(comparison CEq 64 false(OImm 4))=None.
Proof. vm_compute. reflexivity. Qed.
Example disequality_possible:
 feasible(refine_cmp_raw(facts 64)interval_state(comparison CNe 64 false(OImm 4)))=true.
Proof. vm_compute. reflexivity. Qed.
Print Assumptions refine_cmp_raw_sound.
Print Assumptions refine_sound.
"""
    path = tmp_path / "RefinementExact.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 4, result.stdout
