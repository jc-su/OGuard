(*
  Soundness of the transfer function: if a state describes the registers
  before a step, the transferred state describes them after it.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import ZArith.Znumtheory.
From Coq Require Import PArith.PArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import FSets.FMapPositive.
Import ListNotations.

Require Import OpGate.GateIsa.
Require Import OpGate.GateForm.
Require Import OpGate.GateDomain.
Require Import OpGate.GateMeaning.
Require Import OpGate.GateSoundRange.
Require Import OpGate.GateSoundNorm.
Require Import OpGate.GateSoundOp.
Require Import OpGate.GateSoundRefine.
Require Import OpGate.GateSoundJoin.
Require Import OpGate.GateSoundAccess.
Require Import OpGate.GateSoundLoad.
Require Import OpGate.GateSoundLocal.

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

Lemma v2_fuel_le : forall n z, v2_fuel n z <= Z.of_nat n.
Proof.
  induction n as [| n IH]; intros z; cbn [v2_fuel]; [lia |].
  destruct (Z.eqb (z mod 2) 0); [specialize (IH (z / 2)); lia | lia].
Qed.

Lemma some_eq_t : forall (A : Type) (a b : A), Some a = Some b -> b = a.
Proof. intros A a b H. injection H as H. symmetry. exact H. Qed.

Lemma fm_fadd : forall f g, fm (fadd f g) = Z.min (fm f) (fm g).
Proof. intros. unfold fadd. apply mk_fm. Qed.

Lemma v2_le : forall z, v2 z <= 64.
Proof. intros z. unfold v2. destruct (Z.eqb z 0); [lia | apply (v2_fuel_le 64)]. Qed.

(* Registers that agree with rs except at d. *)
Definition agree (d : reg) (rs rs' : reg -> Z) : Prop := forall r, r <> d -> rs' r = rs r.

Lemma tsum_agree : forall wdn d rs rs' ts,
    agree d rs rs' -> existsb (fun t => Pos.eqb (fst (fst t)) d) ts = false ->
    tsum wdn rs' ts = tsum wdn rs ts.
Proof.
  intros wdn d rs rs' ts Ha H. induction ts as [| [[r e] c] ts IH]; cbn [tsum]; [reflexivity |].
  cbn [existsb fst] in H. apply orb_false_iff in H. destruct H as [Hr H].
  rewrite (IH H). unfold atom_val. rewrite (Ha r); [reflexivity |].
  intros ->. rewrite Pos.eqb_refl in Hr. discriminate.
Qed.

Lemma tsum_ext : forall wdn rs rs' ts, (forall r, rs' r = rs r) -> tsum wdn rs' ts = tsum wdn rs ts.
Proof.
  intros wdn rs rs' ts H. induction ts as [| [[r e] c] ts IH]; cbn [tsum]; [reflexivity |].
  rewrite IH. unfold atom_val. rewrite (H r). reflexivity.
Qed.

Lemma sat_agree : forall wdn d rs rs' f v,
    agree d rs rs' -> mentions f d = false -> sat wdn rs f v -> sat wdn rs' f v.
Proof.
  intros wdn d rs rs' f v Ha Hm [e [He Hv]]. exists e. split; [exact He |].
  rewrite (tsum_agree wdn d rs rs' (ft f) Ha Hm). exact Hv.
Qed.

Lemma cmp_agree : forall d rs rs' k,
    agree d rs rs' -> cmp_mentions k d = false -> cmp_holds rs k -> cmp_holds rs' k.
Proof.
  intros d rs rs' k Ha Hm H. unfold cmp_mentions in Hm. apply orb_false_iff in Hm. destruct Hm as [Ha' Hb'].
  unfold cmp_holds in *. rewrite (Ha (ca k)) by (intros E; rewrite E, Pos.eqb_refl in Ha'; discriminate).
  destruct (cb k) as [r | z] eqn:Eb; cbn [opv] in *; [| exact H].
  rewrite (Ha r) by (intros E; rewrite E, Pos.eqb_refl in Hb'; discriminate). exact H.
Qed.

Lemma filter_length_le' : forall {A} (p : A -> bool) l, (length (filter p l) <= length l)%nat.
Proof.
  intros A p l. induction l as [| x l IH]; cbn [filter length]; [lia |].
  destruct (p x); cbn [length]; lia.
Qed.

Lemma filter_length_eq : forall {A} (p : A -> bool) l, length (filter p l) = length l -> filter p l = l.
Proof.
  intros A p l. induction l as [| x l IH]; cbn [filter length]; [reflexivity |].
  destruct (p x); cbn [length].
  - intros H. f_equal. apply IH. lia.
  - intros H. pose proof (filter_length_le' p l) as Hle. lia.
Qed.

(* The scalar selected-step expansion is a proof specification; production
   transfer uses the shared immutable-state evaluator. *)
Definition selected_step (F : lfacts) (st out : astate) (w : Z) (d r : reg) : astate :=
 if negb (Z.eqb w 64 && Z.eqb ((wd F) d) 64 && Z.eqb ((wd F) r) 64) || Pos.eqb d r then out else
 let delta := (normalize F) st (fadd ((as_form F) st r EX) (fscale ((as_form F) st d EX) (-1) 64)) in
 let delta := (materialized_form F) st delta d r in
 if negb (Z.eqb (fm delta) 64 && Z.eqb (flo delta) (fhi delta)) || mentions delta d then out else
 fold_left (fun out k => let '(ts,lo,hi) := k in
   if negb (cons_mentions d k) then out else
   let q := coef ts (d,EX) + coef ts (d,EZ) in
   let ts' := sorted_terms (ts ++ tscale (-q) (ft delta)) in
   match (check_range F) st (form_of_terms ts') with
   | None => out
   | Some (l,h) =>
     let candidate := (ts',Z.min l (lo+q*flo delta),Z.max h (hi+q*flo delta)) in
     (branch_cons1 F) {|amap:=amap st;acons:=[candidate]|} ((projected_constraint F) st (subst_cons d r candidate)) out w d (OReg d) (OReg r)
   end) (acons st) out.
(* Scalar folds are proof specifications for the common ordered jobs. *)
Definition branch_cons1_spec (F : lfacts) (st opposite st1 : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
  let move := fun (r : reg) (other : constraint -> constraint) =>
    if Z.eqb ((wd F) r) w && Z.eqb ((wd F) d) w then
      fold_left (fun s k => if (if cons_mentions r k && (negb (cons_mentions d k) || Pos.eqb r d)
                                then (cons_holds F) opposite (other k)
                                else false)
                            then add_cons s (subst_cons r d k) else s)
                (acons st) st1
    else st1 in
  match a, b with
  | OReg ra, OReg rb => if Z.eqb ((wd F) rb) w then move ra (subst_cons ra rb) else st1
  | OReg ra, OImm z => move ra ((const_cons F) ra (z mod 2 ^ w))
  | OImm _, _ => st1
  end.


Definition sel_cons1 (F : lfacts) (st st1 : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
  match a, b with
  | OImm _, OReg _ => (branch_cons1 F) st st st1 w d b a
  | _, _ => (branch_cons1 F) st st st1 w d a b
  end.

Definition sel_cons_spec (F : lfacts) (st st1 : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
  let st2 := (sel_cons1 F) st st1 w d a b in
  match a, b with
  | OReg ra, OReg rb => if Pos.eqb ra rb then st2 else (sel_cons1 F) st st2 w d b a
  | _, _ => st2
  end.


Definition branch_cons_spec (F : lfacts) (st other out : astate) (w : Z) (d : reg) (a b : opnd) : astate :=
  let base := (branch_cons1 F) st other out w d a b in
  let a' := (copy_operand F) st a in let b' := (copy_operand F) other b in
  let same := fun x y => match x, y with
                        | OReg r, OReg s => Pos.eqb r s
                        | OImm x, OImm y => Z.eqb x y
                        | _, _ => false
                        end in
  if same a a' && same b b' then base else (branch_cons1 F) st other base w d a' b'.

Section PreparedEquivalence.
Variable F : lfacts.
Lemma cons_prepare_exact : forall a k,
 prepared_run F a (check_range F a) (cons_prepare F a k) = cons_holds F a k.
Proof. reflexivity. Qed.

Lemma work_move_exact : forall st out r d other holds,
 work_apply holds
 (flat_map(fun k=>if cons_mentions r k && (negb(cons_mentions d k)||Pos.eqb r d)
 then [(subst_cons r d k,other k)] else []) (acons st)) out =
 fold_left(fun s k=>if(if cons_mentions r k && (negb(cons_mentions d k)||Pos.eqb r d)
 then holds(other k) else false) then add_cons s(subst_cons r d k) else s) (acons st) out.
Proof.
 intros st out r d other holds. unfold work_apply.
 generalize (acons st) as ks. intros ks. revert out.
 induction ks as [|k ks IH];intros out;cbn;[reflexivity|].
 destruct(cons_mentions r k && (negb(cons_mentions d k)||Pos.eqb r d));cbn;apply IH.
Qed.

Lemma branch_work_spec : forall st other out w d a b,
 work_apply (cons_holds F other) (branch_work F st w d a b) out = branch_cons1_spec F st other out w d a b.
Proof.
 intros. unfold branch_work,branch_cons1_spec. destruct a,b;cbn;try reflexivity.
 - destruct(Z.eqb(wd F r0)w);[|reflexivity].
   destruct(Z.eqb(wd F r)w && Z.eqb(wd F d)w);[apply work_move_exact|reflexivity].
 - destruct(Z.eqb(wd F r)w && Z.eqb(wd F d)w);[apply work_move_exact|reflexivity].
Qed.

Lemma branch_cons1_exact : forall st other out w d a b,
 branch_cons1 F st other out w d a b=branch_cons1_spec F st other out w d a b.
Proof. intros. unfold branch_cons1. apply branch_work_spec. Qed.

Lemma branch_work_exact : forall st other out w d a b,
 work_apply (cons_holds F other) (branch_work F st w d a b) out=branch_cons1 F st other out w d a b.
Proof. reflexivity. Qed.

Lemma prepared_apply_exact : forall opposite jobs out,
 prepared_apply F opposite (check_range F opposite)(prepare_work F opposite jobs) out =
 work_apply (cons_holds F opposite) jobs out.
Proof.
 intros opposite jobs. induction jobs as [|[k query]jobs IH];intros out;[reflexivity|].
 cbn [prepared_apply prepare_work work_apply map fold_left fst snd].
 rewrite cons_prepare_exact. apply IH.
Qed.

Lemma shared_check_exact : forall opposite f,shared_check F opposite f=check_range F opposite f.
Proof.
 intros opposite f. unfold shared_check,check_range.
 assert(Hp:forall extra g, project_range_with F (tighten F (cons_index (acons opposite)) opposite
   (rf_with_cache F (cons_index (acons opposite)) true DEPTH opposite)) opposite extra g = project_range F opposite extra g).
 {intros. rewrite project_range_exact. apply project_range_extensional. apply tighten_extensional.
  exact (rf_memo_equivalent F _ DEPTH opposite true). }
 cbv zeta. rewrite Hp. f_equal. apply flat_map_ext. intros x. apply residue_eq_extensional. intros g. apply Hp.
Qed.

Lemma prepared_apply_ext : forall opposite jobs out f g,
 (forall x,f x=g x)->prepared_apply F opposite f jobs out=prepared_apply F opposite g jobs out.
Proof.
 intros opposite jobs. induction jobs as [|[k j]jobs IH];intros out f g H;[reflexivity|].
 cbn [prepared_apply fold_left fst snd].
 assert(Hj:prepared_run F opposite f j=prepared_run F opposite g j).
 {destruct j as [b|[[ts lo]hi]];cbn [prepared_run];[reflexivity|rewrite H;reflexivity]. }
 rewrite Hj. apply IH. exact H.
Qed.

Lemma run_prepared_exact : forall opposite jobs out,
 run_prepared F opposite jobs out=work_apply(cons_holds F opposite)jobs out.
Proof.
 intros. unfold run_prepared. destruct(prepared_pending(prepare_work F opposite jobs)).
 - rewrite (prepared_apply_ext opposite _ out _ (check_range F opposite));[apply prepared_apply_exact|apply shared_check_exact].
 - apply prepared_apply_exact.
Qed.

Lemma branch_cons1_prepared_exact : forall st other out w d a b,
 branch_cons1_prepared F st other out w d a b=branch_cons1 F st other out w d a b.
Proof. intros. unfold branch_cons1_prepared. rewrite run_prepared_exact. apply branch_work_exact. Qed.

Lemma work_apply_app : forall holds first second out,
 work_apply holds(first++second)out=work_apply holds second(work_apply holds first out).
Proof. intros. unfold work_apply. apply fold_left_app. Qed.

Lemma branch_cons_exact : forall st other out w d a b,
 branch_cons F st other out w d a b=branch_cons_spec F st other out w d a b.
Proof.
 intros. unfold branch_cons,branch_jobs,branch_cons_spec. cbv zeta.
 destruct ((match a,copy_operand F st a with OReg r,OReg s=>Pos.eqb r s|OImm x,OImm y=>Z.eqb x y|_,_=>false end) &&
 (match b,copy_operand F other b with OReg r,OReg s=>Pos.eqb r s|OImm x,OImm y=>Z.eqb x y|_,_=>false end)).
 - reflexivity.
 - rewrite work_apply_app. reflexivity.
Qed.

Lemma sel_cons_exact : forall st out w d a b,
 sel_cons F st out w d a b=sel_cons_spec F st out w d a b.
Proof.
 intros. unfold sel_cons,selection_jobs,sel_cons_spec,sel_cons1.
 destruct a,b;cbn [fst snd];try reflexivity.
 destruct(Pos.eqb r r0);[reflexivity|rewrite work_apply_app;reflexivity].
Qed.

Lemma branch_cons_prepared_exact : forall st other out w d a b,
 branch_cons_prepared F st other out w d a b=branch_cons F st other out w d a b.
Proof. intros. unfold branch_cons_prepared,branch_cons. apply run_prepared_exact. Qed.

Lemma sel_cons_prepared_exact : forall st out w d a b,
 sel_cons_prepared F st out w d a b=sel_cons F st out w d a b.
Proof. intros. unfold sel_cons_prepared,sel_cons. apply run_prepared_exact. Qed.

Lemma sel_refined_cons_prepared_exact : forall st st1 w d a b p,
 sel_refined_cons_prepared F st st1 w d a b p = sel_refined_cons F st st1 w d a b p.
Proof.
 intros. unfold sel_refined_cons_prepared,sel_refined_cons.
 destruct a as [r|za],b as [s|zb];try reflexivity;
 unfold sel_refined_cons_prepared_raw,sel_refined_cons_raw,sel_refined_cons_with;
 rewrite sel_cons_prepared_exact;
 repeat match goal with |- context[refine_selection ?f ?s ?p ?truth] => destruct (refine_selection f s p truth) end;
 try reflexivity;rewrite !branch_cons_prepared_exact;reflexivity.
Qed.
End PreparedEquivalence.

Lemma shared_check_cached_exact : forall F st f,shared_check_cached F st f=check_range F st f.
Proof.
 intros F st f. unfold shared_check_cached,check_range.
 assert(Hp:forall extra g,project_range_with F (cache_rf (tighten F (cons_index (acons st)) st
   (rf_with_cache F (cons_index (acons st)) true DEPTH st)) (map fst(PM.elements(amap st)))) st extra g=project_range F st extra g).
 { intros extra g. rewrite project_range_exact. apply project_range_extensional.
   intros r. destruct(cache_rf_same (tighten F (cons_index (acons st)) st (rf_with_cache F (cons_index (acons st)) true DEPTH st)) (map fst(PM.elements(amap st))) r) as [Hr Hs].
   rewrite Hr,Hs. exact(tighten_extensional F _ st _ _ (rf_memo_equivalent F _ DEPTH st true) r). }
 cbv zeta. rewrite Hp. f_equal. apply flat_map_ext. intros x. apply residue_eq_extensional. intros g. apply Hp.
Qed.

Lemma projected_cached_exact : forall F st k,
 projected_cached st (shared_check_cached F st) k=projected_constraint F st k.
Proof. intros F st [[ts lo]hi]. unfold projected_cached,projected_constraint.
 rewrite shared_check_cached_exact. reflexivity. Qed.

Lemma selected_step_cached_exact : forall F st out w d r, selected_step_cached F st out w d r=selected_step F st out w d r.
Proof.
 intros F st out w d r. unfold selected_step_cached,selected_step.
 destruct(negb (Z.eqb w 64 && Z.eqb (wd F d) 64 && Z.eqb (wd F r) 64) || Pos.eqb d r);[reflexivity|].
 set(delta:=materialized_form F st (normalize F st (fadd (as_form F st r EX) (fscale (as_form F st d EX) (-1) 64))) d r).
 destruct(negb (Z.eqb (fm delta) 64 && Z.eqb (flo delta) (fhi delta)) || mentions delta d);[reflexivity|].
 destruct (short_any (cons_mentions d) (acons st)) eqn:Hpending.
 2:{ cbv zeta. revert Hpending. generalize (acons st) as cs.
     intros cs Hnone. revert out. induction cs as [|[[ts lo]hi]cs IH];intros out;[reflexivity|].
     cbn [short_any] in Hnone.
     destruct(cons_mentions d (ts,lo,hi)) eqn:Hmention;[discriminate|].
     cbn [fold_left]. rewrite Hmention. cbn [negb]. apply IH. exact Hnone. }
 cbv zeta. generalize(acons st) as cs. intros cs. revert out.
 induction cs as [|[[ts lo]hi] cs IH];intros out;cbn [fold_left];[reflexivity|].
 rewrite IH. f_equal.
 destruct(negb(cons_mentions d (ts,lo,hi)));[reflexivity|].
 cbv beta iota zeta. rewrite shared_check_cached_exact.
 match goal with |- context[check_range F st ?f]=>destruct(check_range F st f) as [[l h]|] end;[|reflexivity].
 rewrite projected_cached_exact,branch_cons1_prepared_exact. reflexivity.
Qed.


Section Transfer.

Set Default Proof Using "All".

Variable F : lfacts.
Variable prog : program.
Variable E : env.
Hypothesis HE : env_ok F E.

Notation wdn := (wdn F).

(* The relation between d's old and new values that a step records: the new
   value is the old plus the value of step s in registers rs, which s reads
   from registers other than d. *)
Definition step_rel (d : reg) (step : option form) (rs : reg -> Z) (old new : Z) : Prop :=
  match step with
  | None => True
  | Some s => fm s = 64 /\ mentions s d = false /\
              exists e, flo s <= e <= fhi s /\ eqm (wd F d) new (old + (tsum wdn rs (ft s) + e)) /\
                        0 <= new < 2 ^ wd F d
  end.

(* ------------------------------------------------------------------ *)
(* Later steps.                                                        *)

Lemma fact_okv_later : forall t rs x v, fact_okv F t rs x v -> fact_okv F (S t) rs x v.
Proof. intros t rs x v H. destruct x; cbn [fact_okv] in *; exact H. Qed.

Theorem state_ok_later : forall t st rs, state_ok F t st rs -> state_ok F (S t) st rs.
Proof.
  intros t st rs [Hf Hc]. split; [| exact Hc].
  intros r x Hx. unfold fact_ok. apply fact_okv_later. exact (Hf r x Hx).
Qed.

Lemma upd_agree : forall rs d v, agree d rs (upd wdn rs d v).
Proof. intros rs d v r Hr. unfold upd. destruct (Pos.eqb_spec r d); [contradiction | reflexivity]. Qed.

Lemma upd_regs_ok : forall rs d v, regs_ok F E rs -> is_fixed F d = false -> regs_ok F E (upd wdn rs d v).
Proof.
  intros rs d v [Hw Hf] Hd. split.
  - intros r. unfold upd. destruct (Pos.eqb_spec r d) as [-> |]; [apply trunc_range | apply Hw].
  - intros r Hr. unfold upd. destruct (Pos.eqb_spec r d) as [-> |]; [rewrite Hd in Hr; discriminate | apply Hf; exact Hr].
Qed.

Section AtStep.

Variable t : nat.

(* ------------------------------------------------------------------ *)
(* Forgetting a register.                                              *)

Lemma forget_none : forall st f d step l, fold_left (forget_step F st f d step) l None = None.
Proof. intros st f d step l. induction l as [| x l IH]; cbn [fold_left]; [reflexivity | exact IH]. Qed.

Lemma forget_reg_sound : forall st rs rs' f d step g v,
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    wf f -> sat wdn rs f v -> forget_reg F st f d step = Some g -> wf g /\ sat wdn rs' g v.
Proof.
  intros st rs rs' f d step g v Hst Hrs Ha Hstep Hwf [e0 [He0 Hv]] H. unfold forget_reg in H.
  assert (Hwf' : 0 <= fm f <= 64) by exact Hwf.
  (* the fold keeps a form of modulus fm f that allows the partial sum *)
  assert (Hfold : forall l g0 V,
             wf g0 -> fm g0 = fm f -> sat wdn rs' g0 V ->
             fold_left (forget_step F st f d step) l (Some g0) = Some g -> wf g /\ fm g = fm f /\ sat wdn rs' g (V + tsum wdn rs l)).
  2: { destruct (Hfold (ft f) (range_form (fm f) (flo f) (fhi f)) e0) as [Hg [Hm Hs]].
       - unfold wf, range_form. cbn. lia.
       - reflexivity.
       - exists e0. cbn. split; [exact He0 | apply eqm_refl].
       - exact H.
       - split; [exact Hg |]. apply (sat_eqm _ _ _ _ _ Hs). rewrite Hm.
         rewrite Z.add_comm. apply eqm_sym. exact Hv. }
  induction l as [| [[r e] c] l IH]; intros g0 V Hw0 Hm0 Hs0 Hl; cbn [fold_left] in Hl;
    [| cbn [forget_step] in Hl].
  { injection Hl as <-. cbn [tsum]. rewrite Z.add_0_r. split; [exact Hw0 | split; assumption]. }
  (* the range of c times the old atom *)
  assert (Hrange : forall x y, atom_range F st (r, e) = Some (x, y) ->
                     Z.min (x * c) (y * c) <= c * atom_val wdn rs (r, e) <= Z.max (x * c) (y * c)).
  { intros x y Hxy. pose proof (atom_range_sound F E HE t st rs Hst Hrs (r, e) x y Hxy) as Hb.
    assert (x * c <= c * atom_val wdn rs (r, e) <= y * c \/ y * c <= c * atom_val wdn rs (r, e) <= x * c)
      by (destruct (Z.le_ge_cases 0 c); [left | right]; split; nia).
    lia. }
  assert (Hshift : forall x y, atom_range F st (r, e) = Some (x, y) ->
                     fold_left (forget_step F st f d step) l (Some (fshift g0 (Z.min (x * c) (y * c)) (Z.max (x * c) (y * c)))) = Some g ->
                     wf g /\ fm g = fm f /\ sat wdn rs' g (V + tsum wdn rs (((r, e), c) :: l))).
  { intros x y Hxy Hl'. cbn [tsum]. rewrite Z.add_assoc.
    apply (IH (fshift g0 (Z.min (x * c) (y * c)) (Z.max (x * c) (y * c))));
      [apply wf_fshift; exact Hw0 | unfold fshift; rewrite mk_fm; exact Hm0 | | exact Hl'].
    apply sat_fshift; [exact Hw0 | apply Hrange; exact Hxy | exact Hs0]. }
  assert (Hrel : forall s, step = Some s ->
                   (eqm (fm f) (atom_val wdn rs (d, e)) (rs d) /\
                    eqm (fm f) (atom_val wdn rs' (d, e)) (rs' d)) -> r = d ->
                   (forall es, flo s <= es <= fhi s ->
                      eqm (wd F d) (rs' d) (rs d + (tsum wdn rs (ft s) + es)) ->
                      eqm (fm f) (rs' d) (rs d + (tsum wdn rs (ft s) + es))) ->
                   fold_left (forget_step F st f d step) l
                     (Some (fadd g0 (fadd (fatom (fm f) (r, e) c) (fscale s (- c) (fm f))))) = Some g ->
                   wf g /\ fm g = fm f /\ sat wdn rs' g (V + tsum wdn rs (((r, e), c) :: l))).
  { intros s Es Hval Hr Hmod Hl'. subst r step. cbn [step_rel] in Hstep.
    destruct Hstep as [Hfs [Hms [es [Hes [Hsw _]]]]].
    pose proof (Hmod es Hes Hsw) as Hs64.
    (* the step's value, which s reads from registers d does not change *)
    set (SV := tsum wdn rs (ft s) + es) in *.
    assert (Hws : wf s) by (unfold wf; lia).
    assert (Hss : sat wdn rs' s SV).
    { exists es. split; [exact Hes |]. unfold SV. unfold mentions in Hms.
      rewrite (tsum_agree wdn d rs rs' (ft s) Ha Hms). apply eqm_refl. }
    assert (Hwa : wf (fatom (fm f) (d, e) c)) by (unfold fatom; apply wf_mk; lia).
    assert (Hwc : wf (fscale s (- c) (fm f))) by (apply wf_fscale; [exact Hws | lia]).
    assert (Hmc : fm (fscale s (- c) (fm f)) = fm f).
    { unfold fscale. rewrite mk_fm. destruct (Z.eqb (- c) 0); [reflexivity |].
      pose proof (v2_nonneg (- c)). lia. }
    assert (Hwi : wf (fadd (fatom (fm f) (d, e) c) (fscale s (- c) (fm f)))) by (apply wf_fadd; assumption).
    cbn [tsum]. rewrite Z.add_assoc.
    apply (IH _ _ (wf_fadd _ _ Hw0 Hwi)); [rewrite !fm_fadd, Hmc; unfold fatom; rewrite mk_fm, Hm0; lia | | exact Hl'].
    apply sat_fadd; [exact Hw0 | exact Hwi | exact Hs0 |].
    (* c times the new atom less c times the step is c times the old atom *)
    apply (sat_eqm wdn rs' _ (c * atom_val wdn rs' (d, e) + - c * SV)).
    - apply sat_fadd; [exact Hwa | exact Hwc | apply sat_fatom; lia | apply sat_fscale; [exact Hws | lia | exact Hss]].
    - rewrite fm_fadd, Hmc. unfold fatom. rewrite mk_fm, Z.min_id.
      replace (c * atom_val wdn rs' (d, e) + - c * SV)
        with (c * (atom_val wdn rs' (d, e) + - SV)) by ring.
      apply eqm_mul; [lia |].
      replace (atom_val wdn rs (d, e)) with ((atom_val wdn rs (d, e) + SV) + - SV) by ring.
      apply eqm_add; [lia | | apply eqm_refl].
      eapply eqm_trans; [exact (proj2 Hval) |].
      eapply eqm_trans; [exact Hs64 |].
      apply eqm_add; [lia | apply eqm_sym; exact (proj1 Hval) | apply eqm_refl]. }
  (* the step's congruence modulo 2^(wd d) holds modulo 2^(fm f) for a form
     no wider than d, and is an equality when the step does not wrap *)
  assert (Hmodc : forall s, step = Some s -> ((fm f <=? wd F d) || step_exact F st d s) = true ->
            forall es, flo s <= es <= fhi s -> eqm (wd F d) (rs' d) (rs d + (tsum wdn rs (ft s) + es)) ->
            eqm (fm f) (rs' d) (rs d + (tsum wdn rs (ft s) + es))).
  { intros s Es Hc es Hes Hm. apply orb_true_iff in Hc. destruct Hc as [Hc | Hc].
    - apply Z.leb_le in Hc. apply (eqm_weaken (wd F d)); [unfold wf in Hwf; lia | exact Hm].
    - unfold step_exact in Hc.
      destruct (reg_range F st d) as [[x y] |] eqn:Er; [| discriminate].
      destruct (form_range F st s) as [[p q] |] eqn:Ep; [| discriminate].
      apply andb_true_iff in Hc. destruct Hc as [H1 H2]. apply Z.leb_le in H1. apply Z.ltb_lt in H2.
      pose proof (reg_range_sound F E HE t st rs Hst Hrs d x y Er) as Hd.
      pose proof (form_range_sound F E HE t st rs Hst Hrs s p q es Ep Hes) as Hsv.
      subst step. cbn [step_rel] in Hstep. destruct Hstep as [_ [_ [es' [_ [_ Hb]]]]].
      pose proof (wd_range F E HE d) as Hwr.
      assert (Heq : rs' d = rs d + (tsum wdn rs (ft s) + es))
        by (apply (eqm_small_eq (wd F d) (wd F d)); [unfold wd in *; lia | lia | lia | exact Hm]).
      rewrite Heq. apply eqm_refl. }
  destruct (Pos.eqb_spec r d) as [Hrd | Hrd]; cbn [negb] in Hl.
  - destruct step as [s |]; [destruct e | ].
    + subst r. destruct ((fm f <=? wd F d) || step_exact F st d s) eqn:Ec.
      * exact (Hrel s eq_refl (conj (eqm_refl _ _) (eqm_refl _ _)) eq_refl (Hmodc s eq_refl Ec) Hl).
      * destruct (atom_range F st (d, EX)) as [[x0 y0] |] eqn:Exy; [| rewrite forget_none in Hl; discriminate].
        exact (Hshift x0 y0 eq_refl Hl).
    + subst r. destruct ((fm f <=? wd F d) || step_exact F st d s) eqn:Ec.
      * exact (Hrel s eq_refl (conj (eqm_refl _ _) (eqm_refl _ _)) eq_refl (Hmodc s eq_refl Ec) Hl).
      * destruct (atom_range F st (d, EZ)) as [[x0 y0] |] eqn:Exy; [| rewrite forget_none in Hl; discriminate].
        exact (Hshift x0 y0 eq_refl Hl).
    + (* the sign extension is the value, before and after the step *)
      subst r. cbv zeta in Hl.
      destruct (fm f <=? wd F d) eqn:Emod.
      { apply Z.leb_le in Emod.
        apply (Hrel s eq_refl); [| reflexivity | | exact Hl].
        - split; apply (eqm_weaken (wd F d)); [lia | apply sext_eqm | lia | apply sext_eqm].
        - intros es Hes Hq. apply (eqm_weaken (wd F d)); [lia | exact Hq]. }
      destruct (reg_range F st d) as [[x y] |] eqn:Er;
        [destruct (form_range F st s) as [[p q] |] eqn:Ep |].
      2, 3: destruct (atom_range F st (d, ES)) as [[x' y'] |] eqn:Exy; [| rewrite forget_none in Hl; discriminate];
            exact (Hshift x' y' eq_refl Hl).
      destruct ((wd F d <? 64) && (0 <=? x) && (y <? 2 ^ (wd F d - 1)) && (0 <=? x + p) &&
                (y + q <? 2 ^ (wd F d - 1))) eqn:Ec.
      2: { destruct (atom_range F st (d, ES)) as [[x' y'] |] eqn:Exy; [| rewrite forget_none in Hl; discriminate].
           exact (Hshift x' y' eq_refl Hl). }
      rewrite !andb_true_iff in Ec. destruct Ec as [[[[Hw Hx] Hy] Hxp] Hyq].
      apply Z.ltb_lt in Hw, Hy, Hyq. apply Z.leb_le in Hx, Hxp.
      pose proof (reg_range_sound F E HE t st rs Hst Hrs d x y Er) as Hd.
      pose proof (wd_range F E HE d) as Hwr.
      assert (Hp1 : 2 ^ (wd F d - 1) <= 2 ^ wd F d) by (apply Z.pow_le_mono_r; unfold wd in *; lia).
      (* both sides lie below the sign bit, so the step does not wrap *)
      assert (Hnew : forall es, flo s <= es <= fhi s -> eqm (wd F d) (rs' d) (rs d + (tsum wdn rs (ft s) + es)) ->
                       rs' d = rs d + (tsum wdn rs (ft s) + es)).
      { intros es Hes Hm. pose proof (form_range_sound F E HE t st rs Hst Hrs s p q es Ep Hes) as Hsv.
        cbn [step_rel] in Hstep. destruct Hstep as [_ [_ [_ [_ [_ Hb]]]]].
        apply (eqm_small_eq (wd F d) (wd F d)); [unfold wd in *; lia | lia | lia | exact Hm]. }
      cbn [step_rel] in Hstep. destruct Hstep as [Hfs [Hms [es [Hes [Hsw Hbnd]]]]].
      pose proof (Hnew es Hes Hsw) as Hnew1.
      pose proof (form_range_sound F E HE t st rs Hst Hrs s p q es Ep Hes) as Hsv.
      apply (Hrel s eq_refl); [| reflexivity | | exact Hl].
      * cbn [atom_val]. unfold wd in Hy, Hyq, Hw.
        split; rewrite sext_small by lia; apply eqm_refl.
      * intros es0 Hes0 Hmw. rewrite (Hnew es0 Hes0 Hmw). apply eqm_refl.
    + destruct (atom_range F st (r, EShr k)) as [[x y] |] eqn:Exy; [| rewrite forget_none in Hl; discriminate].
      exact (Hshift x y eq_refl Hl).
    + destruct (atom_range F st (r, EBf a n)) as [[x y] |] eqn:Exy; [| rewrite forget_none in Hl; discriminate].
      exact (Hshift x y eq_refl Hl).
    + destruct (atom_range F st (r, e)) as [[x y] |] eqn:Exy; [| destruct e; rewrite forget_none in Hl; discriminate].
      destruct e; exact (Hshift x y eq_refl Hl).
  - assert (Hwa : wf (fatom (fm f) (r, e) c)) by (unfold fatom; apply wf_mk; lia).
    cbn [tsum]. rewrite Z.add_assoc.
    apply (IH _ _ (wf_fadd _ _ Hw0 Hwa)); [rewrite fm_fadd; unfold fatom; rewrite mk_fm, Hm0; lia | | exact Hl].
    apply sat_fadd; [exact Hw0 | exact Hwa | exact Hs0 |].
    replace (c * atom_val wdn rs (r, e)) with (c * atom_val wdn rs' (r, e))
      by (unfold atom_val; rewrite (Ha r Hrd); reflexivity).
    apply sat_fatom. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* Comparisons and constraints through a constant step.                *)

(* A constant step that keeps d's reading moves the value and the reading
   by c. *)
Lemma step_keeps_value : forall st rs rs' d step c sg,
    state_ok F t st rs -> regs_ok F E rs -> step_rel d step rs (rs d) (rs' d) ->
    const_step step = Some c -> step_keeps F st d c sg = true ->
    extend (wdn d) sg (rs' d) = extend (wdn d) sg (rs d) + c.
Proof.
  intros st rs rs' d step c sg Hst Hrs Hstep Hc Hk.
  destruct step as [s |]; [| discriminate]. unfold const_step in Hc.
  destruct (null (ft s) && Z.eqb (flo s) (fhi s)) eqn:Es; [| discriminate]. apply some_eq_t in Hc. subst c.
  apply andb_true_iff in Es. destruct Es as [Hn Hx]. apply Z.eqb_eq in Hx.
  cbn [step_rel] in Hstep. destruct Hstep as [_ [_ [e [He [Hm Hb]]]]].
  destruct (ft s) eqn:Ets; [| discriminate]. cbn [tsum] in Hm. assert (e = flo s) by lia. subst e.
  rewrite Z.add_0_l in Hm.
  pose proof (wd_range F E HE d) as Hw. pose proof (rs_range F E rs Hrs d) as Hr.
  assert (Hw' : 1 <= wd F d <= 64) by (unfold wd; lia).
  (* the new value is the old plus the step, modulo 2^(wd d) *)
  assert (Hmc : eqm (wd F d) (rs' d) (rs d + rep 64 (flo s))).
  { apply (eqm_trans _ _ _ _ Hm). apply eqm_add; [lia | apply eqm_refl |].
    apply (eqm_weaken 64); [lia |]. apply eqm_sym. apply rep_eqm. lia. }
  unfold step_keeps in Hk. destruct sg; cbn [extend].
  - destruct (sreg_range F st d) as [[x y] |] eqn:Er; [| discriminate].
    pose proof (sreg_range_sound F E HE t st rs Hst Hrs d x y Er) as Hd.
    apply andb_true_iff in Hk. destruct Hk as [H1 H2]. apply Z.leb_le in H1. apply Z.ltb_lt in H2.
    symmetry. apply (eqm_extend_eq (wdn d) true (rs' d)); [lia | unfold lo_ok, hi_ok, wd in *; lia |].
    apply (eqm_trans _ _ (rs d + rep 64 (flo s))).
    + apply eqm_add; [lia | | apply eqm_refl]. pose proof (extend_eqm (wdn d) true (rs d)) as Hx1. exact Hx1.
    + apply (eqm_trans _ _ (rs' d)); [apply eqm_sym; unfold wd in Hmc; exact Hmc |].
      apply eqm_sym. apply extend_eqm.
  - destruct (reg_range F st d) as [[x y] |] eqn:Er; [| discriminate].
    pose proof (reg_range_sound F E HE t st rs Hst Hrs d x y Er) as Hd.
    apply andb_true_iff in Hk. destruct Hk as [H1 H2]. apply Z.leb_le in H1. apply Z.ltb_lt in H2.
    assert (Hnew : rs' d = rs d + rep 64 (flo s))
      by (apply (eqm_small_eq (wd F d) (wd F d)); [lia | unfold wd in *; lia | lia | exact Hmc]).
    unfold wd in *. rewrite Hnew, !trunc_small by lia. reflexivity.
Qed.

(* The unsigned reading of a register is its value. *)
Lemma step_keeps_unsigned : forall st rs rs' d step c,
    state_ok F t st rs -> regs_ok F E rs -> step_rel d step rs (rs d) (rs' d) ->
    const_step step = Some c -> step_keeps F st d c false = true -> rs' d = rs d + c.
Proof.
  intros st rs rs' d step c Hst Hrs Hstep Hc Hk.
  pose proof (step_keeps_value st rs rs' d step c false Hst Hrs Hstep Hc Hk) as H. cbn [extend] in H.
  pose proof (rs_range F E rs Hrs d) as Hr.
  assert (Hb : 0 <= rs' d < 2 ^ Z.of_nat (wdn d)).
  { destruct step as [s |]; [| discriminate]. cbn [step_rel] in Hstep.
    destruct Hstep as [_ [_ [_ [_ [_ Hb]]]]]. unfold wd in Hb. exact Hb. }
  rewrite !trunc_small in H by lia. exact H.
Qed.

Lemma cmp_eval_shift : forall cc a b c, cmp_eval cc (a + c) (b + c) = cmp_eval cc a b.
Proof.
  intros cc a b c. destruct cc; cbn [cmp_eval]; apply Bool.eq_iff_eq_true;
    rewrite ?Z.ltb_lt, ?Z.leb_le, ?Z.eqb_eq, ?negb_true_iff, ?Z.eqb_neq; lia.
Qed.

(* A comparison that follows a constant step holds of the new registers
   exactly when it held of the old ones. *)
Lemma cmp_step_direct_sound : forall st rs rs' d step k k',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    cmp_step_direct F st d step k = Some k' -> (cmp_holds rs' k' <-> cmp_holds rs k).
Proof.
  intros st rs rs' d step k k' Hst Hrs Ha Hstep H. unfold cmp_step_direct in H.
  destruct (cmp_mentions k d) eqn:Em; cbn [negb] in H.
  2: { apply some_eq_t in H. subst k'. split; [| exact (cmp_agree d rs rs' k Ha Em)].
       intros Hk. unfold cmp_mentions in Em. apply orb_false_iff in Em. destruct Em as [Ha' Hb'].
       unfold cmp_holds in *. rewrite <- (Ha (ca k)) by (intros E'; rewrite E', Pos.eqb_refl in Ha'; discriminate).
       destruct (cb k) as [r | z]; cbn [opv] in *; [| exact Hk].
       rewrite <- (Ha r) by (intros E'; rewrite E', Pos.eqb_refl in Hb'; discriminate). exact Hk. }
  destruct (const_step step) as [c |] eqn:Ec; [| discriminate].
  destruct (cb k) as [r | z] eqn:Eb.
  - destruct (Pos.eqb (ca k) d && negb (Pos.eqb r d) &&
              Z.eqb (Z.of_nat (cw k)) (wd F d) && step_keeps F st d c (csg k)) eqn:Ek;
      [| discriminate].
    rewrite !andb_true_iff in Ek. destruct Ek as [[[Hca Hrd] Hcw] Hkeep].
    apply Pos.eqb_eq in Hca. apply negb_true_iff, Pos.eqb_neq in Hrd. apply Z.eqb_eq in Hcw.
    assert (Hn : cw k = wdn d) by (unfold wd in Hcw; lia).
    pose proof (step_keeps_value st rs rs' d step c (csg k) Hst Hrs Hstep Ec Hkeep) as Hext.
    destruct (step_cmp c (cc k)) as [cc' |] eqn:Ecc; [| discriminate].
    injection H as <-. unfold cmp_holds. cbn [cc cw csg ca cb opv].
    rewrite Hca, Eb. cbn [opv]. rewrite (Ha r Hrd), Hn, Hext.
    unfold step_cmp in Ecc. destruct (cc k);
      repeat match goal with H : context [if ?b then _ else _] |- _ => destruct b eqn:? end;
      try discriminate; injection Ecc as <-;
      match goal with H : Z.eqb _ _ = true |- _ => apply Z.eqb_eq in H; subst c end;
      cbn [cmp_eval]; rewrite ?Z.ltb_lt, ?Z.leb_le; lia.
  -
  destruct (Pos.eqb (ca k) d && Z.eqb (Z.of_nat (cw k)) (wd F d) && step_keeps F st d c (csg k) &&
            (lo_ok (wd F d) (csg k) <=? extend (cw k) (csg k) z + c) &&
            (extend (cw k) (csg k) z + c <=? hi_ok (wd F d) (csg k))) eqn:Ek; [| discriminate].
  apply some_eq_t in H. subst k'.
  rewrite !andb_true_iff in Ek. destruct Ek as [[[[Hca Hcw] Hkeep] Hlo] Hhi].
  apply Pos.eqb_eq in Hca. apply Z.eqb_eq in Hcw. apply Z.leb_le in Hlo, Hhi.
  pose proof (wd_range F E HE d) as Hw.
  assert (Hn : cw k = wdn d) by (unfold wd in Hcw; lia).
  pose proof (step_keeps_value st rs rs' d step c (csg k) Hst Hrs Hstep Ec Hkeep) as Hext.
  (* the new constant is its own reading *)
  assert (Hz : extend (cw k) (csg k) (extend (cw k) (csg k) z + c) = extend (cw k) (csg k) z + c).
  { rewrite Hn in *. unfold wd in Hlo, Hhi. symmetry. apply eqm_extend_eq; [lia | lia |].
    apply eqm_sym. apply extend_eqm. }
  unfold cmp_holds. cbn [cc cw csg ca cb opv]. rewrite Hz, Hca, Eb. cbn [opv]. rewrite Hn in *. rewrite Hext.
  rewrite cmp_eval_shift. reflexivity.
Qed.

Lemma same_exact_value_sound : forall st rs a b,
 state_ok F t st rs -> regs_ok F E rs -> same_exact_value F st a b = true -> rs a = rs b.
Proof.
 intros st rs a b Hst Hrs H. unfold same_exact_value in H.
 destruct (get st a) as [[f al vr|al|cs ex|cs ex]|] eqn:Ea; try discriminate.
 destruct (get st b) as [[g bl wr|bl|ds ey|ds ey]|] eqn:Eb; try discriminate.
 rewrite !andb_true_iff in H. destruct H as [[[[[[Hw Hm] Hfg] Hf] Hg] Hlo] Hts].
 apply Z.eqb_eq in Hw,Hfg,Hf,Hg,Hlo. apply Z.leb_le in Hm. apply terms_eqb_eq in Hts.
 pose proof (proj1 Hst a _ Ea) as [_ [[e [He Hsa]] _]].
 pose proof (proj1 Hst b _ Eb) as [_ [[e' [He' Hsb]] _]].
 assert (e = flo f) by lia. assert (e' = flo g) by lia. subst e e'.
 rewrite <- Hfg, <- Hts, <- Hlo in Hsb.
 pose proof (rs_range F E rs Hrs a) as Ha.
 pose proof (rs_range F E rs Hrs b) as Hb.
 pose proof (wd_range F E HE a) as Hwa.
 assert (Hmab : eqm (fm f) (rs a) (rs b)).
 { eapply eqm_trans; [exact Hsa|apply eqm_sym;exact Hsb]. }
 apply (eqm_small_eq (fm f) (wd F a)); unfold wd in *.
 - lia.
 - exact Ha.
 - rewrite Hw. exact Hb.
 - exact Hmab.
Qed.

Lemma comparison_alias_sound : forall st rs a d,
 state_ok F t st rs -> regs_ok F E rs -> comparison_alias F st a d = true -> rs d = rs a.
Proof.
 intros st rs a d Hst Hrs H. unfold comparison_alias in H.
 destruct (copy_of F a (get st a)) as [r|] eqn:Er.
 - destruct (Pos.eqb r d) eqn:Ed.
   + apply Pos.eqb_eq in Ed. subst r. exact (copy_of_sound F E HE t st rs Hst Hrs a d Er).
   + symmetry. exact (same_exact_value_sound st rs a d Hst Hrs H).
 - symmetry. exact (same_exact_value_sound st rs a d Hst Hrs H).
Qed.

Lemma cmp_step_sound : forall st rs rs' d step k k',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    cmp_step F st d step k = Some k' -> (cmp_holds rs' k' <-> cmp_holds rs k).
Proof.
  intros st rs rs' d step k k' Hst Hrs Ha Hstep H. unfold cmp_step in H.
  destruct (const_step step) as [c|] eqn:Ec;
    [| exact (cmp_step_direct_sound st rs rs' d step k k' Hst Hrs Ha Hstep H)].
  destruct (comparison_alias F st (ca k) d) eqn:Er;
    [| exact (cmp_step_direct_sound st rs rs' d step k k' Hst Hrs Ha Hstep H)].
  destruct (cmp_step_direct F st d step (cmp_rebase d k)) as [kb|] eqn:Eb;
    [| exact (cmp_step_direct_sound st rs rs' d step k k' Hst Hrs Ha Hstep H)].
  injection H as <-.
  pose proof (cmp_step_direct_sound st rs rs' d step (cmp_rebase d k) kb Hst Hrs Ha Hstep Eb) as He.
  pose proof (comparison_alias_sound st rs (ca k) d Hst Hrs Er) as Hv.
  rewrite He. unfold cmp_holds, cmp_rebase. cbn [cc cw csg ca cb]. rewrite Hv. reflexivity.
Qed.

Lemma tsum_step : forall rs rs' d c dts,
    (forall t0, In t0 dts -> fst (fst t0) = d /\
                  atom_val wdn rs' (fst t0) = atom_val wdn rs (fst t0) + c) ->
    tsum wdn rs' dts = tsum wdn rs dts + fold_left (fun acc (t0 : atom * Z) => acc + snd t0) dts 0 * c.
Proof.
  intros rs rs' d c dts H.
  assert (Hgen : forall l acc, (forall t0, In t0 l -> atom_val wdn rs' (fst t0) = atom_val wdn rs (fst t0) + c) ->
            tsum wdn rs' l + acc * c = tsum wdn rs l + fold_left (fun acc (t0 : atom * Z) => acc + snd t0) l acc * c).
  { induction l as [| [x q] l IH]; intros acc Hl; cbn [tsum fold_left fst snd]; [lia |].
    pose proof (IH (acc + q) (fun t0 Ht0 => Hl t0 (or_intror Ht0))) as Hi.
    pose proof (Hl (x, q) (or_introl eq_refl)) as Hx. cbn [fst] in Hx. rewrite Hx. nia. }
  pose proof (Hgen dts 0 (fun t0 Ht0 => proj2 (H t0 Ht0))) as Hg.
  rewrite Z.mul_0_l, Z.add_0_r in Hg. exact Hg.
Qed.

(* A constraint that follows a constant step holds of the new registers. *)
Lemma cons_step_sound : forall st rs rs' d step k k',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    cons_ok F rs k -> cons_step F st d step k = Some k' -> cons_ok F rs' k'.
Proof.
  intros st rs rs' d step [[ts lo] hi] k' Hst Hrs Ha Hstep Hk H. unfold cons_step in H.
  pose proof (tsum_filter wdn rs (on_reg d) ts) as Hs.
  pose proof (tsum_filter wdn rs' (on_reg d) ts) as Hs'.
  (* the terms without d agree *)
  assert (Hrest : tsum wdn rs' (filter (fun t0 => negb (on_reg d t0)) ts) =
                  tsum wdn rs (filter (fun t0 => negb (on_reg d t0)) ts)).
  { clear -Ha. induction ts as [| [[r e] q] ts IH]; [reflexivity |].
    cbn [filter]. change (on_reg d ((r, e), q)) with (Pos.eqb r d).
    destruct (Pos.eqb_spec r d) as [-> | Hne]; cbn [negb]; [exact IH |].
    cbn [tsum]. rewrite IH. destruct e; cbn [atom_val]; rewrite (Ha r Hne); reflexivity. }
  cbn [cons_ok] in Hk.
  destruct (filter (on_reg d) ts) as [| t1 more] eqn:Edts.
  - apply some_eq_t in H. subst k'. cbn [cons_ok tsum] in *. lia.
  - destruct (const_step step) as [c |] eqn:Ec; [| discriminate].
    match type of H with context [forallb ?p (t1 :: more)] => destruct (forallb p (t1 :: more)) eqn:Eall end;
      [| discriminate]. apply some_eq_t in H. subst k'.
    rewrite forallb_forall in Eall. cbn [cons_ok].
    assert (Hd : forall t0, In t0 (t1 :: more) -> fst (fst t0) = d /\
                   atom_val wdn rs' (fst t0) = atom_val wdn rs (fst t0) + c).
    { intros [[r e] q] Hin. pose proof (Eall _ Hin) as Hok.
      assert (Hr : r = d).
      { rewrite <- Edts in Hin. apply filter_In in Hin. destruct Hin as [_ Hin]. unfold on_reg in Hin.
        apply Pos.eqb_eq. exact Hin. }
      subst r. split; [reflexivity |]. cbn [fst snd] in *.
      destruct e; try discriminate; cbn [atom_val].
      - exact (step_keeps_unsigned st rs rs' d step c Hst Hrs Hstep Ec Hok).
      - exact (step_keeps_unsigned st rs rs' d step c Hst Hrs Hstep Ec Hok).
      - exact (step_keeps_value st rs rs' d step c true Hst Hrs Hstep Ec Hok). }
    pose proof (tsum_step rs rs' d c (t1 :: more) Hd) as Hstp. lia.
Qed.

(* Comparisons cs that hold when P does, and, when exact, only then, moved
   through the step: those kept hold after it when P does, and, when none
   is dropped, only then. *)
Lemma forget_cmps_sound : forall st rs rs' d step cs ex (P : Prop),
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    (P -> Forall (cmp_holds rs) cs) -> (ex = true -> Forall (cmp_holds rs) cs -> P) ->
    let cs' := flat_map (fun k => match cmp_step F st d step k with Some k' => [k'] | None => [] end) cs in
    (P -> Forall (cmp_holds rs') cs') /\
    (ex && Nat.eqb (length cs') (length cs) = true -> Forall (cmp_holds rs') cs' -> P).
Proof.
  intros st rs rs' d step cs ex P Hst Hrs Ha Hstep Hx1 Hx2 cs'.
  set (sh := fun k => match cmp_step F st d step k with Some k' => [k'] | None => [] end) in *.
  (* each comparison kept is one of cs, moved through the step *)
  assert (Hkept : forall k', In k' (flat_map sh cs) -> exists k, In k cs /\ cmp_step F st d step k = Some k').
  { intros k' Hk'. apply in_flat_map in Hk'. destruct Hk' as [k [Hk Hk']]. unfold sh in Hk'.
    destruct (cmp_step F st d step k) as [k1 |] eqn:E1; [| destruct Hk'].
    destruct Hk' as [<- | []]. exists k. split; assumption. }
  split.
  - intros Hv. apply Forall_forall. intros k' Hk'. destruct (Hkept k' Hk') as [k [Hk Hs]].
    apply (proj2 (cmp_step_sound st rs rs' d step k k' Hst Hrs Ha Hstep Hs)).
    exact (proj1 (Forall_forall _ _) (Hx1 Hv) k Hk).
  - intros He Hall. apply andb_true_iff in He. destruct He as [He Hl]. apply Nat.eqb_eq in Hl.
    apply (Hx2 He). apply Forall_forall. intros k Hk.
    (* no comparison was dropped, so each of cs has its moved form *)
    assert (Hall_some : forall k, In k cs -> exists k', cmp_step F st d step k = Some k').
    { assert (Hlen : forall l, length (flat_map sh l) = length l -> forall k, In k l -> exists k', cmp_step F st d step k = Some k').
      { induction l as [| k1 l IH]; intros Hl1 k2 Hk2; [destruct Hk2 |].
        cbn [flat_map length] in Hl1. rewrite app_length in Hl1. unfold sh at 1 in Hl1.
        assert (Hle : (length (flat_map sh l) <= length l)%nat).
        { clear. induction l as [| k l IH]; cbn [flat_map length]; [lia |].
          rewrite app_length. unfold sh at 1. destruct (cmp_step F st d step k); cbn [length]; lia. }
        destruct (cmp_step F st d step k1) as [k1' |] eqn:E1; cbn [length] in Hl1; [| lia].
        destruct Hk2 as [<- | Hk2]; [exists k1'; exact E1 | apply IH; [lia | exact Hk2]]. }
      exact (Hlen cs Hl). }
    destruct (Hall_some k Hk) as [k' Hs].
    apply (proj1 (cmp_step_sound st rs rs' d step k k' Hst Hrs Ha Hstep Hs)).
    apply (proj1 (Forall_forall _ _) Hall). apply in_flat_map. exists k. split; [exact Hk |].
    unfold sh. rewrite Hs. left. reflexivity.
Qed.

Lemma forget_fact_sound : forall st rs rs' d step x y v,
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    fact_okv F t rs x v -> forget_fact F st d step x = Some y -> fact_okv F t rs' y v.
Proof.
  intros st rs rs' d step x y v Hst Hrs Ha Hstep Hx H.
  destruct x as [f al vr | k | cs ex | cs ex];
    cbn [forget_fact] in H.
  - destruct Hx as [Hwf [Hs [Hal Hvr]]]. destruct (mentions f d) eqn:Em.
    + destruct (forget_reg F st f d step) as [g |] eqn:Eg; [| discriminate]. apply some_eq_t in H. subst y.
      destruct (forget_reg_sound st rs rs' f d step g v Hst Hrs Ha Hstep Hwf Hs Eg) as [Hwg Hsg].
      cbn [fact_okv]. split; [exact Hwg |]. split; [exact Hsg |]. split; [| exact Hvr].
      exact (value_align_sound F E HE t rs Hrs st f al v Hst Hwf Hs Hal).
    + apply some_eq_t in H. subst y. cbn [fact_okv]. split; [exact Hwf |]. split; [| split; assumption].
      exact (sat_agree wdn d rs rs' f v Ha Em Hs).
  - apply some_eq_t in H. subst y. exact Hx.
  - destruct Hx as [Hx1 Hx2]. pose proof (forget_cmps_sound st rs rs' d step cs ex _ Hst Hrs Ha Hstep Hx1 Hx2) as Hc.
    cbv zeta in Hc. destruct (flat_map _ cs) as [| k0 cs0]; [discriminate |].
    apply some_eq_t in H. subst y. exact Hc.
  - destruct Hx as [Hx1 Hx2]. pose proof (forget_cmps_sound st rs rs' d step cs ex _ Hst Hrs Ha Hstep Hx1 Hx2) as Hc.
    cbv zeta in Hc. destruct (flat_map _ cs) as [| k0 cs0]; [discriminate |].
    apply some_eq_t in H. subst y. exact Hc.
Qed.

(* The facts the fold of assign keeps: each is a forgotten fact of st. *)
Lemma assign_fold_map : forall st d step r y,
    PM.find r (PM.fold (fun r x acc =>
                          if Pos.eqb r d then acc else
                          match forget_fact F st d step x with Some y => PM.add r y acc | None => acc end)
                       (amap st) (PM.empty fact)) = Some y ->
    r <> d /\ exists x, get st r = Some x /\ forget_fact F st d step x = Some y.
Proof.
  intros st d step r y H. rewrite PM.fold_1 in H.
  assert (Hgen : forall l acc,
             (forall p, In p l -> In p (PM.elements (amap st))) ->
             (forall r' y', PM.find r' acc = Some y' ->
                r' <> d /\ exists x, get st r' = Some x /\ forget_fact F st d step x = Some y') ->
             forall r' y', PM.find r' (fold_left (fun a p =>
                               if Pos.eqb (fst p) d then a else
                               match forget_fact F st d step (snd p) with Some y => PM.add (fst p) y a | None => a end)
                             l acc) = Some y' ->
             r' <> d /\ exists x, get st r' = Some x /\ forget_fact F st d step x = Some y').
  { induction l as [| [r0 x0] l IH]; intros acc Hl Hacc r' y' Hf; cbn [fold_left] in Hf; [exact (Hacc r' y' Hf) |].
    eapply IH; [exact (fun p Hp => Hl p (or_intror Hp)) | | exact Hf].
    intros r1 y1 H1. cbn [fst snd] in H1.
    destruct (Pos.eqb_spec r0 d) as [Hr0 | Hr0]; [exact (Hacc r1 y1 H1) |].
    destruct (forget_fact F st d step x0) as [y0 |] eqn:Ey0; [| exact (Hacc r1 y1 H1)].
    destruct (Pos.eq_dec r1 r0) as [-> | Hne].
    - rewrite PM.gss in H1. apply some_eq_t in H1. subst y1. split; [exact Hr0 |]. exists x0.
      split; [| exact Ey0]. apply PM.elements_complete. apply Hl. left. reflexivity.
    - rewrite PM.gso in H1 by exact Hne. exact (Hacc r1 y1 H1). }
  apply (Hgen _ (PM.empty fact) (fun p Hp => Hp)); [| exact H].
  intros r' y' Hf. rewrite PM.gempty in Hf. discriminate.
Qed.

Lemma fact_node_find : forall l value rt r,
  PM.find r (fact_node l value rt) = PM.find r (PM.Node l value rt).
Proof.
  intros l value rt r. unfold fact_node.
  destruct l, value, rt, r; cbn [PM.find]; try rewrite PM.gleaf; reflexivity.
Qed.

Lemma filter_facts_find : forall f m r,
  PM.find r (filter_facts f m) =
  match PM.find r m with Some x => f x | None => None end.
Proof.
  intros f m. induction m as [|l IHl value rt IHr]; intros r.
  - destruct r; reflexivity.
  - cbn [filter_facts]. rewrite fact_node_find.
    destruct r; cbn [PM.find]; auto.
Qed.

Lemma assigned_facts_find : forall st d step r,
  PM.find r (assigned_facts F st d step) =
  if Pos.eqb r d then None
  else match get st r with Some x => forget_fact F st d step x | None => None end.
Proof.
  intros st d step r. unfold assigned_facts. rewrite filter_facts_find.
  destruct (Pos.eqb_spec r d) as [-> | Hne].
  - rewrite PM.grs. reflexivity.
  - rewrite PM.gro by exact Hne. reflexivity.
Qed.

Lemma assign_map : forall st d step r y,
    PM.find r (assigned_facts F st d step) = Some y ->
    r <> d /\ exists x, get st r = Some x /\ forget_fact F st d step x = Some y.
Proof.
  intros st d step r y H. rewrite assigned_facts_find in H.
  destruct (Pos.eqb_spec r d) as [Heq | Hne]; [discriminate |].
  destruct (get st r) as [x |] eqn:Hx; [|discriminate].
  split; [exact Hne |]. exists x. auto.
Qed.

(* Every surviving lookup equals the previous insertion-based evaluator;
   this changes map construction, not facts or inference precision. *)
Lemma assigned_facts_exact : forall st d step r,
  PM.find r (assigned_facts F st d step) =
  PM.find r (PM.fold (fun r x acc =>
    if Pos.eqb r d then acc else
    match forget_fact F st d step x with Some y => PM.add r y acc | None => acc end)
    (amap st) (PM.empty fact)).
Proof.
  intros st d step r.
  set (old := PM.fold (fun r x acc =>
    if Pos.eqb r d then acc else
    match forget_fact F st d step x with Some y => PM.add r y acc | None => acc end)
    (amap st) (PM.empty fact)).
  destruct (PM.find r old) as [y |] eqn:Hold.
  - pose proof (assign_fold_map st d step r y Hold) as [Hne [x [Hx Hforget]]].
    rewrite assigned_facts_find. destruct (Pos.eqb_spec r d); [contradiction |].
    rewrite Hx, Hforget. reflexivity.
  - destruct (PM.find r (assigned_facts F st d step)) as [y |] eqn:Hnew; [|reflexivity].
    pose proof (assign_map st d step r y Hnew) as [Hne [x [Hx Hforget]]].
    assert (Helem : In (r,x) (PM.elements (amap st))) by (apply PM.elements_correct; exact Hx).
    set (one := fun (acc : PM.t fact) (pair : reg * fact) =>
      if Pos.eqb (fst pair) d then acc else
      match forget_fact F st d step (snd pair) with Some y => PM.add (fst pair) y acc | None => acc end).
    assert (Hkeep : forall l acc,
      (forall pair, In pair l -> In pair (PM.elements (amap st))) ->
      PM.find r acc = Some y -> PM.find r (fold_left one l acc) = Some y).
    { induction l as [|[r0 x0] rest IH]; intros acc Hsub Hacc; cbn [fold_left]; [exact Hacc|].
      apply IH; [intros pair Hin; apply Hsub; right; exact Hin|].
      unfold one. cbn [fst snd]. destruct (Pos.eqb r0 d); [exact Hacc|].
      destruct (forget_fact F st d step x0) as [y0|] eqn:Hy0; [|exact Hacc].
      destruct (Pos.eq_dec r r0) as [Heq|Hdiff].
      - subst r0. assert (Ex : x0=x).
        { pose proof (PM.elements_complete _ r x0 (Hsub (r,x0) (or_introl eq_refl))) as Hx0.
          unfold get in Hx. rewrite Hx in Hx0. congruence. }
        subst x0. rewrite Hforget in Hy0. injection Hy0 as <-. apply PM.gss.
      - rewrite PM.gso by exact Hdiff. exact Hacc. }
    assert (Hhit : forall l acc,
      (forall pair, In pair l -> In pair (PM.elements (amap st))) ->
      In (r,x) l -> PM.find r (fold_left one l acc) = Some y).
    { induction l as [|[r0 x0] rest IH]; intros acc Hsub Hin; [contradiction|].
      cbn [fold_left]. destruct Hin as [Heq|Hin].
      - injection Heq as Er Ex. subst r0. subst x0. apply Hkeep; [intros pair Hp; apply Hsub; right; exact Hp|].
        unfold one. cbn [fst snd]. destruct (Pos.eqb_spec r d); [contradiction|].
        rewrite Hforget. apply PM.gss.
      - apply IH; [intros pair Hp; apply Hsub; right; exact Hp|exact Hin]. }
    unfold old in Hold. rewrite PM.fold_1 in Hold.
    change (PM.find r (fold_left one (PM.elements (amap st)) (PM.empty fact)) = None) in Hold.
    pose proof (Hhit _ (PM.empty fact) (fun pair Hp=>Hp) Helem) as Hsome.
    rewrite Hold in Hsome. discriminate.
Qed.

Lemma assign_sound : forall st rs rs' d new step,
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> step_rel d step rs (rs d) (rs' d) ->
    (forall y, new = Some y -> fact_okv F t rs' y (rs' d)) ->
    state_ok F t (assign F st d new step) rs'.
Proof.
  intros st rs rs' d new step Hst Hrs Ha Hstep Hnew. split.
  - intros r y Hg. unfold get, assign in Hg. cbn [amap] in Hg. unfold fact_ok.
    destruct new as [x |].
    + destruct (Pos.eq_dec r d) as [-> | Hne].
      * rewrite PM.gss in Hg. apply some_eq_t in Hg. subst y. exact (Hnew x eq_refl).
      * rewrite PM.gso in Hg by exact Hne.
        destruct (assign_map st d step r y Hg) as [Hr [x0 [Hx0 Hf]]].
        rewrite (Ha r Hr). apply (forget_fact_sound st rs rs' d step x0 y (rs r) Hst Hrs Ha Hstep); [| exact Hf].
        exact (proj1 Hst r x0 Hx0).
    + destruct (assign_map st d step r y Hg) as [Hr [x0 [Hx0 Hf]]].
      rewrite (Ha r Hr). apply (forget_fact_sound st rs rs' d step x0 y (rs r) Hst Hrs Ha Hstep); [| exact Hf].
      exact (proj1 Hst r x0 Hx0).
  - unfold assign. cbn [acons]. apply Forall_forall. intros k' Hk'. apply in_flat_map in Hk'.
    destruct Hk' as [k [Hk Hk']].
    destruct (cons_step F st d step k) as [k1 |] eqn:E1; [| destruct Hk']. destruct Hk' as [<- | []].
    apply (cons_step_sound st rs rs' d step k k1 Hst Hrs Ha Hstep); [| exact E1].
    exact (proj1 (Forall_forall _ _) (proj2 Hst) _ Hk).
Qed.

(* ------------------------------------------------------------------ *)
(* Steps d := d + s.                                                   *)

Lemma tsum_split_reg : forall rs d ts c e,
    filter (fun t => Pos.eqb (fst (fst t)) d) ts = [((d, e), c)] ->
    tsum wdn rs ts = c * atom_val wdn rs (d, e) + tsum wdn rs (filter (fun t => negb (Pos.eqb (fst (fst t)) d)) ts).
Proof.
  intros rs d ts c e H.
  transitivity (tsum wdn rs (filter (fun t => Pos.eqb (fst (fst t)) d) ts) +
                tsum wdn rs (filter (fun t => negb (Pos.eqb (fst (fst t)) d)) ts)).
  - apply tsum_filter.
  - rewrite H. cbn [tsum]. ring.
Qed.

Lemma narrow_step_sound : forall st rs d f g s v,
    state_ok F t st rs -> regs_ok F E rs -> narrow_step F st d f g = Some s ->
    sat wdn rs f v -> 0 <= v < 2 ^ wd F d -> g = None /\ step_rel d (Some s) rs (rs d) v.
Proof.
  intros st rs d f g s v Hst Hrs H [e0 [He0 Hv]] Hvr. unfold narrow_step in H.
  destruct g as [gg |]; [discriminate |]. split; [reflexivity |].
  set (rest := filter (fun t => negb (Pos.eqb (fst (fst t)) d)) (ft f)) in *.
  assert (Hmr : mentions {| fm := 64; ft := rest; flo := flo f; fhi := fhi f |} d = false).
  { unfold mentions. cbn [ft]. apply Bool.not_true_iff_false. intros Hex.
    apply existsb_exists in Hex. destruct Hex as [u [Hu Hp]]. unfold rest in Hu.
    apply filter_In in Hu. destruct Hu as [_ Hn]. rewrite Hp in Hn. discriminate. }
  destruct (filter (fun t => Pos.eqb (fst (fst t)) d) (ft f)) as [| [[r e] c] [| ? ?]] eqn:Ed; try discriminate.
  2: { destruct c as [| [p' | p' |] | p']; discriminate. }
  destruct c as [| [p | p |] | p]; try discriminate.
  assert (Hr : r = d).
  { assert (Hin : In ((r, e), 1) (filter (fun t => Pos.eqb (fst (fst t)) d) (ft f))) by (rewrite Ed; left; reflexivity).
    apply filter_In in Hin. destruct Hin as [_ Hin]. cbn [fst] in Hin. apply Pos.eqb_eq. exact Hin. }
  subst r.
  (* f's terms are d's term and the rest *)
  assert (Hsplit : tsum wdn rs (ft f) = 1 * atom_val wdn rs (d, e) + tsum wdn rs rest)
    by exact (tsum_split_reg rs d (ft f) 1 e Ed).
  destruct e; try discriminate.
  - destruct (Z.eqb_spec (fm f) 64) as [Hm |]; [| discriminate].
    destruct (Z.eqb_spec (wd F d) 64) as [Hw |]; [| discriminate]. cbn [andb] in H.
    apply some_eq_t in H. subst s. cbn [step_rel fm ft flo fhi].
    split; [reflexivity |]. split; [exact Hmr |]. exists e0. split; [exact He0 |].
    split; [| exact Hvr].
    rewrite Hm, Hsplit in Hv. cbn [atom_val] in Hv. rewrite Hw.
    replace (rs d + (tsum wdn rs rest + e0)) with (1 * rs d + tsum wdn rs rest + e0) by ring. exact Hv.
  - destruct (null rest && Z.eqb (flo f) (fhi f) && (wd F d <? 64) && (wd F d <=? fm f)) eqn:Ec0; [| discriminate].
    rewrite !andb_true_iff in Ec0. destruct Ec0 as [[[Hnull Hex] Hw] Hwm].
    apply Z.eqb_eq in Hex. apply Z.ltb_lt in Hw. apply Z.leb_le in Hwm.
    assert (Hrest : rest = []) by (destruct rest; [reflexivity | discriminate]).
    rewrite Hrest in Hsplit. cbn [tsum] in Hsplit.
    assert (e0 = flo f) by lia. subst e0.
    destruct (reg_range F st d) as [[x y] |] eqn:Er; [| discriminate].
    set (s0 := rep (wd F d) (flo f)) in H.
    destruct ((0 <=? x + s0) && (y + s0 <? 2 ^ wd F d)) eqn:Ec; [| discriminate].
    apply some_eq_t in H. subst s.
    apply andb_true_iff in Ec. destruct Ec as [E1 E2]. apply Z.leb_le in E1. apply Z.ltb_lt in E2.
    pose proof (reg_range_sound F E HE t st rs Hst Hrs d x y Er) as Hd.
    assert (Hwd : 1 <= wd F d <= 64) by (pose proof (wd_range F E HE d); unfold wd; lia).
    rewrite Hsplit in Hv. cbn [atom_val] in Hv.
    (* v and rs d + s0 agree modulo 2^(wd d) and both lie in [0, 2^(wd d)) *)
    assert (Hm : eqm (wd F d) v (rs d + s0)).
    { apply (eqm_trans _ _ (rs d + flo f)).
      - apply (eqm_weaken (fm f)); [lia |].
        replace (rs d + flo f) with (1 * rs d + 0 + flo f) by ring. exact Hv.
      - apply eqm_add; [lia | apply eqm_refl | apply eqm_sym; unfold s0; apply rep_eqm; lia]. }
    unfold eqm in Hm. rewrite !Z.mod_small in Hm by lia.
    cbn [step_rel fconst fm ft flo fhi tsum]. split; [reflexivity |]. split; [reflexivity |].
    exists s0. split; [lia |]. rewrite Hm, Z.add_0_l. split; [apply eqm_refl | lia].
  - destruct (Z.eqb_spec (fm f) 64) as [Hm |]; [| discriminate].
    destruct (Z.eqb_spec (wd F d) 64) as [Hw |]; [| discriminate]. cbn [andb] in H.
    apply some_eq_t in H. subst s. cbn [step_rel fm ft flo fhi].
    split; [reflexivity |]. split; [exact Hmr |]. exists e0. split; [exact He0 |].
    split; [| exact Hvr]. rewrite Hm, Hsplit in Hv. rewrite Hw.
    eapply eqm_trans; [exact Hv |].
    replace (rs d + (tsum wdn rs rest + e0)) with (1 * rs d + tsum wdn rs rest + e0) by ring.
    apply eqm_add; [lia | | apply eqm_refl].
    apply eqm_add; [lia | | apply eqm_refl].
    apply eqm_mul; [lia |]. rewrite <- Hw. apply sext_eqm.
Qed.

(* d := f where f is d's old exact form plus a constant: the value moved by
   that constant. *)
Lemma form_step_sound : forall st rs d f s v,
    state_ok F t st rs -> regs_ok F E rs -> form_step F st d f None = Some s ->
    sat wdn rs f v -> 0 <= v < 2 ^ wd F d -> step_rel d (Some s) rs (rs d) v.
Proof.
  intros st rs d f s v Hst Hrs H [ef [Hef Hv]] Hvr. unfold form_step in H.
  destruct (get st d) as [[h al vr | k | cs ex | cs ex] |] eqn:Eg; try discriminate.
  destruct (Z.eqb (fm f) (fm h) && (wd F d <=? fm f) && Z.eqb (flo f) (fhi f) && Z.eqb (flo h) (fhi h) &&
            terms_eqb (ft f) (ft h)) eqn:Ec; [| discriminate].
  rewrite !andb_true_iff in Ec. destruct Ec as [[[[Hm Hwm] Hf] Hh] Ht].
  apply Z.eqb_eq in Hm, Hf, Hh. apply Z.leb_le in Hwm. apply terms_eqb_eq in Ht.
  pose proof (proj1 Hst d _ Eg) as Hx. cbn [fact_ok fact_okv] in Hx. destruct Hx as [Hwh [[eh [Heh Hold]] _]].
  assert (ef = flo f) by lia. assert (eh = flo h) by lia. subst ef eh.
  pose proof (wd_range F E HE d) as Hw. pose proof (rs_range F E rs Hrs d) as Hr.
  assert (Hw' : 1 <= wd F d <= 64) by (unfold wd; lia).
  assert (Hr' : 0 <= rs d < 2 ^ wd F d) by (unfold wd; lia).
  unfold wf in Hwh.
  (* the new value is the old plus flo f - flo h, modulo 2^(fm f) *)
  assert (Hdiff : eqm (fm f) v (rs d + (flo f - flo h))).
  { rewrite Hm in Hv |- *. rewrite Ht in Hv.
    apply (eqm_trans _ _ (tsum wdn rs (ft h) + flo f)); [exact Hv |].
    replace (tsum wdn rs (ft h) + flo f) with ((tsum wdn rs (ft h) + flo h) + (flo f - flo h)) by ring.
    apply eqm_add; [lia | apply eqm_sym; exact Hold | apply eqm_refl]. }
  apply some_eq_t in H. subst s. cbn [step_rel fconst fm ft flo fhi tsum mentions existsb].
  split; [reflexivity |]. split; [reflexivity |]. exists (rep (wd F d) (flo f - flo h)). split; [lia |].
  split; [| exact Hvr]. rewrite Z.add_0_l.
  apply (eqm_trans _ _ (rs d + (flo f - flo h))); [apply (eqm_weaken (fm f)); [lia | exact Hdiff] |].
  apply eqm_add; [lia | apply eqm_refl | apply eqm_sym; apply rep_eqm; lia].
Qed.

Lemma step_fact_sound : forall st rs rs' d s y,
    state_ok F t st rs -> regs_ok F E rs -> step_rel d (Some s) rs (rs d) (rs' d) ->
    step_fact F st d s = Some y -> fact_okv F t rs' y (rs' d).
Proof.
  intros st rs rs' d s y Hst Hrs [Hfs [_ [es [Hes [Hsw _]]]]] H. unfold step_fact in H.
  set (SV := tsum wdn rs (ft s) + es) in *.
  pose proof (wd_range F E HE d) as Hwr.
  assert (Hw' : 1 <= wd F d <= 64) by (unfold wd; lia).
  (* the step's range bounds its value *)
  assert (Hsr : forall p q, form_range F st s = Some (p, q) -> p <= SV <= q)
    by (intros p q Ep; exact (form_range_sound F E HE t st rs Hst Hrs s p q es Ep Hes)).
  assert (Hshift : forall k, k <= step_align F st s -> k <= wd F d -> aligned k (rs d) -> aligned k (rs' d)).
  { intros k Hk Hkw Ha. unfold step_align in Hk.
    destruct (Z.eqb (flo s) (fhi s)) eqn:Ec; [| apply aligned_nonpos; lia].
    apply Z.eqb_eq in Ec. assert (Heq : es = flo s) by lia.
    apply (aligned_eqm (wd F d) k (rs' d) (rs d + SV)); [lia | lia | exact Hsw |].
    apply aligned_add; [exact Ha |]. unfold SV. rewrite Heq.
    apply (aligned_mono (form_align F st s)); [exact Hk |].
    apply (form_align_sound F E HE t st rs Hst Hrs); [exact Ec | lia]. }
  destruct (get st d) as [x |] eqn:Eg.
  2: { destruct (reg_range F st d) as [[a b] |] eqn:Er; [| discriminate].
       destruct (form_range F st s) as [[p q] |] eqn:Ep; [| discriminate]. apply some_eq_t in H. subst y.
       pose proof (reg_range_sound F E HE t st rs Hst Hrs d a b Er) as Hd. pose proof (Hsr p q eq_refl) as Hq.
       cbn [fact_okv]. split; [unfold wf, range_form; cbn [fm]; unfold wd; lia |].
       split; [| split; [apply aligned_nonpos; lia | discriminate]].
       exists (rs d + SV). unfold range_form. cbn [fm ft flo fhi tsum]. split; [lia |].
       rewrite Z.add_0_l. exact Hsw. }
  pose proof (proj1 Hst d x Eg) as Hx. unfold fact_ok in Hx.
  destruct x as [f al vr | k | | ]; try discriminate.
  - destruct (interval_only (Some (FForm f al vr)) || null (ft f)) eqn:Ei; [| discriminate].
    destruct (form_range F st s) as [[p q] |] eqn:Ep; [| discriminate]. pose proof (Hsr p q eq_refl) as Hq.
    destruct Hx as [Hwf [Hs [Hal Hvr]]]. unfold wf in Hwf.
    assert (Halign : aligned (Z.min (Z.min (reg_align F st d) (step_align F st s)) (wd F d)) (rs' d)).
    { apply Hshift; [lia | lia |]. apply (aligned_mono (reg_align F st d)); [lia |].
      exact (reg_align_sound F E HE t st rs Hst Hrs d). }
    destruct vr as [[lo hi] |]; apply some_eq_t in H; subst y.
    + pose proof (Hvr lo hi eq_refl) as Hd. cbn [fact_okv].
      split; [unfold wf, range_form; cbn [fm]; unfold wd; lia |]. split; [| split; [exact Halign | discriminate]].
      exists (rs d + SV). unfold range_form. cbn [fm ft flo fhi tsum]. split; [lia |].
      rewrite Z.add_0_l. exact Hsw.
    + assert (Hnil : ft f = []).
      { unfold interval_only in Ei. destruct (ft f); [reflexivity |]. cbn in Ei. discriminate. }
      destruct Hs as [e0 [He0 Hv]]. rewrite Hnil in Hv. cbn [tsum] in Hv.
      cbn [fact_okv]. split; [unfold wf, range_form; cbn [fm]; lia |]. split; [| split; [exact Halign | discriminate]].
      exists (e0 + SV). unfold range_form. cbn [fm ft flo fhi tsum]. split; [lia |]. rewrite Z.add_0_l.
      (* rs' d = rs d + SV modulo 2^(wd d) and rs d = e0 modulo 2^(fm f) *)
      apply (eqm_trans _ _ (rs d + SV)); [apply (eqm_weaken (wd F d)); [lia | exact Hsw] |].
      apply eqm_add; [lia | rewrite Z.add_0_l in Hv; apply (eqm_weaken (fm f)); [lia | exact Hv] | apply eqm_refl].
  - destruct (0 <? Z.min (Z.min k (step_align F st s)) (wd F d)) eqn:Ek; [| discriminate]. apply some_eq_t in H. subst y.
    cbn [fact_okv] in *. apply Hshift; [lia | lia |]. apply (aligned_mono k); [lia | exact Hx].
Qed.

(* ------------------------------------------------------------------ *)
(* The operands of a selp.                                             *)

(* The fact of an operand holds of its value read at width w. *)
Lemma sel_fact_sound : forall st rs o w x,
    state_ok F t st rs -> regs_ok F E rs -> 1 <= w <= 64 ->
    sel_fact F st o w = Some x -> fact_okv F t rs x (trunc (Z.to_nat w) (opv rs o)).
Proof.
  intros st rs o w x Hst Hrs Hw H.
  destruct o as [s | z]; cbn [sel_fact opv] in *.
  - destruct (Z.eqb_spec (wd F s) w) as [Hws | Hws]; cbn [negb] in H; [| discriminate].
    assert (Htr : trunc (Z.to_nat w) (rs s) = rs s).
    { pose proof (proj1 Hrs s) as Hr. unfold wdz in Hr. rewrite Hws in Hr.
      unfold trunc. rewrite Z2Nat.id by lia. apply Z.mod_small. lia. }
    rewrite Htr.
    (* the copy of s: its atom *)
    assert (Hcopy : forall al, aligned al (rs s) ->
              fact_okv F t rs (FForm (if Z.eqb w 64 then fatom 64 (s, EX) 1 else fatom w (s, EZ) 1) al None) (rs s)).
    { intros al Ha. cbn [fact_okv].
      destruct (Z.eqb w 64); (split; [unfold wf, fatom; rewrite mk_fm; lia |]);
        (split; [| split; [exact Ha | discriminate]]);
        rewrite <- (Z.mul_1_l (rs s)); [apply (sat_fatom wdn rs 64 (s, EX) 1) | apply (sat_fatom wdn rs w (s, EZ) 1)]; lia. }
    cbv zeta in H.
    destruct (get st s) as [y |] eqn:Es.
    + pose proof (proj1 Hst s y Es) as Hy. unfold fact_ok in Hy.
      destruct y as [f al vr | | | ]; cbv beta iota in H;
        try (injection H as <-; exact Hy).
      destruct vr as [vr |]; [injection H as <-; exact Hy |].
      destruct Hy as [Hwf [Hs [Ha _]]].
      injection H as <-. exact (Hcopy al Ha).
    + injection H as <-. apply Hcopy. apply aligned_nonpos. lia.
  - injection H as <-. cbn [fact_okv]. split; [unfold wf, fconst; cbn [fm]; lia |].
    split; [apply sat_fconst; unfold trunc; rewrite Z2Nat.id by lia; apply eqm_refl |].
    split; [apply aligned_nonpos; lia | discriminate].
Qed.

(* ------------------------------------------------------------------ *)
(* The destination of an instruction.                                  *)

Lemma guard_join_sound : forall st rs rs' d g total x,
    state_ok F t st rs -> regs_ok F E rs ->
    (holds g rs = true -> forall y, x = Some y -> fact_okv F t rs y (rs' d)) ->
    (holds g rs = false -> rs' d = rs d) ->
    (total = true -> holds g rs = false -> forall y, x = Some y -> fact_okv F t rs y (rs d)) ->
    forall z, guard_join F st d g total x = Some z -> fact_okv F t rs z (rs' d).
Proof.
  intros st rs rs' d g total x Hst Hrs H1 H2 H3 z Hz. unfold guard_join in Hz.
  destruct g as [gg |]; [| exact (H1 eq_refl z Hz)].
  destruct (holds (Some gg) rs) eqn:Hg.
  - destruct total; [exact (H1 eq_refl z Hz) |].
    destruct x as [y |]; [| destruct (get st d); discriminate].
    exact (join_fact_sound_r F E HE t rs Hrs st st (get st d) y z (rs' d) Hst (H1 eq_refl y eq_refl) Hz).
  - rewrite (H2 eq_refl). destruct total; [exact (H3 eq_refl eq_refl z Hz) |].
    destruct (get st d) as [x0 |] eqn:Eg; [| discriminate].
    exact (join_fact_sound_l F E HE t rs Hrs st st x0 x z (rs d) Hst (proj1 Hst d x0 Eg) Hz).
Qed.

Lemma place_sound : forall st rs rs' d x v,
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' ->
    (forall y, x = Some y -> fact_okv F t rs y v) ->
    forall z, place F st d x = Some z -> fact_okv F t rs' z v.
Proof.
  intros st rs rs' d x v Hst Hrs Ha Hx z Hz. unfold place in Hz.
  destruct x as [y |]; [| discriminate].
  exact (forget_fact_sound st rs rs' d None y z v Hst Hrs Ha I (Hx y eq_refl) Hz).
Qed.

Lemma transfer_dest_raw_sound : forall st rs rs' d g new total,
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    (holds g rs = true -> forall x, new = Some x -> fact_okv F t rs x (rs' d)) ->
    (holds g rs = false -> rs' d = rs d) ->
    (total = true -> holds g rs = false -> forall x, new = Some x -> fact_okv F t rs x (rs d)) ->
    state_ok F t (transfer_dest_raw F st d g new total) rs'.
Proof.
  intros st rs rs' d g new total Hst Hrs Hrs' Ha H1 H2 H3.
  (* the joined and placed fact *)
  assert (Hplace : state_ok F t (assign F st d (place F st d (guard_join F st d g total new)) None) rs').
  { apply (assign_sound st rs rs' d _ None Hst Hrs Ha I). intros y Hy.
    apply (place_sound st rs rs' d (guard_join F st d g total new) (rs' d) Hst Hrs Ha); [| exact Hy].
    apply (guard_join_sound st rs rs' d g total new Hst Hrs H1 H2 H3). }
  unfold transfer_dest_raw.
  destruct new as [[f al vr | | | ] |]; cbv zeta; try exact Hplace.
  destruct (mentions f d) eqn:Em.
  2: { destruct (form_step F st d f g) as [s |] eqn:Efs; [| exact Hplace].
       destruct g as [gg |]; [unfold form_step in Efs; discriminate |].
       destruct (H1 eq_refl _ eq_refl) as [_ [Hsf _]].
       apply (assign_sound st rs rs' d _ (Some s) Hst Hrs Ha);
         [exact (form_step_sound st rs d f s (rs' d) Hst Hrs Efs Hsf (proj1 Hrs' d)) |].
       intros y Hy.
       apply (place_sound st rs rs' d (guard_join F st d None total (Some (FForm f al vr))) (rs' d) Hst Hrs Ha); [| exact Hy].
       apply (guard_join_sound st rs rs' d None total _ Hst Hrs H1 H2 H3). }
  destruct (narrow_step F st d f g) as [s |] eqn:Ens.
  - destruct (narrow_step_sound st rs d f g s (rs' d) Hst Hrs Ens) as [Hg Hrel].
    + destruct (H1 ltac:(destruct g; [discriminate | reflexivity]) _ eq_refl) as [_ [Hs _]]. exact Hs.
    + exact (proj1 Hrs' d).
    + apply (assign_sound st rs rs' d _ (Some s) Hst Hrs Ha Hrel). intros y Hy.
      exact (step_fact_sound st rs rs' d s y Hst Hrs Hrel Hy).
  - exact Hplace.
Qed.

Lemma canonical_fact_sound : forall rs d x y v,
    (forall z, x = Some z -> fact_okv F t rs z v) ->
    canonical_fact F d x = Some y -> fact_okv F t rs y v.
Proof.
  intros rs d x y v Hx Hy. unfold canonical_fact in Hy.
  destruct x as [[f al vr | k | cs ex | cs ex] |]; try (apply Hx; exact Hy).
  destruct (null (ft f) && Z.eqb (flo f) (fhi f) && (wd F d <=? fm f)) eqn:Hc;
    [| apply Hx; exact Hy].
  injection Hy as <-. rewrite !andb_true_iff in Hc. destruct Hc as [[Ht He] Hm].
  apply Z.eqb_eq in He. apply Z.leb_le in Hm.
  assert (Hnil : ft f = []) by (destruct (ft f); [reflexivity | discriminate]).
  destruct (Hx _ eq_refl) as [Hwf [[e [Heb Hv]] [Hal Hvr]]].
  assert (Ee : e = flo f) by lia. subst e. rewrite Hnil in Hv.
  cbn [GateForm.tsum] in Hv. rewrite Z.add_0_l in Hv.
  pose proof (ok_widths F E HE d) as Hw. change (1 <= wd F d <= 64) in Hw.
  split; [unfold wf, fconst; cbn; lia |]. split.
  - apply sat_fconst. eapply eqm_trans.
    + apply (eqm_weaken (fm f) (wd F d)); [lia | exact Hv].
    + apply eqm_sym. apply rep_eqm. lia.
  - split; assumption.
Qed.

Lemma transfer_dest_sound : forall st rs rs' d g new total,
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    (holds g rs = true -> forall x, new = Some x -> fact_okv F t rs x (rs' d)) ->
    (holds g rs = false -> rs' d = rs d) ->
    (total = true -> holds g rs = false -> forall x, new = Some x -> fact_okv F t rs x (rs d)) ->
    state_ok F t (transfer_dest F st d g new total) rs'.
Proof.
  intros st rs rs' d g new total Hst Hrs Hrs' Ha H1 H2 H3.
  unfold transfer_dest. apply (transfer_dest_raw_sound st rs rs' d g _ total Hst Hrs Hrs' Ha).
  - intros Hg x Hx. eapply canonical_fact_sound; [exact (H1 Hg) | exact Hx].
  - exact H2.
  - intros Ht Hg x Hx. eapply canonical_fact_sound; [exact (H3 Ht Hg) | exact Hx].
Qed.


(* ------------------------------------------------------------------ *)
(* Instructions that forget their destinations.                        *)

Lemma forget_all_write : forall ds st rs arb tt,
    state_ok F t st rs -> regs_ok F E rs -> forallb (fun d => negb (is_fixed F d)) ds = true ->
    state_ok F t (fold_left (fun s d => assign F s d None None) ds st) (write_all wdn arb rs tt ds) /\
    regs_ok F E (write_all wdn arb rs tt ds).
Proof.
  induction ds as [| d ds IH]; intros st rs arb tt Hst Hrs Hds; cbn [fold_left write_all]; [split; assumption |].
  cbn [forallb] in Hds. apply andb_true_iff in Hds. destruct Hds as [Hd Hds]. apply negb_true_iff in Hd.
  apply IH; [| apply upd_regs_ok; assumption | exact Hds].
  apply (assign_sound st rs _ d None None Hst Hrs (upd_agree rs d _) I). intros y Hy. discriminate.
Qed.

Lemma forget_all_same : forall ds st rs,
    state_ok F t st rs -> regs_ok F E rs ->
    state_ok F t (fold_left (fun s d => assign F s d None None) ds st) rs.
Proof.
  induction ds as [| d ds IH]; intros st rs Hst Hrs; cbn [fold_left]; [exact Hst |].
  apply IH; [| exact Hrs].
  apply (assign_sound st rs rs d None None Hst Hrs (fun r _ => eq_refl) I). intros y Hy. discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(* Rule R8: local memory cells.                                        *)

(* Forgetting cell p holds whether or not the step writes p. *)
Lemma forget_cell : forall s r r' p,
    state_ok F t s r -> regs_ok F E r -> is_fixed F p = false ->
    (r' = r \/ exists z, r' = upd wdn r p z) ->
    state_ok F t (assign F s p None None) r' /\ regs_ok F E r'.
Proof.
  intros s r r' p Hs Hr Hp [-> | [z ->]].
  - split; [| exact Hr].
    apply (assign_sound s r r p None None Hs Hr (fun _ _ => eq_refl) I). intros y Hy. discriminate.
  - split; [| apply upd_regs_ok; assumption].
    apply (assign_sound s r _ p None None Hs Hr (upd_agree r p z) I). intros y Hy. discriminate.
Qed.

(* A store changes at most the cell's register. *)
Lemma lstore_cell_cases : forall tt x width vz r p u c,
    lstore_cell wdn (e_arb E) (e_init E) tt x width vz r (p, u, c) = r \/
    exists z, lstore_cell wdn (e_arb E) (e_init E) tt x width vz r (p, u, c) = upd wdn r p z.
Proof.
  intros tt x width vz r p u c. unfold lstore_cell.
  destruct (_ && _); [right; eexists; reflexivity |].
  destruct (_ && _); [right; eexists; reflexivity | left; reflexivity].
Qed.

Lemma is_cell_in : forall p u c, In (p, u, c) (lf_cells F) -> is_cell F p = true.
Proof.
  intros p u c Hin. unfold is_cell. apply existsb_exists. exists (p, u, c). split; [exact Hin | apply Pos.eqb_refl].
Qed.

(* A store the step makes: each cell's facts hold of its new value, when the
   store's location is the local address x and its known value vz is the
   value of v in registers rs1, which agree with r on every register that
   is not a cell. *)
Lemma store_cells_sound : forall l s r g loc width v fr x vz rs1 tt,
    incl l (lf_cells F) -> state_ok F t s r -> regs_ok F E r -> 0 < width ->
    (forall q, is_cell F q = false -> r q = rs1 q) ->
    (forall u lo hi, loc = Some (u, lo, hi) ->
       (exists sz, lookup (lf_locals F) u = Some sz) /\ 0 <= lo /\ exists e, lo <= e <= hi /\ x = e_init E u + e) ->
    (forall o, v = Some o -> g = None -> vz = Some (opv rs1 o)) ->
    state_ok F t (fold_left (store_cell F g loc width v fr) l s)
                 (fold_left (lstore_cell wdn (e_arb E) (e_init E) tt x width vz) l r) /\
    regs_ok F E (fold_left (lstore_cell wdn (e_arb E) (e_init E) tt x width vz) l r).
Proof.
  induction l as [| [[p u'] c'] l IH]; intros s r g loc width v fr x vz rs1 tt Hl Hs Hr Hw Hag Hloc Hv;
    cbn [fold_left]; [split; assumption |].
  assert (Hin : In (p, u', c') (lf_cells F)) by (apply Hl; left; reflexivity).
  destruct (cell_in F E HE t r Hr p u' c' Hin) as [Hpf [Hwp _]].
  pose proof (is_cell_in p u' c' Hin) as Hpc.
  set (r' := lstore_cell wdn (e_arb E) (e_init E) tt x width vz r (p, u', c')).
  assert (Hcases : r' = r \/ exists z, r' = upd wdn r p z) by exact (lstore_cell_cases tt x width vz r p u' c').
  assert (Hstep : state_ok F t (store_cell F g loc width v fr s (p, u', c')) r' /\ regs_ok F E r').
  { unfold store_cell. destruct loc as [[[u lo] hi] |]; [| exact (forget_cell s r r' p Hs Hr Hpf Hcases)].
    destruct (Hloc u lo hi eq_refl) as [[[sz f] Hsz] [Hlo [e [He Hx]]]].
    destruct (cell_op_sound F E HE t r Hr fr u sz f lo hi e width p u' c' Hsz Hlo He Hw Hin) as [HPut HKeep].
    destruct (cell_op_of F fr u lo hi width (p, u', c')) eqn:Eop.
    - (* the store leaves the cell *)
      destruct (HKeep eq_refl) as [K1 K2].
      assert (Hr' : r' = r) by (unfold r', lstore_cell; rewrite Hx, K1, K2; reflexivity).
      rewrite Hr'. split; assumption.
    - (* the store writes the cell *)
      destruct (HPut eq_refl) as [P1 P2].
      assert (Hr' : r' = upd wdn r p (match vz with Some z => z | None => e_arb E tt p end)).
      { unfold r', lstore_cell. rewrite Hx, P1, P2, !Z.eqb_refl. reflexivity. }
      destruct g as [gg |]; [exact (forget_cell s r r' p Hs Hr Hpf Hcases) |].
      destruct v as [[r0 | z] |]; [| | exact (forget_cell s r r' p Hs Hr Hpf Hcases)].
      + destruct (Z.eqb (wd F r0) (wd F p) && negb (is_cell F r0)) eqn:Ec;
          [| exact (forget_cell s r r' p Hs Hr Hpf Hcases)].
        apply andb_true_iff in Ec. destruct Ec as [Ew Ec]. apply negb_true_iff in Ec.
        rewrite (Hv _ eq_refl eq_refl) in Hr'. cbn [opv] in Hr'. rewrite <- (Hag r0 Ec) in Hr'.
        assert (Hrr : regs_ok F E r') by (rewrite Hr'; apply upd_regs_ok; assumption).
        split; [| exact Hrr].
        apply (transfer_dest_sound s r r' p None _ false Hs Hr Hrr);
          [rewrite Hr'; apply upd_agree | | discriminate | discriminate].
        intros _ x0 Hx0. rewrite Hr'. unfold upd at 1. rewrite Pos.eqb_refl.
        apply (value_sound F E HE t s r Hs Hr (IMov None p r0) p x0); [| reflexivity | exact Hx0].
        unfold wf_instr. cbn [dest guard_of]. rewrite Hpf. apply Z.eqb_eq in Ew.
        rewrite Ew, Z.eqb_refl. reflexivity.
      + rewrite (Hv _ eq_refl eq_refl) in Hr'. cbn [opv] in Hr'.
        assert (Hrr : regs_ok F E r') by (rewrite Hr'; apply upd_regs_ok; assumption).
        split; [| exact Hrr].
        apply (transfer_dest_sound s r r' p None _ false Hs Hr Hrr);
          [rewrite Hr'; apply upd_agree | | discriminate | discriminate].
        intros _ x0 Hx0. rewrite Hr'. unfold upd at 1. rewrite Pos.eqb_refl.
        apply (value_sound F E HE t s r Hs Hr (IImm None p z) p x0); [| reflexivity | exact Hx0].
        unfold wf_instr. cbn [dest guard_of]. rewrite Hpf. reflexivity.
    - (* the store may overlap the cell *)
      exact (forget_cell s r r' p Hs Hr Hpf Hcases). }
  destruct Hstep as [Hs' Hr'].
  apply (IH _ _ g loc width v fr x vz rs1 tt); [intros y Hy; apply Hl; right; exact Hy | exact Hs' | exact Hr' | exact Hw | | exact Hloc | exact Hv].
  intros q Hq. rewrite <- (Hag q Hq).
  destruct Hcases as [-> | [z ->]]; [reflexivity |].
  unfold upd. destruct (Pos.eqb_spec q p) as [-> |]; [rewrite Hpc in Hq; discriminate | reflexivity].
Qed.

(* A store the step does not make, under a guard, or at an unknown
   location, forgets or keeps each cell. *)
Lemma store_cells_same : forall l s r g loc width v fr,
    incl l (lf_cells F) -> state_ok F t s r -> regs_ok F E r -> (g <> None \/ loc = None) ->
    state_ok F t (fold_left (store_cell F g loc width v fr) l s) r.
Proof.
  induction l as [| [[p u'] c'] l IH]; intros s r g loc width v fr Hl Hs Hr Hg; cbn [fold_left]; [exact Hs |].
  assert (Hin : In (p, u', c') (lf_cells F)) by (apply Hl; left; reflexivity).
  destruct (cell_in F E HE t r Hr p u' c' Hin) as [Hpf _].
  apply (IH _ r g loc width v fr); [intros y Hy; apply Hl; right; exact Hy | | exact Hr | exact Hg].
  assert (Hf : state_ok F t (assign F s p None None) r) by exact (proj1 (forget_cell s r r p Hs Hr Hpf (or_introl eq_refl))).
  unfold store_cell. destruct loc as [[[u lo] hi] |]; [| exact Hf].
  destruct Hg as [Hg | Hg]; [| discriminate].
  destruct (cell_op_of F fr u lo hi width (p, u', c')); [exact Hs | | exact Hf].
  destruct g as [gg |]; [exact Hf | contradiction].
Qed.

(* Entering a frame forgets the cells of its variables, which the step
   gives arbitrary values. *)
Lemma frame_cells_sound : forall l s r vs tt,
    incl l (lf_cells F) -> state_ok F t s r -> regs_ok F E r ->
    state_ok F t (fold_left (fun s0 k => let '(p, u, _) := k in
                                         if existsb (Pos.eqb u) vs then assign F s0 p None None else s0) l s)
                 (fold_left (lframe_cell wdn (e_arb E) tt vs) l r) /\
    regs_ok F E (fold_left (lframe_cell wdn (e_arb E) tt vs) l r).
Proof.
  induction l as [| [[p u] c] l IH]; intros s r vs tt Hl Hs Hr; cbn [fold_left]; [split; assumption |].
  assert (Hin : In (p, u, c) (lf_cells F)) by (apply Hl; left; reflexivity).
  destruct (cell_in F E HE t r Hr p u c Hin) as [Hpf _].
  assert (Hincl : incl l (lf_cells F)) by (intros y Hy; apply Hl; right; exact Hy).
  replace (lframe_cell wdn (e_arb E) tt vs r (p, u, c))
    with (if existsb (Pos.eqb u) vs then upd wdn r p (e_arb E tt p) else r) by reflexivity.
  cbv beta iota. destruct (existsb (Pos.eqb u) vs).
  - destruct (forget_cell s r (upd wdn r p (e_arb E tt p)) p Hs Hr Hpf (or_intror (ex_intro _ _ eq_refl)))
      as [Hs' Hr'].
    exact (IH _ _ vs tt Hincl Hs' Hr').
  - exact (IH _ _ vs tt Hincl Hs Hr).
Qed.

End AtStep.

(* ------------------------------------------------------------------ *)
(* One step.                                                           *)

Local Notation stepF :=
  (step prog wdn (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F)).

Lemma step_regs_dest : forall t s s' i d,
    fetch prog (blk s) (idx s) = Some i -> dest i = Some d -> stepF t s = Some s' ->
    regs s' = if holds (guard_of i) (regs s) then upd wdn (regs s) d (result (e_mem E) (e_arb E) (e_shm E) t (regs s) i) else regs s.
Proof.
  intros t s s' i d Hf Hd H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'.
  destruct (holds (guard_of i) (regs s)); cbn [negb]; [| reflexivity].
  destruct i; cbn [dest] in Hd; try discriminate; injection Hd as <-; reflexivity.
Qed.

Lemma step_regs_def : forall t s s' g ds,
    fetch prog (blk s) (idx s) = Some (IDef g ds) -> stepF t s = Some s' ->
    regs s' = if holds g (regs s) then write_all wdn (e_arb E) (regs s) t ds else regs s.
Proof.
  intros t s s' g ds Hf H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'.
  cbn [guard_of]. destruct (holds g (regs s)); reflexivity.
Qed.

(* The registers after a global access: its destinations, and the cells a
   generic store or atomic writes in local memory. *)
Definition mem_regs (t : nat) (rs : reg -> Z) (g : guard) (k : mkind) (gen : bool) (a : reg) (off width : Z)
    (v : option opnd) (ds : list reg) : reg -> Z :=
  if holds g rs then
    let rs1 := write_all wdn (e_arb E) rs t ds in
    match k with
    | KLoad => rs1
    | _ => if gen then gstore wdn (e_arb E) (e_init E) (lf_cells F) (lf_local_window F) t rs1 (trunc 64 (rs a + off))
                            width (match k, v with KStore, Some o => Some (opv rs o) | _, _ => None end)
           else rs1
    end
  else rs.

Lemma step_regs_mem : forall t s s' g k gen a off width v fr ds,
    fetch prog (blk s) (idx s) = Some (IMem g k gen a off width v fr ds) -> stepF t s = Some s' ->
    regs s' = mem_regs t (regs s) g k gen a off width v ds.
Proof.
  intros t s s' g k gen a off width v fr ds Hf H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'.
  unfold mem_regs. cbn [guard_of]. destruct (holds g (regs s)); reflexivity.
Qed.

Lemma step_regs_ldl : forall t s s' g bits d a off fr,
    fetch prog (blk s) (idx s) = Some (ILdL g bits d a off fr) -> stepF t s = Some s' ->
    regs s' = if holds g (regs s)
              then upd wdn (regs s) d (lload wdn (e_arb E) (e_init E) (lf_cells F) (lf_locals F) t (regs s) d
                                             (trunc 64 (regs s a + off)) bits fr)
              else regs s.
Proof.
  intros t s s' g bits d a off fr Hf H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'.
  cbn [guard_of]. destruct (holds g (regs s)); reflexivity.
Qed.

Lemma step_regs_stl : forall t s s' g a off width v fr,
    fetch prog (blk s) (idx s) = Some (IStL g a off width v fr) -> stepF t s = Some s' ->
    regs s' = if holds g (regs s)
              then lstore wdn (e_arb E) (e_init E) (lf_cells F) t (regs s) (trunc 64 (regs s a + off)) width
                          (option_map (opv (regs s)) v)
              else regs s.
Proof.
  intros t s s' g a off width v fr Hf H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'.
  cbn [guard_of]. destruct (holds g (regs s)); reflexivity.
Qed.

Lemma step_regs_frame : forall t s s' vs,
    fetch prog (blk s) (idx s) = Some (IFrame vs) -> stepF t s = Some s' ->
    regs s' = lframe wdn (e_arb E) (lf_cells F) t (regs s) vs.
Proof.
  intros t s s' vs Hf H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'. reflexivity.
Qed.

Lemma step_regs_same : forall t s s' i,
    fetch prog (blk s) (idx s) = Some i -> dest i = None ->
    (forall g ds, i <> IDef g ds) ->
    (forall g k gen a off width v fr ds, i <> IMem g k gen a off width v fr ds) ->
    (forall g bits d a off fr, i <> ILdL g bits d a off fr) ->
    (forall g a off width v fr, i <> IStL g a off width v fr) ->
    (forall vs, i <> IFrame vs) ->
    stepF t s = Some s' -> regs s' = regs s.
Proof.
  intros t s s' i Hf Hd Hdef Hmem Hldl Hstl Hfr H. unfold step in H. rewrite Hf in H. apply some_eq_t in H. subst s'.
  destruct (holds (guard_of i) (regs s)); cbn [negb]; [| reflexivity].
  destruct i; cbn [dest] in Hd; try discriminate; try reflexivity; exfalso;
    first [eapply Hmem; reflexivity | eapply Hdef; reflexivity | eapply Hldl; reflexivity | eapply Hstl; reflexivity
          | eapply Hfr; reflexivity].
Qed.

(* ------------------------------------------------------------------ *)
(* Rule R8's instructions.                                             *)

(* A global access forgets its destinations; a generic store or atomic
   writes, keeps, or forgets each cell, as the local address it may have
   decides. *)
Lemma mem_sound : forall t st rs g k gen a off width v fr ds,
    state_ok F t st rs -> regs_ok F E rs -> wf_instr F (IMem g k gen a off width v fr ds) = true ->
    state_ok F (S t) (transfer_live F st (IMem g k gen a off width v fr ds)) (mem_regs t rs g k gen a off width v ds) /\
    regs_ok F E (mem_regs t rs g k gen a off width v ds).
Proof.
  intros t st rs g k gen a off width v fr ds Hst Hrs Hwf.
  pose proof (state_ok_later t st rs Hst) as Hst1.
  unfold wf_instr in Hwf. cbn [guard_of] in Hwf. rewrite !andb_true_iff in Hwf.
  destruct Hwf as [[Hfix [_ Hw]] _]. apply Z.ltb_lt in Hw.
  cbn [transfer_live]. set (st1 := fold_left (fun s0 d => assign F s0 d None None) ds st).
  unfold mem_regs. destruct (holds g rs) eqn:Hg.
  - destruct (forget_all_write (S t) ds st rs (e_arb E) t Hst1 Hrs Hfix) as [Hs1 Hr1]. fold st1 in Hs1.
    set (rs1 := write_all wdn (e_arb E) rs t ds) in *.
    (* the value the store writes *)
    assert (Hv : forall o, (match k, ds with KStore, [] => v | _, _ => None end) = Some o -> g = None ->
                 (match k, v with KStore, Some o => Some (opv rs o) | _, _ => None end) = Some (opv rs1 o)).
    { intros o Ho _. destruct k; [discriminate | | discriminate].
      destruct ds; [| discriminate]. subst v. reflexivity. }
    destruct k; [split; assumption | |]; (destruct gen; [| split; assumption]).
    all: unfold gstore.
    all: destruct (local_of_generic F st (addr_form F st a off) width) as [[[u lo] hi] |] eqn:El.
    1, 3: destruct (local_of_generic_sound F E HE t rs Hrs st a off width u lo hi Hst ltac:(lia) El)
            as [Hu [Hlo [Hwlo [Hwhi Hx]]]];
          rewrite (proj2 (Z.leb_le _ _) Hwlo), (proj2 (Z.leb_le _ _) Hwhi); cbn [andb];
          eapply (store_cells_sound (S t) (lf_cells F) st1 rs1 g (Some (u, lo, hi)) width _ fr _ _ rs1 t (incl_refl _)
                   Hs1 Hr1 Hw (fun _ _ => eq_refl)); [| exact Hv];
          intros u' lo' hi' Huc; injection Huc as <- <- <-; split; [exact Hu | split; [exact Hlo | exact Hx]].
    all: destruct (_ && _);
      [eapply (store_cells_sound (S t) (lf_cells F) st1 rs1 g None width _ fr _ _ rs1 t (incl_refl _)
                Hs1 Hr1 Hw (fun _ _ => eq_refl)); [intros u' lo' hi' Huc; discriminate | exact Hv] |
       split; [exact (store_cells_same (S t) (lf_cells F) st1 rs1 g None width _ fr (incl_refl _) Hs1 Hr1
                        (or_intror eq_refl)) | exact Hr1]].
  - assert (Hs1 : state_ok F (S t) st1 rs) by exact (forget_all_same (S t) ds st rs Hst1 Hrs).
    assert (Hgn : g <> None) by (intros ->; discriminate).
    split; [| exact Hrs].
    destruct k; [exact Hs1 | |]; (destruct gen; [| exact Hs1]).
    all: exact (store_cells_same (S t) (lf_cells F) st1 rs g _ width _ fr (incl_refl _) Hs1 Hrs (or_introl Hgn)).
Qed.

(* A local load gives d the value of the cell of its address and width, or
   forgets d. *)
Lemma ldl_sound : forall t st rs g bits d a off fr,
    state_ok F t st rs -> regs_ok F E rs -> wf_instr F (ILdL g bits d a off fr) = true ->
    let rs' := if holds g rs
               then upd wdn rs d (lload wdn (e_arb E) (e_init E) (lf_cells F) (lf_locals F) t rs d
                                        (trunc 64 (rs a + off)) bits fr)
               else rs in
    state_ok F (S t) (transfer_live F st (ILdL g bits d a off fr)) rs' /\ regs_ok F E rs'.
Proof.
  intros t st rs g bits d a off fr Hst Hrs Hwf rs'.
  pose proof (state_ok_later t st rs Hst) as Hst1.
  unfold wf_instr in Hwf. cbn [guard_of] in Hwf. rewrite !andb_true_iff in Hwf.
  destruct Hwf as [[Hdf _] Hgw]. apply negb_true_iff in Hdf.
  assert (Hrs' : regs_ok F E rs') by (unfold rs'; destruct (holds g rs); [apply upd_regs_ok |]; assumption).
  split; [| exact Hrs'].
  assert (Ha : agree d rs rs') by (unfold rs'; destruct (holds g rs); [apply upd_agree | intros r _; reflexivity]).
  assert (Hold : holds g rs = false -> rs' d = rs d) by (intros Hg; unfold rs'; rewrite Hg; reflexivity).
  assert (Hnew : holds g rs = true ->
                 rs' d = trunc (wdn d) (lload wdn (e_arb E) (e_init E) (lf_cells F) (lf_locals F) t rs d
                                              (trunc 64 (rs a + off)) bits fr))
    by (intros Hg; unfold rs'; rewrite Hg; unfold upd; rewrite Pos.eqb_refl; reflexivity).
  (* without a cell, d is forgotten *)
  assert (Hnone : state_ok F (S t) (transfer_dest F st d g None false) rs')
    by (apply (transfer_dest_sound (S t) st rs rs' d g None false Hst1 Hrs Hrs' Ha); [discriminate | exact Hold | discriminate]).
  cbn [transfer_live]. unfold load_cell.
  destruct (local_at F (addr_form F st a off)) as [[u c] |] eqn:El; [| exact Hnone].
  destruct (live (lf_locals F) fr u) eqn:Hlive; cbn [negb]; [| exact Hnone].
  destruct (local_at_sound F E HE t rs Hrs st a off u c Hst El) as [[[sz f] Hsz] [Hc Hx]].
  lazymatch goal with
  | |- context [find ?f (lf_cells F)] => destruct (find f (lf_cells F)) as [[[p u0] c0] |] eqn:Efd; [| exact Hnone]
  end.
  pose proof (cell_at_sound F E HE t rs Hrs fr u sz f c bits p u0 c0 Hsz (proj1 Hc) Hlive Efd) as Hcell.
  assert (Hload : lload wdn (e_arb E) (e_init E) (lf_cells F) (lf_locals F) t rs d (trunc 64 (rs a + off)) bits fr = rs p)
    by (unfold lload; rewrite Hx, Hcell; reflexivity).
  rewrite Hload in Hnew.
  (* the cell's value, at d's width *)
  assert (Hval : forall i, wf_instr F i = true -> dest i = Some d -> guard_of i = g ->
                 trunc (wdn d) (result (e_mem E) (e_arb E) (e_shm E) t rs i) = trunc (wdn d) (rs p) ->
                 state_ok F (S t) (transfer_dest F st d g (value F st i) false) rs').
  { intros i Hwi Hdi Hgi Hri.
    apply (transfer_dest_sound (S t) st rs rs' d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
    intros Hg y Hy. rewrite (Hnew Hg), <- Hri. apply fact_okv_later.
    exact (value_sound F E HE t st rs Hst Hrs i d y Hwi Hdi Hy). }
  destruct (Z.eqb (wd F d) (wd F p)) eqn:Edp.
  - apply Hval; [| reflexivity | reflexivity | reflexivity].
    unfold wf_instr. cbn [dest guard_of]. rewrite Hdf, Edp. exact Hgw.
  - destruct (wd F p <? wd F d) eqn:Elt; [| exact Hnone].
    apply Hval; [| reflexivity | reflexivity |].
    + assert (E1 : Z.eqb (wd F d) (Z.of_nat (wdn d)) = true) by (apply Z.eqb_eq; reflexivity).
      assert (E2 : (Z.of_nat (wdn p) <=? wd F p) = true) by (apply Z.leb_le; unfold wd; lia).
      unfold wf_instr. cbn [dest guard_of]. rewrite Hdf, E1, E2. exact Hgw.
    + cbn [result extend]. f_equal. apply trunc_small. exact (rs_range F E rs Hrs p).
Qed.

(* A local store writes, keeps, or forgets each cell, as its local address
   decides. *)
Lemma stl_sound : forall t st rs g a off width v fr,
    state_ok F t st rs -> regs_ok F E rs -> wf_instr F (IStL g a off width v fr) = true ->
    let rs' := if holds g rs
               then lstore wdn (e_arb E) (e_init E) (lf_cells F) t rs (trunc 64 (rs a + off)) width (option_map (opv rs) v)
               else rs in
    state_ok F (S t) (transfer_live F st (IStL g a off width v fr)) rs' /\ regs_ok F E rs'.
Proof.
  intros t st rs g a off width v fr Hst Hrs Hwf rs'.
  pose proof (state_ok_later t st rs Hst) as Hst1.
  unfold wf_instr in Hwf. cbn [guard_of dest] in Hwf. rewrite !andb_true_iff in Hwf.
  destruct Hwf as [[_ Hw] _]. apply Z.ltb_lt in Hw.
  cbn [transfer_live]. unfold store_cells, rs'. destruct (holds g rs) eqn:Hg.
  - unfold lstore.
    destruct (local_span F st (addr_form F st a off) width) as [[[u lo] hi] |] eqn:El.
    + destruct (local_span_sound F E HE t rs Hrs st a off width u lo hi Hst El) as [Hu [Hlo [_ Hx]]].
      eapply (store_cells_sound (S t) (lf_cells F) st rs g (Some (u, lo, hi)) width v fr _ _ rs t (incl_refl _)
               Hst1 Hrs Hw (fun _ _ => eq_refl)).
      * intros u' lo' hi' Huc. injection Huc as <- <- <-. split; [exact Hu |]. split; [exact Hlo | exact Hx].
      * intros o Ho _. rewrite Ho. reflexivity.
    + eapply (store_cells_sound (S t) (lf_cells F) st rs g None width v fr _ _ rs t (incl_refl _)
               Hst1 Hrs Hw (fun _ _ => eq_refl)).
      * intros u' lo' hi' Huc. discriminate.
      * intros o Ho _. rewrite Ho. reflexivity.
  - split; [| exact Hrs].
    apply (store_cells_same (S t) (lf_cells F) st rs g _ width v fr (incl_refl _) Hst1 Hrs).
    left. intros ->. discriminate.
Qed.

(* Entering a frame forgets the cells of its variables. *)
Lemma frame_sound : forall t st rs vs,
    state_ok F t st rs -> regs_ok F E rs ->
    state_ok F (S t) (transfer_live F st (IFrame vs)) (lframe wdn (e_arb E) (lf_cells F) t rs vs) /\
    regs_ok F E (lframe wdn (e_arb E) (lf_cells F) t rs vs).
Proof.
  intros t st rs vs Hst Hrs. cbn [transfer_live]. unfold frame_cells, lframe.
  exact (frame_cells_sound (S t) (lf_cells F) st rs vs t (incl_refl _) (state_ok_later t st rs Hst) Hrs).
Qed.

(* ------------------------------------------------------------------ *)
(* Rule R4's relation.                                                 *)

(* d's new value is at most the exact value of m's form, which reads the
   registers other than d. *)
Lemma and_cons1_sound : forall t st w d m rs rs' s,
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' -> 1 <= w <= 64 -> wd F m = w ->
    0 <= rs' d -> rs' d <= trunc (Z.to_nat w) (rs m) ->
    regs_ok F E rs' -> state_ok F (S t) s rs' ->
    state_ok F (S t) (and_cons1 F st w d m s) rs'.
Proof.
  intros t st w d m rs rs' s Hst Hrs Ha Hw Hm Hlo Hhi Hrs' Hs. unfold and_cons1.
  destruct (cmp_range F st (OReg m) w false) as [[x y] |] eqn:Er; [| exact Hs].
  assert (Hom : owidth F (OReg m) w) by (intros r Hr; injection Hr as <-; exact Hm).
  destruct (cmp_exact F E HE t rs Hrs st (OReg m) w false x y Hst Hw Hom Er) as [e [He Hf]].
  pose proof (cmp_range_sound F E HE t rs Hrs st (OReg m) w false x y Hst Hw Hom Er) as Hxy.
  unfold cval, extend in Hf, Hxy. cbn [opv] in Hf, Hxy.
  set (f := cmp_form F st (OReg m) w false) in *.
  destruct (null (ft f) || (2 ^ w - 1 <=? y)); [exact Hs |].
  destruct (existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f)) eqn:Ed; [exact Hs |].
  apply (add_cons_ok F E HE (S t) rs' Hrs' s _ Hs).
  unfold bound_cons. cbn [cons_ok].
  rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app, tsum_tscale. cbn [tsum].
  rewrite (tsum_agree wdn d rs rs' (ft f) Ha Ed).
  destruct (Z.eqb (wd F d) 64); cbn [atom_val]; lia.
Qed.

Lemma and_cons_sound : forall t st st1 n d a rb rs rs',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' ->
    wd F d = Z.of_nat n -> wd F a = Z.of_nat n -> wd F rb = Z.of_nat n ->
    rs' d = trunc (wdn d) (Z.land (trunc n (rs a)) (trunc n (rs rb))) ->
    regs_ok F E rs' -> state_ok F (S t) st1 rs' ->
    state_ok F (S t) (and_cons F st st1 (Z.of_nat n) d a rb) rs'.
Proof.
  intros t st st1 n d a rb rs rs' Hst Hrs Ha Hd Hwa Hwb Hv Hrs' Hst1. unfold and_cons.
  assert (Hw : 1 <= Z.of_nat n <= 64) by (pose proof (wd_range F E HE d); unfold wd in Hd; lia).
  assert (Hn : wdn d = n) by (unfold wd in Hd; lia).
  rewrite Hn in Hv.
  assert (Hpos : 0 < 2 ^ Z.of_nat n) by (apply Z.pow_pos_nonneg; lia).
  pose proof (Z.mod_pos_bound (rs a) _ Hpos) as Hta. pose proof (Z.mod_pos_bound (rs rb) _ Hpos) as Htb.
  fold (trunc n (rs a)) in Hta. fold (trunc n (rs rb)) in Htb.
  pose proof (land_bounds (trunc n (rs a)) (trunc n (rs rb)) ltac:(lia)) as H1.
  pose proof (land_le_r (trunc n (rs a)) (trunc n (rs rb)) ltac:(lia)) as H2.
  assert (Hval : rs' d = Z.land (trunc n (rs a)) (trunc n (rs rb)))
    by (rewrite Hv; unfold trunc at 1; apply Z.mod_small; lia).
  apply (and_cons1_sound t st _ d rb rs rs' _ Hst Hrs Ha Hw Hwb); [lia | rewrite Nat2Z.id; lia | exact Hrs' |].
  apply (and_cons1_sound t st _ d a rs rs' _ Hst Hrs Ha Hw Hwa); [lia | rewrite Nat2Z.id; lia | exact Hrs' | exact Hst1].
Qed.

Lemma mask_const_cons_sound : forall t st out n d a k rs rs',
 state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' ->
 wd F d = Z.of_nat n -> wd F a = Z.of_nat n ->
 rs' d = trunc (wdn d) (Z.land (trunc n (rs a)) (trunc n k)) ->
 regs_ok F E rs' -> state_ok F (S t) out rs' ->
 state_ok F (S t) (mask_const_cons F st out (Z.of_nat n) d a k) rs'.
Proof.
 intros t st out n d a k rs rs' Hst Hrs Ha Hd Hwa Hv Hrs' Hout.
 unfold mask_const_cons.
 destruct (negb (Z.eqb (Z.of_nat n) 64)) eqn:Ew; [exact Hout |].
 destruct (match get out d with
      | Some (FForm f _ _) => Z.eqb (fm f) 64 && null (ft f) && Z.eqb (flo f) (fhi f)
      | _ => false end) eqn:Econstant; [exact Hout |].
 apply negb_false_iff,Z.eqb_eq in Ew. assert (n=64%nat) by lia. subst n.
 destruct (contiguous (k mod 2^Z.of_nat 64)) as [[s j]|] eqn:Ec; [|exact Hout].
 destruct (cmp_range F st (OReg a) (Z.of_nat 64) false) as [[lo hi]|] eqn:Er; [|exact Hout].
 set (f:=cmp_form F st (OReg a) (Z.of_nat 64) false) in *.
 destruct ((0 <? s) && (0 <=? lo) && (hi <? 2^j) && negb (null (ft f)) &&
      negb (existsb (fun t=>Pos.eqb (fst(fst t)) d) (ft f))) eqn:Eb; [|exact Hout].
 rewrite !andb_true_iff in Eb. destruct Eb as [[[[Hs Hlo] Hhi] _] Hkeep].
 apply Z.ltb_lt in Hs,Hhi. apply Z.leb_le in Hlo. apply negb_true_iff in Hkeep.
 assert (Hoa : owidth F (OReg a) 64) by (intros r Hr; injection Hr as <-; exact Hwa).
 destruct (cmp_exact F E HE t rs Hrs st (OReg a) 64 false lo hi Hst ltac:(lia) Hoa Er) as [e [He Hf]].
 pose proof (cmp_range_sound F E HE t rs Hrs st (OReg a) 64 false lo hi Hst ltac:(lia) Hoa Er) as Hr.
 change (flo f <= e <= fhi f) in He. fold f in Hf. unfold cval,extend in Hf,Hr. cbn [opv] in Hf,Hr.
 pose proof (rs_range F E rs Hrs a) as Hra. unfold wd in Hwa.
 rewrite Hwa in Hra.
 assert (Hwd : wdn d = 64%nat) by (unfold wd in Hd; lia).
 change (tsum wdn rs (ft f)+e = trunc 64 (rs a)) in Hf.
 change (lo <= trunc 64 (rs a) <= hi) in Hr.
 rewrite (trunc_small 64 (rs a) Hra) in Hf,Hr,Hv.
 rewrite Hwd in Hv.
 destruct (contiguous_spec _ s j Ec) as [_ [Hsj [Hmask _]]].
 change (k mod 2^64 = 2^j-2^s) in Hmask.
 assert (Hland : Z.land (rs a) (trunc 64 k) = rs a - rs a mod 2^s).
 { unfold trunc. change (Z.land (rs a) (k mod 2^64)=rs a-rs a mod 2^s).
   rewrite Hmask, (land_contig_diff F E HE t st rs Hst Hrs) by lia. rewrite (Z.mod_small (rs a) (2^j)) by lia. reflexivity. }
 pose proof (land_bounds (rs a) (trunc 64 k) ltac:(lia)) as Hlb.
 rewrite (trunc_small 64 (Z.land (rs a) (trunc 64 k))) in Hv by lia.
 rewrite Hland in Hv.
 pose proof (Z.mod_pos_bound (rs a) (2^s) ltac:(apply Z.pow_pos_nonneg; lia)) as Hmod.
 apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
 unfold bound_cons. cbn [cons_ok].
 rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'),tsum_app,tsum_tscale. cbn [tsum atom_val].
 rewrite (tsum_agree wdn d rs rs' (ft f) Ha Hkeep). lia.
Qed.

(* The compared value is the exact value of its form, and lies in the
   range that the constraints give the form. *)
Lemma cmp_deep_exact : forall t st rs o w sg x y,
    state_ok F t st rs -> regs_ok F E rs -> 1 <= w <= 64 -> owidth F o w ->
    cmp_deep_range F st o w sg = Some (x, y) ->
    exists e, flo (cmp_form F st o w sg) <= e <= fhi (cmp_form F st o w sg) /\
              tsum wdn rs (ft (cmp_form F st o w sg)) + e = extend (Z.to_nat w) sg (opv rs o) /\
              x <= extend (Z.to_nat w) sg (opv rs o) <= y.
Proof.
  intros t st rs o w sg x y Hst Hrs Hw Ho H. unfold cmp_deep_range in H.
  destruct (cmp_range F st o w sg) as [[x' y'] |] eqn:Ec.
  - injection H as -> ->.
    destruct (cmp_exact F E HE t rs Hrs st o w sg x y Hst Hw Ho Ec) as [e [He Hf]].
    pose proof (cmp_range_sound F E HE t rs Hrs st o w sg x y Hst Hw Ho Ec) as Hxy.
    unfold cval in Hf, Hxy. exists e. auto.
  - destruct (check_range F st (cmp_form F st o w sg)) as [[x' y'] |] eqn:Er; [| discriminate].
    destruct ((lo_ok w sg <=? x') && (y' <=? hi_ok w sg)) eqn:Eb; [| discriminate].
    injection H as -> ->. apply andb_true_iff in Eb. destruct Eb as [E1 E2]. apply Z.leb_le in E1, E2.
    destruct (cmp_value F E HE t rs Hrs st o w sg Hst Hw Ho) as [e [He Hq]].
    pose proof (check_range_sound F E HE t rs Hrs st _ x y e Hst Er He) as Hxy.
    unfold cval in Hq.
    assert (Hv : tsum wdn rs (ft (cmp_form F st o w sg)) + e = extend (Z.to_nat w) sg (opv rs o)).
    { apply eqm_extend_eq; [lia | rewrite Z2Nat.id by lia; lia | rewrite Z2Nat.id by lia; exact Hq]. }
    exists e. split; [exact He |]. split; [exact Hv | lia].
Qed.

(* d's new value, the remainder of a nonnegative dividend by a divisor at
   least 1, lies below the exact value of the divisor's form, which reads
   the registers other than d. *)
Lemma rem_cons_sound : forall t st st1 (sg : bool) n d a b rs rs',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' ->
    wd F d = Z.of_nat n -> wd F a = Z.of_nat n -> owidth F b (Z.of_nat n) ->
    rs' d = trunc (wdn d) (match div_eval true sg n (rs a) (opv rs b) with Some v => v | None => e_arb E t d end) ->
    regs_ok F E rs' -> state_ok F (S t) st1 rs' ->
    state_ok F (S t) (rem_cons F st st1 sg (Z.of_nat n) d a b) rs'.
Proof.
  intros t st st1 sg n d a b rs rs' Hst Hrs Ha Hd Hwa Hwb Hv Hrs' Hs. unfold rem_cons.
  destruct (allocation_related F (cmp_form F st b (Z.of_nat n) sg)); cbn [negb]; [| exact Hs].
  assert (Hw : 1 <= Z.of_nat n <= 64) by (pose proof (wd_range F E HE d); unfold wd in Hd; lia).
  destruct (cmp_range F st (OReg a) (Z.of_nat n) sg) as [[x y] |] eqn:Ea; [| exact Hs].
  destruct (cmp_deep_range F st b (Z.of_nat n) sg) as [[k0 k1] |] eqn:Eb; [| exact Hs].
  assert (Hoa : owidth F (OReg a) (Z.of_nat n)) by (intros r Hr; injection Hr as <-; exact Hwa).
  pose proof (cmp_range_sound F E HE t rs Hrs st (OReg a) (Z.of_nat n) sg x y Hst Hw Hoa Ea) as Hxa.
  destruct (cmp_deep_exact t st rs b (Z.of_nat n) sg k0 k1 Hst Hrs Hw Hwb Eb) as [e [He [Hf Hxb]]].
  set (f := cmp_form F st b (Z.of_nat n) sg) in *.
  destruct (negb (0 <=? x) || negb (1 <=? k0) || null (ft f)) eqn:Ec; [exact Hs |].
  destruct (existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f)) eqn:Ed; [exact Hs |].
  rewrite !orb_false_iff in Ec. destruct Ec as [[Hx Hk] _]. apply negb_false_iff in Hx, Hk.
  apply Z.leb_le in Hx, Hk.
  unfold cval in Hxa, Hxb, Hf. rewrite Nat2Z.id in Hxa, Hxb, Hf. cbn [opv] in Hxa.
  pose proof (Z.rem_bound_pos (extend n sg (rs a)) (extend n sg (opv rs b)) ltac:(lia) ltac:(lia)) as Hr.
  (* the remainder, which the width holds *)
  assert (Hval : rs' d = Z.rem (extend n sg (rs a)) (extend n sg (opv rs b))).
  { rewrite Hv. unfold div_eval. cbv zeta.
    destruct (Z.eqb_spec (extend n sg (opv rs b)) 0) as [| _]; [lia |].
    destruct (Z.eqb_spec (extend n sg (opv rs b)) (-1)) as [| _]; [lia |].
    replace (sg && false) with false by (destruct sg; reflexivity). cbn [andb orb].
    pose proof (extend_range n sg (opv rs b) ltac:(lia)) as Hb.
    unfold hi_ok in Hb. assert (Hn : wdn d = n) by (unfold wd in Hd; lia). rewrite Hn.
    unfold trunc. apply Z.mod_small.
    destruct sg; [pose proof (pow_half (Z.of_nat n) ltac:(lia)) |]; lia. }
  apply (add_cons_ok F E HE (S t) rs' Hrs' st1 _ Hs).
  unfold bound_cons. cbn [cons_ok].
  rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app, tsum_tscale. cbn [tsum].
  rewrite (tsum_agree wdn d rs rs' (ft f) Ha Ed).
  destruct (Z.eqb (wd F d) 64); cbn [atom_val]; lia.
Qed.

Lemma project_upper_sound : forall t rs st d r,
    regs_ok F E rs -> state_ok F t st rs -> state_ok F t (project_upper F st d r) rs.
Proof.
  intros t rs st d r Hrs Hst. unfold project_upper.
  set (f := form_of_terms (sorted_terms [((d, EX), 1); ((r, EX), -1)])).
  destruct (check_range F st f) as [[lo hi] |] eqn:Er; [| exact Hst].
  destruct ((- 2 ^ 64 <=? lo) && (hi <=? 0)) eqn:Hb; [| exact Hst].
  apply andb_true_iff in Hb. destruct Hb as [Hl Hh]. apply Z.leb_le in Hl, Hh.
  pose proof (check_range_sound F E HE t rs Hrs st f lo hi 0 Hst Er ltac:(unfold f, form_of_terms; cbn; lia)) as Hr.
  assert (Hs : state_ok F t (add_cons st (ft f, - 2 ^ 64, 0)) rs).
  { apply (add_cons_ok F E HE t rs Hrs st _ Hst). cbn [cons_ok]. lia. }
  destruct (hi <? 0) eqn:Ehi; [| exact Hs]. apply Z.ltb_lt in Ehi.
  apply (add_cons_ok F E HE t rs Hrs _ _ Hs). cbn [cons_ok]. lia.
Qed.

Lemma projected_constraint_sound : forall t st rs k,
 state_ok F t st rs -> regs_ok F E rs -> state_ok F t (projected_constraint F st k) rs.
Proof.
 intros t st rs [[ts lo] hi] Hst Hrs. unfold projected_constraint.
 destruct (check_range F st (form_of_terms ts)) as [[l h] |] eqn:Er; [| exact Hst].
 destruct ((lo <=? l) && (h <=? hi)) eqn:Eb; [| exact Hst].
 apply andb_true_iff in Eb. destruct Eb as [Hl Hh]. apply Z.leb_le in Hl,Hh.
 apply (add_cons_ok F E HE t rs Hrs st _ Hst). cbn [cons_ok].
 pose proof (check_range_sound F E HE t rs Hrs st (form_of_terms ts) l h 0 Hst Er ltac:(cbn [form_of_terms flo fhi]; lia)) as Hr.
 cbn [form_of_terms ft] in Hr. lia.
Qed.

Lemma project_fixed_sound : forall t rs st d,
    regs_ok F E rs -> state_ok F t st rs -> state_ok F t (project_fixed F st d) rs.
Proof.
  intros t rs st d Hrs Hst. unfold project_fixed.
  assert (Hbase : state_ok F t
    (fold_left (fun s p => let '(r,_) := p in
       if Z.eqb (wd F r) 64 && is_fixed F r then project_upper F s d r else s)
       (lf_bounds F) st) rs).
  { generalize (lf_bounds F). intros l. revert st Hst.
    induction l as [| [r rg] l IH]; intros st Hst; cbn [fold_left]; [exact Hst |].
    apply IH. destruct (Z.eqb (wd F r) 64 && is_fixed F r); [apply project_upper_sound |]; assumption. }
  set (out := fold_left (fun s p => let '(r,_) := p in
       if Z.eqb (wd F r) 64 && is_fixed F r then project_upper F s d r else s)
       (lf_bounds F) st) in *.
  generalize (fixed_affine_targets F st). intros l. clearbody out. revert out Hbase.
  induction l as [| f l IH]; intros out Hout; cbn [fold_left]; [exact Hout |].
  apply IH. unfold project_affine_upper. apply projected_constraint_sound; assumption.
Qed.

(* x - x / p does not decrease as x grows. *)
Lemma sub_div_mono : forall p x y, 0 < p -> x <= y -> x - x / p <= y - y / p.
Proof.
  intros p x y Hp Hxy.
  pose proof (Z.div_mod x p ltac:(lia)) as Hx. pose proof (Z.div_mod y p ltac:(lia)) as Hy.
  pose proof (Z.mod_pos_bound x p Hp) as Hmx. pose proof (Z.mod_pos_bound y p Hp) as Hmy.
  assert (Hq : x / p <= y / p) by (apply Z.div_le_mono; lia).
  nia.
Qed.

Lemma shift_cons_sound : forall t st st1 (sg : bool) n d a b rs rs',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' ->
    wd F d = Z.of_nat n -> wd F a = Z.of_nat n ->
    rs' d = trunc (wdn d) (binop_eval (if sg then BShrs else BShr) n (rs a) (opv rs b)) ->
    regs_ok F E rs' -> state_ok F (S t) st1 rs' ->
    state_ok F (S t) (shift_cons F st st1 sg (Z.of_nat n) d a b) rs'.
Proof.
  intros t st st1 sg n d a b rs rs' Hst Hrs Ha Hd Hwa Hv Hrs' Hs.
  unfold shift_cons.
  destruct (negb (Z.eqb (Z.of_nat n) 64)); [exact Hs |].
  destruct (const_of F st b 32 false) as [k |] eqn:Ek; [| exact Hs].
  destruct (negb (Z.eqb (Z.min (k mod 2 ^ 32) (Z.of_nat n)) 1)); [exact Hs |].
  set (f := cmp_form F st (OReg a) (Z.of_nat n) sg) in *.
  destruct (negb (bound_related F st f)); [exact Hs |].
  destruct (cmp_range F st (OReg a) (Z.of_nat n) sg) as [[lo hi] |] eqn:Er; [| exact Hs].
  destruct ((0 <=? lo) && Z.eqb (flo f) (fhi f) &&
            negb (existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f))) eqn:Hok; [| exact Hs].
  repeat rewrite andb_true_iff in Hok. destruct Hok as [[Hlo Hpoint] Hkeep].
  apply Z.leb_le in Hlo. apply Z.eqb_eq in Hpoint. apply negb_true_iff in Hkeep.
  assert (Hw : 1 <= Z.of_nat n <= 64) by (pose proof (wd_range F E HE d); unfold wd in Hd; lia).
  assert (Hn : wdn d = n) by (unfold wd in Hd; lia).
  assert (Hoa : owidth F (OReg a) (Z.of_nat n)) by (intros r Hr; injection Hr as <-; exact Hwa).
  destruct (cmp_exact F E HE t rs Hrs st (OReg a) (Z.of_nat n) sg lo hi Hst Hw Hoa Er) as [e [He Hf]].
  pose proof (cmp_range_sound F E HE t rs Hrs st (OReg a) (Z.of_nat n) sg lo hi Hst Hw Hoa Er) as Hrange.
  fold f in He, Hf.
  assert (Heq : e = flo f) by lia. subst e.
  pose proof (const_of_sound F E HE t st rs Hst Hrs b 32 k ltac:(lia) Ek) as Hk.
  assert (Hsh : shamt n (opv rs b) = Z.min (k mod 2 ^ 32) (Z.of_nat n)).
  { unfold shamt. f_equal. destruct b as [r | z]; cbn [opv]; unfold trunc; cbn [Z.of_nat Pos.of_succ_nat].
    - rewrite Hk. reflexivity.
    - rewrite Hk. rewrite Z.mod_mod by lia. reflexivity. }
  set (power := 2 ^ Z.min (k mod 2 ^ 32) (Z.of_nat n)).
  assert (Hp : 0 < power).
  { unfold power. apply Z.pow_pos_nonneg. lia. pose proof (Z.mod_pos_bound k (2 ^ 32) ltac:(lia)). lia. }
  set (x := cval rs (OReg a) (Z.of_nat n) sg) in *.
  assert (Hx : 0 <= x < 2 ^ Z.of_nat n).
  { pose proof (cval_range F E HE t rs Hrs (OReg a) (Z.of_nat n) sg Hw) as Hb.
    fold x in Hb. unfold hi_ok in Hb. destruct sg.
    - assert (2 ^ (Z.of_nat n - 1) <= 2 ^ Z.of_nat n) by (apply Z.pow_le_mono_r; lia). lia.
    - lia. }
  assert (Hdval : rs' d = x / power).
  { assert (Hval : rs' d = trunc n (x / power)).
    { rewrite Hn in Hv. unfold x, cval, extend. cbn [opv]. rewrite Nat2Z.id.
      destruct sg; cbn [binop_eval] in Hv; rewrite Hsh in Hv; exact Hv. }
    clear Hv. rename Hval into Hv.
    rewrite Hv. apply trunc_small. split; [apply Z.div_pos; lia |].
    apply Z.div_lt_upper_bound; nia. }
  apply (project_fixed_sound (S t) rs' _ d Hrs').
  apply (add_cons_ok F E HE (S t) rs' Hrs' st1 _ Hs).
  cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app.
  cbn [tsum atom_val]. rewrite (tsum_agree wdn d rs rs' (ft f) Ha Hkeep).
  rewrite Hdval. fold power. pose proof (Z.div_mod x power ltac:(lia)) as Hdiv.
  pose proof (Z.mod_pos_bound x power Hp) as Hmod. nia.
Qed.

(* Constraints added to a state that holds, each of which holds: f k for
   each k of l that keep selects. *)
Lemma add_kept_sound : forall t rs (keep : constraint -> bool) (f : constraint -> constraint) l s,
    regs_ok F E rs -> state_ok F t s rs -> (forall k, In k l -> keep k = true -> cons_ok F rs (f k)) ->
    state_ok F t (fold_left (fun s k => if keep k then add_cons s (f k) else s) l s) rs.
Proof.
  intros t rs keep f l. induction l as [| k l IH]; intros s Hrs Hs Hk; cbn [fold_left]; [exact Hs |].
  apply IH; [exact Hrs | | intros k' Hk'; apply Hk; right; exact Hk'].
  destruct (keep k) eqn:Ek; [| exact Hs].
  apply (add_cons_ok F E HE t rs Hrs s (f k) Hs). exact (Hk k (or_introl eq_refl) Ek).
Qed.

(* Constraints added in order to a state that holds, each of which holds. *)
Lemma add_all_sound : forall t rs l s,
    regs_ok F E rs -> state_ok F t s rs -> Forall (cons_ok F rs) l -> state_ok F t (fold_left add_cons l s) rs.
Proof.
  intros t rs l. induction l as [| k l IH]; intros s Hrs Hs Hl; cbn [fold_left]; [exact Hs |].
  inversion Hl as [| ? ? Hk Hl']; subst. apply IH; [exact Hrs | | exact Hl'].
  exact (add_cons_ok F E HE t rs Hrs s k Hs Hk).
Qed.

(* The constraints f k added for each k of l that keep selects. *)
Lemma add_guarded_sound : forall t rs (keep : constraint -> bool) (f : constraint -> list constraint) l s,
    regs_ok F E rs -> state_ok F t s rs -> (forall k, In k l -> keep k = true -> Forall (cons_ok F rs) (f k)) ->
    state_ok F t (fold_left (fun s k => if keep k then fold_left add_cons (f k) s else s) l s) rs.
Proof.
  intros t rs keep f l. induction l as [| k l IH]; intros s Hrs Hs Hk; cbn [fold_left]; [exact Hs |].
  apply IH; [exact Hrs | | intros k' Hk'; apply Hk; right; exact Hk'].
  destruct (keep k) eqn:Ek; [| exact Hs].
  exact (add_all_sound t rs (f k) s Hrs Hs (Hk k (or_introl eq_refl) Ek)).
Qed.

(* Each constraint of guarded_cons holds after the load: of T under the
   guard, when the load runs, and of T's range in st, when it does not,
   since p's value, 0 or 1, says which. *)
Lemma guarded_cons_sound : forall t st rs rs' neg p k,
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> wd F p = 1 -> rs' p = rs p ->
    (holds (Some (neg, p)) rs = true -> cons_ok F rs' k) ->
    (holds (Some (neg, p)) rs = false -> tsum wdn rs' (fst (fst k)) = tsum wdn rs (fst (fst k))) ->
    Forall (cons_ok F rs') (guarded_cons F st neg p k).
Proof.
  intros t st rs rs' neg p [[ts lo] hi] Hst Hrs Hrs' Hp Hpv Hrun Hskip.
  unfold guarded_cons.
  destruct (form_range_near F st (form_of_terms ts)) as [[lo0 hi0] |] eqn:Er; [| constructor].
  pose proof (form_range_near_sound F E HE t st rs Hst Hrs (form_of_terms ts) lo0 hi0 0 Er
                ltac:(unfold form_of_terms; cbn [flo fhi]; lia)) as Hr.
  unfold form_of_terms in Hr. cbn [ft] in Hr. rewrite Z.add_0_r in Hr.
  pose proof (rs_range F E rs Hrs p) as Hrp. unfold wd in Hp. rewrite Hp in Hrp. cbn in Hrp.
  cbn [fst cons_ok holds] in Hrun, Hskip.
  assert (Hv : rs p = 0 \/ rs p = 1) by lia.
  destruct neg; repeat constructor; cbn [cons_ok];
    rewrite (tsum_sorted_terms F E HE t rs' Hrs'), tsum_app; cbn [tsum atom_val]; rewrite Hpv;
    destruct Hv as [Hv | Hv]; rewrite Hv in Hrun, Hskip |- *; cbn in Hrun, Hskip;
    first [specialize (Hrun eq_refl) | specialize (Hskip eq_refl)]; lia.
Qed.

(* A guarded load of an index input. When the guard holds, the facts of the
   load under the guard hold of the loaded value; when it fails, d keeps its
   old value, of which its old fact holds and each kept constraint, which
   follows from the state before the load; and the constraints of
   guarded_cons hold either way. *)
Lemma guarded_index_load_sound : forall t st rs rs' g gen bits d a off,
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    wf_instr F (ILoad g gen bits d a off) = true ->
    (holds g rs = true -> rs' = upd wdn rs d (trunc bits (e_mem E t (trunc 64 (rs a + off))))) ->
    (holds g rs = false -> rs' d = rs d) ->
    state_ok F (S t) (guarded_index_load F st g (Z.of_nat bits) d a off) rs'.
Proof.
  intros t st rs rs' g gen bits d a off Hst Hrs Hrs' Ha Hwf Hnew Hold.
  pose proof (state_ok_later t st rs Hst) as Hst1.
  assert (Hwf0 : wf_instr F (ILoad None gen bits d a off) = true).
  { unfold wf_instr in *. cbn [dest guard_of] in *. rewrite andb_true_iff in Hwf.
    rewrite andb_true_r. exact (proj1 Hwf). }
  unfold guarded_index_load.
  destruct (negb (index_width F (Z.of_nat bits))) eqn:Ewidth.
  { apply (transfer_dest_sound (S t) st rs rs' d g None false Hst1 Hrs Hrs' Ha);
      [discriminate | exact Hold | discriminate]. }
  destruct (refine_guard F st g) as [sg |] eqn:Er.
  2: { (* the guard never holds *)
    apply (transfer_dest_sound (S t) st rs rs' d g None false Hst1 Hrs Hrs' Ha); [discriminate | exact Hold |].
    discriminate. }
  set (st2 := index_load F sg (transfer_dest F sg d None None false) (Z.of_nat bits) d a off).
  (* when the guard holds, the load under it gives its facts *)
  assert (Hrun : holds g rs = true -> state_ok F (S t) st2 rs').
  { intros Hg. destruct (refine_guard_sound F E HE t rs Hrs st g Hst Hg) as [s [Es Hsg]].
    rewrite Er in Es. injection Es as <-.
    apply (index_load_sound F E HE t sg _ gen bits d a off rs rs' Hsg Hrs Hwf0 (Hnew Hg) Hrs').
    apply (transfer_dest_sound (S t) sg rs rs' d None None false (state_ok_later t sg rs Hsg) Hrs Hrs' Ha);
      [discriminate | discriminate | discriminate]. }
  (* when it fails, d keeps its old value *)
  assert (Hskip : holds g rs = false -> forall r, rs' r = rs r).
  { intros Hg r. destruct (Pos.eq_dec r d) as [-> | Hne]; [exact (Hold Hg) | exact (Ha r Hne)]. }
  (* the joined fact and the constraints that d's old value meets *)
  assert (Hst3 : state_ok F (S t)
                   (fold_left (fun s k => if cons_mentions d k && cons_holds F st k then add_cons s k else s)
                              (acons st2)
                              (transfer_dest F st d g (match get st2 d with
                                                       | Some (FForm f al vr) =>
                                                           if mentions f d then None else Some (FForm f al vr)
                                                       | _ => None
                                                       end) false)) rs').
  { apply (add_kept_sound (S t) rs' (fun k => cons_mentions d k && cons_holds F st k) (fun k => k));
      [exact Hrs' | |].
    - apply (transfer_dest_sound (S t) st rs rs' d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
      intros Hg x Hx. specialize (Hrun Hg).
      destruct (get st2 d) as [[f al vr | k | cs ex | cs ex] |] eqn:E2; try discriminate.
      destruct (mentions f d) eqn:Em; [discriminate |]. injection Hx as <-.
      pose proof (proj1 Hrun d _ E2) as Hf. unfold fact_ok in Hf. cbn [fact_okv] in Hf |- *.
      destruct Hf as [Hw [Hs Hr]]. split; [exact Hw |]. split; [| exact Hr].
      apply (sat_agree wdn d rs' rs f (rs' d)); [intros r Hne; symmetry; exact (Ha r Hne) | exact Em | exact Hs].
    - intros k Hk Hkeep. destruct (holds g rs) eqn:Hg.
      + exact (proj1 (Forall_forall _ _) (proj2 (Hrun eq_refl)) k Hk).
      + apply andb_true_iff in Hkeep. destruct Hkeep as [_ Hc].
        pose proof (cons_holds_sound F E HE t rs Hrs st k Hst Hc) as Hok.
        destruct k as [[ts lo] hi]. cbn [cons_ok] in *. rewrite (tsum_ext wdn rs rs' ts (Hskip eq_refl)). exact Hok. }
  destruct g as [[neg p] |]; [| exact Hst3].
  destruct (Z.eqb_spec (wd F p) 1) as [Hp1 | _]; [| exact Hst3].
  (* the constraints of the load under a guard of one bit *)
  assert (Hpd : p <> d).
  { intros ->. unfold wf_instr in Hwf0. cbn [dest guard_of] in Hwf0. rewrite !andb_true_iff in Hwf0.
    destruct Hwf0 as [[_ [[[Hbd _] Hb8] _]] _]. apply Z.leb_le in Hbd, Hb8. lia. }
  apply (add_guarded_sound (S t) rs' (fun k => cons_mentions d k && negb (cons_holds F st k))
           (guarded_cons F st neg p)); [exact Hrs' | exact Hst3 |].
  intros k Hk _. apply (guarded_cons_sound t st rs rs' neg p k Hst Hrs Hrs' Hp1 (Ha p Hpd)).
  - intros Hg. exact (proj1 (Forall_forall _ _) (proj2 (Hrun Hg)) k Hk).
  - intros Hg. exact (tsum_ext wdn rs rs' _ (Hskip Hg)).
Qed.

(* Terms with r's atoms read from r', in registers rs1, sum as the terms
   in registers rs2, when r' in rs1 holds r's value in rs2 at the same width
   and every other atom has one value in both. *)
Lemma tsum_subst : forall rs1 rs2 r r' ts,
    rs1 r' = rs2 r -> wdn r' = wdn r -> (forall x e c, In ((x, e), c) ts -> x <> r -> rs1 x = rs2 x) ->
    tsum wdn rs1 (map (fun t => let '((x, e), c) := t in if Pos.eqb x r then ((r', e), c) else t) ts) =
    tsum wdn rs2 ts.
Proof.
  intros rs1 rs2 r r' ts Hv Hw Ho. induction ts as [| [[x e] c] ts IH]; cbn [map]; [reflexivity |].
  destruct (Pos.eqb_spec x r) as [-> | Hne]; cbn [tsum];
    rewrite (IH (fun y e' c' Hin Hne => Ho y e' c' (or_intror Hin) Hne)); f_equal; f_equal; unfold atom_val.
  - rewrite Hv, Hw. reflexivity.
  - rewrite (Ho x e c (or_introl eq_refl) Hne). reflexivity.
Qed.

Lemma constant_terms_tsum : forall rs r v ts,
 rs r = v ->
 tsum wdn rs ts = tsum wdn rs (fst (constant_terms F r v ts)) + snd (constant_terms F r v ts).
Proof.
 intros rs r v ts Hv. induction ts as [|[[x e] c] rest IH]; cbn [constant_terms tsum fst snd]; [lia|].
 destruct (constant_terms F r v rest) as [tail z] eqn:Et. cbn [fst snd] in IH.
 destruct (Pos.eqb_spec x r) as [->|Hne]; cbn [tsum fst snd].
 - unfold atom_val. rewrite Hv. destruct e; cbn; lia.
 - lia.
Qed.

Lemma constant_terms_absent : forall r v ts,
 existsb (fun t => Pos.eqb (fst (fst t)) r) (fst (constant_terms F r v ts)) = false.
Proof.
 intros r v ts. induction ts as [|[[x e] c] rest IH]; cbn [constant_terms fst]; [reflexivity|].
 destruct (constant_terms F r v rest) as [tail z] eqn:Et. cbn [fst] in IH.
 destruct (Pos.eqb x r) eqn:Ex; cbn [fst existsb]; [exact IH|]. rewrite Ex,IH. reflexivity.
Qed.

(* An unguarded selp of a register of d's width and another operand: d holds
   one operand's value, so each moved constraint holds, with d in the
   register's place, of the register's value because the constraint holds in
   the state before, and of the other operand's, a register or a constant,
   because it holds there with that operand in the register's place. *)
Lemma branch_cons1_sound : forall t st other st1 w d a b rs rs',
    regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    ((state_ok F t other rs /\ rs' d = trunc (wdn d) (opv rs b)) \/
     (state_ok F t st rs /\ rs' d = trunc (wdn d) (opv rs a))) ->
    state_ok F (S t) st1 rs' -> state_ok F (S t) (branch_cons1 F st other st1 w d a b) rs'.
Proof.
  intros t st other st1 w d a b rs rs' Hrs Hrs' Ha Hv Hst1.
  (* d holds a register's value when it holds that value read at d's width *)
  assert (Hsel : forall rx, wdn d = wdn rx -> rs' d = trunc (wdn d) (rs rx) -> rs' d = rs rx).
  { intros rx Hn Hx. rewrite Hx, Hn. unfold trunc. apply Z.mod_small. exact (rs_range F E rs Hrs rx). }
  (* no atom of a constraint that does not mention d is d's *)
  assert (Hxd : forall ts lo hi r,
    (cons_mentions d (ts, lo, hi) = false \/ r = d) ->
    forall x e c, In ((x, e), c) ts -> x <> r -> x <> d).
  { intros ts lo hi r [Hnd | Hr] x e c Hx Hne; [| congruence].
    intros ->. unfold cons_mentions in Hnd. cbn [fst] in Hnd.
    assert (Hin' : existsb (fun t => Pos.eqb (fst (fst t)) d) ts = true)
      by (apply existsb_exists; exists ((d, e), c); split; [exact Hx | apply Pos.eqb_refl]).
    congruence. }
  (* d is register r: the constraint holds before *)
  assert (Hsame : forall r k, wdn d = wdn r -> rs' d = rs r -> state_ok F t st rs -> In k (acons st) ->
                  (cons_mentions d k = false \/ r = d) ->
                  cons_ok F rs' (subst_cons r d k)).
  { intros r [[ts lo] hi] Hn Hd Hst Hin Hnd. unfold subst_cons. cbn [cons_ok].
    rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
    pose proof (proj1 (Forall_forall _ _) (proj2 Hst) _ Hin) as Hok. cbn [cons_ok] in Hok.
    rewrite (tsum_subst rs' rs r d ts); [exact Hok | exact Hd | congruence |].
    intros x e c Hx Hne. exact (Ha x (Hxd ts lo hi r Hnd x e c Hx Hne)). }
  (* d is register r': the constraint with r' in r's place holds before *)
  assert (Hother : forall r r' k, wdn d = wdn r -> wdn d = wdn r' -> rs' d = rs r' ->
                   (cons_mentions d k = false \/ r = d) ->
                   state_ok F t other rs -> cons_holds F other (subst_cons r r' k) = true -> cons_ok F rs' (subst_cons r d k)).
  { intros r r' [[ts lo] hi] Hn Hn' Hd Hnd Hotherst Hb.
    pose proof (cons_holds_sound F E HE t rs Hrs other _ Hotherst Hb) as Hok.
    unfold subst_cons in Hok |- *. cbn [cons_ok] in Hok |- *.
    rewrite (tsum_sorted_terms F E HE t rs Hrs) in Hok. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
    set (rsx := fun x => if Pos.eqb x r then rs r' else rs x).
    rewrite (tsum_subst rs rsx r r' ts) in Hok;
      [| unfold rsx; rewrite Pos.eqb_refl; reflexivity | congruence
       | intros x e c _ Hne; unfold rsx; apply Pos.eqb_neq in Hne; rewrite Hne; reflexivity].
    rewrite (tsum_subst rs' rsx r d ts); [exact Hok | unfold rsx; rewrite Pos.eqb_refl; exact Hd | congruence |].
    intros x e c Hx Hne. unfold rsx. rewrite (proj2 (Pos.eqb_neq _ _) Hne).
    exact (Ha x (Hxd ts lo hi r Hnd x e c Hx Hne)). }
  (* d is the constant v: the constraint with v in r's place holds before *)
  assert (Hconst : forall r v k, wdn d = wdn r -> rs' d = v ->
                   (cons_mentions d k = false \/ r = d) ->
                   state_ok F t other rs -> cons_holds F other (const_cons F r v k) = true -> cons_ok F rs' (subst_cons r d k)).
  { intros r v [[ts lo] hi] Hn Hd Hnd Hotherst Hb.
    pose proof (cons_holds_sound F E HE t rs Hrs other _ Hotherst Hb) as Hok.
    unfold const_cons in Hok. destruct (constant_terms F r v ts) as [ts' c] eqn:Es. cbn [cons_ok] in Hok.
    unfold subst_cons. cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
    set (rsx := fun x => if Pos.eqb x r then v else rs x).
    rewrite (tsum_subst rs' rsx r d ts); [| unfold rsx; rewrite Pos.eqb_refl; exact Hd | congruence |].
    - pose proof (constant_terms_tsum rsx r v ts ltac:(unfold rsx; rewrite Pos.eqb_refl; reflexivity)) as Ht.
      rewrite Es in Ht. cbn [fst snd] in Ht.
      pose proof (constant_terms_absent r v ts) as Er. rewrite Es in Er. cbn [fst] in Er.
      assert (Hagree : agree r rs rsx) by (intros x Hne; unfold rsx; apply Pos.eqb_neq in Hne; rewrite Hne; reflexivity).
      rewrite (tsum_agree wdn r rs rsx ts' Hagree Er) in Ht. lia.
    - intros x e c0 Hx Hne. unfold rsx. rewrite (proj2 (Pos.eqb_neq _ _) Hne).
      exact (Ha x (Hxd ts lo hi r Hnd x e c0 Hx Hne)). }
  rewrite branch_cons1_exact. unfold branch_cons1_spec. cbv zeta.
  destruct a as [ra | za]; destruct b as [rb | zb].
  - (* two registers *)
    destruct (Z.eqb (wd F rb) w) eqn:Ewb; [| exact Hst1].
    destruct (Z.eqb (wd F ra) w && Z.eqb (wd F d) w) eqn:Ew; [| exact Hst1].
    apply andb_true_iff in Ew. destruct Ew as [Ewa Ewd]. apply Z.eqb_eq in Ewa, Ewb, Ewd.
    assert (Hna : wdn d = wdn ra) by (unfold wd in Ewa, Ewd; lia).
    assert (Hnb : wdn d = wdn rb) by (unfold wd in Ewb, Ewd; lia).
    apply (add_kept_sound (S t) rs'
             (fun k => cons_mentions ra k && (negb (cons_mentions d k) || Pos.eqb ra d) && cons_holds F other (subst_cons ra rb k))
             (subst_cons ra d)); [exact Hrs' | exact Hst1 |].
    intros k Hin Hk. rewrite !andb_true_iff in Hk. destruct Hk as [[_ Hnd] Hb].
    rewrite orb_true_iff, negb_true_iff, Pos.eqb_eq in Hnd.
    destruct Hv as [[Hotherst Hv] | [Hst Hv]]; cbn [opv] in Hv.
    + exact (Hother ra rb k Hna Hnb (Hsel rb Hnb Hv) Hnd Hotherst Hb).
    + exact (Hsame ra k Hna (Hsel ra Hna Hv) Hst Hin Hnd).
  - (* a register, then a constant *)
    destruct (Z.eqb (wd F ra) w && Z.eqb (wd F d) w) eqn:Ew; [| exact Hst1].
    apply andb_true_iff in Ew. destruct Ew as [Ewa Ewd]. apply Z.eqb_eq in Ewa, Ewd.
    assert (Hna : wdn d = wdn ra) by (unfold wd in Ewa, Ewd; lia).
    apply (add_kept_sound (S t) rs'
             (fun k => cons_mentions ra k && (negb (cons_mentions d k) || Pos.eqb ra d) &&
                       cons_holds F other (const_cons F ra (zb mod 2 ^ w) k))
             (subst_cons ra d)); [exact Hrs' | exact Hst1 |].
    intros k Hin Hk. rewrite !andb_true_iff in Hk. destruct Hk as [[_ Hnd] Hb].
    rewrite orb_true_iff, negb_true_iff, Pos.eqb_eq in Hnd.
    destruct Hv as [[Hotherst Hv] | [Hst Hv]]; cbn [opv] in Hv.
    + apply (Hconst ra (zb mod 2 ^ w) k Hna); [| exact Hnd | exact Hotherst | exact Hb].
      rewrite Hv. unfold trunc. unfold wd in Ewd. rewrite Ewd. reflexivity.
    + exact (Hsame ra k Hna (Hsel ra Hna Hv) Hst Hin Hnd).
  - exact Hst1.
  - exact Hst1.
Qed.


Lemma sel_cons1_sound : forall t st st1 w d a b rs rs',
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    (rs' d = trunc (wdn d) (opv rs b) \/ rs' d = trunc (wdn d) (opv rs a)) ->
    state_ok F (S t) st1 rs' -> state_ok F (S t) (sel_cons1 F st st1 w d a b) rs'.
Proof.
  intros t st st1 w d a b rs rs' Hst Hrs Hrs' Ha Hv Hst1.
  unfold sel_cons1. destruct a, b;
    apply (branch_cons1_sound t st st st1 w d _ _ rs rs' Hrs Hrs' Ha); tauto.
Qed.

Lemma sel_cons_sound : forall t st st1 w d a b p rs rs',
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    rs' d = trunc (wdn d) (if Z.eqb (rs p) 0 then opv rs b else opv rs a) ->
    state_ok F (S t) st1 rs' -> state_ok F (S t) (sel_cons F st st1 w d a b) rs'.
Proof.
  intros t st st1 w d a b p rs rs' Hst Hrs Hrs' Ha Hv Hst1.
  assert (Hcases : rs' d = trunc (wdn d) (opv rs b) \/ rs' d = trunc (wdn d) (opv rs a)).
  { destruct (Z.eqb (rs p) 0); auto. }
  pose proof (sel_cons1_sound t st st1 w d a b rs rs' Hst Hrs Hrs' Ha Hcases Hst1) as Hfirst.
  rewrite sel_cons_exact. unfold sel_cons_spec. destruct a as [ra | za], b as [rb | zb]; try exact Hfirst.
  destruct (Pos.eqb ra rb); [exact Hfirst |].
  apply (sel_cons1_sound t st _ w d (OReg rb) (OReg ra) rs rs' Hst Hrs Hrs' Ha); [tauto | exact Hfirst].
Qed.


Lemma copy_operand_value : forall t st rs o,
    state_ok F t st rs -> regs_ok F E rs -> opv rs (copy_operand F st o) = opv rs o.
Proof.
  intros t st rs [r | z] Hst Hrs; [| reflexivity].
  assert (Hcopy : forall n x, atom_val wdn rs (copy_atom_n F n st x) = atom_val wdn rs x).
  { intros n. induction n as [| n IH]; intros x; [reflexivity |].
    cbn [copy_atom_n]. rewrite IH. exact (copy_atom_val F E HE t rs Hrs st x Hst). }
  pose proof (Hcopy 4%nat (r, EX)) as Hv.
  unfold copy_operand. destruct (copy_atom_n F 4 st (r, EX)) as [r' e].
  destruct e; try reflexivity; destruct (Z.eqb (wd F r') (wd F r));
    cbn [opv atom_val] in *; congruence.
Qed.

Lemma branch_cons_sound : forall t st other st1 w d a b rs rs',
    regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    ((state_ok F t other rs /\ rs' d = trunc (wdn d) (opv rs b)) \/
     (state_ok F t st rs /\ rs' d = trunc (wdn d) (opv rs a))) ->
    state_ok F (S t) st1 rs' -> state_ok F (S t) (branch_cons F st other st1 w d a b) rs'.
Proof.
  intros t st other st1 w d a b rs rs' Hrs Hrs' Ha Hcases Hst1.
  pose proof (branch_cons1_sound t st other st1 w d a b rs rs' Hrs Hrs' Ha Hcases Hst1) as Hbase.
  rewrite branch_cons_exact. unfold branch_cons_spec. cbv zeta.
  destruct (_ && _); [exact Hbase |].
  apply (branch_cons1_sound t st other _ w d _ _ rs rs' Hrs Hrs' Ha); [| exact Hbase].
  destruct Hcases as [[Hs Hv] | [Hs Hv]].
  - left. split; [exact Hs |]. rewrite (copy_operand_value t other rs b Hs Hrs). exact Hv.
  - right. split; [exact Hs |]. rewrite (copy_operand_value t st rs a Hs Hrs). exact Hv.
Qed.

Lemma type_operand_sound : forall t st rs o sg,
 state_ok F t st rs -> regs_ok F E rs -> state_ok F t (type_operand F st o sg) rs.
Proof.
 intros t st rs [r | z] sg Hst Hrs; [| exact Hst]. unfold type_operand.
 destruct (negb (Z.eqb (wd F r) 64)) eqn:Ew; [exact Hst |].
 destruct (get st r); [exact Hst |].
 apply (put_ok F E HE t rs Hrs st r _ Hst).
 cbn [fact_ok fact_okv]. split; [unfold wf; cbn [range_form fm]; lia |].
 split.
 - exists (extend 64 sg (rs r)). split.
   + exact (extend_range 64 sg (rs r) ltac:(lia)).
   + cbn [range_form ft fm tsum]. rewrite Z.add_0_l. apply eqm_sym. exact (extend_eqm 64 sg (rs r)).
 - split; [apply aligned_nonpos; lia | discriminate].
Qed.

Lemma type_comparisons_sound : forall t st rs p,
 state_ok F t st rs -> regs_ok F E rs -> state_ok F t (type_comparisons F st p) rs.
Proof.
 intros t st rs p Hst Hrs. unfold type_comparisons.
 generalize (match get st p with Some (FPred cs _) | Some (FNeg cs _) => cs | _ => [] end).
 intros cs. revert st Hst. induction cs as [| k cs IH]; intros st Hst; cbn [fold_left]; [exact Hst |].
 apply IH. destruct (Nat.eqb (cw k) 64); [apply type_operand_sound; [apply type_operand_sound |] |]; assumption.
Qed.

Lemma refine_selection_sound : forall t st rs p truth,
 state_ok F t st rs -> regs_ok F E rs -> negb (Z.eqb (rs p) 0) = truth ->
 exists out, refine_selection F st p truth = Some out /\ state_ok F t out rs.
Proof.
 intros t st rs p truth Hst Hrs Hp. unfold refine_selection.
 destruct (null (guard_origins F st p)); [apply (refine_sound F E HE t rs Hrs st p truth Hst Hp) |].
 pose proof (type_comparisons_sound t st rs p Hst Hrs) as Htyped.
 destruct (refine_sound F E HE t rs Hrs (type_comparisons F st p) p truth Htyped Hp) as [s [Es Hs]].
 rewrite Es. exact (refine_origins_sound F E HE t rs Hrs s (guard_origins F st p) Hs).
Qed.

Lemma selection_operand_sound : forall t st rs d o,
 state_ok F t st rs -> regs_ok F E rs -> opv rs (selection_operand F st d o) = opv rs o.
Proof.
 intros t st rs d [r|z] Hst Hrs; unfold selection_operand; [|reflexivity].
 destruct (comparison_alias F st r d) eqn:He; [|reflexivity].
 exact (comparison_alias_sound t st rs r d Hst Hrs He).
Qed.

Lemma selection_comparison_raw_sound : forall t st rs d k,
 state_ok F t st rs -> regs_ok F E rs ->
 (cmp_holds rs (selection_comparison_raw F st d k) <-> cmp_holds rs k).
Proof.
 intros t st rs d k Hst Hrs. unfold selection_comparison_raw,cmp_holds.
 cbn [cc cw csg ca cb]. rewrite (selection_operand_sound t st rs d (cb k) Hst Hrs).
 destruct (comparison_alias F st (ca k) d) eqn:He; [|reflexivity].
 pose proof (comparison_alias_sound t st rs (ca k) d Hst Hrs He) as Hv.
 rewrite Hv. reflexivity.
Qed.

Lemma cmp_eval_reverse : forall c x y,
 cmp_eval (reverse_comparison c) y x = cmp_eval c x y.
Proof.
 intros c x y. destruct c; cbn [reverse_comparison cmp_eval]; apply Bool.eq_iff_eq_true;
 rewrite ?Z.ltb_lt, ?Z.leb_le, ?Z.eqb_eq, ?negb_true_iff, ?Z.eqb_neq; lia.
Qed.

Lemma selection_comparison_sound : forall t st rs d k,
 state_ok F t st rs -> regs_ok F E rs ->
 (cmp_holds rs (selection_comparison F st d k) <-> cmp_holds rs k).
Proof.
 intros t st rs d k Hst Hrs. unfold selection_comparison.
 rewrite <- (selection_comparison_raw_sound t st rs d k Hst Hrs).
 remember (selection_comparison_raw F st d k) as k'. clear Heqk'.
 destruct (cb k') as [r|z] eqn:Eb; [|reflexivity].
 destruct (Pos.eqb r d && negb (Pos.eqb (ca k') d)); [|reflexivity].
 unfold cmp_holds. cbn [cc cw csg ca cb opv]. rewrite Eb. cbn [opv].
 rewrite cmp_eval_reverse. reflexivity.
Qed.

Lemma selection_predicate_sound : forall t st rs d p,
 state_ok F t st rs -> regs_ok F E rs -> state_ok F t (selection_predicate F st d p) rs.
Proof.
 intros t st rs d p Hst Hrs.
 assert (Hcs : forall cs, Forall (cmp_holds rs) (map (selection_comparison F st d) cs) <-> Forall (cmp_holds rs) cs).
 { induction cs as [|k cs IH]; cbn [map]; [reflexivity|].
   rewrite !Forall_cons_iff, IH, (selection_comparison_sound t st rs d k Hst Hrs). reflexivity. }
 unfold selection_predicate.
 destruct (get st p) as [[f al vr|al|cs exact|cs exact]|] eqn:Ep; try exact Hst;
 apply (put_ok F E HE t rs Hrs st p); [exact Hst| |exact Hst|];
 pose proof (proj1 Hst p _ Ep) as Hp; cbn [fact_ok fact_okv] in *; rewrite Hcs; exact Hp.
Qed.

Lemma signed_selection_bound_sound : forall t other out rs rs' w d r z,
 regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 ((state_ok F t other rs /\ rs' d = trunc (wdn d) z) \/
  rs' d = trunc (wdn d) (rs r)) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (signed_selection_bound F other out w d r z) rs'.
Proof.
 intros t other out rs rs' w d r z Hrs Hrs' Ha Hcases Hout.
 unfold signed_selection_bound.
 destruct (negb ((0 <? w) && Z.eqb w (wd F d) && Z.eqb w (wd F r)) || Pos.eqb r d) eqn:Ev; [exact Hout |].
 apply orb_false_iff in Ev. destruct Ev as [Ew Hne].
 apply negb_false_iff in Ew. repeat rewrite andb_true_iff in Ew.
 destruct Ew as [[Hw Hd] Hr]. apply Z.ltb_lt in Hw. apply Z.eqb_eq in Hd,Hr. apply Pos.eqb_neq in Hne.
 assert (Hdn : wdn d = Z.to_nat w) by (unfold wd in Hd; lia).
 assert (Hwidth : 1 <= w <= 64) by (pose proof (wd_range F E HE d); unfold wd in Hd; lia).
 destruct (cmp_range F other (OReg r) w true) as [[lo hi] |] eqn:Er; [|exact Hout].
 assert (Hread : cval rs' (OReg r) w true = cval rs (OReg r) w true).
 { unfold cval. cbn [opv]. rewrite (Ha r Hne). reflexivity. }
 assert (Hbounds :
   (cval rs' (OReg d) w true = extend (Z.to_nat w) true z /\ lo <= cval rs (OReg r) w true <= hi) \/
   cval rs' (OReg d) w true = cval rs (OReg r) w true).
 { destruct Hcases as [[Hother Hv] | Hv].
   - left. split.
     + unfold cval. cbn [opv]. rewrite Hv,Hdn,extend_trunc. reflexivity.
     + apply (cmp_range_sound F E HE t rs Hrs other (OReg r) w true lo hi Hother); [exact Hwidth | intros u Hu; injection Hu as <-; lia | exact Er].
   - right. unfold cval. cbn [opv]. rewrite Hv,Hdn,extend_trunc. reflexivity. }
 assert (Hle : extend (Z.to_nat w) true z <= lo -> cmp_holds rs' {|cc:=CLe;cw:=Z.to_nat w;csg:=true;ca:=d;cb:=OReg r|}).
 { intros H. unfold cmp_holds. cbn [cc cw csg ca cb cmp_eval]. apply Z.leb_le. change (cval rs' (OReg d) w true <= cval rs' (OReg r) w true). rewrite Hread. destruct Hbounds as [[He Hb]|He]; rewrite He; lia. }
 assert (Hge : hi <= extend (Z.to_nat w) true z -> cmp_holds rs' {|cc:=CGe;cw:=Z.to_nat w;csg:=true;ca:=d;cb:=OReg r|}).
 { intros H. unfold cmp_holds. cbn [cc cw csg ca cb cmp_eval]. apply Z.leb_le. change (cval rs' (OReg r) w true <= cval rs' (OReg d) w true). rewrite Hread. destruct Hbounds as [[He Hb]|He]; rewrite He; lia. }
 cbv zeta. destruct (extend (Z.to_nat w) true z <=? lo) eqn:El.
 - apply Z.leb_le in El.
   destruct (refine_cmp_sound F E HE (S t) rs' Hrs' out _ Hout (Hle El)) as [s [Es Hs]]. rewrite Es. exact Hs.
 - destruct (hi <=? extend (Z.to_nat w) true z) eqn:Eh; [|exact Hout].
   apply Z.leb_le in Eh.
   destruct (refine_cmp_sound F E HE (S t) rs' Hrs' out _ Hout (Hge Eh)) as [s [Es Hs]]. rewrite Es. exact Hs.
Qed.

Lemma sel_refined_cons_sound : forall t st st1 w d a b p rs rs',
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
    rs' d = trunc (wdn d) (if Z.eqb (rs p) 0 then opv rs b else opv rs a) ->
    state_ok F (S t) st1 rs' -> state_ok F (S t) (sel_refined_cons F st st1 w d a b p) rs'.
Proof.
  intros t st st1 w d a b p rs rs' Hst Hrs Hrs' Ha Hv Hst1.
  rewrite sel_refined_cons_same. unfold sel_refined_cons_raw,sel_refined_cons_with.
  pose proof (selection_predicate_sound t st rs (selection_register d a b) p Hst Hrs) as Hselected.
  remember (selection_predicate F st (selection_register d a b) p) as selected. clear Heqselected Hst.
  rename st into prior. rename selected into st. rename Hselected into Hst.
  pose proof (sel_cons_sound t st st1 w d a b p rs rs' Hst Hrs Hrs' Ha Hv Hst1) as Hbase.
  destruct (refine_selection F st p true) as [sa |] eqn:Ea; [| exact Hbase].
  destruct (refine_selection F st p false) as [sb |] eqn:Eb; [| exact Hbase].
  assert (Hcases : (state_ok F t sb rs /\ rs' d = trunc (wdn d) (opv rs b)) \/
                   (state_ok F t sa rs /\ rs' d = trunc (wdn d) (opv rs a))).
  { destruct (Z.eqb (rs p) 0) eqn:Ep.
    - left. split; [| exact Hv].
      destruct (refine_selection_sound t st rs p false Hst Hrs ltac:(rewrite Ep; reflexivity)) as [s [Es Hok]].
      rewrite Eb in Es. injection Es as <-. exact Hok.
    - right. split; [| exact Hv].
      destruct (refine_selection_sound t st rs p true Hst Hrs ltac:(rewrite Ep; reflexivity)) as [s [Es Hok]].
      rewrite Ea in Es. injection Es as <-. exact Hok. }
  pose proof (branch_cons_sound t sa sb _ w d a b rs rs' Hrs Hrs' Ha Hcases Hbase) as Hfirst.
  pose proof (branch_cons_sound t sb sa _ w d b a rs rs' Hrs Hrs' Ha ltac:(tauto) Hfirst) as Hboth.
  destruct a as [ra|za], b as [rb|zb]; cbn; try exact Hboth.
  - apply (signed_selection_bound_sound t sb _ rs rs' w d ra zb Hrs Hrs' Ha); [|exact Hboth].
    destruct Hcases as [[Hs Hv']|[Hs Hv']]; [left|right]; auto.
  - apply (signed_selection_bound_sound t sa _ rs rs' w d rb za Hrs Hrs' Ha); [|exact Hboth].
    destruct Hcases as [[Hs Hv']|[Hs Hv']]; [right|left]; auto.
Qed.

Lemma minmax_cons_sound : forall t st rs rs' (mn sg : bool) w d a b,
    regs_ok F E rs' -> agree d rs rs' -> wd F d = Z.of_nat w ->
    rs' d = trunc (wdn d) (if mn then Z.min (extend w sg (rs a)) (extend w sg (opv rs b))
                          else Z.max (extend w sg (rs a)) (extend w sg (opv rs b))) ->
    state_ok F t st rs' -> state_ok F t (minmax_cons F st mn sg w d a b) rs'.
Proof.
  intros t st rs rs' mn sg w d a b Hrs Ha Hwd Hv Hst.
  assert (Hn : wdn d = w) by (unfold wd in Hwd; lia).
  assert (Hw : (1 <= w)%nat) by (pose proof (wd_range F E HE d); lia).
  set (x := extend w sg (rs a)) in *.
  set (y := extend w sg (opv rs b)) in *.
  set (m := if mn then Z.min x y else Z.max x y) in *.
  assert (Hd : extend w sg (rs' d) = m).
  { rewrite Hv, Hn, extend_trunc. symmetry.
    apply eqm_extend_eq; [exact Hw | | apply eqm_sym; apply extend_eqm].
    pose proof (extend_range w sg (rs a) Hw).
    pose proof (extend_range w sg (opv rs b) Hw).
    unfold m, x, y. destruct mn; lia. }
  assert (HA : cmp_holds rs' {| cc := if mn then CLe else CGe; cw := w; csg := sg; ca := d; cb := OReg a |}).
  { unfold cmp_holds. cbn [cc cw csg ca cb opv].
    destruct (Pos.eq_dec a d) as [-> | Hne].
    - destruct mn; cbn [cmp_eval]; apply Z.leb_refl.
    - rewrite (Ha a Hne), Hd. fold x. unfold m. destruct mn; cbn [cmp_eval]; apply Z.leb_le; lia. }
  assert (HB : cmp_holds rs' {| cc := if mn then CLe else CGe; cw := w; csg := sg; ca := d; cb := b |}).
  { unfold cmp_holds. cbn [cc cw csg ca cb].
    assert (HB0 : opv rs' b = opv rs b \/ opv rs' b = rs' d).
    { destruct b as [r | z]; cbn [opv].
      - destruct (Pos.eq_dec r d) as [-> | Hne]; [right; reflexivity | left; apply Ha; exact Hne].
      - left; reflexivity. }
    destruct HB0 as [-> | ->].
    - rewrite Hd. fold y. unfold m. destruct mn; cbn [cmp_eval]; apply Z.leb_le; lia.
    - destruct mn; cbn [cmp_eval]; apply Z.leb_refl. }
  unfold minmax_cons.
  destruct (refine_cmp_sound F E HE t rs' Hrs st _ Hst HA) as [s [Hs Hok]]. rewrite Hs.
  destruct (refine_cmp_sound F E HE t rs' Hrs s _ Hok HB) as [s' [Hs' Hok']]. rewrite Hs'. exact Hok'.
Qed.

Lemma minmax_common_cons_sound : forall t st out rs rs' (mn sg : bool) w d a b,
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' ->
    agree d rs rs' -> wd F d = Z.of_nat w -> wd F a = Z.of_nat w -> owidth F b (Z.of_nat w) ->
    rs' d = trunc (wdn d) (if mn then Z.min (extend w sg (rs a)) (extend w sg (opv rs b))
                          else Z.max (extend w sg (rs a)) (extend w sg (opv rs b))) ->
    state_ok F (S t) out rs' ->
    state_ok F (S t) (minmax_common_cons F st out mn sg w d a b) rs'.
Proof.
  intros t st out rs rs' mn sg w d a b Hst Hrs Hrs' Ha Hd Hwa Hwb Hv Hout.
  unfold minmax_common_cons.
  destruct (negb (Nat.eqb w 64)) eqn:Ew; [exact Hout |].
  apply negb_false_iff in Ew. apply Nat.eqb_eq in Ew. subst w.
  destruct (cmp_range F st (OReg a) 64 sg) as [[alo ahi] |] eqn:Ea; [| exact Hout].
  destruct (cmp_range F st b 64 sg) as [[blo bhi] |] eqn:Eb; [| exact Hout].
  destruct (negb (0 <=? (if mn then Z.min alo blo else Z.max alo blo))) eqn:En; [exact Hout |].
  apply negb_false_iff in En. apply Z.leb_le in En.
  assert (Hoa : owidth F (OReg a) 64) by (intros r Hr; injection Hr as <-; exact Hwa).
  destruct (cmp_exact F E HE t rs Hrs st (OReg a) 64 sg alo ahi Hst ltac:(lia) Hoa Ea) as [ea [Hea Hfa]].
  destruct (cmp_exact F E HE t rs Hrs st b 64 sg blo bhi Hst ltac:(lia) Hwb Eb) as [eb [Heb Hfb]].
  pose proof (cmp_range_sound F E HE t rs Hrs st (OReg a) 64 sg alo ahi Hst ltac:(lia) Hoa Ea) as Hra.
  pose proof (cmp_range_sound F E HE t rs Hrs st b 64 sg blo bhi Hst ltac:(lia) Hwb Eb) as Hrb.
  set (fa := cmp_form F st (OReg a) 64 sg) in *.
  set (fb := cmp_form F st b 64 sg) in *.
  set (va := cval rs (OReg a) 64 sg) in *.
  set (vb := cval rs b 64 sg) in *.
  assert (Hn : wdn d = 64%nat) by (unfold wd in Hd; lia).
  assert (Hresult : rs' d = if mn then Z.min va vb else Z.max va vb).
  { rewrite Hv, Hn. change (trunc 64 (if mn then Z.min va vb else Z.max va vb) = (if mn then Z.min va vb else Z.max va vb)).
    apply trunc_small.
    pose proof (extend_range 64 sg (rs a) ltac:(lia)) as Hxa.
    pose proof (extend_range 64 sg (opv rs b) ltac:(lia)) as Hxb.
    change (lo_ok 64 sg <= va <= hi_ok 64 sg) in Hxa.
    change (lo_ok 64 sg <= vb <= hi_ok 64 sg) in Hxb.
    unfold hi_ok, lo_ok in Hxa, Hxb. destruct mn, sg; cbn in *; lia. }
  cbv zeta.
  lazymatch goal with |- state_ok _ _ (fold_left _ ?targets _) _ => generalize targets end.
  intros l. revert out Hout.
  induction l as [| target l IH]; intros out Hout; cbn [fold_left]; [exact Hout |].
  apply IH.
  destruct (negb (forallb (fun t => negb (Pos.eqb (fst (fst t)) d)) target)) eqn:Ek; [exact Hout |].
  apply negb_false_iff in Ek.
  assert (Hkeep : existsb (fun t => Pos.eqb (fst (fst t)) d) target = false).
  { apply Bool.not_true_iff_false. intros Hx. apply existsb_exists in Hx.
    destruct Hx as [term [Hin Hx]].
    pose proof (proj1 (forallb_forall _ _) Ek term Hin) as Hterm.
    change (negb (Pos.eqb (fst (fst term)) d) = true) in Hterm.
    rewrite Hx in Hterm. discriminate. }
  set (da := {| fm := 64; ft := sorted_terms (ft fa ++ tscale (-1) target); flo := flo fa; fhi := fhi fa |}).
  set (db := {| fm := 64; ft := sorted_terms (ft fb ++ tscale (-1) target); flo := flo fb; fhi := fhi fb |}).
  destruct (check_range F st da) as [[lo hi] |] eqn:Eda; [| exact Hout].
  destruct (check_range F st db) as [[lo' hi'] |] eqn:Edb; [| exact Hout].
  pose proof (check_range_sound F E HE t rs Hrs st da lo hi ea Hst Eda Hea) as Hda.
  pose proof (check_range_sound F E HE t rs Hrs st db lo' hi' eb Hst Edb Heb) as Hdb.
  unfold da in Hda. unfold db in Hdb. cbn [ft] in Hda, Hdb.
  rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app, tsum_tscale in Hda, Hdb.
  destruct (if mn then 0 <=? Z.min lo lo' else Z.max hi hi' <=? 0); [| exact Hout].
  apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
  cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app, tsum_tscale.
  cbn [tsum atom_val]. rewrite (tsum_agree wdn d rs rs' target Ha Hkeep), Hresult.
  destruct mn; nia.
Qed.

Lemma minmax_relative_cons_sound : forall t st out rs rs' (mn sg : bool) w d a b,
    state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' ->
    agree d rs rs' -> wd F d = Z.of_nat w -> wd F a = Z.of_nat w -> owidth F b (Z.of_nat w) ->
    rs' d = trunc (wdn d) (if mn then Z.min (extend w sg (rs a)) (extend w sg (opv rs b))
                          else Z.max (extend w sg (rs a)) (extend w sg (opv rs b))) ->
    state_ok F (S t) out rs' ->
    state_ok F (S t) (minmax_relative_cons F st out mn sg w d a b) rs'.
Proof.
  intros t st out rs rs' mn sg w d a b Hst Hrs Hrs' Ha Hd Hwa Hwb Hv Hout.
  unfold minmax_relative_cons.
  destruct (negb (Nat.eqb w 64)) eqn:Ew; [exact Hout |].
  apply negb_false_iff in Ew. apply Nat.eqb_eq in Ew. subst w.
  destruct (cmp_range F st (OReg a) 64 sg) as [[alo ahi] |] eqn:Ea; [| exact Hout].
  destruct (cmp_range F st b 64 sg) as [[blo bhi] |] eqn:Eb; [| exact Hout].
  destruct (negb (0 <=? (if mn then Z.min alo blo else Z.max alo blo))) eqn:En; [exact Hout |].
  apply negb_false_iff in En. apply Z.leb_le in En.
  assert (Hoa : owidth F (OReg a) 64) by (intros r Hr; injection Hr as <-; exact Hwa).
  destruct (cmp_exact F E HE t rs Hrs st (OReg a) 64 sg alo ahi Hst ltac:(lia) Hoa Ea) as [ea [Hea Hfa]].
  destruct (cmp_exact F E HE t rs Hrs st b 64 sg blo bhi Hst ltac:(lia) Hwb Eb) as [eb [Heb Hfb]].
  pose proof (cmp_range_sound F E HE t rs Hrs st (OReg a) 64 sg alo ahi Hst ltac:(lia) Hoa Ea) as Hra.
  pose proof (cmp_range_sound F E HE t rs Hrs st b 64 sg blo bhi Hst ltac:(lia) Hwb Eb) as Hrb.
  set (fa := cmp_form F st (OReg a) 64 sg) in *.
  set (fb := cmp_form F st b 64 sg) in *.
  set (va := cval rs (OReg a) 64 sg) in *.
  set (vb := cval rs b 64 sg) in *.
  assert (Hn : wdn d = 64%nat) by (unfold wd in Hd; lia).
  assert (Hresult : rs' d = if mn then Z.min va vb else Z.max va vb).
  { rewrite Hv, Hn. change (trunc 64 (if mn then Z.min va vb else Z.max va vb) = (if mn then Z.min va vb else Z.max va vb)).
    apply trunc_small.
    pose proof (extend_range 64 sg (rs a) ltac:(lia)) as Hxa.
    pose proof (extend_range 64 sg (opv rs b) ltac:(lia)) as Hxb.
    change (lo_ok 64 sg <= va <= hi_ok 64 sg) in Hxa.
    change (lo_ok 64 sg <= vb <= hi_ok 64 sg) in Hxb.
    unfold hi_ok, lo_ok in Hxa, Hxb. destruct mn, sg; cbn in *; lia. }
  cbv zeta.
  destruct (negb (null (ft fa) || null (ft fb))).
  { eapply minmax_common_cons_sound; eauto. }
  lazymatch goal with |- state_ok _ _ (fold_left _ ?targets _) _ => generalize targets end.
  intros l. revert out Hout.
  induction l as [| target l IH]; intros out Hout; cbn [fold_left]; [exact Hout |].
  apply IH.
  destruct (negb (forallb (fun t => is_fixed F (fst (fst t)) && negb (Pos.eqb (fst (fst t)) d)) target)) eqn:Ek; [exact Hout |].
  apply negb_false_iff in Ek.
  assert (Hkeep : existsb (fun t => Pos.eqb (fst (fst t)) d) target = false).
  { apply Bool.not_true_iff_false. intros Hx. apply existsb_exists in Hx.
    destruct Hx as [term [Hin Hx]].
    pose proof (proj1 (forallb_forall _ _) Ek term Hin) as Hterm.
    change (is_fixed F (fst (fst term)) && negb (Pos.eqb (fst (fst term)) d) = true) in Hterm.
    apply andb_true_iff in Hterm. destruct Hterm as [_ Hnot].
    rewrite Hx in Hnot. discriminate. }
  set (da := {| fm := 64; ft := sorted_terms (ft fa ++ tscale (-1) target); flo := flo fa; fhi := fhi fa |}).
  set (db := {| fm := 64; ft := sorted_terms (ft fb ++ tscale (-1) target); flo := flo fb; fhi := fhi fb |}).
  destruct (form_range F st da) as [[lo hi] |] eqn:Eda; [| exact Hout].
  destruct (form_range F st db) as [[lo' hi'] |] eqn:Edb; [| exact Hout].
  pose proof (form_range_sound F E HE t st rs Hst Hrs da lo hi ea Eda Hea) as Hda.
  pose proof (form_range_sound F E HE t st rs Hst Hrs db lo' hi' eb Edb Heb) as Hdb.
  unfold da in Hda. unfold db in Hdb. cbn [ft] in Hda, Hdb.
  rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app, tsum_tscale in Hda, Hdb.
  apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
  cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app, tsum_tscale.
  cbn [tsum atom_val]. rewrite (tsum_agree wdn d rs rs' target Ha Hkeep), Hresult.
  destruct mn; nia.
Qed.

Lemma sext_range : forall n v, (1 <= n)%nat -> - 2 ^ (Z.of_nat n - 1) <= sext n v < 2 ^ (Z.of_nat n - 1).
Proof.
  intros n v Hn. unfold sext. pose proof (trunc_range n v) as Ht.
  assert (Hp : 2 ^ Z.of_nat n = 2 * 2 ^ (Z.of_nat n - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
  destruct (Z.ltb_spec (trunc n v) (2 ^ (Z.of_nat n - 1))); lia.
Qed.

Lemma sext_trunc : forall n v, sext n (trunc n v) = sext n v.
Proof. intros n v. unfold sext, trunc. rewrite Z.mod_mod by (apply Z.pow_nonzero; lia). reflexivity. Qed.

(* A value within the signed window of n bits is its own sign extension. *)
Lemma sext_id : forall n x, (1 <= n)%nat -> - 2 ^ (Z.of_nat n - 1) <= x < 2 ^ (Z.of_nat n - 1) -> sext n x = x.
Proof.
  intros n x Hn Hx. unfold sext, trunc.
  assert (Hp : 2 ^ Z.of_nat n = 2 * 2 ^ (Z.of_nat n - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
  assert (Hpos : 0 < 2 ^ (Z.of_nat n - 1)) by (apply Z.pow_pos_nonneg; lia).
  destruct (Z.le_gt_cases 0 x) as [H0 | H0].
  - rewrite Z.mod_small by lia. destruct (Z.ltb_spec x (2 ^ (Z.of_nat n - 1))); lia.
  - assert (Hm : x mod 2 ^ Z.of_nat n = x + 2 ^ Z.of_nat n).
    { symmetry. apply (Z.mod_unique x (2 ^ Z.of_nat n) (-1) (x + 2 ^ Z.of_nat n)); [left; lia | ring]. }
    rewrite Hm. destruct (Z.ltb_spec (x + 2 ^ Z.of_nat n) (2 ^ (Z.of_nat n - 1))); lia.
Qed.

(* The signed reading of a sign extension of 32 bits lies in the signed
   window of 32 bits, whatever the fact of the register. *)
Lemma cvt_signed_bound_sound : forall t st out rs' to from sg d s v,
 regs_ok F E rs' -> wd F d = Z.of_nat to ->
 rs' d = trunc (wdn d) (extend from sg v) ->
 state_ok F t out rs' -> state_ok F t (cvt_signed_bound F st out to from sg d s) rs'.
Proof.
 intros t st out rs' to from sg d s v Hrs' Hd Hv Hout. unfold cvt_signed_bound.
 destruct (Nat.eqb to 64 && Nat.eqb from 32 && sg && Z.eqb (wd F s) 64) eqn:Ec; [| exact Hout].
 rewrite !andb_true_iff in Ec. destruct Ec as [[[Hto Hfrom] Hsg] _].
 apply Nat.eqb_eq in Hto, Hfrom. subst to from sg.
 destruct (get out d) as [[f al vr | | |] |]; try exact Hout. destruct vr; [exact Hout |].
 destruct (Z.eqb (fm f) 32 && negb (null (ft f))); [| exact Hout].
 apply (add_cons_ok F E HE t rs' Hrs' out _ Hout).
 assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
 cbn [cons_ok tsum atom_val]. rewrite Hdn, Hv, Hdn. cbn [extend]. rewrite sext_trunc.
 pose proof (sext_range 32 v ltac:(lia)) as H32.
 rewrite (sext_id 64 (sext 32 v)); [| lia |]; change (Z.of_nat 32 - 1) with 31 in *; change (Z.of_nat 64 - 1) with 63; lia.
Qed.

Lemma relational_cvt_sound : forall t st rs to from sg d s x,
 state_ok F t st rs -> regs_ok F E rs ->
 wd F d = Z.of_nat to -> Z.of_nat from <= wd F s ->
 relational_cvt F st to from sg d s = Some x ->
 fact_okv F t rs x (trunc (wdn d) (extend from sg (rs s))).
Proof.
 intros t st rs to from sg d s x Hst Hrs Hd Hs H.
 assert (Hbase : value F st (ICvt None to from sg d s) = Some x ->
   fact_okv F t rs x (trunc (wdn d) (extend from sg (rs s)))).
 { exact (value_cvt F E HE t st rs Hst Hrs None to from sg d s x Hd Hs). }
 unfold relational_cvt in H. cbv zeta in H.
 destruct (negb (Nat.eqb to 64 && Nat.eqb from 32)) eqn:Et; [exact (Hbase H) |].
 destruct (get st s) as [[f al vr | | | ] |] eqn:Eg; try exact (Hbase H).
 destruct (Z.eqb (wd F s) 32 && (32 <=? fm f) && negb (null (ft f))) eqn:Ef; [| exact (Hbase H)].
 destruct (check_range F st f) as [[lo hi] |] eqn:Er; [| exact (Hbase H)].
 destruct ((0 <=? lo) && (hi <? if sg then 2 ^ 31 else 2 ^ 32)) eqn:Eb; [| exact (Hbase H)].
 apply negb_false_iff in Et. apply andb_true_iff in Et. destruct Et as [Ht Hfrom].
 apply Nat.eqb_eq in Ht,Hfrom. subst to from.
 rewrite !andb_true_iff in Ef. destruct Ef as [[Hsw Hfm] _].
 apply Z.eqb_eq in Hsw. apply Z.leb_le in Hfm.
 apply andb_true_iff in Eb. destruct Eb as [Hlo Hhi]. apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi.
 injection H as <-. clear Hbase.
 pose proof (proj1 Hst s _ Eg) as [Hwf [[e [He Hm]] _]].
 pose proof (check_range_sound F E HE t rs Hrs st f lo hi e Hst Er He) as Hr.
 pose proof (rs_range F E rs Hrs s) as Hrs0. change (0 <= rs s < 2 ^ wd F s) in Hrs0. rewrite Hsw in Hrs0.
 assert (Hx : rs s = tsum wdn rs (ft f) + e).
 { apply (eqm_close 32); [lia | apply (eqm_weaken (fm f)); [lia | exact Hm] |].
   destruct sg; cbn in Hhi; lia. }
 assert (Hext : extend 32 sg (rs s) = rs s).
 { destruct sg; cbn [extend].
   - apply sext_small; [lia | change (0 <= rs s < 2 ^ (32 - 1)); rewrite Hx; lia].
   - apply trunc_small. exact Hrs0. }
 assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
 rewrite Hext,Hdn, trunc_small by (cbn; lia).
 cbn [fact_okv]. split; [unfold wf; cbn; lia |].
 split.
 - exists e. cbn [flo fhi fm ft]. split; [exact He | rewrite <- Hx; apply eqm_refl].
 - split; [apply aligned_nonpos; lia | discriminate].
Qed.

Lemma cval_u64_reg : forall rs r,
 regs_ok F E rs -> wd F r = 64 -> cval rs (OReg r) 64 false = rs r.
Proof.
 intros rs r Hrs Hw. unfold cval. cbn [opv extend]. apply trunc_small.
 pose proof (rs_range F E rs Hrs r) as Hr. change (0 <= rs r < 2 ^ wd F r) in Hr.
 rewrite Hw in Hr. exact Hr.
Qed.

Lemma select_between_sound : forall t out rs d low high,
 regs_ok F E rs -> wd F d = 64 -> owidth F low 64 -> owidth F high 64 ->
 cval rs low 64 false <= rs d <= cval rs high 64 false ->
 state_ok F t out rs -> state_ok F t (select_between F d low high out) rs.
Proof.
 intros t out rs d low high Hrs Hd Hwl Hwh [Hl Hh] Hout.
 pose proof (rs_range F E rs Hrs d) as Hdr.
 change (0 <= rs d < 2 ^ wd F d) in Hdr. rewrite Hd in Hdr.
 assert (Hlower : state_ok F t
   (match low with OReg r => add_cons out (sorted_terms [((d,EX),1);((r,EX),-1)],0,2^64-1) | _ => out end) rs).
 { destruct low as [r | z]; [| exact Hout].
   rewrite (cval_u64_reg rs r Hrs (Hwl r eq_refl)) in Hl.
   pose proof (rs_range F E rs Hrs r) as Hr.
   apply (add_cons_ok F E HE t rs Hrs _ _ Hout).
   cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE t rs Hrs). cbn [tsum atom_val]. lia. }
 set (out1 := match low with OReg r => add_cons out (sorted_terms [((d,EX),1);((r,EX),-1)],0,2^64-1) | _ => out end) in *.
 assert (Hupper : state_ok F t
   (match high with OReg r => add_cons out1 (sorted_terms [((d,EX),1);((r,EX),-1)],1-2^64,0) | _ => out1 end) rs).
 { destruct high as [r | z]; [| exact Hlower].
   rewrite (cval_u64_reg rs r Hrs (Hwh r eq_refl)) in Hh.
   pose proof (rs_range F E rs Hrs r) as Hr.
   change (0 <= rs r < 2 ^ wd F r) in Hr. rewrite (Hwh r eq_refl) in Hr.
   apply (add_cons_ok F E HE t rs Hrs _ _ Hlower).
   cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE t rs Hrs). cbn [tsum atom_val]. lia. }
 set (out2 := match high with OReg r => add_cons out1 (sorted_terms [((d,EX),1);((r,EX),-1)],1-2^64,0) | _ => out1 end) in *.
 assert (Hlc : cmp_holds rs {|cc:=CGe;cw:=64;csg:=false;ca:=d;cb:=low|}).
 { unfold cmp_holds. cbn [cc cw csg ca cb cmp_eval].
   change ((cval rs low 64 false <=? cval rs (OReg d) 64 false) = true).
   rewrite (cval_u64_reg rs d Hrs Hd). apply Z.leb_le. exact Hl. }
 assert (Hhc : cmp_holds rs {|cc:=CLe;cw:=64;csg:=false;ca:=d;cb:=high|}).
 { unfold cmp_holds. cbn [cc cw csg ca cb cmp_eval].
   change ((cval rs (OReg d) 64 false <=? cval rs high 64 false) = true).
   rewrite (cval_u64_reg rs d Hrs Hd). apply Z.leb_le. exact Hh. }
 unfold select_between. fold out1 out2.
 destruct (refine_cmp_sound F E HE t rs Hrs out2 _ Hupper Hlc) as [s [Es Hs]]. rewrite Es.
 destruct (refine_cmp_sound F E HE t rs Hrs s _ Hs Hhc) as [s' [Es' Hs']]. rewrite Es'. exact Hs'.
Qed.

Lemma select_order_sound : forall t st out rs rs' w d a b p,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' -> wd F d = w ->
 rs' d = trunc (wdn d) (if Z.eqb (rs p) 0 then opv rs b else opv rs a) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (select_order F st out w d a b) rs'.
Proof.
 intros t st out rs rs' w d a b p Hst Hrs Hrs' Ha Hd Hv Hout.
 unfold select_order.
 lazymatch goal with |- state_ok _ _ (if ?b then _ else _) _ => destruct b; [exact Hout |] end.
 destruct (negb (Z.eqb w 64)) eqn:Ew; [exact Hout |].
 apply negb_false_iff in Ew. apply Z.eqb_eq in Ew. rewrite Ew in Hd. subst w.
 set (valid := fun o => match o with OReg r => negb (Pos.eqb r d) && Z.eqb (wd F r) 64 | OImm _ => true end).
 lazymatch goal with |- state_ok _ _ (if ?test then _ else _) _ => destruct test eqn:Ev; [exact Hout |] end.
 change (negb (valid a && valid b) = false) in Ev.
 apply negb_false_iff in Ev. apply andb_true_iff in Ev. destruct Ev as [Eva Evb].
 assert (Hwidth : forall o, valid o = true -> owidth F o 64).
 { intros o Ho r Hr. subst o. unfold valid in Ho. apply andb_true_iff in Ho. apply Z.eqb_eq. exact (proj2 Ho). }
 assert (Hread : forall o, valid o = true -> cval rs' o 64 false = cval rs o 64 false).
 { intros [r | z] Ho; [| reflexivity]. unfold valid in Ho. apply andb_true_iff in Ho.
   destruct Ho as [Hne _]. apply negb_true_iff in Hne. apply Pos.eqb_neq in Hne.
   unfold cval. cbn [opv]. rewrite (Ha r Hne). reflexivity. }
 destruct (cmp_range F st a 64 false) as [[al ah] |] eqn:Ea; [| exact Hout].
 destruct (cmp_range F st b 64 false) as [[bl bh] |] eqn:Eb; [| exact Hout].
 pose proof (cmp_range_sound F E HE t rs Hrs st a 64 false al ah Hst ltac:(lia) (Hwidth a Eva) Ea) as Hra.
 pose proof (cmp_range_sound F E HE t rs Hrs st b 64 false bl bh Hst ltac:(lia) (Hwidth b Evb) Eb) as Hrb.
 assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
 rewrite Hdn in Hv.
 assert (Hcases : rs' d = cval rs a 64 false \/ rs' d = cval rs b 64 false).
 { destruct (Z.eqb (rs p) 0); [right | left]; exact Hv. }
 cbv zeta. destruct (ah <=? bl) eqn:Eab.
 - apply Z.leb_le in Eab. apply (select_between_sound (S t) out rs' d a b Hrs' Hd (Hwidth a Eva) (Hwidth b Evb)); [| exact Hout].
   rewrite (Hread a Eva), (Hread b Evb). destruct Hcases; lia.
 - destruct (bh <=? al) eqn:Eba; [| exact Hout].
   apply Z.leb_le in Eba. apply (select_between_sound (S t) out rs' d b a Hrs' Hd (Hwidth b Evb) (Hwidth a Eva)); [| exact Hout].
   rewrite (Hread a Eva), (Hread b Evb). destruct Hcases; lia.
Qed.

(* Every proposed selection constraint is checked in both operand states. *)
Lemma selected_step_sound : forall t st out rs rs' w d r,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 (rs' d = trunc (wdn d) (rs d) \/ rs' d = trunc (wdn d) (rs r)) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (selected_step F st out w d r) rs'.
Proof.
 intros t st out rs rs' w d r Hst Hrs Hrs' Ha Hv Hout.
 unfold selected_step.
 destruct (negb (Z.eqb w 64 && Z.eqb (wd F d) 64 && Z.eqb (wd F r) 64) || Pos.eqb d r); [exact Hout |].
 set (delta := materialized_form F st (normalize F st (fadd (as_form F st r EX) (fscale (as_form F st d EX) (-1) 64))) d r).
 destruct (negb (Z.eqb (fm delta) 64 && Z.eqb (flo delta) (fhi delta)) || mentions delta d); [exact Hout |].
 generalize (acons st). intros cs. revert out Hout.
 induction cs as [| [[ts lo] hi] cs IH]; intros out Hout; cbn [fold_left]; [exact Hout |].
 apply IH. destruct (negb (cons_mentions d (ts,lo,hi))); [exact Hout |].
 set (q := coef ts (d,EX) + coef ts (d,EZ)).
 set (ts' := sorted_terms (ts ++ tscale (-q) (ft delta))).
 destruct (check_range F st (form_of_terms ts')) as [[l h] |] eqn:Er; [| exact Hout].
 set (candidate := (ts',Z.min l (lo+q*flo delta),Z.max h (hi+q*flo delta))).
 assert (Hcandidate : state_ok F t {|amap:=amap st;acons:=[candidate]|} rs).
 { split; [exact (proj1 Hst) |]. constructor; [| constructor].
   cbn [cons_ok candidate].
   pose proof (check_range_sound F E HE t rs Hrs st (form_of_terms ts') l h 0 Hst Er ltac:(cbn [form_of_terms flo fhi]; lia)) as Hr.
   cbn [form_of_terms ft] in Hr. lia. }
 apply (branch_cons1_sound t _ _ out w d (OReg d) (OReg r) rs rs' Hrs Hrs' Ha); [| exact Hout].
 destruct Hv as [Hv | Hv].
 - right. split; [exact Hcandidate | exact Hv].
 - left. split; [apply projected_constraint_sound; assumption | exact Hv].
Qed.

Lemma selection_steps_sound : forall t st out rs rs' w d a b p,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 rs' d = trunc (wdn d) (if Z.eqb (rs p) 0 then opv rs b else opv rs a) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (selection_steps F st out w d a b) rs'.
Proof.
 intros t st out rs rs' w d a b p Hst Hrs Hrs' Ha Hv Hout.
 unfold selection_steps.
 destruct a as [r | z], b as [s | z']; try exact Hout.
 rewrite !selected_step_cached_exact.
 destruct (Pos.eqb_spec d r) as [-> | Hdr].
 - apply (selected_step_sound t st out rs rs' w r s Hst Hrs Hrs' Ha); [| exact Hout].
   destruct (Z.eqb (rs p) 0); [right | left]; exact Hv.
 - destruct (Pos.eqb_spec d s) as [-> | Hds]; [| exact Hout].
   apply (selected_step_sound t st out rs rs' w s r Hst Hrs Hrs' Ha); [| exact Hout].
   destruct (Z.eqb (rs p) 0); [left | right]; exact Hv.
Qed.

Lemma known_operand_sound : forall t st rs o,
 state_ok F t st rs -> regs_ok F E rs -> opv rs (known_operand F st o) = opv rs o.
Proof.
 intros t st rs [r | z] Hst Hrs; [| reflexivity]. rewrite known_operand_exact.
 destruct (const_of F st (OReg r) 64 false) as [c |] eqn:Ec; [| reflexivity].
 cbn [opv]. symmetry. exact (const_of_sound F E HE t st rs Hst Hrs (OReg r) 64 c ltac:(lia) Ec).
Qed.

Lemma sentinel_load_sound : forall t st out rs rs' g bits d,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 (holds g rs = false -> rs' d = rs d) -> state_ok F (S t) out rs' ->
 state_ok F (S t) (sentinel_load F st out g bits d) rs'.
Proof.
 intros t st out rs rs' g bits d Hst Hrs Hrs' Ha Hold Hout.
 unfold sentinel_load. destruct g as [[neg p] |]; [destruct neg |]; try exact Hout.
 destruct (negb (Z.eqb bits 64 && Z.eqb (wd F d) 64 && Z.eqb (wd F p) 1)) eqn:Ew; [exact Hout |].
 apply negb_false_iff in Ew. rewrite !andb_true_iff in Ew. destruct Ew as [[_ Hd] Hp].
 apply Z.eqb_eq in Hd,Hp.
 destruct (const_of F st (OReg d) 64 false) as [c |] eqn:Ec; [| exact Hout].
 destruct (Z.eqb c (2^63-1)) eqn:Es; [| exact Hout]. apply Z.eqb_eq in Es. subst c.
 pose proof (const_of_sound F E HE t st rs Hst Hrs (OReg d) 64 (2^63-1) ltac:(lia) Ec) as Hc.
 assert (Hne : p <> d) by (intros ->; lia).
 assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
 pose proof (rs_range F E rs Hrs p) as Hpr.
 change (0 <= rs p < 2 ^ wd F p) in Hpr. rewrite Hp in Hpr.
 pose proof (extend_range 64 true (rs' d) ltac:(lia)) as Hdr.
 apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
 cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
 cbn [tsum atom_val]. rewrite Hdn, (Ha p Hne).
 assert (Hcases : rs p = 0 \/ rs p = 1) by lia.
 destruct Hcases as [Hz | Ho].
 - assert (Hkeep : rs' d = 2^63-1).
   { rewrite Hold; [exact Hc | cbn [holds]; rewrite Hz; reflexivity]. }
   rewrite Hkeep,Hz. rewrite sext_small by (cbn; lia). cbn in Hdr. lia.
 - rewrite Ho. cbn in Hdr. lia.
Qed.

Lemma type_default_sound : forall t out rs d,
 state_ok F t out rs -> regs_ok F E rs -> state_ok F t (type_default F out d) rs.
Proof.
 intros t out rs d Hout Hrs. unfold type_default.
 destruct (get out d); [exact Hout|].
 apply (put_ok F E HE t rs Hrs out d _ Hout).
 pose proof (rs_range F E rs Hrs d) as Hr. pose proof (wd_range F E HE d) as Hw.
 change (0 <= rs d < 2^wd F d) in Hr.
 change (1 <= wd F d <= 64) in Hw.
 cbn [fact_ok fact_okv]. split; [unfold wf,range_form;cbn;lia|].
 split; [|split;[apply aligned_nonpos;lia|discriminate]].
 exists (rs d). cbn [range_form flo fhi ft tsum]. split;[lia|].
 apply eqm_refl.
Qed.

Lemma default_load_sound : forall t st out rs rs' g d,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 (holds g rs = false -> rs' d = rs d) -> state_ok F (S t) out rs' ->
 state_ok F (S t) (default_load F st out g d) rs'.
Proof.
 intros t st out rs rs' [[neg p]|] d Hst Hrs Hrs' Ha Hold Hout; [|exact Hout].
 unfold default_load.
 destruct (negb (Z.eqb (wd F p) 1) || Pos.eqb p d) eqn:Ew; [exact Hout|].
 apply orb_false_iff in Ew. destruct Ew as [Hp Hne]. apply negb_false_iff,Z.eqb_eq in Hp.
 apply Pos.eqb_neq in Hne.
 destruct (const_of F st (OReg d) 64 false) as [c|] eqn:Ec; [|exact Hout].
 pose proof (const_of_sound F E HE t st rs Hst Hrs (OReg d) 64 c ltac:(lia) Ec) as Hc.
 change (rs d = c) in Hc.
 pose proof (rs_range F E rs Hrs p) as Hpr.
 change (0 <= rs p < 2 ^ wd F p) in Hpr. rewrite Hp in Hpr.
 pose proof (rs_range F E rs Hrs d) as Hdr.
 pose proof (rs_range F E rs' Hrs' d) as Hdr'.
 change (0 <= rs d < 2 ^ wd F d) in Hdr.
 change (0 <= rs' d < 2 ^ wd F d) in Hdr'.
 assert (Hcases : rs p = 0 \/ rs p = 1) by lia.
 assert (Hupper : cons_ok F rs' (if neg then
   (sorted_terms [((d,EZ),1);((p,EX),2^wd F d-1)],0,2^wd F d-1+c)
   else (sorted_terms [((d,EZ),1);((p,EX),-(2^wd F d-1))],-(2^wd F d-1),c))).
 { destruct neg; cbn [cons_ok]; rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'); cbn [tsum atom_val]; rewrite (Ha p Hne).
   all: destruct Hcases as [Hz|Hz]; rewrite Hz; try lia.
   - assert (Hkeep:rs' d=rs d) by (apply Hold; cbn [holds]; rewrite Hz; reflexivity). lia.
   - assert (Hkeep:rs' d=rs d) by (apply Hold; cbn [holds]; rewrite Hz; reflexivity). lia. }
 assert (Hlower : cons_ok F rs' (if neg then
   (sorted_terms [((d,EZ),1);((p,EX),-(2^wd F d-1))],c-(2^wd F d-1),2^wd F d-1)
   else (sorted_terms [((d,EZ),1);((p,EX),2^wd F d-1)],c,2*(2^wd F d-1)))).
 { destruct neg; cbn [cons_ok]; rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'); cbn [tsum atom_val]; rewrite (Ha p Hne).
   all: destruct Hcases as [Hz|Hz]; rewrite Hz; try lia.
   - assert (Hkeep:rs' d=rs d) by (apply Hold; cbn [holds]; rewrite Hz; reflexivity). lia.
   - assert (Hkeep:rs' d=rs d) by (apply Hold; cbn [holds]; rewrite Hz; reflexivity). lia. }
 apply (add_cons_ok F E HE (S t) rs' Hrs'); [|exact Hlower].
 apply (add_cons_ok F E HE (S t) rs' Hrs'); [apply type_default_sound;assumption|exact Hupper].
Qed.

Theorem transfer_live_sound : forall t s s' st i,
    fetch prog (blk s) (idx s) = Some i ->
    stepF t s = Some s' ->
    state_ok F t st (regs s) -> regs_ok F E (regs s) ->
    wf_instr F i = true ->
    state_ok F (S t) (transfer_live F st i) (regs s') /\ regs_ok F E (regs s').
Proof.
  intros t s s' st i Hf Hstep Hst Hrs Hwf.
  pose proof (state_ok_later t st (regs s) Hst) as Hst1.
  pose proof Hwf as Hwf'. unfold wf_instr in Hwf'. apply andb_true_iff in Hwf'. destruct Hwf' as [Hwf' _].
  apply andb_true_iff in Hwf'. destruct Hwf' as [Hfix _].
  destruct (dest i) as [d |] eqn:Hd.
  - (* an instruction that writes d *)
    pose proof (step_regs_dest t s s' i d Hf Hd Hstep) as Hs'.
    assert (Hdf : is_fixed F d = false).
    { destruct i; cbn [dest] in Hd; try discriminate; injection Hd as <-; cbn in Hfix; apply negb_true_iff; exact Hfix. }
    set (rs := regs s) in *.
    assert (Hrs' : regs_ok F E (regs s')) by (rewrite Hs'; destruct (holds (guard_of i) rs); [apply upd_regs_ok |]; assumption).
    split; [| exact Hrs'].
    assert (Ha : agree d rs (regs s')) by (rewrite Hs'; destruct (holds (guard_of i) rs); [apply upd_agree | intros r _; reflexivity]).
    assert (Hnew : holds (guard_of i) rs = true -> regs s' d = trunc (wdn d) (result (e_mem E) (e_arb E) (e_shm E) t rs i)).
    { intros Hg. rewrite Hs', Hg. unfold upd. rewrite Pos.eqb_refl. reflexivity. }
    assert (Hold : holds (guard_of i) rs = false -> regs s' d = rs d) by (intros Hg; rewrite Hs', Hg; reflexivity).
    (* the transfer of a writing instruction, and the facts of its value *)
    destruct i as [g d' s0 | g d' z | g o w d' a b | g w d' a b c | g sg d' a b c | g to from sg d' s0
                   | g w d' a p ln | g c w sg d' a b | g d' a b | g d' a b | g mn sg w d' a b | g w d' a b p
                   | g gen bits d' a off | g k gen a off width v fr ds | g ds | g target | g | | g rm sg w d' a b
                   | g bits d' a off | g bits d' a off fr | g a off width v fr | vs | g a off width | g d' a b];
      cbn [dest] in Hd; try discriminate; injection Hd as ->;
      cbn [transfer_live dest guard_of] in *.
    13: { (* a load: d is forgotten, and a load of an index input gives the
           constraints of rule R3, under its guard joined with d's old value *)
      destruct g as [gg |].
      - apply (default_load_sound t st _ rs (regs s') (Some gg) d Hst Hrs Hrs' Ha Hold).
        apply (sentinel_load_sound t st _ rs (regs s') (Some gg) (Z.of_nat bits) d Hst Hrs Hrs' Ha Hold).
        apply (guarded_index_load_sound t st rs (regs s') (Some gg) gen bits d a off Hst Hrs Hrs' Ha Hwf);
          [intros Hg; rewrite Hs', Hg; reflexivity | exact Hold].
      - assert (Hst2 : state_ok F (S t) (transfer_dest F st d None None false) (regs s'))
          by (apply (transfer_dest_sound (S t) st rs (regs s') d None None false Hst1 Hrs Hrs' Ha);
              [discriminate | exact Hold | discriminate]).
        apply (index_load_sound F E HE t st _ gen bits d a off rs (regs s') Hst Hrs Hwf); [| exact Hrs' | exact Hst2].
        rewrite Hs'. cbn [holds]. reflexivity. }
    13: { (* a division; an unguarded remainder also gives the relation of
           rule R4 *)
      assert (Hgen : state_ok F (S t) (transfer_dest F st d g (value F st (IDiv g rm sg w d a b)) false) (regs s')).
      { apply (transfer_dest_sound (S t) st rs (regs s') d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
        intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later.
        exact (value_sound F E HE t st rs Hst Hrs _ d x Hwf eq_refl Hx). }
      destruct g as [gg |]; [exact Hgen |]. destruct rm; [| exact Hgen].
      assert (Hws : wd F d = Z.of_nat w /\ wd F a = Z.of_nat w /\ owidth F b (Z.of_nat w)).
      { unfold wf_instr in Hwf. cbn [guard_of dest] in Hwf. rewrite !andb_true_iff in Hwf.
        destruct Hwf as [[_ [[Hw1 Hw2] Hw3]] _].
        apply Z.eqb_eq in Hw1, Hw2. split; [exact Hw1 |]. split; [exact Hw2 |].
        intros r Hr. subst b. cbn [opnd_width_ok] in Hw3. apply Z.eqb_eq. exact Hw3. }
      destruct Hws as [Hwd [Hwa Hwb]].
      apply (rem_cons_sound t st _ sg w d a b rs (regs s') Hst Hrs Ha Hwd Hwa Hwb); [| exact Hrs' | exact Hgen].
      rewrite (Hnew eq_refl). reflexivity. }
    11: { (* min/max retain their order relation to both operands *)
      assert (Hgen : state_ok F (S t) (transfer_dest F st d g (value F st (IMinMax g mn sg w d a b)) false) (regs s')).
      { apply (transfer_dest_sound (S t) st rs (regs s') d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
        intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later.
        exact (value_sound F E HE t st rs Hst Hrs _ d x Hwf eq_refl Hx). }
      destruct g as [gg |]; [exact Hgen |].
      assert (Hws : wd F d = Z.of_nat w /\ wd F a = Z.of_nat w /\ owidth F b (Z.of_nat w)).
      { unfold wf_instr in Hwf. cbn [guard_of dest] in Hwf. rewrite !andb_true_iff in Hwf.
        destruct Hwf as [[_ [[Hw1 Hw2] Hw3]] _].
        apply Z.eqb_eq in Hw1, Hw2. split; [exact Hw1 |]. split; [exact Hw2 |].
        intros r Hr. subst b. cbn [opnd_width_ok] in Hw3. apply Z.eqb_eq. exact Hw3. }
      destruct Hws as [Hwd [Hwa Hwb]].
      apply (minmax_relative_cons_sound t st _ rs (regs s') mn sg w d a b Hst Hrs Hrs' Ha Hwd Hwa Hwb).
      { rewrite (Hnew eq_refl). reflexivity. }
      apply (minmax_cons_sound (S t) _ rs (regs s') mn sg w d a b Hrs' Ha Hwd);
        [rewrite (Hnew eq_refl); reflexivity | exact Hgen]. }
    11: { (* a selp: one operand's value, of which that operand's fact, or the
           join, holds; an unguarded one of two registers moves constraints *)
      destruct (PM.mem d (lf_payloads F)) eqn:Hpayload.
      { apply (assign_sound (S t) st rs (regs s') d None None Hst1 Hrs Ha I).
        intros y Hy. discriminate. }
      assert (Hsd : state_ok F (S t) (transfer_dest F st d g (sel_value F st a b p (Z.of_nat w)) false) (regs s')).
      2: { destruct g as [gg |]; [exact Hsd |].
           assert (Hwd : wd F d = Z.of_nat w).
           { unfold wf_instr in Hwf. cbn [guard_of dest] in Hwf. rewrite !andb_true_iff in Hwf.
             destruct Hwf as [[_ [Hw1 _]] _]. apply Z.eqb_eq. exact Hw1. }
           apply (selection_steps_sound t st _ rs (regs s') (Z.of_nat w) d a b p Hst Hrs Hrs' Ha).
           { rewrite (Hnew eq_refl). reflexivity. }
           apply (select_order_sound t st _ rs (regs s') (Z.of_nat w) d (known_operand F st a) (known_operand F st b) p Hst Hrs Hrs' Ha Hwd).
           { rewrite (known_operand_sound t st rs a Hst Hrs), (known_operand_sound t st rs b Hst Hrs).
             rewrite (Hnew eq_refl). reflexivity. }
           rewrite sel_refined_cons_prepared_exact.
           apply (sel_refined_cons_sound t st _ (Z.of_nat w) d a b p rs (regs s') Hst Hrs Hrs' Ha); [| exact Hsd].
           rewrite (Hnew eq_refl). reflexivity. }
      apply (transfer_dest_sound (S t) st rs (regs s') d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
      intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later. cbn [result].
      assert (Hwd : wd F d = Z.of_nat w).
      { unfold wf_instr in Hwf. cbn [guard_of dest] in Hwf. rewrite !andb_true_iff in Hwf.
        destruct Hwf as [[_ [Hw1 _]] _]. apply Z.eqb_eq. exact Hw1. }
      assert (Hw : 1 <= Z.of_nat w <= 64) by (pose proof (wd_range F E HE d); unfold wd in Hwd; lia).
      replace (wdn d) with (Z.to_nat (Z.of_nat w)) by (unfold wd in Hwd; lia).
      assert (Hjoin : sel_join F st a b p (Z.of_nat w) = Some x ->
                      fact_okv F t rs x (trunc (Z.to_nat (Z.of_nat w)) (if Z.eqb (rs p) 0 then opv rs b else opv rs a))).
      { intros Hx'. unfold sel_join in Hx'.
        destruct (Z.eqb (rs p) 0) eqn:Ep.
        - (* p fails, so the refinement by its failure succeeds *)
          destruct (refine_sound F E HE t rs Hrs st p false Hst ltac:(rewrite Ep; reflexivity)) as [s1 [Er1 Hs1]].
          rewrite Er1 in Hx'. destruct (refine F st p true) as [s2 |].
          + destruct (sel_fact F s1 b (Z.of_nat w)) as [xb |] eqn:Eb;
              [| destruct (sel_fact F s2 a (Z.of_nat w)); discriminate].
            exact (join_fact_sound_r F E HE t rs Hrs s2 s1 _ xb x _ Hs1
                     (sel_fact_sound t s1 rs b _ xb Hs1 Hrs Hw Eb) Hx').
          + exact (sel_fact_sound t s1 rs b _ x Hs1 Hrs Hw Hx').
        - (* p holds, so the refinement by it succeeds *)
          destruct (refine_sound F E HE t rs Hrs st p true Hst ltac:(rewrite Ep; reflexivity)) as [s1 [Er1 Hs1]].
          rewrite Er1 in Hx'. destruct (refine F st p false) as [s2 |].
          + destruct (sel_fact F s1 a (Z.of_nat w)) as [xa |] eqn:Ea; [| discriminate].
            exact (join_fact_sound_l F E HE t rs Hrs s1 s2 xa _ x _ Hs1
                     (sel_fact_sound t s1 rs a _ xa Hs1 Hrs Hw Ea) Hx').
          + exact (sel_fact_sound t s1 rs a _ x Hs1 Hrs Hw Hx'). }
      unfold sel_value in Hx.
      destruct (refine F st p true) as [s1 |]; [| exact (Hjoin Hx)].
      destruct (refine F st p false) as [s2 |]; [| exact (Hjoin Hx)].
      destruct a as [ra | za]; [exact (Hjoin Hx) |]. destruct b as [rb | zb]; [exact (Hjoin Hx) |].
      destruct (Z.eqb_spec (wd F p) 1) as [Hp1 | Hp1]; [| exact (Hjoin Hx)].
      (* two constants: b + (a - b) * p, with p's value 0 or 1 *)
      injection Hx as <-. cbn [opv].
      pose proof (rs_range F E rs Hrs p) as Hrp. unfold wd in Hp1. rewrite Hp1 in Hrp.
      cbn [fact_okv]. split; [unfold wf; rewrite mk_fm; lia |].
      split; [| split; [apply aligned_nonpos; lia | discriminate]].
      apply (sat_mk wdn rs (Z.of_nat w) _ zb zb _ zb); [lia | lia |]. cbn [tsum atom_val].
      unfold trunc. rewrite Z2Nat.id by lia.
      destruct (Z.eqb_spec (rs p) 0) as [Hz | Hz].
      - rewrite Hz. replace ((za - zb) * 0 + 0 + zb) with zb by ring. apply eqm_mod. lia.
      - assert (rs p = 1) as -> by lia. replace ((za - zb) * 1 + 0 + zb) with za by ring. apply eqm_mod. lia. }
    3: { (* a binary operation; an unguarded and of two registers also gives
           the relation of rule R4 *)
      assert (Hgen : state_ok F (S t) (transfer_dest F st d g (value F st (IBin g o w d a b)) false) (regs s')).
      { apply (transfer_dest_sound (S t) st rs (regs s') d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
        intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later.
        exact (value_sound F E HE t st rs Hst Hrs _ d x Hwf eq_refl Hx). }
      destruct g as [gg |]; [exact Hgen |]. destruct o; try exact Hgen.
      1: {
        assert (Hws : wd F d = Z.of_nat w /\ wd F a = Z.of_nat w).
        { unfold wf_instr in Hwf. cbn [guard_of dest opnd_width_ok] in Hwf.
          rewrite !andb_true_iff in Hwf. destruct Hwf as [[_ [[Hw1 Hw2] _]] _].
          apply Z.eqb_eq in Hw1, Hw2. auto. }
        destruct Hws as [Hwd Hwa].
        apply (shift_cons_sound t st _ false w d a b rs (regs s') Hst Hrs Ha Hwd Hwa);
          [rewrite (Hnew eq_refl); reflexivity | exact Hrs' | exact Hgen]. }
      1: {
        assert (Hws : wd F d = Z.of_nat w /\ wd F a = Z.of_nat w).
        { unfold wf_instr in Hwf. cbn [guard_of dest opnd_width_ok] in Hwf.
          rewrite !andb_true_iff in Hwf. destruct Hwf as [[_ [[Hw1 Hw2] _]] _].
          apply Z.eqb_eq in Hw1, Hw2. auto. }
        destruct Hws as [Hwd Hwa].
        apply (shift_cons_sound t st _ true w d a b rs (regs s') Hst Hrs Ha Hwd Hwa);
          [rewrite (Hnew eq_refl); reflexivity | exact Hrs' | exact Hgen]. }
      destruct b as [rb | z].
      2: {
       assert (Hws : wd F d = Z.of_nat w /\ wd F a = Z.of_nat w).
       { unfold wf_instr in Hwf. cbn [guard_of dest opnd_width_ok] in Hwf.
         rewrite !andb_true_iff in Hwf. destruct Hwf as [[_ [[Hw1 Hw2] _]] _].
         apply Z.eqb_eq in Hw1,Hw2. auto. }
       destruct Hws as [Hwd Hwa].
       apply (mask_const_cons_sound t st _ w d a z rs (regs s') Hst Hrs Ha Hwd Hwa);
         [rewrite (Hnew eq_refl); reflexivity | exact Hrs' | exact Hgen]. }
      assert (Hws : wd F d = Z.of_nat w /\ wd F a = Z.of_nat w /\ wd F rb = Z.of_nat w).
      { unfold wf_instr in Hwf. cbn [guard_of dest opnd_width_ok] in Hwf. rewrite !andb_true_iff in Hwf.
        destruct Hwf as [[_ [[Hw1 Hw2] Hw3]] _]. apply Z.eqb_eq in Hw1, Hw2, Hw3. auto. }
      destruct Hws as [Hwd [Hwa Hwb]].
      apply (and_cons_sound t st _ w d a rb rs (regs s') Hst Hrs Ha Hwd Hwa Hwb); [| exact Hrs' | exact Hgen].
      rewrite (Hnew eq_refl). reflexivity. }
    1: { (* a move: a copy of its source, which keeps its source's constraints *)
      assert (Hgen : state_ok F (S t) (transfer_dest F st d g (value F st (IMov g d s0)) false) (regs s')).
      { apply (transfer_dest_sound (S t) st rs (regs s') d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
        intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later.
        exact (value_sound F E HE t st rs Hst Hrs _ d x Hwf eq_refl Hx). }
      destruct g as [gg |]; [exact Hgen |].
      assert (Hsel : state_ok F (S t) (sel_cons F st (transfer_dest F st d None (value F st (IMov None d s0)) false)
                                                (wd F d) d (OReg s0) (OReg s0)) (regs s')).
      { apply (sel_cons_sound t st _ (wd F d) d (OReg s0) (OReg s0) d rs (regs s') Hst Hrs Hrs' Ha); [| exact Hgen].
        rewrite (Hnew eq_refl). cbn [result opv]. destruct (Z.eqb (rs d) 0); reflexivity. }
      destruct (existsb (cons_mentions s0) (acons st) && Z.eqb (wd F d) (wd F s0)) eqn:Ec; [| exact Hsel].
      destruct (reg_range F st s0) as [[lo hi] |] eqn:Er; [| exact Hsel].
      apply (add_cons_ok F E HE (S t) (regs s') Hrs' _ _ Hsel).
      (* the copy holds its source's value, which lies in the source's range *)
      apply andb_true_iff in Ec. destruct Ec as [_ Hw]. apply Z.eqb_eq in Hw.
      pose proof (reg_range_sound F E HE t st rs Hst Hrs s0 lo hi Er) as Hr.
      pose proof (rs_range F E rs Hrs s0) as Hrs0.
      cbn [cons_ok tsum atom_val]. rewrite (Hnew eq_refl). cbn [result].
      unfold wd in Hw. rewrite trunc_small by (rewrite Hw; exact Hrs0). lia. }
    4: { (* a widening conversion can use a nonwrapping relational range *)
      destruct g as [gg |].
      - apply (transfer_dest_sound (S t) st rs (regs s') d (Some gg) _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
        intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later.
        exact (value_sound F E HE t st rs Hst Hrs _ d x Hwf eq_refl Hx).
      - assert (Hws : wd F d = Z.of_nat to /\ Z.of_nat from <= wd F s0).
        { unfold wf_instr in Hwf. cbn [guard_of dest] in Hwf. rewrite !andb_true_iff in Hwf.
          destruct Hwf as [[_ [Hd Hs]] _]. apply Z.eqb_eq in Hd. apply Z.leb_le in Hs. auto. }
        destruct Hws as [Hwd Hsw].
        apply (cvt_signed_bound_sound (S t) st _ (regs s') to from sg d s0 (rs s0) Hrs' Hwd);
          [rewrite (Hnew eq_refl); reflexivity |].
        apply (transfer_dest_sound (S t) st rs (regs s') d None _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
        intros Hg x Hx. rewrite (Hnew Hg). apply fact_okv_later. cbn [result].
        exact (relational_cvt_sound t st rs to from sg d s0 x Hst Hrs Hwd Hsw Hx). }
    all: apply (transfer_dest_sound (S t) st rs (regs s') d g _ false Hst1 Hrs Hrs' Ha); [| exact Hold | discriminate].
    all: intros Hg x Hx; rewrite (Hnew Hg); apply fact_okv_later.
    all: exact (value_sound F E HE t st rs Hst Hrs _ d x Hwf eq_refl Hx).
  - (* an instruction that writes nothing, forgets its destinations, or
       accesses local memory *)
    destruct i as [g d' s0 | g d' z | g o w d' a b | g w d' a b c | g sg d' a b c | g to from sg d' s0
                   | g w d' a p ln | g c w sg d' a b | g d' a b | g d' a b | g mn sg w d' a b | g w d' a b p
                   | g gen bits d' a off | g k gen a off width v fr ds | g ds | g target | g | | g rm sg w d' a b
                   | g bits d' a off | g bits d' a off fr | g a off width v fr | vs | g a off width | g d' a b];
      cbn [dest] in Hd; try discriminate.
    (* the instructions that leave the registers: a branch, an exit, a nop *)
    3-5: rewrite (step_regs_same t s s' _ Hf eq_refl (fun _ _ E' => ltac:(discriminate))
                    (fun _ _ _ _ _ _ _ _ _ E' => ltac:(discriminate)) (fun _ _ _ _ _ _ E' => ltac:(discriminate))
                    (fun _ _ _ _ _ _ E' => ltac:(discriminate)) (fun _ E' => ltac:(discriminate)) Hstep);
         cbn [transfer_live]; split; assumption.
    + (* a global access *)
      rewrite (step_regs_mem t s s' g k gen a off width v fr ds Hf Hstep).
      exact (mem_sound t st (regs s) g k gen a off width v fr ds Hst Hrs Hwf).
    + (* an instruction whose destinations get arbitrary values *)
      rewrite (step_regs_def t s s' g ds Hf Hstep). cbn [transfer_live]. cbn in Hfix.
      destruct (holds g (regs s));
        [exact (forget_all_write (S t) ds st (regs s) (e_arb E) t Hst1 Hrs Hfix) |
         split; [exact (forget_all_same (S t) ds st (regs s) Hst1 Hrs) | exact Hrs]].
    + (* a local load *)
      rewrite (step_regs_ldl t s s' g bits d' a off fr Hf Hstep).
      exact (ldl_sound t st (regs s) g bits d' a off fr Hst Hrs Hwf).
    + (* a local store *)
      rewrite (step_regs_stl t s s' g a off width v fr Hf Hstep).
      exact (stl_sound t st (regs s) g a off width v fr Hst Hrs Hwf).
    + (* entering a frame *)
      rewrite (step_regs_frame t s s' vs Hf Hstep).
      exact (frame_sound t st (regs s) vs Hst Hrs).
    + (* an access to shared memory, which changes no register *)
      rewrite (step_regs_same t s s' _ Hf eq_refl (fun _ _ E' => ltac:(discriminate))
                 (fun _ _ _ _ _ _ _ _ _ E' => ltac:(discriminate)) (fun _ _ _ _ _ _ E' => ltac:(discriminate))
                 (fun _ _ _ _ _ _ E' => ltac:(discriminate)) (fun _ E' => ltac:(discriminate)) Hstep).
      cbn [transfer_live]. split; assumption.
Qed.

Lemma affine_constraint_sound : forall (t : nat) rs rs' d s ts lo hi,
 regs_ok F E rs' -> agree d rs rs' -> mentions s d = false -> flo s = fhi s ->
 rs' d = rs d + (tsum wdn rs (ft s) + flo s) ->
 forallb (fun t => match snd (fst t) with EX | EZ => true | _ => false end) (filter (on_reg d) ts) = true ->
 cons_ok F rs (ts,lo,hi) ->
 let q := fold_left (fun acc (t : atom * Z) => acc + snd t) (filter (on_reg d) ts) 0 in
 cons_ok F rs' (sorted_terms (ts ++ tscale (-q) (ft s)),lo+q*flo s,hi+q*fhi s).
Proof.
 intros t rs rs' d s ts lo hi Hrs' Ha Hmentions Hconst Hvalue Htypes Hcons q.
 set (delta := tsum wdn rs (ft s) + flo s) in *.
 pose proof (tsum_filter wdn rs (on_reg d) ts) as Hsplit.
 pose proof (tsum_filter wdn rs' (on_reg d) ts) as Hsplit'.
 assert (Hrest : tsum wdn rs' (filter (fun t => negb (on_reg d t)) ts) =
                 tsum wdn rs (filter (fun t => negb (on_reg d t)) ts)).
 { clear -Ha. induction ts as [| [[r e] q] ts IH]; [reflexivity |].
   cbn [filter]. change (on_reg d ((r,e),q)) with (Pos.eqb r d).
   destruct (Pos.eqb_spec r d) as [-> | Hne]; cbn [negb]; [exact IH |].
   cbn [tsum]. rewrite IH. destruct e; cbn [atom_val]; rewrite (Ha r Hne); reflexivity. }
 assert (Hchanged : forall term, In term (filter (on_reg d) ts) ->
   fst (fst term) = d /\ atom_val wdn rs' (fst term) = atom_val wdn rs (fst term) + delta).
 { intros [[r e] c] Hin. pose proof (proj1 (forallb_forall _ _) Htypes _ Hin) as Htype.
   apply filter_In in Hin. destruct Hin as [_ Hr]. change (Pos.eqb r d = true) in Hr. apply Pos.eqb_eq in Hr. subst r. cbn [fst snd] in *. split; [reflexivity |].
   destruct e; try discriminate; cbn [atom_val]; exact Hvalue. }
 pose proof (tsum_step t rs rs' d delta (filter (on_reg d) ts) Hchanged) as Hsum.
 fold q in Hsum.
 pose proof (tsum_agree wdn d rs rs' (ft s) Ha Hmentions) as Hs.
 cbn [cons_ok] in *. rewrite (tsum_sorted_terms F E HE t rs' Hrs'), tsum_app, tsum_tscale, Hs.
 unfold delta in *. rewrite <- Hconst. nia.
Qed.

Lemma affine_step_cons_sound : forall t st out rs rs' d s,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 eqm 64 (rs' d) (rs d + (tsum wdn rs (ft s) + flo s)) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (affine_step_cons F st out d s) rs'.
Proof.
 intros t st out rs rs' d s Hst Hrs Hrs' Ha Hstep Hout.
 unfold affine_step_cons.
 do 3 (lazymatch goal with |- state_ok _ _ (if ?b then _ else _) _ => destruct b eqn:?; [exact Hout |] end).
 destruct (negb (Z.eqb (wd F d) 64 && Z.eqb (fm s) 64 && Z.eqb (flo s) (fhi s)) || mentions s d) eqn:Etype; [exact Hout |].
 apply orb_false_iff in Etype. destruct Etype as [Etype Hmentions].
 apply negb_false_iff in Etype. rewrite !andb_true_iff in Etype. destruct Etype as [[Hd Hfm] Hconst].
 apply Z.eqb_eq in Hd,Hfm,Hconst.
 set (result := {|fm:=64;ft:=sorted_terms ([((d,EX),1)] ++ ft s);flo:=flo s;fhi:=fhi s|}) in *.
 destruct (check_range F st result) as [[lo hi] |] eqn:Er; [| exact Hout].
 destruct ((0 <=? lo) && (hi <? 2^64)) eqn:Ebounds; [| exact Hout].
 apply andb_true_iff in Ebounds. destruct Ebounds as [Hlo Hhi]. apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi.
 pose proof (check_range_sound F E HE t rs Hrs st result lo hi (flo s) Hst Er ltac:(unfold result; cbn [flo fhi]; lia)) as Hraw.
 unfold result in Hraw. cbn [ft] in Hraw.
 rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app in Hraw. cbn [tsum atom_val] in Hraw.
 assert (Hvalue : rs' d = rs d + (tsum wdn rs (ft s) + flo s)).
 { pose proof (rs_range F E rs' Hrs' d) as Hr. change (0 <= rs' d < 2 ^ wd F d) in Hr. rewrite Hd in Hr.
   apply (eqm_close 64); [lia | exact Hstep | lia]. }
 generalize (proj2 Hst). generalize (acons st). intros ks Hks. revert out Hout.
 induction Hks as [| [[ts l] h] ks Hk Hks IH]; intros out Hout; cbn [fold_left]; [exact Hout |].
 apply IH.
 lazymatch goal with |- state_ok _ _ (if ?b then _ else _) _ => destruct b eqn:Ekeep; [exact Hout |] end.
 repeat match goal with H : (_ || _) = false |- _ => apply orb_false_iff in H; destruct H end.
 match goal with H : negb (forallb _ _) = false |- _ => apply negb_false_iff in H end.
 apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
 eapply (affine_constraint_sound (S t) rs rs' d s ts l h Hrs' Ha Hmentions Hconst Hvalue); eassumption.
Qed.

Lemma complement_cons_sound : forall t st out rs rs' d k,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 rs' d = bool_z (cmp_eval (cc k) (extend (cw k) (csg k) (rs (ca k)))
                         (extend (cw k) (csg k) (opv rs (cb k)))) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (complement_cons F st out d k) rs'.
Proof.
 intros t st out rs rs' d k Hst Hrs Hrs' Ha Hv Hout. unfold complement_cons.
 destruct (negb (Z.eqb (wd F d) 1)) eqn:Ed; [exact Hout |].
 set (xs := PM.elements (amap st)).
 assert (Hxs : Forall (fun pair => get st (fst pair) = Some (snd pair)) xs).
 { apply Forall_forall. intros [p x] Hin. apply PM.elements_complete. exact Hin. }
 clearbody xs. revert out Hout. induction Hxs as [| [p x] xs Hx Hxs IH]; intros out Hout;
 cbn [fold_left]; [exact Hout |]. apply IH.
 destruct (Pos.eqb p d) eqn:Ep; [exact Hout |]. apply Pos.eqb_neq in Ep.
 destruct x as [f al vr | al | cs exact | cs exact]; try exact Hout.
 destruct cs as [|old more]; [exact Hout |]. destruct more; [| exact Hout].
 destruct exact; [| exact Hout].
 destruct (Z.eqb (wd F p) 1 && cmpf_eqb (negate old) k) eqn:Ek; [| exact Hout].
 apply andb_true_iff in Ek. destruct Ek as [Hp Hk]. apply Z.eqb_eq in Hp.
 apply cmpf_eqb_eq in Hk. subst k.
 pose proof (proj1 Hst p _ Hx) as Hpred. cbn [fact_ok fact_okv] in Hpred.
 destruct Hpred as [Hyes Hno].
 pose proof (rs_range F E rs Hrs p) as Hpr.
 change (0 <= rs p < 2 ^ wd F p) in Hpr. rewrite Hp in Hpr.
 cbn [negate cc cw csg ca cb] in Hv. rewrite flip_eval in Hv.
 apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
 cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
 cbn [tsum atom_val]. rewrite (Ha p Ep).
 destruct (cmp_eval (cc old) (extend (cw old) (csg old) (rs (ca old)))
                            (extend (cw old) (csg old) (opv rs (cb old)))) eqn:Ecmp.
 - assert (Hcmp : cmp_holds rs old) by exact Ecmp.
   assert (Hpn : rs p <> 0) by (apply (Hno eq_refl); constructor; [exact Hcmp | constructor]).
   cbn [negb bool_z] in Hv. lia.
 - assert (Hz : rs p = 0).
   { destruct (Z.eq_dec (rs p) 0) as [Hz | Hn]; [exact Hz |].
     specialize (Hyes Hn). inversion Hyes as [| z zs Hcmp _]; subst.
     unfold cmp_holds in Hcmp. rewrite Ecmp in Hcmp. discriminate. }
   cbn [negb bool_z] in Hv. lia.
Qed.

Lemma arithmetic_relations_sound : forall t st out rs rs' i,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' ->
 state_ok F (S t) out rs' ->
 (forall op d a b, i = IBin None op 64 d a b ->
   wd F d = 64 /\ agree d rs rs' /\ rs' d = trunc 64 (binop_eval op 64 (rs a) (opv rs b))) ->
 (forall c w sg d a b, i = ISetp None c w sg d a b ->
   agree d rs rs' /\ rs' d = bool_z (cmp_eval c (extend w sg (rs a)) (extend w sg (opv rs b)))) ->
 state_ok F (S t) (arithmetic_relations F st out i) rs'.
Proof.
 intros t st out rs rs' i Hst Hrs Hrs' Hout Hsem Hcmp.
 destruct i; try exact Hout.
 { unfold arithmetic_relations.
 destruct g as [gg |]; [exact Hout |].
 destruct (negb (Nat.eqb w 64)) eqn:Ew; [exact Hout |].
 apply negb_false_iff in Ew. apply Nat.eqb_eq in Ew. subst w.
 destruct (Hsem o d a b eq_refl) as [Hd [Ha Hv]].
 destruct (negb (Pos.eqb d a)) eqn:Ed.
 - destruct o; try exact Hout. destruct b as [src | z]; [| exact Hout].
   destruct (Pos.eqb d src && Z.eqb (wd F a) 64) eqn:Es; [| exact Hout].
   apply andb_true_iff in Es. destruct Es as [Es _]. apply Pos.eqb_eq in Es. subst src.
   apply (affine_step_cons_sound t st out rs rs' d _ Hst Hrs Hrs' Ha); [| exact Hout].
   change (eqm 64 (rs' d) (rs d + (1 * rs a + 0 + 0))).
   cbn [binop_eval opv] in Hv. rewrite Hv. unfold eqm, trunc.
   rewrite Z.mod_mod by discriminate. f_equal. ring.
 - apply negb_false_iff in Ed. apply Pos.eqb_eq in Ed. subst a.
   destruct b as [src | z].
   + destruct (Z.eqb (wd F src) 64) eqn:Es; [| exact Hout].
     destruct o; try exact Hout;
       apply (affine_step_cons_sound t st out rs rs' d _ Hst Hrs Hrs' Ha); try exact Hout.
     * change (eqm 64 (rs' d) (rs d + (1 * rs src + 0 + 0))).
       cbn [binop_eval opv] in Hv. rewrite Hv. unfold eqm, trunc.
       rewrite Z.mod_mod by discriminate. f_equal. ring.
     * change (eqm 64 (rs' d) (rs d + (-1 * rs src + 0 + 0))).
       cbn [binop_eval opv] in Hv. rewrite Hv. unfold eqm, trunc.
       rewrite Z.mod_mod by discriminate. f_equal. ring.
   + destruct o; exact Hout. }
 unfold arithmetic_relations. destruct g as [gg |]; [exact Hout |].
 destruct (Hcmp c w signed d a b eq_refl) as [Ha Hv].
 apply (complement_cons_sound t st out rs rs' d {|cc:=c;cw:=w;csg:=signed;ca:=a;cb:=b|} Hst Hrs Hrs' Ha Hv Hout).

Qed.

Lemma fold_predicate_sound : forall t st rs i,
 regs_ok F E rs -> state_ok F t st rs -> state_ok F t (fold_predicate F st i) rs.
Proof.
 intros t st rs i Hrs Hst.
 unfold fold_predicate.
 destruct (negb (foldable_predicate F st i)); [exact Hst |].
 destruct i as [g d a | g d z | g o w d a b | g w d a b c | g signed d a b c | g to from signed d a
                 | g w d a p ln | g c w signed d a b | g d a b | g d a b | g is_min signed w d a b | g w d a b p
                 | g gen bits d a off | g k gen a off width v fr ds | g ds | g target | g | | g rm sg w d a b
                 | g bits d a off | g bits d a off fr | g a off width v fr | vs | g a off width | g d a b];
 try exact Hst;
 destruct g as [g |]; try exact Hst;
 cbn [fold_predicate];
 destruct (negb (Z.eqb (wd F d) 1)) eqn:Hw; try exact Hst;
 apply negb_false_iff, Z.eqb_eq in Hw;
 destruct (get st d) as [[f al vr | al | cs exact | cs exact] |]; try exact Hst;
 destruct (refine F st d true) as [s |] eqn:Et.
 all: try (exact (put_impossible_predicate F E HE t rs Hrs st d true Hst Hw Et)).
 all: destruct (refine F st d false) as [s' |] eqn:Ef; try exact Hst;
 exact (put_impossible_predicate F E HE t rs Hrs st d false Hst Hw Ef).
Qed.

Lemma add_bound_ok : forall t rs st k,
 regs_ok F E rs -> state_ok F t st rs -> cons_ok F rs k -> state_ok F t (add_bound st k) rs.
Proof.
 intros t rs st [[ts lo] hi] Hrs Hst Hk. unfold add_bound.
 match goal with |- context [if ?b then _ else _] => destruct b end;
   [exact Hst | exact (add_cons_ok F E HE t rs Hrs st _ Hst Hk)].
Qed.

(* The sum of a register's value or signed reading less another
   register's value lies within 2^65 of zero. *)
Lemma unit_pair_range : forall rs x e y, regs_ok F E rs -> (e = EX \/ e = ES) ->
  - 2 ^ 65 <= tsum wdn rs (sorted_terms [((x, e), 1); ((y, EX), -1)]) <= 2 ^ 65.
Proof.
 intros rs x e y Hrs He. rewrite (tsum_sorted_terms F E HE 0 rs Hrs). cbn [tsum].
 pose proof (rs_range F E rs Hrs x) as Rx. pose proof (rs_range F E rs Hrs y) as Ry.
 pose proof (wd_range F E HE x) as Wx. pose proof (wd_range F E HE y) as Wy.
 assert (Px : 2 ^ Z.of_nat (wdn x) <= 2 ^ 64) by (apply Z.pow_le_mono_r; lia).
 assert (Py : 2 ^ Z.of_nat (wdn y) <= 2 ^ 64) by (apply Z.pow_le_mono_r; lia).
 destruct He as [-> | ->]; cbn [atom_val].
 - lia.
 - pose proof (sext_range (wdn x) (rs x) ltac:(lia)) as Sx.
   assert (Qx : 2 ^ (Z.of_nat (wdn x) - 1) <= 2 ^ 64) by (apply Z.pow_le_mono_r; lia).
   lia.
Qed.

Lemma upper_cons_ok : forall rs ts hi, regs_ok F E rs ->
 - 2 ^ 66 <= tsum wdn rs ts <= hi -> cons_ok F rs (upper_cons F ts hi).
Proof.
 intros rs ts hi Hrs H. unfold upper_cons. apply (clip_constraint_sound F E HE 0 rs Hrs). exact H.
Qed.

Lemma lower_cons_ok : forall rs ts lo, regs_ok F E rs ->
 lo <= tsum wdn rs ts <= 2 ^ 66 -> cons_ok F rs (lower_cons F ts lo).
Proof.
 intros rs ts lo Hrs H. unfold lower_cons. apply (clip_constraint_sound F E HE 0 rs Hrs). exact H.
Qed.

Lemma scalar_bound_sound : forall t rs st d n,
 regs_ok F E rs -> state_ok F t st rs -> state_ok F t (scalar_bound F st d n) rs.
Proof.
 intros t rs st d n Hrs Hst. unfold scalar_bound.
 destruct (check_range F st _) as [[lo hi] |] eqn:Er; [| exact Hst].
 pose proof (check_range_sound F E HE t rs Hrs st _ lo hi 0 Hst Er ltac:(cbn; lia)) as R.
 cbn [form_of_terms ft] in R. rewrite Z.add_0_r in R.
 pose proof (unit_pair_range rs d EX n Hrs (or_introl eq_refl)) as T.
 assert (Hside : state_ok F t (if hi <? 2 ^ 63 then add_bound st (upper_cons F (sorted_terms [((d, EX), 1); ((n, EX), -1)]) hi) else st) rs).
 { destruct (hi <? 2 ^ 63); [| exact Hst]. apply (add_bound_ok t rs st _ Hrs Hst).
   apply upper_cons_ok; [exact Hrs |]. assert (P : 2 ^ 65 <= 2 ^ 66) by (apply Z.pow_le_mono_r; lia). lia. }
 destruct (- 2 ^ 63 <? lo); [| exact Hside].
 apply (add_bound_ok t rs _ _ Hrs Hside). apply lower_cons_ok; [exact Hrs |].
 assert (P : 2 ^ 65 <= 2 ^ 66) by (apply Z.pow_le_mono_r; lia). lia.
Qed.

(* Two values that agree modulo 2^n have the same signed reading. *)
Lemma sext_eqm_eq : forall n u v, eqm (Z.of_nat n) u v -> sext n u = sext n v.
Proof. intros n u v H. unfold sext, trunc. unfold eqm in H. rewrite H. reflexivity. Qed.

Lemma offset_bound_sound : forall t rs' a d c value signed out k,
 regs_ok F E rs' -> state_ok F t out rs' -> cons_ok F rs' k -> wd F d = 64 -> wd F a = 64 ->
 0 <= rs' a < 2 ^ 63 ->
 (value = true -> rs' d = rs' a + c) -> (signed = true -> sext 64 (rs' d) = rs' a + c) ->
 state_ok F t (offset_bound F a d c value signed out k) rs'.
Proof.
 intros t rs' a d c value signed out [[ts lo] hi] Hrs' Hout Hk Hd Ha Har Hv Hs.
 assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
 assert (Han : wdn a = 64%nat) by (unfold wd in Ha; lia).
 assert (Hread : forall e, match e with EX | ES => true | _ => false end = true -> atom_val wdn rs' (a, e) = rs' a).
 { intros [| | | k | p q] He; try discriminate; cbn [atom_val]; [reflexivity |].
   rewrite Han. apply sext_id; [lia | change (Z.of_nat 64 - 1) with 63; lia]. }
 unfold offset_bound.
 (* the scalar n with a - n in [lo, hi] *)
 assert (Hn : forall n, match ts with
     | [((u, eu), cu); ((v, ev), cv)] =>
         if Pos.eqb u a && Z.eqb cu 1 && match eu with EX | ES => true | _ => false end && Z.eqb cv (-1) &&
            fixed_scalar_term F ((v, ev), cv) && match ev with EX => true | _ => false end
         then Some v
         else if Pos.eqb v a && Z.eqb cv 1 && match ev with EX | ES => true | _ => false end && Z.eqb cu (-1) &&
                 fixed_scalar_term F ((u, eu), cu) && match eu with EX => true | _ => false end
         then Some u else None
     | _ => None end = Some n -> lo <= rs' a - rs' n <= hi).
 { intros n Hm. cbn [cons_ok] in Hk.
   destruct ts as [| [[u eu] cu] [| [[v ev] cv] [|]]]; try discriminate.
   cbn [tsum] in Hk.
   destruct (Pos.eqb u a && Z.eqb cu 1 && match eu with EX | ES => true | _ => false end && Z.eqb cv (-1) &&
             fixed_scalar_term F ((v, ev), cv) && match ev with EX => true | _ => false end) eqn:E1.
   - injection Hm as <-. repeat rewrite andb_true_iff in E1. destruct E1 as [[[[[Eu Ec] Ee] Ecv] _] Eev].
     apply Pos.eqb_eq in Eu. subst u. apply Z.eqb_eq in Ec, Ecv. subst cu cv.
     destruct ev; try discriminate. rewrite (Hread eu Ee) in Hk. cbn [atom_val] in Hk. lia.
   - destruct (Pos.eqb v a && Z.eqb cv 1 && match ev with EX | ES => true | _ => false end && Z.eqb cu (-1) &&
               fixed_scalar_term F ((u, eu), cu) && match eu with EX => true | _ => false end) eqn:E2; [| discriminate].
     injection Hm as <-. repeat rewrite andb_true_iff in E2. destruct E2 as [[[[[Ev Ec] Ee] Ecu] _] Eeu].
     apply Pos.eqb_eq in Ev. subst v. apply Z.eqb_eq in Ec, Ecu. subst cu cv.
     destruct eu; try discriminate. rewrite (Hread ev Ee) in Hk. cbn [atom_val] in Hk. lia. }
 destruct (match ts with
     | [((u, eu), cu); ((v, ev), cv)] => _ | _ => None end) as [n |] eqn:Em; [| exact Hout].
 specialize (Hn n eq_refl).
 assert (Hval : state_ok F t (if value then add_bound out (clip_constraint F (sorted_terms [((d, EX), 1); ((n, EX), -1)], lo + c, hi + c)) else out) rs').
 { destruct value; [| exact Hout]. apply (add_bound_ok t rs' out _ Hrs' Hout).
   apply (clip_constraint_sound F E HE t rs' Hrs'). cbn [cons_ok].
   rewrite (tsum_sorted_terms F E HE t rs' Hrs'). cbn [tsum atom_val]. rewrite (Hv eq_refl). lia. }
 destruct signed; [| exact Hval].
 apply (add_bound_ok t rs' _ _ Hrs' Hval). apply (clip_constraint_sound F E HE t rs' Hrs'). cbn [cons_ok].
 rewrite (tsum_sorted_terms F E HE t rs' Hrs'). cbn [tsum atom_val]. rewrite Hdn, (Hs eq_refl). lia.
Qed.

Lemma sum_bounds_sound : forall t out rs' i,
 regs_ok F E rs' -> state_ok F t out rs' ->
 (forall d a c, i = IBin None BAdd 64 d a (OImm c) -> d <> a ->
    wd F d = 64 -> rs' d = trunc 64 (rs' a + c)) ->
 state_ok F t (sum_bounds F out i) rs'.
Proof.
 intros t out rs' i Hrs' Hout Hsem. unfold sum_bounds.
 destruct i; try exact Hout. destruct g as [g |]; [exact Hout |].
 destruct o; try exact Hout.
 destruct b as [r | c].
 - destruct (negb (Nat.eqb w 64) || Pos.eqb d a || Pos.eqb d r || negb (Z.eqb (wd F d) 64)); [exact Hout |].
   destruct (get out d) as [[f al vr | al | cs ex | cs ex] |]; try exact Hout.
   destruct (negb _); [exact Hout |].
   generalize (related_scalars F out (map (fun t => fst (fst t)) (ft f))). intros l.
   revert out Hout. induction l as [| n l IH]; intros out Hout; [exact Hout |].
   cbn [fold_left]. apply IH. exact (scalar_bound_sound t rs' out d n Hrs' Hout).
 - destruct (negb (Nat.eqb w 64) || Pos.eqb d a || is_fixed F a || negb (Z.eqb (wd F d) 64 && Z.eqb (wd F a) 64) ||
             negb (midpoint out a)) eqn:Ec; [exact Hout |].
   apply orb_false_iff in Ec. destruct Ec as [Ec _].
   apply orb_false_iff in Ec. destruct Ec as [Ec Hw]. apply orb_false_iff in Ec. destruct Ec as [Ec _].
   apply orb_false_iff in Ec. destruct Ec as [H64 Hda]. apply negb_false_iff, Nat.eqb_eq in H64. subst w.
   apply Pos.eqb_neq in Hda. apply negb_false_iff in Hw. apply andb_true_iff in Hw. destruct Hw as [Hd Hwa].
   apply Z.eqb_eq in Hd, Hwa.
   destruct (reg_range F out a) as [[x y] |] eqn:Er; [| exact Hout].
   destruct (negb ((0 <=? x) && (y <? 2 ^ 63))) eqn:Eb; [exact Hout |].
   apply negb_false_iff, andb_true_iff in Eb. destruct Eb as [Hx Hy]. apply Z.leb_le in Hx. apply Z.ltb_lt in Hy.
   pose proof (reg_range_sound F E HE t out rs' Hout Hrs' a x y Er) as Ha.
   pose proof (Hsem d a c eq_refl Hda Hd) as Hv.
   assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
   set (c' := extend 64 true c).
   assert (Hcm : eqm 64 (rs' a + c) (rs' a + c')).
   { unfold eqm. rewrite Zplus_mod, (Zplus_mod (rs' a) c').
     pose proof (extend_eqm 64 true c) as Hc. unfold eqm in Hc. fold c' in Hc. change (Z.of_nat 64) with 64 in Hc. rewrite Hc. reflexivity. }
   assert (Hvalue : (0 <=? x + c') && (y + c' <? 2 ^ 64) = true -> rs' d = rs' a + c').
   { intros Hb. apply andb_true_iff in Hb. destruct Hb as [H1 H2]. apply Z.leb_le in H1. apply Z.ltb_lt in H2.
     rewrite Hv. unfold trunc. change (2 ^ Z.of_nat 64) with (2 ^ 64). unfold eqm in Hcm. rewrite Hcm.
     apply Z.mod_small. lia. }
   assert (Hsigned : (- 2 ^ 63 <=? x + c') && (y + c' <? 2 ^ 63) && (x + c' <? 0) = true -> sext 64 (rs' d) = rs' a + c').
   { intros Hb. repeat rewrite andb_true_iff in Hb. destruct Hb as [[H1 H2] _]. apply Z.leb_le in H1. apply Z.ltb_lt in H2.
     rewrite Hv, sext_trunc, (sext_eqm_eq 64 (rs' a + c) (rs' a + c') Hcm).
     apply sext_id; [lia | change (Z.of_nat 64 - 1) with 63; lia]. }
   assert (Hbase : state_ok F t (if (- 2 ^ 63 <=? x + c') && (y + c' <? 2 ^ 63) && (x + c' <? 0)
       then add_bound out (sorted_terms [((d, ES), 1); ((a, EX), -1)], c', c') else out) rs').
   { destruct ((- 2 ^ 63 <=? x + c') && (y + c' <? 2 ^ 63) && (x + c' <? 0)) eqn:Es; [| exact Hout].
     apply (add_bound_ok t rs' out _ Hrs' Hout). cbn [cons_ok].
     rewrite (tsum_sorted_terms F E HE t rs' Hrs'). cbn [tsum atom_val]. rewrite Hdn, (Hsigned eq_refl). lia. }
   remember (if (- 2 ^ 63 <=? x + c') && (y + c' <? 2 ^ 63) && (x + c' <? 0)
       then add_bound out (sorted_terms [((d, ES), 1); ((a, EX), -1)], c', c') else out) as base.
   assert (Hks : Forall (cons_ok F rs') (acons base)) by exact (proj2 Hbase).
   clear Heqbase. revert Hks. generalize (acons base) as ks. intros ks Hk. revert base Hbase.
   induction Hk as [| k ks Hk1 Hks' IH]; intros base Hbase; cbn [fold_left]; [exact Hbase |].
   apply IH. eapply (offset_bound_sound t rs' a d c' _ _ base k Hrs' Hbase Hk1 Hd Hwa); [lia | exact Hvalue | exact Hsigned].
Qed.

Lemma selection_bounds_sound : forall t st out rs rs' i,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> state_ok F (S t) out rs' ->
 (forall d a b p, i = ISelp None 64 d (OReg a) (OReg b) p ->
    agree d rs rs' /\ rs' d = trunc (wdn d) (if Z.eqb (rs p) 0 then rs b else rs a)) ->
 state_ok F (S t) (selection_bounds F st out i) rs'.
Proof.
 intros t st out rs rs' i Hst Hrs Hrs' Hout Hsem. unfold selection_bounds.
 destruct i; try exact Hout. destruct g as [g |]; [exact Hout |].
 destruct a as [a | za]; [| exact Hout]. destruct b as [b | zb]; [| exact Hout].
 destruct (negb (Nat.eqb w 64 && Z.eqb (wd F d) 64 && Z.eqb (wd F a) 64 && Z.eqb (wd F b) 64 &&
                 (Pos.eqb d a || Pos.eqb d b))) eqn:Ew; [exact Hout |].
 apply negb_false_iff in Ew. rewrite !andb_true_iff in Ew. destruct Ew as [[[[H64 Hd] Hwa] Hwb] _].
 apply Nat.eqb_eq in H64. subst w.
 apply Z.eqb_eq in Hd, Hwa, Hwb.
 assert (Hdn : wdn d = 64%nat) by (unfold wd in Hd; lia).
 assert (Han : wdn a = 64%nat) by (unfold wd in Hwa; lia).
 assert (Hbn : wdn b = 64%nat) by (unfold wd in Hwb; lia).
 destruct (Hsem d a b p eq_refl) as [Ha Hv]. clear Hsem.
 destruct (form_scalars F st [a; b]) as [| n0 more] eqn:El; [exact Hout |].
 rewrite <- El. clear El.
 destruct (refine_selection F st p true) as [sa |] eqn:Ea; [| exact Hout].
 destruct (refine_selection F st p false) as [sb |] eqn:Eb; [| exact Hout].
 (* in the branch the run takes, d holds that branch's operand, whose state holds *)
 assert (Hcase : forall e, (match e with EX | EZ | ES => True | _ => False end) ->
           (atom_val wdn rs' (d, e) = atom_val wdn rs (a, e) /\ state_ok F t sa rs) \/
           (atom_val wdn rs' (d, e) = atom_val wdn rs (b, e) /\ state_ok F t sb rs)).
 { intros e He.
   assert (Hval : forall x, wdn x = 64%nat -> 0 <= rs x < 2 ^ 64) by
     (intros x Hx; pose proof (rs_range F E rs Hrs x) as R; rewrite Hx in R; exact R).
   destruct (Z.eqb (rs p) 0) eqn:Ep.
   - right. destruct (refine_selection_sound t st rs p false Hst Hrs ltac:(rewrite Ep; reflexivity)) as [s0 [Es Hok]].
     rewrite Eb in Es. injection Es as <-. split; [| exact Hok].
     destruct e; try contradiction; cbn [atom_val];
       rewrite Hv, Hdn, (trunc_small 64 (rs b)) by (apply Hval; exact Hbn); rewrite ?Hbn; reflexivity.
   - left. destruct (refine_selection_sound t st rs p true Hst Hrs ltac:(rewrite Ep; reflexivity)) as [s0 [Es Hok]].
     rewrite Ea in Es. injection Es as <-. split; [| exact Hok].
     destruct e; try contradiction; cbn [atom_val];
       rewrite Hv, Hdn, (trunc_small 64 (rs a)) by (apply Hval; exact Han); rewrite ?Han; reflexivity. }
 generalize (form_scalars F st [a; b]). intros l. revert out Hout.
 induction l as [| n l IH]; intros out Hout; [exact Hout |].
 match goal with |- state_ok ?u ?v (fold_left ?f (?x :: ?xs) ?o) ?c =>
   change (state_ok u v (fold_left f xs (f o x)) c) end.
 apply IH. cbv beta. destruct (Pos.eqb n d) eqn:End; [exact Hout |]. apply Pos.eqb_neq in End.
 assert (Hn : rs' n = rs n) by (apply Ha; exact End).
 match goal with |- state_ok _ _ (fold_left ?f [EX; ES] ?o) _ =>
   assert (Hstep : forall e out0, (e = EX \/ e = ES) -> state_ok F (S t) out0 rs' ->
                   state_ok F (S t) (f out0 e) rs') end.
 { intros e out0 He Hout0. cbv beta zeta.
   destruct (check_range F sa _) as [[l1 h1] |] eqn:E1; [| exact Hout0].
   destruct (check_range F sb _) as [[l2 h2] |] eqn:E2; [| exact Hout0].
   assert (Hb : Z.min l1 l2 <= tsum wdn rs' (sorted_terms [((d, e), 1); ((n, EX), -1)]) <= Z.max h1 h2).
   { rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'). cbn [tsum].
     change (atom_val wdn rs' (n, EX)) with (rs' n). rewrite Hn.
   assert (Hk : match e with EX | EZ | ES => True | _ => False end)
     by (destruct He as [-> | ->]; exact I).
   destruct (Hcase e Hk) as [[Hde Hsa] | [Hde Hsb]]; rewrite Hde.
   - pose proof (check_range_sound F E HE t rs Hrs sa _ l1 h1 0 Hsa E1 ltac:(cbn; lia)) as R.
     cbn [form_of_terms ft] in R. rewrite (tsum_sorted_terms F E HE t rs Hrs) in R. cbn [tsum] in R.
     change (atom_val wdn rs (n, EX)) with (rs n) in R. lia.
   - pose proof (check_range_sound F E HE t rs Hrs sb _ l2 h2 0 Hsb E2 ltac:(cbn; lia)) as R.
     cbn [form_of_terms ft] in R. rewrite (tsum_sorted_terms F E HE t rs Hrs) in R. cbn [tsum] in R.
     change (atom_val wdn rs (n, EX)) with (rs n) in R. lia. }
   pose proof (unit_pair_range rs' d e n Hrs' He) as T.
   assert (Hup : state_ok F (S t) (if Z.max h1 h2 <? 2 ^ 63 then add_bound out0
       (upper_cons F (sorted_terms [((d, e), 1); ((n, EX), -1)]) (Z.max h1 h2)) else out0) rs').
   { destruct (Z.max h1 h2 <? 2 ^ 63); [| exact Hout0].
     apply (add_bound_ok (S t) rs' out0 _ Hrs' Hout0). apply upper_cons_ok; [exact Hrs' |].
     assert (P : 2 ^ 65 <= 2 ^ 66) by (apply Z.pow_le_mono_r; lia). lia. }
   destruct (- 2 ^ 63 <? Z.min l1 l2); [| exact Hup].
   apply (add_bound_ok (S t) rs' _ _ Hrs' Hup). apply lower_cons_ok; [exact Hrs' |].
   assert (P : 2 ^ 65 <= 2 ^ 66) by (apply Z.pow_le_mono_r; lia).
   exact (conj (proj1 Hb) (Z.le_trans _ _ _ (proj2 T) P)). }
 cbn [fold_left]. apply Hstep; [right; reflexivity |]. apply Hstep; [left; reflexivity | exact Hout].
Qed.

Lemma half_bounds_sound : forall t st out rs rs' i,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> state_ok F (S t) out rs' ->
 (forall op d a k, i = IBin None op 64 d a (OImm k) ->
    agree d rs rs' /\ rs' d = trunc 64 (binop_eval op 64 (rs a) k)) ->
 state_ok F (S t) (half_bounds F st out i) rs'.
Proof.
 intros t st out rs rs' i Hst Hrs Hrs' Hout Hsem. unfold half_bounds.
 destruct i; try exact Hout. destruct g as [g |]; [exact Hout |].
 destruct b as [rb | k]; [exact Hout |].
 destruct (negb (match o with BShr | BShrs => true | _ => false end && Nat.eqb w 64 && Z.eqb k 1 &&
                 Z.eqb (wd F d) 64 && Z.eqb (wd F a) 64)) eqn:Ec; [exact Hout |].
 apply negb_false_iff in Ec. repeat rewrite andb_true_iff in Ec. destruct Ec as [[[[Hop H64] Hk] Hd] Hwa].
 apply Nat.eqb_eq in H64. subst w. apply Z.eqb_eq in Hk, Hd, Hwa. subst k.
 destruct (Hsem o d a 1 eq_refl) as [Ha Hv]. clear Hsem.
 set (sg := match o with BShrs => true | _ => false end) in *.
 set (f := cmp_form F st (OReg a) 64 sg) in *.
 match goal with |- context [if ?c then out else _] => destruct c eqn:Eshape; [exact Hout |] end.
 apply orb_false_iff in Eshape. destruct Eshape as [Eshape Hkeep].
 apply orb_false_iff in Eshape. destruct Eshape as [_ Hpoint].
 apply negb_false_iff, Z.eqb_eq in Hpoint.
 destruct (cmp_range F st (OReg a) 64 sg) as [[lo hi] |] eqn:Er; [| exact Hout].
 destruct (0 <=? lo) eqn:Hlo; [| exact Hout]. apply Z.leb_le in Hlo.
 assert (Hw : 1 <= 64 <= 64) by lia.
 assert (Hoa : owidth F (OReg a) 64) by (intros r Hr; injection Hr as <-; exact Hwa).
 destruct (cmp_exact F E HE t rs Hrs st (OReg a) 64 sg lo hi Hst Hw Hoa Er) as [e [He Hf]].
 pose proof (cmp_range_sound F E HE t rs Hrs st (OReg a) 64 sg lo hi Hst Hw Hoa Er) as Hrange.
 fold f in He, Hf. assert (Heq : e = flo f) by lia. subst e.
 set (x := cval rs (OReg a) 64 sg) in *.
 assert (Hx : 0 <= x < 2 ^ 64).
 { pose proof (cval_range F E HE t rs Hrs (OReg a) 64 sg Hw) as Hb.
   fold x in Hb. unfold hi_ok in Hb. destruct sg; lia. }
 assert (Hdval : rs' d = x / 2).
 { rewrite Hv. assert (Hb : binop_eval o 64 (rs a) 1 = x / 2).
   { unfold x, cval, extend, sg. cbn [opv].
     destruct o; try discriminate; cbn [binop_eval]; unfold shamt; reflexivity. }
   rewrite Hb. apply trunc_small. split; [apply Z.div_pos; lia | apply Z.div_lt_upper_bound; lia]. }
 apply (add_bound_ok (S t) rs' out _ Hrs' Hout).
 cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app.
 cbn [tsum atom_val]. rewrite (tsum_agree wdn d rs rs' (ft f) Ha Hkeep), Hdval.
 pose proof (sub_div_mono 2 lo x ltac:(lia) ltac:(lia)) as H1.
 pose proof (sub_div_mono 2 x hi ltac:(lia) ltac:(lia)) as H2. lia.
Qed.

Theorem transfer_sound : forall t s s' st i,
 fetch prog (blk s) (idx s) = Some i -> stepF t s = Some s' ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) -> wf_instr F i = true ->
 state_ok F (S t) (transfer F st i) (regs s') /\ regs_ok F E (regs s').
Proof.
 intros t s s' st i Hf Hstep Hst Hrs Hwf.
 unfold transfer. destruct (is_bottom st) eqn:Eb.
 { exfalso. exact (bottom_not_ok F t st (regs s) Eb Hst). }
 destruct (transfer_live_sound t s s' st i Hf Hstep Hst Hrs Hwf) as [Hout Hrs'].
 split; [| exact Hrs'].
 apply (fold_predicate_sound (S t) _ (regs s') i Hrs').
 apply (selection_bounds_sound t st _ (regs s) (regs s') i Hst Hrs Hrs').
 2: { intros d a b p Ei. subst i.
      pose proof (step_regs_dest t s s' (ISelp None 64 d (OReg a) (OReg b) p) d Hf eq_refl Hstep) as Hnew.
      cbn [guard_of holds result opv] in Hnew. split.
      - rewrite Hnew. apply upd_agree.
      - rewrite Hnew. unfold upd. rewrite Pos.eqb_refl. reflexivity. }
 apply (sum_bounds_sound (S t) _ (regs s') i Hrs').
 2: { intros d a c Ei Hda Hd. subst i.
      pose proof (step_regs_dest t s s' (IBin None BAdd 64 d a (OImm c)) d Hf eq_refl Hstep) as Hnew.
      cbn [guard_of holds result binop_eval opv] in Hnew.
      assert (Hdn : wdn d = 64%nat) by (apply Nat2Z.inj; exact Hd).
      rewrite Hnew. unfold upd. rewrite Pos.eqb_refl.
      replace (Pos.eqb a d) with false by (symmetry; apply Pos.eqb_neq; congruence).
      rewrite Hdn. reflexivity. }
 apply (half_bounds_sound t st _ (regs s) (regs s') i Hst Hrs Hrs').
 2: { intros op d a k Ei. subst i.
      pose proof (step_regs_dest t s s' (IBin None op 64 d a (OImm k)) d Hf eq_refl Hstep) as Hnew.
      assert (Hd : wd F d = 64).
      { unfold wf_instr in Hwf. cbn [dest guard_of] in Hwf. rewrite !andb_true_iff in Hwf.
        apply Z.eqb_eq. tauto. }
      assert (Hdn : wdn d = 64%nat) by (apply Nat2Z.inj; exact Hd).
      cbn [guard_of holds result opv] in Hnew. split.
      - rewrite Hnew. apply upd_agree.
      - rewrite Hnew. unfold upd. rewrite Pos.eqb_refl, Hdn. reflexivity. }
 apply (arithmetic_relations_sound t st _ (regs s) (regs s') i Hst Hrs Hrs' Hout).
 { intros op d a b Ei. subst i.
 assert (Hd : wd F d = 64).
 { unfold wf_instr in Hwf. cbn [dest guard_of] in Hwf. rewrite !andb_true_iff in Hwf.
   apply Z.eqb_eq. tauto. }
 pose proof (step_regs_dest t s s' (IBin None op 64 d a b) d Hf eq_refl Hstep) as Hnew.
 cbn [guard_of holds result] in Hnew. split; [exact Hd |]. split.
 - rewrite Hnew. apply upd_agree.
 - rewrite Hnew. unfold upd. rewrite Pos.eqb_refl.
   change (trunc (wdn d) (binop_eval op 64 (regs s a) (opv (regs s) b)) = trunc 64 (binop_eval op 64 (regs s a) (opv (regs s) b))).
   assert (Hdn : wdn d = 64%nat).
   { apply Nat2Z.inj. change (wd F d = 64). exact Hd. }
   rewrite Hdn. reflexivity. }
 intros c w sg d a b Ei. subst i.
 assert (Hd : wd F d = 1).
 { unfold wf_instr in Hwf. cbn [dest guard_of] in Hwf. rewrite !andb_true_iff in Hwf.
   apply Z.eqb_eq. tauto. }
 pose proof (step_regs_dest t s s' (ISetp None c w sg d a b) d Hf eq_refl Hstep) as Hnew.
 cbn [guard_of holds result] in Hnew. split.
 - rewrite Hnew. apply upd_agree.
 - rewrite Hnew. unfold upd. rewrite Pos.eqb_refl.
   assert (Hdn : wdn d = 1%nat) by (apply Nat2Z.inj; exact Hd).
   rewrite Hdn. destruct (cmp_eval c (extend w sg (regs s a)) (extend w sg (opv (regs s) b))); reflexivity.

Qed.

End Transfer.

(* Scalar kept-list specifications occur only in equivalence proofs. *)
Definition keep_scalar (F : lfacts) (widen_now:bool)(k:constraint)(s:astate):list constraint :=
 let '(ts,lo,hi):=k in
 match filter(fun k'=>let '(ts',_,_):=k' in terms_eqb ts ts')(acons s) with
 | (_,lo0,hi0)::more=>
   let '(lo',hi'):=fold_left(fun acc k'=>let '(l,h):=acc in let '(_,l',h'):=k' in (Z.max l l',Z.min h h'))more(lo0,hi0) in
   if widen_now && ((lo' <? lo)||(hi <? hi')) then [] else [(ts,Z.min lo lo',Z.max hi hi')]
 | []=>if (cons_holds F) s k then [k] else [] end.

Definition join_kept_raw (F : lfacts) (old out : astate) (widen_now loop_edge : bool) : list constraint :=
  let old := if widen_now then (clip_join_state F) old else old in
  let out := if widen_now then (clip_join_state F) out else out in

  (* a constraint of one side, in the other: the hull with that side's
     constraint of the same terms, or the constraint itself when it follows
     from the other side, as on the entry edge of a loop whose counter
     starts at a constant *)
  let keep := fun (k : constraint) (s : astate) =>
    let '(ts, lo, hi) := k in
    (* the tightest of s's constraints with the same terms *)
    match filter (fun k' => let '(ts', _, _) := k' in terms_eqb ts ts') (acons s) with
    | (_, lo0, hi0) :: more =>
        let '(lo', hi') := fold_left (fun acc k' => let '(l, h) := acc in let '(_, l', h') := k' in (Z.max l l', Z.min h h'))
                                     more (lo0, hi0) in
        if widen_now && ((lo' <? lo) || (hi <? hi')) then [] else [(ts, Z.min lo lo', Z.max hi hi')]
    | [] => if (cons_holds F) s k then [k] else []
    end in
  (* a constraint of one side read through that side's copies, when that
     changes it: a loop whose first test compares a copy of its register,
     and whose back edge copies the next position into it, as the Numba
     front end's range loop, keeps the bound on the register itself. Only a
     join that does not widen proposes them; a later join keeps them as any
     other constraint, so the loop's states converge *)
  let moved := fun (s other : astate) =>
    flat_map (fun k => let '(ts, lo, hi) := k in
                let ts' := sorted_terms ((copy_terms F) s ts) in
                if terms_eqb ts ts' then [] else keep (ts', lo, hi) other) (acons s) in
  (* A forward join can merge a search's zero-iteration exit with its
     loop exit. Infer the cursor sum there too, before branch-local
     intermediates disappear from the common state. *)
  let kept := flat_map (fun k => keep k out) (acons old)
              ++ flat_map (fun k => let '(ts, _, _) := k in
                             if widen_now then [] else if existsb (fun k' => let '(ts', _, _) := k' in terms_eqb ts ts') (acons old) then []
                             else keep k old) (acons out)
              ++ (if widen_now then [] else moved old out ++ moved out old
                    ++ (pair_candidates F) old out ++ (pair_candidates F) out old ++ (if loop_edge then [] else (exit_sum_candidates F) old out ++ (exit_sum_candidates F) out old ++ (sentinel_candidates F) old out ++ (sentinel_candidates F) out old) ++ (if loop_edge then (signed_candidates F) old out ++ (signed_candidates F) out old ++ (triple_candidates F) old out ++ (triple_candidates F) out old ++ (moving_bound_candidates F) old out ++ (moving_bound_candidates F) out old ++ (order_candidates F) old out else []) ++ (bound_candidates F) old out loop_edge ++ (proportion_candidates F) old out ++ (exit_bound_candidates F) old out ++ (start_bound_candidates F) old out) in
  (* registers that step by a constant, and the others *)
kept.


(* The prior predicate-bit proposal is retained only for exactness proofs. *)
Definition join_predicate_bits (F : lfacts) (old out : astate) (p : reg) (j : option fact) : option fact :=
 if negb (Z.eqb ((wd F) p) 1) then j else
 let loses := match j with None | Some (FPred _ false) | Some (FNeg _ false) => true | _ => false end in
 if negb loses then j else
 match (join_bit F) old p,(join_bit F) out p with Some c,Some c' =>
   if Z.eqb c c' then Some (FForm (fconst 1 c) 0 None) else j
 | _,_ => j end.

Section PreparedJoinEquivalence.
Variable F:lfacts.
Lemma keep_job_exact : forall w k s,keep_run F s (check_range F s)(keep_job F w k s)=keep_scalar F w k s.
Proof.
 intros w [[ts lo]hi]s. unfold keep_job,keep_scalar.
 destruct(filter(fun k'=>let '(ts',_,_):=k' in terms_eqb ts ts')(acons s))as[|[[ks l]h]more].
 - cbn [keep_run]. rewrite cons_prepare_exact. reflexivity.
 - destruct (fold_left _ more (l,h));reflexivity.
Qed.

Lemma keep_apply_ext : forall s jobs a b,(forall f,a f=b f)->keep_apply F s a jobs=keep_apply F s b jobs.
Proof.
 intros s jobs a b H. unfold keep_apply. apply flat_map_ext.
 intros [xs|[k j]];[reflexivity|]. destruct j as[answer|[[ts lo]hi]];[reflexivity|].
 cbn [keep_run prepared_run]. rewrite H. reflexivity.
Qed.

Lemma keep_apply_exact : forall w s ks,
 keep_apply F s (check_range F s)(map(fun k=>keep_job F w k s)ks)=flat_map(fun k=>keep_scalar F w k s)ks.
Proof.
 intros w s ks. induction ks as[|k ks IH];[reflexivity|].
 cbn [map keep_apply flat_map]. unfold keep_apply in IH. rewrite keep_job_exact,IH. reflexivity.
Qed.

Lemma prepared_keep_two_exact : forall w s left right,
 prepared_keep_two F w s left right=
 (flat_map(fun k=>keep_scalar F w k s)left,flat_map(fun k=>keep_scalar F w k s)right).
Proof.
 intros. unfold prepared_keep_two.
 destruct(keep_pending(map(fun k=>keep_job F w k s)left)||keep_pending(map(fun k=>keep_job F w k s)right)).
 - assert(Hcheck:forall f,shared_check_cached F s f=check_range F s f).
   {apply shared_check_cached_exact. }
   rewrite (keep_apply_ext s _ _ _ Hcheck),(keep_apply_ext s _ _ _ Hcheck),!keep_apply_exact. reflexivity.
 - rewrite !keep_apply_exact. reflexivity.
Qed.

Lemma moved_jobs_exact:forall s other w ks,
 flat_map(fun k=>keep_scalar F w k other)(moved_jobs F s ks)=
 flat_map(fun k=>let '(ts,lo,hi):=k in let ts':=sorted_terms(copy_terms F s ts)in
 if terms_eqb ts ts' then [] else keep_scalar F w (ts',lo,hi)other)ks.
Proof.
 intros s other w ks. induction ks as[|[[ts lo]hi]ks IH];[reflexivity|].
 unfold moved_jobs in IH. cbn [moved_jobs flat_map]. rewrite flat_map_app. cbn [flat_map].
 destruct(terms_eqb ts(sorted_terms(copy_terms F s ts)));cbn [flat_map app].
 - exact IH.
 - rewrite app_nil_r. f_equal. exact IH.
Qed.

Lemma novel_jobs_exact:forall old w ks,
 flat_map(fun k=>keep_scalar F w k old)(novel_jobs old ks)=
 flat_map(fun k=>let '(ts,_,_):=k in if existsb(fun k'=>let '(ts',_,_):=k' in terms_eqb ts ts')(acons old)then[]else keep_scalar F w k old)ks.
Proof.
 intros old w ks. induction ks as[|[[ts lo]hi]ks IH];[reflexivity|].
 unfold novel_jobs in IH. cbn [novel_jobs filter flat_map]. destruct(existsb(fun k'=>let '(ts',_,_):=k' in terms_eqb ts ts')(acons old));cbn [negb filter flat_map app].
 - exact IH.
 - f_equal. exact IH.
Qed.

Lemma join_kept_prepared_exact:forall old out w e,
 join_kept_prepared F old out w e=join_kept_raw F old out w e.
Proof.
 intros old out w e. unfold join_kept_prepared. rewrite !prepared_keep_two_exact.
 destruct w;cbn [fst snd].
 - unfold join_kept_raw. cbn [fst snd].
   unfold keep_scalar. rewrite !app_nil_r.
   assert(H:forall(xs:list constraint),flat_map(fun k=>let '(ts,_,_):=k in @nil constraint)xs=[]).
   {intros xs. induction xs as[|[[ts lo]hi]xs IH];cbn;auto. } rewrite H. rewrite !app_nil_r. reflexivity.
 - rewrite !moved_jobs_exact,novel_jobs_exact.
   unfold join_kept_raw,keep_scalar. reflexivity.
Qed.

Lemma join_bit_impossible : forall st p,join_bit_possible F st p=false->join_bit F st p=None.
Proof.
 intros st p H. unfold join_bit_possible in H. unfold join_bit,predicate_constant.
 destruct(known_bit F st p);[discriminate|].
 destruct(negb(Z.eqb(wd F p)1));[reflexivity|].
 destruct(get st p)as[f|];[destruct f as [form al vr|k|cs exact|cs exact]|];try reflexivity;
 destruct exact;congruence.
Qed.

Lemma join_predicate_bits_ready_exact : forall old out p j,
 join_predicate_bits_ready F old out p j=join_predicate_bits F old out p j.
Proof.
 intros old out p j. unfold join_predicate_bits_ready,join_predicate_bits.
 destruct(negb(Z.eqb(wd F p)1));[reflexivity|].
 destruct(negb(match j with None|Some(FPred _ false)|Some(FNeg _ false)=>true|_=>false end));[reflexivity|].
 destruct(join_bit_possible F old p)eqn:Ho;destruct(join_bit_possible F out p)eqn:Hi;cbn.
 - reflexivity.
 - rewrite(join_bit_impossible out p Hi). destruct(join_bit F old p);reflexivity.
 - rewrite(join_bit_impossible old p Ho). reflexivity.
 - rewrite(join_bit_impossible old p Ho). reflexivity.
Qed.

End PreparedJoinEquivalence.
