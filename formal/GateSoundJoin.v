(*
  Soundness of the order of facts and of their join:
  fact_le x g means every value x allows g allows; the join of two facts
  allows every value either allows; and state_le a b means every register
  state a describes b describes.
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
Require Import OpGate.GateSoundAccess.
Require Import OpGate.GateSoundRefine.

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

(* Scalar expansion retained only as the soundness proof specification.
   GateDomain has one active constraint-classification implementation. *)
Definition cons_holds_spec (F : lfacts) (a : astate) (k : constraint) : bool :=
  let '(ts0, lo, hi) := k in
  let ts := (copy_terms F) a ts0 in
  let read := fun (_ : unit) => (canon_terms F) a ts in
  if short_any (fun k' => let '(ts', lo', hi') := k' in
                        (if (lo <=? lo') && (hi' <=? hi) then
                           if terms_eqb ts0 ts' then true else
                           if terms_eqb (value_terms ts) (value_terms ((copy_terms F) a ts'))
                           then terms_eqb (read tt) ((canon_terms F) a ((copy_terms F) a ts')) else false
                         else false)
                        && (lo <=? lo') && (hi' <=? hi)) (acons a)
  then true else
  if negb (forallb (fun t => (is_fixed F) (fst (fst t)) || is_some (get a (fst (fst t)))) ts0) then false else
  match (near_constraint F) a ts lo hi with
  | Some answer => answer
  | None =>
  match (if (cursor_sum F) ts || (Z.eqb lo hi && (mixed_cursor_sum F) ts) || (affine_cursor_bound F) ts || (pair_cursor F) a ts then (check_range F) a else (form_range_near F) a) (form_of_terms ts) with
  | None => false
  | Some (p, q) =>
      if (lo <=? p) && (q <=? hi) then true else
      match (form_range_near F) a ((normalize F) a (form_of_terms ts)) with
      | Some (x, y) => (lo <=? x) && (y <=? hi) && (Z.max q y - Z.min p x <? 2 ^ 64)
      | None => false
      end
  end
  end.

Lemma cons_holds_exact : forall F a k,
 cons_holds F a k = cons_holds_spec F a k.
Proof.
 intros F a [[ts lo]hi]. unfold cons_holds,cons_holds_spec,cons_prepare.
 cbv beta iota zeta.
 fold (near_constraint F a (copy_terms F a ts) lo hi).
 repeat match goal with
 | |- context[if ?b then _ else _] => destruct b eqn:?
 | |- context[match near_constraint _ _ _ _ _ with _ => _ end] =>
     destruct (near_constraint F a (copy_terms F a ts) lo hi) eqn:?
 end; cbn [prepared_run cons_finish]; reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* Equality tests and lists.                                           *)

Lemma some_inj : forall (A : Type) (a b : A), Some a = Some b -> b = a.
Proof. intros A a b H. injection H as H. symmetry. exact H. Qed.

Lemma filter_all : forall (A : Type) (p : A -> bool) l, (forall a, In a l -> p a = true) -> filter p l = l.
Proof.
  intros A p l H. induction l as [| a l IH]; cbn [filter]; [reflexivity |].
  rewrite (H a (or_introl eq_refl)). f_equal. apply IH. intros b Hb. apply H. right. exact Hb.
Qed.

Lemma fold_and : forall (A : Type) (P : A -> bool) l b,
    fold_left (fun acc p => acc && P p) l b = true -> b = true /\ (forall p, In p l -> P p = true).
Proof.
  intros A P l. induction l as [| a l IH]; intros b H; cbn [fold_left] in H.
  - split; [exact H | intros p []].
  - destruct (IH _ H) as [H1 H2]. apply andb_true_iff in H1. destruct H1 as [Hb Ha].
    split; [exact Hb |]. intros p [<- | Hp]; [exact Ha | exact (H2 p Hp)].
Qed.

(* ------------------------------------------------------------------ *)
(* Rounding and alignment of forms.                                    *)

Lemma form_align_le_const : forall (rf : reg -> Z) f, form_align_with rf f <= const_align f.
Proof.
  intros rf [m ts lo hi]. unfold form_align_with. cbn [ft].
  generalize (const_align {| fm := m; ft := ts; flo := lo; fhi := hi |}) as k0.
  induction ts as [| [x c] ts IH]; intros k0; cbn [fold_left]; [lia |].
  eapply Z.le_trans; [apply IH | lia].
Qed.

Lemma sat_range : forall wdn rs m m' lo hi v e,
    0 <= m' <= m -> lo <= e <= hi -> eqm m v e -> sat wdn rs (range_form m' lo hi) v.
Proof.
  intros wdn rs m m' lo hi v e Hm He Hv. exists e. unfold range_form. cbn [fm ft flo fhi tsum].
  split; [exact He |]. rewrite Z.add_0_l. eapply eqm_weaken; [| exact Hv]. exact Hm.
Qed.

Section Join.

Set Default Proof Using "All".

Variable F : lfacts.
Variable E : env.
Hypothesis HE : env_ok F E.
Variable t : nat.
Variable rs : reg -> Z.
Hypothesis Hrs : regs_ok F E rs.

Notation wdn := (wdn F).
Notation aval := (atom_val wdn rs).
Notation sat := (sat wdn rs).
Notation tsum := (tsum wdn rs).
Notation fact_okv := (fact_okv F t rs).

(* The value of a form whose constant is exact is a multiple of its
   alignment, up to its modulus. *)
Lemma form_value_aligned : forall st f v,
    state_ok F t st rs -> wf f -> sat f v -> aligned (Z.min (form_align F st f) (fm f)) v.
Proof.
  intros st f v Hst Hwf [e [He Hv]]. unfold wf in Hwf.
  destruct (Z.eq_dec (flo f) (fhi f)) as [Heq | Hne].
  - assert (e = flo f) by lia. subst e.
    pose proof (form_align_sound F E HE t st rs Hst Hrs f Heq ltac:(lia)) as Ha.
    apply (aligned_eqm (fm f) _ v (tsum (ft f) + flo f)); [lia | lia | exact Hv |].
    apply (aligned_mono (form_align F st f)); [lia | exact Ha].
  - apply aligned_nonpos. unfold form_align.
    pose proof (form_align_le_const (ra F DEPTH st) f) as Hc.
    unfold const_align in Hc. rewrite (proj2 (Z.eqb_neq _ _) Hne) in Hc. lia.
Qed.

Lemma value_align_sound : forall st f al v,
    state_ok F t st rs -> wf f -> sat f v -> aligned al v -> aligned (value_align F st f al) v.
Proof.
  intros st f al v Hst Hwf Hs Ha. unfold value_align.
  apply aligned_max; [exact Ha | apply form_value_aligned; assumption].
Qed.

(* ------------------------------------------------------------------ *)
(* The order of facts.                                                 *)

Lemma sub_sat : forall f h v,
    terms_eqb (ft f) (ft h) = true -> 0 <= fm h <= fm f -> flo h <= flo f -> fhi f <= fhi h ->
    sat f v -> sat h v.
Proof.
  intros f h v Ht Hm Hlo Hhi [e [He Hv]]. apply terms_eqb_eq in Ht.
  exists e. split; [lia |]. rewrite <- Ht. eapply eqm_weaken; [| exact Hv]. exact Hm.
Qed.

Lemma cmps_sub_holds : forall a b, cmps_sub a b = true -> Forall (cmp_holds rs) b -> Forall (cmp_holds rs) a.
Proof.
  intros a b H Hb. unfold cmps_sub in H. rewrite forallb_forall in H. rewrite Forall_forall in *.
  intros k Hk. specialize (H k Hk). apply existsb_exists in H. destruct H as [l [Hl Hkl]].
  apply cmpf_eqb_eq in Hkl. subst l. exact (Hb k Hl).
Qed.

(* A form fact's value range: its value is one of the range modulo the
   form's modulus. *)
Lemma value_range_form : forall a al vr v p q,
    fact_okv (FForm a al vr) v -> value_range (FForm a al vr) = Some (p, q) ->
    exists e, p <= e <= q /\ eqm (fm a) v e.
Proof.
  intros a al vr v p q [_ [[e [He Hs]] [_ Hvr]]] H. unfold value_range in H.
  destruct vr as [[c d] |]; cbv beta iota in H.
  - apply some_inj in H. injection H as -> ->. exists v. split; [apply Hvr; reflexivity | apply eqm_refl].
  - destruct (ft a) as [| ? ?] eqn:Ea; cbn [null] in H; [| discriminate].
    apply some_inj in H. injection H as -> ->. exists e. split; [exact He |].
    cbn [tsum] in Hs. rewrite Z.add_0_l in Hs. exact Hs.
Qed.

(* A form's terms are the terms another form shares and the rest. *)
Lemma tsum_common : forall a b, tsum (ft a) = tsum (common_terms a b) + tsum (ft (rest_form a b)).
Proof. intros a b. unfold common_terms, rest_form. cbn [ft]. apply tsum_filter. Qed.

Theorem fact_le_sound : forall st x g v,
    state_ok F t st rs -> fact_okv x v -> fact_le F st x g = true ->
    fact_okv g v.
Proof.
  intros st x g v Hst Hx H.
  destruct x as [f al vr | k | cs ex | cs ex];
    destruct g as [h al' vr' | k' | cs' ex' | cs' ex'];
    rewrite fact_le_same in H; cbv beta iota zeta in H; try discriminate.
  - (* form <= form *)
    destruct Hx as [Hwf [Hs [Ha Hvr]]]. rewrite !andb_true_iff in H.
    destruct H as [[[[H0 H64] Hsub] Hal] Hv]. apply Z.leb_le in H0, H64, Hal.
    assert (Hwh : wf h) by (unfold wf; lia).
    cbn [fact_okv]. split; [exact Hwh |]. split.
    + apply orb_true_iff in Hsub. destruct Hsub as [Hsub | Hsh].
      2: { (* g's terms are the terms x shares with it, and the rest lies in g's range *)
           rewrite !andb_true_iff in Hsh. destruct Hsh as [[[_ Hc] Hm] Hr]. apply Z.leb_le in Hm.
           apply terms_eqb_eq in Hc.
           destruct (form_range F st (rest_form f h)) as [[c d] |] eqn:Er; [| discriminate].
           apply andb_true_iff in Hr. destruct Hr as [Hc' Hd']. apply Z.leb_le in Hc', Hd'.
           destruct Hs as [e [He Hve]].
           pose proof (form_range_sound F E HE t st rs Hst Hrs (rest_form f h) c d e Er He) as Hb.
           exists (tsum (ft (rest_form f h)) + e). split; [lia |].
           rewrite <- Hc. rewrite (tsum_common f h) in Hve. rewrite Z.add_assoc.
           apply (eqm_weaken (fm f)); [lia | exact Hve]. }
      apply orb_true_iff in Hsub. destruct Hsub as [Hsub | Hin].
      2: { (* x's value range, or its form's range, lies in g's constant range *)
           apply andb_true_iff in Hin. destruct Hin as [Hnil Hin].
           destruct (ft h) eqn:Eh; [| discriminate].
           apply orb_true_iff in Hin. destruct Hin as [Hin | Hin].
           - destruct vr as [[c d] |]; [| discriminate]. apply andb_true_iff in Hin. destruct Hin as [Hc Hd].
             apply Z.leb_le in Hc, Hd. specialize (Hvr c d eq_refl).
             exists v. rewrite Eh. cbn [tsum].
             split; [lia | rewrite Z.add_0_l; apply eqm_refl].
           - apply andb_true_iff in Hin. destruct Hin as [Hm Hin]. apply Z.leb_le in Hm.
             destruct (form_range F st f) as [[c d] |] eqn:Er; [| discriminate].
             apply andb_true_iff in Hin. destruct Hin as [Hc Hd]. apply Z.leb_le in Hc, Hd.
             destruct Hs as [e [He Hve]].
             pose proof (form_range_sound F E HE t st rs Hst Hrs f c d e Er He) as Hr.
             exists (tsum (ft f) + e). rewrite Eh. cbn [GateForm.tsum].
             split; [lia | rewrite Z.add_0_l].
             apply (eqm_weaken (fm f)); [lia | exact Hve]. }
      apply orb_true_iff in Hsub. destruct Hsub as [Hsub | Hsub]; rewrite !andb_true_iff in Hsub;
        destruct Hsub as [[[Ht Hm] Hlo] Hhi]; apply Z.leb_le in Hm, Hlo, Hhi.
      * apply (sub_sat f); [exact Ht | lia | exact Hlo | exact Hhi | exact Hs].
      * destruct (normalize_sound F E HE t st rs Hst Hrs f v Hwf Hs) as [Hwn [Hfn Hsn]].
        destruct (normalize_exact_fm F E HE t st rs Hst Hrs h Hwh) as [Hwx Hfx].
        apply (normalize_exact_bwd F E HE t st rs Hst Hrs h v Hwh).
        apply (sub_sat (normalize F st f)); [exact Ht | unfold wf in Hwh; lia | exact Hlo | exact Hhi | exact Hsn].
    + split.
      * apply (aligned_mono (value_align F st f al)); [exact Hal |]. apply value_align_sound; assumption.
      * intros a b Ev. subst vr'. destruct vr as [[c d] |]; [| discriminate].
        rewrite !andb_true_iff in Hv. destruct Hv as [Hac Hdb]. apply Z.leb_le in Hac, Hdb.
        specialize (Hvr c d eq_refl). lia.
  - (* form <= alignment *)
    destruct Hx as [Hwf [Hs [Ha _]]]. apply andb_true_iff in H. destruct H as [_ Hk]. apply Z.leb_le in Hk.
    destruct (normalize_sound F E HE t st rs Hst Hrs f v Hwf Hs) as [Hwn [_ Hsn]].
    cbn [fact_okv]. apply (aligned_mono (value_align F st (normalize F st f) al)); [exact Hk |].
    apply value_align_sound; assumption.
  - (* alignment <= alignment *)
    apply Z.leb_le in H. cbn [fact_okv] in *. apply (aligned_mono k); assumption.
  (* predicates, and predicates that fail: the same argument *)
  - apply andb_true_iff in H. destruct H as [Hsub He]. destruct Hx as [Hx1 Hx2]. cbn [fact_okv]. split;
      [intros Hv; apply (cmps_sub_holds cs' cs Hsub); exact (Hx1 Hv) |].
    intros Hex Hall. apply orb_true_iff in He. destruct He as [He | He]; [rewrite Hex in He; discriminate |].
    apply andb_true_iff in He. destruct He as [Hex0 Hsub'].
    apply Hx2; [exact Hex0 | exact (cmps_sub_holds cs cs' Hsub' Hall)].
  - apply andb_true_iff in H. destruct H as [Hsub He]. destruct Hx as [Hx1 Hx2]. cbn [fact_okv]. split;
      [intros Hv; apply (cmps_sub_holds cs' cs Hsub); exact (Hx1 Hv) |].
    intros Hex Hall. apply orb_true_iff in He. destruct He as [He | He]; [rewrite Hex in He; discriminate |].
    apply andb_true_iff in He. destruct He as [Hex0 Hsub'].
    apply Hx2; [exact Hex0 | exact (cmps_sub_holds cs cs' Hsub' Hall)].
Qed.

(* ------------------------------------------------------------------ *)
(* The join of facts.                                                  *)

Lemma join_form_terms : forall sf sg a al vr b bl v,
    state_ok F t sf rs -> fact_okv (FForm a al vr) v -> terms_eqb (ft a) (ft b) = true ->
    fact_okv (FForm {| fm := Z.max 0 (Z.min (fm a) (fm b)); ft := ft a; flo := Z.min (flo a) (flo b);
                       fhi := Z.max (fhi a) (fhi b) |}
                    (Z.min (value_align F sf a al) (value_align F sg b bl)) None) v.
Proof.
  intros sf sg a al vr b bl v Hsf [Hwf [Hs [Ha _]]] Ht. unfold wf in Hwf. cbn [fact_okv].
  split; [unfold wf; cbn [fm]; lia |]. split.
  - destruct Hs as [e [He Hv]]. exists e. cbn [fm ft flo fhi]. split; [lia |].
    eapply eqm_weaken; [| exact Hv]. lia.
  - split; [| discriminate]. apply (aligned_mono (value_align F sf a al)); [lia |].
    apply value_align_sound; [exact Hsf | unfold wf; lia | exact Hs | exact Ha].
Qed.

Lemma join_form_terms_r : forall sf sg a al b bl vr' v,
    state_ok F t sg rs -> fact_okv (FForm b bl vr') v -> terms_eqb (ft a) (ft b) = true ->
    fact_okv (FForm {| fm := Z.max 0 (Z.min (fm a) (fm b)); ft := ft a; flo := Z.min (flo a) (flo b);
                       fhi := Z.max (fhi a) (fhi b) |}
                    (Z.min (value_align F sf a al) (value_align F sg b bl)) None) v.
Proof.
  intros sf sg a al b bl vr' v Hsg [Hwf [Hs [Ha _]]] Ht. apply terms_eqb_eq in Ht. unfold wf in Hwf.
  cbn [fact_okv]. split; [unfold wf; cbn [fm]; lia |]. split.
  - destruct Hs as [e [He Hv]]. exists e. cbn [fm ft flo fhi]. rewrite Ht. split; [lia |].
    eapply eqm_weaken; [| exact Hv]. lia.
  - split; [| discriminate]. apply (aligned_mono (value_align F sg b bl)); [lia |].
    apply value_align_sound; [exact Hsg | unfold wf; lia | exact Hs | exact Ha].
Qed.

(* A value in a form fact's value range satisfies the joined range form. *)
Lemma join_range_form : forall a al vr v p q m lo hi,
    fact_okv (FForm a al vr) v -> value_range (FForm a al vr) = Some (p, q) ->
    0 <= m <= fm a -> m <= 64 -> lo <= p -> q <= hi ->
    wf (range_form m lo hi) /\ sat (range_form m lo hi) v.
Proof.
  intros a al vr v p q m lo hi Hx Hr Hm H64 Hlo Hhi.
  destruct (value_range_form a al vr v p q Hx Hr) as [e [He Hv]].
  split; [unfold wf, range_form; cbn [fm]; lia |].
  apply (sat_range wdn rs (fm a) m lo hi v e); [lia | lia | exact Hv].
Qed.

(* The same for a fact's range in its state, which may be its form's range. *)
Lemma join_range_in : forall st a al vr v p q m lo hi,
    state_ok F t st rs -> fact_okv (FForm a al vr) v -> fact_range_in F st (FForm a al vr) = Some (p, q) ->
    0 <= m <= fm a -> m <= 64 -> lo <= p -> q <= hi ->
    wf (range_form m lo hi) /\ sat (range_form m lo hi) v.
Proof.
  intros st a al vr v p q m lo hi Hst Hx Hr Hm H64 Hlo Hhi. unfold fact_range_in in Hr.
  destruct (value_range (FForm a al vr)) as [[p' q'] |] eqn:Ev.
  - apply some_inj in Hr. injection Hr as E1 E2. subst.
    exact (join_range_form a al vr v _ _ m lo hi Hx Ev Hm H64 Hlo Hhi).
  - destruct Hx as [_ [[e [He Hs]] _]].
    pose proof (form_range_sound F E HE t st rs Hst Hrs a p q e Hr He) as Hb.
    split; [unfold wf, range_form; cbn [fm]; lia |].
    apply (sat_range wdn rs (fm a) m lo hi v (tsum (ft a) + e)); [lia | lia | exact Hs].
Qed.

Lemma join_common_sound_l : forall sf sg a al vr b al' v z,
    state_ok F t sf rs -> fact_okv (FForm a al vr) v -> aligned al' v ->
    join_common F sf sg a b al' = Some z -> fact_okv z v.
Proof.
  intros sf sg a al vr b al' v z Hsf Hx Hal H. unfold join_common in H.
  destruct (negb (null (common_terms a b)) && terms_eqb (common_terms a b) (common_terms b a)); [| discriminate].
  destruct (form_range F sf (rest_form a b)) as [[p q] |] eqn:E1; [| discriminate].
  destruct (form_range F sg (rest_form b a)) as [[r s] |]; [| discriminate].
  apply some_inj in H. subst z.
  destruct Hx as [Hwf [[e [He Hv]] _]]. unfold wf in Hwf.
  pose proof (form_range_sound F E HE t sf rs Hsf Hrs (rest_form a b) p q e E1 He) as Hb.
  cbn [fact_okv]. split; [unfold wf; cbn [fm]; lia |]. split; [| split; [exact Hal | discriminate]].
  exists (tsum (ft (rest_form a b)) + e). cbn [fm ft flo fhi]. split; [lia |].
  rewrite (tsum_common a b) in Hv. rewrite Z.add_assoc.
  eapply eqm_weaken; [| exact Hv]. lia.
Qed.

Lemma join_common_sound_r : forall sf sg a b bl vr' al' v z,
    state_ok F t sg rs -> fact_okv (FForm b bl vr') v -> aligned al' v ->
    join_common F sf sg a b al' = Some z -> fact_okv z v.
Proof.
  intros sf sg a b bl vr' al' v z Hsg Hy Hal H. unfold join_common in H.
  destruct (negb (null (common_terms a b)) && terms_eqb (common_terms a b) (common_terms b a)) eqn:Ec;
    [| discriminate].
  apply andb_true_iff in Ec. destruct Ec as [_ Ec]. apply terms_eqb_eq in Ec.
  destruct (form_range F sf (rest_form a b)) as [[p q] |]; [| discriminate].
  destruct (form_range F sg (rest_form b a)) as [[r s] |] eqn:E2; [| discriminate].
  apply some_inj in H. subst z.
  destruct Hy as [Hwf [[e [He Hv]] _]]. unfold wf in Hwf.
  pose proof (form_range_sound F E HE t sg rs Hsg Hrs (rest_form b a) r s e E2 He) as Hb.
  cbn [fact_okv]. split; [unfold wf; cbn [fm]; lia |]. split; [| split; [exact Hal | discriminate]].
  exists (tsum (ft (rest_form b a)) + e). cbn [fm ft flo fhi]. split; [lia |].
  rewrite Ec. rewrite (tsum_common b a) in Hv. rewrite Z.add_assoc.
  eapply eqm_weaken; [| exact Hv]. lia.
Qed.

Lemma filter_holds : forall P (cs : list cmpf), Forall (cmp_holds rs) cs -> Forall (cmp_holds rs) (filter P cs).
Proof.
  intros P cs H. rewrite Forall_forall in *. intros k Hk. apply filter_In in Hk. apply H. apply Hk.
Qed.

Lemma filter_in_holds : forall (cs cs' : list cmpf),
    Forall (cmp_holds rs) cs' -> Forall (cmp_holds rs) (filter (fun k => existsb (cmpf_eqb k) cs') cs).
Proof.
  intros cs cs' H. rewrite Forall_forall in *. intros k Hk. apply filter_In in Hk. destruct Hk as [_ Hk].
  apply existsb_exists in Hk. destruct Hk as [l [Hl Hkl]]. apply cmpf_eqb_eq in Hkl. subst l. exact (H k Hl).
Qed.

Lemma filter_sub_all : forall (cs cs' : list cmpf),
    cmps_sub cs cs' = true -> filter (fun k => existsb (cmpf_eqb k) cs') cs = cs.
Proof.
  intros cs cs' H. apply filter_all. unfold cmps_sub in H. rewrite forallb_forall in H. exact H.
Qed.

Theorem join_fact_sound_l : forall sf sg x y z v,
    state_ok F t sf rs -> fact_okv x v -> join_fact F sf sg (Some x) y = Some z ->
    fact_okv z v.
Proof.
  intros sf sg x [y |] z v Hsf Hx H; [| discriminate].
  unfold join_fact in H. cbv beta iota zeta in H.
  destruct (fact_le F sf x y) eqn:E1; [apply some_inj in H; subst z; exact (fact_le_sound sf x y v Hsf Hx E1) |].
  destruct (fact_le F sg y x) eqn:E2; [apply some_inj in H; subst z; exact Hx |].
  destruct x as [a al vr | k | cs ex | cs ex];
    destruct y as [b bl vr' | k' | cs' ex' | cs' ex'];
    cbv beta iota zeta in H; try discriminate.
  - (* forms *)
    destruct (terms_eqb (ft a) (ft b)) eqn:Et.
    + apply some_inj in H. subst z. apply (join_form_terms sf sg a al vr b bl v); assumption.
    + destruct (fact_range_in F sf (FForm a al vr)) as [[p q] |] eqn:Ev1;
        [destruct (fact_range_in F sg (FForm b bl vr')) as [[r s] |] eqn:Ev2 |].
      1: { apply some_inj in H. subst z. pose proof Hx as Hx'. destruct Hx' as [Hwf [_ [Ha _]]]. unfold wf in Hwf.
        destruct (join_range_in sf a al vr v p q (Z.max 0 (Z.min (fm a) (fm b))) (Z.min p r) (Z.max q s) Hsf Hx Ev1)
          as [Hw Hs']; try lia.
        cbn [fact_okv]. split; [exact Hw |]. split; [exact Hs' |]. split; [| discriminate].
        apply (aligned_mono (value_align F sf a al)); [lia |].
        apply value_align_sound; [exact Hsf | unfold wf; lia | apply Hx | exact Ha]. }
      (* the shared terms *)
      all: pose proof Hx as Hx'; destruct Hx' as [Hwf [Hs [Ha _]]]; unfold wf in Hwf;
        apply (join_common_sound_l sf sg a al vr b (Z.min (value_align F sf a al) (value_align F sg b bl)) v z Hsf Hx);
        [apply (aligned_mono (value_align F sf a al)); [lia |];
         apply value_align_sound; [exact Hsf | unfold wf; lia | exact Hs | exact Ha] | exact H].
  - (* alignments *)
    destruct (0 <? Z.min k k') eqn:Ek; [| discriminate]. apply some_inj in H. subst z.
    cbn [fact_okv] in *. apply (aligned_mono k); [lia | exact Hx].
  (* predicates, and predicates that fail: the same argument *)
  - destruct (filter (fun k => existsb (cmpf_eqb k) cs') cs) as [| k0 c0] eqn:Ef; [discriminate |].
    apply some_inj in H. subst z. rewrite <- Ef. destruct Hx as [Hx1 Hx2]. cbn [fact_okv]. split.
    + intros Hv. apply filter_holds. exact (Hx1 Hv).
    + intros He Hall. rewrite !andb_true_iff in He. destruct He as [[[He0 _] Hs1] _].
      rewrite (filter_sub_all cs cs' Hs1) in Hall. exact (Hx2 He0 Hall).
  - destruct (filter (fun k => existsb (cmpf_eqb k) cs') cs) as [| k0 c0] eqn:Ef; [discriminate |].
    apply some_inj in H. subst z. rewrite <- Ef. destruct Hx as [Hx1 Hx2]. cbn [fact_okv]. split.
    + intros Hv. apply filter_holds. exact (Hx1 Hv).
    + intros He Hall. rewrite !andb_true_iff in He. destruct He as [[[He0 _] Hs1] _].
      rewrite (filter_sub_all cs cs' Hs1) in Hall. exact (Hx2 He0 Hall).
Qed.

Theorem join_fact_sound_r : forall sf sg x y z v,
    state_ok F t sg rs -> fact_okv y v -> join_fact F sf sg x (Some y) = Some z ->
    fact_okv z v.
Proof.
  intros sf sg [x |] y z v Hsg Hy H; [| discriminate].
  unfold join_fact in H. cbv beta iota zeta in H.
  destruct (fact_le F sf x y) eqn:E1; [apply some_inj in H; subst z; exact Hy |].
  destruct (fact_le F sg y x) eqn:E2; [apply some_inj in H; subst z; exact (fact_le_sound sg y x v Hsg Hy E2) |].
  destruct x as [a al vr | k | cs ex | cs ex];
    destruct y as [b bl vr' | k' | cs' ex' | cs' ex'];
    cbv beta iota zeta in H; try discriminate.
  - (* forms *)
    destruct (terms_eqb (ft a) (ft b)) eqn:Et.
    + apply some_inj in H. subst z. apply join_form_terms_r with (vr' := vr'); assumption.
    + destruct (fact_range_in F sf (FForm a al vr)) as [[p q] |] eqn:Ev1;
        [destruct (fact_range_in F sg (FForm b bl vr')) as [[r s] |] eqn:Ev2 |].
      1: { apply some_inj in H. subst z. pose proof Hy as Hy'. destruct Hy' as [Hwf [_ [Ha _]]]. unfold wf in Hwf.
        destruct (join_range_in sg b bl vr' v r s (Z.max 0 (Z.min (fm a) (fm b))) (Z.min p r) (Z.max q s) Hsg Hy Ev2)
          as [Hw Hs']; try lia.
        cbn [fact_okv]. split; [exact Hw |]. split; [exact Hs' |]. split; [| discriminate].
        apply (aligned_mono (value_align F sg b bl)); [lia |].
        apply value_align_sound; [exact Hsg | unfold wf; lia | apply Hy | exact Ha]. }
      (* the shared terms *)
      all: pose proof Hy as Hy'; destruct Hy' as [Hwf [Hs [Ha _]]]; unfold wf in Hwf;
        apply (join_common_sound_r sf sg a b bl vr' (Z.min (value_align F sf a al) (value_align F sg b bl)) v z Hsg Hy);
        [apply (aligned_mono (value_align F sg b bl)); [lia |];
         apply value_align_sound; [exact Hsg | unfold wf; lia | exact Hs | exact Ha] | exact H].
  - (* alignments *)
    destruct (0 <? Z.min k k') eqn:Ek; [| discriminate]. apply some_inj in H. subst z.
    cbn [fact_okv] in *. apply (aligned_mono k'); [lia | exact Hy].
  (* predicates, and predicates that fail: the same argument *)
  - destruct (filter (fun k => existsb (cmpf_eqb k) cs') cs) as [| k0 c0] eqn:Ef; [discriminate |].
    apply some_inj in H. subst z. rewrite <- Ef. destruct Hy as [Hy1 Hy2]. cbn [fact_okv]. split.
    + intros Hv. apply filter_in_holds. exact (Hy1 Hv).
    + intros He Hall. rewrite !andb_true_iff in He. destruct He as [[[_ He'] Hs1] Hs2].
      rewrite (filter_sub_all cs cs' Hs1) in Hall. exact (Hy2 He' (cmps_sub_holds cs' cs Hs2 Hall)).
  - destruct (filter (fun k => existsb (cmpf_eqb k) cs') cs) as [| k0 c0] eqn:Ef; [discriminate |].
    apply some_inj in H. subst z. rewrite <- Ef. destruct Hy as [Hy1 Hy2]. cbn [fact_okv]. split.
    + intros Hv. apply filter_in_holds. exact (Hy1 Hv).
    + intros He Hall. rewrite !andb_true_iff in He. destruct He as [[[_ He'] Hs1] Hs2].
      rewrite (filter_sub_all cs cs' Hs1) in Hall. exact (Hy2 He' (cmps_sub_holds cs' cs Hs2 Hall)).
Qed.

(* ------------------------------------------------------------------ *)
(* The order of states.                                                *)

Lemma eff_fact_sound : forall a r x,
    state_ok F t a rs -> eff_fact F a r = Some x -> fact_okv x (rs r).
Proof.
  intros a r x Ha H. unfold eff_fact in H.
  destruct (get a r) as [y |] eqn:Eg; [apply some_inj in H; subst y; exact (proj1 Ha r x Eg) |].
  destruct (is_fixed F r) eqn:Ef; [| discriminate].
  destruct (lookup (lf_bounds F) r) as [[lo hi] |] eqn:El; [| discriminate].
  apply some_inj in H. subst x.
  pose proof (ok_bounds F E HE r lo hi Ef El) as Hb. rewrite <- (proj2 Hrs r Ef) in Hb.
  pose proof (ok_widths F E HE r) as Hw. unfold wdz in Hw.
  cbn [fact_okv]. split; [unfold wf, range_form; cbn [fm]; lia |].
  split; [apply (sat_range wdn rs (wd F r) (wd F r) lo hi (rs r) (rs r)); [lia | lia | apply eqm_refl] |].
  split; [apply aligned_nonpos; lia | discriminate].
Qed.

Lemma branch_excludes_sound : forall st r truth k,
 state_ok F t st rs -> negb (Z.eqb (rs r) 0) = truth ->
 branch_excludes F st r truth k = true -> ~ cmp_holds rs k.
Proof.
 intros st r truth k Hst Hp Hbad Hk.
 destruct (refine_sound F E HE t rs Hrs st r truth Hst Hp) as [s [Es Hs]].
 destruct (refine_cmp_sound F E HE t rs Hrs s k Hs Hk) as [s' [Es' Hs']].
 unfold branch_excludes in Hbad. rewrite Es, Es' in Hbad.
 exact (bottom_not_ok F t s' rs Hbad Hs').
Qed.

Lemma predicate_fact_le_sound : forall st r g,
 state_ok F t st rs -> predicate_fact_le F st r g = true -> fact_okv g (rs r).
Proof.
 intros st r g Hst H. destruct g as [f al vr | al | cs exact | cs exact]; try discriminate.
 destruct cs as [|k ks]; [discriminate |]. destruct ks; [| discriminate].
 unfold predicate_fact_le in H. apply andb_true_iff in H. destruct H as [Hyes Hno].
 cbn [fact_okv]. split.
 - intros Hr. constructor; [| constructor].
   assert (Hp : negb (Z.eqb (rs r) 0) = true) by (apply negb_true_iff, Z.eqb_neq; exact Hr).
   pose proof (branch_excludes_sound st r true (negate k) Hst Hp Hyes) as Hbad.
   unfold cmp_holds in *. cbn [negate cc cw csg ca cb] in Hbad. rewrite flip_eval in Hbad.
   destruct (cmp_eval (cc k) _ _) eqn:Ecmp; [reflexivity |]. exfalso. apply Hbad. reflexivity.
 - intros Hex Hk Hz. subst exact. cbn [negb orb] in Hno.
   assert (Hp : negb (Z.eqb (rs r) 0) = false) by (rewrite Hz; reflexivity).
   apply (branch_excludes_sound st r false k Hst Hp Hno).
   inversion Hk; assumption.
Qed.

Lemma predicate_constant_sound : forall st p c,
 state_ok F t st rs -> predicate_constant F st p = Some c -> rs p = c.
Proof.
 intros st p c Hst H. unfold predicate_constant in H.
 destruct (negb (Z.eqb (wd F p) 1)) eqn:Hw; [discriminate |].
 apply negb_false_iff, Z.eqb_eq in Hw.
 destruct (get st p) as [[f al vr | al | cs exact | cs exact] |]; try discriminate;
 destruct exact; try discriminate;
 destruct (refine F st p true) as [s |] eqn:Et.
 - destruct (refine F st p false) as [s' |] eqn:Ef; [discriminate |].
   injection H as <-. exact (impossible_predicate_value F E HE t rs Hrs st p false Hst Hw Ef).
 - injection H as <-. exact (impossible_predicate_value F E HE t rs Hrs st p true Hst Hw Et).
 - destruct (refine F st p false) as [s' |] eqn:Ef; [discriminate |].
   injection H as <-. exact (impossible_predicate_value F E HE t rs Hrs st p false Hst Hw Ef).
 - injection H as <-. exact (impossible_predicate_value F E HE t rs Hrs st p true Hst Hw Et).
Qed.

Lemma register_fact_le_sound : forall st r g,
 state_ok F t st rs -> register_fact_le F st r g = true -> fact_okv g (rs r).
Proof.
 intros st r g Hst H. unfold register_fact_le in H.
 destruct (fact_le_o F st (eff_fact F st r) (Some g)) eqn:Eordinary.
 { unfold fact_le_o in Eordinary. destruct (eff_fact F st r) as [x |] eqn:Ex; [| discriminate].
   exact (fact_le_sound st x g (rs r) Hst (eff_fact_sound st r x Hst Ex) Eordinary). }
 destruct g as [f al vr | al | cs exact | cs exact]; try discriminate.
 - destruct vr; [discriminate |].
   destruct (Z.eqb (wd F r) 1 && Z.eqb (fm f) 1 && null (ft f) && (al <=? 0)) eqn:Ebit.
   { rewrite !andb_true_iff in Ebit. destruct Ebit as [[[Hw Hm] Hts] Hal].
     apply Z.eqb_eq in Hw,Hm. apply Z.leb_le in Hal.
     destruct (ft f) as [|u us] eqn:Ets; [| discriminate].
     assert (Hr : flo f <= rs r <= fhi f).
     { destruct ((flo f <=? 0) && (1 <=? fhi f)) eqn:Eall.
       - apply andb_true_iff in Eall. destruct Eall as [Hlo Hhi]. apply Z.leb_le in Hlo,Hhi.
         pose proof (rs_range F E rs Hrs r) as Hrange. unfold wd in Hw. rewrite Hw in Hrange. cbn in Hrange. lia.
       - destruct (predicate_constant F st r) as [c |] eqn:Ec; [| discriminate].
         pose proof (predicate_constant_sound st r c Hst Ec) as Hv.
         apply andb_true_iff in H. destruct H as [Hlo Hhi]. apply Z.leb_le in Hlo,Hhi. lia. }
     cbn [fact_okv]. split; [unfold wf; lia |]. split.
     - exists (rs r). rewrite Ets,Hm. cbn [tsum]. split; [exact Hr |]. rewrite Z.add_0_l. apply eqm_refl.
     - split; [apply aligned_nonpos; exact Hal | discriminate]. }

   destruct (Z.eqb (fm f) 64 && Z.eqb (wd F r) 64 && null (ft f) && (al <=? 0)) eqn:Ef; [| discriminate].
   rewrite !andb_true_iff in Ef. destruct Ef as [[[Hm Hw] Hts] Hal].
   apply Z.eqb_eq in Hm,Hw. apply Z.leb_le in Hal.
   destruct (ft f) as [|u us] eqn:Ets; [| discriminate].
   destruct (check_range F st (fatom 64 (r,EX) 1)) as [[lo hi] |] eqn:Er; [| discriminate].
   apply andb_true_iff in H. destruct H as [Hlo Hhi]. apply Z.leb_le in Hlo,Hhi.
   pose proof (check_range_sound F E HE t rs Hrs st (fatom 64 (r,EX) 1) lo hi 0 Hst Er ltac:(change (0 <= 0 <= 0); lia)) as Hr.
   change (lo <= 1 * rs r + 0 + 0 <= hi) in Hr.
   cbn [fact_okv]. split; [unfold wf; lia |]. split.
   + exists (rs r). rewrite Ets,Hm. cbn [tsum]. split; [lia |]. rewrite Z.add_0_l. apply eqm_refl.
   + split; [apply aligned_nonpos; exact Hal | discriminate].
 - exact (predicate_fact_le_sound st r (FPred cs exact) Hst H).
Qed.

(* The sum of the terms and the sum of their normal form agree modulo 2^64;
   when both ranges lie within one 2^64 window, they are equal. *)
(* A register's copy has its value. *)
Lemma copy_atom_val : forall a x, state_ok F t a rs -> aval (copy_atom F a x) = aval x.
Proof.
  intros a [r e] Ha. unfold copy_atom.
  destruct (copy_of F r (get a r)) as [r2 |] eqn:Ec; [| destruct e; reflexivity].
  pose proof (copy_of_sound F E HE t a rs Ha Hrs r r2 Ec) as Hv.
  destruct (copy_of_shape F E HE t a rs Ha Hrs r _ r2 Ec) as [f [al [vr [e' [Hx [_ [_ [_ [Hft [He' _]]]]]]]]]].
  rewrite Hx, Hft.
  destruct e; try reflexivity; cbn [atom_val]; destruct He' as [-> | ->]; exact Hv.
Qed.

Lemma copy_terms_tsum : forall a ts, state_ok F t a rs -> tsum (copy_terms F a ts) = tsum ts.
Proof.
  intros a ts Ha. unfold copy_terms. induction ts as [| [x c] ts IH]; [reflexivity |].
  cbn [map fst snd tsum]. rewrite IH. f_equal. f_equal.
  generalize 4%nat as n. intros n. revert x. induction n as [| n IHn]; intros x; [reflexivity |].
  cbn [copy_atom_n]. rewrite IHn. exact (copy_atom_val a x Ha).
Qed.

Lemma near_constraint_sound : forall a ts lo hi,
    state_ok F t a rs -> near_constraint F a ts lo hi = Some true ->
    cons_ok F rs (ts,lo,hi).
Proof.
  intros a ts lo hi Ha Hc. unfold near_constraint, range_constraint in Hc.
  destruct (form_range_near F a (form_of_terms ts)) as [[p q]|] eqn:Er; [|discriminate].
  pose proof (form_range_near_sound F E HE t a rs Ha Hrs (form_of_terms ts) p q 0 Er
    ltac:(unfold form_of_terms; cbn [flo fhi]; lia)) as Hr.
  change (p <= tsum ts + 0 <= q) in Hr. cbn [cons_ok].
  destruct ((lo <=? p) && (q <=? hi)) eqn:Eb.
  - apply andb_true_iff in Eb. destruct Eb as [Hl Hh]. apply Z.leb_le in Hl,Hh. lia.
  - destruct ((hi <? p) || (q <? lo)); discriminate.
Qed.

(* The negative classification does not assume that an abstract state has
   a concrete model. Whenever a model exists, this constraint is false. *)
Lemma near_constraint_refutes : forall a ts lo hi,
    state_ok F t a rs -> near_constraint F a ts lo hi = Some false ->
    ~ cons_ok F rs (ts,lo,hi).
Proof.
  intros a ts lo hi Ha Hc. unfold near_constraint, range_constraint in Hc.
  destruct (form_range_near F a (form_of_terms ts)) as [[p q]|] eqn:Er; [|discriminate].
  pose proof (form_range_near_sound F E HE t a rs Ha Hrs (form_of_terms ts) p q 0 Er
    ltac:(unfold form_of_terms; cbn [flo fhi]; lia)) as Hr.
  change (p <= tsum ts + 0 <= q) in Hr. cbn [cons_ok].
  destruct ((lo <=? p) && (q <=? hi)); [discriminate|].
  destruct ((hi <? p) || (q <? lo)) eqn:Eb; [|discriminate].
  apply orb_true_iff in Eb. destruct Eb as [Eb|Eb]; apply Z.ltb_lt in Eb; lia.
Qed.

Lemma cons_holds_sound : forall a k,
    state_ok F t a rs -> cons_holds F a k = true -> cons_ok F rs k.
Proof.
  intros a [[ts0 lo] hi] Ha Hc. rewrite cons_holds_exact in Hc. unfold cons_holds_spec in Hc. cbv beta iota zeta in Hc. cbn [cons_ok].
  lazymatch type of Hc with (if ?c then _ else _) = true => destruct c eqn:Ex end.
  - apply existsb_exists in Ex. destruct Ex as [[[ts' lo'] hi'] [Hk' Hx]].
    rewrite !andb_true_iff in Hx. destruct Hx as [[Ht Hlo] Hhi]. apply Z.leb_le in Hlo, Hhi.
    pose proof (proj1 (Forall_forall _ _) (proj2 Ha) _ Hk') as Hok. cbn [cons_ok] in Hok.
    destruct ((lo <=? lo') && (hi' <=? hi)) eqn:Eb; [| discriminate].
    destruct (terms_eqb ts0 ts') eqn:Et.
    { apply terms_eqb_eq in Et. subst ts'. lia. }
    destruct (terms_eqb (value_terms (copy_terms F a ts0))
                         (value_terms (copy_terms F a ts'))) eqn:Es; [| discriminate].
    apply terms_eqb_eq in Ht.
    (* both terms read through a's copies, whose sums are the terms' own *)
    assert (Hsorted : forall l, tsum (canon_terms F a l) = tsum l)
      by (intros l; exact (canon_terms_tsum F E HE t rs Hrs a l Ha)).
    assert (Hsum : tsum ts0 = tsum ts').
    { rewrite <- (copy_terms_tsum a ts0 Ha), <- (copy_terms_tsum a ts' Ha), <- Hsorted, Ht, Hsorted.
      reflexivity. }
    lia.
  - lazymatch type of Hc with (if ?c then _ else _) = true => destruct c; [discriminate |] end.
    rewrite <- (copy_terms_tsum a ts0 Ha).
    set (ts := copy_terms F a ts0) in *.
    set (f := form_of_terms ts) in *.
    destruct (near_constraint F a ts lo hi) as [answer|] eqn:Enear.
    { destruct answer; [exact (near_constraint_sound a ts lo hi Ha Enear)|discriminate]. }
    destruct ((if cursor_sum F ts || (Z.eqb lo hi && mixed_cursor_sum F ts) || affine_cursor_bound F ts || pair_cursor F a ts then check_range F a else form_range_near F a) f) as [[p q] |] eqn:Er; [| discriminate].
    assert (Hr : p <= tsum (ft f) + 0 <= q).
    { destruct (cursor_sum F ts || (Z.eqb lo hi && mixed_cursor_sum F ts) || affine_cursor_bound F ts || pair_cursor F a ts) eqn:Et; cbv beta iota in Er.
      - exact (check_range_sound F E HE t rs Hrs a f p q 0 Ha Er ltac:(unfold f, form_of_terms; cbn [flo fhi]; lia)).
      - exact (form_range_near_sound F E HE t a rs Ha Hrs f p q 0 Er ltac:(unfold f, form_of_terms; cbn [flo fhi]; lia)). }
    unfold f, form_of_terms in Hr. cbn [ft] in Hr. rewrite Z.add_0_r in Hr.
    destruct ((lo <=? p) && (q <=? hi)) eqn:Ed.
    + apply andb_true_iff in Ed. destruct Ed as [Hlo Hhi]. apply Z.leb_le in Hlo, Hhi. lia.
    + destruct (form_range_near F a (normalize F a f)) as [[x y] |] eqn:En; [| discriminate].
      rewrite !andb_true_iff in Hc. destruct Hc as [[Hlo Hhi] Hw].
      apply Z.leb_le in Hlo, Hhi. apply Z.ltb_lt in Hw.
      assert (Hwf : wf f) by (unfold wf, f, form_of_terms; cbn [fm]; lia).
      assert (Hs : sat f (tsum ts)).
      { exists 0. unfold f, form_of_terms. cbn [flo fhi ft fm].
        split; [lia | rewrite Z.add_0_r; apply eqm_refl]. }
      destruct (normalize_sound F E HE t a rs Ha Hrs f (tsum ts) Hwf Hs) as [_ [Hfm [e [He Hm]]]].
      pose proof (form_range_near_sound F E HE t a rs Ha Hrs (normalize F a f) x y e En He) as Hn.
      rewrite Hfm in Hm. change (fm f) with 64 in Hm.
      assert (Heq : tsum ts = tsum (ft (normalize F a f)) + e) by (apply (eqm_close 64); [lia | exact Hm | lia]).
      lia.
Qed.

Lemma clip_constraint_sound : forall k,
 cons_ok F rs k -> cons_ok F rs (clip_constraint F k).
Proof.
 intros [[ts lo] hi] Hk. unfold clip_constraint.
 destruct (plain_sum (natural_range F) ts 0 0) as [[l h]|] eqn:Er; [|exact Hk].
 pose proof (plain_sum_sound wdn rs (natural_range F) ts 0 0 l h 0
   (fun x c a b _ Hx => natural_range_sound F E HE rs Hrs x a b Hx)
   Er ltac:(lia)) as Hr.
 cbn [cons_ok] in *. rewrite Z.add_0_r in Hr.
 split; [apply Z.max_lub|apply Z.min_glb]; tauto.
Qed.

Lemma clip_join_state_sound : forall st,
 state_ok F t st rs -> state_ok F t (clip_join_state F st) rs.
Proof.
 intros st [Hm Hc]. split; [exact Hm|].
 cbn [clip_join_state acons]. apply Forall_forall. intros k Hk.
 apply in_map_iff in Hk. destruct Hk as [k' [<- Hk']].
 apply clip_constraint_sound. rewrite Forall_forall in Hc. exact (Hc k' Hk').
Qed.

Theorem state_le_sound : forall a b,
    state_ok F t a rs -> state_le F a b = true -> state_ok F t b rs.
Proof.
  intros a b Ha H. unfold state_le in H. apply orb_true_iff in H.
  destruct H as [Hb | H]; [exfalso; exact (bottom_not_ok F t a rs Hb Ha) |].
  apply andb_true_iff in H. destruct H as [Hm Hc]. split.
  - intros r x Hg. rewrite PositiveMap.fold_1 in Hm.
    destruct (fold_and _ (fun p => register_fact_le F a (fst p) (snd p)) _ true Hm) as [_ Hall].
    assert (Hin : In (r, x) (PositiveMap.elements (amap b))) by (apply PositiveMap.elements_correct; exact Hg).
    specialize (Hall _ Hin). cbn [fst snd] in Hall.
    exact (register_fact_le_sound a r x Ha Hall).
  - unfold cons_le in Hc. rewrite forallb_forall in Hc. apply Forall_forall.
    intros k Hk. exact (cons_holds_sound a k Ha (Hc k Hk)).
Qed.

End Join.
