(*
  Soundness of reading registers as forms and of normalization: each form
  that `as_form`, `narrow`, or `opnd_form` gives allows the register's value,
  `normalize` keeps every value its input allows, and `normalize_exact`
  allows no other value than its input does.
*)

From Coq Require Import ZArith.ZArith.
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

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

Lemma sat_eqm : forall wdn rs f a b, sat wdn rs f a -> eqm (fm f) a b -> sat wdn rs f b.
Proof.
  intros wdn rs f a b [e [He Ha]] H. exists e. split; [exact He |].
  eapply eqm_trans; [apply eqm_sym; exact H | exact Ha].
Qed.

Lemma sext_eqm : forall w v, eqm (Z.of_nat w) (sext w v) v.
Proof.
  intros w v. unfold sext. pose proof (trunc_eqm w v) as Ht.
  destruct (Z.ltb (trunc w v) (2 ^ (Z.of_nat w - 1))); [exact Ht |].
  eapply eqm_trans; [| exact Ht]. unfold eqm.
  replace (trunc w v - 2 ^ Z.of_nat w) with (trunc w v + (-1) * 2 ^ Z.of_nat w) by ring.
  apply Z.mod_add. pose proof (pow_nonneg_pos (Z.of_nat w)). lia.
Qed.

Lemma eqm_sub_l : forall m a b c, 0 <= m -> eqm m a (b + c) -> eqm m (a - c) b.
Proof.
  intros m a b c Hm H. replace b with ((b + c) - c) by ring.
  replace (a - c) with (a + - c) by ring. replace (b + c - c) with ((b + c) + - c) by ring.
  apply eqm_add; [exact Hm | exact H | apply eqm_refl].
Qed.

Section Norm.

(* Every lemma takes all of the section's variables, so that callers pass
   the same arguments to each. *)
Set Default Proof Using "All".

Variable F : lfacts.
Variable E : env.
Hypothesis HE : env_ok F E.
Variable t : nat.
Variable st : astate.
Variable rs : reg -> Z.
Hypothesis Hst : state_ok F t st rs.
Hypothesis Hrs : regs_ok F E rs.

Notation wdn := (wdn F).
Notation aval := (atom_val wdn rs).
Notation sat := (sat wdn rs).
Notation tsum := (tsum wdn rs).

Lemma rrs : forall r a b, reg_range F st r = Some (a, b) -> a <= rs r <= b.
Proof. exact (reg_range_sound F E HE t st rs Hst Hrs). Qed.
Lemma srs : forall r a b, sreg_range F st r = Some (a, b) -> a <= sext (wdn r) (rs r) <= b.
Proof. exact (sreg_range_sound F E HE t st rs Hst Hrs). Qed.
Lemma frs : forall f a b e, form_range F st f = Some (a, b) -> flo f <= e <= fhi f -> a <= tsum (ft f) + e <= b.
Proof. exact (form_range_sound F E HE t st rs Hst Hrs). Qed.
Lemma ras : forall r, aligned (reg_align F st r) (rs r).
Proof. exact (reg_align_sound F E HE t st rs Hst Hrs). Qed.
Lemma fas : forall f, flo f = fhi f -> 0 <= fm f -> aligned (form_align F st f) (tsum (ft f) + flo f).
Proof. exact (form_align_sound F E HE t st rs Hst Hrs). Qed.
Lemma rsr : forall r, 0 <= rs r < 2 ^ Z.of_nat (wdn r).
Proof. exact (rs_range F E rs Hrs). Qed.
Lemma wdr : forall r, 1 <= Z.of_nat (wdn r) <= 64.
Proof. exact (wd_range F E HE). Qed.

Lemma form_fact : forall r f al vr, get st r = Some (FForm f al vr) ->
    wf f /\ sat f (rs r) /\ aligned al (rs r) /\ (forall lo hi, vr = Some (lo, hi) -> lo <= rs r <= hi).
Proof. intros r f al vr H. exact (proj1 Hst r _ H). Qed.

Lemma wd_32 : forall r, wd F r = 32 -> wdn r = 32%nat.
Proof. intros r H. unfold wd in H. lia. Qed.

(* ------------------------------------------------------------------ *)
(* A register as a form.                                               *)

Lemma tsum_map_eqm : forall (g : atom * Z -> atom * Z) m ts,
    0 <= m -> (forall x c, snd (g (x, c)) = c /\ eqm m (aval (fst (g (x, c)))) (aval x)) ->
    eqm m (tsum (map g ts)) (tsum ts).
Proof.
  intros g m ts Hm Hg. induction ts as [| [x c] ts IH]; cbn [map tsum]; [apply eqm_refl |].
  destruct (g (x, c)) as [x' c'] eqn:Eg. destruct (Hg x c) as [Hc Hv]. rewrite Eg in Hc, Hv.
  cbn [fst snd] in Hc, Hv. subst c'. cbn [tsum].
  apply eqm_add; [exact Hm | apply eqm_mul; [exact Hm | exact Hv] | exact IH].
Qed.

Lemma tsum_swap_ext : forall ts, eqm 32 (tsum (swap_ext F st ts)) (tsum ts).
Proof.
  intros ts. unfold swap_ext. apply tsum_map_eqm; [lia |].
  intros [r e] c.
  destruct e; cbn [fst snd]; try (split; [reflexivity | apply eqm_refl]).
  - destruct (Z.eqb (wd F r) 32) eqn:E1; cbn [andb]; [| split; [reflexivity | apply eqm_refl]].
    destruct (is_some (atom_range F st (r, EZ))); cbn [negb]; [split; [reflexivity | apply eqm_refl] |].
    destruct (is_some (atom_range F st (r, ES))); cbn [fst snd]; [| split; [reflexivity | apply eqm_refl]].
    split; [reflexivity |]. apply Z.eqb_eq in E1. cbn [atom_val]. rewrite (wd_32 r E1). apply sext_eqm.
  - destruct (Z.eqb (wd F r) 32) eqn:E1; cbn [andb]; [| split; [reflexivity | apply eqm_refl]].
    destruct (is_some (atom_range F st (r, ES))); cbn [negb]; [split; [reflexivity | apply eqm_refl] |].
    destruct (is_some (atom_range F st (r, EZ))); cbn [fst snd]; [| split; [reflexivity | apply eqm_refl]].
    split; [reflexivity |]. apply Z.eqb_eq in E1. cbn [atom_val]. rewrite (wd_32 r E1).
    apply eqm_sym. apply sext_eqm.
Qed.

(* A lifted form allows the extension of the low 32 bits of every value
   that its input allows. *)
Lemma lift_sound : forall f e g v,
    sat f v -> lift F st f e = Some g ->
    fm g = 64 /\ wf g /\
    ((e = EZ /\ sat g (trunc 32 v)) \/ (e = ES /\ sat g (sext 32 v))).
Proof.
  intros f e g v Hs Hl. unfold lift in Hl.
  destruct (Z.ltb (fm f) 32) eqn:E1; [discriminate |]. apply Z.ltb_ge in E1. rename E1 into Hm.
  set (f' := if Z.eqb (fm f) 32 then mk 32 (swap_ext F st (ft f)) (flo f) (fhi f) else f) in *.
  assert (Hs' : sat f' v /\ 32 <= fm f').
  { unfold f'. destruct (Z.eqb (fm f) 32) eqn:E2.
    - apply Z.eqb_eq in E2. destruct Hs as [x [Hx Hv]]. split; [| rewrite mk_fm; lia].
      apply (sat_mk _ rs 32 _ _ _ _ x); [lia | exact Hx |]. rewrite E2 in Hv.
      eapply eqm_trans; [exact Hv |]. apply eqm_add; [lia | apply eqm_sym, tsum_swap_ext | apply eqm_refl].
    - split; [exact Hs | lia]. }
  destruct Hs' as [[x [Hx Hv]] Hm'].
  destruct (bounded_form_range F st f' (match e with ES => -2^31 | _ => 0 end) (match e with ES => 2^31-1 | _ => 2^32-1 end)) as [[lo hi] |] eqn:Er; [| discriminate].
  pose proof (bounded_form_range_sound F E HE t st rs Hst Hrs f' (match e with ES => -2^31 | _ => 0 end) (match e with ES => 2^31-1 | _ => 2^32-1 end) lo hi x Er Hx) as Hb.
  assert (Hv32 : eqm 32 v (tsum (ft f') + x)) by (eapply eqm_weaken; [| exact Hv]; lia).
  assert (Ht : eqm 32 (trunc 32 v) (tsum (ft f') + x)).
  { eapply eqm_trans; [apply trunc_eqm | exact Hv32]. }
  destruct e; try discriminate.
  - destruct ((0 <=? lo) && (hi <? 2 ^ 32)) eqn:Ec; [| discriminate]. inversion Hl; subst g.
    apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1. apply Z.ltb_lt in Ec2.
    split; [reflexivity |]. split; [unfold wf; cbn; lia |]. left. split; [reflexivity |].
    exists x. cbn [fm ft flo fhi]. split; [exact Hx |].
    pose proof (trunc_range 32 v) as Htr. cbn [Z.of_nat Pos.of_succ_nat] in Htr.
    rewrite (eqm_small_eq 32 32 (trunc 32 v) (tsum (ft f') + x)); [apply eqm_refl | lia | lia | lia | exact Ht].
  - destruct ((- 2 ^ 31 <=? lo) && (hi <? 2 ^ 31)) eqn:Ec; [| discriminate]. inversion Hl; subst g.
    apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1. apply Z.ltb_lt in Ec2.
    split; [reflexivity |]. split; [unfold wf; cbn; lia |]. right. split; [reflexivity |].
    exists x. cbn [fm ft flo fhi]. split; [exact Hx |].
    pose proof (trunc_range 32 v) as Htr. cbn [Z.of_nat Pos.of_succ_nat] in Htr.
    assert (Hsx : sext 32 v = tsum (ft f') + x).
    { replace (sext 32 v) with (sext 32 (trunc 32 v))
        by (unfold sext; rewrite (trunc_small 32 (trunc 32 v)) by exact Htr; reflexivity).
      apply (eqm_sext_eq 32 32); [lia | cbn; lia | exact Htr | cbn; lia | exact Ht]. }
    rewrite Hsx. apply eqm_refl.
Qed.

Lemma tsum_swap_to : forall e ts, e = EZ \/ e = ES -> eqm 32 (tsum (swap_to F e ts)) (tsum ts).
Proof.
  intros e ts He. unfold swap_to. apply tsum_map_eqm; [lia |].
  intros [r e'] c.
  destruct e'; cbn [fst snd]; try (split; [reflexivity | apply eqm_refl]);
    (destruct (Z.eqb (wd F r) 32) eqn:E1; cbn [fst snd]; [| split; [reflexivity | apply eqm_refl]]);
    apply Z.eqb_eq in E1; split; try reflexivity;
    destruct He as [-> | ->]; cbn [atom_val]; rewrite ?(wd_32 r E1);
    solve [apply eqm_refl | apply (sext_eqm 32) | apply eqm_sym, (sext_eqm 32)].
Qed.

Lemma lift_by_range_sound : forall r f e g,
    sat f (rs r) -> lift_by_range F st r f e = Some g ->
    fm g = 64 /\ wf g /\
    ((e = EZ /\ sat g (trunc 32 (rs r))) \/ (e = ES /\ sat g (sext 32 (rs r)))).
Proof.
  intros r f e g Hs H. unfold lift_by_range in H.
  assert (He : e = EZ \/ e = ES) by (destruct e; try discriminate; auto).
  assert (Hmatch : forall A (x y : A), match e with EZ | ES => x | _ => y end = x)
    by (intros; destruct He as [-> | ->]; reflexivity).
  rewrite Hmatch in H.
  destruct (negb (Z.eqb (wd F r) 32) || negb (Z.eqb (fm f) 32)) eqn:Ew; [discriminate |].
  apply orb_false_iff in Ew. destruct Ew as [Ew1 Ew2].
  apply negb_false_iff, Z.eqb_eq in Ew1. apply negb_false_iff, Z.eqb_eq in Ew2.
  set (f' := mk 32 (swap_to F e (ft f)) (flo f) (fhi f)) in H.
  cbv zeta in H.
  set (ar := atom_range_with F (rft_on F (cons_index (acons st)) st
               (range_reads (cons_index (acons st)) (ft f' ++ [((r,e),1)])))) in H.
  assert (Har : forall x a b, ar x = Some (a,b) -> a <= aval x <= b).
  { intros x a b Hx. unfold ar in Hx.
    exact (atom_range_with_sound F E HE rs Hrs _ x a b
      (rft_on_sound F E HE t st rs Hst Hrs _) Hx). }
  set (arn := fun x => match ar x with Some g => Some g | None => natural_range F x end).
  assert (Harn : forall x a b, arn x = Some (a,b) -> a <= aval x <= b).
  { intros x a b Hx. unfold arn in Hx. destruct (ar x) as [[lo hi]|] eqn:Er.
    - injection Hx as <- <-. exact (Har x lo hi Er).
    - exact (natural_range_sound F E HE rs Hrs x a b Hx). }
  fold arn in H.

  (* f' holds of r's value modulo 2^32 *)
  assert (Hs' : sat f' (rs r)).
  { destruct Hs as [x [Hx Hv]]. apply (sat_mk wdn rs 32 _ _ _ _ x); [lia | exact Hx |].
    rewrite Ew2 in Hv. eapply eqm_trans; [exact Hv |].
    apply eqm_add; [lia | apply eqm_sym, tsum_swap_to; exact He | apply eqm_refl]. }
  destruct Hs' as [x [Hx Hv']]. unfold f' in Hv' at 1. rewrite mk_fm in Hv'.
  (* V, r's extension: r's value modulo 2^32, and its atom's value *)
  pose proof (rsr r) as Hr. rewrite (wd_32 r Ew1) in Hr. cbn in Hr.
  set (V := match e with ES => sext 32 (rs r) | _ => trunc 32 (rs r) end).
  assert (HV32 : eqm 32 V (rs r)).
  { unfold V. destruct He as [-> | ->]; [apply (trunc_eqm 32) | apply (sext_eqm 32)]. }
  assert (HVa : V = aval (r, e)).
  { unfold V. destruct He as [-> | ->]; cbn [atom_val]; [rewrite trunc_small by (cbn; lia); reflexivity |].
    rewrite (wd_32 r Ew1). reflexivity. }
  assert (Hnat : fst (match e with ES => (- 2 ^ 31, 2 ^ 31 - 1) | _ => (0, 2 ^ 32 - 1) end) <= V <=
                 snd (match e with ES => (- 2 ^ 31, 2 ^ 31 - 1) | _ => (0, 2 ^ 32 - 1) end)).
  { unfold V. destruct He as [-> | ->]; cbn [fst snd].
    - rewrite trunc_small by (cbn; lia). cbn. lia.
    - unfold sext. rewrite trunc_small by (cbn; lia). cbn.
      destruct (Z.ltb_spec (rs r) 2147483648); lia. }

  set (n := match e with ES => (- 2 ^ 31, 2 ^ 31 - 1) | _ => (0, 2 ^ 32 - 1) end) in *.
  set (bounds := match form_range_with F st ar {| fm := 64; ft := [((r,e),1)]; flo := 0; fhi := 0 |} with
    | Some (a,b) => (Z.max a (fst n), Z.min b (snd n)) | None => n end) in *.
  assert (HB : fst bounds <= V <= snd bounds).
  { unfold bounds. destruct (form_range_with F st ar {| fm := 64; ft := [((r,e),1)]; flo := 0; fhi := 0 |})
      as [[a b]|] eqn:Ea; [|exact Hnat].
    pose proof (form_range_with_sound F E t st rs Hst Hrs ar _ a b 0 Har Ea ltac:(cbn; lia)) as Hab.
    cbn [ft tsum] in Hab. rewrite <- HVa in Hab. cbn [fst snd]. lia. }
  set (al := Z.min 32 (reg_align F st r)) in *.
  assert (HA : aligned al V).
  { apply (aligned_eqm 32 al V (rs r)); [lia|unfold al; lia|exact HV32|].
    eapply aligned_mono; [unfold al; apply Z.le_min_r|apply ras]. }
  destruct bounds as [a0 b0] eqn:EB. cbn [fst snd] in HB.
  pose proof (round_range_sound al a0 b0 V HB HA) as HR.
  destruct (round al (a0,b0)) as [a b] eqn:ER. cbn [fst snd] in HR.
  destruct (form_range_with F st arn f') as [[p q]|] eqn:Ep; [|discriminate].
  pose proof (form_range_with_sound F E t st rs Hst Hrs arn f' p q x Harn Ep Hx) as HS.
  destruct ((q-a <? 2^32) && (b-p <? 2^32)) eqn:Ec; [|discriminate].
  apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.ltb_lt in Ec1,Ec2.
  injection H as <-.
  assert (HVS : V = tsum (ft f') + x).
  { apply (eqm_close 32); [lia|eapply eqm_trans; [exact HV32|exact Hv']|lia]. }
  split; [reflexivity|]. split; [unfold wf; cbn; lia|].
  assert (Hg : sat {| fm := 64; ft := ft f'; flo := flo f'; fhi := fhi f' |} V).
  { exists x. cbn [fm ft flo fhi]. split; [exact Hx|rewrite HVS; apply eqm_refl]. }
  destruct He as [-> | ->]; [left | right]; split; try reflexivity; exact Hg.
Qed.

Lemma lift_reg_sound : forall r f e g,
    sat f (rs r) -> lift_reg F st r f e = Some g ->
    fm g = 64 /\ wf g /\
    ((e = EZ /\ sat g (trunc 32 (rs r))) \/ (e = ES /\ sat g (sext 32 (rs r)))).
Proof.
  intros r f e g Hs H. unfold lift_reg in H.
  destruct (lift F st f e) as [g' |] eqn:El.
  - injection H as <-. exact (lift_sound f e g' (rs r) Hs El).
  - exact (lift_by_range_sound r f e g Hs H).
Qed.

Lemma atom64_sound : forall r e, sat (atom64 r e) (aval (r, e)) /\ fm (atom64 r e) = 64.
Proof.
  intros r e. unfold atom64. split; [| unfold fatom; apply mk_fm].
  rewrite <- (Z.mul_1_l (aval (r, e))). apply sat_fatom. lia.
Qed.

Lemma as_form_sound : forall r e, sat (as_form F st r e) (aval (r, e)) /\ fm (as_form F st r e) = 64.
Proof.
  intros r e. unfold as_form.
  destruct (interval_only (get st r)); [apply atom64_sound |].
  destruct (get st r) as [x |] eqn:Eg; [| apply atom64_sound].
  destruct x; try apply atom64_sound.
  pose proof (form_fact r f al vr Eg) as [Hwf [Hs _]].
  destruct e; try apply atom64_sound.
  - destruct (Z.eqb (fm f) 64 && Z.eqb (wd F r) 64) eqn:E1; [| apply atom64_sound].
    apply andb_true_iff in E1. destruct E1 as [E1 _]. apply Z.eqb_eq in E1. split; [exact Hs | exact E1].
  - destruct (Z.eqb (wd F r) 32) eqn:E1; [| apply atom64_sound]. apply Z.eqb_eq in E1.
    destruct (lift_reg F st r (fat f 32) EZ) as [g |] eqn:El; [| apply atom64_sound].
    pose proof (sat_fat _ rs f 32 (rs r) Hwf ltac:(lia) Hs) as Hs32.
    destruct (lift_reg_sound r _ EZ g Hs32 El) as [Hg [_ [[_ Hz] | [He _]]]]; [| discriminate].
    split; [| exact Hg]. cbn [atom_val].
    pose proof (rsr r) as Hr. rewrite (wd_32 r E1) in Hr.
    rewrite trunc_small in Hz by exact Hr. exact Hz.
  - destruct (Z.eqb (wd F r) 32) eqn:E1; [| apply atom64_sound]. apply Z.eqb_eq in E1.
    destruct (lift_reg F st r (fat f 32) ES) as [g |] eqn:El; [| apply atom64_sound].
    pose proof (sat_fat _ rs f 32 (rs r) Hwf ltac:(lia) Hs) as Hs32.
    destruct (lift_reg_sound r _ ES g Hs32 El) as [Hg [_ [[He _] | [_ Hz]]]]; [discriminate |].
    split; [| exact Hg]. cbn [atom_val]. rewrite (wd_32 r E1). exact Hz.
Qed.

Lemma as_form_wf : forall r e, wf (as_form F st r e).
Proof. intros r e. unfold wf. rewrite (proj2 (as_form_sound r e)). lia. Qed.

Lemma narrow_sound : forall r w rel,
    0 <= w <= 64 -> sat (narrow st r w rel) (rs r) /\ wf (narrow st r w rel) /\ fm (narrow st r w rel) <= w.
Proof.
  intros r w rel Hw. unfold narrow.
  assert (Hatom : sat (fatom w (r, EZ) 1) (rs r) /\ wf (fatom w (r, EZ) 1) /\ fm (fatom w (r, EZ) 1) <= w).
  { unfold fatom, wf. rewrite mk_fm. split; [| lia].
    rewrite <- (Z.mul_1_l (rs r)). change (rs r) with (aval (r, EZ)). apply sat_fatom. lia. }
  destruct (get st r) as [x |] eqn:Eg; [| exact Hatom].
  destruct x; try exact Hatom.
  pose proof (form_fact r f al vr Eg) as [Hwf [Hs _]]. unfold wf in Hwf.
  assert (Hat : sat (fat f w) (rs r) /\ wf (fat f w) /\ fm (fat f w) <= w).
  { unfold fat, wf. rewrite mk_fm. split; [| lia]. apply sat_fat; [exact Hwf | lia | exact Hs]. }
  destruct (rel && negb (null (ft f))); [exact Hat |].
  destruct (negb (interval_only (Some (FForm f al vr)))); [exact Hat |].
  destruct (null (ft f) && negb (is_some vr) && Z.ltb (fm f) w) eqn:E1; [| exact Hatom].
  apply andb_true_iff in E1. destruct E1 as [_ E1]. apply Z.ltb_lt in E1.
  split; [exact Hs | split; [exact Hwf | lia]].
Qed.

Lemma opnd_form_sound : forall o w,
    0 <= w <= 64 -> sat (opnd_form F st o w) (opv rs o) /\ wf (opnd_form F st o w) /\ fm (opnd_form F st o w) <= w.
Proof.
  intros [r | z] w Hw; unfold opnd_form; cbn [opv].
  - destruct (Z.eqb w 64) eqn:E1.
    + apply Z.eqb_eq in E1. subst. destruct (as_form_sound r EX) as [Hs Hm].
      split; [exact Hs | unfold wf; lia].
    + apply narrow_sound. exact Hw.
  - split; [| unfold fconst, wf; cbn; lia].
    apply sat_fconst. apply eqm_sym. apply eqm_mod. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* Copies and shifts.                                                  *)

Lemma copy_of_shape : forall r x r2,
    copy_of F r x = Some r2 ->
    exists f al vr e', x = Some (FForm f al vr) /\ wd F r <= fm f /\ flo f = 0 /\ fhi f = 0 /\
                       ft f = [((r2, e'), 1)] /\ (e' = EX \/ e' = EZ) /\ wd F r2 <= wd F r.
Proof.
  intros r x r2 H. unfold copy_of in H.
  destruct x as [[] |]; try discriminate.
  destruct ((wd F r <=? fm f) && Z.eqb (flo f) 0 && Z.eqb (fhi f) 0) eqn:E1; [| discriminate].
  rewrite !andb_true_iff in E1. destruct E1 as [[E1 E2] E3].
  apply Z.leb_le in E1. apply Z.eqb_eq in E2. apply Z.eqb_eq in E3.
  destruct (ft f) as [| [[r' ex] c] l] eqn:Et; [discriminate |].
  destruct l as [| y l]; [| destruct ex; destruct c as [| cp | cp]; try destruct cp; discriminate].
  destruct ex; try discriminate; destruct c as [| cp | cp]; try discriminate; destruct cp; try discriminate;
    destruct (wd F r' <=? wd F r) eqn:Ew; try discriminate; inversion H; subst r';
    apply Z.leb_le in Ew.
  - exists f, al, vr, EX. repeat split; auto.
  - exists f, al, vr, EZ. repeat split; auto.
Qed.

Lemma copy_of_sound : forall r r2, copy_of F r (get st r) = Some r2 -> rs r2 = rs r.
Proof.
  intros r r2 H. destruct (copy_of_shape r _ r2 H) as [f [al [vr [e' [Hx [H1 [H2 [H3 [H4 [H5 H6]]]]]]]]]].
  pose proof (form_fact r f al vr Hx) as [Hwf [[e [He Hs]] _]]. unfold wf in Hwf.
  assert (e = 0) by lia. subst e. rewrite H4 in Hs. cbn [tsum] in Hs.
  assert (Hv : atom_val wdn rs (r2, e') = rs r2) by (destruct H5; subst e'; reflexivity).
  rewrite Hv in Hs. pose proof (rsr r) as Hr. pose proof (rsr r2) as Hr2. unfold wd in *.
  assert (2 ^ Z.of_nat (wdn r2) <= 2 ^ Z.of_nat (wdn r)) by (apply Z.pow_le_mono_r; lia).
  symmetry. apply (eqm_small_eq (fm f) (Z.of_nat (wdn r))); [lia | lia | lia |].
  eapply eqm_trans; [exact Hs |]. unfold eqm. f_equal. ring.
Qed.

Lemma aligned_pos : forall k v, 0 < k -> aligned k v -> (2 ^ k | v).
Proof. unfold aligned. intros k v Hk H. rewrite Z.max_r in H by lia. exact H. Qed.

Lemma fold_max_le : forall ks k0 b, k0 <= b -> (forall k, In k ks -> k <= b) -> fold_left Z.max ks k0 <= b.
Proof.
  induction ks as [| k ks IH]; intros k0 b H0 H; cbn [fold_left]; [exact H0 |].
  apply IH; [apply Z.max_lub; [exact H0 | apply H; left; reflexivity] | intros; apply H; right; assumption].
Qed.

(* Rewriting r's shifts below k as its shift by k keeps every term's value
   when r's value is a multiple of 2^k. *)
Lemma tsum_shift_map : forall r k ts,
    k <= reg_align F st r ->
    tsum (map (fun t0 => let '((r', e), c) := t0 in
                 if Pos.eqb r r' then
                   match shr_of e with
                   | Some j => if (0 <=? j) && (j <? k) then ((r, EShr k), c * 2 ^ (k - j)) else t0
                   | None => t0
                   end
                 else t0) ts) = tsum ts.
Proof.
  intros r k ts Hk.
  induction ts as [| [[r' e] c] ts IH]; [reflexivity |].
  cbn [map tsum]. rewrite IH.
  destruct (Pos.eqb r r') eqn:Er; [| reflexivity]. apply Pos.eqb_eq in Er. subst r'.
  destruct (shr_of e) as [j |] eqn:Ej; [| reflexivity].
  destruct ((0 <=? j) && (j <? k)) eqn:Ejk; [| reflexivity].
  apply andb_true_iff in Ejk. destruct Ejk as [Ej0 Ejk]. apply Z.leb_le in Ej0. apply Z.ltb_lt in Ejk.
  cbn [tsum]. f_equal.
  assert (Hdiv : (2 ^ k | rs r)) by (apply aligned_pos; [lia | apply (aligned_mono (reg_align F st r)); [exact Hk | apply ras]]).
  destruct Hdiv as [q Hq].
  assert (Hpk : 2 ^ k = 2 ^ (k - j) * 2 ^ j) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
  assert (Hp : 0 < 2 ^ j) by (apply pow_nonneg_pos; lia).
  assert (Hp2 : 0 < 2 ^ (k - j)) by (apply pow_nonneg_pos; lia).
  cbn [atom_val].
  replace (rs r / 2 ^ k) with q by (rewrite Hq; rewrite Z.div_mul; [reflexivity | lia]).
  destruct e; cbn [shr_of] in Ej; inversion Ej; subst j; cbn [atom_val].
  - rewrite Hq, Hpk. rewrite Z.pow_0_r. rewrite Z.sub_0_r. ring.
  - rewrite Hq, Hpk. rewrite Z.mul_assoc, Z.div_mul by lia. ring.
Qed.

Lemma tsum_merge_reg : forall ts r, tsum (merge_reg F st ts r) = tsum ts.
Proof.
  intros ts r. unfold merge_reg.
  set (al := reg_align F st r).
  set (ks := flat_map (fun t0 => let '((r', e), _) := t0 in
                        if Pos.eqb r r' then match e with EShr k => if k <=? al then [k] else [] | _ => [] end
                        else []) ts).
  destruct ks as [| k0 ks'] eqn:Eks; [reflexivity |].
  set (k := fold_left Z.max (k0 :: ks') k0).
  assert (Hk : k <= al).
  { unfold k. apply fold_max_le.
    - assert (In k0 ks) by (rewrite Eks; left; reflexivity).
      unfold ks in H. apply in_flat_map in H. destruct H as [[[r' e] c] [_ Hin]].
      destruct (Pos.eqb r r'); [| inversion Hin]. destruct e; try inversion Hin.
      destruct (k1 <=? al) eqn:Ek; [| inversion Hin]. destruct Hin as [<- | []]. apply Z.leb_le. exact Ek.
    - intros k' Hin. rewrite <- Eks in Hin. unfold ks in Hin. apply in_flat_map in Hin.
      destruct Hin as [[[r' e] c] [_ Hin]].
      destruct (Pos.eqb r r'); [| inversion Hin]. destruct e; try inversion Hin.
      destruct (k1 <=? al) eqn:Ek; [| inversion Hin]. destruct Hin as [<- | []]. apply Z.leb_le. exact Ek. }
  apply tsum_shift_map. exact Hk.
Qed.

Lemma tsum_merge_fold : forall rl ts, tsum (fold_left (merge_reg F st) rl ts) = tsum ts.
Proof.
  induction rl as [| r rl IH]; intros ts; cbn [fold_left]; [reflexivity |].
  rewrite IH. apply tsum_merge_reg.
Qed.

Lemma merge_shifts_iff : forall f v, 0 <= fm f -> (sat (merge_shifts F st f) v <-> sat f v).
Proof.
  intros f v Hm. unfold merge_shifts. split.
  - intros H. apply sat_mk_inv in H; [| exact Hm]. destruct H as [e [He Hv]].
    rewrite tsum_merge_fold in Hv. exists e. split; assumption.
  - intros [e [He Hv]]. apply (sat_mk _ rs (fm f) _ _ _ _ e Hm He). rewrite tsum_merge_fold. exact Hv.
Qed.

Lemma merge_shifts_fm : forall f, fm (merge_shifts F st f) = fm f.
Proof. intros. unfold merge_shifts. apply mk_fm. Qed.

(* ------------------------------------------------------------------ *)
(* Normalization.                                                      *)

(* The form a term becomes allows the term's value; in the exact mode it
   has a single constant, so it allows nothing else. *)
Definition term_ok (ex : bool) (m : Z) (g : form) (x : atom) (c : Z) : Prop :=
  wf g /\ m <= fm g /\ sat g (c * aval x) /\
  (ex = true -> flo g = fhi g /\ eqm (fm g) (tsum (ft g) + flo g) (c * aval x)).

Lemma fatom_term_ok : forall ex m x c, 0 <= m <= 64 -> term_ok ex m (fatom m x c) x c.
Proof.
  intros ex m x c Hm. unfold term_ok, fatom, wf. rewrite mk_fm.
  split; [lia | split; [lia | split; [apply sat_fatom; lia |]]].
  intros _. unfold mk. pose proof (tsum_canon wdn rs m [(x, c)] ltac:(lia)) as Hc.
  cbn [tsum] in Hc. rewrite Z.add_0_r in Hc.
  destruct (canon m [(x, c)]) as [| y l] eqn:E1; cbn [fm ft flo fhi].
  - split; [reflexivity |]. rewrite Z.add_0_r. exact Hc.
  - rewrite Z.eqb_refl. cbn [fm ft flo fhi]. split; [reflexivity |].
    assert (rep m 0 = 0) by (unfold rep; rewrite !pow2_eq, Z.mod_0_l by (pose proof (pow_nonneg_pos m); lia);
                             destruct (Z.ltb_spec (2 ^ (m - 1)) 0); [pose proof (pow_nonneg_pos (m - 1)) | ]; lia).
    rewrite H, Z.add_0_r. exact Hc.
Qed.

Lemma scale_term_ok : forall ex m f c x,
    0 <= m <= 64 -> wf f -> fm f = 64 -> sat f (aval x) -> (ex = true -> flo f = fhi f) ->
    term_ok ex m (fscale f c 64) x c.
Proof.
  intros ex m f c x Hm Hwf Hf Hs Hp. unfold term_ok.
  assert (Hfm : fm (fscale f c 64) = 64).
  { unfold fscale. rewrite mk_fm. destruct (Z.eqb c 0); [reflexivity | pose proof (v2_nonneg c); lia]. }
  split; [apply wf_fscale; [exact Hwf | lia] |]. rewrite Hfm. split; [lia |].
  split; [apply sat_fscale; [exact Hwf | lia | exact Hs] |].
  intros Hex. specialize (Hp Hex). destruct Hs as [e [He Hv]]. assert (e = flo f) by lia. subst e.
  unfold fscale. rewrite Hf in *.
  set (m' := if Z.eqb c 0 then 64 else Z.min 64 (64 + v2 c)).
  assert (Hm' : m' = 64) by (unfold m'; destruct (Z.eqb c 0); [reflexivity | pose proof (v2_nonneg c); lia]).
  rewrite Hm'. rewrite <- Hp.
  rewrite Z.min_id, Z.max_id.
  assert (Hmk : forall ts z, flo (mk 64 ts z z) = fhi (mk 64 ts z z) /\
                             eqm 64 (tsum (ft (mk 64 ts z z)) + flo (mk 64 ts z z)) (tsum ts + z)).
  { intros ts z. unfold mk. pose proof (tsum_canon wdn rs 64 ts ltac:(lia)) as Hc.
    destruct (canon 64 ts); cbn [ft flo fhi].
    - split; [reflexivity |]. apply eqm_add; [lia | exact Hc | apply eqm_refl].
    - rewrite Z.eqb_refl. cbn [ft flo fhi]. split; [reflexivity |].
      apply eqm_add; [lia | exact Hc | apply rep_eqm; lia]. }
  destruct (Hmk (tscale c (ft f)) (flo f * c)) as [Hpt Heq]. split; [exact Hpt |].
  eapply eqm_trans; [exact Heq |]. rewrite tsum_tscale.
  replace (c * tsum (ft f) + flo f * c) with (c * (tsum (ft f) + flo f)) by ring.
  apply eqm_mul; [lia |]. apply eqm_sym. exact Hv.
Qed.

(* An atom of a register fixed with one launch value is that value's. *)
Lemma fixed_atom_sound : forall r e v, fixed_atom F r e = Some v -> aval (r, e) = v.
Proof.
  intros r e v H. unfold fixed_atom in H.
  destruct (is_fixed F r) eqn:Ef; [| discriminate].
  destruct (lookup (lf_bounds F) r) as [[lo hi] |] eqn:Eb; [| discriminate].
  destruct (Z.eqb_spec lo hi) as [<- |]; [| discriminate].
  pose proof (ok_bounds F E HE r lo lo Ef Eb) as Hb. rewrite <- (proj2 Hrs r Ef) in Hb.
  assert (Hr : rs r = lo) by lia.
  injection H as <-. destruct e; cbn [atom_val]; rewrite Hr; reflexivity.
Qed.

Lemma signed_term_ok : forall ex m r c,
    0 <= m <= 64 -> wd F r = 64 -> term_ok ex m (fatom m (r, ES) c) (r, EX) c.
Proof.
  intros ex m r c Hm Hw.
  destruct (fatom_term_ok ex m (r, ES) c Hm) as [Hg [Hgm [Hs Hback]]].
  assert (Heq : eqm (fm (fatom m (r, ES) c)) (c * aval (r, ES)) (c * aval (r, EX))).
  { unfold fatom. rewrite mk_fm. apply eqm_mul; [lia |].
    apply (eqm_weaken (wd F r)); [lia |]. apply sext_eqm. }
  split; [exact Hg |]. split; [exact Hgm |].
  split; [exact (sat_eqm wdn rs _ _ _ Hs Heq) |].
  intros Hex. destruct (Hback Hex) as [Hpoint Hback']. split; [exact Hpoint |].
  eapply eqm_trans; eauto.
Qed.

Lemma norm_term_ok : forall ex m x c, 0 <= m <= 64 -> term_ok ex m (fst (norm_term F ex st m x c)) x c.
Proof.
  intros ex m [r e] c Hm. unfold norm_term. cbn [fst].
  destruct (fixed_atom F r e) as [v |] eqn:Efx.
  { (* a register fixed with one launch value: the constant *)
    cbn [fst]. pose proof (fixed_atom_sound r e v Efx) as Hv.
    unfold term_ok. cbn [fm ft flo fhi]. split; [unfold wf; cbn [fm]; lia |]. split; [lia |].
    split; [exists (c * v); cbn [flo fhi ft tsum]; split; [lia | rewrite Hv, Z.add_0_l; apply eqm_refl] |].
    intros _. split; [reflexivity |]. cbn [tsum]. rewrite Hv, Z.add_0_l. apply eqm_refl. }
  (* a shift or field of a copy is that of the original *)
  assert (Hcopy : forall e', match e' with EShr _ | EBf _ _ => True | _ => False end ->
            term_ok ex m (fst (match copy_of F r (get st r) with
                               | Some r2 => (fatom m (r2, e') c, true)
                               | None => (fatom m (r, e') c, false)
                               end)) (r, e') c).
  { intros e' He'. destruct (copy_of F r (get st r)) as [r2 |] eqn:Ec; cbn [fst]; [| apply fatom_term_ok; exact Hm].
    pose proof (copy_of_sound r r2 Ec) as Hr2.
    destruct (fatom_term_ok ex m (r2, e') c Hm) as [H1 [H2 [H3 H4]]].
    assert (Hv : atom_val wdn rs (r2, e') = atom_val wdn rs (r, e'))
      by (destruct e'; try contradiction; cbn [atom_val]; rewrite Hr2; reflexivity).
    unfold term_ok. rewrite <- Hv. tauto. }
  destruct (interval_only (get st r)).
  { destruct e; try (apply fatom_term_ok; exact Hm).
    destruct (Z.eqb_spec (wd F r) 64) as [Hw |]; [| apply fatom_term_ok; exact Hm].
    destruct (match fact_range F r (get st r) with
              | Some (lo, hi) => (0 <=? lo) && (hi <? 2 ^ 63)
              | None => false
              end); [apply fatom_term_ok; exact Hm |].
    destruct (sreg_range F st r) as [[lo hi] |]; [| apply fatom_term_ok; exact Hm].
    destruct (lo <? 0); [apply signed_term_ok; assumption | apply fatom_term_ok; exact Hm]. }
  destruct (get st r) as [y |] eqn:Eg.
  - destruct y;
      try (destruct e; [apply fatom_term_ok; exact Hm | apply fatom_term_ok; exact Hm |
                        apply fatom_term_ok; exact Hm | apply Hcopy; exact I | apply Hcopy; exact I]).
    pose proof (form_fact r f al vr Eg) as [Hwf [Hs _]].
    destruct e.
    + destruct (Z.eqb (fm f) 64 && (negb ex || Z.eqb (flo f) (fhi f))) eqn:E1; cbn [fst];
        [| apply fatom_term_ok; exact Hm].
      apply andb_true_iff in E1. destruct E1 as [E1 E2]. apply Z.eqb_eq in E1.
      apply scale_term_ok; [exact Hm | exact Hwf | exact E1 | exact Hs |].
      intros ->. cbn in E2. apply Z.eqb_eq. exact E2.
    + destruct (negb (Z.eqb (wd F r) 32)) eqn:E0; cbn [fst]; [apply fatom_term_ok; exact Hm |].
      apply negb_false_iff in E0. apply Z.eqb_eq in E0.
      destruct (lift_reg F st r (fat f 32) EZ) as [h |] eqn:El; [| apply fatom_term_ok; exact Hm].
      destruct (negb ex || Z.eqb (flo h) (fhi h)) eqn:E2; cbn [fst]; [| apply fatom_term_ok; exact Hm].
      pose proof (sat_fat _ rs f 32 (rs r) Hwf ltac:(lia) Hs) as Hs32.
      destruct (lift_reg_sound r _ EZ h Hs32 El) as [Hh [Hwh [[_ Hz] | [He _]]]]; [| discriminate].
      pose proof (rsr r) as Hr. rewrite (wd_32 r E0) in Hr. rewrite trunc_small in Hz by exact Hr.
      apply scale_term_ok; [exact Hm | exact Hwh | exact Hh | exact Hz |].
      intros ->. cbn in E2. apply Z.eqb_eq. exact E2.
    + destruct (Z.eqb_spec (wd F r) 64) as [Hw64 | Hw64].
      { destruct (form_range F st f) as [rg |]; [| apply fatom_term_ok; exact Hm].
        destruct (Z.eqb (fm f) 64 && sext_ok 64 rg && (negb ex || Z.eqb (flo f) (fhi f))) eqn:Ec;
          cbn [fst]; [| apply fatom_term_ok; exact Hm].
        rewrite !andb_true_iff in Ec. destruct Ec as [[Hfm _] Hp]. apply Z.eqb_eq in Hfm.
        apply scale_term_ok; [exact Hm | exact Hwf | exact Hfm | |].
        - apply (sat_eqm wdn rs f (rs r)); [exact Hs |]. rewrite Hfm, <- Hw64.
          apply eqm_sym. apply sext_eqm.
        - intros ->. cbn in Hp. apply Z.eqb_eq. exact Hp. }
      destruct (negb (Z.eqb (wd F r) 32)) eqn:E0; cbn [fst]; [apply fatom_term_ok; exact Hm |].
      apply negb_false_iff in E0. apply Z.eqb_eq in E0.
      destruct (lift_reg F st r (fat f 32) ES) as [h |] eqn:El; [| apply fatom_term_ok; exact Hm].
      destruct (negb ex || Z.eqb (flo h) (fhi h)) eqn:E2; cbn [fst]; [| apply fatom_term_ok; exact Hm].
      pose proof (sat_fat _ rs f 32 (rs r) Hwf ltac:(lia) Hs) as Hs32.
      destruct (lift_reg_sound r _ ES h Hs32 El) as [Hh [Hwh [[He _] | [_ Hz]]]]; [discriminate |].
      apply scale_term_ok; [exact Hm | exact Hwh | exact Hh | cbn [atom_val]; rewrite (wd_32 r E0); exact Hz |].
      intros ->. cbn in E2. apply Z.eqb_eq. exact E2.
    + apply Hcopy. exact I.
    + apply Hcopy. exact I.
  - destruct e; [apply fatom_term_ok; exact Hm | apply fatom_term_ok; exact Hm |
                 apply fatom_term_ok; exact Hm | apply Hcopy; exact I | apply Hcopy; exact I].
Qed.

Definition norm_step (ex : bool) (m : Z) (acc : form * bool) (t0 : atom * Z) : form * bool :=
  let '(o, ch) := acc in let '(x, c) := t0 in
  let '(g, b) := norm_term F ex st m x c in (fadd o g, ch || b).

Lemma norm_fold_fwd : forall ex m L acc ch v,
    0 <= m <= 64 -> wf acc -> fm acc <= m -> sat acc v ->
    let res := fold_left (norm_step ex m) L (acc, ch) in
    wf (fst res) /\ fm (fst res) = fm acc /\ sat (fst res) (v + tsum L).
Proof.
  intros ex m L. induction L as [| [x c] L IH]; intros acc ch v Hm Hwf Hfm Hs; cbn [fold_left tsum].
  - rewrite Z.add_0_r. auto.
  - pose proof (norm_term_ok ex m x c Hm) as Ht.
    destruct (norm_term F ex st m x c) as [g b] eqn:En. cbn [fst] in Ht.
    assert (Hstep : norm_step ex m (acc, ch) (x, c) = (fadd acc g, ch || b))
      by (unfold norm_step; rewrite En; reflexivity).
    rewrite Hstep. destruct Ht as [Hwg [Hfg [Hsg _]]].
    assert (Hwa : wf (fadd acc g)) by (apply wf_fadd; assumption).
    assert (Hfa : fm (fadd acc g) = fm acc) by (unfold fadd; rewrite mk_fm; lia).
    destruct (IH (fadd acc g) (ch || b) (v + c * aval x) Hm Hwa ltac:(lia)
                 (sat_fadd _ rs acc g v (c * aval x) Hwf Hwg Hs Hsg)) as [H1 [H2 H3]].
    split; [exact H1 | split; [lia |]]. replace (v + (c * aval x + tsum L)) with (v + c * aval x + tsum L) by ring.
    exact H3.
Qed.

Lemma norm_fold_bwd : forall m L acc ch w,
    0 <= m <= 64 -> wf acc -> fm acc <= m ->
    sat (fst (fold_left (norm_step true m) L (acc, ch))) w -> sat acc (w - tsum L).
Proof.
  intros m L. induction L as [| [x c] L IH]; intros acc ch w Hm Hwf Hfm Hs; cbn [fold_left tsum] in *.
  - rewrite Z.sub_0_r. exact Hs.
  - pose proof (norm_term_ok true m x c Hm) as Ht.
    destruct (norm_term F true st m x c) as [g b] eqn:En. cbn [fst] in Ht.
    assert (Hstep : norm_step true m (acc, ch) (x, c) = (fadd acc g, ch || b))
      by (unfold norm_step; rewrite En; reflexivity).
    rewrite Hstep in Hs. destruct Ht as [Hwg [Hfg [_ Hex]]]. destruct (Hex eq_refl) as [Hpt Heq].
    assert (Hwa : wf (fadd acc g)) by (apply wf_fadd; assumption).
    assert (Hfa : fm (fadd acc g) = fm acc) by (unfold fadd; rewrite mk_fm; lia).
    specialize (IH (fadd acc g) (ch || b) w Hm Hwa ltac:(lia) Hs).
    unfold fadd in IH. unfold wf in *. apply sat_mk_inv in IH; [| lia].
    destruct IH as [e [He Hv]]. rewrite tsum_app in Hv.
    exists (e - flo g). split; [lia |].
    replace (Z.min (fm acc) (fm g)) with (fm acc) in Hv by lia.
    (* w - tsum L = tsum acc + (tsum g + flo g) + (e - flo g), and tsum g + flo g is c * x *)
    assert (Hg : eqm (fm acc) (tsum (ft g) + flo g) (c * aval x)) by (eapply eqm_weaken; [| exact Heq]; lia).
    assert (H1 : eqm (fm acc) (w - tsum L) (tsum (ft acc) + (e - flo g) + (tsum (ft g) + flo g))).
    { eapply eqm_trans; [exact Hv |]. unfold eqm. f_equal. ring. }
    assert (H2 : eqm (fm acc) (w - tsum L) (tsum (ft acc) + (e - flo g) + c * aval x)).
    { eapply eqm_trans; [exact H1 |]. apply eqm_add; [lia | apply eqm_refl | exact Hg]. }
    replace (w - (c * aval x + tsum L)) with ((w - tsum L) - c * aval x) by ring.
    apply eqm_sub_l; [lia | exact H2].
Qed.

Lemma norm_once_fwd : forall ex f v,
    wf f -> sat f v -> wf (fst (norm_once F ex st f)) /\ fm (fst (norm_once F ex st f)) = fm f /\
                       sat (fst (norm_once F ex st f)) v.
Proof.
  intros ex f v Hwf [e [He Hv]]. unfold norm_once. fold (norm_step ex (fm f)).
  pose proof (Hwf) as Hwf'. unfold wf in Hwf'.
  set (acc0 := {| fm := fm f; ft := []; flo := flo f; fhi := fhi f |}).
  assert (Hs0 : sat acc0 e)
    by (exists e; unfold acc0; cbn [fm ft flo fhi tsum]; split; [exact He | unfold eqm; f_equal; ring]).
  destruct (norm_fold_fwd ex (fm f) (ft f) acc0 false e Hwf' ltac:(unfold acc0, wf; cbn; lia) ltac:(cbn; lia) Hs0)
    as [H1 [H2 H3]].
  destruct (fold_left (norm_step ex (fm f)) (ft f) (acc0, false)) as [out ch] eqn:Eo. cbn [fst] in *.
  cbn [fm acc0] in H2.
  assert (Hso : sat out v) by (apply (sat_eqm _ rs out (e + tsum (ft f))); [exact H3 | rewrite H2; apply eqm_sym;
                                  eapply eqm_trans; [exact Hv | unfold eqm; f_equal; ring]]).
  assert (Hfat : sat (fat out (fm f)) v) by (apply sat_fat; [exact H1 | lia | exact Hso]).
  split; [unfold wf; rewrite merge_shifts_fm; unfold fat; rewrite mk_fm; lia |].
  split; [rewrite merge_shifts_fm; unfold fat; rewrite mk_fm; lia |].
  apply merge_shifts_iff; [unfold fat; rewrite mk_fm; lia | exact Hfat].
Qed.

Lemma norm_fold_fm : forall ex m L acc ch,
    0 <= m <= 64 -> wf acc -> fm acc <= m ->
    wf (fst (fold_left (norm_step ex m) L (acc, ch))) /\ fm (fst (fold_left (norm_step ex m) L (acc, ch))) = fm acc.
Proof.
  intros ex m L. induction L as [| [x c] L IH]; intros acc ch Hm Hwf Hfm; cbn [fold_left]; [auto |].
  pose proof (norm_term_ok ex m x c Hm) as Ht.
  destruct (norm_term F ex st m x c) as [g b] eqn:En. cbn [fst] in Ht.
  assert (Hstep : norm_step ex m (acc, ch) (x, c) = (fadd acc g, ch || b))
    by (unfold norm_step; rewrite En; reflexivity).
  rewrite Hstep. destruct Ht as [Hwg [Hfg _]].
  assert (Hwa : wf (fadd acc g)) by (apply wf_fadd; assumption).
  assert (Hfa : fm (fadd acc g) = fm acc) by (unfold fadd; rewrite mk_fm; unfold wf in *; lia).
  destruct (IH (fadd acc g) (ch || b) Hm Hwa ltac:(lia)) as [H1 H2]. split; [exact H1 | lia].
Qed.

Lemma norm_once_bwd : forall f v, wf f -> sat (fst (norm_once F true st f)) v -> sat f v.
Proof.
  intros f v Hwf Hs. unfold norm_once in Hs. fold (norm_step true (fm f)) in Hs.
  pose proof (Hwf) as Hwf'. unfold wf in Hwf'.
  set (acc0 := {| fm := fm f; ft := []; flo := flo f; fhi := fhi f |}) in *.
  assert (Hwa : wf acc0) by (unfold acc0, wf; cbn; lia).
  pose proof (norm_fold_fm true (fm f) (ft f) acc0 false Hwf' Hwa ltac:(cbn; lia)) as [Hwo Hfo].
  pose proof (norm_fold_bwd (fm f) (ft f) acc0 false) as Hbwd.
  destruct (fold_left (norm_step true (fm f)) (ft f) (acc0, false)) as [out ch] eqn:Eo. cbn [fst] in *.
  cbn [fm acc0] in Hfo.
  assert (Hfm0 : 0 <= fm (fat out (fm f))) by (unfold fat; rewrite mk_fm; lia).
  apply (proj1 (merge_shifts_iff (fat out (fm f)) v Hfm0)) in Hs.
  unfold fat in Hs. apply sat_mk_inv in Hs; [| lia]. destruct Hs as [e [He Hv]].
  assert (Hso : sat out v) by (exists e; split; [exact He | rewrite Hfo; replace (fm f) with (Z.min (fm f) (fm out)) by lia; exact Hv]).
  specialize (Hbwd v Hwf' Hwa ltac:(cbn; lia) Hso).
  destruct Hbwd as [e' [He' Hv']]. unfold acc0 in He', Hv'. cbn [fm ft flo fhi tsum] in He', Hv'.
  exists e'. split; [exact He' |].
  replace v with ((v - tsum (ft f)) + tsum (ft f)) by ring.
  replace (tsum (ft f) + e') with ((0 + e') + tsum (ft f)) by ring.
  apply eqm_add; [lia | exact Hv' | apply eqm_refl].
Qed.

Lemma normalize_n_fwd : forall ex n f v,
    wf f -> sat f v -> wf (normalize_n F ex n st f) /\ fm (normalize_n F ex n st f) = fm f /\ sat (normalize_n F ex n st f) v.
Proof.
  intros ex n. induction n as [| n IH]; intros f v Hwf Hs; cbn [normalize_n]; [auto |].
  destruct (norm_once_fwd ex f v Hwf Hs) as [H1 [H2 H3]].
  destruct (norm_once F ex st f) as [g ch] eqn:En. cbn [fst] in *.
  destruct ch; [| auto].
  destruct (IH g v H1 H3) as [H4 [H5 H6]]. split; [exact H4 | split; [lia | exact H6]].
Qed.

Lemma normalize_n_bwd : forall n f v, wf f -> sat (normalize_n F true n st f) v -> sat f v.
Proof.
  intros n. induction n as [| n IH]; intros f v Hwf Hs; cbn [normalize_n] in Hs; [exact Hs |].
  pose proof (norm_once_fwd true f 0 Hwf) as Hinfo.
  pose proof (norm_once_bwd f) as Hb.
  destruct (norm_once F true st f) as [g ch] eqn:En. cbn [fst] in *.
  assert (Hwg : wf g).
  { unfold norm_once in En. fold (norm_step true (fm f)) in En. pose proof Hwf as Hwf'. unfold wf in Hwf'.
    assert (Hwa : wf {| fm := fm f; ft := []; flo := flo f; fhi := fhi f |}) by (unfold wf; cbn; lia).
    pose proof (norm_fold_fm true (fm f) (ft f) _ false Hwf' Hwa ltac:(cbn; lia)) as [_ Hfo].
    destruct (fold_left (norm_step true (fm f)) (ft f) ({| fm := fm f; ft := []; flo := flo f; fhi := fhi f |}, false))
      as [out c'] eqn:Eo. cbn [fst fm] in Hfo.
    inversion En; subst g. unfold wf. rewrite merge_shifts_fm. unfold fat. rewrite mk_fm. lia. }
  destruct ch; [apply Hb; [exact Hwf | apply IH; [exact Hwg | exact Hs]] | apply Hb; [exact Hwf | exact Hs]].
Qed.

Lemma normalize_sound : forall f v, wf f -> sat f v ->
    wf (normalize F st f) /\ fm (normalize F st f) = fm f /\ sat (normalize F st f) v.
Proof. intros. apply normalize_n_fwd; assumption. Qed.

Lemma normalize_exact_bwd : forall f v, wf f -> sat (normalize_exact F st f) v -> sat f v.
Proof. intros f v Hwf Hs. exact (normalize_n_bwd 12 f v Hwf Hs). Qed.

Lemma normalize_exact_fm : forall f, wf f -> wf (normalize_exact F st f) /\ fm (normalize_exact F st f) = fm f.
Proof.
  intros f Hwf. unfold normalize_exact.
  assert (H : forall n g, wf g -> wf (normalize_n F true n st g) /\ fm (normalize_n F true n st g) = fm g).
  { intros n. induction n as [| n IH]; intros g Hwg; cbn [normalize_n]; [auto |].
    assert (Hwo : wf (fst (norm_once F true st g)) /\ fm (fst (norm_once F true st g)) = fm g).
    { unfold norm_once. fold (norm_step true (fm g)). pose proof Hwg as Hwg'. unfold wf in Hwg'.
      assert (Hwa : wf {| fm := fm g; ft := []; flo := flo g; fhi := fhi g |}) by (unfold wf; cbn; lia).
      pose proof (norm_fold_fm true (fm g) (ft g) _ false Hwg' Hwa ltac:(cbn; lia)) as [_ Hfo].
      destruct (fold_left (norm_step true (fm g)) (ft g) ({| fm := fm g; ft := []; flo := flo g; fhi := fhi g |}, false))
        as [out c'] eqn:Eo. cbn [fst fm] in *.
      unfold wf. rewrite !merge_shifts_fm. unfold fat. rewrite !mk_fm. lia. }
    destruct (norm_once F true st g) as [h ch]. cbn [fst] in Hwo. destruct Hwo as [Hw1 Hw2].
    destruct ch; [destruct (IH h Hw1) as [H1 H2]; split; [exact H1 | lia] | auto]. }
  apply H. exact Hwf.
Qed.

End Norm.
