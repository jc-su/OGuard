(*
  Soundness of refinement: when a comparison, a predicate, or a guard holds
  of the registers, refining a state that holds of them gives a state that
  still holds of them. In particular a refinement that finds no state (None)
  shows that the comparison, predicate, or guard does not hold.
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

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

(* ------------------------------------------------------------------ *)
(* Arithmetic.                                                         *)

Lemma terms_eqb_eq : forall a b, terms_eqb a b = true -> a = b.
Proof.
  induction a as [| [x c] a IH]; intros [| [y d] b] H; cbn [terms_eqb] in H; try discriminate; [reflexivity |].
  rewrite !andb_true_iff in H. destruct H as [[Hx Hc] Hr].
  apply atom_eqb_eq in Hx. apply Z.eqb_eq in Hc. subst. f_equal. exact (IH b Hr).
Qed.

Lemma some_inv : forall (A : Type) (a b : A), Some a = Some b -> b = a.
Proof. intros A a b H. injection H as ->. reflexivity. Qed.

(* Rounding a range inward to multiples of 2^k keeps its multiples of 2^k. *)
Lemma round_in : forall k a b v,
    aligned k v -> a <= v <= b -> fst (round k (a, b)) <= v <= snd (round k (a, b)).
Proof.
  intros k a b v [q Hq] Hv. unfold round. cbn [fst snd].
  assert (Hs : 0 < 2 ^ Z.max 0 k) by (apply pow_nonneg_pos; lia). subst v.
  split.
  - assert (H1 : - q <= (- a) / 2 ^ Z.max 0 k) by (apply Z.div_le_lower_bound; lia). nia.
  - assert (H1 : q <= b / 2 ^ Z.max 0 k) by (apply Z.div_le_lower_bound; lia). nia.
Qed.

(* Two values in one window of 2^w consecutive integers that agree modulo
   2^w are equal. *)
Lemma eqm_window : forall w a b lo,
    0 <= w -> lo <= a < lo + 2 ^ w -> lo <= b < lo + 2 ^ w -> eqm w a b -> a = b.
Proof.
  intros w a b lo Hw Ha Hb H. unfold eqm in H.
  assert (Hp : 0 < 2 ^ w) by (apply pow_nonneg_pos; lia).
  assert (Hd : (a - lo) mod 2 ^ w = (b - lo) mod 2 ^ w).
  { rewrite (Zminus_mod a), (Zminus_mod b), H. reflexivity. }
  rewrite !Z.mod_small in Hd by lia. lia.
Qed.

Lemma pow_half : forall w, 1 <= w -> 2 ^ w = 2 * 2 ^ (w - 1).
Proof. intros w Hw. rewrite <- Z.pow_succ_r by lia. f_equal. lia. Qed.

Lemma hi_lo_ok : forall w sg, 1 <= w -> hi_ok w sg = lo_ok w sg + 2 ^ w - 1.
Proof. intros w [|] Hw; unfold hi_ok, lo_ok; [rewrite (pow_half w Hw) |]; lia. Qed.

Lemma extend_range : forall n sg v,
    (1 <= n)%nat -> lo_ok (Z.of_nat n) sg <= extend n sg v <= hi_ok (Z.of_nat n) sg.
Proof.
  intros n [|] v Hn; unfold lo_ok, hi_ok, extend.
  - unfold sext. pose proof (trunc_range n v) as Ht. pose proof (pow_half (Z.of_nat n) ltac:(lia)) as H2.
    destruct (Z.ltb_spec (trunc n v) (2 ^ (Z.of_nat n - 1))); lia.
  - pose proof (trunc_range n v). lia.
Qed.

Lemma extend_eqm : forall n sg v, eqm (Z.of_nat n) (extend n sg v) v.
Proof. intros n [|] v; unfold extend; [apply sext_eqm | apply trunc_eqm]. Qed.

(* A value in [lo_ok, hi_ok] that agrees with a w-bit read modulo 2^w is
   that read. *)
Lemma eqm_extend_eq : forall n sg v V,
    (1 <= n)%nat -> lo_ok (Z.of_nat n) sg <= V <= hi_ok (Z.of_nat n) sg ->
    eqm (Z.of_nat n) V (extend n sg v) -> V = extend n sg v.
Proof.
  intros n sg v V Hn HV H. pose proof (extend_range n sg v Hn) as Hx.
  rewrite hi_lo_ok in HV, Hx by lia.
  apply (eqm_window (Z.of_nat n) V (extend n sg v) (lo_ok (Z.of_nat n) sg)); [lia | lia | lia | exact H].
Qed.

Lemma ceil_div_pos : forall a c x, 0 < c -> a <= c * x -> ceil_div a c <= x.
Proof.
  intros a c x Hc H. unfold ceil_div.
  assert (- x <= (- a) / c) by (apply Z.div_le_lower_bound; lia). lia.
Qed.

Lemma floor_div_pos : forall b c x, 0 < c -> c * x <= b -> x <= b / c.
Proof. intros b c x Hc H. apply Z.div_le_lower_bound; lia. Qed.

Lemma ceil_div_neg : forall b c x, c < 0 -> c * x <= b -> ceil_div b c <= x.
Proof.
  intros b c x Hc H. unfold ceil_div.
  rewrite <- (Z.div_opp_opp (- b) c) by lia. rewrite Z.opp_involutive.
  assert (- x <= b / (- c)) by (apply Z.div_le_lower_bound; lia). lia.
Qed.

Lemma floor_div_neg : forall a c x, c < 0 -> a <= c * x -> x <= a / c.
Proof.
  intros a c x Hc H. rewrite <- (Z.div_opp_opp a c) by lia.
  apply Z.div_le_lower_bound; lia.
Qed.

(* A sum split into one atom's terms and the rest. *)
Lemma tsum_coef : forall wdn rs ts x,
    tsum wdn rs ts =
    fold_right (fun u s => if atom_eqb (fst u) x then snd u + s else s) 0 ts * atom_val wdn rs x +
    tsum wdn rs (filter (fun u => negb (atom_eqb (fst u) x)) ts).
Proof.
  intros wdn rs ts x. induction ts as [| [y c] ts IH]; cbn [tsum fold_right filter fst snd]; [lia |].
  destruct (atom_eqb y x) eqn:E; cbn [negb tsum].
  - apply atom_eqb_eq in E. subst y. rewrite IH. ring.
  - rewrite IH. ring.
Qed.

(* ------------------------------------------------------------------ *)
(* Refinement.                                                         *)

Section Refine.

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
Notation sok := (fun s => state_ok F t s rs).

Let rsr := rs_range F E rs Hrs.
Let wdr := wd_range F E HE.

Lemma put_ok : forall st r x,
    state_ok F t st rs -> fact_ok F t rs r x -> state_ok F t (put st r x) rs.
Proof.
  intros st r x [Hf Hc] Hx. split; [| exact Hc].
  intros r' y Hy. unfold get, put in Hy. cbn [amap] in Hy.
  destruct (Pos.eq_dec r' r) as [-> | Hne].
  - rewrite PM.gss in Hy. apply some_inv in Hy. subst y. exact Hx.
  - rewrite PM.gso in Hy by exact Hne. exact (Hf r' y Hy).
Qed.

Lemma add_cons_ok : forall st k,
    state_ok F t st rs -> cons_ok F rs k -> state_ok F t (add_cons st k) rs.
Proof.
  intros st k [Hf Hc] Hk. unfold add_cons.
  destruct (existsb (constraint_eqb k) (acons st)); [split; assumption |].
  split; [exact Hf |]. cbn [acons]. constructor; assumption.
Qed.

Lemma sext_range : forall n v, (1 <= n)%nat -> - 2 ^ (Z.of_nat n - 1) <= sext n v <= 2 ^ (Z.of_nat n - 1) - 1.
Proof. intros n v Hn. exact (extend_range n true v Hn). Qed.

Lemma range_form_fact : forall r lo hi al v,
    lo <= v <= hi -> eqm (wd F r) (rs r) v -> aligned al (rs r) ->
    fact_ok F t rs r (FForm (range_form (wd F r) lo hi) al None).
Proof.
  intros r lo hi al v Hv He Ha. pose proof (wdr r) as Hw. cbn [fact_ok fact_okv].
  split; [unfold wf, range_form; cbn [fm]; unfold wd; lia |].
  split; [| split; [exact Ha | discriminate]].
  exists v. unfold range_form. cbn [fm ft flo fhi GateForm.tsum]. split; [exact Hv | rewrite Z.add_0_l; exact He].
Qed.

(* The general case: a register without a relational fact, narrowed to a
   range of its own. *)
Lemma tighten_general : forall st r e lo hi,
    state_ok F t st rs -> (e = EX \/ e = EZ \/ e = ES) -> lo <= aval (r, e) <= hi ->
    exists s,
      (let al := reg_align F st r in
       let w := wd F r in
       let cur := match (match e with ES => sreg_range F st r | _ => reg_range F st r end) with
                  | Some g => g
                  | None => match e with ES => (- 2 ^ (w - 1), 2 ^ (w - 1) - 1) | _ => (0, 2 ^ w - 1) end
                  end in
       let g := round al (Z.max (fst cur) lo, Z.min (snd cur) hi) in
       if snd g <? fst g then None else Some (put st r (FForm (range_form w (fst g) (snd g)) al None)))
      = Some s /\ state_ok F t s rs.
Proof.
  intros st r e lo hi Hst He Hv. cbv zeta.
  pose proof (rsr r) as Hr. pose proof (wdr r) as Hw.
  assert (Hwd : wd F r = Z.of_nat (wdn r)) by reflexivity.
  pose proof (reg_align_sound F E HE t st rs Hst Hrs r) as Hal.
  set (al := reg_align F st r) in *.
  (* v is the value e reads; cur bounds it *)
  assert (Hc : exists c1 c2 v, aval (r, e) = v /\ c1 <= v <= c2 /\ aligned al v /\ eqm (wd F r) (rs r) v /\
               match (match e with ES => sreg_range F st r | _ => reg_range F st r end) with
               | Some g => g
               | None => match e with ES => (- 2 ^ (wd F r - 1), 2 ^ (wd F r - 1) - 1) | _ => (0, 2 ^ wd F r - 1) end
               end = (c1, c2)).
  { destruct He as [-> | [-> | ->]]; cbn [atom_val].
    1, 2: destruct (reg_range F st r) as [[c1 c2] |] eqn:Ec;
          [exists c1, c2, (rs r); pose proof (reg_range_sound F E HE t st rs Hst Hrs r c1 c2 Ec)
          | exists 0, (2 ^ wd F r - 1), (rs r)];
          refine (conj eq_refl (conj _ (conj Hal (conj (eqm_refl _ _) eq_refl)))); try rewrite Hwd; lia.
    all: pose proof (sext_aligned F E HE rs Hrs r al Hal) as Hsa.
    all: pose proof (sext_eqm (wdn r) (rs r)) as Hse. rewrite <- Hwd in Hse.
    destruct (sreg_range F st r) as [[c1 c2] |] eqn:Ec.
    - exists c1, c2, (sext (wdn r) (rs r)). pose proof (sreg_range_sound F E HE t st rs Hst Hrs r c1 c2 Ec).
      refine (conj eq_refl (conj _ (conj Hsa (conj (eqm_sym _ _ _ Hse) eq_refl)))); lia.
    - exists (- 2 ^ (wd F r - 1)), (2 ^ (wd F r - 1) - 1), (sext (wdn r) (rs r)).
      pose proof (sext_range (wdn r) (rs r) ltac:(lia)) as Hsr. rewrite <- Hwd in Hsr.
      refine (conj eq_refl (conj _ (conj Hsa (conj (eqm_sym _ _ _ Hse) eq_refl)))); lia. }
  destruct Hc as [c1 [c2 [v [Hav [Hcv [Halv [Heq Hcur]]]]]]]. rewrite Hcur. cbn [fst snd].
  rewrite Hav in Hv.
  pose proof (round_in al (Z.max c1 lo) (Z.min c2 hi) v Halv ltac:(lia)) as Hg.
  destruct (round al (Z.max c1 lo, Z.min c2 hi)) as [g1 g2]. cbn [fst snd] in *.
  destruct (Z.ltb_spec g2 g1); [lia |].
  eexists; split; [reflexivity |]. apply put_ok; [exact Hst |].
  exact (range_form_fact r g1 g2 al v Hg Heq Hal).
Qed.

(* A metadata fact, narrowed. *)
Lemma add_align_ok : forall st r n,
    state_ok F t st rs -> aligned n (rs r) -> state_ok F t (add_align st r n) rs.
Proof.
  intros st r n Hst Ha. unfold add_align.
  destruct (get st r) as [x |] eqn:Eg.
  - pose proof (proj1 Hst r x Eg) as Hx. destruct x; try exact Hst.
    + apply put_ok; [exact Hst |]. cbn [fact_ok fact_okv] in *. destruct Hx as [H1 [H2 [H3 H4]]].
      split; [exact H1 | split; [exact H2 | split; [apply aligned_max; assumption | exact H4]]].
    + apply put_ok; [exact Hst |]. cbn [fact_ok fact_okv] in *. apply aligned_max; assumption.
  - apply put_ok; [exact Hst |]. cbn [fact_ok fact_okv]. exact Ha.
Qed.

Lemma tighten_reg_sound : forall st r e lo hi,
    state_ok F t st rs -> lo <= aval (r, e) <= hi ->
    exists s, tighten_reg F st r e lo hi = Some s /\ state_ok F t s rs.
Proof.
  intros st r e lo hi Hst Hv.
  assert (Hdone : exists s, Some st = Some s /\ state_ok F t s rs) by (exists st; auto).
  assert (He : e = EX \/ e = EZ \/ e = ES \/
               (exists n, e = EBf 0 n /\ lo = 0 /\ hi = 0 /\ 0 <= n /\
                          tighten_reg F st r e lo hi = Some (add_align st r n)) \/
               tighten_reg F st r e lo hi = Some st).
  { destruct e as [| | | k | p n]; auto. unfold tighten_reg.
    destruct (Z.eqb p 0 && Z.eqb lo 0 && Z.eqb hi 0 && (0 <=? n)) eqn:Ec; [| auto].
    rewrite !andb_true_iff in Ec. destruct Ec as [[[Ep Elo] Ehi] En].
    apply Z.eqb_eq in Ep, Elo, Ehi. apply Z.leb_le in En. subst p.
    right; right; right; left. exists n. auto. }
  destruct He as [He | [He | [He | [[n [-> [-> [-> [Hn He]]]]] | He]]]];
    [| | | rewrite He | rewrite He; exact Hdone].
  4: { (* r's low n bits are zero: r is a multiple of 2^n *)
       exists (add_align st r n). split; [reflexivity |]. apply add_align_ok; [exact Hst |].
       cbn [atom_val] in Hv. rewrite Z.pow_0_r, Z.div_1_r in Hv.
       unfold aligned. rewrite Z.max_r by lia. apply Z.mod_divide; [apply Z.pow_nonzero; lia | lia]. }
  all: assert (He' : e = EX \/ e = EZ \/ e = ES) by auto.
  all: subst e; unfold tighten_reg; cbv iota; destruct (get st r) as [x |] eqn:Eg.
  all: try (exact (tighten_general st r _ lo hi Hst He' Hv)).
  all: destruct x as [f al vr | k | cs ex | cs ex].
  all: try (exact (tighten_general st r _ lo hi Hst He' Hv)).
  all: try exact Hdone.
  all: destruct (interval_only (Some (FForm f al vr)) || null (ft f)); cbn [negb];
         [exact (tighten_general st r _ lo hi Hst He' Hv) | exact Hdone].
Qed.

(* ------------------------------------------------------------------ *)
(* Compared values.                                                    *)

(* The value comparison operand o reads at width w. *)
Definition cval (o : opnd) (w : Z) (sg : bool) : Z := extend (Z.to_nat w) sg (opv rs o).

Definition owidth (o : opnd) (w : Z) : Prop := forall r, o = OReg r -> wd F r = w.

(* A register's form, read at the comparison's width and extension, keeps
   the value modulo 2^w. *)
Lemma reading_sound : forall st f w sg v X,
    state_ok F t st rs -> 1 <= w <= 64 -> wf f -> w <= fm f -> sat f v -> eqm w v X ->
    wf (reading F f w sg) /\ w <= fm (reading F f w sg) /\
    exists v', sat (reading F f w sg) v' /\ eqm w v' X.
Proof.
  intros st f w sg v X Hst Hw Hwf Hm Hs Hv.
  assert (Hdone : wf f /\ w <= fm f /\ exists v', sat f v' /\ eqm w v' X)
    by (split; [exact Hwf |]; split; [exact Hm | exists v; split; assumption]).
  (* x read with either extension agrees with x modulo 2^w *)
  assert (Hx : forall x e1, (e1 = EZ \/ e1 = ES) -> wd F x = w -> eqm w (aval (x, e1)) (rs x)).
  { intros x e1 He1 Hwx. destruct He1 as [-> | ->]; cbn [atom_val]; [apply eqm_refl |].
    unfold wd in Hwx. rewrite <- Hwx. apply sext_eqm. }
  assert (Hatom : forall x e, (e = EZ \/ e = ES) -> ft f = [((x, e), 1)] -> flo f = 0 -> fhi f = 0 -> wd F x = w ->
            wf (atom64 x (if sg then ES else EZ)) /\ w <= fm (atom64 x (if sg then ES else EZ)) /\
            exists v', sat (atom64 x (if sg then ES else EZ)) v' /\ eqm w v' X).
  { intros x e He Ef Hlo Hhi Hwx.
    destruct (atom64_sound F E HE t st rs Hst Hrs x (if sg then ES else EZ)) as [Has Ham].
    split; [unfold wf; lia |]. split; [lia |]. exists (aval (x, if sg then ES else EZ)). split; [exact Has |].
    destruct Hs as [e0 [He0 Hv0]]. rewrite Ef in Hv0. cbn [GateForm.tsum] in Hv0.
    assert (e0 = 0) by lia. subst e0. rewrite Z.mul_1_l, !Z.add_0_r in Hv0.
    apply (eqm_weaken (fm f) w) in Hv0; [| lia].
    eapply eqm_trans; [apply (Hx x); [destruct sg; auto | exact Hwx] |].
    eapply eqm_trans; [apply eqm_sym, (Hx x e He Hwx) |].
    eapply eqm_trans; [apply eqm_sym, Hv0 | exact Hv]. }
  unfold reading. destruct (ft f) as [| [[x e] c] [| u L]] eqn:Ef; try exact Hdone.
  match goal with |- context [if ?b then _ else _] => destruct b eqn:Ec end; [| exact Hdone].
  rewrite !andb_true_iff in Ec. destruct Ec as [[[[Hc Hlo] Hhi] Hwx] He].
  apply Z.eqb_eq in Hc, Hlo, Hhi, Hwx. subst c.
  apply (Hatom x e); [destruct e; try discriminate; auto | reflexivity | exact Hlo | exact Hhi | exact Hwx].
Qed.

Lemma cmp_form_sound : forall st o w sg,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth o w ->
    wf (cmp_form F st o w sg) /\ w <= fm (cmp_form F st o w sg) /\
    exists v, sat (cmp_form F st o w sg) v /\ eqm w v (cval o w sg).
Proof.
  intros st o w sg Hst Hw Ho. unfold cval.
  destruct o as [r | z]; unfold cmp_form; cbn [opv].
  - pose proof (Ho r eq_refl) as Hwr. pose proof (rsr r) as Hr.
    assert (Hn : Z.to_nat w = wdn r) by (unfold wd in Hwr; lia).
    assert (Hcv : eqm w (rs r) (extend (Z.to_nat w) sg (rs r))).
    { apply eqm_sym. pose proof (extend_eqm (Z.to_nat w) sg (rs r)) as H. rewrite Z2Nat.id in H by lia. exact H. }
    assert (Hat : wf (atom64 r (if sg then ES else EZ)) /\ w <= fm (atom64 r (if sg then ES else EZ)) /\
                  exists v, sat (atom64 r (if sg then ES else EZ)) v /\ eqm w v (extend (Z.to_nat w) sg (rs r))).
    { destruct (atom64_sound F E HE t st rs Hst Hrs r (if sg then ES else EZ)) as [Hs Hm].
      split; [unfold wf; lia |]. split; [lia |]. exists (aval (r, if sg then ES else EZ)). split; [exact Hs |].
      destruct sg; cbn [atom_val]; [rewrite Hn; apply eqm_refl | exact Hcv]. }
    destruct (Z.eqb_spec (wd F r) 64) as [H64 | H64].
    + destruct (sg && (match get st r with None | Some (FAlign _) => true | x => interval_only x end) &&
          (match fact_range F r (get st r) with
           | Some (lo, hi) => negb ((0 <=? lo) && (hi <? 2 ^ 63))
           | None => true
           end)) eqn:Esg.
      { rewrite !andb_true_iff in Esg. destruct Esg as [[-> _] _]. exact Hat. }
      destruct (as_form_sound F E HE t st rs Hst Hrs r EX) as [Hs Hm].
      pose proof (as_form_wf F E HE t st rs Hst Hrs r EX) as Hwf.
      destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hwf Hs) as [Hw' [Hm' Hs']].
      split; [exact Hw' |]. split; [lia |]. exists (rs r). split; [exact Hs' | exact Hcv].
    + destruct (get st r) as [x |] eqn:Eg; [| exact Hat].
      destruct x as [f al vr | | | ]; try exact Hat.
      destruct ((wd F r <=? fm f) && negb (interval_only (Some (FForm f al vr)))) eqn:Ec; [| exact Hat].
      apply andb_true_iff in Ec. destruct Ec as [Ec _]. apply Z.leb_le in Ec.
      pose proof (proj1 Hst r _ Eg) as Hf. cbn [fact_ok fact_okv] in Hf. destruct Hf as [Hwf [Hs _]].
      apply (reading_sound st f w sg (rs r)); [exact Hst | exact Hw | exact Hwf | lia | exact Hs | exact Hcv].
  - split; [unfold wf, fconst; cbn; lia |]. split; [cbn; lia |].
    exists (extend (Z.to_nat w) sg z). split; [apply sat_fconst; apply eqm_refl | apply eqm_refl].
Qed.

(* The integer value of the comparison form agrees with the compared value
   modulo 2^w. *)
Lemma cmp_value : forall st o w sg,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth o w ->
    exists e, flo (cmp_form F st o w sg) <= e <= fhi (cmp_form F st o w sg) /\
              eqm w (tsum (ft (cmp_form F st o w sg)) + e) (cval o w sg).
Proof.
  intros st o w sg Hst Hw Ho. destruct (cmp_form_sound st o w sg Hst Hw Ho) as [Hwf [Hm [v [[e [He Hv]] Hc]]]].
  exists e. split; [exact He |]. eapply eqm_trans; [| exact Hc].
  apply eqm_sym. apply (eqm_weaken (fm (cmp_form F st o w sg))); [lia | exact Hv].
Qed.

Lemma cv_aligned : forall st r w sg,
    state_ok F t st rs -> wd F r = w -> 1 <= w <= 64 ->
    aligned (Z.min (reg_align F st r) (w - 1)) (cval (OReg r) w sg).
Proof.
  intros st r w sg Hst Hwr Hw. unfold cval. cbn [opv].
  pose proof (reg_align_sound F E HE t st rs Hst Hrs r) as Ha.
  apply (aligned_mono _ (Z.min (reg_align F st r) (w - 1))) in Ha; [| lia].
  apply (aligned_eqm w _ _ (rs r)); [lia | lia | | exact Ha].
  pose proof (extend_eqm (Z.to_nat w) sg (rs r)) as H. rewrite Z2Nat.id in H by lia. exact H.
Qed.

Lemma aligned_range_sound : forall st o w sg f x y e,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth o w ->
    aligned_range F st o f w = Some (x, y) -> flo f <= e <= fhi f ->
    eqm w (tsum (ft f) + e) (cval o w sg) -> x <= tsum (ft f) + e <= y.
Proof.
  intros st o w sg f x y e Hst Hw Ho H He Hq. unfold aligned_range in H.
  destruct (form_range F st f) as [[g1 g2] |] eqn:Ef; [| discriminate].
  pose proof (form_range_sound F E HE t st rs Hst Hrs f g1 g2 e Ef He) as Hg.
  destruct o as [r | z]; apply some_inv in H; [| injection H as <- <-; exact Hg].
  pose proof (cv_aligned st r w sg Hst (Ho r eq_refl) Hw) as Ha.
  assert (Hv : aligned (Z.min (reg_align F st r) (w - 1)) (tsum (ft f) + e))
    by (apply (aligned_eqm w _ _ (cval (OReg r) w sg)); [lia | lia | exact Hq | exact Ha]).
  pose proof (round_in (Z.min (reg_align F st r) (w - 1)) g1 g2 _ Hv Hg) as Hr.
  rewrite <- H in Hr. exact Hr.
Qed.

(* A value in the comparison's bounds that agrees with the compared value
   modulo 2^w is the compared value. *)
Lemma cval_exact : forall o w sg V,
    1 <= w <= 64 -> lo_ok w sg <= V <= hi_ok w sg -> eqm w V (cval o w sg) -> V = cval o w sg.
Proof.
  intros o w sg V Hw HV Hq. unfold cval in *.
  apply eqm_extend_eq; [lia | rewrite Z2Nat.id by lia; exact HV | rewrite Z2Nat.id by lia; exact Hq].
Qed.

Lemma cval_range : forall o w sg, 1 <= w <= 64 -> lo_ok w sg <= cval o w sg <= hi_ok w sg.
Proof.
  intros o w sg Hw. unfold cval. pose proof (extend_range (Z.to_nat w) sg (opv rs o) ltac:(lia)) as H.
  rewrite Z2Nat.id in H by lia. exact H.
Qed.

(* The sums of the term lists that relate builds are exact. *)
Lemma tsum_sorted_terms : forall ts, tsum (sorted_terms ts) = tsum ts.
Proof. intros ts. unfold sorted_terms. rewrite tsum_nonzero. apply tsum_tsort. Qed.

Lemma tsum_coef_split : forall ts x, tsum ts = coef ts x * aval x + tsum (without ts x).
Proof.
  intros ts x. induction ts as [| [y c] ts IH]; [cbn; ring |].
  unfold coef, without in *. cbn [fold_right filter fst snd tsum].
  destruct (atom_eqb y x) eqn:Ey; cbn [negb tsum].
  - apply atom_eqb_eq in Ey. subst y. rewrite IH. ring.
  - rewrite IH. ring.
Qed.

Lemma relate_sound : forall st fa fb dlo dhi va vb,
    state_ok F t st rs ->
    (exists ea, flo fa <= ea <= fhi fa /\ tsum (ft fa) + ea = va) ->
    (exists eb, flo fb <= eb <= fhi fb /\ tsum (ft fb) + eb = vb) ->
    dlo <= va - vb <= dhi ->
    state_ok F t (relate st fa fb dlo dhi) rs.
Proof.
  intros st fa fb dlo dhi va vb [Hf Hc] [ea [Hea Hva]] [eb [Heb Hvb]] Hd. unfold relate.
  split; [exact Hf |]. cbn [acons]. constructor.
  - cbn [cons_ok]. rewrite tsum_sorted_terms, tsum_app, tsum_tscale. lia.
  - apply Forall_app. split; [| exact Hc].
    destruct (ft fb) as [| [xb cb] [| ? ?]] eqn:Efb;
      [apply Forall_nil | | destruct cb as [| [q | q |] | q]; apply Forall_nil].
    destruct cb as [| [q | q |] | q]; try (apply Forall_nil; fail).
    destruct (Z.eqb (flo fb) 0 && Z.eqb (fhi fb) 0) eqn:E0; [| apply Forall_nil].
    apply andb_true_iff in E0. destruct E0 as [Hlo Hhi]. apply Z.eqb_eq in Hlo, Hhi.
    assert (Hxb : aval xb = vb) by (try rewrite Efb in Hvb; cbn [tsum] in Hvb; lia).
    apply Forall_forall. intros k Hk. apply in_flat_map in Hk. destruct Hk as [[[ts lo] hi] [Hin Hk]].
    cbv beta iota in Hk.
    pose proof (proj1 (Forall_forall _ _) Hc _ Hin) as Hts. cbn [cons_ok] in Hts.
    rewrite (tsum_coef_split ts xb), Hxb in Hts.
    destruct (Z.eqb_spec (coef ts xb) 1) as [Hco | Hco]; [| destruct Hk].
    destruct Hk as [<- | []]. cbn [cons_ok]. rewrite Hco in Hts. rewrite tsum_sorted_terms, tsum_app. lia.
Qed.

(* A comparison form whose range lies in the comparison's bounds has the
   compared value as its integer value. *)
Lemma cmp_exact : forall st o w sg x y,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth o w -> cmp_range F st o w sg = Some (x, y) ->
    exists e, flo (cmp_form F st o w sg) <= e <= fhi (cmp_form F st o w sg) /\
              tsum (ft (cmp_form F st o w sg)) + e = cval o w sg.
Proof.
  intros st o w sg x y Hst Hw Ho H. unfold cmp_range in H.
  destruct (aligned_range F st o (cmp_form F st o w sg) w) as [[x' y'] |] eqn:Ea; [| discriminate].
  destruct ((lo_ok w sg <=? x') && (y' <=? hi_ok w sg)) eqn:Eb; [| discriminate].
  apply andb_true_iff in Eb. destruct Eb as [E1 E2]. apply Z.leb_le in E1, E2.
  destruct (cmp_value st o w sg Hst Hw Ho) as [e [He Hq]].
  pose proof (aligned_range_sound st o w sg _ _ _ e Hst Hw Ho Ea He Hq) as Hxy.
  exists e. split; [exact He |]. apply cval_exact; [exact Hw | lia | exact Hq].
Qed.

(* The atom of the compared register has the compared value as its integer
   value when its range lies in the comparison's bounds. *)
Lemma cmp_atom_sound : forall st a w sg ga,
    state_ok F t st rs -> 1 <= w <= 64 -> wd F a = w -> cmp_atom F st a w sg = Some ga ->
    exists e, flo ga <= e <= fhi ga /\ tsum (ft ga) + e = cval (OReg a) w sg.
Proof.
  intros st a w sg ga Hst Hw Ha H. unfold cmp_atom in H.
  assert (Hn : Z.to_nat w = wdn a) by (unfold wd in Ha; lia).
  assert (Hcv : eqm w (rs a) (extend (Z.to_nat w) sg (rs a))).
  { apply eqm_sym. pose proof (extend_eqm (Z.to_nat w) sg (rs a)) as Hx. rewrite Z2Nat.id in Hx by lia. exact Hx. }
  assert (Hg : forall x, eqm w (aval (a, x)) (cval (OReg a) w sg) ->
            (match form_range F st (atom64 a x) with
             | Some (lo, hi) => if (lo_ok w sg <=? lo) && (hi <=? hi_ok w sg) then Some (atom64 a x) else None
             | None => None end) = Some ga ->
            exists e, flo ga <= e <= fhi ga /\ tsum (ft ga) + e = cval (OReg a) w sg).
  { intros x Hx Hm. destruct (form_range F st (atom64 a x)) as [[lo hi] |] eqn:Er; [| discriminate].
    destruct ((lo_ok w sg <=? lo) && (hi <=? hi_ok w sg)) eqn:Eb; [| discriminate].
    apply andb_true_iff in Eb. destruct Eb as [E1 E2]. apply Z.leb_le in E1, E2.
    apply some_inv in Hm. subst ga.
    destruct (atom64_sound F E HE t st rs Hst Hrs a x) as [[e [He Hq]] Hfm].
    pose proof (form_range_sound F E HE t st rs Hst Hrs _ _ _ e Er He) as Hr.
    exists e. split; [exact He |].
    apply cval_exact; [exact Hw | lia |].
    eapply eqm_trans; [| exact Hx]. apply eqm_sym. apply (eqm_weaken (fm (atom64 a x))); [lia | exact Hq]. }
  apply (Hg (if Z.eqb (wd F a) 64 then EX else if sg then ES else EZ)); [| exact H].
  unfold cval; cbn [opv].
  destruct (Z.eqb (wd F a) 64); [exact Hcv |].
  destruct sg; cbn [atom_val]; [rewrite Hn; apply eqm_refl | exact Hcv].
Qed.

Lemma relate_cmp_sound : forall st a fa fb w sg dlo dhi vb,
    state_ok F t st rs -> 1 <= w <= 64 -> wd F a = w ->
    (exists ea, flo fa <= ea <= fhi fa /\ tsum (ft fa) + ea = cval (OReg a) w sg) ->
    (exists eb, flo fb <= eb <= fhi fb /\ tsum (ft fb) + eb = vb) ->
    dlo <= cval (OReg a) w sg - vb <= dhi ->
    state_ok F t (relate_cmp F st a fa fb w sg dlo dhi) rs.
Proof.
  intros st a fa fb w sg dlo dhi vb Hst Hw Ha Hfa Hfb Hd. unfold relate_cmp.
  pose proof (relate_sound st fa fb dlo dhi _ vb Hst Hfa Hfb Hd) as H1.
  destruct (cmp_atom F st a w sg) as [ga |] eqn:Eg; [| exact H1].
  destruct (terms_eqb (ft fa) (ft ga)); [exact H1 |].
  apply (relate_sound _ ga fb dlo dhi (cval (OReg a) w sg) vb H1); [| exact Hfb | exact Hd].
  exact (cmp_atom_sound st a w sg ga Hst Hw Ha Eg).
Qed.

(* A refinement that the registers satisfy is not refused: its last
   constraint holds of them, and so lies within its terms' range. *)
Lemma settle_sound : forall st s,
    state_ok F t st rs -> state_ok F t s rs -> settle F st s = Some s.
Proof.
  intros st s Hst Hs. unfold settle.
  destruct (acons s) as [| [[ts lo] hi] more] eqn:Ec; [reflexivity |].
  destruct (form_range_near F st (form_of_terms ts)) as [[x y] |] eqn:Er; [| reflexivity].
  pose proof (form_range_near_sound F E HE t st rs Hst Hrs (form_of_terms ts) x y 0 Er
                ltac:(unfold form_of_terms; cbn [flo fhi]; lia)) as Hr.
  unfold form_of_terms in Hr. cbn [ft] in Hr.
  pose proof (proj2 Hs) as Hk. rewrite Ec in Hk. inversion Hk as [| k l Hok _]; subst. cbn [cons_ok] in Hok.
  destruct ((y <? lo) || (hi <? x)) eqn:Eb; [| reflexivity].
  apply orb_true_iff in Eb. rewrite !Z.ltb_lt in Eb. lia.
Qed.

Lemma cmp_range_sound : forall st o w sg x y,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth o w ->
    cmp_range F st o w sg = Some (x, y) -> x <= cval o w sg <= y.
Proof.
  intros st o w sg x y Hst Hw Ho H. unfold cmp_range in H.
  destruct (aligned_range F st o (cmp_form F st o w sg) w) as [[x' y'] |] eqn:Ea; [| discriminate].
  destruct ((lo_ok w sg <=? x') && (y' <=? hi_ok w sg)) eqn:Eb; [| discriminate].
  apply some_inv in H. injection H as <- <-. apply andb_true_iff in Eb. destruct Eb as [E1 E2].
  apply Z.leb_le in E1, E2.
  destruct (cmp_value st o w sg Hst Hw Ho) as [e [He Hq]].
  pose proof (aligned_range_sound st o w sg _ _ _ e Hst Hw Ho Ea He Hq) as Hxy.
  rewrite <- (cval_exact o w sg (tsum (ft (cmp_form F st o w sg)) + e) Hw ltac:(lia) Hq). exact Hxy.
Qed.

(* ------------------------------------------------------------------ *)
(* Constraining a form and an operand.                                 *)

Lemma constrain_form_none : forall out st f lo' hi' e,
    state_ok F t out rs -> state_ok F t st rs -> flo f <= e <= fhi f ->
    lo' <= tsum (ft f) + e <= hi' ->
    exists s, constrain_form F out st f lo' hi' None = Some s /\ state_ok F t s rs.
Proof.
  intros out st f lo' hi' e Hout Hst He Hv'. unfold constrain_form. cbv beta iota zeta.
  destruct (Z.ltb_spec hi' lo'); [lia |].
  match goal with
  | |- context [fold_left ?g (ft f) (Some out)] =>
      assert (Hstep : forall o u, state_ok F t o rs -> exists s, g (Some o) u = Some s /\ state_ok F t s rs);
      [| assert (Hfold : forall L o, state_ok F t o rs -> exists s, fold_left g L (Some o) = Some s /\ state_ok F t s rs);
         [| destruct (Hfold (ft f) out Hout) as [s [Es Hs]]; rewrite Es]]
  end.
  - intros o [x c0] Ho. cbv beta iota zeta. cbn [fst].
    set (C := fold_right (fun u s => if atom_eqb (fst u) x then snd u + s else s) 0 (ft f)).
    destruct (Z.eqb_spec C 0) as [HC | HC]; [exists o; split; [reflexivity | exact Ho] |].
    match goal with |- context [form_range F st ?oth] => destruct (form_range F st oth) as [[p q] |] eqn:Er end;
      [| exists o; split; [reflexivity | exact Ho]].
    pose proof (form_range_sound F E HE t st rs Hst Hrs _ p q e Er ltac:(cbn [flo fhi]; exact He)) as Hpq.
    cbn [ft] in Hpq. pose proof (tsum_coef wdn rs (ft f) x) as Hsplit. fold C in Hsplit.
    assert (Hcx : lo' - q <= C * aval x <= hi' - p) by lia.
    destruct x as [r0 e1]. cbn [fst snd].
    destruct (Z.ltb_spec 0 C) as [Hpos | Hneg].
    + apply tighten_reg_sound; [exact Ho |].
      split; [apply ceil_div_pos | apply floor_div_pos]; lia.
    + apply tighten_reg_sound; [exact Ho |].
      split; [apply ceil_div_neg | apply floor_div_neg]; lia.
  - intros L. induction L as [| u L IH]; intros o Ho; cbn [fold_left]; [exists o; split; [reflexivity | exact Ho] |].
    destruct (Hstep o u Ho) as [s [Es Hs]]. rewrite Es. exact (IH s Hs).
  - (* the bound holds of f's normal form when the two cannot differ by a
       multiple of 2^64 *)
    assert (HN : forall o1, state_ok F t o1 rs -> forall p q,
                 form_range_nat F o1 (normalize F o1 f) = Some (p, q) ->
                 Z.eqb (fm f) 64 = true -> q - lo' < 2 ^ 64 -> hi' - p < 2 ^ 64 ->
                 cons_ok F rs (ft (normalize F o1 f), lo' - fhi (normalize F o1 f), hi' - flo (normalize F o1 f))).
    { intros o1 Ho1 p q Er Hm Hq Hp. apply Z.eqb_eq in Hm.
      assert (Hwf : wf f) by (unfold wf; lia).
      assert (Hsf : sat f (tsum (ft f) + e)) by (exists e; split; [exact He | apply eqm_refl]).
      destruct (normalize_sound F E HE t o1 rs Ho1 Hrs f _ Hwf Hsf) as [_ [HmN [eN [HeN HvN]]]].
      pose proof (form_range_nat_sound F E HE t o1 rs Ho1 Hrs _ p q eN Er HeN) as HpN.
      rewrite HmN, Hm in HvN.
      assert (Heq : tsum (ft (normalize F o1 f)) + eN = tsum (ft f) + e)
        by (apply (eqm_close 64); [lia | apply eqm_sym; exact HvN | lia]).
      cbn [cons_ok]. lia. }
    assert (Hc : cons_ok F rs (ft f, lo' - fhi f, hi' - flo f)) by (cbn [cons_ok]; lia).
    revert Hc. destruct (ft f) as [| [[r0 e1] c1] [| u' L]]; intros Hc;
      [exists s; auto | cbv beta iota | eexists; split; [reflexivity | apply add_cons_ok; assumption]].
    destruct (match e1 with EBf _ _ | EShr _ => true | _ => false end);
      [eexists; split; [reflexivity | apply add_cons_ok; assumption] |].
    destruct (get s r0) as [[g0 al0 vr0 | | | ] |]; try (exists s; split; [reflexivity | exact Hs]).
    cbv beta iota. destruct (negb (interval_only (Some (FForm g0 al0 vr0))) && negb (null (ft g0)));
      [| exists s; split; [reflexivity | exact Hs]].
    match goal with |- context [form_range_nat F ?o (normalize F ?o f)] => set (o1 := o) end.
    assert (Ho1 : state_ok F t o1 rs) by (apply add_cons_ok; [exact Hs | exact Hc]).
    destruct (form_range_nat F o1 (normalize F o1 f)) as [[p q] |] eqn:Er;
      [| exists o1; split; [reflexivity | exact Ho1]].
    match goal with |- context [if ?c then _ else _] => destruct c eqn:Ec end;
      [| exists o1; split; [reflexivity | exact Ho1]].
    rewrite !andb_true_iff in Ec. destruct Ec as [[[Hm _] Hq] Hp]. apply Z.ltb_lt in Hq, Hp.
    eexists; split; [reflexivity |]. apply add_cons_ok; [exact Ho1 | exact (HN o1 Ho1 p q Er Hm Hq Hp)].
Qed.

Lemma constrain_form_sound : forall out st f lo hi rg e,
    state_ok F t out rs -> state_ok F t st rs -> flo f <= e <= fhi f ->
    lo <= tsum (ft f) + e <= hi -> (forall x y, rg = Some (x, y) -> x <= tsum (ft f) + e <= y) ->
    exists s, constrain_form F out st f lo hi rg = Some s /\ state_ok F t s rs.
Proof.
  intros out st f lo hi rg e Hout Hst He Hv Hrg. destruct rg as [[x y] |].
  - change (constrain_form F out st f lo hi (Some (x, y))) with (constrain_form F out st f (Z.max lo x) (Z.min hi y) None).
    specialize (Hrg x y eq_refl). apply (constrain_form_none out st f _ _ e Hout Hst He). lia.
  - exact (constrain_form_none out st f lo hi e Hout Hst He Hv).
Qed.

Lemma single_value : forall f r w sg,
    (match ft f with
     | [((r', e), c)] =>
         Z.eqb c 1 && Z.eqb (flo f) 0 && Z.eqb (fhi f) 0 && Pos.eqb r' r && Z.eqb (wd F r) w &&
         match e with EX | EZ => negb sg | ES => sg | _ => false end
     | _ => false
     end) = true ->
    flo f <= 0 <= fhi f /\ tsum (ft f) + 0 = cval (OReg r) w sg.
Proof.
  intros f r w sg H. unfold cval. cbn [opv].
  destruct (ft f) as [| [[r' e1] c] [| u L]]; try discriminate.
  rewrite !andb_true_iff in H. destruct H as [[[[[Hc Hlo] Hhi] Hr] Hw] Hs].
  apply Z.eqb_eq in Hc, Hlo, Hhi, Hw. apply Pos.eqb_eq in Hr. subst c r'.
  pose proof (rsr r) as Hrr. pose proof (wdr r) as Hwr.
  assert (Hn : Z.to_nat w = wdn r) by (unfold wd in Hw; lia).
  split; [lia |]. cbn [GateForm.tsum]. rewrite Hn.
  destruct e1, sg; try discriminate; cbn [atom_val extend]; try (rewrite trunc_small by lia); lia.
Qed.

Lemma constrain_sound : forall st o lo hi w sg,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth o w ->
    (forall l, lo = Some l -> l <= cval o w sg) -> (forall h, hi = Some h -> cval o w sg <= h) ->
    exists s, constrain F st o lo hi w sg = Some s /\ state_ok F t s rs.
Proof.
  intros st o lo hi w sg Hst Hw Ho Hlo Hhi.
  destruct o as [r | z]; [| exists st; split; [reflexivity | exact Hst]].
  pose proof (Ho r eq_refl) as Hwr. pose proof (rsr r) as Hrr. pose proof (wdr r) as Hwdr.
  assert (Hn : Z.to_nat w = wdn r) by (unfold wd in Hwr; lia).
  unfold constrain. cbv zeta.
  set (f := cmp_form F st (OReg r) w sg).
  set (rg := aligned_range F st (OReg r) f w).
  set (cv := cval (OReg r) w sg).
  destruct (cmp_value st (OReg r) w sg Hst Hw Ho) as [e0 [He0 Hq0]]. fold f cv in He0, Hq0.
  (* a known range of the form holds its value *)
  assert (Hrg : forall V, eqm w V cv -> forall e, flo f <= e <= fhi f -> V = tsum (ft f) + e ->
                forall x y, rg = Some (x, y) -> x <= V <= y).
  { intros V HV e He -> x y Er. exact (aligned_range_sound st (OReg r) w sg f x y e Hst Hw Ho Er He HV). }
  (* the form's range, aligned or with atoms in their natural ranges *)
  assert (Hnr : forall x y, match rg with Some g => Some g | None => form_range_nat F st f end = Some (x, y) ->
                x <= tsum (ft f) + e0 <= y).
  { intros x y Er. destruct rg as [[x' y'] |] eqn:Erg; cbv beta iota in Er.
    - injection Er as <- <-. exact (aligned_range_sound st (OReg r) w sg f x' y' e0 Hst Hw Ho Erg He0 Hq0).
    - exact (form_range_nat_sound F E HE t st rs Hst Hrs f x y e0 Er He0). }
  match goal with |- context [if negb ?A && negb ?B && negb ?C then Some st else _] =>
    remember A as exact eqn:Eexact; remember B as single eqn:Esingle; remember C as wrap eqn:Ewrap end.
  destruct (negb exact && negb single && negb wrap) eqn:Enb; [exists st; split; [reflexivity | exact Hst] |].
  (* the form's integer value is the compared value *)
  assert (Hval : exists e, flo f <= e <= fhi f /\ tsum (ft f) + e = cv).
  { apply andb_false_iff in Enb. destruct Enb as [Enb | Ew].
    - apply andb_false_iff in Enb.
      destruct Enb as [Ex | Es]; apply negb_false_iff in Ex || apply negb_false_iff in Es.
      + subst exact.
        destruct (match rg with Some g => Some g | None => form_range_nat F st f end) as [[x y] |] eqn:Er;
          [| discriminate].
        apply andb_true_iff in Ex. destruct Ex as [E1 E2]. apply Z.leb_le in E1, E2.
        exists e0. split; [exact He0 |].
        pose proof (Hnr x y eq_refl).
        apply cval_exact; [exact Hw | lia | exact Hq0].
      + subst single. destruct (single_value f r w sg Es) as [H1 H2].
        exists 0. split; [exact H1 | exact H2].
    - (* an unsigned upper bound that no wrapped value of the form meets *)
      apply negb_false_iff in Ew. subst wrap.
      destruct sg; [discriminate |]. destruct lo as [l |]; [discriminate |].
      destruct hi as [h |]; [| discriminate].
      destruct (match rg with Some g => Some g | None => form_range_nat F st f end) as [[x y] |] eqn:Er;
        [| discriminate].
      rewrite !andb_true_iff in Ew. destruct Ew as [[E1 E2] E3]. apply Z.ltb_lt in E1, E2, E3.
      pose proof (Hnr x y eq_refl) as HV.
      pose proof (Hhi h eq_refl) as Hh. fold cv in Hh, Hq0.
      pose proof (cval_range (OReg r) w false Hw) as Hc. fold cv in Hc. cbn [lo_ok hi_ok] in Hc.
      exists e0. split; [exact He0 |].
      destruct (Z_lt_le_dec (tsum (ft f) + e0) 0) as [Hneg | Hpos].
      + (* a negative value wraps to at least x + 2^w, past h *)
        exfalso.
        assert (Hq1 : eqm w (tsum (ft f) + e0 + 2 ^ w) cv).
        { unfold eqm in *. rewrite <- Hq0.
          replace (tsum (ft f) + e0 + 2 ^ w) with (tsum (ft f) + e0 + 1 * 2 ^ w) by ring.
          apply Z.mod_add. apply Z.pow_nonzero; lia. }
        pose proof (cval_exact (OReg r) w false (tsum (ft f) + e0 + 2 ^ w) Hw ltac:(cbn [lo_ok hi_ok]; lia) Hq1) as Hx. fold cv in Hx.
        lia.
      + apply cval_exact; [exact Hw | cbn [lo_ok hi_ok]; lia | exact Hq0]. }
  destruct Hval as [e [He Hve]].
  assert (Hin : forall x y, rg = Some (x, y) -> x <= cv <= y).
  { intros x y Er. rewrite <- Hve.
    apply (Hrg (tsum (ft f) + e) ltac:(rewrite Hve; apply eqm_refl) e He eq_refl x y Er). }
  pose proof (cval_range (OReg r) w sg Hw) as Hcr. fold cv in Hcr.
  set (lo' := match lo with
              | Some v => v
              | None => match rg with Some (x, _) => Z.max x (lo_ok w sg) | None => lo_ok w sg end
              end).
  set (hi' := match hi with
              | Some v => v
              | None => match rg with Some (_, y) => Z.min y (hi_ok w sg) | None => hi_ok w sg end
              end).
  assert (Hl : lo' <= cv).
  { unfold lo'. destruct lo as [l |]; [apply Hlo; reflexivity |].
    destruct rg as [[x y] |]; [pose proof (Hin x y eq_refl); lia | lia]. }
  assert (Hh : cv <= hi').
  { unfold hi'. destruct hi as [h |]; [apply Hhi; reflexivity |].
    destruct rg as [[x y] |]; [pose proof (Hin x y eq_refl); lia | lia]. }
  pose proof (round_in (Z.min (reg_align F st r) (w - 1)) lo' hi' cv
                (cv_aligned st r w sg Hst Hwr Hw) (conj Hl Hh)) as Hround.
  destruct (round (Z.min (reg_align F st r) (w - 1)) (lo', hi')) as [lo2 hi2]. cbn [fst snd] in Hround.
  (* the register's own read is the compared value *)
  assert (Hown : lo2 <= aval (r, if sg then ES else if Z.eqb (wd F r) 64 then EX else EZ) <= hi2).
  { assert (Hcv : cv = extend (wdn r) sg (rs r)) by (unfold cv, cval; cbn [opv]; rewrite Hn; reflexivity).
    destruct sg; cbn [atom_val extend] in *; [lia |].
    rewrite trunc_small in Hcv by lia. destruct (Z.eqb (wd F r) 64); cbn [atom_val]; lia. }
  assert (Hcf : forall out, state_ok F t out rs ->
                exists s, constrain_form F out st f lo2 hi2 rg = Some s /\ state_ok F t s rs).
  { intros out Hout. apply (constrain_form_sound out st f lo2 hi2 rg e Hout Hst He); [lia |].
    intros x y Er. rewrite Hve. apply (Hin x y Er). }
  match goal with |- context [if ?own then tighten_reg F st r ?ex lo2 hi2 else Some st] =>
    destruct own; [destruct (tighten_reg_sound st r ex lo2 hi2 Hst Hown) as [o [Eo Ho']]; rewrite Eo | ];
    [exact (Hcf o Ho') | exact (Hcf st Hst)] end.
Qed.

(* ------------------------------------------------------------------ *)
(* Comparisons.                                                        *)

(* The difference of two compared values, as the difference form D. *)
Lemma diff_value : forall st a b w sg,
    state_ok F t st rs -> 1 <= w <= 64 -> owidth (OReg a) w -> owidth b w ->
    let D := normalize F st (fadd (cmp_form F st (OReg a) w sg) (fscale (cmp_form F st b w sg) (-1) 64)) in
    w <= fm D ->
    exists e, flo D <= e <= fhi D /\ eqm w (tsum (ft D) + e) (cval (OReg a) w sg - cval b w sg).
Proof.
  intros st a b w sg Hst Hw Hoa Hob D HD.
  destruct (cmp_form_sound st (OReg a) w sg Hst Hw Hoa) as [Hwa [Hma [pa [Hsa Hqa]]]].
  destruct (cmp_form_sound st b w sg Hst Hw Hob) as [Hwb [Hmb [pb [Hsb Hqb]]]].
  pose proof (sat_fscale wdn rs _ (-1) 64 pb Hwb ltac:(lia) Hsb) as Hsc.
  pose proof (wf_fscale _ (-1) 64 Hwb ltac:(lia)) as Hwc.
  pose proof (sat_fadd wdn rs _ _ _ _ Hwa Hwc Hsa Hsc) as Hs.
  pose proof (wf_fadd _ _ Hwa Hwc) as Hwf.
  destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hwf Hs) as [HwD [HmD [e [He Hv]]]].
  fold D in HwD, HmD, He, Hv. exists e. split; [exact He |].
  apply (eqm_weaken (fm D) w) in Hv; [| unfold wf in HwD; lia].
  apply eqm_sym in Hv. eapply eqm_trans; [exact Hv |].
  replace (pa + -1 * pb) with (pa + - pb) by ring. replace (cval (OReg a) w sg - cval b w sg) with (cval (OReg a) w sg + - cval b w sg) by ring.
  apply eqm_add; [lia | exact Hqa | apply eqm_opp; [lia | exact Hqb]].
Qed.

(* Two values of one window of 2^w values differ by a multiple of 2^w only
   when they are equal. *)
Lemma window_diff_zero : forall w lo a b,
    0 <= w -> lo <= a < lo + 2 ^ w -> lo <= b < lo + 2 ^ w -> (a - b) mod 2 ^ w = 0 -> a = b.
Proof.
  intros w lo a b Hw Ha Hb H. apply (eqm_window w a b lo Hw Ha Hb). unfold eqm.
  assert (Hp : 0 < 2 ^ w) by (apply pow_nonneg_pos; lia).
  replace a with (b + (a - b)) by ring. rewrite Z.add_mod, H, Z.add_0_r, Z.mod_mod by lia. reflexivity.
Qed.

Lemma small_multiple_zero : forall w V, 0 <= w -> - 2 ^ w < V < 2 ^ w -> V mod 2 ^ w = 0 -> V = 0.
Proof.
  intros w V Hw HV H. assert (Hp : 0 < 2 ^ w) by (apply pow_nonneg_pos; lia).
  pose proof (Z.div_mod V (2 ^ w) ltac:(lia)) as Hd. rewrite H, Z.add_0_r in Hd.
  assert (Hq : V / 2 ^ w = 0) by nia. rewrite Hq in Hd. lia.
Qed.

(* The relation of a's own atom to a constant holds, and a state that holds
   is never refuted. *)
Lemma relate_atom_sound : forall st a fb w sg dlo dhi vb,
    state_ok F t st rs -> 1 <= w <= 64 -> wd F a = w ->
    (exists eb, flo fb <= eb <= fhi fb /\ tsum (ft fb) + eb = vb) ->
    dlo <= cval (OReg a) w sg - vb <= dhi ->
    exists s, relate_atom F st a fb w sg dlo dhi = Some s /\ state_ok F t s rs.
Proof.
  intros st a fb w sg dlo dhi vb Hst Hw Ha Hfb Hd. unfold relate_atom.
  destruct (cmp_atom F st a w sg) as [ga |] eqn:Eg; [| exists st; auto].
  destruct (existsb (cons_mentions a) (acons st)); [| exists st; auto].
  pose proof (relate_sound st ga fb dlo dhi (cval (OReg a) w sg) vb Hst
                (cmp_atom_sound st a w sg ga Hst Hw Ha Eg) Hfb Hd) as H.
  exists (relate st ga fb dlo dhi). split; [apply settle_sound; assumption | exact H].
Qed.

(* A constant's comparison form holds its value. *)
Lemma cmp_form_const : forall st z w sg,
    exists e, flo (cmp_form F st (OImm z) w sg) <= e <= fhi (cmp_form F st (OImm z) w sg) /\
              tsum (ft (cmp_form F st (OImm z) w sg)) + e = cval (OImm z) w sg.
Proof.
  intros st z w sg. exists (cval (OImm z) w sg). unfold cmp_form, fconst, cval. cbn [flo fhi ft opv GateForm.tsum].
  split; [lia | ring].
Qed.

Theorem refine_cmp_raw_sound : forall st k,
    state_ok F t st rs -> cmp_holds rs k ->
    exists s, refine_cmp_raw F st k = Some s /\ state_ok F t s rs.
Proof.
  intros st [c n sg a b] Hst Hk. unfold cmp_holds in Hk. cbn [cc cw csg ca cb] in Hk.
  unfold refine_cmp_raw, cmp_wf. cbn [cc cw csg ca cb]. cbv zeta.
  assert (Hdone : exists s, Some st = Some s /\ state_ok F t s rs) by (exists st; auto).
  destruct (Z.eqb (wd F a) (Z.of_nat n) && opnd_width_ok F b (Z.of_nat n)) eqn:Ewf; cbn [negb]; [| exact Hdone].
  apply andb_true_iff in Ewf. destruct Ewf as [Ea Eb]. apply Z.eqb_eq in Ea.
  set (w := Z.of_nat n) in *.
  assert (Hw : 1 <= w <= 64) by (pose proof (wdr a); unfold wd in Ea; lia).
  assert (Hoa : owidth (OReg a) w) by (intros r Hr; injection Hr as <-; exact Ea).
  assert (Hob : owidth b w) by (intros r Hr; subst b; cbn [opnd_width_ok] in Eb; apply Z.eqb_eq; exact Eb).
  assert (Hext : forall o, extend n sg (opv rs o) = cval o w sg) by (intros o; unfold cval, w; rewrite Nat2Z.id; reflexivity).
  change (rs a) with (opv rs (OReg a)) in Hk. rewrite !Hext in Hk.
  set (va := cval (OReg a) w sg) in *. set (vb := cval b w sg) in *.
  pose proof (cval_range (OReg a) w sg Hw) as Hra0. pose proof (cval_range b w sg Hw) as Hrb0.
  fold va in Hra0. fold vb in Hrb0.
  assert (Hfar : hi_ok w sg - lo_ok w sg = 2 ^ w - 1) by (rewrite hi_lo_ok by lia; lia).
  (* a < b, as a <= b - d, then the relation of the two forms *)
  assert (Hlt : forall d, 0 <= d -> va + d <= vb ->
            exists s, (match (match cmp_range F st b w sg with
                              | Some (_, y) => constrain F st (OReg a) None (Some (y - d)) w sg
                              | None => Some st end) with
                       | None => None
                       | Some s1 =>
                           match (match cmp_range F st (OReg a) w sg, b with
                                  | Some (x, _), OReg _ => constrain F s1 b (Some (x + d)) None w sg
                                  | _, _ => Some s1 end), cmp_range F st (OReg a) w sg, cmp_range F st b w sg, b with
                           | Some s2, Some _, Some _, OReg _ =>
                               settle F s2 (relate_cmp F s2 a (cmp_form F st (OReg a) w sg) (cmp_form F st b w sg) w sg (- (2 ^ w - 1)) (- d))
                           | Some s2, _, _, OImm _ =>
                               relate_atom F s2 a (cmp_form F st b w sg) w sg (- (2 ^ w - 1)) (- d)
                           | _, _, _, _ =>
                               match cmp_range F st (OReg a) w sg, b with
                               | Some (x, _), OReg _ => constrain F s1 b (Some (x + d)) None w sg
                               | _, _ => Some s1 end
                           end
                       end) = Some s /\ state_ok F t s rs).
  { intros d Hd Hab.
    assert (H1 : exists s1, (match cmp_range F st b w sg with
                             | Some (_, y) => constrain F st (OReg a) None (Some (y - d)) w sg
                             | None => Some st end) = Some s1 /\ state_ok F t s1 rs).
    { destruct (cmp_range F st b w sg) as [[xb yb] |] eqn:Erb; [| exact Hdone].
      pose proof (cmp_range_sound st b w sg xb yb Hst Hw Hob Erb) as Hb'. fold vb in Hb'.
      apply constrain_sound; [exact Hst | exact Hw | exact Hoa | discriminate |].
      intros h Hh. injection Hh as <-. fold va. lia. }
    destruct H1 as [s1 [E1 Hs1]]. rewrite E1.
    assert (H2 : exists s2, (match cmp_range F st (OReg a) w sg, b with
                             | Some (x, _), OReg _ => constrain F s1 b (Some (x + d)) None w sg
                             | _, _ => Some s1 end) = Some s2 /\ state_ok F t s2 rs).
    { destruct (cmp_range F st (OReg a) w sg) as [[xa ya] |] eqn:Era; [| exists s1; auto].
      pose proof (cmp_range_sound st (OReg a) w sg xa ya Hst Hw Hoa Era) as Ha'. fold va in Ha'.
      destruct b as [rb | z]; [| exists s1; auto].
      apply constrain_sound; [exact Hs1 | exact Hw | exact Hob | | discriminate].
      intros l Hl. injection Hl as <-. fold vb. lia. }
    destruct H2 as [s2 [E2 Hs2]]. rewrite E2.
    destruct b as [rb | z].
    2: { (* a constant: the relation of a's own atom *)
      destruct (cmp_range F st (OReg a) w sg) as [[xa ya] |];
        destruct (cmp_range F st (OImm z) w sg) as [[xb yb] |]; cbv iota beta;
        (apply (relate_atom_sound s2 a _ w sg _ _ vb Hs2 Hw (Hoa a eq_refl)); [exact (cmp_form_const st z w sg) | lia]). }
    destruct (cmp_range F st (OReg a) w sg) as [[xa ya] |] eqn:Era; [| exists s2; split; [reflexivity | exact Hs2]].
    destruct (cmp_range F st (OReg rb) w sg) as [[xb yb] |] eqn:Erb; [| exists s2; split; [reflexivity | exact Hs2]].
    eexists; split; [apply settle_sound; [exact Hs2 |] |].
    all: apply (relate_cmp_sound s2 a _ _ w sg _ _ vb Hs2 Hw (Hoa a eq_refl));
      [exact (cmp_exact st (OReg a) w sg xa ya Hst Hw Hoa Era) | exact (cmp_exact st (OReg rb) w sg xb yb Hst Hw Hob Erb) | lia]. }
  (* a > b, as b <= a - d *)
  assert (Hgt : forall d, 0 <= d -> vb + d <= va ->
            exists s, (match b with
                       | OReg rb =>
                           match (match cmp_range F st (OReg a) w sg with
                                  | Some (_, y) => constrain F st b None (Some (y - d)) w sg
                                  | None => Some st end) with
                           | None => None
                           | Some s1 =>
                               match (match cmp_range F st b w sg with
                                      | Some (x, _) => constrain F s1 (OReg a) (Some (x + d)) None w sg
                                      | None => Some s1 end), cmp_range F st (OReg a) w sg, cmp_range F st b w sg with
                               | Some s2, Some _, Some _ =>
                                   settle F s2 (relate_cmp F s2 a (cmp_form F st (OReg a) w sg) (cmp_form F st b w sg) w sg d (2 ^ w - 1))
                               | _, _, _ =>
                                   match cmp_range F st b w sg with
                                   | Some (x, _) => constrain F s1 (OReg a) (Some (x + d)) None w sg
                                   | None => Some s1 end
                               end
                           end
                       | OImm z =>
                           match constrain F st (OReg a) (Some (extend n sg z + d)) None w sg with
                           | Some s1 => relate_atom F s1 a (cmp_form F st b w sg) w sg d (2 ^ w - 1)
                           | None => None
                           end
                       end) = Some s /\ state_ok F t s rs).
  { intros d Hd Hab. destruct b as [rb | z].
    - assert (H1 : exists s1, (match cmp_range F st (OReg a) w sg with
                               | Some (_, y) => constrain F st (OReg rb) None (Some (y - d)) w sg
                               | None => Some st end) = Some s1 /\ state_ok F t s1 rs).
      { destruct (cmp_range F st (OReg a) w sg) as [[xa ya] |] eqn:Era; [| exact Hdone].
        pose proof (cmp_range_sound st (OReg a) w sg xa ya Hst Hw Hoa Era) as Ha'. fold va in Ha'.
        apply constrain_sound; [exact Hst | exact Hw | exact Hob | discriminate |].
        intros h Hh. injection Hh as <-. fold vb. lia. }
      destruct H1 as [s1 [E1 Hs1]]. rewrite E1.
      assert (H2 : exists s2, (match cmp_range F st (OReg rb) w sg with
                               | Some (x, _) => constrain F s1 (OReg a) (Some (x + d)) None w sg
                               | None => Some s1 end) = Some s2 /\ state_ok F t s2 rs).
      { destruct (cmp_range F st (OReg rb) w sg) as [[xb yb] |] eqn:Erb; [| exists s1; auto].
        pose proof (cmp_range_sound st (OReg rb) w sg xb yb Hst Hw Hob Erb) as Hb'. fold vb in Hb'.
        apply constrain_sound; [exact Hs1 | exact Hw | exact Hoa | | discriminate].
        intros l Hl. injection Hl as <-. fold va. lia. }
      destruct H2 as [s2 [E2 Hs2]]. rewrite E2.
      destruct (cmp_range F st (OReg a) w sg) as [[xa ya] |] eqn:Era; [| exists s2; split; [reflexivity | exact Hs2]].
      destruct (cmp_range F st (OReg rb) w sg) as [[xb yb] |] eqn:Erb; [| exists s2; split; [reflexivity | exact Hs2]].
      eexists; split; [apply settle_sound; [exact Hs2 |] |].
      all: apply (relate_cmp_sound s2 a _ _ w sg _ _ vb Hs2 Hw (Hoa a eq_refl));
        [exact (cmp_exact st (OReg a) w sg xa ya Hst Hw Hoa Era) | exact (cmp_exact st (OReg rb) w sg xb yb Hst Hw Hob Erb) | lia].
    - assert (Hz : extend n sg z = vb) by (unfold vb; rewrite <- (Hext (OImm z)); reflexivity).
      destruct (constrain_sound st (OReg a) (Some (extend n sg z + d)) None w sg Hst Hw Hoa)
        as [s1 [E1 Hs1]]; [intros l Hl; injection Hl as <-; fold va; lia | discriminate |].
      rewrite E1.
      apply (relate_atom_sound s1 a _ w sg _ _ vb Hs1 Hw (Hoa a eq_refl)); [exact (cmp_form_const st z w sg) | lia]. }
  destruct c; cbv iota; cbn [cmp_eval] in Hk;
    [apply Z.ltb_lt in Hk; specialize (Hlt 1 ltac:(lia) ltac:(lia))
    | apply Z.leb_le in Hk; specialize (Hlt 0 ltac:(lia) ltac:(lia))
    | apply Z.ltb_lt in Hk; apply (Hgt 1); lia
    | apply Z.leb_le in Hk; apply (Hgt 0); lia | |].
  1, 2: destruct b as [rb | z]; [exact Hlt |];
    destruct (cmp_range F st (OReg a) w sg) as [[xa ya] |] in Hlt;
    destruct (cmp_range F st (OImm z) w sg) as [[xb yb] |] in *;
    exact Hlt.
  (* equal or not: the difference D of the two forms *)
  all: set (D := normalize F st (fadd (cmp_form F st (OReg a) w sg) (fscale (cmp_form F st b w sg) (-1) 64))).
  all: destruct (Z.ltb_spec (fm D) w) as [HmD | HmD]; [exact Hdone |].
  all: destruct (diff_value st a b w sg Hst Hw Hoa Hob HmD) as [e [He Hq]]; fold D va vb in He, Hq.
  all: pose proof (cval_range (OReg a) w sg Hw) as Hra; pose proof (cval_range b w sg Hw) as Hrb;
       fold va in Hra; fold vb in Hrb.
  all: assert (Hp : 0 < 2 ^ w) by (apply pow_nonneg_pos; lia).
  all: assert (Hwin : 2 ^ w = hi_ok w sg - lo_ok w sg + 1) by (rewrite hi_lo_ok by lia; lia).
  all: destruct (null (ft D) && Z.eqb (flo D) (fhi D)) eqn:Ec.
  1, 3: apply andb_true_iff in Ec; destruct Ec as [Hn Hlh]; apply Z.eqb_eq in Hlh;
        destruct (ft D) eqn:Eft; [| discriminate]; cbn [GateForm.tsum] in Hq;
        assert (e = flo D) by lia; subst e; unfold eqm in Hq.
  1: { (* equal, D a constant *)
    apply Z.eqb_eq in Hk. rewrite Hk, Z.sub_diag, Z.mod_0_l in Hq by lia. rewrite Z.add_0_l in Hq.
    rewrite Hq. cbn [Z.eqb Bool.eqb]. exact Hdone. }
  1: { (* not equal, D a constant *)
    apply negb_true_iff, Z.eqb_neq in Hk. rewrite Z.add_0_l in Hq.
    destruct (Z.eqb_spec (flo D mod 2 ^ w) 0) as [H0 | H0]; [| exact Hdone].
    exfalso. apply Hk. apply (window_diff_zero w (lo_ok w sg)); [lia | lia | lia |]. rewrite <- Hq. exact H0. }
  (* D with terms: its range *)
  all: destruct (form_range F st D) as [[x y] |] eqn:Er; [| exact Hdone].
  all: pose proof (form_range_sound F E HE t st rs Hst Hrs D x y e Er He) as Hxy.
  all: destruct ((x <=? - 2 ^ w) || (2 ^ w <=? y)) eqn:Ewin; [exact Hdone |].
  all: apply orb_false_iff in Ewin; destruct Ewin as [Eb1 Eb2]; apply Z.leb_gt in Eb1, Eb2.
  all: set (al := Z.min (Z.min (reg_align F st a) (opnd_align F st b)) (w - 1)).
  - (* equal: D is 0 *)
    apply Z.eqb_eq in Hk. unfold eqm in Hq. rewrite Hk, Z.sub_diag, Z.mod_0_l in Hq by lia.
    pose proof (small_multiple_zero w (tsum (ft D) + e) ltac:(lia) ltac:(lia) Hq) as H0.
    apply (constrain_form_sound st st D 0 0 _ e Hst Hst He); [lia |].
    intros x' y' Hg. injection Hg as <- <-. rewrite H0. apply round_in; [apply aligned_zero | lia].
  - (* not equal: D is a nonzero multiple of 2^al *)
    apply negb_true_iff, Z.eqb_neq in Hk.
    set (V := tsum (ft D) + e) in *.
    assert (HV0 : V <> 0).
    { intros HV. apply Hk. apply (window_diff_zero w (lo_ok w sg)); [lia | lia | lia |].
      unfold eqm in Hq. rewrite <- Hq, HV. apply Z.mod_0_l. lia. }
    assert (Haa : aligned al va).
    { pose proof (cv_aligned st a w sg Hst Ea Hw) as H. fold va in H.
      apply (aligned_mono _ al) in H; [exact H | unfold al; lia]. }
    assert (Hab : aligned al vb).
    { pose proof (opnd_align_sound F E HE t st rs Hst Hrs b) as H.
      apply (aligned_mono _ al) in H; [| unfold al; lia].
      apply (aligned_eqm w al vb (opv rs b)); [lia | unfold al; lia | | exact H].
      unfold vb, cval. pose proof (extend_eqm (Z.to_nat w) sg (opv rs b)) as He'. rewrite Z2Nat.id in He' by lia.
      exact He'. }
    assert (HaV : aligned al V).
    { apply (aligned_eqm w al V (va - vb)); [lia | unfold al; lia | exact Hq |].
      replace (va - vb) with (va + - vb) by ring. apply aligned_add; [exact Haa | apply aligned_opp; exact Hab]. }
    pose proof (round_in al x y V HaV Hxy) as Hg.
    destruct (round al (x, y)) as [g1 g2] eqn:Eg. cbn [fst snd] in *.
    destruct HaV as [q Hq'].
    assert (Hpa : 0 < 2 ^ Z.max 0 al) by (apply pow_nonneg_pos; lia).
    (* 2^al is at most the power V is a multiple of; for al < 0 it is 0 *)
    assert (Hpow : 2 ^ al <= 2 ^ Z.max 0 al).
    { destruct (Z.le_gt_cases 0 al); [rewrite Z.max_r by lia; lia | rewrite Z.pow_neg_r by lia; lia]. }
    destruct (Z.eqb_spec g1 0) as [Hg1 | Hg1].
    + assert (Hq1 : 1 <= q).
      { destruct (Z.le_gt_cases q 0) as [Hq0 | Hq0]; [| lia]. assert (q * 2 ^ Z.max 0 al <= 0) by nia. lia. }
      apply (constrain_form_sound st st D _ _ _ e Hst Hst He); [fold V; split; [nia | lia] |].
      intros x' y' Hs. injection Hs as <- <-. fold V. lia.
    + destruct (Z.eqb_spec g2 0) as [Hg2 | Hg2]; [| exact Hdone].
      assert (Hq1 : q <= -1).
      { destruct (Z.le_gt_cases 0 q) as [Hq0 | Hq0]; [| lia]. assert (0 <= q * 2 ^ Z.max 0 al) by nia. lia. }
      apply (constrain_form_sound st st D _ _ _ e Hst Hst He); [fold V; split; [lia | nia] |].
      intros x' y' Hs. injection Hs as <- <-. fold V. lia.
Qed.

Lemma extend_equality : forall n sg x y,
 (extend n sg x = extend n sg y <-> trunc n x = trunc n y).
Proof.
 intros n sg x y. split.
 - intros H. pose proof (extend_eqm n sg x) as Hx.
   pose proof (extend_eqm n sg y) as Hy. unfold eqm in Hx,Hy.
   unfold trunc. rewrite <- Hx, <- Hy, H. reflexivity.
 - intros H. destruct sg; unfold extend, sext; rewrite H; reflexivity.
Qed.

Lemma unsigned_equality_holds : forall qrs k,
 cmp_holds qrs k -> cmp_holds qrs (unsigned_equality k).
Proof.
 intros qrs [c n sg a b] H. destruct c; try exact H;
 unfold cmp_holds, unsigned_equality in *; cbn [cc cw csg ca cb cmp_eval extend] in *.
 - apply Z.eqb_eq in H. apply Z.eqb_eq. apply (proj1 (extend_equality n sg _ _)). exact H.
 - apply negb_true_iff, Z.eqb_neq in H. apply negb_true_iff, Z.eqb_neq.
   intros He. apply H. apply (proj2 (extend_equality n sg _ _)). exact He.
Qed.

Theorem refine_cmp_sound : forall st k,
 state_ok F t st rs -> cmp_holds rs k ->
 exists s, refine_cmp F st k = Some s /\ state_ok F t s rs.
Proof.
 intros st k Hst Hk. unfold refine_cmp. apply refine_cmp_raw_sound; [exact Hst |].
 apply unsigned_equality_holds. exact Hk.
Qed.

Theorem refine_all_sound : forall cs st,
    state_ok F t st rs -> Forall (cmp_holds rs) cs ->
    exists s, refine_all F st cs = Some s /\ state_ok F t s rs.
Proof.
  intros cs. induction cs as [| k cs IH]; intros st Hst Hall; cbn [refine_all]; [exists st; auto |].
  inversion Hall as [| k' cs' Hk Hcs]; subst.
  destruct (refine_cmp_sound st k Hst Hk) as [s [Es Hs]]. rewrite Es. exact (IH s Hs Hcs).
Qed.


Lemma cmps_split : forall cs, Forall (cmp_holds rs) cs \/ exists k, In k cs /\ ~ cmp_holds rs k.
Proof.
  intros cs. induction cs as [| k cs IH]; [left; constructor |].
  destruct (cmp_eval (cc k) (extend (cw k) (csg k) (rs (ca k))) (extend (cw k) (csg k) (opv rs (cb k)))) eqn:Ek.
  - destruct IH as [IH | [k' [Hin Hk']]].
    + left. constructor; [exact Ek | exact IH].
    + right. exists k'. split; [right; exact Hin | exact Hk'].
  - right. exists k. split; [left; reflexivity |]. unfold cmp_holds. rewrite Ek. discriminate.
Qed.

(* Comparisons that hold when all is true, and, when they are exact, only then. *)
Lemma common_constraints_left : forall a b,
 Forall (cons_ok F rs) a -> Forall (cons_ok F rs) (common_constraints a b).
Proof.
 intros a b Ha. apply Forall_forall. intros k Hk.
 unfold common_constraints in Hk. apply in_flat_map in Hk.
 destruct Hk as [[[ts lo] hi] [Hin Hk]].
 pose proof (proj1 (Forall_forall _ _) Ha _ Hin) as Hc.
 destruct (find (fun k' => let '(ts',_,_) := k' in terms_eqb ts ts') b) as [[[ts' lo'] hi'] |] eqn:Efind; [| contradiction].
 destruct Hk as [<- | []]. cbn [cons_ok] in *.
 split; [eapply Z.le_trans; [apply Z.le_min_l | exact (proj1 Hc)] | eapply Z.le_trans; [exact (proj2 Hc) | apply Z.le_max_l]].
Qed.

Lemma common_constraints_right : forall a b,
 Forall (cons_ok F rs) b -> Forall (cons_ok F rs) (common_constraints a b).
Proof.
 intros a b Hb. apply Forall_forall. intros k Hk.
 unfold common_constraints in Hk. apply in_flat_map in Hk.
 destruct Hk as [[[ts lo] hi] [Hin Hk]].
 destruct (find (fun k' => let '(ts',_,_) := k' in terms_eqb ts ts') b) as [[[ts' lo'] hi'] |] eqn:Efind; [| contradiction].
 apply find_some in Efind. destruct Efind as [Hin' Heq]. cbn in Heq. apply terms_eqb_eq in Heq. subst ts'.
 pose proof (proj1 (Forall_forall _ _) Hb _ Hin') as Hc.
 destruct Hk as [<- | []]. cbn [cons_ok] in *.
 split; [eapply Z.le_trans; [apply Z.le_min_r | exact (proj1 Hc)] | eapply Z.le_trans; [exact (proj2 Hc) | apply Z.le_max_r]].
Qed.

Lemma common_fold_sound : forall states cs,
 (Forall (cons_ok F rs) cs \/ exists s, In s states /\ Forall (cons_ok F rs) (acons s)) ->
 Forall (cons_ok F rs) (fold_left (fun cs s => common_constraints cs (acons s)) states cs).
Proof.
 induction states as [|s states IH]; intros cs H; cbn.
 - destruct H as [H | [s [[] _]]]; exact H.
 - apply IH. destruct H as [H | [s' [[Heq | Hin] Hs]]].
   + left. apply common_constraints_left. exact H.
   + subst s'. left. apply common_constraints_right. exact Hs.
   + right. exists s'. auto.
Qed.

Lemma join_alternatives_sound : forall st states s,
 state_ok F t st rs -> In s states -> state_ok F t s rs ->
 exists out, join_alternatives st states = Some out /\ state_ok F t out rs.
Proof.
 intros st states s Hst Hin Hs. destruct states as [|a rest]; [contradiction |].
 destruct rest as [|b rest].
 - destruct Hin as [<- | []]. exists a. auto.
 - eexists. split; [reflexivity |]. split; [exact (proj1 Hst) |].
   apply common_fold_sound. destruct Hin as [<- | Hin].
   + left. exact (proj2 Hs).
   + right. exists s. split; [exact Hin | exact (proj2 Hs)].
Qed.

Lemma refine_cmps_sound : forall st cs exact all,
    state_ok F t st rs -> (all = true -> Forall (cmp_holds rs) cs) ->
    (exact = true -> Forall (cmp_holds rs) cs -> all = true) ->
    exists s, refine_cmps F st cs exact all = Some s /\ state_ok F t s rs.
Proof.
  intros st cs exact all Hst Hf1 Hf2. unfold refine_cmps.
  assert (Hdone : exists s, Some st = Some s /\ state_ok F t s rs) by (exists st; auto).
  destruct all; [exact (refine_all_sound cs st Hst (Hf1 eq_refl)) |].
  destruct exact; cbn [negb]; [| exact Hdone].
  destruct (cmps_split cs) as [Hall | [k [Hin Hnk]]]; [discriminate (Hf2 eq_refl Hall) |].
  pose proof (proj2 (negate_holds rs k) Hnk) as Hneg.
  destruct (refine_cmp_sound st (negate k) Hst Hneg) as [s [Es Hs]].
  apply (join_alternatives_sound st _ s Hst); [| exact Hs].
  apply in_flat_map. exists k. split; [exact Hin |]. rewrite Es. left. reflexivity.
Qed.


Theorem refine_pred_sound : forall st p b,
    state_ok F t st rs -> negb (Z.eqb (rs p) 0) = b ->
    exists s, refine_pred F st p b = Some s /\ state_ok F t s rs.
Proof.
  intros st p b Hst Hb. unfold refine_pred.
  assert (Hdone : exists s, Some st = Some s /\ state_ok F t s rs) by (exists st; auto).
  destruct (get st p) as [x |] eqn:Eg; [| exact Hdone].
  pose proof (proj1 Hst p _ Eg) as Hf. unfold fact_ok in Hf.
  destruct x as [f al vr | | cs ex | cs ex]; try exact Hdone; cbn [fact_okv] in Hf.
  - (* a predicate set to a constant *)
    destruct (null (ft f) && Z.eqb (flo f) (fhi f) && (1 <=? fm f) && Z.eqb (wd F p) 1) eqn:Ec; [| exact Hdone].
    rewrite !andb_true_iff in Ec. destruct Ec as [[[Hn Hlh] Hm] Hw].
    apply Z.eqb_eq in Hlh, Hw. apply Z.leb_le in Hm.
    destruct Hf as [Hwf [[e [He Hv]] _]].
    destruct (ft f) eqn:Eft; [| discriminate]. cbn [GateForm.tsum] in Hv.
    assert (e = flo f) by lia. subst e.
    pose proof (rsr p) as Hr. unfold wd in Hw. rewrite Hw in Hr.
    apply (eqm_weaken (fm f) 1) in Hv; [| lia]. unfold eqm in Hv. rewrite Z.pow_1_r, Z.add_0_l in Hv.
    rewrite (Z.mod_small (rs p)) in Hv by (rewrite Z.pow_1_r in Hr; lia).
    rewrite <- Hv, <- Hb. rewrite Bool.eqb_reflx. exact Hdone.
  - destruct Hf as [Hf1 Hf2]. apply refine_cmps_sound; [exact Hst | |].
    + intros ->. apply Hf1. intros H0. rewrite H0 in Hb. discriminate.
    + intros He Hall. rewrite <- Hb. apply negb_true_iff, Z.eqb_neq. exact (Hf2 He Hall).
  - destruct Hf as [Hf1 Hf2]. apply refine_cmps_sound; [exact Hst | |].
    + intros Hn. apply Hf1. rewrite <- Hb in Hn. apply negb_true_iff, negb_false_iff, Z.eqb_eq in Hn. exact Hn.
    + intros He Hall. rewrite <- Hb, (Hf2 He Hall). reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* A predicate's known value.                                          *)

Lemma subst_pred_sound : forall st p v, state_ok F t st rs -> rs p = v -> state_ok F t (subst_pred st p v) rs.
Proof.
  intros st p v [Hf Hc] Hp. split.
  - intros r x Hx. unfold get, subst_pred in Hx. cbn [amap] in Hx.
    unfold PositiveMap.map in Hx. rewrite PositiveMap.gmapi in Hx.
    destruct (PM.find r (amap st)) as [y |] eqn:Ey; [| discriminate]. cbn [option_map] in Hx. injection Hx as <-.
    pose proof (Hf r y Ey) as Hy. unfold fact_ok in *. destruct y as [f al vr | | | ]; try exact Hy.
    cbn [fact_okv] in *. destruct Hy as [Hwf [[e [He Hv]] [Hal Hvr]]].
    unfold subst_form. pose proof (subst_terms_tsum wdn rs p v (ft f) Hp) as Ht.
    destruct (subst_terms p v (ft f)) as [ts k] eqn:Es. cbn [fst snd] in Ht.
    split; [exact Hwf |]. split; [| split; [exact Hal | exact Hvr]].
    exists (e + k). cbn [fm ft flo fhi]. split; [lia |]. rewrite <- Ht in Hv.
    replace (tsum ts + (e + k)) with (tsum ts + k + e) by ring. exact Hv.
  - unfold subst_pred. cbn [acons]. apply Forall_forall. intros k' Hk'. apply in_map_iff in Hk'.
    destruct Hk' as [[[ts lo] hi] [Hk' Hin]].
    pose proof (proj1 (Forall_forall _ _) Hc _ Hin) as Hok. cbn [cons_ok] in Hok.
    pose proof (subst_terms_tsum wdn rs p v ts Hp) as Ht.
    destruct (subst_terms p v ts) as [ts' c] eqn:Es. subst k'. cbn [cons_ok fst snd] in *. lia.
Qed.

Theorem refine_sound : forall st p b,
    state_ok F t st rs -> negb (Z.eqb (rs p) 0) = b ->
    exists s, refine F st p b = Some s /\ state_ok F t s rs.
Proof.
  intros st p b Hst Hb. unfold refine.
  destruct (refine_pred_sound st p b Hst Hb) as [s [Es Hs]]. rewrite Es.
  eexists; split; [reflexivity |].
  destruct (Z.eqb_spec (wd F p) 1) as [H1 | H1]; [| exact Hs].
  apply subst_pred_sound; [exact Hs |].
  pose proof (rsr p) as Hr. unfold wd in H1. rewrite H1 in Hr.
  destruct b; destruct (Z.eqb_spec (rs p) 0); cbn in Hb; try discriminate; lia.
Qed.

(* A rejected branch fixes the value of a one-bit predicate. *)
Lemma impossible_predicate_value : forall st p b,
 state_ok F t st rs -> wd F p = 1 -> refine F st p b = None ->
 rs p = if b then 0 else 1.
Proof.
 intros st p b Hst Hw Hbad.
 assert (Hneq : negb (Z.eqb (rs p) 0) <> b).
 { intros Hb. destruct (refine_sound st p b Hst Hb) as [s [Es Hs]]. congruence. }
 pose proof (rsr p) as Hr. unfold wd in Hw. rewrite Hw in Hr.
 destruct b; destruct (Z.eqb_spec (rs p) 0); cbn in Hneq; lia.
Qed.

Lemma put_impossible_predicate : forall st p b,
 state_ok F t st rs -> wd F p = 1 -> refine F st p b = None ->
 state_ok F t (put st p (FForm (fconst 1 (if b then 0 else 1)) 0 None)) rs.
Proof.
 intros st p b Hst Hw Hbad. apply put_ok; [exact Hst |].
 pose proof (impossible_predicate_value st p b Hst Hw Hbad) as Hp.
 cbn [fact_ok fact_okv]. split; [unfold fconst,wf; cbn; lia |].
 split; [apply sat_fconst; rewrite Hp; apply eqm_refl |].
 split; [apply aligned_nonpos; lia | discriminate].
Qed.

Theorem refine_guard_sound : forall st g,
    state_ok F t st rs -> holds g rs = true ->
    exists s, refine_guard F st g = Some s /\ state_ok F t s rs.
Proof.
  intros st [[neg p] |] Hst Hg; cbn [refine_guard]; [| exists st; auto].
  apply refine_sound; [exact Hst |]. cbn [holds] in Hg.
  destruct neg, (Z.eqb (rs p) 0); cbn in Hg |- *; congruence.
Qed.

End Refine.
