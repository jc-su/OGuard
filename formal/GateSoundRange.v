(*
  Soundness of the checker's ranges and alignments: every range that
  `reg_range`, `sreg_range`, `atom_range`, or `form_range` gives contains the
  value, and every alignment that `reg_align` or `form_align` gives divides
  it, in every register state that the abstract state describes.
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

Local Open Scope Z_scope.

(* Conversion unfolds the fuel wrappers before the fuel-indexed functions,
   so that it never unrolls the fuel. *)
Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

(* ------------------------------------------------------------------ *)
(* Arithmetic.                                                         *)

Lemma pow_nonneg_pos : forall k, 0 <= k -> 0 < 2 ^ k.
Proof. intros. apply Z.pow_pos_nonneg; lia. Qed.

Lemma trunc_range : forall n v, 0 <= trunc n v < 2 ^ Z.of_nat n.
Proof. intros. unfold trunc. apply Z.mod_pos_bound. apply pow_nonneg_pos. lia. Qed.

Lemma trunc_small : forall n v, 0 <= v < 2 ^ Z.of_nat n -> trunc n v = v.
Proof. intros. unfold trunc. apply Z.mod_small. assumption. Qed.

Lemma trunc_eqm : forall n v, eqm (Z.of_nat n) (trunc n v) v.
Proof. intros. unfold eqm, trunc. apply Z.mod_mod. pose proof (pow_nonneg_pos (Z.of_nat n)). lia. Qed.

Lemma sext_small : forall w v, (1 <= w)%nat -> 0 <= v < 2 ^ (Z.of_nat w - 1) -> sext w v = v.
Proof.
  intros w v Hw Hv. unfold sext.
  assert (H2 : 2 ^ Z.of_nat w = 2 * 2 ^ (Z.of_nat w - 1)).
  { replace (Z.of_nat w) with (Z.of_nat w - 1 + 1) at 1 by lia. rewrite Z.pow_add_r by lia. lia. }
  rewrite trunc_small by lia. destruct (Z.ltb_spec v (2 ^ (Z.of_nat w - 1))); lia.
Qed.

(* Two values equal modulo 2^m that both lie in [0, 2^w), w <= m, are equal. *)
Lemma eqm_small_eq : forall m w a b,
    0 <= w <= m -> 0 <= a < 2 ^ w -> 0 <= b < 2 ^ w -> eqm m a b -> a = b.
Proof.
  intros m w a b Hw Ha Hb H.
  apply (eqm_weaken m w a b) in H; [| lia]. unfold eqm in H.
  rewrite !Z.mod_small in H by lia. exact H.
Qed.

(* A w-bit value equal modulo 2^m (m >= w) to a small signed x reads as x. *)
Lemma eqm_sext_eq : forall m w v x,
    (1 <= w)%nat -> Z.of_nat w <= m -> 0 <= v < 2 ^ Z.of_nat w ->
    - 2 ^ (Z.of_nat w - 1) <= x < 2 ^ (Z.of_nat w - 1) -> eqm m v x -> sext w v = x.
Proof.
  intros m w v x Hw Hm Hv Hx H.
  assert (H2 : 2 ^ Z.of_nat w = 2 * 2 ^ (Z.of_nat w - 1)).
  { replace (Z.of_nat w) with (Z.of_nat w - 1 + 1) at 1 by lia. rewrite Z.pow_add_r by lia. lia. }
  apply (eqm_weaken m (Z.of_nat w) v x) in H; [| lia]. unfold eqm in H.
  unfold sext. rewrite trunc_small by lia.
  rewrite (Z.mod_small v) in H by lia.
  destruct (Z.le_gt_cases 0 x) as [Hx0 | Hx0].
  - rewrite Z.mod_small in H by lia. subst. destruct (Z.ltb_spec x (2 ^ (Z.of_nat w - 1))); lia.
  - assert (Hm2 : x mod 2 ^ Z.of_nat w = x + 2 ^ Z.of_nat w).
    { rewrite <- (Z.mod_add x 1) by lia. rewrite Z.mul_1_l. apply Z.mod_small. lia. }
    rewrite Hm2 in H. subst. destruct (Z.ltb_spec (x + 2 ^ Z.of_nat w) (2 ^ (Z.of_nat w - 1))); lia.
Qed.

Lemma mul_between : forall a b v c,
    a <= v <= b -> Z.min (a * c) (b * c) <= v * c <= Z.max (a * c) (b * c).
Proof.
  intros a b v c H. destruct (Z.le_ge_cases 0 c).
  - split; [apply Z.min_le_iff; left | apply Z.max_le_iff; right]; nia.
  - split; [apply Z.min_le_iff; right | apply Z.max_le_iff; left]; nia.
Qed.

Lemma div_between : forall a b v k,
    0 < k -> a <= v <= b -> a / k <= v / k <= b / k.
Proof. intros. split; apply Z.div_le_mono; lia. Qed.

Lemma aligned_mono : forall k k' v, k' <= k -> aligned k v -> aligned k' v.
Proof.
  unfold aligned. intros k k' v Hk [q Hq]. exists (q * 2 ^ (Z.max 0 k - Z.max 0 k')).
  rewrite Hq, <- Z.mul_assoc, <- Z.pow_add_r by lia. f_equal. f_equal. lia.
Qed.

Lemma aligned_max : forall a b v, aligned a v -> aligned b v -> aligned (Z.max a b) v.
Proof.
  intros a b v Ha Hb. destruct (Z.le_ge_cases a b).
  - rewrite Z.max_r by lia. exact Hb.
  - rewrite Z.max_l by lia. exact Ha.
Qed.

Lemma aligned_zero : forall k, aligned k 0.
Proof. intros. exists 0. lia. Qed.

Lemma aligned_nonpos : forall k v, k <= 0 -> aligned k v.
Proof. intros k v Hk. unfold aligned. rewrite Z.max_l by lia. exists v. lia. Qed.

Lemma aligned_add : forall k a b, aligned k a -> aligned k b -> aligned k (a + b).
Proof. unfold aligned. intros. apply Z.divide_add_r; assumption. Qed.

Lemma aligned_opp : forall k v, aligned k v -> aligned k (- v).
Proof. unfold aligned. intros k v H. apply Z.divide_opp_r. exact H. Qed.

Lemma aligned_v2 : forall z, aligned (v2 z) z.
Proof. intros z. unfold aligned. rewrite Z.max_r by apply v2_nonneg. apply v2_divides. Qed.

Lemma aligned_mul : forall j k c v, 0 <= j -> (2 ^ j | c) -> aligned k v -> aligned (j + k) (c * v).
Proof.
  unfold aligned. intros j k c v Hj [p Hp] [q Hq]. subst.
  destruct (Z.le_ge_cases 0 k).
  - rewrite !Z.max_r by lia. exists (p * q). rewrite Z.pow_add_r by lia. ring.
  - exists (p * q * 2 ^ (Z.max 0 k) * 2 ^ (j - Z.max 0 (j + k))).
    rewrite (Z.max_l 0 k) by lia. rewrite Z.pow_0_r.
    replace (2 ^ j) with (2 ^ (j - Z.max 0 (j + k)) * 2 ^ Z.max 0 (j + k)) by
      (rewrite <- Z.pow_add_r by lia; f_equal; lia).
    ring.
Qed.

(* A multiple of 2^k modulo 2^m, k <= m: the representative is one too. *)
Lemma aligned_eqm : forall m k a b, 0 <= m -> k <= m -> eqm m a b -> aligned k b -> aligned k a.
Proof.
  unfold aligned, eqm. intros m k a b Hm Hk H [q Hq].
  destruct (Z.le_ge_cases k 0) as [Hk0 | Hk0].
  - rewrite Z.max_l by lia. exists a. lia.
  - rewrite Z.max_r in * by lia.
    assert (Hp : 0 < 2 ^ m) by (apply pow_nonneg_pos; lia).
    assert (Hd : (2 ^ k | 2 ^ m)) by (exists (2 ^ (m - k)); rewrite <- Z.pow_add_r by lia; f_equal; lia).
    assert (Hab : (2 ^ m | a - b)).
    { apply Z.mod_divide; [lia |]. rewrite Zminus_mod, H, Z.sub_diag. apply Z.mod_0_l. lia. }
    replace a with (b + (a - b)) by ring.
    apply Z.divide_add_r; [exists q; exact Hq | eapply Z.divide_trans; [exact Hd | exact Hab]].
Qed.

(* ------------------------------------------------------------------ *)
(* Sums of terms.                                                      *)

Section Sums.

Variable wdn : reg -> nat.
Variable rs : reg -> Z.

Lemma tsum_filter : forall (p : atom * Z -> bool) ts,
    tsum wdn rs ts = tsum wdn rs (filter p ts) + tsum wdn rs (filter (fun t => negb (p t)) ts).
Proof.
  intros p ts. induction ts as [| [x c] ts IH]; cbn [filter tsum]; [lia |].
  destruct (p (x, c)); cbn [negb tsum]; lia.
Qed.

(* The terms with register p's value atoms read as v, and the constant they
   add, sum to the terms' sum in every valuation where p holds v. *)
Lemma subst_terms_tsum : forall p v ts, rs p = v ->
    tsum wdn rs (fst (subst_terms p v ts)) + snd (subst_terms p v ts) = tsum wdn rs ts.
Proof.
  intros p v ts Hp. unfold subst_terms. cbn [fst snd].
  assert (Hgen : forall l acc,
            tsum wdn rs (filter (fun t0 => negb (value_atom p (fst t0))) l) +
            fold_left (fun acc (t0 : atom * Z) => if value_atom p (fst t0) then acc + snd t0 * v else acc) l acc =
            tsum wdn rs l + acc).
  { induction l as [| [[r e] c] l IH]; intros acc; cbn [filter fold_left tsum fst snd]; [lia |].
    destruct (value_atom p (r, e)) eqn:Ev; cbn [negb tsum].
    - transitivity (tsum wdn rs l + (acc + c * v)); [apply IH |].
      unfold value_atom in Ev. cbn [fst snd] in Ev. apply andb_true_iff in Ev. destruct Ev as [Er Ee].
      apply Pos.eqb_eq in Er. subst r. destruct e; try discriminate; cbn [atom_val]; rewrite Hp; lia.
    - rewrite <- !Z.add_assoc. f_equal. apply IH. }
  pose proof (Hgen ts 0). lia.
Qed.

(* Terms whose atoms each lie in a range: the sum lies in the sum of the
   ranges. *)
Lemma plain_sum_sound : forall (ar : atom -> option range) ts lo hi L H e,
    (forall x c a b, In (x, c) ts -> ar x = Some (a, b) -> a <= atom_val wdn rs x <= b) ->
    plain_sum ar ts lo hi = Some (L, H) -> lo <= e <= hi -> L <= tsum wdn rs ts + e <= H.
Proof.
  intros ar ts. induction ts as [| [x c] ts IH]; intros lo hi L H e Har Hs He; cbn [plain_sum tsum] in *.
  - inversion Hs. subst. lia.
  - destruct (ar x) as [[a b] |] eqn:Ex; [| discriminate].
    pose proof (mul_between a b (atom_val wdn rs x) c (Har x c a b (or_introl eq_refl) Ex)) as Hm.
    specialize (IH _ _ L H (e + c * atom_val wdn rs x)
                  (fun x' c' a' b' Hi He' => Har x' c' a' b' (or_intror Hi) He') Hs).
    rewrite (Z.mul_comm c) in *. lia.
Qed.

Lemma scaled_sum : forall q pick ts,
    scaled q pick ts = true -> tsum wdn rs pick = q * tsum wdn rs ts.
Proof.
  intros q pick. induction pick as [| [x c'] pick IH]; intros ts H; destruct ts as [| [y c] ts];
    cbn [scaled tsum] in *; try discriminate; [lia |].
  rewrite !andb_true_iff in H. destruct H as [[Hx Hc] Hr].
  apply atom_eqb_eq in Hx. apply Z.eqb_eq in Hc. subst. rewrite (IH ts Hr). ring.
Qed.

Lemma tsum_nonzero : forall ts, tsum wdn rs (filter (fun t => negb (Z.eqb (snd t) 0)) ts) = tsum wdn rs ts.
Proof.
  intros ts. induction ts as [| [x c] ts IH]; cbn [filter]; [reflexivity |].
  cbn [snd]. destruct (Z.eqb_spec c 0) as [Hc | Hc]; cbn [negb].
  - subst c. rewrite IH. cbn [tsum]. ring.
  - cbn [tsum]. rewrite IH. reflexivity.
Qed.

End Sums.

(* ------------------------------------------------------------------ *)
(* Ranges.                                                             *)

Section Ranges.

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

Lemma rs_range : forall r, 0 <= rs r < 2 ^ Z.of_nat (wdn r).
Proof. intros. apply (proj1 Hrs). Qed.

Lemma wd_range : forall r, 1 <= Z.of_nat (wdn r) <= 64.
Proof. intros. apply (ok_widths F E HE). Qed.

(* plain_range's second range reads x as 2^n (x >> n) + x mod 2^n: the three
   terms it adds sum to 0, since a register lies below 2^w for its width w,
   so its bits from n are x >> n. *)
Lemma plain_range_sound : forall (ar : atom -> option range) f L H e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    plain_range F ar f = Some (L, H) -> flo f <= e <= fhi f -> L <= tsum wdn rs (ft f) + e <= H.
Proof.
  intros ar f L H e Har Hp He. unfold plain_range in Hp. cbv zeta in Hp.
  assert (H0 : forall L' H', plain_sum ar (ft f) (flo f) (fhi f) = Some (L', H') ->
                             L' <= tsum wdn rs (ft f) + e <= H')
    by (intros L' H' Hs; apply (plain_sum_sound wdn rs ar (ft f) (flo f) (fhi f));
        [intros x c a b _ Hx; exact (Har x a b Hx) | exact Hs | exact He]).
  destruct (low_bits (ft f)) as [[[[[r ex] hi] n] c] |]; [| exact (H0 L H Hp)].
  match type of Hp with (if ?b then _ else _) = _ => destruct b eqn:Eok end; [| exact (H0 L H Hp)].
  rewrite !andb_true_iff in Eok. destruct Eok as [[Hwh Hhi] Hn]. apply Z.leb_le in Hn.
  match type of Hp with inter _ (plain_sum _ ?l _ _) = _ => set (ts := l) in Hp end.
  (* the atom of the high bits is x >> n *)
  assert (Hhv : aval (r, hi) = rs r / 2 ^ n).
  { destruct hi as [| | | k | a m]; cbn in Hhi; try discriminate; cbn [atom_val].
    - apply Z.eqb_eq in Hhi. subst k. reflexivity.
    - rewrite !andb_true_iff in Hhi. destruct Hhi as [[Ha Hm] Hw].
      apply Z.eqb_eq in Ha. apply Z.leb_le in Hm, Hw. subst a.
      pose proof (rs_range r) as Hr. unfold wd in Hw.
      assert (Hp2 : 0 < 2 ^ n) by (apply Z.pow_pos_nonneg; lia).
      assert (2 ^ Z.of_nat (wdn r) <= 2 ^ (n + m)) by (apply Z.pow_le_mono_r; lia).
      apply Z.mod_small. split; [apply Z.div_pos; lia |].
      apply Z.div_lt_upper_bound; [lia |]. rewrite <- Z.pow_add_r by lia. lia. }
  assert (Hev : aval (r, ex) = rs r) by (destruct ex; cbn in Hwh; try discriminate; reflexivity).
  assert (Hts : tsum wdn rs ts = tsum wdn rs (ft f)).
  { unfold ts. rewrite tsum_nonzero, tsum_tsort, tsum_app. cbn [tsum]. rewrite Hhv, Hev.
    cbn [atom_val]. rewrite Z.pow_0_r, Z.div_1_r.
    pose proof (Z.div_mod (rs r) (2 ^ n) ltac:(apply Z.pow_nonzero; lia)) as Hd.
    set (q := rs r / 2 ^ n) in *. set (m := rs r mod 2 ^ n) in *. rewrite Hd. ring. }
  assert (H1 : forall L' H', plain_sum ar ts (flo f) (fhi f) = Some (L', H') ->
                             L' <= tsum wdn rs (ft f) + e <= H')
    by (intros L' H' Hs; rewrite <- Hts; apply (plain_sum_sound wdn rs ar ts (flo f) (fhi f));
        [intros x c' a b _ Hx; exact (Har x a b Hx) | exact Hs | exact He]).
  destruct (plain_sum ar (ft f) (flo f) (fhi f)) as [[a b] |];
    destruct (plain_sum ar ts (flo f) (fhi f)) as [[a' b'] |]; cbn [inter] in Hp; try discriminate;
    injection Hp as <- <-.
  - pose proof (H0 a b eq_refl). pose proof (H1 a' b' eq_refl). lia.
  - exact (H0 _ _ eq_refl).
  - exact (H1 _ _ eq_refl).
Qed.

Lemma fact_of : forall r x, get st r = Some x -> fact_ok F t rs r x.
Proof. intros. apply (proj1 Hst); assumption. Qed.

Lemma fact_range_sound : forall r lo hi, fact_range F r (get st r) = Some (lo, hi) -> lo <= rs r <= hi.
Proof.
  intros r lo hi H. unfold fact_range in H.
  destruct (get st r) as [x |] eqn:Eg.
  - pose proof (fact_of r x Eg) as Hx. destruct x; cbn in Hx; try discriminate.
    destruct vr as [[a b] |].
    + inversion H; subst; apply (proj2 (proj2 (proj2 Hx))); reflexivity.
    + cbn in H. destruct (Nat.eqb (List.length (ft f)) 0 && (wd F r <=? fm f) && (0 <=? flo f) && (fhi f <? 2 ^ wd F r)) eqn:Hb; [| discriminate].
      injection H as <- <-. rewrite !andb_true_iff in Hb.
      destruct Hb as [[[Hnil Hwidth] Hlo] Hhi].
      destruct (ft f) eqn:Et; [| discriminate].
      apply Z.leb_le in Hwidth, Hlo. apply Z.ltb_lt in Hhi.
      destruct (proj1 (proj2 Hx)) as [e [He Hm]].
      rewrite Et in Hm. cbn [tsum] in Hm.
      pose proof (rs_range r) as Hr. pose proof (wd_range r) as Hw.
      assert (rs r = e).
      { apply (eqm_small_eq (fm f) (wd F r));
          [unfold wd in *; lia | exact Hr | lia | exact Hm]. }
      lia.
  - destruct (is_fixed F r) eqn:Hf; [| discriminate].
    rewrite (proj2 Hrs r Hf). apply (ok_bounds F E HE r lo hi Hf H).
Qed.

Lemma cons_range_sound : forall ar f k L H e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    cons_ok F rs k -> cons_range ar f k = Some (L, H) -> flo f <= e <= fhi f ->
    L <= tsum wdn rs (ft f) + e <= H.
Proof.
  intros ar f [[ts clo] chi] L H e Har Hk Hc He. cbn [cons_ok] in Hk. unfold cons_range in Hc.
  pose proof (tsum_filter wdn rs (fun t0 => tmem ts (fst t0)) (ft f)) as Hsplit.
  pose proof (tsum_filter wdn rs (fun t0 => tmem (ft f) (fst t0)) ts) as Hsplit'.
  destruct (filter (fun t0 => tmem ts (fst t0)) (ft f)) as [| [x1 c1] pick'] eqn:Ep; [discriminate |].
  destruct (filter (fun t0 => tmem (ft f) (fst t0)) ts) as [| [x0 c0] core'] eqn:Ec; [discriminate |].
  destruct (Z.eqb c0 0 || negb (Z.eqb (Z.rem c1 c0) 0)) eqn:Eq; [discriminate |].
  destruct (scaled (Z.quot c1 c0) ((x1, c1) :: pick') ((x0, c0) :: core')) eqn:Esc; [| cbn in Hc; discriminate].
  cbn [negb] in Hc.
  destruct (plain_sum ar (filter (fun t0 => negb (tmem ts (fst t0))) (ft f)) (flo f) (fhi f))
    as [[a b] |] eqn:Er; [| discriminate].
  destruct (plain_sum ar (filter (fun t0 => negb (tmem (ft f) (fst t0))) ts) 0 0)
    as [[u v] |] eqn:Eu; [| discriminate].
  inversion Hc; subst L H. clear Hc.
  pose proof (scaled_sum wdn rs _ _ _ Esc) as Hsc.
  pose proof (plain_sum_sound wdn rs ar _ _ _ _ _ e (fun x c a0 b0 _ Ha => Har x a0 b0 Ha) Er He) as Hr.
  pose proof (plain_sum_sound wdn rs ar _ _ _ _ _ 0 (fun x c a0 b0 _ Ha => Har x a0 b0 Ha) Eu ltac:(lia)) as Hx.
  (* the constraint's terms in f: sum(ts) less the others *)
  assert (Hcore : clo - v <= tsum wdn rs ((x0, c0) :: core') <= chi - u) by lia.
  pose proof (mul_between (clo - v) (chi - u) (tsum wdn rs ((x0, c0) :: core')) (Z.quot c1 c0) Hcore) as Hm.
  rewrite (Z.mul_comm (tsum wdn rs ((x0, c0) :: core'))) in Hm. lia.
Qed.


Lemma projection_at_sound : forall ar f k c1 c0 L H e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    cons_ok F rs k -> projection_at F ar f k c1 c0 = Some (L, H) -> flo f <= e <= fhi f ->
    L <= tsum wdn rs (ft f) + e <= H.
Proof using F E rs Hrs.
  intros ar f [[ts clo] chi] c1 c0 L H e Har Hk Hc He. cbn [cons_ok] in Hk. unfold projection_at in Hc.
  destruct (Z.eqb c0 0) eqn:Eq; [discriminate |]. apply Z.eqb_neq in Eq.
  set (n := Z.abs c0) in *. set (q := c1 * Z.sgn c0) in *.
  assert (Hn : 0 < n) by (unfold n; lia).
  destruct (plain_range F ar {| fm:=64; ft:=filter (fun t0 => negb (Z.eqb (snd t0) 0)) (tsort (tscale n (ft f) ++ tscale (-q) ts)); flo:=n * flo f; fhi:=n * fhi f |})
    as [[a b] |] eqn:Er; [| discriminate].
  inversion Hc; subst L H. clear Hc.
  pose proof (plain_range_sound ar _ a b (n * e) Har Er ltac:(cbn; nia)) as Hr.
  cbn [ft] in Hr.
  rewrite tsum_nonzero, tsum_tsort, tsum_app, !tsum_tscale in Hr.
  pose proof (mul_between clo chi (tsum wdn rs ts) q Hk) as Hm.
  assert (Hlow : a + Z.min (clo * q) (chi * q) <= (tsum wdn rs (ft f) + e) * n) by nia.
  assert (Hhigh : (tsum wdn rs (ft f) + e) * n <= b + Z.max (clo * q) (chi * q)) by nia.
  assert (Hneg : (- (tsum wdn rs (ft f) + e)) * n <= - (a + Z.min (clo * q) (chi * q))) by nia.
  apply (Z.div_le_mono _ _ n Hn) in Hneg, Hhigh.
  rewrite Z.div_mul in Hneg, Hhigh by lia. lia.
Qed.

Definition inter_sound (o : option range) (v : Z) : Prop :=
  forall a b, o = Some (a, b) -> a <= v <= b.

Lemma inter_ok : forall x y v, inter_sound x v -> inter_sound y v -> inter_sound (inter x y) v.
Proof.
  unfold inter_sound, inter. intros [[a b] |] [[c d] |] v Hx Hy a' b' H; try discriminate; inversion H; subst.
  - specialize (Hx a b eq_refl). specialize (Hy c d eq_refl). lia.
  - exact (Hx a' b' eq_refl).
  - exact (Hy a' b' eq_refl).
Qed.

Lemma projected_cons_range_sound : forall ar f k L H e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    cons_ok F rs k -> projected_cons_range F ar f k = Some (L, H) -> flo f <= e <= fhi f ->
    L <= tsum wdn rs (ft f) + e <= H.
Proof using F E rs Hrs.
  intros ar f [[ts clo] chi] L H e Har Hk Hc He.
  unfold projected_cons_range in Hc.
  assert (Hfold : forall (ps : list (Z * Z)) acc,
    inter_sound acc (tsum wdn rs (ft f) + e) ->
    inter_sound (fold_left (fun acc p => inter acc (projection_at F ar f (ts,clo,chi) (fst p) (snd p))) ps acc)
      (tsum wdn rs (ft f) + e)).
  { induction ps as [| [c1 c0] ps IH]; intros acc Hacc; [exact Hacc |].
    apply IH. apply inter_ok; [exact Hacc |].
    intros lo hi Hr. exact (projection_at_sound ar f (ts,clo,chi) c1 c0 lo hi e Har Hk Hr He). }
  exact (Hfold _ None ltac:(intros lo hi Hnone; discriminate) L H Hc).
Qed.

(* Cached lookup returns exactly the same range as the original function. *)
Lemma cached_range_same : forall ar xs x, cached_range ar xs x = ar x.
Proof.
  intros ar xs x. unfold cached_range.
  destruct (find (fun p => atom_eqb x (fst p)) (map (fun y => (y, ar y)) xs))
    as [[y v] |] eqn:Hfind; [| reflexivity].
  apply find_some in Hfind. destruct Hfind as [Hin Heq].
  apply in_map_iff in Hin. destruct Hin as [z [Hz Hin]].
  inversion Hz; subst. cbn [fst] in Heq. apply atom_eqb_eq in Heq. subst. reflexivity.
Qed.

Lemma form_range_in_sound : forall cs ar f L H e,
    Forall (cons_ok F rs) cs ->
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    form_range_in F cs ar f = Some (L, H) -> flo f <= e <= fhi f ->
    L <= tsum wdn rs (ft f) + e <= H.
Proof.
  intros cs ar f L H e Hcs Har Hf He. unfold form_range_in in Hf.
  set (xs := dependencies_in cs (ft f)) in Hf.
  set (ar' := cached_range ar xs) in Hf.
  assert (Har' : forall x a b, ar' x = Some (a, b) -> a <= aval x <= b).
  { intros x a b Hx. unfold ar' in Hx. rewrite cached_range_same in Hx. exact (Har x a b Hx). }

  assert (Hinv : forall ks acc, Forall (cons_ok F rs) ks -> inter_sound acc (tsum wdn rs (ft f) + e) ->
            inter_sound (fold_left (fun acc k => inter acc (cons_range ar' f k)) ks acc) (tsum wdn rs (ft f) + e)).
  { intros ks. induction ks as [| k ks IH]; intros acc Hks Hacc; cbn [fold_left]; [exact Hacc |].
    inversion Hks; subst. apply IH; [assumption |]. apply inter_ok; [exact Hacc |].
    intros a b Hk. apply (cons_range_sound ar' f k); assumption. }
  apply (Hinv cs (plain_range F ar' f) Hcs); [| exact Hf].
  intros a b Hp. apply (plain_range_sound ar' f); [intros x a' b' Hx; exact (Har' x a' b' Hx) | exact Hp | exact He].
Qed.

Lemma form_range_with_sound : forall ar f L H e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    form_range_with F st ar f = Some (L, H) -> flo f <= e <= fhi f ->
    L <= tsum wdn rs (ft f) + e <= H.
Proof. intros ar f L H e Har Hf He. exact (form_range_in_sound (acons st) ar f L H e (proj2 Hst) Har Hf He). Qed.

(* Each constraint of a register's bucket is a constraint of the list. *)
Lemma cons_bucket_add : forall k regs ix r k',
    In k' (cons_bucket (fold_left (fun ix r => PM.add r (k :: cons_bucket ix r) ix) regs ix) r) ->
    k' = k \/ In k' (cons_bucket ix r).
Proof.
  intros k regs. induction regs as [| r0 regs IH]; intros ix r k' H; cbn [fold_left] in H; [right; exact H |].
  destruct (IH _ r k' H) as [-> | Hin]; [left; reflexivity |].
  unfold cons_bucket at 1 in Hin. destruct (Pos.eq_dec r r0) as [-> | Hne].
  - rewrite PM.gss in Hin. destruct Hin as [-> | Hin]; [left; reflexivity | right; exact Hin].
  - rewrite PM.gso in Hin by exact Hne. right. exact Hin.
Qed.

Lemma cons_bucket_in : forall cs r k, In k (cons_bucket (cons_index cs) r) -> In k cs.
Proof.
  intros cs r. induction cs as [| c cs IH]; intros k H.
  - unfold cons_index, cons_bucket in H. cbn [fold_right] in H. rewrite PM.gempty in H. destruct H.
  - unfold cons_index in H. cbn [fold_right] in H. fold (cons_index cs) in H.
    destruct (cons_bucket_add c (cons_regs c) (cons_index cs) r k H) as [-> | Hin];
      [left; reflexivity | right; exact (IH k Hin)].
Qed.

(* An index of the state's constraints: each bucket holds constraints of st. *)
Definition index_ok (ix : PM.t (list constraint)) : Prop :=
  forall r k, In k (cons_bucket ix r) -> In k (acons st).

Lemma cons_index_ok : index_ok (cons_index (acons st)).
Proof. intros r k H. exact (cons_bucket_in (acons st) r k H). Qed.

Lemma cons_near_ok : forall ix ts, index_ok ix -> Forall (cons_ok F rs) (cons_near ix ts).
Proof.
  intros ix ts Hix. apply Forall_forall. intros k Hk. unfold cons_near in Hk.
  apply in_flat_map in Hk. destruct Hk as [term [_ Hk]].
  exact (proj1 (Forall_forall _ _) (proj2 Hst) k (Hix _ k Hk)).
Qed.

Lemma form_range_ix_sound : forall ix ar f L H e, index_ok ix ->
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    form_range_ix F ix ar f = Some (L, H) -> flo f <= e <= fhi f ->
    L <= tsum wdn rs (ft f) + e <= H.
Proof.
  intros ix ar f L H e Hix Har Hf He.
  exact (form_range_in_sound (cons_near ix (ft f)) ar f L H e (cons_near_ok ix (ft f) Hix) Har Hf He).
Qed.

(* The range functions at each fuel are sound. *)
Definition rfuns_ok (fs : rfuns) : Prop :=
  (forall r a b, f_rr fs r = Some (a, b) -> a <= rs r <= b) /\
  (forall r a b, f_sr fs r = Some (a, b) -> a <= sext (wdn r) (rs r) <= b).

Lemma sext_of_small : forall r a b,
    0 <= a -> b < 2 ^ (wd F r - 1) -> a <= rs r <= b -> a <= sext (wdn r) (rs r) <= b.
Proof.
  intros r a b Ha Hb H. pose proof (wd_range r). rewrite sext_small; [lia | lia |].
  unfold wd in Hb. lia.
Qed.

Lemma atom_range_with_sound : forall fs x a b,
    rfuns_ok fs -> atom_range_with F fs x = Some (a, b) -> a <= aval x <= b.
Proof.
  intros fs [r e] a b [Hrr Hsr] H. pose proof (rs_range r) as Hr. pose proof (wd_range r) as Hw.
  destruct e; cbn [atom_range_with atom_val] in *.
  - exact (Hrr r a b H).
  - unfold wd in H. destruct (f_rr fs r) as [[lo hi] |] eqn:E1; inversion H; subst; [| lia].
    pose proof (Hrr r lo hi E1). lia.
  - destruct (f_sr fs r) as [[lo hi] |] eqn:E1; inversion H; subst; [exact (Hsr r _ _ E1) |].
    unfold sext, wd. pose proof (trunc_range (wdn r) (rs r)) as Ht.
    assert (H2 : 2 ^ Z.of_nat (wdn r) = 2 * 2 ^ (Z.of_nat (wdn r) - 1))
      by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
    destruct (Z.ltb_spec (trunc (wdn r) (rs r)) (2 ^ (Z.of_nat (wdn r) - 1))); lia.
  - destruct (k <? 0) eqn:Ek; [discriminate |]. apply Z.ltb_ge in Ek.
    assert (Hp : 0 < 2 ^ k) by (apply pow_nonneg_pos; lia).
    unfold shr_range in H. destruct (f_rr fs r) as [[lo hi] |] eqn:E1; inversion H; subst.
    + apply div_between; [lia | exact (Hrr r lo hi E1)].
    + split; [apply Z.div_pos; lia | apply Z.div_le_mono; [lia |]].
      assert (2 ^ Z.of_nat (wdn r) <= 2 ^ 64) by (apply Z.pow_le_mono_r; lia). lia.
  - destruct ((a0 <? 0) || (n <? 0)) eqn:Ean; [discriminate |].
    apply orb_false_iff in Ean. destruct Ean as [Ea En]. apply Z.ltb_ge in Ea. apply Z.ltb_ge in En.
    assert (Hp : 0 < 2 ^ a0) by (apply pow_nonneg_pos; lia).
    assert (Hn : 0 < 2 ^ n) by (apply pow_nonneg_pos; lia).
    assert (Hmod : forall y, 0 <= y mod 2 ^ n < 2 ^ n) by (intros; apply Z.mod_pos_bound; lia).
    destruct (f_rr fs r) as [[lo hi] |] eqn:E1.
    + destruct ((0 <=? lo) && (hi / 2 ^ a0 <? 2 ^ n)) eqn:Ec.
      * inversion H; subst. apply andb_true_iff in Ec. destruct Ec as [E2 E3].
        apply Z.leb_le in E2. apply Z.ltb_lt in E3.
        pose proof (Hrr r lo hi E1) as Hv. pose proof (div_between lo hi (rs r) (2 ^ a0) Hp Hv).
        rewrite Z.mod_small by (split; [apply Z.div_pos; lia | lia]). lia.
      * destruct (Z.eqb lo hi) eqn:Elh.
        -- apply Z.eqb_eq in Elh. subst hi. inversion H; subst.
           pose proof (Hrr r lo lo E1) as Hv. assert (rs r = lo) as -> by lia. lia.
        -- inversion H; subst. pose proof (Hmod (rs r / 2 ^ a0)). lia.
    + inversion H; subst. pose proof (Hmod (rs r / 2 ^ a0)). lia.
Qed.

Lemma fact_range_sext : forall r a b,
    match fact_range F r (get st r) with
    | Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd F r - 1)) then Some (lo, hi) else None
    | None => None
    end = Some (a, b) -> a <= sext (wdn r) (rs r) <= b.
Proof.
  intros r a b H. destruct (fact_range F r (get st r)) as [[lo hi] |] eqn:Ef; [| discriminate].
  destruct ((0 <=? lo) && (hi <? 2 ^ (wd F r - 1))) eqn:Ec; [| discriminate].
  inversion H; subst. apply andb_true_iff in Ec. destruct Ec as [E1 E2].
  apply Z.leb_le in E1. apply Z.ltb_lt in E2. apply sext_of_small; [lia | lia |].
  exact (fact_range_sound r a b Ef).
Qed.

(* A value below 2^w congruent to X modulo 2^m, for m >= w, is X's residue,
   which reduce bounds. *)
Lemma reduce_sound : forall w m v X x y c d,
    0 <= w <= m -> 0 <= v < 2 ^ w -> eqm m v X -> x <= X <= y -> reduce w (x, y) = Some (c, d) ->
    c <= v <= d.
Proof.
  intros w m v X x y c d Hw Hv Hm HX H. unfold reduce in H.
  destruct (Z.eqb_spec (x / 2 ^ w) (y / 2 ^ w)) as [Hq |]; [| discriminate]. injection H as <- <-.
  assert (Hp : 0 < 2 ^ w) by (apply Z.pow_pos_nonneg; lia).
  assert (Hvm : v = X mod 2 ^ w).
  { apply (eqm_weaken m w v X) in Hm; [| lia]. unfold eqm in Hm. rewrite Z.mod_small in Hm by lia. exact Hm. }
  assert (HqX : X / 2 ^ w = x / 2 ^ w).
  { pose proof (Z.div_le_mono x X (2 ^ w) Hp (proj1 HX)). pose proof (Z.div_le_mono X y (2 ^ w) Hp (proj2 HX)). lia. }
  rewrite Hvm. rewrite !Z.mod_eq by lia. rewrite HqX, <- Hq.
  set (q := x / 2 ^ w). lia.
Qed.

Lemma cache_rf_same : forall fs regs r,
 f_rr (cache_rf fs regs) r = f_rr fs r /\ f_sr (cache_rf fs regs) r = f_sr fs r.
Proof.
 intros fs regs r. unfold cache_rf. cbn [f_rr f_sr].
 destruct (PM.find r (pair_table (map (fun u => (u,(f_rr fs u,f_sr fs u))) regs))) as [[rr sr]|] eqn:He;
   [| split; reflexivity].
 apply (pair_table_find (fun u => (f_rr fs u,f_sr fs u))) in He. injection He as -> ->. split; reflexivity.
Qed.

Lemma cache_rf_ok : forall fs regs, rfuns_ok fs -> rfuns_ok (cache_rf fs regs).
Proof.
 intros fs regs [Hrr Hsr]. split; intros r a b H.
 - rewrite (proj1 (cache_rf_same fs regs r)) in H. exact (Hrr r a b H).
 - rewrite (proj2 (cache_rf_same fs regs r)) in H. exact (Hsr r a b H).
Qed.

Lemma range_pair_exact : forall ix prev r,
 range_pair F ix st prev r=(f_rr (range_layer F ix st prev) r,f_sr (range_layer F ix st prev) r).
Proof.
 intros ix prev r. unfold range_pair,range_layer,range_unsigned_signed. cbn [f_rr f_sr].
 destruct (get st r) as [[f al vr| | |]|];try reflexivity.
 destruct (fm f >=? wd F r);destruct vr;reflexivity.
Qed.
Lemma cache_range_layer_same : forall regs ix prev r,
 f_rr(cache_range_layer F ix st prev(layer_pairs F regs ix st prev))r=f_rr(range_layer F ix st prev)r /\
 f_sr(cache_range_layer F ix st prev(layer_pairs F regs ix st prev))r=f_sr(range_layer F ix st prev)r.
Proof.
 intros regs ix prev r. unfold cache_range_layer,layer_pairs. cbn [f_rr f_sr].
 destruct (PM.find r (pair_table (map(fun r=>(r,range_pair F ix st prev r)) regs))) as [[rr sr]|] eqn:H;[|split;reflexivity].
 apply (pair_table_find (range_pair F ix st prev)) in H. rewrite range_pair_exact in H. injection H as Hr Hs.
 unfold range_layer in Hr,Hs. cbn [f_rr f_sr] in Hr,Hs. split;assumption.
Qed.
Lemma range_base_sound : rfuns_ok(range_base F st).
Proof. split;intros r a b H;[exact(fact_range_sound r a b H)|exact(fact_range_sext r a b H)]. Qed.
Lemma range_layer_sound : forall ix prev,index_ok ix -> rfuns_ok prev -> rfuns_ok(range_layer F ix st prev).
Proof.
 intros ix prev Hix Hprev. unfold range_layer.
    set (fr := form_range_ix F ix (atom_range_with F prev)).
    assert (Hfr : forall f a b e, fr f = Some (a, b) -> flo f <= e <= fhi f -> a <= tsum wdn rs (ft f) + e <= b).
    { intros f a b e H He. apply (form_range_ix_sound ix (atom_range_with F prev) f); [exact Hix | | exact H | exact He].
      intros x a' b' Hx. exact (atom_range_with_sound prev x a' b' Hprev Hx). }
    assert (Hrr : forall r a b,
               match get st r with
               | Some (FForm f _ None) =>
                   if fm f >=? wd F r then
                     match fr f with Some g => reduce (wd F r) g | None => None end
                   else None
               | x => fact_range F r x
               end = Some (a, b) -> a <= rs r <= b).
    { intros r a b H. destruct (get st r) as [x |] eqn:Eg; [| exact (fact_range_sound r a b (eq_trans (f_equal _ Eg) H))].
      destruct x; try exact (fact_range_sound r a b (eq_trans (f_equal _ Eg) H)).
      destruct vr as [vr |]; [exact (fact_range_sound r a b (eq_trans (f_equal _ Eg) H)) |].
      destruct (fm f >=? wd F r) eqn:Em; [| discriminate].
      destruct (fr f) as [[g1 g2] |] eqn:Ef; [| discriminate].
      pose proof (fact_of r _ Eg) as [_ [[e [He Hsat]] _]]. cbn [fact_ok fact_okv] in *.
      pose proof (Hfr f g1 g2 e Ef He) as Hb.
      apply Z.geb_le in Em. pose proof (rs_range r). pose proof (wd_range r). unfold wd in *.
      exact (reduce_sound (Z.of_nat (wdn r)) (fm f) (rs r) _ g1 g2 a b ltac:(lia) ltac:(lia) Hsat Hb H). }
    split; [exact Hrr |].
    intros r a b H. cbn [f_sr] in H.
    set (rr := fun r0 => match get st r0 with
                          | Some (FForm f _ None) =>
                              if fm f >=? wd F r0 then
                                match fr f with Some g => reduce (wd F r0) g | None => None end
                              else None
                          | x => fact_range F r0 x
                          end) in *.
    assert (Hvia : match rr r with
                   | Some (lo, hi) => if (0 <=? lo) && (hi <? 2 ^ (wd F r - 1)) then Some (lo, hi) else None
                   | None => None
                   end = Some (a, b) -> a <= sext (wdn r) (rs r) <= b).
    { intros H'. destruct (rr r) as [[lo hi] |] eqn:Er; [| discriminate].
      destruct ((0 <=? lo) && (hi <? 2 ^ (wd F r - 1))) eqn:Ec; [| discriminate].
      inversion H'; subst. apply andb_true_iff in Ec. destruct Ec as [E1 E2].
      apply Z.leb_le in E1. apply Z.ltb_lt in E2. apply sext_of_small; [lia | lia |].
      exact (Hrr r a b Er). }
    unfold rr in Hvia.
    destruct (get st r) as [x |] eqn:Eg; [| exact (Hvia H)].
    destruct x; try exact (Hvia H).
    destruct (fm f >=? wd F r) eqn:Em; [| exact (Hvia H)].
    destruct (fr f) as [[g1 g2] |] eqn:Ef; [| exact (Hvia H)].
    destruct (sext_ok (wd F r) (g1, g2)) eqn:Es; [| exact (Hvia H)].
    inversion H; subst.
    pose proof (fact_of r _ Eg) as [_ [[e [He Hsat]] _]]. cbn [fact_ok fact_okv] in *.
    pose proof (Hfr f a b e Ef He) as Hb.
    unfold sext_ok in Es. cbn [fst snd] in Es. apply andb_true_iff in Es. destruct Es as [E1 E2].
    apply Z.leb_le in E1. apply Z.ltb_lt in E2. apply Z.geb_le in Em.
    pose proof (rs_range r). pose proof (wd_range r). unfold wd in *.
    rewrite (eqm_sext_eq (fm f) (wdn r) (rs r) (tsum wdn rs (ft f) + e)); [lia | lia | lia | lia | lia | exact Hsat].
Qed.


Lemma cache_range_layer_sound : forall regs ix prev,index_ok ix -> rfuns_ok prev ->
 rfuns_ok(cache_range_layer F ix st prev(layer_pairs F regs ix st prev)).
Proof.
 intros regs ix prev Hix Hp. destruct(range_layer_sound ix prev Hix Hp)as[Hr Hs]. split;intros r a b H.
 - rewrite(proj1(cache_range_layer_same regs ix prev r))in H. exact(Hr r a b H).
 - rewrite(proj2(cache_range_layer_same regs ix prev r))in H. exact(Hs r a b H).
Qed.
Lemma rf_plain_sound : forall ix n,index_ok ix -> rfuns_ok(rf_plain F ix n st).
Proof. intros ix n Hix. induction n;cbn[rf_plain];[apply range_base_sound|apply (range_layer_sound ix _ Hix);exact IHn]. Qed.
Lemma rf_stable_layers_sound : forall regs ix n prev pairs,index_ok ix -> rfuns_ok prev ->
 rfuns_ok(rf_stable_layers F regs ix st n prev pairs).
Proof.
 intros regs ix n. induction n as [|n IH];intros prev pairs Hix Hp;cbn[rf_stable_layers];[exact Hp|].
 destruct(range_pairs_eqb (layer_pairs F regs ix st prev) pairs);[exact Hp|].
 apply IH;[exact Hix|apply cache_range_layer_sound;assumption].
Qed.
Lemma rf_cached_layers_sound : forall regs ix n,index_ok ix -> rfuns_ok(rf_cached_layers F regs ix n st).
Proof.
 intros regs ix n Hix. unfold rf_cached_layers. destruct n;[apply cache_rf_ok,range_base_sound|].
 apply rf_stable_layers_sound;[exact Hix|]. apply cache_range_layer_sound;[exact Hix|]. apply cache_rf_ok,range_base_sound.
Qed.
Lemma rf_on_sound : forall regs ix n,index_ok ix -> rfuns_ok(rf_on F regs ix n st).
Proof.
 intros regs ix n Hix. destruct n;[apply range_base_sound|].
 apply (range_layer_sound ix _ Hix),rf_cached_layers_sound,Hix.
Qed.
Lemma rf_with_cache_sound : forall ix memo n,index_ok ix -> rfuns_ok(rf_with_cache F ix memo n st).
Proof.
 intros ix memo n Hix. unfold rf_with_cache. destruct memo;[apply rf_on_sound,Hix|apply rf_plain_sound,Hix].
Qed.

Lemma rf_sound : forall n, rfuns_ok (rf F n st).
Proof. intros n. exact (rf_with_cache_sound _ false n cons_index_ok). Qed.

(* A fold of constraint ranges over valid constraints stays sound. *)
Lemma cons_fold_sound : forall ar f ks acc e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    Forall (cons_ok F rs) ks -> flo f <= e <= fhi f -> inter_sound acc (tsum wdn rs (ft f) + e) ->
    inter_sound (fold_left (fun acc k => inter acc (cons_range ar f k)) ks acc) (tsum wdn rs (ft f) + e).
Proof.
  intros ar f ks. induction ks as [| k ks IH]; intros acc e Har Hks He Hacc; cbn [fold_left]; [exact Hacc |].
  inversion Hks; subst. apply IH; [exact Har | assumption | exact He |]. apply inter_ok; [exact Hacc |].
  intros a b Hk. apply (cons_range_sound ar f k); assumption.
Qed.

Lemma projected_cons_fold_sound : forall ar f ks acc e,
    (forall x a b, ar x = Some (a, b) -> a <= aval x <= b) ->
    Forall (cons_ok F rs) ks -> flo f <= e <= fhi f -> inter_sound acc (tsum wdn rs (ft f) + e) ->
    inter_sound (fold_left (fun acc k => inter acc (projected_cons_range F ar f k)) ks acc) (tsum wdn rs (ft f) + e).
Proof.
  intros ar f ks. induction ks as [| k ks IH]; intros acc e Har Hks He Hacc; cbn [fold_left]; [exact Hacc |].
  inversion Hks; subst. apply IH; [exact Har | assumption | exact He |]. apply inter_ok; [exact Hacc |].
  intros a b Hk. apply (projected_cons_range_sound ar f k); assumption.
Qed.

(* A value that reads as at least 0 when sign-extended is its own
   sign extension. *)
Lemma sext_nonneg : forall n v, (1 <= n)%nat -> 0 <= v < 2 ^ Z.of_nat n -> 0 <= sext n v -> sext n v = v.
Proof.
  intros n v Hn Hv Hs. unfold sext in *. unfold trunc in *. rewrite Z.mod_small in * by lia.
  destruct (Z.ltb_spec v (2 ^ (Z.of_nat n - 1))); [reflexivity | lia].
Qed.

Lemma cons_reg_range_sound : forall ks r, (forall k, In k ks -> In k (acons st)) ->
    forall a b, cons_reg_range F ks st r = Some (a, b) -> a <= rs r <= b.
Proof.
  intros ks r Hks a b H. unfold cons_reg_range in H.
  assert (Hok : Forall (cons_ok F rs) ks).
  { apply Forall_forall. intros k Hk. exact (proj1 (Forall_forall _ _) (proj2 Hst) k (Hks k Hk)). }
  set (ar := atom_range_with F (range_base F st)) in *.
  assert (Har : forall x a b, ar x = Some (a, b) -> a <= aval x <= b)
    by (intros x a' b' Hx; exact (atom_range_with_sound _ x a' b' range_base_sound Hx)).
  assert (Hvia : forall e, inter_sound (fold_left (fun acc k => inter acc (cons_range ar {| fm := 64; ft := [((r, e), 1)]; flo := 0; fhi := 0 |} k))
                                                  ks None) (aval (r, e))).
  { intros e. pose proof (cons_fold_sound ar {| fm := 64; ft := [((r, e), 1)]; flo := 0; fhi := 0 |} ks None 0 Har Hok
                            ltac:(cbn; lia) ltac:(intros a' b' Hn; discriminate)) as Hf.
    cbn [ft tsum] in Hf. replace (1 * aval (r, e) + 0 + 0) with (aval (r, e)) in Hf by ring. exact Hf. }
  pose proof (rs_range r) as Hr. pose proof (wd_range r) as Hw.
  revert H. apply inter_ok; [apply inter_ok; [exact (Hvia EX) | exact (Hvia EZ)] |].
  intros lo hi Hes. pose proof (Hvia ES) as Hs.
  destruct (fold_left _ ks None) as [[l h] |] eqn:E1 in Hes; [| discriminate].
  destruct (0 <=? l) eqn:El; [| discriminate]. injection Hes as <- <-. apply Z.leb_le in El.
  specialize (Hs l h). cbn [atom_val] in Hs. rewrite E1 in Hs. specialize (Hs eq_refl).
  rewrite (sext_nonneg (wdn r) (rs r)) in Hs; [lia | lia | lia | lia].
Qed.

Lemma tighten_sound : forall ix fs, index_ok ix -> rfuns_ok fs -> rfuns_ok (tighten F ix st fs).
Proof.
  intros ix fs Hix [Hrr Hsr]. unfold tighten. split; cbn [f_rr f_sr].
  - intros r a b H.
    exact (inter_ok _ _ (rs r) (Hrr r) (cons_reg_range_sound _ r (Hix r)) a b H).
  - intros r a b H. revert H. apply inter_ok; [exact (Hsr r) |].
    intros lo hi H.
    destruct (inter (f_rr fs r) (cons_reg_range F (cons_bucket ix r) st r))
      as [[l h] |] eqn:E1; [| discriminate].
    destruct ((0 <=? l) && (h <? 2 ^ (wd F r - 1))) eqn:Ec; [| discriminate].
    injection H as <- <-. apply andb_true_iff in Ec. destruct Ec as [E2 E3].
    apply Z.leb_le in E2. apply Z.ltb_lt in E3. apply sext_of_small; [lia | lia |].
    exact (inter_ok _ _ (rs r) (Hrr r) (cons_reg_range_sound _ r (Hix r)) l h E1).
Qed.

Lemma rft_on_sound : forall regs, rfuns_ok (rft_on F (cons_index (acons st)) st regs).
Proof. intros regs. unfold rft_on. apply (tighten_sound _ _ cons_index_ok). apply (rf_on_sound _ _ DEPTH cons_index_ok). Qed.

Lemma reg_range_sound : forall r a b, reg_range F st r = Some (a, b) -> a <= rs r <= b.
Proof. intros r a b H. unfold reg_range in H. exact (proj1 (rft_on_sound [r]) r a b H). Qed.

Lemma sreg_range_sound : forall r a b, sreg_range F st r = Some (a, b) -> a <= sext (wdn r) (rs r) <= b.
Proof. intros r a b H. unfold sreg_range in H. exact (proj2 (rft_on_sound [r]) r a b H). Qed.

Lemma atom_range_sound : forall x a b, atom_range F st x = Some (a, b) -> a <= aval x <= b.
Proof. intros x a b H. unfold atom_range in H. exact (atom_range_with_sound _ x a b (rft_on_sound _) H). Qed.

Lemma form_range_sound : forall f a b e,
    form_range F st f = Some (a, b) -> flo f <= e <= fhi f -> a <= tsum wdn rs (ft f) + e <= b.
Proof.
  intros f a b e H He. unfold form_range in H.
  apply (form_range_ix_sound (cons_index (acons st))
           (atom_range_with F (rft_on F (cons_index (acons st)) st (range_reads (cons_index (acons st)) (ft f)))) f);
    [exact cons_index_ok | | exact H | exact He].
  intros x a' b' Hx. exact (atom_range_with_sound _ x a' b' (rft_on_sound _) Hx).
Qed.

Lemma form_range_near_sound : forall f a b e,
    form_range_near F st f = Some (a, b) -> flo f <= e <= fhi f -> a <= tsum wdn rs (ft f) + e <= b.
Proof.
  intros f a b e H He. unfold form_range_near in H.
  apply (form_range_ix_sound (cons_index (acons st)) (atom_range_with F (rf_with_cache F (cons_index (acons st)) false 1 st)) f);
    [exact cons_index_ok | | exact H | exact He].
  intros x a' b' Hx. exact (atom_range_with_sound _ x a' b' (rf_with_cache_sound _ false 1 cons_index_ok) Hx).
Qed.

Lemma bounded_form_range_sound : forall f lower upper a b e,
 bounded_form_range F st f lower upper = Some (a,b) -> flo f <= e <= fhi f ->
 a <= tsum wdn rs (ft f) + e <= b.
Proof.
 intros f lower upper a b e H He. unfold bounded_form_range in H.
 destruct (form_range_near F st f) as [[p q]|] eqn:Er.
 - destruct ((lower <=? p) && (q <=? upper)).
   + injection H as <- <-. exact (form_range_near_sound f p q e Er He).
   + exact (form_range_sound f a b e H He).
 - exact (form_range_sound f a b e H He).
Qed.

Lemma natural_range_sound : forall x a b, natural_range F x = Some (a, b) -> a <= aval x <= b.
Proof.
  intros [r e] a b H. pose proof (rs_range r) as Hr.
  destruct e; cbn [natural_range atom_val] in *; try discriminate; injection H as <- <-; unfold wd.
  - lia.
  - lia.
  - pose proof (wd_range r) as Hw. unfold sext.
    assert (Hp : 2 ^ Z.of_nat (wdn r) = 2 * 2 ^ (Z.of_nat (wdn r) - 1)).
    { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
    pose proof (trunc_range (wdn r) (rs r)) as Ht.
    destruct (Z.ltb_spec (trunc (wdn r) (rs r)) (2 ^ (Z.of_nat (wdn r) - 1))); lia.
Qed.

Lemma form_range_nat_sound : forall f a b e,
    form_range_nat F st f = Some (a, b) -> flo f <= e <= fhi f -> a <= tsum wdn rs (ft f) + e <= b.
Proof.
  intros f a b e H He. unfold form_range_nat in H.
  set (ranges := rft_on F (cons_index (acons st)) st (range_reads (cons_index (acons st)) (ft f))) in H.
  apply (form_range_with_sound
           (fun x => match atom_range_with F ranges x with Some g => Some g | None => natural_range F x end) f);
    [| exact H | exact He].
  intros x a' b' Hx. destruct (atom_range_with F ranges x) as [[p q] |] eqn:Ex.
  - injection Hx as <- <-. exact (atom_range_with_sound _ x p q (rft_on_sound _) Ex).
  - exact (natural_range_sound x a' b' Hx).
Qed.

(* ------------------------------------------------------------------ *)
(* Alignment.                                                          *)

Lemma sext_aligned : forall r k, aligned k (rs r) -> aligned k (sext (wdn r) (rs r)).
Proof.
  intros r k H. pose proof (rs_range r) as Hr. pose proof (wd_range r) as Hw.
  unfold sext. rewrite trunc_small by lia.
  destruct (Z.ltb_spec (rs r) (2 ^ (Z.of_nat (wdn r) - 1))) as [Hl | Hl]; [exact H |].
  destruct (Z.le_gt_cases (Z.max 0 k) (Z.of_nat (wdn r))) as [Hk | Hk].
  - unfold aligned in *. apply Z.divide_sub_r; [exact H |]. exists (2 ^ (Z.of_nat (wdn r) - Z.max 0 k)).
    rewrite <- Z.pow_add_r by lia. f_equal. lia.
  - (* a multiple of 2^k below 2^w with k > w is zero, below 2^(w-1) *)
    exfalso. destruct H as [q Hq].
    assert (HK : 2 ^ Z.of_nat (wdn r) < 2 ^ Z.max 0 k) by (apply Z.pow_lt_mono_r; lia).
    assert (Hh : 0 < 2 ^ (Z.of_nat (wdn r) - 1)) by (apply pow_nonneg_pos; lia).
    set (K := 2 ^ Z.max 0 k) in *. set (W := 2 ^ Z.of_nat (wdn r)) in *.
    set (W1 := 2 ^ (Z.of_nat (wdn r) - 1)) in *.
    assert (Hq1 : 1 <= q) by nia.
    nia.
Qed.

Lemma atom_align_ok : forall (ra : reg -> Z) x,
    (forall r, aligned (ra r) (rs r)) ->
    aligned (match snd x with EX | EZ | ES => ra (fst x) | _ => 0 end) (aval x).
Proof.
  intros ra [r e] Hra. destruct e; cbn [snd fst atom_val]; try exact (Hra r).
  - apply sext_aligned. exact (Hra r).
  - apply aligned_nonpos. lia.
  - apply aligned_nonpos. lia.
Qed.

Lemma form_align_with_sound : forall (ra : reg -> Z) f,
    (forall r, aligned (ra r) (rs r)) -> flo f = fhi f -> 0 <= fm f ->
    aligned (form_align_with ra f) (tsum wdn rs (ft f) + flo f).
Proof.
  intros ra f Hra Hlo Hm. unfold form_align_with.
  assert (Hconst : aligned (const_align f) (flo f)).
  { unfold const_align. rewrite Hlo, Z.eqb_refl.
    destruct (Z.eqb (fhi f) 0) eqn:E0; [apply Z.eqb_eq in E0; rewrite E0; apply aligned_zero |].
    apply (aligned_mono (v2 (fhi f))); [lia |]. unfold aligned.
    rewrite Z.max_r by (apply v2_nonneg). apply v2_divides. }
  rewrite Hlo in Hconst |- *.
  (* the fold keeps the minimum; every term and the constant are multiples of it *)
  assert (Hg : forall ts k s, aligned k s ->
            (forall x c, In (x, c) ts -> aligned (v2 c + match snd x with EX | EZ | ES => ra (fst x) | _ => 0 end) (c * aval x)) ->
            aligned (fold_left (fun k0 '(x, c) => Z.min k0 (v2 c + match snd x with EX | EZ | ES => ra (fst x) | _ => 0 end)) ts k)
                    (tsum wdn rs ts + s)).
  { intros ts. induction ts as [| [x c] ts IH]; intros k s Hk Hts; cbn [fold_left tsum]; [rewrite Z.add_0_l; exact Hk |].
    replace (c * aval x + tsum wdn rs ts + s) with (tsum wdn rs ts + (c * aval x + s)) by ring.
    apply IH.
    - apply aligned_add; [eapply aligned_mono; [apply Z.le_min_r | apply Hts; left; reflexivity] |
                          eapply aligned_mono; [apply Z.le_min_l | exact Hk]].
    - intros x' c' Hi. apply Hts. right. exact Hi. }
  apply Hg; [exact Hconst |].
  intros x c _. apply aligned_mul; [apply v2_nonneg | apply v2_divides | apply atom_align_ok; exact Hra].
Qed.

Lemma ra_sound : forall n r, aligned (ra F n st r) (rs r).
Proof.
  induction n as [| n IH]; intros r; cbn [ra].
  - destruct (get st r) as [x |] eqn:Eg.
    + pose proof (fact_of r x Eg) as Hx. destruct x; cbn [fact_ok fact_okv] in Hx; try (apply aligned_nonpos; lia).
      * apply aligned_max; [apply aligned_nonpos; lia | apply (proj1 (proj2 (proj2 Hx)))].
      * exact Hx.
    + destruct (is_fixed F r) eqn:Hf; [| apply aligned_nonpos; lia].
      destruct (lookup (lf_aligns F) r) as [k |] eqn:El; [| apply aligned_nonpos; lia].
      rewrite (proj2 Hrs r Hf). exact (ok_aligns F E HE r k Hf El).
  - destruct (get st r) as [x |] eqn:Eg.
    + pose proof (fact_of r x Eg) as Hx. destruct x; cbn [fact_ok fact_okv] in Hx; try (apply aligned_nonpos; lia).
      * destruct Hx as [_ [[e [He Hsat]] [Hal _]]].
        destruct ((fm f >=? wd F r) || match ft f with [] => true | _ => false end).
        -- apply aligned_max; [exact Hal |].
           destruct (Z.eqb (flo f) (fhi f)) eqn:Elh.
           ++ apply Z.eqb_eq in Elh. assert (e = flo f) by lia. subst e.
              destruct (Z.le_gt_cases 0 (fm f)) as [Hm | Hm].
              ** apply (aligned_eqm (fm f) _ (rs r) (tsum wdn rs (ft f) + flo f)); [lia | apply Z.le_min_r | exact Hsat |].
                 eapply aligned_mono; [apply Z.le_min_l |]. apply form_align_with_sound; [exact (IH) | exact Elh | exact Hm].
              ** apply aligned_nonpos. lia.
           ++ (* a range constant gives no alignment *)
              apply aligned_nonpos. unfold form_align_with.
              assert (Hc : const_align f = 0) by (unfold const_align; rewrite Elh; reflexivity).
              rewrite Hc.
              assert (Hfold : forall ts k, k <= 0 ->
                        fold_left (fun k0 '(x, c) => Z.min k0 (v2 c + match snd x with EX | EZ | ES => ra F n st (fst x) | _ => 0 end)) ts k <= 0).
              { intros ts. induction ts as [| [x c] ts IHt]; intros k Hk; cbn [fold_left]; [exact Hk |].
                apply IHt. lia. }
              pose proof (Hfold (ft f) 0 (Z.le_refl 0)). lia.
        -- apply aligned_max; [apply aligned_nonpos; lia | exact Hal].
      * exact Hx.
    + destruct (is_fixed F r) eqn:Hf; [| apply aligned_nonpos; lia].
      destruct (lookup (lf_aligns F) r) as [k |] eqn:El; [| apply aligned_nonpos; lia].
      rewrite (proj2 Hrs r Hf). exact (ok_aligns F E HE r k Hf El).
Qed.

Lemma reg_align_sound : forall r, aligned (reg_align F st r) (rs r).
Proof. intros. unfold reg_align. apply ra_sound. Qed.

Lemma form_align_sound : forall f,
    flo f = fhi f -> 0 <= fm f -> aligned (form_align F st f) (tsum wdn rs (ft f) + flo f).
Proof.
  intros. unfold form_align. apply form_align_with_sound; [intros r; apply ra_sound | assumption | assumption].
Qed.

End Ranges.

Lemma round_range_sound : forall k x y v,
    x <= v <= y -> (2 ^ Z.max 0 k | v) -> fst (round k (x, y)) <= v <= snd (round k (x, y)).
Proof.
  intros k x y v Hv [q Hq]. unfold round. cbn [fst snd].
  set (s := 2 ^ Z.max 0 k) in *. assert (Hp : 0 < s) by (apply Z.pow_pos_nonneg; lia). subst v.
  split.
  - assert (H : - x / s >= - q) by (apply Z.le_ge, Z.div_le_lower_bound; lia). nia.
  - assert (H : y / s >= q) by (apply Z.le_ge, Z.div_le_lower_bound; lia). nia.
Qed.

(* The cached evaluators give the ranges of the uncached layers. These
   pointwise lemmas need no extensionality axiom. Each evaluator reads the
   atom ranges of the registers of the terms it is given alone, so two
   atom range functions that agree on those registers give one result. *)
Definition terms_in (P : reg -> Prop) (ts : terms) : Prop := forall t, In t ts -> P (fst (fst t)).
Definition ranges_agree (P : reg -> Prop) (ar br : atom -> option range) : Prop :=
 forall x, P (fst x) -> ar x = br x.

Lemma term_regs_in : forall t ts, In t ts -> In (fst (fst t)) (term_regs ts).
Proof. intros t ts Ht. exact (in_map (fun t => fst (fst t)) ts t Ht). Qed.
Lemma terms_in_app : forall P a b, terms_in P a -> terms_in P b -> terms_in P (a ++ b).
Proof. intros P a b Ha Hb t Ht. apply in_app_or in Ht. destruct Ht as [Ht|Ht];[exact (Ha t Ht)|exact (Hb t Ht)]. Qed.
Lemma terms_in_filter : forall P (g : atom * Z -> bool) ts, terms_in P ts -> terms_in P (filter g ts).
Proof. intros P g ts H t Ht. apply filter_In in Ht. exact (H t (proj1 Ht)). Qed.
Lemma terms_in_tscale : forall P k ts, terms_in P ts -> terms_in P (tscale k ts).
Proof. intros P k ts H t Ht. unfold tscale in Ht. apply in_map_iff in Ht. destruct Ht as [[x c] [<- Hin]]. exact (H (x,c) Hin). Qed.
Lemma tinsert_atoms : forall x c ts t, In t (tinsert x c ts) -> fst t = x \/ exists u, In u ts /\ fst u = fst t.
Proof.
 intros x c ts. induction ts as [|[y d] ts IH];intros t Ht;cbn [tinsert] in Ht.
 - destruct Ht as [<-|[]]. left. reflexivity.
 - destruct (atom_eqb x y).
   + destruct Ht as [<-|Ht];right;[exists (y,d)|exists t];split;try reflexivity;[left;reflexivity|right;exact Ht].
   + destruct (atom_ltb x y).
     * destruct Ht as [<-|Ht];[left;reflexivity|right;exists t;split;[exact Ht|reflexivity]].
     * destruct Ht as [<-|Ht];[right;exists (y,d);split;[left;reflexivity|reflexivity]|].
       destruct (IH t Ht) as [Hx|[u [Hu Hut]]];[left;exact Hx|right;exists u;split;[right;exact Hu|exact Hut]].
Qed.
Lemma terms_in_tsort : forall P ts, terms_in P ts -> terms_in P (tsort ts).
Proof.
 intros P ts. induction ts as [|[x c] ts IH];intros H t Ht;cbn [tsort] in Ht;[destruct Ht|].
 destruct (tinsert_atoms x c (tsort ts) t Ht) as [Hx|[u [Hu Hut]]].
 - rewrite Hx. exact (H (x,c) (or_introl eq_refl)).
 - rewrite <- Hut. apply IH;[intros t' Ht';apply H;right;exact Ht'|exact Hu].
Qed.
(* The register whose low bits low_bits finds is a register of the terms. *)
Lemma low_bits_in : forall P ts r e hi n c, terms_in P ts -> low_bits ts = Some (r,e,hi,n,c) -> P r.
Proof.
 intros P ts r e hi n c Hts. unfold low_bits.
 match goal with |- fold_left ?g _ _ = _ -> _ => set (step := g) end.
 assert (Hinv : forall l acc, terms_in P l ->
   (forall r' e' hi' n' c', acc = Some (r',e',hi',n',c') -> P r') ->
   forall r' e' hi' n' c', fold_left step l acc = Some (r',e',hi',n',c') -> P r').
 { induction l as [|t l IH];intros acc Hl Hacc;cbn [fold_left];[exact Hacc|].
   apply IH;[intros u Hu;apply Hl;right;exact Hu|].
   assert (Ht : P (fst (fst t))) by (apply Hl;left;reflexivity).
   intros r' e' hi' n' c' H. unfold step in H.
   destruct acc as [v|];[exact (Hacc r' e' hi' n' c' H)|].
   destruct t as [[r0 e0] c0]. cbn [fst] in Ht.
   destruct e0 as [| | |k|a m];cbv beta iota in H;try discriminate H.
   - match type of H with context [find ?g ts] => destruct (find g ts) as [[[? ?] ?]|] end;[|discriminate H]. injection H as Hr _ _ _ _. rewrite <- Hr. exact Ht.
   - destruct (Z.eqb a 0).
     + match type of H with context [find ?g ts] => destruct (find g ts) as [[[? ?] ?]|] end;[|discriminate H]. injection H as Hr _ _ _ _. rewrite <- Hr. exact Ht.
     + destruct (Z.eqb (Z.rem c0 (2 ^ a)) 0);[|discriminate H].
       match type of H with context [find ?g ts] => destruct (find g ts) as [[[? ?] ?]|] end;[|discriminate H]. injection H as Hr _ _ _ _. rewrite <- Hr. exact Ht. }
 apply (Hinv ts None Hts). intros r' e' hi' n' c' H. discriminate.
Qed.

Lemma plain_sum_local : forall P ar br ts lo hi, ranges_agree P ar br -> terms_in P ts ->
 plain_sum ar ts lo hi = plain_sum br ts lo hi.
Proof.
 intros P ar br ts. induction ts as [|[x c] ts IH];intros lo hi H Hts;cbn;[reflexivity|].
 rewrite (H x (Hts (x,c) (or_introl eq_refl))). destruct (br x) as [[a b]|];[|reflexivity].
 apply IH;[exact H|intros t Ht;apply Hts;right;exact Ht].
Qed.
Lemma plain_range_local : forall F P ar br f, ranges_agree P ar br -> terms_in P (ft f) ->
 plain_range F ar f = plain_range F br f.
Proof.
 intros F P ar br f H Hf. unfold plain_range. rewrite (plain_sum_local P ar br (ft f) _ _ H Hf).
 destruct (low_bits (ft f)) as [[[[[r e] hi] n] c]|] eqn:El;[|reflexivity].
 pose proof (low_bits_in P (ft f) r e hi n c Hf El) as Hr.
 destruct (match e with EX | EZ => true | _ => false end && _ && _);[|reflexivity].
 rewrite (plain_sum_local P ar br _ _ _ H);[reflexivity|].
 apply terms_in_filter,terms_in_tsort,terms_in_app;[exact Hf|].
 intros t Ht. destruct Ht as [<-|[<-|[<-|[]]]];exact Hr.
Qed.
(* A constraint that shares no atom with f gives f no range. *)
Lemma cons_range_local : forall P ar br f ts lo hi, ranges_agree P ar br -> terms_in P (ft f) ->
 (existsb (fun t => tmem ts (fst t)) (ft f) = true -> terms_in P ts) ->
 cons_range ar f (ts,lo,hi) = cons_range br f (ts,lo,hi).
Proof.
 intros P ar br f ts lo hi H Hf Hk. unfold cons_range. cbv beta iota zeta.
 destruct (filter (fun t => tmem ts (fst t)) (ft f)) as [|[x c1] pick] eqn:Ep;[reflexivity|].
 assert (Hts : terms_in P ts).
 { apply Hk. apply existsb_exists. exists (x,c1).
   assert (Hin : In (x,c1) (filter (fun t => tmem ts (fst t)) (ft f))) by (rewrite Ep;left;reflexivity).
   apply filter_In in Hin. exact Hin. }
 destruct (filter (fun t => tmem (ft f) (fst t)) ts) as [|[y c0] core];[reflexivity|].
 destruct (Z.eqb c0 0 || negb (Z.eqb (Z.rem c1 c0) 0));[reflexivity|].
 destruct (negb (scaled (Z.quot c1 c0) ((x,c1)::pick) ((y,c0)::core)));[reflexivity|].
 rewrite (plain_sum_local P ar br _ _ _ H (terms_in_filter _ _ _ Hf)).
 rewrite (plain_sum_local P ar br _ _ _ H (terms_in_filter _ _ _ Hts)). reflexivity.
Qed.
Lemma form_range_in_local : forall F P cs ar br f, ranges_agree P ar br -> terms_in P (ft f) ->
 (forall ts lo hi, In (ts,lo,hi) cs -> existsb (fun t => tmem ts (fst t)) (ft f) = true -> terms_in P ts) ->
 form_range_in F cs ar f = form_range_in F cs br f.
Proof.
 intros F P cs ar br f H Hf Hcs. unfold form_range_in. cbv zeta.
 set (D := dependencies_in cs (ft f)). clearbody D.
 assert (HC : ranges_agree P (cached_range ar D) (cached_range br D)).
 { intros x Hx. rewrite !cached_range_same. exact (H x Hx). }
 rewrite (plain_range_local F P _ _ f HC Hf).
 generalize (plain_range F (cached_range br D) f) as acc.
 induction cs as [|[[ts lo] hi] ks IH];intros acc;cbn [fold_left];[reflexivity|].
 rewrite (cons_range_local P _ _ f ts lo hi HC Hf (Hcs ts lo hi (or_introl eq_refl))).
 apply IH. intros ts' lo' hi' Hin. apply (Hcs ts' lo' hi'). right. exact Hin.
Qed.
Lemma plain_range_extensional : forall F ar br, (forall x, ar x=br x) -> forall f,
 plain_range F ar f=plain_range F br f.
Proof. intros F ar br H f. apply (plain_range_local F (fun _ => True));[intros x _;apply H|intros t _;exact I]. Qed.
Lemma form_range_in_extensional : forall F cs ar br, (forall x,ar x=br x) -> forall f,
 form_range_in F cs ar f=form_range_in F cs br f.
Proof.
 intros F cs ar br H f. apply (form_range_in_local F (fun _ => True));
 [intros x _;apply H|intros t _;exact I|intros ts lo hi _ _ t _;exact I].
Qed.
Lemma form_range_with_extensional : forall F st ar br, (forall x,ar x=br x) -> forall f,
 form_range_with F st ar f=form_range_with F st br f.
Proof. intros F st ar br H f. exact (form_range_in_extensional F (acons st) ar br H f). Qed.
Lemma form_range_ix_extensional : forall F ix ar br, (forall x,ar x=br x) -> forall f,
 form_range_ix F ix ar f=form_range_ix F ix br f.
Proof. intros F ix ar br H f. exact (form_range_in_extensional F (cons_near ix (ft f)) ar br H f). Qed.

Definition rf_equivalent (a b:rfuns) := forall r,f_rr a r=f_rr b r /\ f_sr a r=f_sr b r.
(* Range functions that agree on the registers that satisfy P. *)
Definition rf_agree (P : reg -> Prop) (a b : rfuns) : Prop :=
 forall r, P r -> f_rr a r = f_rr b r /\ f_sr a r = f_sr b r.
Lemma atom_range_local : forall F P a b, rf_agree P a b -> ranges_agree P (atom_range_with F a) (atom_range_with F b).
Proof.
 intros F P a b H [r e] Hr. cbn [fst] in Hr. destruct (H r Hr) as [Hrr Hsr].
 destruct e;cbn [atom_range_with];rewrite ?Hrr,?Hsr;reflexivity.
Qed.
Lemma atom_range_extensional : forall F a b,rf_equivalent a b -> forall x,
 atom_range_with F a x=atom_range_with F b x.
Proof. intros F a b H x. apply (atom_range_local F (fun _ => True));[intros r _;exact (H r)|exact I]. Qed.
Lemma cache_rf_equivalent : forall fs regs,rf_equivalent (cache_rf fs regs) fs.
Proof. exact cache_rf_same. Qed.

(* The registers whose ranges in the layer below a layer reads at r. *)
Definition layer_reads (F : lfacts) (ix : PM.t (list constraint)) (st : astate) (r : reg) : list reg :=
 match get st r with
 | Some (FForm f _ _) => if fm f >=? wd F r then range_reads ix (ft f) else []
 | _ => []
 end.
Lemma form_range_ix_reads : forall F P ix ar br f, ranges_agree P ar br ->
 (forall u, In u (range_reads ix (ft f)) -> P u) -> form_range_ix F ix ar f = form_range_ix F ix br f.
Proof.
 intros F P ix ar br f H Hr. unfold form_range_ix. apply (form_range_in_local F P);[exact H| |].
 - intros t Ht. apply Hr. unfold range_reads. apply in_or_app. left. apply term_regs_in. exact Ht.
 - intros ts lo hi Hin _ t Ht. apply Hr. unfold range_reads. apply in_or_app. right.
   apply in_flat_map. exists (ts,lo,hi). split;[exact Hin|apply term_regs_in;exact Ht].
Qed.
Lemma range_layer_local : forall F ix st a b r, rf_agree (fun u => In u (layer_reads F ix st r)) a b ->
 f_rr (range_layer F ix st a) r = f_rr (range_layer F ix st b) r /\
 f_sr (range_layer F ix st a) r = f_sr (range_layer F ix st b) r.
Proof.
 intros F ix st a b r H. unfold range_layer. cbn[f_rr f_sr].
 destruct (get st r) as [[f al vr| | |]|] eqn:Hg;try(split;reflexivity).
 destruct (fm f >=? wd F r) eqn:Hm;[|split;destruct vr;reflexivity].
 assert (HF : form_range_ix F ix (atom_range_with F a) f = form_range_ix F ix (atom_range_with F b) f).
 { apply (form_range_ix_reads F (fun u => In u (layer_reads F ix st r)));[apply atom_range_local;exact H|].
   intros u Hu. unfold layer_reads. rewrite Hg,Hm. exact Hu. }
 rewrite HF. split;destruct vr;reflexivity.
Qed.
Lemma range_layer_extensional : forall F ix st a b,
 rf_equivalent a b -> rf_equivalent(range_layer F ix st a)(range_layer F ix st b).
Proof. intros F ix st a b H r. apply range_layer_local. intros u _. exact (H u). Qed.
Lemma rf_equivalent_trans : forall a b c,rf_equivalent a b -> rf_equivalent b c -> rf_equivalent a c.
Proof. intros a b c H1 H2 r. destruct(H1 r) as [H1r H1s]. destruct(H2 r) as [H2r H2s]. split;congruence. Qed.
Lemma range_pairs_eqb_eq : forall a b,range_pairs_eqb a b=true -> a=b.
Proof.
 assert(Ho:forall x y,range_option_eqb x y=true -> x=y).
 { intros [[p q]|] [[p' q']|] H;cbn in H;try discriminate;[|reflexivity].
   apply andb_true_iff in H as [H1 H2]. apply Z.eqb_eq in H1,H2. subst. reflexivity. }
 induction a as [|[r [x y]] a IH];intros [|[r' [x' y']] b] H;cbn in H;try discriminate;[reflexivity|].
 apply andb_true_iff in H as [H Hrest]. apply andb_true_iff in H as [H Hy]. apply andb_true_iff in H as [Hr Hx].
 apply Pos.eqb_eq in Hr. apply Ho in Hx,Hy. rewrite (IH b Hrest). subst. reflexivity.
Qed.
Lemma layer_pairs_equivalent : forall F regs ix st a b,rf_equivalent a b ->
 layer_pairs F regs ix st a=layer_pairs F regs ix st b.
Proof.
 intros F regs ix st a b H. unfold layer_pairs. apply map_ext. intros r. f_equal.
 rewrite !range_pair_exact. destruct(range_layer_extensional F ix st a b H r) as [Hr Hs]. rewrite Hr,Hs. reflexivity.
Qed.
Lemma cache_range_layer_equivalent : forall F regs ix st prev,
 rf_equivalent(cache_range_layer F ix st prev(layer_pairs F regs ix st prev))(range_layer F ix st prev).
Proof. intros F regs ix st prev r. exact(cache_range_layer_same F st regs ix prev r). Qed.
Lemma map_pairs_same : forall (f g:reg -> option range * option range) regs r,
 map(fun u=>(u,f u)) regs=map(fun u=>(u,g u)) regs -> In r regs -> f r=g r.
Proof.
 intros f g regs r. induction regs as [|u regs IH];intros H Hin;[destruct Hin|].
 cbn in H. injection H as Hu Hrest. destruct Hin as [<-|Hin];[exact Hu|exact(IH Hrest Hin)].
Qed.

(* A register whose form a layer evaluates, which reach_graph collects. *)
Definition layer_node (F : lfacts) (st : astate) (r : reg) : bool :=
 match get st r with
 | Some (FForm f _ _) => (wd F r <=? fm f) && negb (Nat.eqb (length (ft f)) 0)
 | _ => false
 end.
(* A register of regs, or one whose range in a layer reads no register of
   the layer below. *)
Definition covered (F : lfacts) (st : astate) (regs : list reg) (u : reg) : Prop :=
 layer_node F st u = true -> In u regs.
(* The layers read at each register of regs only covered registers. *)
Definition reads_closed (F : lfacts) (ix : PM.t (list constraint)) (st : astate) (regs : list reg) : Prop :=
 forall r u, In r regs -> In u (layer_reads F ix st r) -> covered F st regs u.

Lemma layer_reads_node : forall F ix st r, layer_node F st r = false -> layer_reads F ix st r = [].
Proof.
 intros F ix st r H. unfold layer_node,layer_reads in *.
 destruct (get st r) as [[f al vr| | |]|];try reflexivity.
 rewrite Z.geb_leb. destruct (wd F r <=? fm f);[|reflexivity].
 cbn [andb] in H. destruct (ft f);[reflexivity|discriminate].
Qed.
Lemma covered_reads : forall F ix st regs, reads_closed F ix st regs ->
 forall r u, covered F st regs r -> In u (layer_reads F ix st r) -> covered F st regs u.
Proof.
 intros F ix st regs Hc r u Hr Hu. destruct (layer_node F st r) eqn:Hn.
 - exact (Hc r u (Hr Hn) Hu).
 - rewrite (layer_reads_node F ix st r Hn) in Hu. destruct Hu.
Qed.
Lemma range_layer_covered : forall F ix st regs a b, reads_closed F ix st regs ->
 rf_agree (covered F st regs) a b ->
 rf_agree (covered F st regs) (range_layer F ix st a) (range_layer F ix st b).
Proof.
 intros F ix st regs a b Hc H r Hr. apply range_layer_local.
 intros u Hu. exact (H u (covered_reads F ix st regs Hc r u Hr Hu)).
Qed.
(* Once two consecutive plain layers agree on the covered registers, every
   later layer agrees on them too. *)
Lemma rf_plain_stable : forall F ix st regs j, reads_closed F ix st regs ->
 rf_agree (covered F st regs) (rf_plain F ix (S j) st) (rf_plain F ix j st) ->
 forall i, rf_agree (covered F st regs) (rf_plain F ix (j+i) st) (rf_plain F ix j st).
Proof.
 intros F ix st regs j Hc H i. induction i as [|i IH];[rewrite Nat.add_0_r;intros r _;split;reflexivity|].
 rewrite Nat.add_succ_r. intros r Hr. cbn[rf_plain].
 destruct (range_layer_covered F ix st regs _ _ Hc IH r Hr) as [H1 H2].
 destruct (H r Hr) as [H3 H4]. cbn [rf_plain] in H3,H4. split;congruence.
Qed.
Lemma rf_stable_layers_agree : forall F ix st regs m j prev, reads_closed F ix st regs ->
 rf_equivalent prev (rf_plain F ix (S j) st) ->
 rf_agree (covered F st regs)
   (rf_stable_layers F regs ix st m prev (layer_pairs F regs ix st (rf_plain F ix j st)))
   (rf_plain F ix (S j+m) st).
Proof.
 intros F ix st regs m. induction m as [|m IH];intros j prev Hc Hp;cbn[rf_stable_layers].
 - rewrite Nat.add_0_r. intros r _. exact (Hp r).
 - rewrite(layer_pairs_equivalent F regs ix st prev (rf_plain F ix (S j) st) Hp).
   destruct(range_pairs_eqb (layer_pairs F regs ix st (rf_plain F ix (S j) st))
     (layer_pairs F regs ix st (rf_plain F ix j st))) eqn:He.
   + apply range_pairs_eqb_eq in He.
     assert(Hstep:rf_agree (covered F st regs) (rf_plain F ix (S (S j)) st) (rf_plain F ix (S j) st)).
     { intros r Hr. destruct(in_dec Pos.eq_dec r regs) as [Hin|Hout].
       - pose proof (map_pairs_same _ _ _ r He Hin) as Hpair.
         rewrite !range_pair_exact in Hpair. injection Hpair as Hrr Hsr. cbn[rf_plain]. split;assumption.
       - assert (Hn : layer_node F st r = false).
         { destruct (layer_node F st r) eqn:E;[exfalso;exact (Hout (Hr E))|reflexivity]. }
         cbn[rf_plain]. apply range_layer_local. intros u Hu.
         rewrite (layer_reads_node F ix st r Hn) in Hu. destruct Hu. }
     intros r Hr. destruct (Hp r) as [H1 H2].
     destruct (rf_plain_stable F ix st regs (S j) Hc Hstep (S m) r Hr) as [H3 H4]. split;congruence.
   + replace (S j+S m)%nat with (S (S j)+m)%nat by lia. apply IH;[exact Hc|].
     rewrite <- (layer_pairs_equivalent F regs ix st prev _ Hp).
     apply rf_equivalent_trans with (range_layer F ix st prev);[apply cache_range_layer_equivalent|].
     cbn[rf_plain]. apply range_layer_extensional. exact Hp.
Qed.
Lemma rf_cached_layers_agree : forall F ix st regs n, reads_closed F ix st regs ->
 rf_agree (covered F st regs) (rf_cached_layers F regs ix n st) (rf_plain F ix n st).
Proof.
 intros F ix st regs n Hc. unfold rf_cached_layers. destruct n as [|n];[intros r _;apply cache_rf_equivalent|].
 assert(Hbase:rf_equivalent(cache_rf(range_base F st) regs)(rf_plain F ix 0 st)) by apply cache_rf_equivalent.
 rewrite(layer_pairs_equivalent F regs ix st _ _ Hbase).
 apply (rf_stable_layers_agree F ix st regs n 0);[exact Hc|].
 apply rf_equivalent_trans with (range_layer F ix st (cache_rf(range_base F st) regs)).
 - rewrite <- (layer_pairs_equivalent F regs ix st _ _ Hbase). apply cache_range_layer_equivalent.
 - cbn[rf_plain]. apply range_layer_extensional. exact Hbase.
Qed.
(* The layers cached on registers closed under the layers' reads give the
   ranges of the uncached layers at every covered register. *)
Lemma rf_on_agree : forall F ix st regs n, reads_closed F ix st regs ->
 rf_agree (covered F st regs) (rf_on F regs ix n st) (rf_plain F ix n st).
Proof.
 intros F ix st regs n Hc. destruct n as [|n];[intros r _;split;reflexivity|].
 cbn [rf_on rf_plain]. apply range_layer_covered;[exact Hc|]. apply rf_cached_layers_agree. exact Hc.
Qed.
Lemma cache_regs_covered : forall F st r, covered F st (range_cache_regs st) r.
Proof.
 intros F st r Hn. unfold layer_node in Hn. unfold range_cache_regs.
 destruct (get st r) as [[f al vr| | |]|] eqn:Hg;try discriminate.
 apply in_flat_map. exists (r,FForm f al vr). split;[apply PM.elements_correct;exact Hg|].
 cbn [snd fst]. apply andb_true_iff in Hn as [_ Hn].
 destruct (Nat.eqb (length (ft f)) 0);[discriminate|left;reflexivity].
Qed.
Lemma rf_memo_equivalent : forall F ix n st memo,
 rf_equivalent(rf_with_cache F ix memo n st)(rf_with_cache F ix false n st).
Proof.
 intros F ix n st memo. unfold rf_with_cache. destruct memo;[|intros r;split;reflexivity].
 intros r. apply (rf_on_agree F ix st (range_cache_regs st) n);[|apply cache_regs_covered].
 intros r' u _ _. apply cache_regs_covered.
Qed.
Lemma tighten_local : forall F ix st P a b, rf_agree P a b -> rf_agree P (tighten F ix st a) (tighten F ix st b).
Proof.
 intros F ix st P a b H r Hr. destruct (H r Hr) as [Hrr Hsr].
 unfold tighten;cbn [f_rr f_sr]. rewrite Hrr,Hsr. split;reflexivity.
Qed.
Lemma tighten_extensional : forall F ix st a b,rf_equivalent a b ->rf_equivalent (tighten F ix st a) (tighten F ix st b).
Proof. intros F ix st a b H r. apply (tighten_local F ix st (fun _ => True));[intros u _;exact (H u)|exact I]. Qed.

(* reach_regs collects every register of regs whose form a layer
   evaluates, and is closed under the layers' reads. Each round of
   reach_graph keeps its nodes, adds registers with a fact, and reads next
   the registers that the layers of the added registers read. *)
Section Reach.
Variable F : lfacts.
Variable ix : PM.t (list constraint).
Variable st : astate.

Lemma mem_add_cases : forall (m : PM.t unit) r u, PM.mem u (PM.add r tt m) = true -> u = r \/ PM.mem u m = true.
Proof.
 intros m r u H. rewrite PM.mem_find in *. destruct (Pos.eq_dec u r) as [->|Hne];[left;reflexivity|].
 rewrite PM.gso in H by exact Hne. right. exact H.
Qed.
Lemma mem_add_self : forall (m : PM.t unit) r, PM.mem r (PM.add r tt m) = true.
Proof. intros m r. rewrite PM.mem_find, PM.gss. reflexivity. Qed.
Lemma mem_add_keep : forall (m : PM.t unit) r u, PM.mem u m = true -> PM.mem u (PM.add r tt m) = true.
Proof.
 intros m r u H. rewrite PM.mem_find in *. destruct (Pos.eq_dec u r) as [->|Hne];[rewrite PM.gss;reflexivity|].
 rewrite PM.gso by exact Hne. exact H.
Qed.

Definition reach_grows (a b : PM.t unit * list reg) : Prop :=
 (forall r, PM.mem r (fst a) = true -> PM.mem r (fst b) = true) /\
 incl (snd a) (snd b) /\
 (forall r, PM.mem r (fst b) = true -> PM.mem r (fst a) = true \/ incl (layer_reads F ix st r) (snd b)) /\
 (snd b = snd a \/ exists r x, PM.mem r (fst a) = false /\ PM.mem r (fst b) = true /\ get st r = Some x).

Lemma reach_grows_refl : forall a, reach_grows a a.
Proof. intros a. split;[|split;[|split]];[intros r H;exact H|intros x H;exact H|intros r H;left;exact H|left;reflexivity]. Qed.
Lemma reach_grows_trans : forall a b c, reach_grows a b -> reach_grows b c -> reach_grows a c.
Proof.
 intros a b c [H1 [H2 [H3 H4]]] [G1 [G2 [G3 G4]]]. split;[|split;[|split]].
 - intros r Hr. apply G1,H1,Hr.
 - intros x Hx. apply G2,H2,Hx.
 - intros r Hr. destruct (G3 r Hr) as [Hb|Hc];[|right;exact Hc].
   destruct (H3 r Hb) as [Ha|Hi];[left;exact Ha|right;intros x Hx;apply G2,Hi,Hx].
 - destruct G4 as [Eq|[r [x [Hb [Hc Hg]]]]].
   + destruct H4 as [Eq'|[r [x [Ha [Hb Hg]]]]];[left;congruence|].
     right. exists r,x. split;[exact Ha|split;[apply G1,Hb|exact Hg]].
   + right. exists r,x. split;[|split;[exact Hc|exact Hg]].
     destruct (PM.mem r (fst a)) eqn:Ha;[|reflexivity]. rewrite (H1 r Ha) in Hb. discriminate.
Qed.
Lemma reach_step_grows : forall a r, reach_grows a (reach_step F ix st a r).
Proof.
 intros [nodes next] r. unfold reach_step.
 destruct (PM.mem r nodes) eqn:Hm;[apply reach_grows_refl|].
 destruct (get st r) as [[f al vr| | |]|] eqn:Hg;try apply reach_grows_refl.
 destruct ((wd F r <=? fm f) && negb (Nat.eqb (length (ft f)) 0)) eqn:Hc;[|apply reach_grows_refl].
 split;[|split;[|split]];cbn [fst snd].
 - intros u Hu. apply mem_add_keep,Hu.
 - intros x Hx. apply in_or_app. right. exact Hx.
 - intros u Hu. destruct (mem_add_cases nodes r u Hu) as [->|Hn];[right|left;exact Hn].
   unfold layer_reads. rewrite Hg. apply andb_true_iff in Hc as [Hw _]. rewrite Z.geb_leb,Hw.
   intros x Hx. apply in_or_app. left. exact Hx.
 - right. exists r,(FForm f al vr). split;[exact Hm|split;[apply mem_add_self|exact Hg]].
Qed.
Lemma reach_fold_grows : forall l a, reach_grows a (fold_left (reach_step F ix st) l a).
Proof.
 intros l. induction l as [|r l IH];intros a;cbn [fold_left];[apply reach_grows_refl|].
 apply reach_grows_trans with (reach_step F ix st a r);[apply reach_step_grows|apply IH].
Qed.
Lemma reach_fold_seeds : forall l a r, In r l -> layer_node F st r = true ->
 PM.mem r (fst (fold_left (reach_step F ix st) l a)) = true.
Proof.
 intros l. induction l as [|r0 l IH];intros a r Hr Hn;[destruct Hr|]. cbn [fold_left].
 destruct Hr as [<-|Hr];[|apply IH;assumption].
 apply (proj1 (reach_fold_grows l _)).
 destruct a as [nodes next]. unfold reach_step.
 destruct (PM.mem r0 nodes) eqn:Hm;[exact Hm|]. unfold layer_node in Hn.
 destruct (get st r0) as [[f al vr| | |]|];try discriminate. rewrite Hn. apply mem_add_self.
Qed.

(* The registers with a fact that nodes does not hold. *)
Definition reach_left (nodes : PM.t unit) : nat :=
 length (filter (fun p => negb (PM.mem (fst p) nodes)) (PM.elements (amap st))).
Lemma filter_length_lt : forall (A : Type) (f g : A -> bool) l a,
 (forall b, g b = true -> f b = true) -> In a l -> f a = true -> g a = false ->
 (length (filter g l) < length (filter f l))%nat.
Proof.
 intros A f g l a Hfg. induction l as [|b l IH];intros Ha Hf Hg;[destruct Ha|].
 assert (Hle : forall l', (length (filter g l') <= length (filter f l'))%nat).
 { induction l' as [|c l' IH'];cbn;[lia|]. destruct (g c) eqn:Ec;[rewrite (Hfg c Ec);cbn;lia|].
   destruct (f c);cbn;lia. }
 destruct Ha as [<-|Ha];cbn.
 - rewrite Hf,Hg. cbn. pose proof (Hle l). lia.
 - destruct (g b) eqn:Eb;[rewrite (Hfg b Eb);cbn;pose proof (IH Ha Hf Hg);lia|].
   destruct (f b);cbn;pose proof (IH Ha Hf Hg);lia.
Qed.
Lemma reach_left_lt : forall n1 n2 r x, (forall u, PM.mem u n1 = true -> PM.mem u n2 = true) ->
 PM.mem r n1 = false -> PM.mem r n2 = true -> get st r = Some x -> (reach_left n2 < reach_left n1)%nat.
Proof.
 intros n1 n2 r x H H1 H2 Hg. unfold reach_left. apply (filter_length_lt _ _ _ _ (r,x)).
 - intros [u y]. cbn [fst]. destruct (PM.mem u n2) eqn:E2;[discriminate|].
   destruct (PM.mem u n1) eqn:E1;[rewrite (H u E1) in E2;discriminate|reflexivity].
 - apply PM.elements_correct. exact Hg.
 - cbn [fst]. rewrite H1. reflexivity.
 - cbn [fst]. rewrite H2. reflexivity.
Qed.
Lemma reach_left_empty : (reach_left (PM.empty unit) <= PM.cardinal (amap st))%nat.
Proof.
 unfold reach_left. rewrite PM.cardinal_1. generalize (PM.elements (amap st)) as l.
 induction l as [|p l IH];cbn;[lia|]. destruct (negb _);cbn;lia.
Qed.

(* Every register that a layer at a node reads and that a layer evaluates
   is a node or waits in pending. *)
Definition reach_inv (nodes : PM.t unit) (pending : list reg) : Prop :=
 forall r u, PM.mem r nodes = true -> In u (layer_reads F ix st r) -> layer_node F st u = true ->
 PM.mem u nodes = true \/ In u pending.

Lemma reach_graph_spec : forall fuel nodes pending, reach_inv nodes pending -> (reach_left nodes < fuel)%nat ->
 (forall r u, PM.mem r (reach_graph F fuel ix st pending nodes) = true -> In u (layer_reads F ix st r) ->
   layer_node F st u = true -> PM.mem u (reach_graph F fuel ix st pending nodes) = true) /\
 (forall r, PM.mem r nodes = true -> PM.mem r (reach_graph F fuel ix st pending nodes) = true) /\
 (forall r, In r pending -> layer_node F st r = true -> PM.mem r (reach_graph F fuel ix st pending nodes) = true).
Proof.
 intros fuel. induction fuel as [|fuel IH];intros nodes pending Hinv Hleft;[lia|].
 cbn [reach_graph].
 destruct (fold_left (reach_step F ix st) pending (nodes,[])) as [n1 x1] eqn:Hfold.
 pose proof (reach_fold_grows pending (nodes,[])) as Hg. rewrite Hfold in Hg.
 destruct Hg as [G1 [_ [G3 G4]]]. cbn [fst snd] in G1,G3,G4.
 assert (Hseed : forall r, In r pending -> layer_node F st r = true -> PM.mem r n1 = true).
 { intros r Hr Hn. pose proof (reach_fold_seeds pending (nodes,[]) r Hr Hn) as H. rewrite Hfold in H. exact H. }
 assert (Hinv1 : reach_inv n1 x1).
 { intros r u Hr Hu Hn. destruct (G3 r Hr) as [H0|Hi];[|right;exact (Hi u Hu)].
   destruct (Hinv r u H0 Hu Hn) as [Hm|Hp];[left;exact (G1 u Hm)|left;exact (Hseed u Hp Hn)]. }
 destruct x1 as [|x xs].
 - split;[|split;[exact G1|exact Hseed]].
   intros r u Hr Hu Hn. destruct (Hinv1 r u Hr Hu Hn) as [Hm|[]]. exact Hm.
 - destruct G4 as [Eq|[r [y [Hr0 [Hr1 Hgr]]]]];[discriminate|].
   assert (Hl : (reach_left n1 < fuel)%nat).
   { pose proof (reach_left_lt nodes n1 r y G1 Hr0 Hr1 Hgr). lia. }
   destruct (IH n1 (x::xs) Hinv1 Hl) as [R1 [R2 _]]. split;[exact R1|split].
   + intros u Hu. apply R2,G1,Hu.
   + intros u Hu Hn. apply R2,Hseed;assumption.
Qed.

Lemma reach_regs_mem : forall regs u, In u (reach_regs F ix st regs) <->
 PM.mem u (reach_graph F (S (PM.cardinal (amap st))) ix st regs (PM.empty unit)) = true.
Proof.
 intros regs u. unfold reach_regs. set (m := reach_graph F _ ix st regs (PM.empty unit)). split.
 - intros H. apply in_map_iff in H. destruct H as [[v y] [<- Hin]]. cbn [fst].
   apply PM.elements_complete in Hin. rewrite PM.mem_find,Hin. reflexivity.
 - intros H. rewrite PM.mem_find in H. destruct (PM.find u m) as [y|] eqn:Hf;[|discriminate].
   apply in_map_iff. exists (u,y). split;[reflexivity|]. apply PM.elements_correct. exact Hf.
Qed.
Lemma reach_spec : forall regs,
 (forall r u, In r (reach_regs F ix st regs) -> In u (layer_reads F ix st r) ->
   layer_node F st u = true -> In u (reach_regs F ix st regs)) /\
 (forall r, In r regs -> layer_node F st r = true -> In r (reach_regs F ix st regs)).
Proof.
 intros regs. destruct (reach_graph_spec (S (PM.cardinal (amap st))) (PM.empty unit) regs) as [R1 [_ R3]].
 - intros r u H. rewrite PM.mem_find,PM.gempty in H. discriminate.
 - pose proof reach_left_empty. lia.
 - split.
   + intros r u Hr Hu Hn. apply reach_regs_mem. apply reach_regs_mem in Hr. exact (R1 r u Hr Hu Hn).
   + intros r Hr Hn. apply reach_regs_mem. exact (R3 r Hr Hn).
Qed.
Lemma reach_closed : forall regs, reads_closed F ix st (reach_regs F ix st regs).
Proof. intros regs r u Hr Hu Hn. exact (proj1 (reach_spec regs) r u Hr Hu Hn). Qed.
Lemma reach_seeds : forall regs u, In u regs -> covered F st (reach_regs F ix st regs) u.
Proof. intros regs u Hu Hn. exact (proj2 (reach_spec regs) u Hu Hn). Qed.
End Reach.

(* The ranges of a query of regs are those of the uncached layers on the
   registers that reach_regs covers, and on regs in particular. *)
Lemma rft_on_agree : forall F ix st regs,
 rf_agree (covered F st (reach_regs F ix st regs)) (rft_on F ix st regs) (tighten F ix st (rf_plain F ix DEPTH st)).
Proof. intros F ix st regs. unfold rft_on. apply tighten_local. apply rf_on_agree. apply reach_closed. Qed.

(* The ranges of the uncached layers of depth DEPTH. *)
Definition plain_ranges (F : lfacts) (st : astate) : rfuns :=
 let ix := cons_index (acons st) in tighten F ix st (rf_plain F ix DEPTH st).

Lemma rft_on_exact : forall F st regs u, In u regs ->
 f_rr (rft_on F (cons_index (acons st)) st regs) u = f_rr (plain_ranges F st) u /\
 f_sr (rft_on F (cons_index (acons st)) st regs) u = f_sr (plain_ranges F st) u.
Proof. intros F st regs u Hu. apply rft_on_agree. apply reach_seeds. exact Hu. Qed.

Theorem reg_range_exact : forall F st r, reg_range F st r = f_rr (plain_ranges F st) r.
Proof. intros F st r. exact (proj1 (rft_on_exact F st [r] r (or_introl eq_refl))). Qed.
Theorem sreg_range_exact : forall F st r, sreg_range F st r = f_sr (plain_ranges F st) r.
Proof. intros F st r. exact (proj2 (rft_on_exact F st [r] r (or_introl eq_refl))). Qed.
Theorem atom_range_exact : forall F st x, atom_range F st x = atom_range_with F (plain_ranges F st) x.
Proof.
 intros F st x. unfold atom_range. apply (atom_range_local F (fun u => In u [fst x]));[|left;reflexivity].
 intros u Hu. exact (rft_on_exact F st [fst x] u Hu).
Qed.
Theorem form_range_exact : forall F st f,
 form_range F st f = form_range_ix F (cons_index (acons st)) (atom_range_with F (plain_ranges F st)) f.
Proof.
 intros F st f. unfold form_range. cbv zeta.
 apply (form_range_ix_reads F (fun u => In u (range_reads (cons_index (acons st)) (ft f))));[|intros u Hu;exact Hu].
 apply atom_range_local. intros u Hu. exact (rft_on_exact F st _ u Hu).
Qed.

(* Each constraint of cs is in the bucket of each of its registers. *)
Lemma cons_regs_in : forall ts lo hi t, In t ts -> In (fst (fst t)) (cons_regs (ts,lo,hi)).
Proof.
 intros ts lo hi t Ht. unfold cons_regs.
 assert (H : forall l acc, In (fst (fst t)) acc \/ In t l ->
   In (fst (fst t)) (fold_left (fun rs t => let r := fst (fst t) in if existsb (Pos.eqb r) rs then rs else r :: rs) l acc)).
 { induction l as [|u l IH];intros acc Hacc;cbn [fold_left].
   - destruct Hacc as [Ha|[]]. exact Ha.
   - apply IH. cbv zeta. destruct Hacc as [Ha|[<-|Hl]].
     + left. match goal with |- context [if ?b then _ else _] => destruct b end;[exact Ha|right;exact Ha].
     + left. match goal with |- context [if ?b then _ else _] => destruct b eqn:E end;[|left;reflexivity].
       apply existsb_exists in E. destruct E as [r [Hr Hrt]]. apply Pos.eqb_eq in Hrt. rewrite <- Hrt in Hr. exact Hr.
     + right. exact Hl. }
 apply H. right. exact Ht.
Qed.
Lemma cons_bucket_keep : forall k regs ix r k', In k' (cons_bucket ix r) ->
 In k' (cons_bucket (fold_left (fun ix r => PM.add r (k :: cons_bucket ix r) ix) regs ix) r).
Proof.
 intros k regs. induction regs as [|r0 regs IH];intros ix r k' H;cbn [fold_left];[exact H|].
 apply IH. unfold cons_bucket at 1. destruct (Pos.eq_dec r r0) as [->|Hne].
 - rewrite PM.gss. right. exact H.
 - rewrite PM.gso by exact Hne. exact H.
Qed.
Lemma cons_bucket_new : forall k regs ix r, In r regs ->
 In k (cons_bucket (fold_left (fun ix r => PM.add r (k :: cons_bucket ix r) ix) regs ix) r).
Proof.
 intros k regs. induction regs as [|r0 regs IH];intros ix r Hr;[destruct Hr|]. cbn [fold_left].
 destruct Hr as [<-|Hr];[|apply IH;exact Hr].
 apply cons_bucket_keep. unfold cons_bucket. rewrite PM.gss. left. reflexivity.
Qed.
Lemma cons_bucket_complete : forall cs k r, In k cs -> In r (cons_regs k) -> In k (cons_bucket (cons_index cs) r).
Proof.
 intros cs. induction cs as [|c cs IH];intros k r Hk Hr;[destruct Hk|].
 unfold cons_index. cbn [fold_right]. fold (cons_index cs).
 destruct Hk as [<-|Hk];[apply cons_bucket_new;exact Hr|apply cons_bucket_keep;apply IH;assumption].
Qed.
(* A constraint of the state that shares an atom with f is a constraint on
   f's registers, so form_range_with reads the registers of range_reads. *)
Lemma cons_near_complete : forall cs ts lo hi f, In (ts,lo,hi) cs ->
 existsb (fun t => tmem ts (fst t)) (ft f) = true -> In (ts,lo,hi) (cons_near (cons_index cs) (ft f)).
Proof.
 intros cs ts lo hi f Hin Hs. apply existsb_exists in Hs. destruct Hs as [u [Hu Hm]].
 unfold cons_near. apply in_flat_map. exists u. split;[exact Hu|].
 apply cons_bucket_complete;[exact Hin|]. unfold tmem in Hm. apply existsb_exists in Hm.
 destruct Hm as [[y c] [Hy Hxy]]. apply atom_eqb_eq in Hxy. destruct u as [x cu]. cbn [fst] in *. subst x.
 exact (cons_regs_in ts lo hi (y,c) Hy).
Qed.
Theorem form_range_nat_exact : forall F st f, form_range_nat F st f =
 form_range_with F st (fun x => match atom_range_with F (plain_ranges F st) x with
                                | Some g => Some g | None => natural_range F x end) f.
Proof.
 intros F st f. unfold form_range_nat,form_range_with. cbv zeta.
 apply (form_range_in_local F (fun u => In u (range_reads (cons_index (acons st)) (ft f)))).
 - intros x Hx. rewrite (atom_range_local F _ _ _ (fun u Hu => rft_on_exact F st _ u Hu) x Hx). reflexivity.
 - intros t Ht. unfold range_reads. apply in_or_app. left. apply term_regs_in. exact Ht.
 - intros ts lo hi Hin Hs t Ht. unfold range_reads. apply in_or_app. right. apply in_flat_map.
   exists (ts,lo,hi). split;[exact (cons_near_complete _ ts lo hi f Hin Hs)|apply term_regs_in;exact Ht].
Qed.

(* Constant recognition reads the register's range as const_of does. *)
Theorem known_operand_exact : forall F st o,
 known_operand F st o =
 match o with
 | OReg _ => match const_of F st o 64 false with Some c => OImm c | None => o end
 | _ => o
 end.
Proof.
 intros F st [r | z]; [|reflexivity].
 unfold known_operand, const_of, opnd_range.
 destruct (reg_range F st r) as [[lo hi]|]; [destruct (Z.eqb lo hi)|]; reflexivity.
Qed.
