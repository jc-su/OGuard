(*
  Soundness of the facts the checker gives the results of integer
  instructions: masks, disjoint ors, shifts, sums, and products.
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

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

(* ------------------------------------------------------------------ *)
(* Comparisons.                                                        *)

Lemma flip_eval : forall c a b, cmp_eval (flip c) a b = negb (cmp_eval c a b).
Proof.
  intros [] a b; cbn [flip cmp_eval];
    try (destruct (Z.ltb_spec a b), (Z.leb_spec b a); reflexivity || lia);
    try (destruct (Z.leb_spec a b), (Z.ltb_spec b a); reflexivity || lia).
Qed.

(* A comparison's negation holds exactly when the comparison fails. *)
Lemma negate_holds : forall rs k, cmp_holds rs (negate k) <-> ~ cmp_holds rs k.
Proof.
  intros rs k. unfold cmp_holds. cbn [negate cc cw csg ca cb]. rewrite flip_eval.
  destruct (cmp_eval (cc k) _ _); cbn [negb]; split; intros H; congruence.
Qed.

Lemma cmpf_eqb_eq : forall k l, cmpf_eqb k l = true -> k = l.
Proof.
  intros [c1 w1 s1 a1 b1] [c2 w2 s2 a2 b2] H. unfold cmpf_eqb in H. cbn [cc cw csg ca cb] in H.
  rewrite !andb_true_iff in H. destruct H as [[[[Hc Hw] Hs] Ha] Hb].
  apply Nat.eqb_eq in Hw. apply Bool.eqb_prop in Hs. apply Pos.eqb_eq in Ha. subst.
  assert (c1 = c2) by (destruct c1, c2; first [reflexivity | discriminate]). subst.
  destruct b1 as [r1 | z1], b2 as [r2 | z2]; try discriminate.
  - apply Pos.eqb_eq in Hb. subst. reflexivity.
  - apply Z.eqb_eq in Hb. subst. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* Bits.                                                               *)

Lemma testbit_small : forall v b i, 0 <= v < 2 ^ b -> 0 <= b <= i -> Z.testbit v i = false.
Proof.
  intros v b i Hv Hi. rewrite <- (Z.mod_small v (2 ^ b)) by lia.
  apply Z.mod_pow2_bits_high. lia.
Qed.

Lemma land_split : forall a b, Z.land a b + Z.ldiff a b = a.
Proof.
  intros a b. rewrite Z.add_comm. rewrite Z.add_nocarry_lxor.
  - rewrite Z.lxor_lor.
    + apply Z.lor_ldiff_and.
    + apply Z.bits_inj'. intros n Hn. rewrite Z.land_spec, Z.ldiff_spec, Z.land_spec, Z.bits_0.
      destruct (Z.testbit a n), (Z.testbit b n); reflexivity.
  - apply Z.bits_inj'. intros n Hn. rewrite Z.land_spec, Z.ldiff_spec, Z.land_spec, Z.bits_0.
    destruct (Z.testbit a n), (Z.testbit b n); reflexivity.
Qed.

Lemma ldiff_nonneg : forall a b, 0 <= a -> 0 <= Z.ldiff a b.
Proof. intros a b Ha. rewrite Z.ldiff_land. apply Z.land_nonneg. left. exact Ha. Qed.

Lemma land_bounds : forall a b, 0 <= a -> 0 <= Z.land a b <= a.
Proof.
  intros a b Ha. pose proof (land_split a b). pose proof (ldiff_nonneg a b Ha).
  split; [apply Z.land_nonneg; left; exact Ha | lia].
Qed.

Lemma land_le_r : forall a b, 0 <= b -> 0 <= Z.land a b <= b.
Proof. intros a b Hb. rewrite Z.land_comm. apply land_bounds. exact Hb. Qed.

Lemma lor_land : forall a b, Z.lor a b + Z.land a b = a + b.
Proof.
  intros a b.
  (* a = ldiff a b + land a b, and lor a b = ldiff a b + b without carries *)
  assert (H1 : Z.lor a b = Z.ldiff a b + b).
  { rewrite Z.add_nocarry_lxor.
    - rewrite Z.lxor_lor.
      + apply Z.bits_inj'. intros n Hn. rewrite Z.lor_spec, Z.lor_spec, Z.ldiff_spec.
        destruct (Z.testbit a n), (Z.testbit b n); reflexivity.
      + apply Z.bits_inj'. intros n Hn. rewrite Z.land_spec, Z.ldiff_spec, Z.bits_0.
        destruct (Z.testbit a n), (Z.testbit b n); reflexivity.
    - apply Z.bits_inj'. intros n Hn. rewrite Z.land_spec, Z.ldiff_spec, Z.bits_0.
      destruct (Z.testbit a n), (Z.testbit b n); reflexivity. }
  pose proof (land_split a b). lia.
Qed.

Lemma lor_disjoint : forall a b, Z.land a b = 0 -> Z.lor a b = a + b.
Proof. intros a b H. pose proof (lor_land a b). lia. Qed.

Lemma lor_bounds : forall a b, 0 <= a -> 0 <= b -> a <= Z.lor a b <= a + b.
Proof.
  intros a b Ha Hb. pose proof (lor_land a b). pose proof (land_le_r a b Hb). lia.
Qed.

Lemma land_low : forall x n, 0 <= n -> Z.land x (2 ^ n - 1) = x mod 2 ^ n.
Proof. intros x n Hn. rewrite <- Z.land_ones by exact Hn. rewrite Z.ones_equiv. f_equal; lia. Qed.

Lemma testbit_contig : forall s j i, 0 <= s <= j -> 0 <= i ->
    Z.testbit (2 ^ j - 2 ^ s) i = (s <=? i) && (i <? j).
Proof.
  intros s j i Hs Hi.
  assert (Hd : 2 ^ j - 2 ^ s = Z.ldiff (Z.ones j) (Z.ones s)).
  { rewrite <- Z.sub_nocarry_ldiff.
    - rewrite !Z.ones_equiv. lia.
    - apply Z.bits_inj'. intros n Hn. rewrite Z.ldiff_spec, !Z.testbit_ones by lia. rewrite Z.bits_0.
      destruct (Z.leb_spec 0 n), (Z.ltb_spec n s), (Z.ltb_spec n j); cbn; try reflexivity; lia. }
  rewrite Hd, Z.ldiff_spec, !Z.testbit_ones by lia.
  destruct (Z.leb_spec 0 i), (Z.ltb_spec i s), (Z.ltb_spec i j), (Z.leb_spec s i); cbn; try reflexivity; lia.
Qed.

(* A mask that keeps the bits [s, j): the field of those bits, in place. *)
Lemma land_contig : forall x s j, 0 <= s <= j ->
    Z.land x (2 ^ j - 2 ^ s) = 2 ^ s * ((x / 2 ^ s) mod 2 ^ (j - s)).
Proof.
  intros x s j Hs. apply Z.bits_inj'. intros i Hi.
  rewrite Z.land_spec, testbit_contig by lia.
  destruct (Z.leb_spec s i) as [Hsi | Hsi].
  - rewrite Z.mul_comm, Z.mul_pow2_bits by lia.
    rewrite Z.testbit_mod_pow2 by lia. rewrite Z.div_pow2_bits by lia.
    replace (i - s + s) with i by ring.
    destruct (Z.ltb_spec i j), (Z.ltb_spec (i - s) (j - s)), (Z.testbit x i); cbn; try reflexivity; lia.
  - rewrite Z.mul_comm, Z.mul_pow2_bits by lia. rewrite (Z.testbit_neg_r _ (i - s)) by lia.
    rewrite andb_false_r. reflexivity.
Qed.

(* x and k share a bit only at or above k's lowest set bit. *)
Lemma land_aligned : forall x k, 0 < k -> aligned (v2 k) (Z.land x k).
Proof.
  intros x k Hk. unfold aligned. rewrite Z.max_r by apply v2_nonneg.
  apply Z.mod_divide; [pose proof (pow_nonneg_pos (v2 k) (v2_nonneg k)); lia |].
  apply Z.bits_inj'. intros i Hi. rewrite Z.bits_0.
  destruct (Z.ltb_spec i (v2 k)) as [Hl | Hl].
  - rewrite Z.mod_pow2_bits_low by exact Hl. rewrite Z.land_spec.
    assert (Hb : Z.testbit k i = false).
    { destruct (v2_divides k) as [q Hq]. rewrite Hq, Z.mul_pow2_bits by apply v2_nonneg.
      apply Z.testbit_neg_r. lia. }
    rewrite Hb, andb_false_r. reflexivity.
  - apply Z.mod_pow2_bits_high. pose proof (v2_nonneg k). lia.
Qed.

Lemma bitlen_bound : forall k, 0 <= k -> k < 2 ^ bitlen k.
Proof.
  intros k Hk. unfold bitlen. destruct (Z.leb_spec k 0); [cbn; lia |].
  destruct (Z.log2_spec k ltac:(lia)) as [_ H2]. rewrite Z.pow_succ_r in H2 by apply Z.log2_nonneg.
  rewrite Z.pow_add_r by (try apply Z.log2_nonneg; lia). rewrite Z.pow_1_r. lia.
Qed.

Lemma bitlen_nonneg : forall k, 0 <= bitlen k.
Proof. intros k. unfold bitlen. destruct (Z.leb k 0); [lia | pose proof (Z.log2_nonneg k); lia]. Qed.

(* A value whose low t bits are at most y and a multiple of 2^a, and a k
   below 2^t with no bit from a to y's bit length: they share no bit. *)
Lemma land_zero_low : forall x k t y a,
    0 <= t -> 0 <= k < 2 ^ t -> 0 <= x mod 2 ^ t <= y -> 0 <= a <= bitlen y -> aligned a (x mod 2 ^ t) ->
    Z.land (2 ^ bitlen y - 2 ^ a) k = 0 -> Z.land x k = 0.
Proof.
  intros x k t y a Ht Hk Hr Ha Hal Hm.
  pose proof (bitlen_bound y ltac:(lia)) as Hby. pose proof (bitlen_nonneg y) as Hb0.
  set (b := bitlen y) in *.
  apply Z.bits_inj'. intros i Hi. rewrite Z.land_spec, Z.bits_0.
  destruct (Z.ltb_spec i t) as [Hit | Hit].
  - rewrite <- (Z.mod_pow2_bits_low x t i Hit).
    destruct (Z.ltb_spec i a) as [Hia | Hia].
    + (* below the alignment, the low bits have no bit *)
      unfold aligned in Hal. rewrite Z.max_r in Hal by lia. destruct Hal as [q Hq].
      rewrite Hq, Z.mul_pow2_bits by lia. rewrite Z.testbit_neg_r by lia. reflexivity.
    + destruct (Z.ltb_spec i b) as [Hib | Hib].
      * (* from the alignment to y's bit length, k has no bit *)
        assert (Hz : Z.testbit (Z.land (2 ^ b - 2 ^ a) k) i = false) by (rewrite Hm; apply Z.bits_0).
        rewrite Z.land_spec in Hz.
        replace (2 ^ b - 2 ^ a) with (Z.shiftl (Z.ones (b - a)) a) in Hz.
        2: { rewrite Z.shiftl_mul_pow2 by lia. rewrite Z.ones_equiv, <- Z.sub_1_r.
             rewrite Z.mul_sub_distr_r, Z.mul_1_l, <- Z.pow_add_r by lia.
             replace (b - a + a) with b by ring. reflexivity. }
        rewrite Z.shiftl_spec, Z.testbit_ones in Hz by lia.
        destruct (Z.leb_spec 0 (i - a)), (Z.ltb_spec (i - a) (b - a)); cbn in Hz; try lia.
        rewrite Hz, andb_false_r. reflexivity.
      * rewrite (testbit_small (x mod 2 ^ t) b i) by lia. reflexivity.
  - rewrite (testbit_small k t i) by lia. rewrite andb_false_r. reflexivity.
Qed.

Lemma contiguous_spec : forall k s j,
    contiguous k = Some (s, j) -> 0 < k /\ 0 <= s < j /\ k = 2 ^ j - 2 ^ s /\ s = v2 k.
Proof.
  intros k s j H. unfold contiguous in H.
  destruct (Z.leb_spec k 0); [discriminate |].
  destruct (Z.eqb (k + 2 ^ v2 k) (2 ^ Z.log2 (k + 2 ^ v2 k))) eqn:E1; [| discriminate].
  inversion H; subst s j. apply Z.eqb_eq in E1.
  pose proof (v2_nonneg k) as Hv. pose proof (pow_nonneg_pos (v2 k) Hv) as Hp.
  pose proof (Z.log2_nonneg (k + 2 ^ v2 k)) as Hl.
  split; [lia |]. split; [| split; [lia | reflexivity]].
  split; [exact Hv |].
  destruct (Z.lt_ge_cases (v2 k) (Z.log2 (k + 2 ^ v2 k))) as [Hlt | Hge]; [exact Hlt |].
  assert (2 ^ Z.log2 (k + 2 ^ v2 k) <= 2 ^ v2 k) by (apply Z.pow_le_mono_r; lia). lia.
Qed.

Lemma contiguous_top : forall k s j bw, contiguous k = Some (s, j) -> 0 <= bw -> k < 2 ^ bw -> j <= bw.
Proof.
  intros k s j bw H Hbw Hk. destruct (contiguous_spec k s j H) as [Hk0 [Hs [Hkj _]]].
  destruct (Z.le_gt_cases j bw) as [Hj | Hj]; [exact Hj |].
  assert (2 ^ bw <= 2 ^ (j - 1)) by (apply Z.pow_le_mono_r; lia).
  assert (2 ^ s <= 2 ^ (j - 1)) by (apply Z.pow_le_mono_r; lia).
  assert (2 ^ j = 2 * 2 ^ (j - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
  lia.
Qed.

(* The largest multiple of 2^s at most y bounds every such multiple. *)
Lemma aligned_below : forall s r y, 0 <= s -> aligned s r -> r <= y -> r <= y - y mod 2 ^ s.
Proof.
  intros s r y Hs [q Hq] Hr. rewrite Z.max_r in Hq by lia. subst r.
  pose proof (pow_nonneg_pos s Hs) as Hp.
  assert (Hm : y - y mod 2 ^ s = 2 ^ s * (y / 2 ^ s)) by (rewrite Z.mod_eq by lia; ring).
  rewrite Hm. assert (q <= y / 2 ^ s) by (apply Z.div_le_lower_bound; lia). nia.
Qed.

(* The value facts of a register's new value, which do not depend on any
   other register than through the atoms of a form. *)
Definition ffact_ok (wdn : reg -> nat) (rs : reg -> Z) (x : fact) (v : Z) : Prop :=
  match x with
  | FForm f al vr => wf f /\ sat wdn rs f v /\ aligned al v /\ (forall lo hi, vr = Some (lo, hi) -> lo <= v <= hi)
  | FAlign k => aligned k v
  | _ => False
  end.

Lemma lor_already_set : forall x k, Z.land (Z.lnot x) k = 0 -> Z.lor x k = x.
Proof.
  intros x k H. apply Z.bits_inj'. intros i Hi.
  assert (Hb : Z.testbit (Z.land (Z.lnot x) k) i = false) by (rewrite H; apply Z.bits_0).
  rewrite Z.land_spec, Z.lnot_spec in Hb by exact Hi.
  rewrite Z.lor_spec.
  destruct (Z.testbit x i), (Z.testbit k i); first [reflexivity | discriminate Hb].
Qed.

Lemma lor_add_missing : forall x k, Z.lor x k = x + Z.land (Z.lnot x) k.
Proof.
 intros x k. pose proof (lor_land x k) as H1. pose proof (land_split k x) as H2.
 rewrite Z.ldiff_land, (Z.land_comm k x), (Z.land_comm k (Z.lnot x)) in H2. lia.
Qed.

Lemma land_eqm_bounded : forall w x y k,
 0 <= w -> eqm w x y -> 0 <= k < 2^w -> Z.land x k = Z.land y k.
Proof.
 intros w x y k Hw He Hk. apply Z.bits_inj'. intros i Hi. rewrite !Z.land_spec.
 destruct (Z.lt_ge_cases i w) as [Hlo | Hhi].
 - rewrite <- (Z.mod_pow2_bits_low x w i), <- (Z.mod_pow2_bits_low y w i) by lia.
   unfold eqm in He. rewrite He. reflexivity.
 - rewrite (testbit_small k w i) by lia. rewrite !andb_false_r. reflexivity.
Qed.

Section Ops.

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

Let frs := form_range_sound F E HE t st rs Hst Hrs.
Let rrs := reg_range_sound F E HE t st rs Hst Hrs.
Let srs := sreg_range_sound F E HE t st rs Hst Hrs.
Let ras := reg_align_sound F E HE t st rs Hst Hrs.
Let fas := form_align_sound F E HE t st rs Hst Hrs.
Let rsr := rs_range F E rs Hrs.
Let wdr := wd_range F E HE.

Lemma range_form_ok : forall w lo hi v, 0 <= w <= 64 -> lo <= v <= hi -> wf (range_form w lo hi) /\ sat (range_form w lo hi) v.
Proof.
  intros w lo hi v Hw Hv. unfold range_form, wf. cbn. split; [lia |].
  exists v. cbn. split; [exact Hv | unfold eqm; f_equal; ring].
Qed.

Lemma atom_align_sound : forall x, aligned (atom_align F st x) (aval x).
Proof.
  intros [r e]. unfold atom_align. destruct e; cbn [fst snd atom_val]; try (apply aligned_nonpos; lia).
  - apply ras.
  - apply ras.
  - apply (sext_aligned F E HE rs Hrs). apply ras.
Qed.

Lemma split_at_sum : forall f n,
    let '(A, B, base) := split_at F st f n in
    tsum (ft f) = tsum A + tsum B /\ aligned n (tsum A) /\ aligned n base /\
    (flo f = fhi f -> base = flo f - flo f mod 2 ^ n) /\ (flo f <> fhi f -> base = 0).
Proof.
  intros f n. unfold split_at.
  split; [| split; [| split; [| split]]].
  - rewrite (tsum_filter wdn rs (fun t0 => let '(x, c) := t0 in n <=? v2 c + atom_align F st x) (ft f)).
    f_equal. f_equal. apply filter_ext. intros [x c]. cbn. rewrite Z.ltb_antisym. reflexivity.
  - induction (ft f) as [| [x c] ts IH]; cbn [filter tsum]; [apply aligned_zero |].
    destruct (n <=? v2 c + atom_align F st x) eqn:E1; cbn [tsum]; [| exact IH].
    apply aligned_add; [| exact IH]. apply Z.leb_le in E1.
    apply (aligned_mono (v2 c + atom_align F st x)); [lia |].
    apply aligned_mul; [apply v2_nonneg | apply v2_divides | apply atom_align_sound].
  - destruct (Z.eqb (flo f) (fhi f)); [| apply aligned_zero].
    destruct (Z.le_gt_cases n 0) as [Hn | Hn]; [apply aligned_nonpos; exact Hn |].
    unfold aligned. rewrite Z.max_r by lia. exists (flo f / 2 ^ n).
    pose proof (pow_nonneg_pos n ltac:(lia)). rewrite Z.mod_eq by lia. ring.
  - intros H. rewrite H, Z.eqb_refl. rewrite <- H. reflexivity.
  - intros H. destruct (Z.eqb_spec (flo f) (fhi f)); [contradiction | reflexivity].
Qed.

(* ------------------------------------------------------------------ *)
(* Masks.                                                              *)

(* The w-bit value a form allows, when its modulus covers w and the form's
   range is known, is at most the range's top. *)
Lemma value_le_top : forall fa x bw x0 y0,
    wf fa -> bw <= fm fa -> sat fa x -> 0 <= x < 2 ^ bw -> 0 <= bw ->
    form_range F st fa = Some (x0, y0) -> 0 <= x0 -> x <= y0.
Proof.
  intros fa x bw x0 y0 Hwf Hbw [e [He Hv]] Hx Hb Hr Hx0.
  pose proof (frs fa x0 y0 e Hr He) as Hb2.
  apply (eqm_weaken (fm fa) bw) in Hv; [| unfold wf in Hwf; lia].
  unfold eqm in Hv. rewrite Z.mod_small in Hv by lia.
  rewrite Hv. transitivity (tsum (ft fa) + e); [| lia].
  apply Z.mod_le; [lia | apply pow_nonneg_pos; lia].
Qed.

(* A form whose range lies within [0, 2^bw) is the w-bit value it allows. *)
Lemma value_eq_form : forall fa x bw e,
    wf fa -> bw <= fm fa -> 0 <= bw -> 0 <= x < 2 ^ bw ->
    flo fa <= e <= fhi fa -> eqm (fm fa) x (tsum (ft fa) + e) ->
    0 <= tsum (ft fa) + e < 2 ^ bw -> x = tsum (ft fa) + e.
Proof.
  intros fa x bw e Hwf Hbw Hb Hx He Hv Hr. unfold wf in Hwf.
  apply (eqm_small_eq (fm fa) bw); [lia | lia | lia | exact Hv].
Qed.

Lemma mask_reg_shape : forall fa a bw r,
    mask_reg F st fa a bw = r -> r = a \/
      (bw <= fm fa /\ flo fa = 0 /\ fhi fa = 0 /\ wd F r <= bw /\
       exists e, ft fa = [((r, e), 1)] /\ (e = EX \/ e = EZ)).
Proof.
  intros fa a bw r H. unfold mask_reg in H.
  destruct (negb (plain st a) && (bw <=? fm fa) && Z.eqb (flo fa) 0 && Z.eqb (fhi fa) 0) eqn:E1; [| left; congruence].
  rewrite !andb_true_iff in E1. destruct E1 as [[[_ E1] E2] E3].
  apply Z.leb_le in E1. apply Z.eqb_eq in E2. apply Z.eqb_eq in E3.
  destruct (ft fa) as [| [[r' ex] c] l] eqn:Et; [left; congruence |].
  destruct l as [| y l]; [| left; destruct ex; destruct c as [| cp | cp]; try destruct cp; congruence].
  destruct ex; destruct c as [| cp | cp]; try destruct cp; try (left; congruence);
    destruct (plain st r' && (wd F r' <=? bw)) eqn:E4; try (left; congruence);
    apply andb_true_iff in E4; destruct E4 as [_ E4]; apply Z.leb_le in E4; subst r; right;
    (split; [lia | split; [exact E2 | split; [exact E3 | split; [exact E4 |]]]]).
  - exists EX. split; [reflexivity | left; reflexivity].
  - exists EZ. split; [reflexivity | right; reflexivity].
Qed.

Lemma mask_reg_value : forall fa a bw x,
    wf fa -> sat fa x -> 0 <= bw -> 0 <= x < 2 ^ bw -> x = rs a -> x = rs (mask_reg F st fa a bw).
Proof.
  intros fa a bw x Hwf Hs Hb Hx Ha.
  destruct (mask_reg_shape fa a bw _ eq_refl) as [Heq | [H1 [H2 [H3 [H4 [e [H5 H6]]]]]]];
    [rewrite Heq; exact Ha |].
  set (r := mask_reg F st fa a bw) in *.
  destruct Hs as [e' [He' Hv]]. assert (e' = 0) by lia. subst e'.
  rewrite H5 in Hv. cbn [tsum] in Hv.
  assert (Hval : atom_val wdn rs (r, e) = rs r) by (destruct H6; subst e; reflexivity).
  rewrite Hval in Hv. pose proof (rsr r) as Hr. unfold wd in H4.
  assert (2 ^ Z.of_nat (wdn r) <= 2 ^ bw) by (apply Z.pow_le_mono_r; lia).
  apply (eqm_small_eq (fm fa) bw); [unfold wf in Hwf; lia | lia | lia |].
  eapply eqm_trans; [exact Hv | unfold eqm; f_equal; ring].
Qed.

Lemma mask_low_linear_sound : forall fa k bw n x,
    0 <= bw <= 64 -> 0 <= n <= 64 -> k = 2 ^ n - 1 -> k < 2 ^ bw ->
    wf fa -> sat fa x -> 0 <= x < 2 ^ bw -> ffact_ok wdn rs (mask_low_linear F st fa k bw n) (Z.land x k).
Proof.
  intros fa k bw n x Hbw Hn Hk Hkb Hwf Hs Hx.
  assert (Hpn : 0 < 2 ^ n) by (apply pow_nonneg_pos; lia).
  assert (HR : Z.land x k = x mod 2 ^ n) by (rewrite Hk; apply land_low; lia).
  pose proof (Z.mod_pos_bound x (2 ^ n) Hpn) as HRb.
  assert (Hrange : wf (range_form bw 0 k) /\ sat (range_form bw 0 k) (Z.land x k)).
  { apply range_form_ok; [lia | pose proof (land_le_r x k ltac:(lia)); lia]. }
  unfold mask_low_linear.
  assert (Hwhole : ffact_ok wdn rs
            (match mask_rg F st fa bw with
             | Some (x0, y0) => if (0 <=? x0) && (y0 <=? k) then FForm fa 0 None else FForm (range_form bw 0 k) 0 None
             | None => FForm (range_form bw 0 k) 0 None
             end) (Z.land x k)).
  { unfold mask_rg.
    destruct (bw <=? fm fa) eqn:Eb; [| cbn; split; [apply Hrange | split; [apply Hrange | split; [apply aligned_nonpos; lia | discriminate]]]].
    destruct (form_range F st fa) as [[x0 y0] |] eqn:Er;
      [| cbn; split; [apply Hrange | split; [apply Hrange | split; [apply aligned_nonpos; lia | discriminate]]]].
    destruct ((0 <=? x0) && (y0 <=? k)) eqn:Ec;
      [| cbn; split; [apply Hrange | split; [apply Hrange | split; [apply aligned_nonpos; lia | discriminate]]]].
    apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1, Ec2. apply Z.leb_le in Eb.
    destruct Hs as [e [He Hv]] eqn:Hs'.
    pose proof (frs fa x0 y0 e Er He) as Hb.
    assert (Hxe : x = tsum (ft fa) + e) by (apply (value_eq_form fa x bw e); [exact Hwf | lia | lia | lia | exact He | exact Hv | lia]).
    assert (Hxk : x mod 2 ^ n = x) by (apply Z.mod_small; lia).
    cbn. split; [exact Hwf | split; [| split; [apply aligned_nonpos; lia | discriminate]]].
    rewrite HR, Hxk. exists e. split; [exact He | exact Hv]. }
  pose proof (split_at_sum fa n) as Hsplit.
  destruct (split_at F st fa n) as [[A B] base] eqn:Es.
  destruct Hsplit as [Hsum [HA [Hbase [Hbase1 Hbase2]]]].
  destruct (if n <=? fm fa then form_range F st (mk 64 B (flo fa - base) (fhi fa - base)) else None)
    as [[p q] |] eqn:Er; [| exact Hwhole].
  destruct ((0 <=? p) && (q <=? k)) eqn:Ec; [| exact Hwhole].
  apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1, Ec2.
  destruct (n <=? fm fa) eqn:En; [| discriminate]. apply Z.leb_le in En.
  destruct Hs as [e [He Hv]].
  (* the B part with the constant less base is the value modulo 2^n *)
  set (e' := e - base).
  assert (He' : flo fa - base <= e' <= fhi fa - base) by (unfold e'; lia).
  destruct (sat_mk wdn rs 64 B (flo fa - base) (fhi fa - base) (tsum B + e') e' ltac:(lia) He' (eqm_refl _ _))
    as [e'' [He'' Hv'']]. rewrite mk_fm in Hv''.
  pose proof (frs _ p q e'' Er He'') as Hpq.
  set (V := tsum (ft (mk 64 B (flo fa - base) (fhi fa - base))) + e'') in *.
  assert (HVn : eqm n (tsum B + e') V) by (eapply eqm_weaken; [| exact Hv'']; lia).
  assert (HxB : eqm n x (tsum B + e')).
  { apply (eqm_weaken (fm fa) n) in Hv; [| unfold wf in Hwf; lia].
    eapply eqm_trans; [exact Hv |]. rewrite Hsum.
    unfold aligned in HA, Hbase. destruct (Z.le_gt_cases n 0) as [Hn0 | Hn0].
    - unfold eqm. assert (n = 0) by lia. subst n. cbn. rewrite !Z.mod_1_r. reflexivity.
    - rewrite Z.max_r in HA, Hbase by lia. destruct HA as [qa Hqa]. destruct Hbase as [qb Hqb].
      unfold eqm. replace (tsum A + tsum B + e) with (tsum B + e' + (qa + qb) * 2 ^ n)
        by (unfold e'; rewrite Hqa, Hqb; ring).
      rewrite Z.mod_add by lia. reflexivity. }
  assert (HxV : x mod 2 ^ n = V).
  { assert (Hm : eqm n x V) by (eapply eqm_trans; [exact HxB | exact HVn]). unfold eqm in Hm.
    rewrite Hm. apply Z.mod_small. lia. }
  cbn. unfold wf. rewrite mk_fm. split; [lia |].
  split; [| split; [apply aligned_nonpos; lia | discriminate]].
  rewrite HR, HxV. apply (sat_mk wdn rs bw B (flo fa - base) (fhi fa - base) V e'); [lia | exact He' |].
  apply (eqm_weaken 64 bw); [lia |]. apply eqm_sym. exact Hv''.
Qed.

Lemma mask_low_residue_sound : forall fa k bw n x,
    0 <= bw <= 64 -> 0 <= n <= 64 -> k = 2 ^ n - 1 -> k < 2 ^ bw ->
    wf fa -> sat fa x -> 0 <= x < 2 ^ bw -> ffact_ok wdn rs (mask_low_residue F st fa k bw n) (Z.land x k).
Proof.
  intros fa k bw n x Hbw Hn Hk Hkb Hwf Hs Hx.
  assert (Hpn : 0 < 2 ^ n) by (apply pow_nonneg_pos; lia).
  assert (HR : Z.land x k = x mod 2 ^ n) by (rewrite Hk; apply land_low; lia).
  pose proof (Z.mod_pos_bound x (2 ^ n) Hpn) as HRb.
  assert (Hrange : wf (range_form bw 0 k) /\ sat (range_form bw 0 k) (Z.land x k)).
  { apply range_form_ok; [lia | pose proof (land_le_r x k ltac:(lia)); lia]. }
  unfold mask_low_residue.
  assert (Hwhole : ffact_ok wdn rs
            (match mask_rg F st fa bw with
             | Some (x0, y0) => if (0 <=? x0) && (y0 <=? k) then FForm fa 0 None else FForm (range_form bw 0 k) 0 None
             | None => FForm (range_form bw 0 k) 0 None
             end) (Z.land x k)).
  { unfold mask_rg.
    destruct (bw <=? fm fa) eqn:Eb; [| cbn; split; [apply Hrange | split; [apply Hrange | split; [apply aligned_nonpos; lia | discriminate]]]].
    destruct (form_range F st fa) as [[x0 y0] |] eqn:Er;
      [| cbn; split; [apply Hrange | split; [apply Hrange | split; [apply aligned_nonpos; lia | discriminate]]]].
    destruct ((0 <=? x0) && (y0 <=? k)) eqn:Ec;
      [| cbn; split; [apply Hrange | split; [apply Hrange | split; [apply aligned_nonpos; lia | discriminate]]]].
    apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1, Ec2. apply Z.leb_le in Eb.
    destruct Hs as [e [He Hv]] eqn:Hs'.
    pose proof (frs fa x0 y0 e Er He) as Hb.
    assert (Hxe : x = tsum (ft fa) + e) by (apply (value_eq_form fa x bw e); [exact Hwf | lia | lia | lia | exact He | exact Hv | lia]).
    assert (Hxk : x mod 2 ^ n = x) by (apply Z.mod_small; lia).
    cbn. split; [exact Hwf | split; [| split; [apply aligned_nonpos; lia | discriminate]]].
    rewrite HR, Hxk. exists e. split; [exact He | exact Hv]. }
  pose proof (split_at_sum fa n) as Hsplit.
  destruct (split_at F st fa n) as [[A B] base] eqn:Es.
  destruct Hsplit as [Hsum [HA [Hbase [Hbase1 Hbase2]]]].
  destruct (if n <=? fm fa then form_range F st (mk n B (flo fa - base) (fhi fa - base)) else None)
    as [[p q] |] eqn:Er; [| exact Hwhole].
  destruct ((0 <=? p) && (q <=? k)) eqn:Ec; [| exact Hwhole].
  apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1, Ec2.
  destruct (n <=? fm fa) eqn:En; [| discriminate]. apply Z.leb_le in En.
  destruct Hs as [e [He Hv]].
  (* the B part with the constant less base is the value modulo 2^n *)
  set (e' := e - base).
  assert (He' : flo fa - base <= e' <= fhi fa - base) by (unfold e'; lia).
  destruct (sat_mk wdn rs n B (flo fa - base) (fhi fa - base) (tsum B + e') e' ltac:(lia) He' (eqm_refl _ _))
    as [e'' [He'' Hv'']]. rewrite mk_fm in Hv''.
  pose proof (frs _ p q e'' Er He'') as Hpq.
  set (V := tsum (ft (mk n B (flo fa - base) (fhi fa - base))) + e'') in *.
  assert (HVn : eqm n (tsum B + e') V) by exact Hv''.
  assert (HxB : eqm n x (tsum B + e')).
  { apply (eqm_weaken (fm fa) n) in Hv; [| unfold wf in Hwf; lia].
    eapply eqm_trans; [exact Hv |]. rewrite Hsum.
    unfold aligned in HA, Hbase. destruct (Z.le_gt_cases n 0) as [Hn0 | Hn0].
    - unfold eqm. assert (n = 0) by lia. subst n. cbn. rewrite !Z.mod_1_r. reflexivity.
    - rewrite Z.max_r in HA, Hbase by lia. destruct HA as [qa Hqa]. destruct Hbase as [qb Hqb].
      unfold eqm. replace (tsum A + tsum B + e) with (tsum B + e' + (qa + qb) * 2 ^ n)
        by (unfold e'; rewrite Hqa, Hqb; ring).
      rewrite Z.mod_add by lia. reflexivity. }
  assert (HxV : x mod 2 ^ n = V).
  { assert (Hm : eqm n x V) by (eapply eqm_trans; [exact HxB | exact HVn]). unfold eqm in Hm.
    rewrite Hm. apply Z.mod_small. lia. }
  cbn. unfold wf. rewrite mk_fm. split; [lia |].
  split; [| split; [apply aligned_nonpos; lia | discriminate]].
  rewrite HR, HxV. apply (sat_mk wdn rs bw
    (ft (mk n B (flo fa - base) (fhi fa - base)))
    (flo (mk n B (flo fa - base) (fhi fa - base)))
    (fhi (mk n B (flo fa - base) (fhi fa - base))) V e''); [lia | exact He'' |].
  unfold V. apply eqm_refl.
Qed.

Lemma mask_low_sound : forall fa k bw n x,
 0 <= bw <= 64 -> 0 <= n <= 64 -> k = 2 ^ n - 1 -> k < 2 ^ bw ->
 wf fa -> sat fa x -> 0 <= x < 2 ^ bw ->
 ffact_ok wdn rs (mask_low F st fa k bw n) (Z.land x k).
Proof.
 intros fa k bw n x Hbw Hn Hk Hkb Hwf Hsat Hx.
 pose proof (mask_low_linear_sound fa k bw n x Hbw Hn Hk Hkb Hwf Hsat Hx) as Hlinear.
 unfold mask_low.
 destruct (mask_low_linear F st fa k bw n) as [f al vr| | | ]; try exact Hlinear.
 destruct vr; [exact Hlinear|].
 destruct (null (ft f) && Z.eqb (flo f) 0 && Z.eqb (fhi f) k && (al <=? 0)); [|exact Hlinear].
 apply mask_low_residue_sound; assumption.
Qed.

Lemma land_contig_diff : forall x s j, 0 <= s <= j ->
    Z.land x (2 ^ j - 2 ^ s) = x mod 2 ^ j - x mod 2 ^ s.
Proof.
  intros x s j Hs. rewrite land_contig by exact Hs.
  assert (Hp : 0 < 2 ^ s) by (apply pow_nonneg_pos; lia).
  assert (Hq : 0 < 2 ^ (j - s)) by (apply pow_nonneg_pos; lia).
  replace (2 ^ j) with (2 ^ s * 2 ^ (j - s)) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
  rewrite Z.rem_mul_r by lia. ring.
Qed.

Section Mid.

Variables (fa : form) (k bw s j x : Z).
Hypotheses (Hbw : 0 <= bw <= 64) (Hs : 0 < s < j) (Hj : j <= bw) (Hk : k = 2 ^ j - 2 ^ s)
           (Hwf : wf fa) (Hsat : sat fa x) (Hx : 0 <= x < 2 ^ bw).

Lemma mid_facts :
  Z.land x k = x mod 2 ^ j - x mod 2 ^ s /\
  Z.land x k = 2 ^ s * ((x / 2 ^ s) mod 2 ^ (j - s)) /\
  0 <= Z.land x k <= k /\ Z.land x k <= x /\ aligned s (Z.land x k).
Proof.
  assert (Hps : 0 < 2 ^ s) by (apply pow_nonneg_pos; lia).
  assert (Hkpos : 0 < k) by (rewrite Hk; assert (2 ^ s < 2 ^ j) by (apply Z.pow_lt_mono_r; lia); lia).
  assert (HRf : Z.land x k = 2 ^ s * ((x / 2 ^ s) mod 2 ^ (j - s))) by (rewrite Hk; apply land_contig; lia).
  split; [rewrite Hk; apply land_contig_diff; lia |]. split; [exact HRf |].
  pose proof (land_le_r x k ltac:(lia)). pose proof (land_bounds x k ltac:(lia)).
  split; [lia |]. split; [lia |].
  unfold aligned. rewrite Z.max_r by lia. rewrite HRf. apply Z.divide_factor_l.
Qed.

Lemma mask_rel_sound : ffact_ok wdn rs (mask_rel F st fa k bw s j) (Z.land x k).
Proof.
  destruct mid_facts as [HR [HRf [HRk [HRx HRal]]]].
  assert (Hps : 0 < 2 ^ s) by (apply pow_nonneg_pos; lia).
  assert (Hpj : 0 < 2 ^ j) by (apply pow_nonneg_pos; lia).
  pose proof (Z.mod_pos_bound x (2 ^ s) Hps) as Hms.
  (* modulo 2^j, the result is x less its low s bits *)
  assert (Hrelj : sat (fshift (fat fa j) (- (2 ^ s - 1)) 0) (Z.land x k)).
  { assert (H1 : sat (fshift (fat fa j) (- (2 ^ s - 1)) 0) (x + - (x mod 2 ^ s))).
    { apply sat_fshift; [apply wf_fat; [exact Hwf | lia] | lia | apply sat_fat; [exact Hwf | lia | exact Hsat]]. }
    apply (sat_eqm wdn rs _ _ _ H1). unfold fshift, fat. rewrite !mk_fm.
    apply (eqm_weaken j); [unfold wf in Hwf; lia |]. rewrite HR.
    replace (x mod 2 ^ j - x mod 2 ^ s) with (x mod 2 ^ j + - (x mod 2 ^ s)) by ring.
    apply eqm_add; [lia | apply eqm_sym, eqm_mod; lia | apply eqm_refl]. }
  assert (Hwrel : wf (fshift (fat fa j) (- (2 ^ s - 1)) 0)) by (apply wf_fshift, wf_fat; [exact Hwf | lia]).
  assert (Hk0 : forall lo hi, (if j <? bw then Some (0, k) else None) = Some (lo, hi) -> lo <= Z.land x k <= hi)
    by (intros lo hi H; destruct (j <? bw); inversion H; subst lo hi; lia).
  unfold mask_rel. unfold mask_rg.
  destruct (bw <=? fm fa) eqn:Eb.
  2: { cbn. split; [exact Hwrel | split; [exact Hrelj | split; [exact HRal | exact Hk0]]]. }
  apply Z.leb_le in Eb.
  destruct (form_range F st fa) as [[x0 y0] |] eqn:Er.
  2: { cbn. split; [exact Hwrel | split; [exact Hrelj | split; [exact HRal | exact Hk0]]]. }
  assert (Hvr : forall lo hi,
             (if 0 <=? x0 then Some (0, if 0 <=? x0 then Z.min k (y0 - y0 mod 2 ^ s) else k)
              else if j <? bw then Some (0, k) else None) = Some (lo, hi) ->
             lo <= Z.land x k <= hi).
  { intros lo hi H. destruct (Z.leb_spec 0 x0) as [Hx0 | Hx0]; [| exact (Hk0 lo hi H)]. inversion H; subst lo hi.
    pose proof (value_le_top fa x bw x0 y0 Hwf Eb Hsat Hx ltac:(lia) Er Hx0) as Hxy.
    pose proof (aligned_below s (Z.land x k) y0 ltac:(lia) HRal ltac:(lia)). lia. }
  assert (Hjb : 2 ^ j <= 2 ^ bw) by (apply Z.pow_le_mono_r; lia).
  destruct ((0 <=? x0) && (y0 <? 2 ^ j)) eqn:Ec; cbn [andb].
  - apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2]. apply Z.leb_le in Ec1. apply Z.ltb_lt in Ec2.
    destruct Hsat as [e [He Hv]] eqn:Hs'.
    pose proof (frs fa x0 y0 e Er He) as Hb.
    assert (Hxe : x = tsum (ft fa) + e) by (apply (value_eq_form fa x bw e); [exact Hwf | lia | lia | lia | exact He | exact Hv | lia]).
    assert (Hxj : x mod 2 ^ j = x) by (apply Z.mod_small; lia).
    split; [apply wf_fshift; exact Hwf |].
    split; [| split; [exact HRal |]].
    + rewrite HR, Hxj. replace (x - x mod 2 ^ s) with (x + - (x mod 2 ^ s)) by ring.
      apply sat_fshift; [exact Hwf | lia | exact Hsat].
    + destruct (Z.eqb j bw); [intros lo hi Hn; discriminate | exact Hvr].
  - split; [exact Hwrel | split; [exact Hrelj | split; [exact HRal | exact Hvr]]].
Qed.

Lemma mask_shr_sound : forall fct, mask_shr F st fa bw s j = Some fct -> ffact_ok wdn rs fct (Z.land x k).
Proof.
  intros fct H. destruct mid_facts as [HR [HRf [HRk [HRx HRal]]]].
  unfold mask_shr in H. unfold mask_rg in H.
  destruct (ft fa) as [| [[r ex] c] l] eqn:Et; [discriminate |].
  destruct l as [| y l]; [| destruct (bw <=? fm fa); try destruct (form_range F st fa) as [[? ?] |]; discriminate].
  destruct (bw <=? fm fa) eqn:Eb; [| discriminate].
  destruct (form_range F st fa) as [[x0 y0] |] eqn:Er; [| discriminate].
  destruct (shr_of ex) as [kk |] eqn:Esh; [| discriminate].
  destruct (Z.eqb bw 64 && Z.eqb (flo fa) 0 && Z.eqb (fhi fa) 0 && (0 <=? kk) && (0 <? c) && Z.eqb c (2 ^ v2 c)
            && (v2 c <=? s) && (0 <=? x0) && (y0 <? 2 ^ j)) eqn:Ec; [| discriminate].
  inversion H; subst fct. clear H.
  rewrite !andb_true_iff in Ec. destruct Ec as [[[[[[[[E1 E2] E3] Ek] E4] E5] E6] E7] E8].
  apply Z.eqb_eq in E1, E2, E3, E5. apply Z.ltb_lt in E4, E8. apply Z.leb_le in Ek, E6, E7. apply Z.leb_le in Eb.
  destruct Hsat as [e [He Hv]]. assert (e = 0) by lia. subst e.
  pose proof (frs fa x0 y0 0 Er He) as Hb. rewrite Et in Hb, Hv. cbn [tsum] in Hb, Hv.
  rewrite Z.add_0_r in Hb, Hv. rewrite Z.add_0_r in Hb, Hv.
  assert (Hpj : 2 ^ j <= 2 ^ 64) by (apply Z.pow_le_mono_r; lia).
  pose proof Hx as Hx64. rewrite E1 in Hx64.
  assert (Hxe : x = c * atom_val wdn rs (r, ex)).
  { apply (eqm_small_eq (fm fa) 64); [unfold wf in Hwf; lia | lia | lia | exact Hv]. }
  (* the atom is the register shifted right by kk, which is 0 for the
     register itself *)
  assert (Hav : atom_val wdn rs (r, ex) = rs r / 2 ^ kk).
  { destruct ex; cbn [shr_of] in Esh; try discriminate; injection Esh as <-; cbn [atom_val].
    - rewrite Z.pow_0_r, Z.div_1_r. reflexivity.
    - reflexivity. }
  rewrite Hav in Hxe, Hb.
  pose proof (v2_nonneg c) as Hv2.
  assert (Hpk : 0 < 2 ^ kk) by (apply pow_nonneg_pos; lia).
  assert (Hps : 0 < 2 ^ s) by (apply pow_nonneg_pos; lia).
  assert (Hpd : 0 < 2 ^ (s - v2 c)) by (apply pow_nonneg_pos; lia).
  assert (Hpjs : 0 < 2 ^ (j - s)) by (apply pow_nonneg_pos; lia).
  assert (Hq : x / 2 ^ s = rs r / 2 ^ (kk + s - v2 c)).
  { rewrite Hxe, E5. replace (2 ^ s) with (2 ^ v2 c * 2 ^ (s - v2 c)) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
    rewrite <- E5. rewrite Z.div_mul_cancel_l by lia. rewrite Z.div_div by lia.
    rewrite <- Z.pow_add_r by lia. replace (kk + (s - v2 c)) with (kk + s - v2 c) by ring. reflexivity. }
  assert (Hsmall : x / 2 ^ s < 2 ^ (j - s)).
  { apply Z.div_lt_upper_bound; [lia |]. rewrite <- Z.pow_add_r by lia. replace (s + (j - s)) with j by ring. lia. }
  cbn [ffact_ok]. split; [unfold fatom, wf; rewrite mk_fm; lia |].
  split; [| split; [apply aligned_nonpos; lia | discriminate]].
  rewrite HRf, Z.mod_small by (split; [apply Z.div_pos; lia | exact Hsmall]).
  rewrite Hq. change (rs r / 2 ^ (kk + s - v2 c)) with (atom_val wdn rs (r, EShr (kk + s - v2 c))).
  apply sat_fatom. lia.
Qed.

Lemma mask_mid_sound : ffact_ok wdn rs (mask_mid F st fa k bw s j) (Z.land x k).
Proof.
  unfold mask_mid. destruct (mask_shr F st fa bw s j) as [fct |] eqn:E1.
  - apply mask_shr_sound. exact E1.
  - apply mask_rel_sound.
Qed.

End Mid.

Lemma mask_other_sound : forall fa k bw x,
    0 <= bw <= 64 -> 0 < k < 2 ^ bw -> wf fa -> sat fa x -> 0 <= x < 2 ^ bw ->
    ffact_ok wdn rs (mask_other F st fa k bw) (Z.land x k).
Proof.
  intros fa k bw x Hbw Hk Hwf Hsat Hx.
  pose proof (land_le_r x k ltac:(lia)) as HRk. pose proof (land_bounds x k ltac:(lia)) as HRx.
  pose proof (land_aligned x k ltac:(lia)) as HRal. pose proof (v2_nonneg k) as Hv.
  unfold mask_other, mask_rg.
  assert (Hr : forall top, 0 <= Z.land x k <= top -> ffact_ok wdn rs (FForm (range_form bw 0 top) (v2 k) None) (Z.land x k)).
  { intros top Htop. destruct (range_form_ok bw 0 top (Z.land x k) ltac:(lia) Htop) as [H1 H2].
    cbn. split; [exact H1 | split; [exact H2 | split; [exact HRal | discriminate]]]. }
  destruct (bw <=? fm fa) eqn:Eb; [| apply Hr; lia].
  apply Z.leb_le in Eb.
  destruct (form_range F st fa) as [[x0 y0] |] eqn:Er; [| apply Hr; lia].
  destruct (Z.leb_spec 0 x0) as [Hx0 | Hx0]; [| apply Hr; lia].
  apply Hr. pose proof (value_le_top fa x bw x0 y0 Hwf Eb Hsat Hx ltac:(lia) Er Hx0) as Hxy.
  pose proof (aligned_below (v2 k) (Z.land x k) y0 Hv HRal ltac:(lia)). lia.
Qed.

(* The fact of fa & k at width bw, where fa allows the w-bit value x of
   register a. *)
Lemma mask_sound : forall fa a k bw x fct,
    0 <= bw <= 64 -> 0 <= k < 2 ^ bw -> wf fa -> sat fa x -> 0 <= x < 2 ^ bw -> x = rs a ->
    mask F st fa a k bw = Some fct -> ffact_ok wdn rs fct (Z.land x k).
Proof.
  intros fa a k bw x fct Hbw Hk Hwf Hsat Hx Ha H. unfold mask in H.
  destruct (Z.eqb k 0 || (k <? 2 ^ reg_align F st a) || above_range F st a k) eqn:Hz.
  - assert (Hl : Z.land x k = 0).
    { apply orb_true_iff in Hz. destruct Hz as [Hz | Habove].
      2: { (* every bit x may have lies below k's bits *)
           unfold above_range in Habove.
           destruct (reg_range F st a) as [[lo hi] |] eqn:Er; [| discriminate].
           apply andb_true_iff in Habove. destruct Habove as [Hlo Hland].
           apply Z.leb_le in Hlo. apply Z.eqb_eq in Hland.
           pose proof (rrs a lo hi Er) as Hr. rewrite <- Ha in Hr.
           pose proof (bitlen_bound hi ltac:(lia)) as Hb. pose proof (bitlen_nonneg hi) as Hb0.
           rewrite <- (Z.mod_small x (2 ^ bitlen hi)) by lia. rewrite <- Z.land_ones by exact Hb0.
           rewrite <- Z.land_assoc, Hland. apply Z.land_0_r. }
      apply orb_true_iff in Hz. destruct Hz as [Hk0 | Hlt]; [apply Z.eqb_eq in Hk0; subst k; apply Z.land_0_r |].
      apply Z.ltb_lt in Hlt. set (al := reg_align F st a) in Hlt.
      assert (Hal : 0 <= al) by (destruct (Z.lt_ge_cases al 0) as [Hn |]; [rewrite Z.pow_neg_r in Hlt by lia; lia | lia]).
      pose proof (ras a) as Hx2. fold al in Hx2. rewrite <- Ha in Hx2. unfold aligned in Hx2.
      rewrite Z.max_r in Hx2 by lia.
      rewrite <- (Z.mod_small k (2 ^ al)) by lia. rewrite <- Z.land_ones by lia.
      rewrite (Z.land_comm k (Z.ones al)), Z.land_assoc, (Z.land_ones x al) by lia.
      rewrite (proj2 (Z.mod_divide x (2 ^ al) ltac:(lia)) Hx2). apply Z.land_0_l. }
    inversion H; subst fct. rewrite Hl. cbn. unfold fconst, wf. cbn.
    split; [lia | split; [apply sat_fconst; apply eqm_refl | split; [apply aligned_nonpos; lia | discriminate]]].
  - apply orb_false_iff in Hz. destruct Hz as [Hz _]. apply orb_false_iff in Hz. destruct Hz as [Hk0 _].
    apply Z.eqb_neq in Hk0.
    set (a' := mask_reg F st fa a bw) in H.
    assert (Ha' : x = rs a') by (apply mask_reg_value; [exact Hwf | exact Hsat | lia | exact Hx | exact Ha]).
    destruct (contiguous k) as [[s j] |] eqn:Ec.
    + destruct (contiguous_spec k s j Ec) as [Hk1 [Hsj [Hkj Hsv]]].
      pose proof (contiguous_top k s j bw Ec ltac:(lia) ltac:(lia)) as Hjb.
      (* a's bit field *)
      assert (Hbf : ffact_ok wdn rs (FForm (fatom bw (a', EBf s (j - s)) (2 ^ s)) 0 None) (Z.land x k)).
      { cbn [ffact_ok].
        split; [unfold fatom, wf; rewrite mk_fm; lia |].
        split; [| split; [apply aligned_nonpos; lia | discriminate]].
        rewrite Hkj, land_contig by lia. rewrite Ha'.
        change ((rs a' / 2 ^ s) mod 2 ^ (j - s)) with (atom_val wdn rs (a', EBf s (j - s))).
        apply sat_fatom. lia. }
      (* the bit field to a's width, above which a has no bit *)
      assert (Hbf2 : s < wd F a' ->
                ffact_ok wdn rs (FForm (fatom bw (a', EBf s (Z.min j (wd F a') - s)) (2 ^ s)) 0 None) (Z.land x k)).
      { intros Hsw. cbn [ffact_ok].
        split; [unfold fatom, wf; rewrite mk_fm; lia |].
        split; [| split; [apply aligned_nonpos; lia | discriminate]].
        rewrite Hkj, land_contig by lia. rewrite Ha'.
        assert (Hm : (rs a' / 2 ^ s) mod 2 ^ (j - s) = (rs a' / 2 ^ s) mod 2 ^ (Z.min j (wd F a') - s)).
        { pose proof (rsr a') as Hr. change (Z.of_nat (wdn a')) with (wd F a') in Hr.
          destruct (Z.le_ge_cases j (wd F a')) as [Hj | Hj]; [rewrite Z.min_l by lia; reflexivity |].
          rewrite Z.min_r by lia.
          assert (Hp : 0 < 2 ^ s) by (apply pow_nonneg_pos; lia).
          assert (Hq : 0 <= rs a' / 2 ^ s < 2 ^ (wd F a' - s)).
          { split; [apply Z.div_pos; lia |].
            apply Z.div_lt_upper_bound; [lia |]. rewrite <- Z.pow_add_r by lia.
            replace (s + (wd F a' - s)) with (wd F a') by lia. lia. }
          assert (2 ^ (wd F a' - s) <= 2 ^ (j - s)) by (apply Z.pow_le_mono_r; lia).
          rewrite !Z.mod_small by lia. reflexivity. }
        rewrite Hm.
        change ((rs a' / 2 ^ s) mod 2 ^ (Z.min j (wd F a') - s))
          with (atom_val wdn rs (a', EBf s (Z.min j (wd F a') - s))).
        apply sat_fatom. lia. }
      destruct (plain st a' && (s <? wd F a')) eqn:Ep.
      * inversion H; subst fct. apply andb_true_iff in Ep. apply Hbf2. apply Z.ltb_lt. exact (proj2 Ep).
      * destruct (Z.eqb_spec s 0) as [Hs0 | Hs0].
        -- destruct ((j <=? wd F a') &&
                     match mask_low F st fa k bw j with
                     | FForm f al None => null (ft f) && Z.eqb (flo f) 0 && Z.eqb (fhi f) (2 ^ j - 1) && (al <=? 0)
                     | _ => false
                     end);
             inversion H; subst fct; [exact Hbf |].
           rewrite Hs0, Z.pow_0_r in Hkj.
           apply mask_low_sound; [lia | lia | exact Hkj | lia | exact Hwf | exact Hsat | exact Hx].
        -- inversion H; subst fct. apply mask_mid_sound; [lia | lia | exact Hjb | exact Hkj | exact Hwf | exact Hsat | exact Hx].
    + inversion H; subst fct. apply mask_other_sound; [lia | lia | exact Hwf | exact Hsat | exact Hx].
Qed.

(* ------------------------------------------------------------------ *)
(* Operands.                                                           *)

Lemma operand_a : forall a bw, 0 <= bw <= 64 ->
    let fa := normalize F st (opnd_form F st (OReg a) bw) in wf fa /\ fm fa <= bw /\ sat fa (rs a).
Proof.
  intros a bw Hbw fa. destruct (opnd_form_sound F E HE t st rs Hst Hrs (OReg a) bw Hbw) as [Hs [Hw Hm]].
  cbn [opv] in Hs. destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hw Hs) as [H1 [H2 H3]].
  unfold fa. split; [exact H1 | split; [lia | exact H3]].
Qed.

Lemma operand_b : forall b bw, 0 <= bw <= 64 ->
    let fb := normalize F st (opnd_form F st b bw) in wf fb /\ fm fb <= bw /\ sat fb (opv rs b).
Proof.
  intros b bw Hbw fb. destruct (opnd_form_sound F E HE t st rs Hst Hrs b bw Hbw) as [Hs [Hw Hm]].
  destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hw Hs) as [H1 [H2 H3]].
  unfold fb. split; [exact H1 | split; [lia | exact H3]].
Qed.

Lemma const_of_sound : forall b bw k,
    0 <= bw -> const_of F st b bw false = Some k ->
    match b with OReg r => rs r = k | OImm z => k = z mod 2 ^ bw end.
Proof.
  intros [r | z] bw k Hbw H; unfold const_of, opnd_range in H.
  - destruct (reg_range F st r) as [[x y] |] eqn:Er; [| discriminate].
    destruct (Z.eqb_spec x y) as [Hxy |]; [| discriminate]. subst y. inversion H; subst x.
    pose proof (rrs r k k Er). lia.
  - cbn [extend] in H. rewrite Z.eqb_refl in H. inversion H. unfold trunc.
    rewrite Z2Nat.id by exact Hbw. reflexivity.
Qed.

Lemma form_align_nonpos : forall f, flo f <> fhi f -> form_align F st f <= 0.
Proof.
  intros f H. unfold form_align, form_align_with.
  assert (Hc : const_align f = 0) by (unfold const_align; destruct (Z.eqb_spec (flo f) (fhi f)); [contradiction | reflexivity]).
  rewrite Hc.
  assert (Hg : forall ts k, k <= 0 ->
            fold_left (fun k0 '(x, c) => Z.min k0 (v2 c + match snd x with EX | EZ | ES => ra F DEPTH st (fst x) | _ => 0 end)) ts k <= 0).
  { intros ts. induction ts as [| [x c] ts IH]; intros k Hk; cbn [fold_left]; [exact Hk |]. apply IH. lia. }
  apply Hg. lia.
Qed.

Lemma falign_sound : forall f v, wf f -> sat f v -> aligned (falign F st f) v.
Proof.
  intros f v Hwf [e [He Hv]]. unfold falign. unfold wf in Hwf.
  destruct (Z.eqb_spec (flo f) (fhi f)) as [Hlh | Hlh].
  - assert (e = flo f) by lia. subst e.
    apply (aligned_eqm (fm f) _ v (tsum (ft f) + flo f)); [lia | apply Z.le_min_r | exact Hv |].
    eapply aligned_mono; [apply Z.le_min_l | apply fas; [exact Hlh | lia]].
  - apply aligned_nonpos. pose proof (form_align_nonpos f Hlh). lia.
Qed.

Lemma land_aligned_small : forall T x y, 0 <= T -> aligned T x -> 0 <= y < 2 ^ T -> Z.land x y = 0.
Proof.
  intros T x y HT [q Hq] Hy. rewrite Z.max_r in Hq by lia.
  apply Z.bits_inj'. intros i Hi. rewrite Z.land_spec, Z.bits_0.
  destruct (Z.ltb_spec i T) as [HiT | HiT].
  - rewrite Hq, Z.mul_pow2_bits by lia. rewrite Z.testbit_neg_r by lia. reflexivity.
  - rewrite (testbit_small y T i) by lia. apply andb_false_r.
Qed.

Lemma sat_trunc : forall f w X, wf f -> fm f <= Z.of_nat w -> sat f X -> sat f (trunc w X).
Proof.
  intros f w X Hwf Hm Hs. apply (sat_eqm _ _ f X); [exact Hs |]. apply eqm_sym.
  apply (eqm_weaken (Z.of_nat w)); [unfold wf in Hwf; lia | apply trunc_eqm].
Qed.

Lemma aligned_trunc : forall k w X, k <= Z.of_nat w -> aligned k X -> aligned k (trunc w X).
Proof. intros k w X Hk H. apply (aligned_eqm (Z.of_nat w) k _ X); [lia | exact Hk | apply trunc_eqm | exact H]. Qed.

Lemma opnd_align_sound : forall b, aligned (opnd_align F st b) (opv rs b).
Proof.
  intros [r | z]; cbn [opnd_align opv]; [apply ras |].
  unfold aligned. rewrite Z.max_r by apply v2_nonneg. apply v2_divides.
Qed.

Definition form_ok (f : form) (v : Z) (bw : Z) : Prop := wf f /\ fm f <= bw /\ sat f v.

Lemma addsub_sound : forall w sg a b, 1 <= Z.of_nat w <= 64 -> (sg = 1 \/ sg = -1) ->
    ffact_ok wdn rs (addsub_fact F st sg (Z.of_nat w) a b) (trunc w (rs a + sg * opv rs b)).
Proof.
  intros w sg a b Hw Hsg. unfold addsub_fact. set (bw := Z.of_nat w) in *.
  assert (Hbw : 0 <= bw <= 64) by lia.
  destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
  destruct (operand_b b bw Hbw) as [Hwb [Hmb Hsb]].
  set (X := rs a + sg * opv rs b).
  assert (Hal : aligned (Z.min (Z.min (reg_align F st a) (opnd_align F st b)) bw) (trunc w X)).
  { apply aligned_trunc; [apply Z.le_min_r |]. unfold X. apply aligned_add.
    - eapply aligned_mono; [| apply ras]. lia.
    - apply (aligned_mono (0 + opnd_align F st b)); [lia |].
      apply aligned_mul; [lia | exists sg; lia | apply opnd_align_sound]. }
  assert (Hcomb : forall fa fb, form_ok fa (rs a) bw -> form_ok fb (opv rs b) bw ->
            ffact_ok wdn rs (FForm (fadd fa (fscale fb sg bw)) (Z.min (Z.min (reg_align F st a) (opnd_align F st b)) bw) None)
                     (trunc w X)).
  { intros fa fb [H1 [H2 H3]] [H4 [H5 H6]].
    assert (Hws : wf (fscale fb sg bw)) by (apply wf_fscale; [exact H4 | lia]).
    assert (Hs : sat (fscale fb sg bw) (sg * opv rs b)) by (apply sat_fscale; [exact H4 | lia | exact H6]).
    cbn [ffact_ok]. split; [apply wf_fadd; assumption |].
    split; [| split; [exact Hal | discriminate]].
    apply sat_trunc; [apply wf_fadd; assumption | unfold fadd; rewrite mk_fm; lia |].
    apply sat_fadd; assumption. }
  destruct (Z.eqb bw 64); [apply Hcomb; split; auto; split; auto |].
  set (ra := normalize F st (narrow st a bw true)).
  set (rb := normalize F st (match b with OImm _ => opnd_form F st b bw | OReg r => narrow st r bw true end)).
  assert (Hra : form_ok ra (rs a) bw).
  { destruct (narrow_sound F E HE t st rs Hst Hrs a bw true Hbw) as [H1 [H2 H3]].
    destruct (normalize_sound F E HE t st rs Hst Hrs _ _ H2 H1) as [H4 [H5 H6]].
    split; [exact H4 | split; [unfold ra; lia | exact H6]]. }
  assert (Hrb : form_ok rb (opv rs b) bw).
  { unfold rb. destruct b as [r | z].
    - destruct (narrow_sound F E HE t st rs Hst Hrs r bw true Hbw) as [H1 [H2 H3]].
      destruct (normalize_sound F E HE t st rs Hst Hrs _ _ H2 H1) as [H4 [H5 H6]].
      split; [exact H4 | split; [lia | exact H6]].
    - split; [exact Hwb | split; [exact Hmb | exact Hsb]]. }
  destruct (Nat.ltb (length (ft (fadd ra (fscale rb sg bw))))
                    (length (ft (fadd (normalize F st (opnd_form F st (OReg a) bw)) (fscale (normalize F st (opnd_form F st b bw)) sg bw)))));
    [apply Hcomb; assumption | apply Hcomb; split; auto; split; auto].
Qed.

Lemma fadd_fm : forall f g, fm (fadd f g) = Z.min (fm f) (fm g).
Proof. intros. unfold fadd. apply mk_fm. Qed.

Lemma fscale_fm_bounds : forall f k cap, 0 <= fm f -> 0 <= cap -> 0 <= fm (fscale f k cap) <= cap.
Proof. intros. unfold fscale. rewrite mk_fm. destruct (Z.eqb k 0); [lia | pose proof (v2_nonneg k); lia]. Qed.

Lemma aligned_mul2 : forall i j x y, aligned i x -> aligned j y -> aligned (i + j) (x * y).
Proof.
  intros i j x y Hx Hy. apply (aligned_mono (Z.max 0 i + Z.max 0 j)); [lia |].
  unfold aligned in *. rewrite (Z.max_r 0 (Z.max 0 i + Z.max 0 j)) by lia.
  destruct Hx as [p Hp]. destruct Hy as [q Hq]. exists (p * q).
  rewrite Hp, Hq, Z.pow_add_r by lia. ring.
Qed.

Lemma range_fact_ok : forall w lo hi al v,
    0 <= Z.of_nat w <= 64 -> lo <= v <= hi -> aligned al (trunc w v) ->
    ffact_ok wdn rs (FForm (range_form (Z.of_nat w) lo hi) al None) (trunc w v).
Proof.
  intros w lo hi al v Hw Hv Hal. cbn [ffact_ok].
  destruct (range_form_ok (Z.of_nat w) lo hi v Hw Hv) as [H1 H2].
  split; [exact H1 | split; [| split; [exact Hal | discriminate]]].
  apply sat_trunc; [exact H1 | unfold range_form; cbn; lia | exact H2].
Qed.

Lemma mul_sound : forall w a b fct,
    1 <= Z.of_nat w <= 64 -> (forall r, b = OReg r -> wd F r = Z.of_nat w) ->
    mul_fact F st (Z.of_nat w) a b = Some fct -> ffact_ok wdn rs fct (trunc w (rs a * opv rs b)).
Proof.
  intros w a b fct Hw Hb H. unfold mul_fact in H. set (bw := Z.of_nat w) in *.
  assert (Hbw : 0 <= bw <= 64) by lia.
  destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
  (* the product modulo 2^bw depends on b's value modulo 2^bw only *)
  assert (Hbv : forall v, eqm bw (opv rs b) v -> eqm bw (rs a * opv rs b) (rs a * v)).
  { intros v Hv. apply eqm_mul; [lia | exact Hv]. }
  destruct (const_of F st b bw false) as [k |] eqn:Ec.
  - inversion H; subst fct. pose proof (const_of_sound b bw k ltac:(lia) Ec) as Hk.
    assert (Hkm : eqm bw (opv rs b) (k mod 2 ^ bw)).
    { destruct b as [r | z]; cbn [opv]; [rewrite Hk; apply eqm_sym, eqm_mod; lia | rewrite Hk; apply eqm_sym; rewrite Z.mod_mod by (apply Z.pow_nonzero; lia); apply eqm_mod; lia]. }
    cbn [ffact_ok]. assert (Hws : wf (fscale (normalize F st (opnd_form F st (OReg a) bw)) (k mod 2 ^ bw) bw))
      by (apply wf_fscale; [exact Hwa | lia]).
    split; [exact Hws |]. split; [| split; [apply aligned_nonpos; lia | discriminate]].
    apply (sat_eqm _ _ _ ((k mod 2 ^ bw) * rs a)); [apply sat_fscale; [exact Hwa | lia | exact Hsa] |].
    apply (eqm_weaken bw); [apply fscale_fm_bounds; [exact (proj1 Hwa) | lia] |].
    eapply eqm_trans; [| apply eqm_sym, trunc_eqm]. rewrite Z.mul_comm. apply eqm_sym. apply Hbv. exact Hkm.
  - destruct (const_of F st (OReg a) bw false) as [k |] eqn:Eca.
    { (* a is the constant: the product is k times b's form *)
      inversion H; subst fct. pose proof (const_of_sound (OReg a) bw k ltac:(lia) Eca) as Hk.
      destruct (operand_b b bw Hbw) as [Hwb [Hmb Hsb]].
      cbn [ffact_ok]. assert (Hws : wf (fscale (normalize F st (opnd_form F st b bw)) (k mod 2 ^ bw) bw))
        by (apply wf_fscale; [exact Hwb | lia]).
      split; [exact Hws |]. split; [| split; [apply aligned_nonpos; lia | discriminate]].
      apply (sat_eqm _ _ _ ((k mod 2 ^ bw) * opv rs b)); [apply sat_fscale; [exact Hwb | lia | exact Hsb] |].
      apply (eqm_weaken bw); [apply fscale_fm_bounds; [exact (proj1 Hwb) | lia] |].
      eapply eqm_trans; [| apply eqm_sym, trunc_eqm]. rewrite (Z.mul_comm (rs a)).
      apply eqm_sym. rewrite (Z.mul_comm (k mod 2 ^ bw)). apply eqm_mul; [lia |].
      rewrite Hk. apply eqm_sym, eqm_mod. lia. }
    set (al := Z.min bw (reg_align F st a + match b with OReg r => reg_align F st r | OImm _ => 0 end)) in *.
    (* b's value as the product reads it, modulo 2^bw *)
    set (bv := match b with OReg r => rs r | OImm z => trunc w z end).
    assert (Hbv' : eqm bw (rs a * opv rs b) (rs a * bv)).
    { apply Hbv. unfold bv. destruct b as [r | z]; cbn [opv]; [apply eqm_refl | apply eqm_sym, trunc_eqm]. }
    assert (Htr : trunc w (rs a * opv rs b) = trunc w (rs a * bv)) by (unfold trunc; exact Hbv').
    rewrite Htr.
    assert (Hal : aligned al (trunc w (rs a * bv))).
    { apply aligned_trunc; [unfold al; lia |]. apply (aligned_mono (reg_align F st a + match b with OReg r => reg_align F st r | OImm _ => 0 end)); [unfold al; lia |].
      apply aligned_mul2; [apply ras |]. unfold bv. destruct b; [apply ras | apply aligned_nonpos; lia]. }
    assert (Hfalign : 0 < al -> ffact_ok wdn rs (FAlign al) (trunc w (rs a * bv))) by (intros; exact Hal).
    destruct (reg_range F st a) as [[a0 a1] |] eqn:Ea;
      [| destruct (0 <? al) eqn:Ea0; inversion H; subst fct; apply Hfalign; apply Z.ltb_lt; exact Ea0].
    destruct (opnd_range F st b bw false) as [[b0 b1] |] eqn:Eb;
      [| destruct (0 <? al) eqn:Ea0; inversion H; subst fct; apply Hfalign; apply Z.ltb_lt; exact Ea0].
    destruct ((0 <=? a0) && (0 <=? b0) && (a1 * b1 <? 2 ^ bw)) eqn:Er;
      [| destruct (0 <? al) eqn:Ea0; inversion H; subst fct; apply Hfalign; apply Z.ltb_lt; exact Ea0].
    inversion H; subst fct. rewrite !andb_true_iff in Er. destruct Er as [[E1 E2] E3].
    apply Z.leb_le in E1, E2. apply Z.ltb_lt in E3.
    pose proof (rrs a a0 a1 Ea) as Ha.
    assert (Hbr : b0 <= bv <= b1).
    { unfold bv. destruct b as [r | z]; unfold opnd_range in Eb.
      - exact (rrs r b0 b1 Eb).
      - cbn [extend] in Eb. unfold bw in Eb. rewrite Nat2Z.id in Eb. inversion Eb. lia. }
    assert (Hp : a0 * b0 <= rs a * bv <= a1 * b1) by nia.
    apply range_fact_ok; [lia | exact Hp | exact Hal].
Qed.

Lemma lor_small : forall x y n, 0 <= n -> 0 <= x < 2 ^ n -> 0 <= y < 2 ^ n -> 0 <= Z.lor x y < 2 ^ n.
Proof.
  intros x y n Hn Hx Hy. split; [apply Z.lor_nonneg; lia |].
  assert (H : Z.lor x y = Z.lor x y mod 2 ^ n).
  { symmetry. apply Z.bits_inj'. intros i Hi. rewrite Z.testbit_mod_pow2 by lia. rewrite Z.lor_spec.
    destruct (Z.ltb_spec i n); [reflexivity |]. rewrite (testbit_small x n i), (testbit_small y n i) by lia. reflexivity. }
  rewrite H. apply Z.mod_pos_bound. apply pow_nonneg_pos. lia.
Qed.

Lemma lxor_small : forall x y n, 0 <= n -> 0 <= x < 2 ^ n -> 0 <= y < 2 ^ n -> 0 <= Z.lxor x y < 2 ^ n.
Proof.
  intros x y n Hn Hx Hy. split; [apply Z.lxor_nonneg; lia |].
  assert (H : Z.lxor x y = Z.lxor x y mod 2 ^ n).
  { symmetry. apply Z.bits_inj'. intros i Hi. rewrite Z.testbit_mod_pow2 by lia. rewrite Z.lxor_spec.
    destruct (Z.ltb_spec i n); [reflexivity |]. rewrite (testbit_small x n i), (testbit_small y n i) by lia. reflexivity. }
  rewrite H. apply Z.mod_pos_bound. apply pow_nonneg_pos. lia.
Qed.

(* A power of two that divides both values divides their or and their xor. *)
Lemma aligned_lor : forall k x y, aligned k x -> aligned k y -> aligned k (Z.lor x y).
Proof.
  intros k x y [p Hp] [q Hq]. assert (HK : 0 <= Z.max 0 k) by lia.
  exists (Z.lor p q). rewrite Hp, Hq, <- !Z.shiftl_mul_pow2 by exact HK. rewrite <- Z.shiftl_lor.
  reflexivity.
Qed.

Lemma aligned_lxor : forall k x y, aligned k x -> aligned k y -> aligned k (Z.lxor x y).
Proof.
  intros k x y [p Hp] [q Hq]. assert (HK : 0 <= Z.max 0 k) by lia.
  exists (Z.lxor p q). rewrite Hp, Hq, <- !Z.shiftl_mul_pow2 by exact HK. rewrite <- Z.shiftl_lxor.
  reflexivity.
Qed.

(* The terms whose values are multiples of 2^n sum to one. *)
Lemma filter_aligned : forall n ts,
    aligned n (tsum (filter (fun u => let '(x1, c) := u in n <=? v2 c + atom_align F st x1) ts)).
Proof.
  intros n ts. induction ts as [| [x1 c] ts IH]; cbn [filter tsum]; [apply aligned_zero |].
  destruct (n <=? v2 c + atom_align F st x1) eqn:E1; cbn [tsum]; [| exact IH].
  apply aligned_add; [| exact IH]. apply Z.leb_le in E1.
  apply (aligned_mono (v2 c + atom_align F st x1)); [lia |].
  apply aligned_mul; [apply v2_nonneg | apply v2_divides | apply atom_align_sound].
Qed.

(* fa | k for a constant k: fa + k when fa has none of k's bits, else in
   [fa, fa + k]. *)
(* Bit disjointness depends only on the form modulo its known width;
   the represented integer may itself be negative, as for a complement. *)
Lemma low_bits_clear_disjoint : forall fa x k,
    wf fa -> sat fa x -> low_bits_clear F st fa k = true -> Z.land x k = 0.
Proof.
  intros fa x k Hwf Hs El.
  pose proof Hwf as Hwf0. unfold wf in Hwf0.
    unfold low_bits_clear in El. cbv zeta in El.
    destruct (Z.leb k 0 || Z.ltb (fm fa) (bitlen k) || negb (Z.eqb (flo fa) (fhi fa))) eqn:E1; [discriminate |].
    rewrite !orb_false_iff in E1. destruct E1 as [[E1 E2] E3].
    apply Z.leb_gt in E1. apply Z.ltb_ge in E2. apply negb_false_iff, Z.eqb_eq in E3.
    set (tb := bitlen k) in *.
    set (rest := filter (fun u => let '(x0, c) := u in v2 c + atom_align F st x0 <? tb) (ft fa)) in *.
    destruct (form_range F st (mk 64 rest (flo fa) (flo fa))) as [[x0 y0] |] eqn:Er; [| discriminate].
    rewrite !andb_true_iff in El. destruct El as [[E4 E5] E6].
    apply Z.leb_le in E4. apply Z.ltb_lt in E5. apply Z.eqb_eq in E6.
    pose proof (bitlen_bound k ltac:(lia)) as Hkt. pose proof (bitlen_nonneg k) as Htb. fold tb in Hkt, Htb.
    destruct Hs as [e [He Hv]]. assert (e = flo fa) by lia. subst e.
    (* x's low tb bits are those of the rest *)
    rewrite (tsum_filter wdn rs (fun u => let '(x1, c) := u in tb <=? v2 c + atom_align F st x1) (ft fa)) in Hv.
    set (A := filter (fun u => let '(x1, c) := u in tb <=? v2 c + atom_align F st x1) (ft fa)) in Hv.
    replace (filter (fun t0 => negb (let '(x1, c) := t0 in tb <=? v2 c + atom_align F st x1)) (ft fa)) with rest in Hv
      by (unfold rest; apply filter_ext; intros [x1 c]; rewrite Z.ltb_antisym; reflexivity).
    assert (HA : aligned tb (tsum A)) by apply filter_aligned.
    destruct (sat_mk wdn rs 64 rest (flo fa) (flo fa) (tsum rest + flo fa) (flo fa) ltac:(lia) ltac:(lia) (eqm_refl _ _))
      as [e' [He' Hv']]. rewrite mk_fm in Hv'.
    pose proof (frs _ x0 y0 e' Er He') as Hb.
    assert (Hlow : x mod 2 ^ tb = tsum (ft (mk 64 rest (flo fa) (flo fa))) + e').
    { assert (Hx1 : eqm tb x (tsum rest + flo fa)).
      { apply (eqm_weaken (fm fa) tb) in Hv; [| unfold wf in Hwf; lia]. eapply eqm_trans; [exact Hv |].
        unfold aligned in HA. destruct (Z.le_gt_cases tb 0) as [Ht0 | Ht0].
        - assert (tb = 0) by lia. unfold eqm. rewrite H. cbn. rewrite !Z.mod_1_r. reflexivity.
        - rewrite Z.max_r in HA by lia. destruct HA as [qa Hqa]. unfold eqm.
          replace (tsum A + tsum rest + flo fa) with (tsum rest + flo fa + qa * 2 ^ tb) by (rewrite Hqa; ring).
          rewrite Z.mod_add by (apply Z.pow_nonzero; lia). reflexivity. }
      assert (Hx2 : eqm tb x (tsum (ft (mk 64 rest (flo fa) (flo fa))) + e')).
      { eapply eqm_trans; [exact Hx1 |]. apply (eqm_weaken 64); [lia | exact Hv']. }
      unfold eqm in Hx2. rewrite Hx2. apply Z.mod_small. lia. }
    (* the low bits are a multiple of 2^a for the rest's alignment a *)
    set (RF := mk 64 rest (flo fa) (flo fa)) in *.
    assert (Hal : aligned (Z.min (Z.max 0 (falign F st RF)) (bitlen y0)) (x mod 2 ^ tb)).
    { apply (aligned_mono (Z.max 0 (falign F st RF))); [lia |].
      assert (Hwrf : wf RF) by (unfold wf, RF; rewrite mk_fm; lia).
      assert (Hsrf : sat RF (x mod 2 ^ tb)) by (exists e'; split; [exact He' | rewrite Hlow; apply eqm_refl]).
      pose proof (falign_sound RF (x mod 2 ^ tb) Hwrf Hsrf) as H0. unfold aligned in *.
      replace (Z.max 0 (Z.max 0 (falign F st RF))) with (Z.max 0 (falign F st RF)) by lia. exact H0. }
    pose proof (bitlen_nonneg y0) as Hy0.
    apply (land_zero_low x k tb y0 (Z.min (Z.max 0 (falign F st RF)) (bitlen y0)));
        [lia | lia | rewrite Hlow; lia | lia | exact Hal | exact E6].
Qed.



Lemma or_complement_sound : forall fa x k bw fct,
  0 <= bw <= 64 -> wf fa -> sat fa x -> 0 <= k < 2^bw ->
  or_complement F st fa k = Some fct -> ffact_ok wdn rs fct (Z.lor x k).
Proof.
 intros fa x k bw fct Hbw Hwf Hs Hk Hc. unfold or_complement in Hc.
 set (nf := normalize F st (fshift (fscale fa (-1) 64) (-1) (-1))) in *.
 assert (Hwn0 : wf (fshift (fscale fa (-1) 64) (-1) (-1)))
   by (apply wf_fshift, wf_fscale; [exact Hwf | lia]).
 assert (Hsn0 : sat (fshift (fscale fa (-1) 64) (-1) (-1)) (Z.lnot x)).
 { replace (Z.lnot x) with ((-1)*x+(-1)) by (pose proof (Z.add_lnot_diag x); lia).
   apply sat_fshift; [apply wf_fscale; [exact Hwf | lia] | lia |].
   apply sat_fscale; [exact Hwf | lia | exact Hs]. }
 destruct (normalize_sound F E HE t st rs Hst Hrs _ (Z.lnot x) Hwn0 Hsn0) as [Hwn [_ Hsn]]. fold nf in Hwn,Hsn.
 destruct (negb (Z.eqb (fm nf) 64 && Z.eqb (flo nf) 0 && Z.eqb (fhi nf) 0)) eqn:En; [discriminate |].
 apply negb_false_iff in En. rewrite !andb_true_iff in En. destruct En as [[Hm Hl] Hh].
 apply Z.eqb_eq in Hm,Hl,Hh.
 destruct (ft nf) as [|[[r e] c] ts] eqn:Et; [discriminate |].
 destruct ts; [|discriminate].
 destruct (Z.eqb c 1 && match e with EX | EZ => true | _ => false end) eqn:Ec; [|discriminate].
 apply andb_true_iff in Ec. destruct Ec as [Hc1 He]. apply Z.eqb_eq in Hc1. subst c.
 assert (Ha : atom_val wdn rs (r,e) = rs r) by (destruct e; first [reflexivity | discriminate He]).
 assert (Hq : eqm 64 (Z.lnot x) (rs r)).
 { destruct Hsn as [v [Hv Hq]]. rewrite Hl,Hh in Hv. assert (v=0) by lia. subst v.
   rewrite Hm,Et in Hq. cbn [tsum] in Hq. rewrite Ha in Hq.
   replace (1*rs r+0+0) with (rs r) in Hq by ring. exact Hq. }
 destruct (contiguous (k mod 2^wd F r)) as [[s j]|] eqn:Econt; [|discriminate].
 injection Hc as <-.
 pose proof (wdr r) as Hwr. pose proof (rsr r) as Hrr.
 assert (Hwidth : 0 <= wd F r) by (unfold wd; lia).
 change (0 <= rs r < 2^wd F r) in Hrr.
 assert (Hk64 : 0 <= k < 2^64).
 { assert (2^bw <= 2^64) by (apply Z.pow_le_mono_r; lia). lia. }
 assert (Hmissing : Z.land (Z.lnot x) k = Z.land (rs r) (k mod 2^wd F r)).
 { rewrite (land_eqm_bounded 64 _ _ k ltac:(lia) Hq Hk64).
   rewrite (Z.land_comm (rs r) k), (Z.land_comm (rs r) (k mod 2^wd F r)).
   apply (land_eqm_bounded (wd F r)); [exact Hwidth | apply eqm_sym, eqm_mod; exact Hwidth | exact Hrr]. }
 destruct (contiguous_spec _ s j Econt) as [_ [Hsj [Hmask _]]].
 rewrite Hmask, land_contig in Hmissing by lia.
 rewrite lor_add_missing, Hmissing.
 cbn [ffact_ok]. assert (Hwb : wf (fatom 64 (r,EBf s (j-s)) (2^s)))
   by (unfold wf,fatom; rewrite mk_fm; lia).
 split; [apply wf_fadd; assumption |]. split.
 - apply sat_fadd; [exact Hwf | exact Hwb | exact Hs |].
   change (sat (fatom 64 (r,EBf s (j-s)) (2^s)) ((2^s) * atom_val wdn rs (r,EBf s (j-s)))).
   apply sat_fatom. lia.
 - split; [apply aligned_nonpos; lia | discriminate].
Qed.

Lemma or_const_sound : forall fa x k bw,
    0 <= bw <= 64 -> wf fa -> fm fa <= bw -> sat fa x -> 0 <= x < 2 ^ bw -> 0 <= k < 2 ^ bw ->
    ffact_ok wdn rs (or_const F st fa k) (Z.lor x k).
Proof.
  intros fa x k bw Hbw Hwf Hm Hs Hx Hk. unfold or_const.
  destruct (low_bits_clear F st fa k) eqn:El.
  - rewrite (lor_disjoint x k (low_bits_clear_disjoint fa x k Hwf Hs El)). cbn [ffact_ok].
    split; [apply wf_fshift; exact Hwf |].
    split; [apply sat_fshift; [exact Hwf | lia | exact Hs] |].
    split; [apply aligned_nonpos; lia | discriminate].
  - set (nf := fshift (fscale fa (-1) 64) (-1) (-1)).
    destruct (low_bits_clear F st (normalize F st nf) k) eqn:En.
    + assert (Hwscale : wf (fscale fa (-1) 64)) by (apply wf_fscale; [exact Hwf | lia]).
      assert (Hwn : wf nf) by (apply wf_fshift; exact Hwscale).
      assert (Hsn : sat nf (Z.lnot x)).
      { replace (Z.lnot x) with ((-1) * x + (-1)) by (pose proof (Z.add_lnot_diag x); lia).
        apply sat_fshift; [exact Hwscale | lia |]. apply sat_fscale; [exact Hwf | lia | exact Hs]. }
      destruct (normalize_sound F E HE t st rs Hst Hrs nf (Z.lnot x) Hwn Hsn) as [Hwn' [_ Hsn']].
      pose proof (low_bits_clear_disjoint (normalize F st nf) (Z.lnot x) k Hwn' Hsn' En) as Hz.
      rewrite (lor_already_set x k Hz). cbn [ffact_ok].
      split; [exact Hwf |]. split; [exact Hs |]. split; [apply aligned_nonpos; lia | discriminate].
    + destruct (or_complement F st fa k) as [fct |] eqn:Ec.
      * exact (or_complement_sound fa x k bw fct Hbw Hwf Hs Hk Ec).
      * pose proof (lor_bounds x k ltac:(lia) ltac:(lia)) as Hb.
        cbn [ffact_ok]. split; [apply wf_fshift; exact Hwf |].
        split; [| split; [apply aligned_nonpos; lia | discriminate]].
      replace (Z.lor x k) with (x + (Z.lor x k - x)) by ring.
      apply sat_fshift; [exact Hwf | lia | exact Hs].
Qed.

Lemma trunc_reg : forall r w, wd F r = Z.of_nat w -> trunc w (rs r) = rs r.
Proof. intros r w H. apply trunc_small. pose proof (rsr r). unfold wd in H. rewrite <- H. lia. Qed.

Lemma or_fact_sound : forall w a b fct,
    1 <= Z.of_nat w <= 64 -> wd F a = Z.of_nat w -> (forall r, b = OReg r -> wd F r = Z.of_nat w) ->
    or_fact F st (Z.of_nat w) a b = Some fct ->
    ffact_ok wdn rs fct (trunc w (Z.lor (trunc w (rs a)) (trunc w (opv rs b)))).
Proof.
  intros w a b fct Hw Ha Hb H. unfold or_fact in H. set (bw := Z.of_nat w) in *.
  assert (Hbw : 0 <= bw <= 64) by lia.
  destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
  rewrite (trunc_reg a w Ha).
  pose proof (rsr a) as Hra. unfold wd in Ha. rewrite Ha in Hra. fold bw in Hra.
  assert (Hty : forall y, 0 <= trunc w y < 2 ^ bw) by (intros; apply trunc_range).
  assert (Hsmall : forall y, trunc w (Z.lor (rs a) (trunc w y)) = Z.lor (rs a) (trunc w y)).
  { intros y. apply trunc_small. fold bw. apply lor_small; [lia | lia | apply Hty]. }
  rewrite Hsmall.
  destruct b as [rb | z].
  - cbn [opv]. rewrite (trunc_reg rb w (Hb rb eq_refl)).
    destruct (const_of F st (OReg rb) bw false) as [k |] eqn:Ec.
    + inversion H; subst fct. pose proof (const_of_sound (OReg rb) bw k ltac:(lia) Ec) as Hk. cbn in Hk.
      pose proof (rsr rb) as Hrb. pose proof (Hb rb eq_refl) as Hwb. unfold wd in Hwb. rewrite Hwb in Hrb.
      rewrite Hk in *. rewrite (Z.mod_small k) by lia.
      apply (or_const_sound _ _ _ bw); [lia | exact Hwa | exact Hma | exact Hsa | lia | lia].
    + destruct (operand_b (OReg rb) bw Hbw) as [Hwb [Hmb Hsb]]. cbn [opv] in Hsb.
      pose proof (rsr rb) as Hrb. pose proof (Hb rb eq_refl) as Hwb'. unfold wd in Hwb'. rewrite Hwb' in Hrb.
      set (fa := normalize F st (opnd_form F st (OReg a) bw)) in *.
      set (fb := normalize F st (opnd_form F st (OReg rb) bw)) in *.
      assert (Hsum : Z.land (rs a) (rs rb) = 0 -> ffact_ok wdn rs (FForm (fadd fa fb) 0 None) (Z.lor (rs a) (rs rb))).
      { intros Hz. rewrite (lor_disjoint _ _ Hz). cbn [ffact_ok].
        split; [apply wf_fadd; assumption |].
        split; [apply sat_fadd; assumption | split; [apply aligned_nonpos; lia | discriminate]]. }
      (* a multiple of 2^T and a value below 2^T share no bit *)
      assert (Hc1 : forall b0 b1, reg_range F st rb = Some (b0, b1) ->
                (0 <=? b0) && (bitlen b1 <=? falign F st fa) = true -> Z.land (rs a) (rs rb) = 0).
      { intros b0 b1 Eb Ec'. apply andb_true_iff in Ec'. destruct Ec' as [E1 E2]. apply Z.leb_le in E1, E2.
        pose proof (rrs rb b0 b1 Eb). pose proof (bitlen_bound b1 ltac:(lia)). pose proof (bitlen_nonneg b1).
        apply (land_aligned_small (falign F st fa)); [lia | apply falign_sound; assumption |].
        split; [lia |]. assert (2 ^ bitlen b1 <= 2 ^ falign F st fa) by (apply Z.pow_le_mono_r; lia). lia. }
      assert (Hc2 : forall a0 a1, reg_range F st a = Some (a0, a1) ->
                (0 <=? a0) && (bitlen a1 <=? falign F st fb) = true -> Z.land (rs a) (rs rb) = 0).
      { intros a0 a1 Ea Ec'. apply andb_true_iff in Ec'. destruct Ec' as [E1 E2]. apply Z.leb_le in E1, E2.
        pose proof (rrs a a0 a1 Ea). pose proof (bitlen_bound a1 ltac:(lia)). pose proof (bitlen_nonneg a1).
        rewrite Z.land_comm. apply (land_aligned_small (falign F st fb)); [lia | apply falign_sound; assumption |].
        split; [lia |]. assert (2 ^ bitlen a1 <= 2 ^ falign F st fb) by (apply Z.pow_le_mono_r; lia). lia. }
      cbv zeta in H. fold fa fb in H.
      destruct (reg_range F st a) as [[a0 a1] |] eqn:Ea; destruct (reg_range F st rb) as [[b0 b1] |] eqn:Eb.
      * destruct ((0 <=? b0) && (bitlen b1 <=? falign F st fa)) eqn:E1;
          [inversion H; subst fct; apply Hsum; exact (Hc1 b0 b1 eq_refl E1) |].
        destruct ((0 <=? a0) && (bitlen a1 <=? falign F st fb)) eqn:E2;
          [inversion H; subst fct; apply Hsum; exact (Hc2 a0 a1 eq_refl E2) |].
        cbn [orb] in H.
        destruct ((0 <=? a0) && (0 <=? b0) && (a1 + b1 <? 2 ^ bw)) eqn:Er; [| discriminate].
        inversion H; subst fct. rewrite !andb_true_iff in Er. destruct Er as [[E3 E4] E5].
        apply Z.leb_le in E3, E4. apply Z.ltb_lt in E5.
        pose proof (rrs a a0 a1 Ea) as HA. pose proof (rrs rb b0 b1 Eb) as HB.
        pose proof (lor_bounds (rs a) (rs rb) ltac:(lia) ltac:(lia)).
        rewrite <- (trunc_small w (Z.lor (rs a) (rs rb))) by (fold bw; lia).
        apply range_fact_ok; [lia | lia |]. rewrite trunc_small by (fold bw; lia).
        apply aligned_lor.
        -- apply (aligned_mono (falign F st fa)); [lia | apply falign_sound; assumption].
        -- apply (aligned_mono (falign F st fb)); [lia | apply falign_sound; assumption].
      * cbn [orb] in H.
        destruct ((0 <=? a0) && (bitlen a1 <=? falign F st fb)) eqn:E2; [| discriminate].
        inversion H; subst fct. apply Hsum. exact (Hc2 a0 a1 eq_refl E2).
      * rewrite orb_false_r in H.
        destruct ((0 <=? b0) && (bitlen b1 <=? falign F st fa)) eqn:E1; [| discriminate].
        inversion H; subst fct. apply Hsum. exact (Hc1 b0 b1 eq_refl E1).
      * discriminate.
  - cbn [opv]. inversion H; subst fct.
    unfold trunc. fold bw. apply (or_const_sound _ _ _ bw); [lia | exact Hwa | exact Hma | exact Hsa | lia |].
    apply Z.mod_pos_bound. apply pow_nonneg_pos. lia.
Qed.

Lemma xor_fact_sound : forall w a b fct,
    1 <= Z.of_nat w <= 64 -> wd F a = Z.of_nat w -> (forall r, b = OReg r -> wd F r = Z.of_nat w) ->
    xor_fact F st (Z.of_nat w) a b = Some fct ->
    ffact_ok wdn rs fct (trunc w (Z.lxor (trunc w (rs a)) (trunc w (opv rs b)))).
Proof.
  intros w a b fct Hw Ha Hb H. unfold xor_fact in H. set (bw := Z.of_nat w) in *.
  assert (Hbw : 0 <= bw <= 64) by lia.
  destruct b as [rb | z].
  2: {
    destruct (const_of F st (OReg a) bw false) as [c |] eqn:Ec.
    - inversion H; subst fct. cbn [opv].
      pose proof (const_of_sound (OReg a) bw c ltac:(lia) Ec) as Hc. cbn [opv] in Hc.
      rewrite (trunc_reg a w Ha), Hc. unfold trunc at 2. fold bw.
      apply range_fact_ok; [lia | lia | apply aligned_nonpos; lia].
    - destruct (reg_range F st a) as [[a0 a1] |] eqn:Ea; [| discriminate].
      set (k := z mod 2 ^ bw) in *.
      set (n := Z.max (bitlen a1) (bitlen k)) in *.
      destruct ((0 <=? a0) && (2 ^ n <=? 2 ^ bw)) eqn:Ebound; [| discriminate].
      inversion H; subst fct. apply andb_true_iff in Ebound.
      destruct Ebound as [Hlo Hhi]. apply Z.leb_le in Hlo,Hhi.
      pose proof (rrs a a0 a1 Ea) as HA.
      assert (Hk : 0 <= k < 2 ^ bw) by (unfold k; apply Z.mod_pos_bound; apply pow_nonneg_pos; lia).
      pose proof (bitlen_bound a1 ltac:(lia)) as Hba.
      pose proof (bitlen_bound k ltac:(lia)) as Hbk.
      pose proof (bitlen_nonneg a1). pose proof (bitlen_nonneg k).
      assert (2 ^ bitlen a1 <= 2 ^ n) by (apply Z.pow_le_mono_r; unfold n; lia).
      assert (2 ^ bitlen k <= 2 ^ n) by (apply Z.pow_le_mono_r; unfold n; lia).
      pose proof (lxor_small (rs a) k n ltac:(unfold n; lia) ltac:(lia) ltac:(lia)) as Hx.
      cbn [opv]. rewrite (trunc_reg a w Ha). change (trunc w z) with k.
      apply range_fact_ok; [lia | lia |].
      rewrite trunc_small by (fold bw; lia).
      apply aligned_lxor.
      + apply (aligned_mono (reg_align F st a)); [lia | apply ras].
      + apply (aligned_mono (v2 k)); [lia |].
        unfold aligned. rewrite Z.max_r by apply v2_nonneg. apply v2_divides.
  }
  cbv zeta in H. cbn [opv].
  rewrite (trunc_reg a w Ha), (trunc_reg rb w (Hb rb eq_refl)).
  destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
  destruct (operand_b (OReg rb) bw Hbw) as [Hwb [Hmb Hsb]]. cbn [opv] in Hsb.
  set (fa := normalize F st (opnd_form F st (OReg a) bw)) in *.
  set (fb := normalize F st (opnd_form F st (OReg rb) bw)) in *.
  destruct (reg_range F st a) as [[a0 a1] |] eqn:Ea; [| discriminate].
  destruct (reg_range F st rb) as [[b0 b1] |] eqn:Eb; [| discriminate].
  set (n := Z.max (bitlen a1) (bitlen b1)) in *.
  destruct ((0 <=? a0) && (0 <=? b0) && (2 ^ n <=? 2 ^ bw)) eqn:Ec; [| discriminate].
  inversion H; subst fct. rewrite !andb_true_iff in Ec. destruct Ec as [[E1 E2] E3].
  apply Z.leb_le in E1, E2, E3.
  pose proof (rrs a a0 a1 Ea) as HA. pose proof (rrs rb b0 b1 Eb) as HB.
  pose proof (bitlen_bound a1 ltac:(lia)) as Hba. pose proof (bitlen_bound b1 ltac:(lia)) as Hbb.
  pose proof (bitlen_nonneg a1). pose proof (bitlen_nonneg b1).
  assert (2 ^ bitlen a1 <= 2 ^ n) by (apply Z.pow_le_mono_r; unfold n; lia).
  assert (2 ^ bitlen b1 <= 2 ^ n) by (apply Z.pow_le_mono_r; unfold n; lia).
  pose proof (lxor_small (rs a) (rs rb) n ltac:(unfold n; lia) ltac:(lia) ltac:(lia)) as Hx.
  apply range_fact_ok; [lia | lia |].
  rewrite trunc_small by (fold bw; lia).
  apply aligned_lxor.
  - apply (aligned_mono (falign F st fa)); [lia | apply falign_sound; assumption].
  - apply (aligned_mono (falign F st fb)); [lia | apply falign_sound; assumption].
Qed.

Lemma tsum_div : forall k ts, 0 <= k -> forallb (fun t0 => Z.eqb (snd t0 mod 2 ^ k) 0) ts = true ->
    tsum ts = 2 ^ k * tsum (map (fun t0 => (fst t0, snd t0 / 2 ^ k)) ts).
Proof.
  intros k ts Hk H. assert (Hp : 0 < 2 ^ k) by (apply pow_nonneg_pos; lia).
  induction ts as [| [x c] ts IH]; cbn [map tsum fst snd] in *; [ring |].
  apply andb_true_iff in H. destruct H as [H1 H2]. apply Z.eqb_eq in H1. cbn [snd] in H1.
  rewrite (IH H2). rewrite (Z.div_mod c (2 ^ k)) at 1 by lia. rewrite H1. ring.
Qed.

Lemma aligned_shift_sound : forall fa k bw signed g x,
    0 <= k -> 0 < bw <= 64 -> wf fa -> sat fa x -> 0 <= x < 2 ^ bw ->
    aligned_shift F st fa k bw signed = Some g ->
    wf g /\ fm g = bw /\ (if signed then x < 2 ^ (bw - 1) else True) /\ sat g (x / 2 ^ k).
Proof.
  intros fa k bw signed g x Hk Hbw Hwf Hsat Hx H. unfold aligned_shift in H.
  destruct (Z.ltb (fm fa) bw || negb (Z.eqb (flo fa) (fhi fa)) || negb (Z.eqb (flo fa mod 2 ^ k) 0)
            || negb (forallb (fun t0 => Z.eqb (snd t0 mod 2 ^ k) 0) (ft fa))) eqn:E1; [discriminate |].
  rewrite !orb_false_iff in E1. destruct E1 as [[[E1 E2] E3] E4].
  apply Z.ltb_ge in E1. apply negb_false_iff, Z.eqb_eq in E2, E3. apply negb_false_iff in E4.
  destruct (form_range F st fa) as [[x0 y0] |] eqn:Er; [| discriminate].
  assert (Hp : 0 < 2 ^ k) by (apply pow_nonneg_pos; lia).
  destruct ((0 <=? x0) && (y0 <? (if signed then 2 ^ (bw - 1) else 2 ^ bw))) eqn:Ec; [| discriminate].
  inversion H; subst g. clear H. apply andb_true_iff in Ec. destruct Ec as [Ec1 Ec2].
  apply Z.leb_le in Ec1. apply Z.ltb_lt in Ec2.
  destruct Hsat as [e [He Hv]]. assert (e = flo fa) by lia. subst e.
  pose proof (frs fa x0 y0 (flo fa) Er He) as Hb.
  assert (Htop : (if signed then 2 ^ (bw - 1) else 2 ^ bw) <= 2 ^ bw)
    by (destruct signed; [apply Z.pow_le_mono_r; lia | lia]).
  assert (Hxe : x = tsum (ft fa) + flo fa) by (apply (value_eq_form fa x bw (flo fa)); [exact Hwf | lia | lia | lia | exact He | exact Hv | lia]).
  split; [unfold wf; rewrite mk_fm; lia |]. split; [apply mk_fm |].
  split; [destruct signed; [lia | exact I] |].
  apply (sat_mk wdn rs bw _ (flo fa / 2 ^ k) (flo fa / 2 ^ k) (x / 2 ^ k) (flo fa / 2 ^ k)); [lia | lia |].
  rewrite Hxe. rewrite (tsum_div k (ft fa) Hk E4).
  rewrite (Z.div_mod (flo fa) (2 ^ k)) at 1 by lia. rewrite E3, Z.add_0_r.
  rewrite <- Z.mul_add_distr_l. rewrite Z.mul_comm, Z.div_mul by lia. apply eqm_refl.
Qed.

Lemma tsum_filter_split : forall (p : atom * Z -> bool) ts,
 tsum ts = tsum (filter p ts) + tsum (filter (fun t => negb (p t)) ts).
Proof.
 intros p ts. induction ts as [|[x c] rest IH]; cbn [filter tsum]; [ring|].
 destruct (p (x,c)); cbn [filter tsum negb]; rewrite IH; ring.
Qed.

Lemma quotient_shift_sound : forall fa k bw signed g x,
 0 <= k -> 0 < bw <= 64 -> wf fa -> sat fa x -> 0 <= x < 2 ^ bw ->
 quotient_shift F st fa k bw signed = Some g ->
 wf g /\ fm g = bw /\ (if signed then x < 2 ^ (bw - 1) else True) /\ sat g (x / 2 ^ k).
Proof.
 intros fa k bw signed g x Hk Hbw Hwf Hsat Hx H.
 unfold quotient_shift in H.
 destruct (ft fa) as [|t1 [|t2 rest]] eqn:Et; try discriminate.
 destruct (Z.ltb (fm fa) bw || negb (Z.eqb (flo fa) (fhi fa))) eqn:Ec; [discriminate|].
 apply orb_false_iff in Ec. destruct Ec as [Hfm Hconst].
 apply Z.ltb_ge in Hfm. apply negb_false_iff,Z.eqb_eq in Hconst.
 rewrite <- Et in H.
 set (multiple := fun t : atom*Z => Z.eqb (snd t mod 2 ^ k) 0) in *.
 set (high := filter multiple (ft fa)) in *.
 set (low := filter (fun t : atom*Z => negb (Z.eqb (snd t mod 2 ^ k) 0)) (ft fa)) in *.
 destruct (null high || null low); [discriminate|].
 destruct (form_range F st fa) as [[lo hi]|] eqn:Er; [|discriminate].
 destruct (form_range F st {|fm:=64;ft:=low;flo:=flo fa mod 2 ^ k;fhi:=flo fa mod 2 ^ k|}) as [[rlo rhi]|] eqn:Erem; [|discriminate].
 destruct ((0 <=? lo) && (hi <? (if signed then 2 ^ (bw-1) else 2 ^ bw)) && (0 <=? rlo) && (rhi <? 2 ^ k)) eqn:Eb; [|discriminate].
 injection H as <-. rewrite !andb_true_iff in Eb. destruct Eb as [[[Elo Ehi] Erlo] Erhi].
 apply Z.leb_le in Elo,Erlo. apply Z.ltb_lt in Ehi,Erhi.
 destruct Hsat as [e [He Hv]]. assert (e = flo fa) by lia. subst e.
 pose proof (frs fa lo hi (flo fa) Er He) as Hrange.
 pose proof (frs _ rlo rhi (flo fa mod 2 ^ k) Erem ltac:(cbn;lia)) as Hrem.
 cbn [ft flo fhi] in Hrem.
 assert (Htop : (if signed then 2 ^ (bw-1) else 2 ^ bw) <= 2 ^ bw) by (destruct signed; [apply Z.pow_le_mono_r;lia|lia]).
 assert (Hxe : x = tsum (ft fa) + flo fa) by (apply (value_eq_form fa x bw (flo fa)); [exact Hwf|lia|lia|lia|exact He|exact Hv|lia]).
 assert (Hdiv : forallb multiple high = true).
 { apply forallb_forall. intros term Hin. unfold high in Hin. apply filter_In in Hin. exact (proj2 Hin). }
 pose proof (tsum_div k high Hk Hdiv) as Hhigh.
 pose proof (tsum_filter_split multiple (ft fa)) as Hsplit. change (tsum (ft fa) = tsum high + tsum low) in Hsplit.
 assert (Hp : 0 < 2 ^ k) by (apply pow_nonneg_pos;lia).
 assert (Hq : x / 2 ^ k = tsum (map (fun t => (fst t,snd t / 2 ^ k)) high) + flo fa / 2 ^ k).
 { rewrite Hxe,Hsplit,Hhigh. rewrite (Z.div_mod (flo fa) (2 ^ k)) at 1 by lia.
   replace (2 ^ k * tsum (map (fun t => (fst t,snd t / 2 ^ k)) high) + tsum low +
     (2 ^ k * (flo fa / 2 ^ k) + flo fa mod 2 ^ k)) with
     ((tsum low + flo fa mod 2 ^ k) +
       (tsum (map (fun t => (fst t,snd t / 2 ^ k)) high) + flo fa / 2 ^ k) * 2 ^ k) by ring.
   rewrite Z.div_add by lia. rewrite Z.div_small by lia. ring. }
 split; [unfold wf;rewrite mk_fm;lia|]. split; [apply mk_fm|].
 split; [destruct signed;[lia|exact I]|].
 apply (sat_mk wdn rs bw _ _ _ _ (flo fa / 2 ^ k)); [lia|lia|]. rewrite Hq. apply eqm_refl.
Qed.

Lemma exact_shift_sound : forall fa k bw signed g x,
 0 <= k -> 0 < bw <= 64 -> wf fa -> sat fa x -> 0 <= x < 2 ^ bw ->
 exact_shift F st fa k bw signed = Some g ->
 wf g /\ fm g = bw /\ (if signed then x < 2 ^ (bw-1) else True) /\ sat g (x / 2 ^ k).
Proof.
 intros fa k bw signed g x Hk Hbw Hwf Hsat Hx H. unfold exact_shift in H.
 destruct (aligned_shift F st fa k bw signed) as [v|] eqn:Ea.
 - injection H as <-. eapply aligned_shift_sound; eassumption.
 - eapply quotient_shift_sound; eassumption.
Qed.

Lemma shift_fact_sound : forall signed w a k,
    1 <= Z.of_nat w <= 64 -> wd F a = Z.of_nat w -> 0 <= k <= Z.of_nat w ->
    ffact_ok wdn rs (shift_fact F st signed (Z.of_nat w) a k)
             (trunc w ((if signed then sext w (rs a) else trunc w (rs a)) / 2 ^ k)).
Proof.
  intros signed w a k Hw Ha Hk. unfold shift_fact.
  set (bw := Z.of_nat w) in *.
  assert (Hbw : 0 <= bw <= 64) by lia.
  destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
  pose proof (rsr a) as Hra. unfold wd in Ha. rewrite Ha in Hra. fold bw in Hra.
  assert (Hwn : wdn a = w) by (apply Nat2Z.inj; exact Ha).
  rewrite (trunc_small w (rs a)) by (fold bw; lia).
  assert (Hp : 0 < 2 ^ k) by (apply pow_nonneg_pos; lia).
  destruct (exact_shift F st (normalize F st (opnd_form F st (OReg a) bw)) k bw signed) as [g |] eqn:Ee.
  - destruct (exact_shift_sound _ k bw signed g (rs a) ltac:(lia) ltac:(lia) Hwa Hsa Hra Ee) as [H1 [H2 [H3 H4]]].
    cbn [ffact_ok]. split; [exact H1 |]. split; [| split; [apply aligned_nonpos; lia | discriminate]].
    apply sat_trunc; [exact H1 | lia |].
    destruct signed; [| exact H4].
    rewrite sext_small by (lia || (fold bw; lia)). exact H4.
  - destruct (Z.eqb bw 64 && negb signed) eqn:E64.
    + apply andb_true_iff in E64. destruct E64 as [E64 Es]. apply Z.eqb_eq in E64. destruct signed; [discriminate |].
      destruct (atom64_sound F E HE t st rs Hst Hrs a (EShr k)) as [Hs Hm].
      cbn [ffact_ok]. split; [unfold wf; lia |]. split; [| split; [apply aligned_nonpos; lia | discriminate]].
      apply sat_trunc; [unfold wf; lia | lia | exact Hs].
    + assert (Hrange : forall lo hi v, lo <= v <= hi -> ffact_ok wdn rs (FForm (range_form bw lo hi) 0 None) (trunc w v)).
      { intros lo hi v Hv. apply range_fact_ok; [lia | exact Hv | apply aligned_nonpos; lia]. }
      destruct signed.
      * destruct (sreg_range F st a) as [[x0 y0] |] eqn:Er.
        -- apply Hrange. pose proof (srs a x0 y0 Er) as Hs'. rewrite Hwn in Hs'. split; apply Z.div_le_mono; lia.
        -- (* a sign-extended w-bit value shifted right *)
           assert (Hsx : - 2 ^ (bw - 1) <= sext w (rs a) < 2 ^ (bw - 1)).
           { unfold sext. rewrite trunc_small by (fold bw; lia).
             assert (H2 : 2 ^ bw = 2 * 2 ^ (bw - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
             fold bw. destruct (Z.ltb_spec (rs a) (2 ^ (bw - 1))); lia. }
           destruct (Z.ltb_spec k bw) as [Hkb | Hkb].
           ++ apply Hrange.
              assert (Hd : 2 ^ (bw - 1) = 2 ^ (bw - 1 - k) * 2 ^ k) by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
              split.
              ** apply Z.div_le_lower_bound; [lia |]. rewrite Hd in Hsx. nia.
              ** assert (sext w (rs a) / 2 ^ k < 2 ^ (bw - 1 - k)); [| lia].
                 apply Z.div_lt_upper_bound; [lia |]. rewrite Hd in Hsx. nia.
           ++ apply Hrange. assert (k = bw) by lia. subst k.
              assert (H2 : 2 ^ bw = 2 * 2 ^ (bw - 1)) by (rewrite <- Z.pow_succ_r by lia; f_equal; lia).
              split.
              ** apply Z.div_le_lower_bound; [lia | lia].
              ** assert (sext w (rs a) / 2 ^ bw < 1); [| lia]. apply Z.div_lt_upper_bound; lia.
      * destruct (reg_range F st a) as [[x0 y0] |] eqn:Er.
        -- apply Hrange. pose proof (rrs a x0 y0 Er). split; apply Z.div_le_mono; lia.
        -- apply Hrange. split; [apply Z.div_pos; lia | apply Z.div_le_mono; lia].
Qed.

(* ------------------------------------------------------------------ *)
(* Binary operations.                                                  *)

Definition bin_wf (o : binop) (w : nat) (a : reg) (b : opnd) : Prop :=
  1 <= Z.of_nat w <= 64 /\ wd F a = Z.of_nat w /\
  (forall r, b = OReg r -> match o with BShl | BShr | BShrs => wd F r = 32 | _ => wd F r = Z.of_nat w end).

Lemma binop_fact_sound : forall o w a b fct,
    bin_wf o w a b -> binop_fact F st o (Z.of_nat w) a b = Some fct ->
    ffact_ok wdn rs fct (trunc w (binop_eval o w (rs a) (opv rs b))).
Proof.
  intros o w a b fct [Hw [Ha Hb]] H. unfold binop_fact in H. set (bw := Z.of_nat w) in *.
  assert (Hbw : 0 <= bw <= 64) by lia.
  destruct o; cbn [binop_eval].
  - inversion H; subst fct. rewrite <- (Z.mul_1_l (opv rs b)). apply addsub_sound; [lia | left; reflexivity].
  - inversion H; subst fct. replace (rs a - opv rs b) with (rs a + -1 * opv rs b) by ring.
    apply addsub_sound; [lia | right; reflexivity].
  - apply mul_sound; [lia | exact Hb | exact H].
  - (* shl *)
    destruct (const_of F st b 32 false) as [k |] eqn:Ec; [| discriminate]. inversion H; subst fct.
    pose proof (const_of_sound b 32 k ltac:(lia) Ec) as Hk.
    destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
    assert (Hsh : shamt w (opv rs b) = Z.min (k mod 2 ^ 32) bw).
    { unfold shamt. f_equal. destruct b as [r | z]; cbn [opv]; unfold trunc; cbn [Z.of_nat Pos.of_succ_nat].
      - rewrite Hk. reflexivity.
      - rewrite Hk. rewrite Z.mod_mod by lia. reflexivity. }
    rewrite Hsh. set (s := Z.min (k mod 2 ^ 32) bw).
    assert (Hs0 : 0 <= s) by (unfold s; pose proof (Z.mod_pos_bound k (2 ^ 32) ltac:(lia)); lia).
    cbn [ffact_ok]. split; [apply wf_fscale; [exact Hwa | lia] |].
    split.
    2: { split; [|discriminate]. apply aligned_trunc; [unfold s, bw; lia |].
         unfold aligned. rewrite Z.max_r by exact Hs0. apply Z.divide_factor_r. }
    apply sat_trunc; [apply wf_fscale; [exact Hwa | lia] | apply fscale_fm_bounds; [exact (proj1 Hwa) | lia] |].
    rewrite Z.mul_comm. apply sat_fscale; [exact Hwa | lia | exact Hsa].
  - (* shr *)
    destruct (const_of F st b 32 false) as [k |] eqn:Ec; [| discriminate]. inversion H; subst fct.
    pose proof (const_of_sound b 32 k ltac:(lia) Ec) as Hk.
    assert (Hsh : shamt w (opv rs b) = Z.min (k mod 2 ^ 32) bw).
    { unfold shamt. f_equal. destruct b as [r | z]; cbn [opv]; unfold trunc; cbn [Z.of_nat Pos.of_succ_nat].
      - rewrite Hk. reflexivity.
      - rewrite Hk. rewrite Z.mod_mod by lia. reflexivity. }
    rewrite Hsh. apply (shift_fact_sound false w a); [lia | exact Ha |].
    pose proof (Z.mod_pos_bound k (2 ^ 32) ltac:(lia)). lia.
  - (* shrs *)
    destruct (const_of F st b 32 false) as [k |] eqn:Ec; [| discriminate]. inversion H; subst fct.
    pose proof (const_of_sound b 32 k ltac:(lia) Ec) as Hk.
    assert (Hsh : shamt w (opv rs b) = Z.min (k mod 2 ^ 32) bw).
    { unfold shamt. f_equal. destruct b as [r | z]; cbn [opv]; unfold trunc; cbn [Z.of_nat Pos.of_succ_nat].
      - rewrite Hk. reflexivity.
      - rewrite Hk. rewrite Z.mod_mod by lia. reflexivity. }
    rewrite Hsh. apply (shift_fact_sound true w a); [lia | exact Ha |].
    pose proof (Z.mod_pos_bound k (2 ^ 32) ltac:(lia)). lia.
  - (* and *)
    pose proof (rsr a) as Hra. pose proof Ha as Ha'. unfold wd in Ha'. rewrite Ha' in Hra. fold bw in Hra.
    destruct (const_of F st b bw false) as [k |] eqn:Ec.
    2: { destruct b as [rb | z]; [| discriminate].
         pose proof (Hb rb eq_refl) as Hrb. cbv beta iota in Hrb. cbn [opv].
         pose proof (rsr rb) as Hrr. pose proof Hrb as Hrb'. unfold wd in Hrb'. rewrite Hrb' in Hrr. fold bw in Hrr.
         rewrite (trunc_reg a w Ha), (trunc_reg rb w Hrb).
         pose proof (land_bounds (rs a) (rs rb) ltac:(lia)) as Hla.
         pose proof (land_le_r (rs a) (rs rb) ltac:(lia)) as Hlb.
         rewrite trunc_small by (fold bw; lia).
         destruct (const_of F st (OReg a) bw false) as [k |] eqn:Eca.
         - (* a is the constant k: b masked by k *)
           pose proof (const_of_sound (OReg a) bw k ltac:(lia) Eca) as Hk. cbn beta iota in Hk.
           destruct (operand_a rb bw Hbw) as [Hwb [Hmb Hsb]].
           assert (Hkm : k mod 2 ^ bw = rs a) by (rewrite Hk; apply Z.mod_small; lia).
           rewrite Z.land_comm, <- Hkm.
           apply (mask_sound (normalize F st (opnd_form F st (OReg rb) bw)) rb (k mod 2 ^ bw) bw (rs rb) fct);
             [lia | lia | exact Hwb | exact Hsb | lia | reflexivity | exact H].
         - (* at most the least upper bound of the operands' ranges *)
           assert (Hfact : forall m, 0 <= m -> Z.land (rs a) (rs rb) <= m ->
                     ffact_ok wdn rs (FForm (range_form bw 0 m) 0 None) (Z.land (rs a) (rs rb))).
           { intros m Hm Hle. cbn [ffact_ok].
             split; [unfold wf, range_form; cbn [fm]; lia |]. split; [| split; [apply aligned_nonpos; lia | discriminate]].
             exists (Z.land (rs a) (rs rb)). unfold range_form. cbn [fm ft flo fhi tsum].
             split; [lia | rewrite Z.add_0_l; apply eqm_refl]. }
           cbv zeta in H.
           destruct (reg_range F st a) as [[xa ya] |] eqn:Era;
             [pose proof (rrs a xa ya Era) as Hya |];
             destruct (reg_range F st rb) as [[xb yb] |] eqn:Erb;
             try pose proof (rrs rb xb yb Erb) as Hyb; try discriminate;
             match type of H with
             | (if ?m <? 0 then _ else _) = _ =>
                 destruct (Z.ltb_spec m 0); [discriminate |]; inversion H; subst fct; apply Hfact; lia
             end. }
    pose proof (const_of_sound b bw k ltac:(lia) Ec) as Hk.
    destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]].
    assert (Hkb : trunc w (opv rs b) = k mod 2 ^ bw).
    { destruct b as [r | z]; cbn [opv]; unfold trunc; fold bw; [rewrite Hk; reflexivity | rewrite Hk, Z.mod_mod by lia; reflexivity]. }
    rewrite Hkb, (trunc_reg a w Ha).
    pose proof (Z.mod_pos_bound k (2 ^ bw) ltac:(apply pow_nonneg_pos; lia)) as Hkr.
    rewrite trunc_small by (fold bw; pose proof (land_le_r (rs a) (k mod 2 ^ bw) ltac:(lia)); lia).
    apply (mask_sound (normalize F st (opnd_form F st (OReg a) bw)) a (k mod 2 ^ bw) bw (rs a) fct);
      [lia | lia | exact Hwa | exact Hsa | lia | reflexivity | exact H].
  - (* or *)
    apply or_fact_sound; [lia | exact Ha | exact Hb | exact H].
  - (* xor *)
    apply xor_fact_sound; [lia | exact Ha | exact Hb | exact H].
Qed.

(* ------------------------------------------------------------------ *)
(* Values of instructions.                                             *)

(* Unlike inversion, keeps the terms as written. *)
Lemma some_eq : forall (A : Type) (a b : A), Some a = Some b -> b = a.
Proof. intros A a b H. congruence. Qed.

Lemma ffact_fact : forall x v, ffact_ok wdn rs x v -> fact_okv F t rs x v.
Proof. intros x v H. destruct x; cbn [ffact_ok] in H; try contradiction; cbn [fact_okv]; exact H. Qed.

Lemma wdn_of : forall r w, wd F r = Z.of_nat w -> wdn r = w.
Proof. intros r w H. unfold wd in H. lia. Qed.

Lemma form_fact_ok : forall f v, wf f -> sat f v -> fact_okv F t rs (FForm f 0 None) v.
Proof.
  intros f v H1 H3. cbn [fact_okv]. split; [exact H1 | split; [exact H3 | split; [apply aligned_nonpos; lia | discriminate]]].
Qed.

Lemma value_mov : forall g d s x, wd F d = wd F s -> value F st (IMov g d s) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (rs s)).
Proof.
  intros g d s x Hw H. cbn [value] in H.
  assert (Hts : trunc (wdn d) (rs s) = rs s).
  { apply trunc_small. pose proof (rsr s). unfold wd in Hw. rewrite Hw. lia. }
  rewrite Hts.
  destruct (get st s) as [y |] eqn:Eg.
  - destruct y; try (inversion H; subst x; exact (proj1 Hst s _ Eg)).
    all: destruct (Z.eqb (wd F s) (wd F d)) eqn:E1; [| discriminate]; inversion H; subst x;
      destruct (opnd_form_sound F E HE t st rs Hst Hrs (OReg s) (wd F d) ltac:(pose proof (wdr d); unfold wd; lia))
        as [Hs [Hwf Hm]];
      apply form_fact_ok; [exact Hwf | exact Hs].
  - destruct (Z.eqb (wd F s) (wd F d)) eqn:E1; [| discriminate]. inversion H; subst x.
    destruct (opnd_form_sound F E HE t st rs Hst Hrs (OReg s) (wd F d) ltac:(pose proof (wdr d); unfold wd; lia))
      as [Hs [Hwf Hm]].
    apply form_fact_ok; [exact Hwf | exact Hs].
Qed.

Lemma value_imm : forall g d z x, value F st (IImm g d z) = Some x -> fact_okv F t rs x (trunc (wdn d) z).
Proof.
  intros g d z x H. cbn [value] in H. inversion H; subst x.
  apply form_fact_ok; [unfold fconst, wf; cbn; pose proof (wdr d); unfold wd; lia |].
  apply sat_fconst. unfold trunc. fold (wd F d). apply eqm_refl.
Qed.

Lemma value_bin : forall g o w d a b x,
    bin_wf o w a b -> wd F d = Z.of_nat w -> value F st (IBin g o w d a b) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (binop_eval o w (rs a) (opv rs b))).
Proof.
  intros g o w d a b x Hw Hd H. cbn [value] in H. rewrite (wdn_of d w Hd).
  apply ffact_fact. apply (binop_fact_sound o w a b x Hw H).
Qed.

Lemma value_mad_eq : forall g w d a b c,
    value F st (IMad g w d a b c) =
    match const_of F st b (Z.of_nat w) false with
    | Some k => Some (FForm (fadd (fscale (normalize F st (opnd_form F st (OReg a) (Z.of_nat w))) (k mod 2 ^ Z.of_nat w) (Z.of_nat w))
                                  (normalize F st (opnd_form F st c (Z.of_nat w)))) 0 None)
    | None =>
        match const_of F st (OReg a) (Z.of_nat w) false with
        | Some k => Some (FForm (fadd (fscale (normalize F st (opnd_form F st b (Z.of_nat w))) (k mod 2 ^ Z.of_nat w) (Z.of_nat w))
                                      (normalize F st (opnd_form F st c (Z.of_nat w)))) 0 None)
        | None => None
        end
    end.
Proof. reflexivity. Qed.

Lemma value_mad : forall g w d a b c x,
    1 <= Z.of_nat w <= 64 -> wd F d = Z.of_nat w -> value F st (IMad g w d a b c) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (rs a * opv rs b + opv rs c)).
Proof.
  intros g w d a b c x Hw Hd H. rewrite value_mad_eq in H. rewrite (wdn_of d w Hd).
  set (bw := Z.of_nat w) in *. assert (Hbw : 0 <= bw <= 64) by lia.
  destruct (const_of F st b bw false) as [k |] eqn:Ec.
  2: { (* a is the constant: k times b's form, plus c's *)
       destruct (const_of F st (OReg a) bw false) as [k |] eqn:Eca; [| discriminate].
       apply some_eq in H. subst x.
       pose proof (const_of_sound (OReg a) bw k ltac:(lia) Eca) as Hk. cbn [opv] in Hk.
       destruct (operand_b b bw Hbw) as [Hwb [Hmb Hsb]]. destruct (operand_b c bw Hbw) as [Hwc [Hmc Hsc]].
       assert (Hws : wf (fscale (normalize F st (opnd_form F st b bw)) (k mod 2 ^ bw) bw))
         by (apply wf_fscale; [exact Hwb | lia]).
       assert (Hms : 0 <= fm (fscale (normalize F st (opnd_form F st b bw)) (k mod 2 ^ bw) bw) <= bw)
         by (apply fscale_fm_bounds; [exact (proj1 Hwb) | lia]).
       pose proof (proj1 Hwc) as Hwc0.
       apply form_fact_ok; [apply wf_fadd; assumption |].
       apply (sat_eqm _ _ _ ((k mod 2 ^ bw) * opv rs b + opv rs c)).
       - apply sat_fadd; [exact Hws | exact Hwc | apply sat_fscale; [exact Hwb | lia | exact Hsb] | exact Hsc].
       - rewrite fadd_fm. apply (eqm_weaken bw); [lia |].
         eapply eqm_trans; [| apply eqm_sym, trunc_eqm]. fold bw.
         apply eqm_add; [lia | | apply eqm_refl].
         rewrite (Z.mul_comm (k mod 2 ^ bw)), (Z.mul_comm (rs a)). apply eqm_mul; [lia |].
         rewrite Hk. apply eqm_mod. lia. }
  apply some_eq in H. subst x.
  pose proof (const_of_sound b bw k ltac:(lia) Ec) as Hk.
  destruct (operand_a a bw Hbw) as [Hwa [Hma Hsa]]. destruct (operand_b c bw Hbw) as [Hwc [Hmc Hsc]].
  assert (Hws : wf (fscale (normalize F st (opnd_form F st (OReg a) bw)) (k mod 2 ^ bw) bw))
    by (apply wf_fscale; [exact Hwa | lia]).
  assert (Hms : 0 <= fm (fscale (normalize F st (opnd_form F st (OReg a) bw)) (k mod 2 ^ bw) bw) <= bw)
    by (apply fscale_fm_bounds; [exact (proj1 Hwa) | lia]).
  pose proof (proj1 Hwc) as Hwc0.
  apply form_fact_ok; [apply wf_fadd; assumption |].
  apply (sat_eqm _ _ _ ((k mod 2 ^ bw) * rs a + opv rs c)).
  - apply sat_fadd; [exact Hws | exact Hwc | apply sat_fscale; [exact Hwa | lia | exact Hsa] | exact Hsc].
  - rewrite fadd_fm. apply (eqm_weaken bw); [lia |].
    eapply eqm_trans; [| apply eqm_sym, trunc_eqm]. fold bw.
    apply eqm_add; [lia | | apply eqm_refl]. rewrite Z.mul_comm. apply eqm_mul; [lia |].
    destruct b as [r | z]; cbn [opv]; [rewrite Hk; apply eqm_mod; lia | rewrite Hk, Z.mod_mod by (apply Z.pow_nonzero; lia); apply eqm_mod; lia].
Qed.

Lemma value_wide : forall g signed d a b c x,
    wd F d = 64 -> wd F a = 32 -> (forall r, b = OReg r -> wd F r = 32) ->
    value F st (IWide g signed d a b c) = Some x ->
    fact_okv F t rs x
      (trunc (wdn d) (extend 32 signed (rs a) * extend 32 signed (opv rs b) + match c with Some r => rs r | None => 0 end)).
Proof.
  intros g signed d a b c x Hd Ha Hb H. cbn [value] in H. rewrite (wdn_of d 64 Hd).
  (* an operand's value as the product reads it, when the analysis knows it
     as the constant k *)
  assert (Hk : forall o k, (forall r, o = OReg r -> wd F r = 32) -> const_of F st o 32 signed = Some k ->
                 extend 32 signed (opv rs o) = k).
  { intros o k Ho Ec. unfold const_of, opnd_range in Ec. destruct o as [r | z]; cbn [opv].
    - pose proof (Ho r eq_refl) as Hr. pose proof (wdn_of r 32 Hr) as Hrn. pose proof (rsr r) as Hrr. rewrite Hrn in Hrr.
      destruct signed.
      + destruct (sreg_range F st r) as [[p q] |] eqn:Es; [| discriminate].
        destruct (Z.eqb_spec p q); [| discriminate]. subst q. inversion Ec; subst p.
        pose proof (srs r k k Es) as Hsr. rewrite Hrn in Hsr. cbn [extend]. lia.
      + destruct (reg_range F st r) as [[p q] |] eqn:Es; [| discriminate].
        destruct (Z.eqb_spec p q); [| discriminate]. subst q. inversion Ec; subst p.
        pose proof (rrs r k k Es). cbn [extend]. rewrite trunc_small by (cbn; lia). lia.
    - rewrite Z.eqb_refl in Ec. inversion Ec. reflexivity. }
  (* k times the extension of a 32-bit register, as a form *)
  assert (Hsc : forall r k, wd F r = 32 ->
            wf (fscale (normalize F st (as_form F st r (if signed then ES else EZ))) k 64) /\
            fm (fscale (normalize F st (as_form F st r (if signed then ES else EZ))) k 64) <= 64 /\
            sat (fscale (normalize F st (as_form F st r (if signed then ES else EZ))) k 64) (k * extend 32 signed (rs r))).
  { intros r k Hr.
    destruct (as_form_sound F E HE t st rs Hst Hrs r (if signed then ES else EZ)) as [Hs Hm].
    assert (Hav : atom_val wdn rs (r, if signed then ES else EZ) = extend 32 signed (rs r)).
    { pose proof (wdn_of r 32 Hr) as Hrn. pose proof (rsr r) as Hrr. rewrite Hrn in Hrr.
      destruct signed; cbn [atom_val extend]; [rewrite Hrn; reflexivity | rewrite trunc_small by (cbn; lia); reflexivity]. }
    rewrite Hav in Hs.
    destruct (normalize_sound F E HE t st rs Hst Hrs _ _ (as_form_wf F E HE t st rs Hst Hrs r _) Hs)
      as [Hw1 [Hm1 Hs1]].
    split; [apply wf_fscale; [exact Hw1 | lia] |].
    split; [apply fscale_fm_bounds; [unfold wf in Hw1; lia | lia] |].
    apply sat_fscale; [exact Hw1 | lia | exact Hs1]. }
  (* the product of the operands, as a form *)
  assert (Hp : exists f, match const_of F st b 32 signed with
                         | Some k => Some (fscale (normalize F st (as_form F st a (if signed then ES else EZ))) k 64)
                         | None =>
                             match b, const_of F st (OReg a) 32 signed with
                             | OReg rb, Some k => Some (fscale (normalize F st (as_form F st rb (if signed then ES else EZ))) k 64)
                             | _, _ => None
                             end
                         end = Some f /\ wf f /\ fm f <= 64 /\
                         sat f (extend 32 signed (rs a) * extend 32 signed (opv rs b))).
  { destruct (const_of F st b 32 signed) as [k |] eqn:Ec.
    - rewrite (Hk b k Hb Ec). destruct (Hsc a k Ha) as [Hw [Hm Hs]].
      eexists. split; [reflexivity |]. rewrite Z.mul_comm. exact (conj Hw (conj Hm Hs)).
    - destruct b as [rb | z]; [| cbv beta iota in H; discriminate].
      destruct (const_of F st (OReg a) 32 signed) as [k |] eqn:Ea; [| cbv beta iota in H; discriminate].
      pose proof (Hk (OReg a) k (fun r E0 => ltac:(injection E0 as <-; exact Ha)) Ea) as Hka.
      cbn [opv] in Hka. rewrite Hka.
      destruct (Hsc rb k (Hb rb eq_refl)) as [Hw [Hm Hs]].
      eexists. split; [reflexivity |]. exact (conj Hw (conj Hm Hs)). }
  destruct Hp as [f [Ef [Hws [Hms Hss]]]]. cbv zeta in H. rewrite Ef in H. inversion H; subst x.
  destruct c as [r |].
  - destruct (as_form_sound F E HE t st rs Hst Hrs r EX) as [Hsr Hmr]. cbn [atom_val] in Hsr.
    destruct (normalize_sound F E HE t st rs Hst Hrs _ _ (as_form_wf F E HE t st rs Hst Hrs r _) Hsr)
      as [Hw2 [Hm2 Hs2]].
    apply form_fact_ok; [apply wf_fadd; assumption |].
    apply sat_trunc; [apply wf_fadd; assumption | rewrite fadd_fm; change (Z.of_nat 64) with 64; lia |].
    apply sat_fadd; assumption.
  - apply form_fact_ok; [exact Hws |].
    apply sat_trunc; [exact Hws | change (Z.of_nat 64) with 64; lia |]. rewrite Z.add_0_r. exact Hss.
Qed.

Lemma extend_same : forall r n signed, wdn r = n -> trunc n (extend n signed (rs r)) = rs r.
Proof.
  intros r n signed Hn. pose proof (rsr r) as Hr. rewrite Hn in Hr.
  assert (Ht : trunc n (rs r) = rs r) by (apply trunc_small; exact Hr).
  destruct signed; cbn [extend].
  - rewrite <- Ht at 2. unfold trunc. apply sext_eqm.
  - rewrite Ht. exact Ht.
Qed.

Lemma converted_form_sound : forall s n signed g,
 n = wdn s -> converted_form F st s n signed = Some g ->
 wf g /\ sat g (extend n signed (rs s)).
Proof.
 intros s n signed g Hn H. unfold converted_form in H.
 destruct (get st s) as [[f al vr| | | ]|] eqn:Eg; try discriminate.
 destruct (null (ft f) || negb (Nat.leb 1 n) || (fm f <? Z.of_nat n)) eqn:Evalid; [discriminate|].
 rewrite !orb_false_iff in Evalid. destruct Evalid as [[_ En] Em].
 apply negb_false_iff,Nat.leb_le in En. apply Z.ltb_ge in Em.
 destruct (bounded_form_range F st f (lo_ok (Z.of_nat n) signed) (hi_ok (Z.of_nat n) signed)) as [[lo hi]|] eqn:Er; [|discriminate].
 destruct ((lo_ok (Z.of_nat n) signed <=? lo) && (hi <=? hi_ok (Z.of_nat n) signed)) eqn:Eb; [|discriminate].
 apply andb_true_iff in Eb. destruct Eb as [Elo Ehi]. apply Z.leb_le in Elo,Ehi.
 injection H as <-. pose proof (proj1 Hst s _ Eg) as [Hwf [[e [He Hv]] _]].
 pose proof (bounded_form_range_sound F E HE t st rs Hst Hrs f (lo_ok (Z.of_nat n) signed) (hi_ok (Z.of_nat n) signed) lo hi e Er He) as Hb.
 pose proof (rsr s) as Hr. rewrite <- Hn in Hr.
 split; [unfold wf;cbn;lia|]. exists e. cbn [ft flo fhi fm]. split; [exact He|].
 assert (Hvalue : extend n signed (rs s) = tsum (ft f) + e).
 { destruct signed; cbn [extend lo_ok hi_ok] in *.
   - apply (eqm_sext_eq (fm f) n); assumption || lia.
   - rewrite trunc_small by exact Hr.
     apply (eqm_small_eq (fm f) (Z.of_nat n)); assumption || lia. }
 rewrite Hvalue. apply eqm_refl.
Qed.

Lemma value_cvt : forall g to from signed d s x,
    wd F d = Z.of_nat to -> Z.of_nat from <= wd F s -> value F st (ICvt g to from signed d s) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (extend from signed (rs s))).
Proof.
  intros g to from signed d s x Hd Hs H. rewrite (wdn_of d to Hd).
  assert (Heq : value F st (ICvt g to from signed d s) =
    if Nat.eqb to 64 && Nat.eqb from 32 then
      let e := if signed then ES else EZ in
      if Z.eqb (wd F s) 32 then Some (FForm (normalize F st (as_form F st s e)) 0 None) else
      match get st s with
      | Some (FForm f _ _) as x =>
          if interval_only x then None else
          match lift F st (fat f 32) e with
          | Some g => Some (FForm (normalize F st g) 0 None)
          | None => Some (FForm (fat f 32) 0 (if signed then None else Some (0, 2 ^ 32 - 1)))
          end
      | _ => None
      end
    else if Nat.eqb to 64 && Nat.eqb from 64 then Some (FForm (normalize F st (as_form F st s EX)) 0 None)
    else if Nat.eqb to 32 && (Nat.eqb from 64 || Nat.eqb from 32 && Z.eqb (wd F s) 64) then
      Some (FForm (fat (normalize F st (as_form F st s EX)) 32) 0 None)
    else if Nat.eqb to from && Z.eqb (wd F s) (Z.of_nat from) then get st s
    else if Nat.eqb from (wdn s) then
      match converted_form F st s from signed with
      | Some f=>Some (FForm (fat f (Z.of_nat to)) 0 None)
      | None=>Some (FForm (fat (normalize F st (as_form F st s (if signed then ES else EZ))) (Z.of_nat to)) 0 None)
      end
    else None) by reflexivity.
  rewrite Heq in H. clear Heq.
  destruct (Nat.eqb to 64 && Nat.eqb from 32) eqn:E1.
  { apply andb_true_iff in E1. destruct E1 as [E1 E2]. apply Nat.eqb_eq in E1, E2. subst to from. cbv zeta in H.
    set (e := if signed then ES else EZ) in *.
    assert (Hext : forall g v, sat g (if signed then sext 32 v else trunc 32 v) -> sat g (extend 32 signed v))
      by (intros g0 v Hg; destruct signed; exact Hg).
    destruct (Z.eqb (wd F s) 32) eqn:E3.
    - apply some_eq in H. subst x. apply Z.eqb_eq in E3. pose proof (wdn_of s 32 E3) as Hsn.
      destruct (as_form_sound F E HE t st rs Hst Hrs s e) as [Has Ham].
      destruct (normalize_sound F E HE t st rs Hst Hrs _ _ (as_form_wf F E HE t st rs Hst Hrs s e) Has)
        as [Hw1 [Hm1 Hs1]].
      apply form_fact_ok; [exact Hw1 |]. apply sat_trunc; [exact Hw1 | lia |].
      assert (Hv : atom_val wdn rs (s, e) = extend 32 signed (rs s)).
      { unfold e. pose proof (rsr s) as Hr. rewrite Hsn in Hr.
        destruct signed; cbn [atom_val extend]; [rewrite Hsn; reflexivity | rewrite trunc_small by (cbn; lia); reflexivity]. }
      rewrite <- Hv. exact Hs1.
    - destruct (get st s) as [[f al vr | | | ] |] eqn:Eg; try discriminate.
      destruct (interval_only (Some (FForm f al vr))); [discriminate |].
      pose proof (proj1 Hst s _ Eg) as [Hwf [Hsf _]].
      pose proof (sat_fat _ rs f 32 (rs s) Hwf ltac:(lia) Hsf) as Hs32.
      destruct (lift F st (fat f 32) e) as [g' |] eqn:El.
      + apply some_eq in H. subst x.
        destruct (lift_sound F E HE t st rs Hst Hrs _ e g' (rs s) Hs32 El) as [Hg [Hwg Hor]].
        assert (Hsg : sat g' (extend 32 signed (rs s))).
        { unfold e in Hor. destruct signed; destruct Hor as [[He Hz] | [He Hz]]; try discriminate; exact Hz. }
        destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hwg Hsg) as [Hw1 [Hm1 Hs1]].
        apply form_fact_ok; [exact Hw1 |]. apply sat_trunc; [exact Hw1 | lia | exact Hs1].
      + (* the residue of the source modulo 2^32 *)
        apply some_eq in H. subst x. cbn [fact_okv].
        split; [apply wf_fat; [exact Hwf | lia] |].
        split.
        * apply (sat_eqm _ _ _ (rs s)); [exact Hs32 |].
          apply (eqm_weaken 32); [unfold fat; rewrite mk_fm; unfold wf in Hwf; lia |].
          eapply eqm_trans; [| apply (eqm_weaken 64); [lia | apply eqm_sym, trunc_eqm]].
          destruct signed; cbn [extend]; [apply eqm_sym, sext_eqm | apply eqm_sym, trunc_eqm].
        * split; [apply aligned_nonpos; lia |].
          intros lo hi Hvr. destruct signed; [discriminate |]. injection Hvr as <- <-.
          cbn [extend]. unfold trunc.
          pose proof (Z.mod_pos_bound (rs s) (2 ^ Z.of_nat 32) ltac:(apply pow_nonneg_pos; lia)) as Hb.
          change (Z.of_nat 32) with 32 in *. change (Z.of_nat 64) with 64.
          rewrite (Z.mod_small (rs s mod 2 ^ 32) (2 ^ 64)) by lia. lia. }
  destruct (Nat.eqb to 64 && Nat.eqb from 64) eqn:E2.
  { apply andb_true_iff in E2. destruct E2 as [E2a E2b]. apply Nat.eqb_eq in E2a, E2b. subst to from.
    apply some_eq in H. subst x.
    destruct (as_form_sound F E HE t st rs Hst Hrs s EX) as [Has Ham]. cbn [atom_val] in Has.
    destruct (normalize_sound F E HE t st rs Hst Hrs _ _ (as_form_wf F E HE t st rs Hst Hrs s EX) Has)
      as [Hw1 [Hm1 Hs1]].
    apply form_fact_ok; [exact Hw1 |]. apply (sat_eqm _ _ _ (rs s)); [exact Hs1 |].
    rewrite Hm1, Ham. eapply eqm_trans; [| apply eqm_sym, trunc_eqm].
    destruct signed; cbn [extend]; [apply eqm_sym, sext_eqm | apply eqm_sym, trunc_eqm]. }
  destruct (Nat.eqb to 32 && (Nat.eqb from 64 || Nat.eqb from 32 && Z.eqb (wd F s) 64)) eqn:E3.
  { apply andb_true_iff in E3. destruct E3 as [E3a E3b]. apply Nat.eqb_eq in E3a. subst to.
    assert (Hv32 : eqm 32 (trunc 32 (extend from signed (rs s))) (rs s)).
    { eapply eqm_trans; [apply trunc_eqm |]. apply orb_true_iff in E3b. destruct E3b as [E3b | E3b].
      - apply Nat.eqb_eq in E3b. subst from. apply (eqm_weaken 64); [lia |].
        destruct signed; cbn [extend]; [apply sext_eqm | apply trunc_eqm].
      - apply andb_true_iff in E3b. destruct E3b as [E3b _]. apply Nat.eqb_eq in E3b. subst from.
        destruct signed; cbn [extend]; [apply sext_eqm | apply trunc_eqm]. }
    apply some_eq in H. subst x.
    destruct (as_form_sound F E HE t st rs Hst Hrs s EX) as [Has Ham]. cbn [atom_val] in Has.
    destruct (normalize_sound F E HE t st rs Hst Hrs _ _ (as_form_wf F E HE t st rs Hst Hrs s EX) Has)
      as [Hw1 [Hm1 Hs1]].
    apply form_fact_ok; [apply wf_fat; [exact Hw1 | lia] |].
    apply (sat_eqm _ _ _ (rs s)); [apply sat_fat; [exact Hw1 | lia | exact Hs1] |].
    unfold fat; rewrite mk_fm; apply (eqm_weaken 32); [rewrite Hm1, Ham; lia | apply eqm_sym, Hv32]. }
  destruct (Nat.eqb to from && Z.eqb (wd F s) (Z.of_nat from)) eqn:E4.
  {
  apply andb_true_iff in E4. destruct E4 as [E4 E5]. apply Nat.eqb_eq in E4. apply Z.eqb_eq in E5. subst to.
  rewrite (extend_same s from signed (wdn_of s from E5)).
  destruct (get st s) as [y |] eqn:Eg; [| discriminate]. apply some_eq in H. subst x.
  exact (proj1 Hst s _ Eg). }
  destruct (Nat.eqb from (wdn s)) eqn:E5; [|discriminate].
  apply Nat.eqb_eq in E5.
  destruct (converted_form F st s from signed) as [lifted|] eqn:Elift.
  { apply some_eq in H. subst x.
    destruct (converted_form_sound s from signed lifted E5 Elift) as [Hwf Hsat].
    apply form_fact_ok; [apply wf_fat;[exact Hwf|lia]|].
    apply sat_trunc; [apply wf_fat;[exact Hwf|lia]|unfold fat;rewrite mk_fm;lia|].
    apply sat_fat; [exact Hwf|lia|exact Hsat]. }

  apply some_eq in H. subst x.
  set (e := if signed then ES else EZ).
  destruct (as_form_sound F E HE t st rs Hst Hrs s e) as [Has Ham].
  destruct (normalize_sound F E HE t st rs Hst Hrs _ _ (as_form_wf F E HE t st rs Hst Hrs s e) Has)
    as [Hw1 [Hm1 Hs1]].
  assert (Hv : atom_val wdn rs (s,e) = extend from signed (rs s)).
  { unfold e. destruct signed; cbn [atom_val extend]; [rewrite E5; reflexivity|].
    rewrite trunc_small; [reflexivity|rewrite E5; exact (rsr s)]. }
  rewrite Hv in Hs1.
  apply form_fact_ok; [apply wf_fat; [exact Hw1|lia]|].
  apply sat_trunc; [apply wf_fat; [exact Hw1|lia]|unfold fat; rewrite mk_fm; lia|].
  apply sat_fat; [exact Hw1|lia|exact Hs1].
Qed.

Lemma value_bfe : forall g w d a p ln x,
    1 <= Z.of_nat w <= 64 -> wd F d = Z.of_nat w -> wd F a = Z.of_nat w ->
    value F st (IBfe g w d a p ln) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (trunc ln (trunc w (rs a) / 2 ^ Z.of_nat p))).
Proof.
  intros g w d a p ln x Hw Hd Ha H. rewrite (wdn_of d w Hd). rewrite (trunc_reg a w Ha).
  assert (Heq : value F st (IBfe g w d a p ln) =
    if Z.of_nat p + Z.of_nat ln <=? wd F a then Some (FForm (fatom (Z.of_nat w) (a, EBf (Z.of_nat p) (Z.of_nat ln)) 1) 0 None)
    else Some (FForm (range_form (Z.of_nat w) 0 (match reg_range F st a with
                                                  | Some (_, y) => Z.min (2 ^ Z.of_nat ln - 1) (y / 2 ^ Z.of_nat p)
                                                  | None => 2 ^ Z.of_nat ln - 1 end)) 0 None)) by reflexivity.
  rewrite Heq in H. clear Heq.
  assert (Hp : 0 < 2 ^ Z.of_nat p) by (apply pow_nonneg_pos; lia).
  assert (Hl : 0 < 2 ^ Z.of_nat ln) by (apply pow_nonneg_pos; lia).
  pose proof (rsr a) as Hra.
  assert (Hv : trunc ln (rs a / 2 ^ Z.of_nat p) = (rs a / 2 ^ Z.of_nat p) mod 2 ^ Z.of_nat ln) by reflexivity.
  rewrite Hv.
  pose proof (Z.mod_pos_bound (rs a / 2 ^ Z.of_nat p) (2 ^ Z.of_nat ln) Hl) as Hm.
  destruct (Z.of_nat p + Z.of_nat ln <=? wd F a) eqn:E1.
  - apply some_eq in H. subst x.
    apply form_fact_ok; [unfold fatom, wf; rewrite mk_fm; lia |].
    apply sat_trunc; [unfold fatom, wf; rewrite mk_fm; lia | unfold fatom; rewrite mk_fm; lia |].
    change ((rs a / 2 ^ Z.of_nat p) mod 2 ^ Z.of_nat ln) with (atom_val wdn rs (a, EBf (Z.of_nat p) (Z.of_nat ln))).
    rewrite <- (Z.mul_1_l (atom_val wdn rs _)). apply sat_fatom. lia.
  - apply some_eq in H. subst x.
    apply ffact_fact. apply range_fact_ok; [lia | | apply aligned_nonpos; lia].
    destruct (reg_range F st a) as [[x0 y0] |] eqn:Er; [| lia].
    pose proof (rrs a x0 y0 Er) as Hr.
    assert (Hle : (rs a / 2 ^ Z.of_nat p) mod 2 ^ Z.of_nat ln <= rs a / 2 ^ Z.of_nat p)
      by (apply Z.mod_le; [apply Z.div_pos; lia | lia]).
    assert (rs a / 2 ^ Z.of_nat p <= y0 / 2 ^ Z.of_nat p) by (apply Z.div_le_mono; lia).
    split; [lia | apply Z.min_glb; lia].
Qed.

Lemma value_setp_raw : forall c w signed d a b x,
    wd F d = 1 -> Some (FPred [{|cc:=c;cw:=w;csg:=signed;ca:=a;cb:=b|}] true) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (bool_z (cmp_eval c (extend w signed (rs a)) (extend w signed (opv rs b))))).
Proof.
  intros c w signed d a b x Hd H. inversion H; subst x. clear H.
  rewrite (wdn_of d 1 Hd). cbn [fact_okv].
  destruct (cmp_eval c (extend w signed (rs a)) (extend w signed (opv rs b))) eqn:Eb; cbn [bool_z].
  - split; [intros _ | intros _ _; cbn; lia].
    constructor; [| constructor]. unfold cmp_holds. cbn [cc cw csg ca cb]. exact Eb.
  - split; [intros Hne; exfalso; apply Hne; reflexivity |].
    intros _ Hall. inversion Hall as [| k l Hk Hl]; subst. unfold cmp_holds in Hk. cbn [cc cw csg ca cb] in Hk.
    rewrite Eb in Hk. discriminate.
Qed.

Lemma opposite_predicate_sound : forall r x,
 opposite_predicate (get st r) = Some x ->
 fact_okv F t rs x (trunc 1 (bool_z (Z.eqb (rs r) 0))).
Proof.
 intros r x H. unfold opposite_predicate in H.
 destruct (get st r) as [[f al vr | al | cs exact | cs exact] |] eqn:Er; try discriminate;
 injection H as <-; pose proof (proj1 Hst r _ Er) as [Hyes Hno];
 destruct (Z.eqb_spec (rs r) 0) as [Hz | Hz]; cbn [fact_okv bool_z trunc].
 - split; [discriminate |]. intros He Hcs. exfalso. exact (Hno He Hcs Hz).
 - split; [intros _; exact (Hyes Hz) | intros; reflexivity].
 - split; [intros _; exact (Hyes Hz) | intros; discriminate].
 - split; [intros H; exfalso; apply H; reflexivity |]. intros He Hcs. exfalso. exact (Hz (Hno He Hcs)).
Qed.

Lemma bit_extend_zero : forall w signed v,
 (1 <= w)%nat -> 0 <= v <= 1 ->
 Z.eqb (extend w signed v) (extend w signed 0) = Z.eqb v 0.
Proof.
 intros w signed v Hw Hv.
 assert (Hpow : 2 <= 2 ^ Z.of_nat w) by (change 2 with (2 ^ 1) at 1; apply Z.pow_le_mono_r; lia).
 assert (Hhalf : 0 < 2 ^ (Z.of_nat w - 1)) by (apply pow_nonneg_pos;lia).
 assert (Ht : trunc w v = v) by (apply trunc_small;lia).
 assert (H0 : trunc w 0 = 0) by (apply trunc_small;lia).
 destruct signed; cbn [extend]; [|rewrite Ht,H0;reflexivity].
 unfold sext. rewrite Ht,H0.
 assert (Hz : (0 <? 2 ^ (Z.of_nat w - 1)) = true) by (apply Z.ltb_lt;lia).
 rewrite Hz. destruct (v <? 2 ^ (Z.of_nat w - 1)) eqn:Ev; [reflexivity|].
 apply Z.ltb_ge in Ev. assert (Hv0 : (v =? 0) = false) by (apply Z.eqb_neq;lia).
 rewrite Hv0. apply Z.eqb_neq. lia.
Qed.

Lemma copied_zero_predicate_sound : forall c w a b x signed,
 copied_zero_predicate F st c w a b = Some x ->
 fact_okv F t rs x (trunc 1 (bool_z (cmp_eval c (extend w signed (rs a)) (extend w signed (opv rs b))))).
Proof.
 intros c w a b x signed H. unfold copied_zero_predicate in H.
 destruct c; destruct b as [r|z]; try discriminate; destruct z; try discriminate;
 destruct (Nat.leb 1 w) eqn:Ew; try discriminate;
 destruct (copy_of F a (get st a)) as [p|] eqn:Ec; try discriminate;
 destruct (Z.eqb (wd F p) 1) eqn:Ep; try discriminate;
 apply Nat.leb_le in Ew; apply Z.eqb_eq in Ep;
 pose proof (copy_of_sound F E HE t st rs Hst Hrs a p Ec) as Hcopy;
 pose proof (rsr p) as Hp; change (Z.of_nat (wdn p)) with (wd F p) in Hp; rewrite Ep in Hp;
 cbn [cmp_eval opv]; rewrite <- Hcopy,bit_extend_zero by (try exact Ew;cbn in Hp;lia).
 - exact (opposite_predicate_sound p x H).
 - assert (Hvalue : trunc 1 (bool_z (negb (rs p =? 0))) = rs p).
   { destruct (Z.eqb_spec (rs p) 0) as [Hz|Hz].
     - rewrite Hz. reflexivity.
     - assert (rs p = 1) as -> by (cbn in Hp;lia). reflexivity. }
   rewrite Hvalue. exact (proj1 Hst p x H).
Qed.

Lemma value_setp : forall g c w signed d a b x,
 wd F d = 1 -> value F st (ISetp g c w signed d a b) = Some x ->
 fact_okv F t rs x (trunc (wdn d) (bool_z (cmp_eval c (extend w signed (rs a)) (extend w signed (opv rs b))))).
Proof.
 intros g c w signed d a b x Hd H. cbn [value] in H.
 destruct (copied_zero_predicate F st c w a b) as [recovered|] eqn:Ecopy.
 { injection H as <-. rewrite (wdn_of d 1 Hd). eapply copied_zero_predicate_sound. exact Ecopy. }
 destruct (Nat.eqb w 1 && Z.eqb (wd F a) 1) eqn:Hw.
 2: { eapply value_setp_raw; [exact Hd | exact H]. }
 apply andb_true_iff in Hw. destruct Hw as [Hwn Ha]. apply Nat.eqb_eq in Hwn. subst w. apply Z.eqb_eq in Ha.
 destruct c; destruct b as [r | z]; try (eapply value_setp_raw; [exact Hd | exact H]).
 destruct z; try (eapply value_setp_raw; [exact Hd | exact H]).
 destruct (opposite_predicate (get st a)) as [y |] eqn:Ey; [| eapply value_setp_raw; [exact Hd | exact H]].
 injection H as <-. rewrite (wdn_of d 1 Hd). cbn [opv].
 assert (Hv : cmp_eval CEq (extend 1 signed (rs a)) (extend 1 signed 0) = Z.eqb (rs a) 0).
 { pose proof (rsr a) as Hr. unfold wd in Ha. rewrite Ha in Hr.
   destruct (Z.eqb_spec (rs a) 0) as [Hz | Hz].
   - rewrite Hz. destruct signed; reflexivity.
   - assert (Hone : rs a = 1) by (cbn in Hr; lia). rewrite Hone. destruct signed; reflexivity. }
 rewrite Hv. exact (opposite_predicate_sound a y Ey).
Qed.

(* The predicate part of a register's fact: its comparisons and whether
   they are exact. *)
Definition pcmps (o : option fact) : list cmpf := match o with Some (FPred cs _) => cs | _ => [] end.
Definition pexact (o : option fact) : bool := match o with Some (FPred _ e) => e | _ => false end.

Lemma pred_union_eq : forall x y,
    pred_union x y = match pcmps x ++ pcmps y with [] => None | cs => Some (FPred cs (pexact x && pexact y)) end.
Proof. intros [[] |] [[] |]; cbn; try reflexivity; rewrite ?andb_false_r; reflexivity. Qed.

Lemma pred_fact : forall r,
    (rs r <> 0 -> Forall (cmp_holds rs) (pcmps (get st r))) /\
    (pexact (get st r) = true -> Forall (cmp_holds rs) (pcmps (get st r)) -> rs r <> 0).
Proof.
  intros r. destruct (get st r) as [[] |] eqn:Eg; cbn [pcmps pexact]; try (split; [intros; constructor | discriminate]).
  pose proof (proj1 Hst r _ Eg) as Hp. cbn [fact_ok fact_okv] in Hp. exact Hp.
Qed.

Lemma value_andp_raw : forall d a b x,
    wd F d = 1 -> pred_union (get st a) (get st b) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (bool_z (negb (Z.eqb (rs a) 0) && negb (Z.eqb (rs b) 0)))).
Proof.
  intros d a b x Hd H. rewrite (wdn_of d 1 Hd).
  rewrite pred_union_eq in H.
  destruct (pred_fact a) as [Hxa Hea]. destruct (pred_fact b) as [Hxb Heb].
  destruct (pcmps (get st a) ++ pcmps (get st b)) as [| k ks] eqn:Ec; [discriminate |].

  apply some_eq in H. subst x. cbn [fact_okv]. rewrite <- Ec.
  destruct (Z.eqb_spec (rs a) 0) as [Ha0 | Ha0], (Z.eqb_spec (rs b) 0) as [Hb0 | Hb0].
  - split; [intros Hne; exfalso; apply Hne; reflexivity |]. intros He Hall. apply andb_true_iff in He.
    apply Forall_app in Hall. exfalso. exact (Hea (proj1 He) (proj1 Hall) Ha0).
  - split; [intros Hne; exfalso; apply Hne; reflexivity |]. intros He Hall. apply andb_true_iff in He.
    apply Forall_app in Hall. exfalso. exact (Hea (proj1 He) (proj1 Hall) Ha0).
  - split; [intros Hne; exfalso; apply Hne; reflexivity |]. intros He Hall. apply andb_true_iff in He.
    apply Forall_app in Hall. exfalso. exact (Heb (proj2 He) (proj2 Hall) Hb0).
  - split; [intros _; apply Forall_app; split; [apply Hxa | apply Hxb]; assumption | intros _ _; discriminate].
Qed.

(* The comparisons that hold when register r's predicate fails. *)
Lemma fails_fact : forall r,
    (rs r = 0 -> Forall (cmp_holds rs) (fst (fails_cmps (get st r)))) /\
    (snd (fails_cmps (get st r)) = true -> Forall (cmp_holds rs) (fst (fails_cmps (get st r))) -> rs r = 0).
Proof.
  intros r. destruct (get st r) as [x |] eqn:Eg; [| split; [intros; constructor | discriminate]].
  pose proof (proj1 Hst r _ Eg) as Hp. unfold fact_ok in Hp.
  destruct x as [f al vr | k | cs ex | cs ex]; cbn [fails_cmps fst snd];
    try (split; [intros; constructor | discriminate]); [| exact Hp].
  destruct cs as [| k [| k' cs]]; try (split; [intros; constructor | discriminate]).
  destruct ex; cbn [fst snd]; [| split; [intros; constructor | discriminate]].
  cbn [fact_okv] in Hp. destruct Hp as [H1 H2]. split.
  - intros H0. constructor; [| constructor]. apply negate_holds. intros Hk.
    exact (H2 eq_refl (Forall_cons _ Hk (Forall_nil _)) H0).
  - intros _ Hall. inversion Hall as [| ? ? Hn _]. subst.
    destruct (Z.eqb_spec (rs r) 0) as [| Hne]; [assumption |].
    specialize (H1 Hne). inversion H1 as [| ? ? Hk _]. subst.
    exfalso. exact (proj1 (negate_holds rs k) Hn Hk).
Qed.

Lemma value_orp_raw : forall d a b x,
    wd F d = 1 -> pred_disj (get st a) (get st b) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (bool_z (negb (Z.eqb (rs a) 0) || negb (Z.eqb (rs b) 0)))).
Proof.
  intros d a b x Hd H. rewrite (wdn_of d 1 Hd). unfold pred_disj in H.
  destruct (fails_fact a) as [Hxa Hea]. destruct (fails_fact b) as [Hxb Heb].
  destruct (fails_cmps (get st a)) as [ca ea], (fails_cmps (get st b)) as [cb eb]. cbn [fst snd] in *.
  destruct (ca ++ cb) as [| k ks] eqn:Ec; [discriminate |].
  apply some_eq in H. subst x. cbn [fact_okv]. rewrite <- Ec.
  destruct (Z.eqb_spec (rs a) 0) as [Ha0 | Ha0], (Z.eqb_spec (rs b) 0) as [Hb0 | Hb0]; cbn.
  - split; [intros _; apply Forall_app; split; [apply Hxa | apply Hxb]; assumption | intros _ _; reflexivity].
  - split; [discriminate |]. intros He Hall. apply andb_true_iff in He.
    apply Forall_app in Hall. exfalso. exact (Hb0 (Heb (proj2 He) (proj2 Hall))).
  - split; [discriminate |]. intros He Hall. apply andb_true_iff in He.
    apply Forall_app in Hall. exfalso. exact (Ha0 (Hea (proj1 He) (proj1 Hall))).
  - split; [discriminate |]. intros He Hall. apply andb_true_iff in He.
    apply Forall_app in Hall. exfalso. exact (Ha0 (Hea (proj1 He) (proj1 Hall))).
Qed.

Lemma known_bit_sound : forall r b,
 known_bit F st r = Some b -> rs r = if b then 1 else 0.
Proof.
 intros r b H. unfold known_bit in H.
 destruct (fact_range F r (get st r)) as [[lo hi] |] eqn:Er; [| discriminate].
 pose proof (fact_range_sound F E HE t st rs Hst Hrs r lo hi Er) as Hr.
 destruct (Z.eqb lo 0 && Z.eqb hi 0) eqn:Ez.
 - injection H as <-. apply andb_true_iff in Ez. destruct Ez as [El Eh]. apply Z.eqb_eq in El,Eh. lia.
 - destruct (Z.eqb lo 1 && Z.eqb hi 1) eqn:Eo; [| discriminate].
   injection H as <-. apply andb_true_iff in Eo. destruct Eo as [El Eh]. apply Z.eqb_eq in El,Eh. lia.
Qed.

Lemma bit_fact_sound : forall b x,
 bit_fact b = Some x -> fact_okv F t rs x (trunc 1 (bool_z b)).
Proof.
 intros b x H. injection H as <-. apply form_fact_ok.
 - unfold fconst, wf; cbn; lia.
 - apply sat_fconst. destruct b; apply eqm_refl.
Qed.

Lemma predicate_bit_value : forall r,
 wd F r = 1 -> trunc 1 (bool_z (negb (Z.eqb (rs r) 0))) = rs r.
Proof.
 intros r Hw. pose proof (rsr r) as Hr. unfold wd in Hw. rewrite Hw in Hr.
 destruct (Z.eqb_spec (rs r) 0) as [-> | Hz]; cbn; lia.
Qed.

Lemma value_andp : forall g d a b x,
 wd F d = 1 -> wd F a = 1 -> wd F b = 1 -> value F st (IAndp g d a b) = Some x ->
 fact_okv F t rs x (trunc (wdn d) (bool_z (negb (Z.eqb (rs a) 0) && negb (Z.eqb (rs b) 0)))).
Proof.
 intros g d a b x Hd Ha Hb H. cbn [value] in H. rewrite (wdn_of d 1 Hd).
 destruct (known_bit F st a) as [ba |] eqn:Ea; destruct (known_bit F st b) as [bb |] eqn:Eb.
 - pose proof (known_bit_sound a ba Ea) as Hva. pose proof (known_bit_sound b bb Eb) as Hvb.
   destruct ba,bb; cbn in H; rewrite Hva,Hvb; cbn;
   try (exact (bit_fact_sound false x H));
   try (exact (bit_fact_sound true x H));
   try (rewrite <- Hvb; exact (proj1 Hst b x H));
   try (rewrite <- Hva; exact (proj1 Hst a x H)).
 - pose proof (known_bit_sound a ba Ea) as Hva. destruct ba; cbn in H; rewrite Hva; cbn;
   try (exact (bit_fact_sound false x H));
   try (exact (bit_fact_sound true x H));
   try (rewrite predicate_bit_value by exact Hb; exact (proj1 Hst b x H)).
 - pose proof (known_bit_sound b bb Eb) as Hvb. destruct bb; cbn in H; rewrite Hvb;
   rewrite ?andb_true_r, ?andb_false_r, ?orb_true_r, ?orb_false_r;
   try (exact (bit_fact_sound false x H));
   try (exact (bit_fact_sound true x H));
   try (rewrite predicate_bit_value by exact Ha; exact (proj1 Hst a x H)).
 - rewrite <- (wdn_of d 1 Hd). exact (value_andp_raw d a b x Hd H).
Qed.

Lemma value_orp : forall g d a b x,
 wd F d = 1 -> wd F a = 1 -> wd F b = 1 -> value F st (IOrp g d a b) = Some x ->
 fact_okv F t rs x (trunc (wdn d) (bool_z (negb (Z.eqb (rs a) 0) || negb (Z.eqb (rs b) 0)))).
Proof.
 intros g d a b x Hd Ha Hb H. cbn [value] in H. rewrite (wdn_of d 1 Hd).
 destruct (known_bit F st a) as [ba |] eqn:Ea; destruct (known_bit F st b) as [bb |] eqn:Eb.
 - pose proof (known_bit_sound a ba Ea) as Hva. pose proof (known_bit_sound b bb Eb) as Hvb.
   destruct ba,bb; cbn in H; rewrite Hva,Hvb; cbn;
   try (exact (bit_fact_sound false x H));
   try (exact (bit_fact_sound true x H));
   try (rewrite <- Hvb; exact (proj1 Hst b x H));
   try (rewrite <- Hva; exact (proj1 Hst a x H)).
 - pose proof (known_bit_sound a ba Ea) as Hva. destruct ba; cbn in H; rewrite Hva; cbn;
   try (exact (bit_fact_sound false x H));
   try (exact (bit_fact_sound true x H));
   try (rewrite predicate_bit_value by exact Hb; exact (proj1 Hst b x H)).
 - pose proof (known_bit_sound b bb Eb) as Hvb. destruct bb; cbn in H; rewrite Hvb;
   rewrite ?andb_true_r, ?andb_false_r, ?orb_true_r, ?orb_false_r;
   try (exact (bit_fact_sound false x H));
   try (exact (bit_fact_sound true x H));
   try (rewrite predicate_bit_value by exact Ha; exact (proj1 Hst a x H)).
 - rewrite <- (wdn_of d 1 Hd). exact (value_orp_raw d a b x Hd H).
Qed.

(* The exclusive or holds when exactly one of a and b holds, and then each
   comparison that the facts of both list holds. *)
Lemma value_xorp : forall g d a b x,
    wd F d = 1 -> value F st (IXorp g d a b) = Some x ->
    fact_okv F t rs x (trunc (wdn d) (bool_z (xorb (negb (Z.eqb (rs a) 0)) (negb (Z.eqb (rs b) 0))))).
Proof.
  intros g d a b x Hd H. cbn [value] in H. rewrite (wdn_of d 1 Hd). unfold pred_common in H.
  destruct (get st a) as [[f1 al1 vr1 | k1 | ca ea | c1 e1] |] eqn:Ea; try discriminate.
  destruct (get st b) as [[f2 al2 vr2 | k2 | cb eb | c2 e2] |] eqn:Eb; try discriminate.
  destruct (filter (fun k => existsb (cmpf_eqb k) cb) ca) as [| k ks] eqn:Ec; [discriminate |].
  apply some_eq in H. subst x. cbn [fact_okv]. rewrite <- Ec.
  pose proof (proj1 Hst a _ Ea) as Hpa. pose proof (proj1 Hst b _ Eb) as Hpb.
  cbn [fact_ok fact_okv] in Hpa, Hpb.
  destruct (Z.eqb_spec (rs a) 0) as [Ha0 | Ha0], (Z.eqb_spec (rs b) 0) as [Hb0 | Hb0].
  - split; [intros Hne; exfalso; apply Hne; reflexivity | discriminate].
  - split; [intros _ | discriminate]. apply Forall_forall. intros k' Hk'.
    apply filter_In in Hk'. destruct Hk' as [_ Hkb].
    apply existsb_exists in Hkb. destruct Hkb as [l [Hl Hkl]]. apply cmpf_eqb_eq in Hkl. subst l.
    exact (proj1 (Forall_forall _ _) (proj1 Hpb Hb0) k' Hl).
  - split; [intros _ | discriminate]. apply Forall_forall. intros k' Hk'.
    apply filter_In in Hk'. destruct Hk' as [Hka _].
    exact (proj1 (Forall_forall _ _) (proj1 Hpa Ha0) k' Hka).
  - split; [intros Hne; exfalso; apply Hne; reflexivity | discriminate].
Qed.

Lemma sext_lower : forall n v, (1 <= n)%nat -> - 2 ^ (Z.of_nat n - 1) <= sext n v.
Proof.
  intros n v Hn. unfold sext. pose proof (trunc_range n v) as Ht.
  assert (Hp : 2 ^ Z.of_nat n = 2 * 2 ^ (Z.of_nat n - 1)).
  { rewrite <- Z.pow_succ_r by lia. f_equal. lia. }
  assert (0 < 2 ^ (Z.of_nat n - 1)) by (apply Z.pow_pos_nonneg; lia).
  destruct (Z.ltb_spec (trunc n v) (2 ^ (Z.of_nat n - 1))); lia.
Qed.

Lemma opnd_range_sound : forall o w signed lo hi,
    (forall r, o = OReg r -> wd F r = Z.of_nat w) ->
    opnd_range F st o (Z.of_nat w) signed = Some (lo, hi) -> lo <= extend w signed (opv rs o) <= hi.
Proof.
  intros [r | z] w signed lo hi Hw H; unfold opnd_range in H; cbn [opv].
  - pose proof (Hw r eq_refl) as Hr. destruct signed; cbn [extend].
    + rewrite <- (wdn_of r w Hr). exact (srs r lo hi H).
    + rewrite (trunc_reg r w Hr). exact (rrs r lo hi H).
  - rewrite Nat2Z.id in H. apply some_eq in H. injection H as H1 H2. lia.
Qed.

(* Extending a value to width w keeps an alignment of at most w. *)
Lemma aligned_extend : forall k w signed v, k <= Z.of_nat w -> aligned k v -> aligned k (extend w signed v).
Proof.
  intros k w signed v Hk H.
  destruct (Z.le_gt_cases k 0) as [Hk0 | Hk0]; [apply aligned_nonpos; exact Hk0 |].
  assert (Ht : aligned k (trunc w v)) by (apply aligned_trunc; [exact Hk | exact H]).
  destruct signed; cbn [extend]; [| exact Ht].
  unfold sext. cbv zeta. destruct (Z.ltb (trunc w v) (2 ^ (Z.of_nat w - 1))); [exact Ht |].
  replace (trunc w v - 2 ^ Z.of_nat w) with (trunc w v + - 2 ^ Z.of_nat w) by ring.
  apply aligned_add; [exact Ht |]. apply aligned_opp. unfold aligned.
  exists (2 ^ (Z.of_nat w - Z.max 0 k)). rewrite <- Z.pow_add_r by lia. f_equal. lia.
Qed.

Lemma value_minmax : forall g is_min signed w d a b x,
    1 <= Z.of_nat w <= 64 -> wd F d = Z.of_nat w -> wd F a = Z.of_nat w ->
    (forall r, b = OReg r -> wd F r = Z.of_nat w) ->
    value F st (IMinMax g is_min signed w d a b) = Some x ->
    fact_okv F t rs x
      (trunc (wdn d) (if is_min then Z.min (extend w signed (rs a)) (extend w signed (opv rs b))
                      else Z.max (extend w signed (rs a)) (extend w signed (opv rs b)))).
Proof.
  intros g is_min signed w d a b x Hw Hd Ha Hb H. cbn [value] in H. rewrite (wdn_of d w Hd).
  assert (HaR : forall r, OReg a = OReg r -> wd F r = Z.of_nat w) by (intros r Hr; injection Hr as <-; exact Ha).
  set (al := Z.min (Z.min (reg_align F st a) (opnd_align F st b)) (Z.of_nat w)) in H.
  (* the result is one of the operands, each extended to the width *)
  assert (Hal : aligned al (trunc w (if is_min then Z.min (extend w signed (rs a)) (extend w signed (opv rs b))
                                     else Z.max (extend w signed (rs a)) (extend w signed (opv rs b))))).
  { apply aligned_trunc; [apply Z.le_min_r |].
    assert (HA : aligned al (extend w signed (rs a))).
    { apply aligned_extend; [apply Z.le_min_r |]. eapply aligned_mono; [| apply ras]. lia. }
    assert (HB : aligned al (extend w signed (opv rs b))).
    { apply aligned_extend; [apply Z.le_min_r |]. eapply aligned_mono; [| apply opnd_align_sound]. lia. }
    destruct is_min; [destruct (Z.min_spec (extend w signed (rs a)) (extend w signed (opv rs b))) as [[_ ->] | [_ ->]]
                     | destruct (Z.max_spec (extend w signed (rs a)) (extend w signed (opv rs b))) as [[_ ->] | [_ ->]]];
      assumption. }
  assert (Hfin : forall lo hi v, lo <= v <= hi -> aligned al (trunc w v) ->
                   fact_okv F t rs (FForm (range_form (Z.of_nat w) lo hi) al None) (trunc w v)).
  { intros lo hi v Hv Hv'. apply ffact_fact. apply range_fact_ok; [lia | exact Hv | exact Hv']. }
  pose proof (sext_lower w (rs a) ltac:(lia)) as Hla. pose proof (sext_lower w (opv rs b) ltac:(lia)) as Hlb.
  pose proof (trunc_range w (rs a)) as Hua. pose proof (trunc_range w (opv rs b)) as Hub.
  (* a value read at the width and truncated back is its truncation *)
  assert (Htr : forall v, trunc w (extend w signed v) = trunc w v).
  { intros v. assert (Hp : 0 < 2 ^ Z.of_nat w) by (apply Z.pow_pos_nonneg; lia).
    destruct signed; unfold extend, sext, trunc.
    - destruct (Z.ltb_spec (v mod 2 ^ Z.of_nat w) (2 ^ (Z.of_nat w - 1))).
      + rewrite Z.mod_mod by lia. reflexivity.
      + replace (v mod 2 ^ Z.of_nat w - 2 ^ Z.of_nat w) with (v mod 2 ^ Z.of_nat w + (-1) * 2 ^ Z.of_nat w) by ring.
        rewrite Z.mod_add, Z.mod_mod by lia. reflexivity.
    - rewrite Z.mod_mod by lia. reflexivity. }
  set (m := if is_min then Z.min (extend w signed (rs a)) (extend w signed (opv rs b))
            else Z.max (extend w signed (rs a)) (extend w signed (opv rs b))) in *.
  (* the operand that the ranges decide *)
  assert (HdA : m = extend w signed (rs a) -> fact_okv F t rs (FForm (normalize F st (opnd_form F st (OReg a) (Z.of_nat w))) 0 None) (trunc w m)).
  { intros Hm. rewrite Hm, Htr. destruct (operand_a a (Z.of_nat w) ltac:(lia)) as [Hw1 [_ Hs1]].
    rewrite (trunc_small w (rs a)) by (pose proof (rsr a); unfold wd in Ha; rewrite Ha in *; lia).
    exact (form_fact_ok _ _ Hw1 Hs1). }
  assert (HdB : m = extend w signed (opv rs b) -> fact_okv F t rs (FForm (normalize F st (opnd_form F st b (Z.of_nat w))) 0 None) (trunc w m)).
  { intros Hm. rewrite Hm, Htr. destruct (operand_b b (Z.of_nat w) ltac:(lia)) as [Hw1 [Hm1 [e [He Hs1]]]].
    apply form_fact_ok; [exact Hw1 |]. exists e. split; [exact He |].
    apply (eqm_trans _ _ (opv rs b)); [| exact Hs1].
    apply (eqm_weaken (Z.of_nat w)); [unfold wf in Hw1; lia |]. unfold eqm, trunc. apply Z.mod_mod.
    apply Z.pow_nonzero; lia. }
  destruct (opnd_range F st (OReg a) (Z.of_nat w) signed) as [[a0 a1] |] eqn:Ea;
    [pose proof (opnd_range_sound (OReg a) w signed a0 a1 HaR Ea) as Hxa |];
    (destruct (opnd_range F st b (Z.of_nat w) signed) as [[b0 b1] |] eqn:Eb;
     [pose proof (opnd_range_sound b w signed b0 b1 Hb Eb) as Hxb |]).
  1: { destruct (if is_min then a1 <=? b0 else b1 <=? a0) eqn:D1.
       { apply some_eq in H. subst x. apply HdA. unfold m.
         destruct is_min; [apply Z.leb_le in D1; apply Z.min_l | apply Z.leb_le in D1; apply Z.max_l];
           cbn [opv] in *; lia. }
       destruct (if is_min then b1 <=? a0 else a1 <=? b0) eqn:D2.
       { apply some_eq in H. subst x. apply HdB. unfold m.
         destruct is_min; [apply Z.leb_le in D2; apply Z.min_r | apply Z.leb_le in D2; apply Z.max_r];
           cbn [opv] in *; lia. }
       clear D1 D2. unfold m in *.
       destruct is_min, signed; apply some_eq in H; subst x; apply Hfin;
         try exact Hal; try (cbn [extend] in Hxa); try (cbn [extend] in Hxb); cbn [opv extend] in *; lia. }
  all: unfold m in *; destruct is_min, signed; cbn [andb] in H; try discriminate; apply some_eq in H; subst x; apply Hfin;
    try exact Hal; try (cbn [extend] in Hxa); try (cbn [extend] in Hxb); cbn [opv extend] in *; lia.
Qed.

Lemma value_div : forall g rm sg w d a b x,
    1 <= Z.of_nat w <= 64 -> wd F d = Z.of_nat w -> wd F a = Z.of_nat w ->
    (forall r, b = OReg r -> wd F r = Z.of_nat w) ->
    value F st (IDiv g rm sg w d a b) = Some x ->
    fact_okv F t rs x
      (trunc (wdn d) (match div_eval rm sg w (rs a) (opv rs b) with Some v => v | None => e_arb E t d end)).
Proof.
  intros g rm sg w d a b x Hw Hd Ha Hb H. cbn [value] in H. unfold div_fact in H. rewrite (wdn_of d w Hd).
  assert (HaR : forall r, OReg a = OReg r -> wd F r = Z.of_nat w) by (intros r Hr; injection Hr as <-; exact Ha).
  assert (Hfin : forall lo hi q, lo <= q <= hi ->
                   fact_okv F t rs (FForm (range_form (Z.of_nat w) lo hi) 0 None) (trunc w q)).
  { intros lo hi q Hq. apply ffact_fact. apply range_fact_ok; [lia | exact Hq | apply aligned_nonpos; lia]. }
  destruct (const_of F st b (Z.of_nat w) sg) as [k |] eqn:Ek.
  2: { (* a remainder by a register of positive range *)
    destruct (opnd_range F st (OReg a) (Z.of_nat w) sg) as [[x0 y0] |] eqn:Ea; [| discriminate].
    destruct (rm && (0 <=? x0)) eqn:Erm; [| discriminate].
    apply andb_true_iff in Erm. destruct Erm as [-> Hx0]. apply Z.leb_le in Hx0.
    destruct (opnd_range F st b (Z.of_nat w) sg) as [[k0 k1] |] eqn:Eb; [| discriminate].
    destruct (Z.ltb_spec 0 k0) as [Hk0 |]; [| discriminate].
    apply some_eq in H. subst x.
    pose proof (opnd_range_sound b w sg k0 k1 Hb Eb) as Hxb.
    pose proof (opnd_range_sound (OReg a) w sg x0 y0 HaR Ea) as Hxa. cbn [opv] in Hxa.
    unfold div_eval.
    destruct (Z.eqb_spec (extend w sg (opv rs b)) 0) as [| _]; [lia |].
    destruct (Z.eqb_spec (extend w sg (opv rs b)) (-1)) as [| _]; [lia |].
    replace (sg && false) with false by (destruct sg; reflexivity). cbn [andb orb].
    apply Hfin.
    pose proof (Z.rem_bound_pos (extend w sg (rs a)) (extend w sg (opv rs b)) ltac:(lia) ltac:(lia)).
    pose proof (Z.rem_le (extend w sg (rs a)) (extend w sg (opv rs b)) ltac:(lia) ltac:(lia)). lia. }
  destruct (opnd_range F st (OReg a) (Z.of_nat w) sg) as [[x0 y0] |] eqn:Ea; [| discriminate].
  destruct (Z.ltb_spec 0 k) as [Hk |]; [| discriminate].
  (* the divisor, as the division reads it, is k *)
  assert (Hbk : extend w sg (opv rs b) = k).
  { unfold const_of in Ek. destruct (opnd_range F st b (Z.of_nat w) sg) as [[p q] |] eqn:Eb; [| discriminate].
    destruct (Z.eqb_spec p q) as [-> |]; [| discriminate]. apply some_eq in Ek. subst k.
    pose proof (opnd_range_sound b w sg q q Hb Eb). lia. }
  pose proof (opnd_range_sound (OReg a) w sg x0 y0 HaR Ea) as Hxa. cbn [opv] in Hxa.
  unfold div_eval. rewrite Hbk.
  destruct (Z.eqb_spec k 0) as [| _]; [lia |].
  destruct (Z.eqb_spec k (-1)) as [| _]; [lia |].
  replace (sg && false) with false by (destruct sg; reflexivity). cbn [andb orb].
  set (v := extend w sg (rs a)) in *.
  destruct rm.
  - destruct (Z.leb_spec 0 x0) as [Hx0 | Hx0]; apply some_eq in H; subst x; apply Hfin.
    + pose proof (Z.rem_bound_pos v k ltac:(lia) ltac:(lia)).
      pose proof (Z.rem_le v k ltac:(lia) ltac:(lia)). lia.
    + pose proof (Z.rem_bound_abs v k ltac:(lia)) as Hr. rewrite (Z.abs_eq k) in Hr by lia.
      destruct (Z.abs_spec (Z.rem v k)) as [[_ Ha'] | [_ Ha']]; lia.
  - apply some_eq in H. subst x. apply Hfin.
    split; apply Z.quot_le_mono; lia.
Qed.

(* The fact the checker gives an exact instruction's result holds of the
   value the instruction writes. *)
Theorem value_sound : forall i d x,
    wf_instr F i = true -> dest i = Some d -> value F st i = Some x ->
    fact_okv F t rs x (trunc (wdn d) (result (e_mem E) (e_arb E) (e_shm E) t rs i)).
Proof.
  intros i d x Hwf Hd H.
  assert (Hw : forall r, 1 <= wd F r <= 64) by exact wdr.
  assert (Hop : forall o w, opnd_width_ok F o w = true -> forall r, o = OReg r -> wd F r = w).
  { intros [r' | z] w Ho r Hr; [injection Hr as <-; apply Z.eqb_eq; exact Ho | discriminate]. }
  destruct i as [g d' s | g d' z | g o w d' a b | g w d' a b c | g signed d' a b c | g to from signed d' s
                 | g w d' a p ln | g c w signed d' a b | g d' a b | g d' a b | g is_min signed w d' a b | g w d' a b p
                 | g gen bits d' a off | g k gen a off width v fr ds | g ds | g target | g | | g rm sg w d' a b
                 | g bits d' a off | g bits d' a off fr | g a off width v fr | vs | g a off width | g d' a b];
    cbn [dest] in Hd; try discriminate; injection Hd as ->;
    unfold wf_instr in Hwf; cbn [dest guard_of] in Hwf; rewrite !andb_true_iff in Hwf; cbn [result].
  - destruct Hwf as [[_ Hm] _]. apply Z.eqb_eq in Hm. exact (value_mov g d s x Hm H).
  - exact (value_imm g d z x H).
  - destruct Hwf as [[_ [[Hd Ha] Hb]] _]. apply Z.eqb_eq in Hd, Ha. pose proof (Hw d).
    apply (value_bin g o w d a b x); [| exact Hd | exact H].
    split; [lia |]. split; [exact Ha |].
    intros r' Hr'. destruct o; exact (Hop b _ Hb r' Hr').
  - destruct Hwf as [[_ [[[Hd _] _] _]] _]. apply Z.eqb_eq in Hd. pose proof (Hw d).
    apply (value_mad g w d a b c x); [lia | exact Hd | exact H].
  - destruct Hwf as [[_ [[[Hd Ha] Hb] _]] _]. apply Z.eqb_eq in Hd, Ha.
    exact (value_wide g signed d a b c x Hd Ha (Hop b 32 Hb) H).
  - destruct Hwf as [[_ [Hd Hs]] _]. apply Z.eqb_eq in Hd. apply Z.leb_le in Hs.
    exact (value_cvt g to from signed d s x Hd Hs H).
  - destruct Hwf as [[_ [[[Hd Ha] _] _]] _]. apply Z.eqb_eq in Hd, Ha. pose proof (Hw d).
    apply (value_bfe g w d a p ln x); [lia | exact Hd | exact Ha | exact H].
  - destruct Hwf as [[_ [[Hd _] _]] _]. apply Z.eqb_eq in Hd. exact (value_setp g c w signed d a b x Hd H).
  - destruct Hwf as [[_ [[Hd Ha] Hb]] _]. apply Z.eqb_eq in Hd,Ha,Hb. exact (value_andp g d a b x Hd Ha Hb H).
  - destruct Hwf as [[_ [[Hd Ha] Hb]] _]. apply Z.eqb_eq in Hd,Ha,Hb. exact (value_orp g d a b x Hd Ha Hb H).
  - destruct Hwf as [[_ [[Hd Ha] Hb]] _]. apply Z.eqb_eq in Hd, Ha. pose proof (Hw d).
    apply (value_minmax g is_min signed w d a b x); [lia | exact Hd | exact Ha | exact (Hop b _ Hb) | exact H].
  - destruct Hwf as [[_ [[Hd Ha] Hb]] _]. apply Z.eqb_eq in Hd, Ha. pose proof (Hw d).
    apply (value_div g rm sg w d a b x); [lia | exact Hd | exact Ha | exact (Hop b _ Hb) | exact H].
  - destruct Hwf as [[_ [[Hd _] _]] _]. apply Z.eqb_eq in Hd. exact (value_xorp g d a b x Hd H).
Qed.

End Ops.
