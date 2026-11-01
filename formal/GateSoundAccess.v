(*
  Soundness of the access checks: an access the check accepts, made while
  its guard holds from registers the state describes, stays in its region.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import ZArith.Znumtheory.
From Coq Require Import PArith.PArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
From Coq Require Import Sorting.Sorted.
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

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

(* ------------------------------------------------------------------ *)
(* Canonical term lists name each atom once.                           *)

Lemma key_ltb_spec : forall a b c a' b' c',
    key_ltb (a, b, c) (a', b', c') = true <-> (a < a' \/ (a = a' /\ (b < b' \/ (b = b' /\ c < c')))).
Proof.
  intros. unfold key_ltb.
  rewrite orb_true_iff, andb_true_iff, orb_true_iff, andb_true_iff, !Z.ltb_lt, !Z.eqb_eq. tauto.
Qed.

Lemma ext_key_inj : forall e f, ext_key e = ext_key f -> e = f.
Proof. intros [] [] H; cbn in H; inversion H; reflexivity. Qed.

Lemma atom_ltb_spec : forall r e r' e',
    atom_ltb (r, e) (r', e') = true <->
    ((r < r')%positive \/ (r = r' /\ key_ltb (ext_key e) (ext_key e') = true)).
Proof.
  intros. unfold atom_ltb. cbn [fst snd].
  rewrite orb_true_iff, andb_true_iff, Pos.ltb_lt, Pos.eqb_eq. tauto.
Qed.

Lemma key_ltb_irrefl : forall k, key_ltb k k = false.
Proof.
  intros [[a b] c]. destruct (key_ltb (a, b, c) (a, b, c)) eqn:E; [| reflexivity].
  apply key_ltb_spec in E. lia.
Qed.

Lemma key_ltb_trans : forall k1 k2 k3, key_ltb k1 k2 = true -> key_ltb k2 k3 = true -> key_ltb k1 k3 = true.
Proof.
  intros [[a b] c] [[a' b'] c'] [[a'' b''] c''] H1 H2.
  apply key_ltb_spec in H1, H2. apply key_ltb_spec. lia.
Qed.

Lemma key_ltb_total : forall k1 k2, k1 <> k2 -> key_ltb k1 k2 = false -> key_ltb k2 k1 = true.
Proof.
  intros [[a b] c] [[a' b'] c'] Hne H. apply key_ltb_spec.
  destruct (key_ltb (a, b, c) (a', b', c')) eqn:E; [discriminate |].
  assert (Hn : ~ (a < a' \/ (a = a' /\ (b < b' \/ (b = b' /\ c < c'))))).
  { intros Hc. apply key_ltb_spec in Hc. congruence. }
  assert (Hd : a <> a' \/ b <> b' \/ c <> c').
  { destruct (Z.eq_dec a a'), (Z.eq_dec b b'), (Z.eq_dec c c'); subst; try tauto. }
  lia.
Qed.

Lemma atom_ltb_irrefl : forall x, atom_ltb x x = false.
Proof.
  intros [r e]. destruct (atom_ltb (r, e) (r, e)) eqn:E; [| reflexivity].
  apply atom_ltb_spec in E. destruct E as [E | [_ E]]; [lia | rewrite key_ltb_irrefl in E; discriminate].
Qed.

Lemma atom_ltb_trans : forall x y z, atom_ltb x y = true -> atom_ltb y z = true -> atom_ltb x z = true.
Proof.
  intros [r e] [r' e'] [r'' e''] H1 H2. apply atom_ltb_spec in H1, H2. apply atom_ltb_spec.
  destruct H1 as [H1 | [-> H1]], H2 as [H2 | [-> H2]]; try (left; lia).
  right. split; [reflexivity | exact (key_ltb_trans _ _ _ H1 H2)].
Qed.

Lemma atom_ltb_total : forall x y, atom_eqb x y = false -> atom_ltb x y = false -> atom_ltb y x = true.
Proof.
  intros [r e] [r' e'] Hne Hlt. apply atom_ltb_spec.
  assert (Hn : ~ ((r < r')%positive \/ (r = r' /\ key_ltb (ext_key e) (ext_key e') = true))).
  { intros Hc. apply atom_ltb_spec in Hc. congruence. }
  destruct (Pos.lt_total r r') as [H | [H | H]]; [tauto | subst r' | left; exact H].
  right. split; [reflexivity |]. apply key_ltb_total.
  - intros Hk. apply ext_key_inj in Hk. subst e'.
    assert (atom_eqb (r, e) (r, e) = true) by (apply atom_eqb_eq; reflexivity). congruence.
  - destruct (key_ltb (ext_key e) (ext_key e')) eqn:E; [tauto | reflexivity].
Qed.

Definition tlt (p q : atom * Z) : Prop := atom_ltb (fst p) (fst q) = true.

Lemma tinsert_atoms : forall x c ts y, In y (map fst (tinsert x c ts)) -> y = x \/ In y (map fst ts).
Proof.
  intros x c ts. induction ts as [| [z d] ts IH]; intros y H; cbn [tinsert map In] in *.
  - destruct H as [H | []]. left. symmetry. exact H.
  - destruct (atom_eqb x z) eqn:E.
    + cbn [map In fst] in H. right. exact H.
    + destruct (atom_ltb x z); cbn [map In fst] in H.
      * destruct H as [H | H]; [left; symmetry; exact H | right; exact H].
      * destruct H as [H | H]; [right; left; exact H |]. destruct (IH y H) as [H' | H']; [left | right; right]; exact H'.
Qed.

Lemma tinsert_sorted : forall x c ts, StronglySorted tlt ts -> StronglySorted tlt (tinsert x c ts).
Proof.
  intros x c ts. induction ts as [| [z d] ts IH]; intros Hs; cbn [tinsert].
  - constructor; constructor.
  - inversion Hs as [| a l Hs' Hf]; subst.
    destruct (atom_eqb x z) eqn:E.
    + constructor; [exact Hs' |]. rewrite Forall_forall in Hf |- *. intros q Hq. exact (Hf q Hq).
    + destruct (atom_ltb x z) eqn:El.
      * constructor; [exact Hs |]. constructor; [exact El |].
        rewrite Forall_forall in Hf |- *. intros q Hq. exact (atom_ltb_trans _ _ _ El (Hf q Hq)).
      * constructor; [exact (IH Hs') |]. rewrite Forall_forall in Hf |- *. intros [y e'] Hq.
        assert (Hy : In y (map fst (tinsert x c ts))) by (apply in_map_iff; exists (y, e'); auto).
        destruct (tinsert_atoms x c ts y Hy) as [-> | Hin].
        -- unfold tlt. cbn [fst]. exact (atom_ltb_total _ _ E El).
        -- apply in_map_iff in Hin. destruct Hin as [[y' e''] [Hy' Hin]]. cbn [fst] in Hy'. subst y'.
           exact (Hf _ Hin).
Qed.

Lemma tsort_sorted : forall ts, StronglySorted tlt (tsort ts).
Proof.
  induction ts as [| [x c] ts IH]; cbn [tsort]; [constructor | apply tinsert_sorted; exact IH].
Qed.

Lemma treduce_atoms : forall m ts y, In y (map fst (treduce m ts)) -> In y (map fst ts).
Proof.
  intros m ts. induction ts as [| [x c] ts IH]; intros y H; cbn [treduce] in H; [exact H |]. rewrite pow2_eq in H.
  destruct (Z.eqb (rep m c mod 2 ^ m) 0); cbn [map In fst] in H |- *.
  - right. exact (IH y H).
  - destruct H as [H | H]; [left; exact H | right; exact (IH y H)].
Qed.

Lemma treduce_sorted : forall m ts, StronglySorted tlt ts -> StronglySorted tlt (treduce m ts).
Proof.
  intros m ts. induction ts as [| [x c] ts IH]; intros Hs; cbn [treduce]; [constructor |]. rewrite pow2_eq.
  inversion Hs as [| a l Hs' Hf]; subst.
  destruct (Z.eqb (rep m c mod 2 ^ m) 0); [exact (IH Hs') |].
  constructor; [exact (IH Hs') |]. rewrite Forall_forall in Hf |- *. intros [y e'] Hq.
  assert (Hy : In y (map fst (treduce m ts))) by (apply in_map_iff; exists (y, e'); auto).
  apply treduce_atoms in Hy. apply in_map_iff in Hy. destruct Hy as [[y' e''] [Hy' Hin]]. cbn [fst] in Hy'. subst y'.
  exact (Hf _ Hin).
Qed.

Lemma sorted_nodup : forall ts, StronglySorted tlt ts -> NoDup (map fst ts).
Proof.
  induction ts as [| [x c] ts IH]; intros Hs; cbn [map]; [constructor |].
  inversion Hs as [| a l Hs' Hf]; subst. constructor; [| exact (IH Hs')].
  intros Hin. apply in_map_iff in Hin. destruct Hin as [[y d] [Hy Hin]]. cbn [fst] in Hy. subst y.
  rewrite Forall_forall in Hf. specialize (Hf _ Hin). unfold tlt in Hf. cbn [fst] in Hf.
  rewrite atom_ltb_irrefl in Hf. discriminate.
Qed.

Lemma mk_ft : forall m ts lo hi, ft (mk m ts lo hi) = canon m ts.
Proof.
  intros. unfold mk. destruct (canon m ts); [reflexivity |]. destruct (Z.eqb lo hi); reflexivity.
Qed.

Lemma mk_nodup : forall m ts lo hi, NoDup (map fst (ft (mk m ts lo hi))).
Proof.
  intros. rewrite mk_ft. unfold canon. apply sorted_nodup, treduce_sorted, tsort_sorted.
Qed.

(* ------------------------------------------------------------------ *)
(* Forms, rounding, and comparisons.                                   *)


(* Rounding a range inward to multiples of 2^k keeps each multiple in it. *)
Lemma acc_round_in : forall k a b x y v,
    round k (a, b) = (x, y) -> a <= v <= b -> (2 ^ Z.max 0 k | v) -> x <= v <= y.
Proof.
  intros k a b x y v H Hv Hd. pose proof (round_range_sound k a b v Hv Hd) as Hr. rewrite H in Hr. exact Hr.
Qed.

Lemma eqm_solve : forall a R e, eqm 64 (a + -1 * R) (0 + e) -> eqm 64 R (a - e).
Proof.
  intros a R e H. unfold eqm in *.
  assert (HD : ((a + -1 * R) - (0 + e)) mod 2 ^ 64 = 0) by (rewrite Zminus_mod, H, Z.sub_diag; reflexivity).
  replace R with ((a - e) - ((a + -1 * R) - (0 + e))) at 1 by ring.
  rewrite Zminus_mod, HD, Z.sub_0_r, Z.mod_mod by lia. reflexivity.
Qed.

Lemma acc_some : forall (A : Type) (a b : A), Some a = Some b -> a = b.
Proof. intros A a b H. injection H as H. exact H. Qed.

Lemma falign_le : forall F st f, flo f = fhi f -> 0 <= fm f -> form_align F st f <= fm f.
Proof.
  intros F st f Hx Hm. unfold form_align, form_align_with.
  assert (Hle : forall l k,
             fold_left (fun k '(x, c) =>
                          let ka := match snd x with EX | EZ | ES => ra F DEPTH st (fst x) | _ => 0 end in
                          Z.min k (v2 c + ka)) l k <= k).
  { induction l as [| [x c] l IH]; intros k; cbn [fold_left]; [lia |]. eapply Z.le_trans; [apply IH | lia]. }
  eapply Z.le_trans; [apply Hle |]. unfold const_align. rewrite Hx, Z.eqb_refl.
  destruct (Z.eqb (fhi f) 0); lia.
Qed.

Lemma trunc64_eqm : forall a b, eqm 64 a b -> trunc 64 a = trunc 64 b.
Proof. intros a b H. unfold trunc. change (Z.of_nat 64) with 64. exact H. Qed.

Lemma sext64_inj : forall a b, 0 <= a < 2 ^ 64 -> 0 <= b < 2 ^ 64 -> sext 64 a = sext 64 b -> a = b.
Proof.
  intros a b Ha Hb H. unfold sext in H. rewrite !trunc_small in H by (change (Z.of_nat 64) with 64; lia).
  change (Z.of_nat 64 - 1) with 63 in H. change (Z.of_nat 64) with 64 in H.
  destruct (Z.ltb_spec a (2 ^ 63)), (Z.ltb_spec b (2 ^ 63)); lia.
Qed.

Lemma fold_first : forall (f : cmpf -> option range) l a r,
    fold_left (fun acc k => match acc with Some r => Some r | None => f k end) l a = Some r ->
    a = Some r \/ exists k, In k l /\ f k = Some r.
Proof.
  intros f l. induction l as [| k l IH]; intros a r H; cbn [fold_left] in H; [left; exact H |].
  destruct (IH _ _ H) as [H1 | [k' [Hin Hk]]].
  - destruct a as [a |]; [left; exact H1 | right; exists k; split; [left; reflexivity | exact H1]].
  - right. exists k'. split; [right; exact Hin | exact Hk].
Qed.

(* A form whose alignment is positive has a single constant. *)
Lemma falign_exact : forall F st f, 1 <= form_align F st f -> flo f = fhi f.
Proof.
  intros F st f H. unfold form_align, form_align_with in H.
  assert (Hle : forall l k,
             fold_left (fun k '(x, c) =>
                          let ka := match snd x with EX | EZ | ES => ra F DEPTH st (fst x) | _ => 0 end in
                          Z.min k (v2 c + ka)) l k <= k).
  { induction l as [| [x c] l IH]; intros k; cbn [fold_left]; [lia |]. eapply Z.le_trans; [apply IH | lia]. }
  pose proof (Hle (ft f) (const_align f)) as Hc. unfold const_align in H, Hc.
  destruct (Z.eqb_spec (flo f) (fhi f)); [assumption | lia].
Qed.

Lemma in_firstn_in : forall (A : Type) n (l : list A) x, In x (firstn n l) -> In x l.
Proof.
  intros A n l x H. rewrite <- (firstn_skipn n l). apply in_or_app. left. exact H.
Qed.

(* The bounded enumerator returns exactly the old prefix, in the same
   order. This optimization changes cost, not the retained constraints. *)
Lemma flat_map_budget_prefix : forall (A B : Type) (f : nat -> A -> list B) (g : A -> list B),
    (forall n x, f n x = firstn n (g x)) ->
    forall xs n, flat_map_budget n f xs = firstn n (flat_map g xs).
Proof.
  intros A B f g Hf xs. induction xs as [| x xs IH]; intros n.
  - destruct n; reflexivity.
  - destruct n as [| n]; [reflexivity |].
    cbn [flat_map_budget flat_map].
    rewrite Hf, IH, firstn_app. f_equal. rewrite firstn_length.
  f_equal. lia.
Qed.

Lemma flat_map_budget_pair_prefix : forall (A B C : Type) (f : A -> B -> list C) xs ys n,
    flat_map_budget n (fun m x => flat_map_budget m (fun k y => firstn k (f x y)) ys) xs =
    firstn n (flat_map (fun x => flat_map (f x) ys) xs).
Proof.
  intros A B C f xs ys n. apply flat_map_budget_prefix.
  intros m x. apply flat_map_budget_prefix. intros k y. reflexivity.
Qed.

(* ------------------------------------------------------------------ *)
(* The access checks.                                                  *)



Section Access.

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

Lemma fixed_rs : forall r, is_fixed F r = true -> rs r = e_init E r.
Proof. intros r H. exact (proj2 Hrs r H). Qed.

Lemma role_rs : forall P rl, role_of F P = Some rl -> rs P = e_init E P.
Proof. intros P rl H. apply fixed_rs. exact (ok_roles_fixed F E HE P rl H). Qed.

Lemma rs64 : forall r, 0 <= rs r < 2 ^ 64.
Proof.
  intros r. pose proof (rs_range F E rs Hrs r). pose proof (wd_range F E HE r).
  assert (2 ^ Z.of_nat (wdn r) <= 2 ^ 64) by (apply Z.pow_le_mono_r; lia). lia.
Qed.

(* The address form of an access at a + off. *)
Lemma addr_sound : forall st a off,
    state_ok F t st rs ->
    wf (fshift (normalize F st (as_form F st a EX)) off off) /\
    fm (fshift (normalize F st (as_form F st a EX)) off off) = 64 /\
    NoDup (map fst (ft (fshift (normalize F st (as_form F st a EX)) off off))) /\
    sat (fshift (normalize F st (as_form F st a EX)) off off) (rs a + off).
Proof.
  intros st a off Hst.
  destruct (as_form_sound F E HE t st rs Hst Hrs a EX) as [Hs Hm].
  pose proof (as_form_wf F E HE t st rs Hst Hrs a EX) as Hw.
  destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hw Hs) as [Hw2 [Hm2 Hs2]].
  cbn [atom_val] in Hs2.
  split; [apply wf_fshift; exact Hw2 |]. split; [unfold fshift; rewrite mk_fm; lia |].
  split; [unfold fshift; apply mk_nodup |].
  apply sat_fshift; [exact Hw2 | lia | exact Hs2].
Qed.

(* The same for a's value through its zero extension. *)
Lemma addr_sound_ext : forall st a off ex, (ex = EX \/ ex = EZ) ->
    state_ok F t st rs ->
    wf (fshift (normalize F st (as_form F st a ex)) off off) /\
    fm (fshift (normalize F st (as_form F st a ex)) off off) = 64 /\
    NoDup (map fst (ft (fshift (normalize F st (as_form F st a ex)) off off))) /\
    sat (fshift (normalize F st (as_form F st a ex)) off off) (rs a + off).
Proof.
  intros st a off ex Hex Hst.
  destruct (as_form_sound F E HE t st rs Hst Hrs a ex) as [Hs Hm].
  pose proof (as_form_wf F E HE t st rs Hst Hrs a ex) as Hw.
  destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hw Hs) as [Hw2 [Hm2 Hs2]].
  assert (Hv : aval (a, ex) = rs a) by (destruct Hex as [-> | ->]; reflexivity).
  rewrite Hv in Hs2.
  split; [apply wf_fshift; exact Hw2 |]. split; [unfold fshift; rewrite mk_fm; lia |].
  split; [unfold fshift; apply mk_nodup |].
  apply sat_fshift; [exact Hw2 | lia | exact Hs2].
Qed.

Lemma value_ext_exact : forall a, value_ext F a = EX \/ value_ext F a = EZ.
Proof. intros a. unfold value_ext. destruct (Z.eqb (wd F a) 64); [left | right]; reflexivity. Qed.

Lemma filter_notin : forall (ts : terms) (x : atom),
    ~ In x (map fst ts) -> filter (fun t0 => negb (atom_eqb (fst t0) x)) ts = ts.
Proof.
  induction ts as [| [y d] ts IH]; intros x Hn; [reflexivity |]. cbn [filter fst]. cbn [map In fst] in Hn.
  destruct (atom_eqb y x) eqn:E1.
  - apply atom_eqb_eq in E1. subst. exfalso. apply Hn. left. reflexivity.
  - cbn [negb]. f_equal. apply IH. tauto.
Qed.

Lemma tsum_without : forall ts x c,
    NoDup (map fst ts) -> In (x, c) ts -> tsum ts = tsum (without ts x) + c * aval x.
Proof.
  induction ts as [| [y d] ts IH]; intros x c Hn Hin; [destruct Hin |].
  inversion Hn as [| a l Hna Hn']; subst. cbn [map fst] in Hna. unfold without. cbn [filter fst tsum].
  destruct Hin as [Heq | Hin].
  - injection Heq as -> ->. rewrite (proj2 (atom_eqb_eq x x) eq_refl). cbn [negb].
    rewrite filter_notin by exact Hna. lia.
  - assert (Hyx : atom_eqb y x = false).
    { destruct (atom_eqb y x) eqn:E1; [| reflexivity]. apply atom_eqb_eq in E1. subst.
      exfalso. apply Hna. apply in_map_iff. exists (x, c). auto. }
    rewrite Hyx. cbn [negb tsum]. rewrite (IH x c Hn' Hin). unfold without. lia.
Qed.

(* A form with one term of atom x taken out holds of the value less that term. *)
Lemma sat_strip : forall f x c v,
    fm f = 64 -> NoDup (map fst (ft f)) -> In (x, c) (ft f) -> sat f v ->
    sat (mk 64 (without (ft f) x) (flo f) (fhi f)) (v - c * aval x).
Proof.
  intros f x c v Hm Hn Hin [e [He Hv]]. apply (sat_mk wdn rs 64 _ _ _ _ e); [lia | exact He |].
  rewrite Hm in Hv. rewrite (tsum_without _ x c Hn Hin) in Hv.
  pose proof (eqm_add 64 _ _ (- (c * aval x)) (- (c * aval x)) ltac:(lia) Hv (eqm_refl 64 _)) as H.
  replace (v - c * aval x) with (v + - (c * aval x)) by ring.
  replace (tsum (without (ft f) x) + e) with (tsum (without (ft f) x) + c * aval x + e + - (c * aval x)) by ring.
  exact H.
Qed.

(* A value of a form lies, modulo 2^64, in the form's range. *)
Lemma acc_sat_range : forall st f v L H,
    state_ok F t st rs -> fm f = 64 -> sat f v -> form_range F st f = Some (L, H) ->
    exists X, L <= X <= H /\ eqm 64 v X /\ (flo f = fhi f -> X = tsum (ft f) + flo f).
Proof.
  intros st f v L H Hst Hm [e [He Hv]] Hr. exists (tsum (ft f) + e).
  pose proof (form_range_sound F E HE t st rs Hst Hrs f L H e Hr He). rewrite Hm in Hv.
  split; [lia | split; [exact Hv | intros Hx; f_equal; lia]].
Qed.

Lemma near_acc_sat_range : forall st f v L H,
    state_ok F t st rs -> fm f = 64 -> sat f v -> form_range_near F st f = Some (L, H) ->
    exists X, L <= X <= H /\ eqm 64 v X /\ (flo f = fhi f -> X = tsum (ft f) + flo f).
Proof.
  intros st f v L H Hst Hm [e [He Hv]] Hr. exists (tsum (ft f) + e).
  pose proof (form_range_near_sound F E HE t st rs Hst Hrs f L H e Hr He). rewrite Hm in Hv.
  split; [lia | split; [exact Hv | intros Hx; f_equal; lia]].
Qed.

(* A value X of a form modulo 2^64 is a multiple of 2^al for each al at
   most the form's alignment. *)
Lemma form_aligned : forall st f R X al,
    state_ok F t st rs -> fm f = 64 -> sat f R -> eqm 64 R X -> al <= form_align F st f ->
    (2 ^ Z.max 0 al | X).
Proof.
  intros st f R X al Hst Hm Hs HX Hal.
  destruct (Z.le_gt_cases al 0) as [H0 | H0].
  - rewrite Z.max_l by lia. exists X. cbn. lia.
  - pose proof (falign_exact F st f ltac:(lia)) as Hex.
    pose proof (falign_le F st f Hex ltac:(lia)) as Hle.
    pose proof (form_align_sound F E HE t st rs Hst Hrs f Hex ltac:(lia)) as Ha.
    destruct Hs as [e [He Hv]]. assert (e = flo f) by lia. subst e. rewrite Hm in Hv.
    assert (Hal' : aligned al X).
    { apply (aligned_eqm 64 _ X (tsum (ft f) + flo f)); [lia | lia | |].
      - eapply eqm_trans; [apply eqm_sym; exact HX | exact Hv].
      - apply (aligned_mono (form_align F st f)); [lia | exact Ha]. }
    exact Hal'.
Qed.

(* The comparisons of a guard that holds hold. *)
Lemma guard_cmps_sound : forall st g,
    state_ok F t st rs -> holds g rs = true -> Forall (cmp_holds rs) (guard_cmps st g).
Proof.
  intros st [[neg p] |] Hst Hg; cbn [guard_cmps]; [| constructor].
  destruct neg; [constructor |].
  destruct (get st p) as [x |] eqn:Ep; [| constructor]. destruct x; try constructor.
  pose proof (proj1 Hst p _ Ep) as Hp. cbn [fact_ok fact_okv] in Hp. apply (proj1 Hp).
  cbn [holds xorb] in Hg. destruct (Z.eqb_spec (rs p) 0); [discriminate | assumption].
Qed.

Lemma norm_reg : forall st r,
    state_ok F t st rs ->
    wf (normalize F st (as_form F st r EX)) /\ fm (normalize F st (as_form F st r EX)) = 64 /\
    sat (normalize F st (as_form F st r EX)) (rs r).
Proof.
  intros st r Hst.
  destruct (as_form_sound F E HE t st rs Hst Hrs r EX) as [Hs Hm].
  pose proof (as_form_wf F E HE t st rs Hst Hrs r EX) as Hw.
  destruct (normalize_sound F E HE t st rs Hst Hrs _ _ Hw Hs) as [Hw2 [Hm2 Hs2]].
  split; [exact Hw2 | split; [lia | exact Hs2]].
Qed.

Lemma fm_fscale64 : forall f k, fm f = 64 -> fm (fscale f k 64) = 64.
Proof.
  intros f k H. unfold fscale. rewrite mk_fm. destruct (Z.eqb k 0); [reflexivity |].
  pose proof (v2_nonneg k). lia.
Qed.

(* ------------------------------------------------------------------ *)
(* Bounds from the guard.                                              *)

(* The alignment rest_align gives holds of each value of rest modulo 2^64,
   when rest is the address a + off less base P. *)
Lemma rest_align_sound : forall st a off P rest X,
    state_ok F t st rs -> fm rest = 64 -> sat rest (rs a + off - rs P) ->
    eqm 64 (rs a + off - rs P) X -> aligned (GateDomain.rest_align F st a off P rest) X.
Proof.
  intros st a off P rest X Hst Hm Hs HX. unfold GateDomain.rest_align.
  set (m := Z.min (Z.min (reg_align F st a) (v2 off)) (Z.min (reg_align F st P) 64)).
  destruct (Z.max_spec (form_align F st rest) m) as [[_ ->] | [_ ->]].
  - apply (aligned_eqm 64 m X (rs a + off - rs P)); [lia | unfold m; lia | apply eqm_sym; exact HX |].
    pose proof (reg_align_sound F E HE t st rs Hst Hrs a) as Ha.
    pose proof (reg_align_sound F E HE t st rs Hst Hrs P) as HP.
    replace (rs a + off - rs P) with (rs a + off + - rs P) by ring.
    apply aligned_add; [apply aligned_add |].
    + apply (aligned_mono (reg_align F st a)); [unfold m; lia | exact Ha].
    + apply (aligned_mono (v2 off)); [unfold m; lia | apply aligned_v2].
    + apply aligned_opp. apply (aligned_mono (reg_align F st P)); [unfold m; lia | exact HP].
  - exact (form_aligned st rest _ X _ Hst Hm Hs HX (Z.le_refl _)).
Qed.

Lemma guard_bound_sound : forall st g rest R al x y,
    state_ok F t st rs -> holds g rs = true -> fm rest = 64 -> sat rest R ->
    (forall X, eqm 64 R X -> aligned al X) ->
    guard_bound F st g rest al = Some (x, y) -> exists X, x <= X <= y /\ eqm 64 R X.
Proof.
  intros st g rest R al x y Hst Hg Hm Hs Hal H. unfold guard_bound in H. cbv zeta in H.
  apply fold_first in H. destruct H as [H | [k [Hk H]]]; [discriminate |].
  pose proof (proj1 (Forall_forall _ _) (guard_cmps_sound st g Hst Hg) k Hk) as Hck.
  unfold cmp_holds in Hck. cbv beta in H.
  destruct (Nat.eqb_spec (cw k) 64) as [Ew | Ew]; [| discriminate]. cbn [negb] in H. rewrite Ew in Hck.
  destruct (norm_reg st (ca k) Hst) as [Hwx [Hmx Hsx]].
  assert (Hwr : wf rest) by (unfold wf; lia).
  pose proof (sat_fscale wdn rs rest (-1) 64 R Hwr ltac:(lia) Hs) as HsR.
  pose proof (wf_fscale rest (-1) 64 Hwr ltac:(lia)) as HwR.
  pose proof (fm_fscale64 rest (-1) Hm) as HmR.
  destruct (cc k) eqn:Ecc; destruct (cb k) as [yr | z] eqn:Ecb; try discriminate; cbn [cmp_eval opv] in Hck.
  - (* CLt with a constant *)
    destruct (csg k) eqn:Es; [discriminate |].
    match type of H with (if ?c then _ else _) = _ => destruct c eqn:Ec; [| discriminate] end.
    rewrite !andb_true_iff in Ec. destruct Ec as [[[Hn HmD] Hc0] Hc1].
    apply Z.eqb_eq in HmD. apply Z.leb_le in Hc0. apply Z.ltb_lt in Hc1.
    pose proof (sat_fadd wdn rs _ _ _ _ Hwx HwR Hsx HsR) as HsD.
    set (D := fadd (normalize F st (as_form F st (ca k) EX)) (fscale rest (-1) 64)) in *.
    destruct (ft D) eqn:EftD; [| discriminate].
    destruct HsD as [e [He Hv]]. rewrite HmD, EftD in Hv. cbn [tsum] in Hv.
    apply acc_some in H.
    cbn [extend] in Hck. apply Z.ltb_lt in Hck. rewrite trunc_small in Hck by (pose proof (rs64 (ca k)); change (Z.of_nat 64) with 64; lia).
    assert (Hz : trunc 64 z = z \/ trunc 64 z = 0).
    { destruct (Z.eq_dec z (2 ^ 64)) as [-> | Hne]; [right; reflexivity |].
      left. apply trunc_small. change (Z.of_nat 64) with 64. lia. }
    exists (rs (ca k) - e). split.
    + apply (acc_round_in _ _ _ _ _ _ H); [| apply Hal].
      * pose proof (rs64 (ca k)). destruct Hz as [Hz | Hz]; rewrite Hz in Hck; lia.
      * apply eqm_solve. exact Hv.
    + apply eqm_solve. exact Hv.
  - (* CLe with a constant *)
    destruct (csg k) eqn:Es; [discriminate |].
    match type of H with (if ?c then _ else _) = _ => destruct c eqn:Ec; [| discriminate] end.
    rewrite !andb_true_iff in Ec. destruct Ec as [[[Hn HmD] Hc0] Hc1].
    apply Z.eqb_eq in HmD. apply Z.leb_le in Hc0. apply Z.ltb_lt in Hc1.
    pose proof (sat_fadd wdn rs _ _ _ _ Hwx HwR Hsx HsR) as HsD.
    set (D := fadd (normalize F st (as_form F st (ca k) EX)) (fscale rest (-1) 64)) in *.
    destruct (ft D) eqn:EftD; [| discriminate].
    destruct HsD as [e [He Hv]]. rewrite HmD, EftD in Hv. cbn [tsum] in Hv.
    apply acc_some in H.
    cbn [extend] in Hck. apply Z.leb_le in Hck.
    rewrite trunc_small in Hck by (pose proof (rs64 (ca k)); change (Z.of_nat 64) with 64; lia).
    rewrite (trunc_small 64 z) in Hck by (change (Z.of_nat 64) with 64; lia).
    exists (rs (ca k) - e). split.
    + apply (acc_round_in _ _ _ _ _ _ H); [pose proof (rs64 (ca k)); lia |].
      apply Hal. apply eqm_solve. exact Hv.
    + apply eqm_solve. exact Hv.
  - (* CEq of two registers *)
    match type of H with (if ?c then _ else _) = _ => destruct c eqn:Ec; [| discriminate] end.
    rewrite !andb_true_iff in Ec. destruct Ec as [Hn HmD]. apply Z.eqb_eq in HmD.
    destruct (norm_reg st yr Hst) as [Hwy [Hmy Hsy]].
    pose proof (sat_fscale wdn rs _ (-1) 64 _ Hwy ltac:(lia) Hsy) as Hsy'.
    pose proof (wf_fscale _ (-1) 64 Hwy ltac:(lia)) as Hwy'.
    pose proof (sat_fadd wdn rs _ _ _ _ Hwx Hwy' Hsx Hsy') as Hs1.
    pose proof (wf_fadd _ _ Hwx Hwy') as Hw1.
    pose proof (sat_fadd wdn rs _ _ _ _ Hw1 HwR Hs1 HsR) as HsD.
    set (D := fadd (fadd (normalize F st (as_form F st (ca k) EX))
                         (fscale (normalize F st (as_form F st yr EX)) (-1) 64)) (fscale rest (-1) 64)) in *.
    destruct (ft D) eqn:EftD; [| discriminate].
    destruct HsD as [e [He Hv]]. rewrite HmD, EftD in Hv. cbn [tsum] in Hv.
    apply acc_some in H.
    assert (Hxy : rs (ca k) = rs yr).
    { apply Z.eqb_eq in Hck. pose proof (rs64 (ca k)). pose proof (rs64 yr).
      destruct (csg k); cbn [extend] in Hck.
      - apply sext64_inj; [lia | lia | exact Hck].
      - rewrite !trunc_small in Hck by (change (Z.of_nat 64) with 64; lia). exact Hck. }
    assert (HR : eqm 64 R (- e)).
    { rewrite Hxy in Hv. replace (rs yr + -1 * rs yr + -1 * R) with (0 + -1 * R) in Hv by ring.
      pose proof (eqm_solve 0 R e Hv) as H'. replace (- e) with (0 - e) by ring. exact H'. }
    exists (- e). split; [| exact HR].
    apply (acc_round_in _ _ _ _ _ _ H); [lia | apply (Hal _ HR)].
Qed.

(* ------------------------------------------------------------------ *)
(* One base parameter.                                                 *)

Lemma base_rest : forall A P v,
    fm A = 64 -> NoDup (map fst (ft A)) -> sat A v -> base_params F A = [P] ->
    (exists rl, role_of F P = Some rl) /\ sat (mk 64 (without (ft A) (P, EX)) (flo A) (fhi A)) (v - rs P).
Proof.
  intros A P v Hm Hn Hs Hb.
  assert (Hin : In P (base_params F A)) by (rewrite Hb; left; reflexivity).
  unfold base_params in Hin. apply in_flat_map in Hin. destruct Hin as [[[r e] c] [Hin Hp]].
  destruct e; cbv beta iota in Hp; try (destruct Hp).
  destruct ((c =? 1) && is_some (role_of F r)) eqn:Ec; [| destruct Hp].
  destruct Hp as [-> | []].
  apply andb_true_iff in Ec. destruct Ec as [Ec Er]. apply Z.eqb_eq in Ec. subst c.
  split; [destruct (role_of F P) as [rl |]; [exists rl; reflexivity | discriminate] |].
  pose proof (sat_strip A (P, EX) 1 v Hm Hn Hin Hs) as H. cbn [atom_val] in H.
  replace (v - rs P) with (v - 1 * rs P) by ring. exact H.
Qed.

Lemma eqm_base : forall v p X, eqm 64 (v - p) X -> eqm 64 v (p + X).
Proof.
  intros v p X H. replace v with (p + (v - p)) by ring. apply eqm_add; [lia | apply eqm_refl | exact H].
Qed.

(* An access within the extent of a span. *)
(* A bound form has the same value in every state, its value at launch. *)
Lemma bound_value : forall f, bound_form F f = true -> tsum (ft f) + flo f = bval F E f.
Proof.
  intros f H. unfold bound_form in H. rewrite !andb_true_iff in H. destruct H as [[_ _] Hts].
  unfold bval. f_equal. rewrite forallb_forall in Hts.
  induction (ft f) as [| [[r e] c] ts IH]; cbn [tsum]; [reflexivity |].
  pose proof (Hts _ (or_introl eq_refl)) as Hr. cbv beta iota in Hr. apply andb_true_iff in Hr.
  destruct Hr as [Hfix He]. destruct e; try discriminate. cbn [atom_val].
  rewrite (fixed_rs r Hfix), IH; [reflexivity |]. intros u Hu. exact (Hts u (or_intror Hu)).
Qed.

Lemma bound_sat : forall f, bound_form F f = true -> wf f /\ fm f = 64 /\ flo f = fhi f /\ sat f (bval F E f).
Proof.
  intros f H. pose proof (bound_value f H) as Hv. unfold bound_form in H. rewrite !andb_true_iff in H.
  destruct H as [[Hm Hlh] _]. apply Z.eqb_eq in Hm, Hlh.
  split; [unfold wf; lia |]. split; [exact Hm |]. split; [exact Hlh |].
  exists (flo f). split; [lia |]. rewrite <- Hv. apply eqm_refl.
Qed.

(* A representative X of rest inside the span: at or past the base, and
   ending at or before the extent. *)
(* ------------------------------------------------------------------ *)
(* The constraints of an access check.                                 *)

Lemma canon_atom_with_val : forall ranges x, rfuns_ok F rs ranges -> aval (canon_atom_with F ranges x) = aval x.
Proof.
  intros ranges [r e] Hst. unfold canon_atom_with.
  destruct e; cbn [atom_val]; try reflexivity.
  destruct (f_rr ranges r) as [[lo hi] |] eqn:Er; [| reflexivity].
  destruct ((0 <=? lo) && (hi <? 2 ^ (wd F r - 1))) eqn:Ec; [| reflexivity].
  apply andb_true_iff in Ec. destruct Ec as [E1 E2]. apply Z.leb_le in E1. apply Z.ltb_lt in E2.
  pose proof (proj1 Hst r lo hi Er) as Hr.
  pose proof (wd_range F E HE r) as Hw. unfold wd in E2. cbn [atom_val].
  symmetry. apply sext_small; lia.
Qed.

Lemma canon_terms_with_tsum : forall ranges ts, rfuns_ok F rs ranges -> tsum (canon_terms_with F ranges ts) = tsum ts.
Proof.
  intros ranges ts Hst. unfold canon_terms_with. rewrite (tsum_sorted_terms F E HE t rs Hrs).
  induction ts as [| [x c] ts IH]; cbn [map tsum fst snd]; [reflexivity |].
  rewrite (canon_atom_with_val ranges x Hst), IH. reflexivity.
Qed.

Lemma canon_cons_with_ok : forall ranges k, rfuns_ok F rs ranges -> cons_ok F rs k -> cons_ok F rs (canon_cons_with F ranges k).
Proof.
  intros ranges [[ts lo] hi] Hst Hk. cbn [cons_ok canon_cons_with] in *. rewrite (canon_terms_with_tsum ranges ts Hst). exact Hk.
Qed.

Lemma canon_atom_val : forall st x, state_ok F t st rs -> aval (canon_atom F st x) = aval x.
Proof. intros st x Hst. apply canon_atom_with_val. exact (rft_on_sound F E HE t st rs Hst Hrs _). Qed.
Lemma canon_terms_tsum : forall st ts, state_ok F t st rs -> tsum (canon_terms F st ts) = tsum ts.
Proof. intros st ts Hst. apply canon_terms_with_tsum. exact (rft_on_sound F E HE t st rs Hst Hrs _). Qed.
Lemma canon_cons_ok : forall st k, state_ok F t st rs -> cons_ok F rs k -> cons_ok F rs (canon_cons F st k).
Proof.
 intros st [[ts lo] hi] Hst Hk. apply canon_cons_with_ok; [exact (rft_on_sound F E HE t st rs Hst Hrs _) | exact Hk].
Qed.

Lemma cons_lin_ok : forall a b k1 k2,
    0 <= a -> cons_ok F rs k1 -> cons_ok F rs k2 -> cons_ok F rs (cons_lin a b k1 k2).
Proof.
  intros a b [[t1 l1] h1] [[t2 l2] h2] Ha H1 H2. cbn [cons_ok cons_lin] in *.
  rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app, !tsum_tscale.
  destruct (Z.leb_spec 0 b); nia.
Qed.

Lemma eliminate_ok : forall x k1 k2, cons_ok F rs k1 -> cons_ok F rs k2 -> cons_ok F rs (eliminate x k1 k2).
Proof.
  intros x [[t1 l1] h1] [[t2 l2] h2] H1 H2. unfold eliminate. apply cons_lin_ok; [lia | exact H1 | exact H2].
Qed.

Lemma fm_step_ok : forall x cs, Forall (cons_ok F rs) cs -> Forall (cons_ok F rs) (fm_step x cs).
Proof.
  intros x cs H. unfold fm_step. cbv zeta. apply Forall_app. split.
  - apply Forall_forall. intros k Hk. apply filter_In in Hk. exact (proj1 (Forall_forall _ _) H k (proj1 Hk)).
  - destruct (find _ _) as [ke |] eqn:Ef.
    + (* each constraint with x, combined with the equality ke *)
      apply find_some in Ef. destruct Ef as [Hke _]. apply filter_In in Hke.
      pose proof (proj1 (Forall_forall _ _) H ke (proj1 Hke)) as He.
      destruct ke as [[te le] he]. apply Forall_forall. intros k Hk.
      apply in_flat_map in Hk. destruct Hk as [k1 [Hk1 Hk]]. apply filter_In in Hk1.
      pose proof (proj1 (Forall_forall _ _) H k1 (proj1 Hk1)) as H1.
      destruct k1 as [[t1 l1] h1].
      destruct (terms_eqb t1 te); [destruct Hk | destruct Hk as [<- | []]].
      exact (eliminate_ok x _ _ H1 He).
    + rewrite flat_map_budget_pair_prefix.
      apply Forall_forall. intros k Hk. apply in_firstn_in in Hk.
      apply in_flat_map in Hk. destruct Hk as [k1 [Hk1 Hk]]. apply in_flat_map in Hk. destruct Hk as [k2 [Hk2 Hk]].
      apply filter_In in Hk1, Hk2.
      pose proof (proj1 (Forall_forall _ _) H k1 (proj1 Hk1)) as H1.
      pose proof (proj1 (Forall_forall _ _) H k2 (proj1 Hk2)) as H2.
      destruct k1 as [[t1 l1] h1]. destruct k2 as [[t2 l2] h2].
      destruct (terms_eqb t1 t2); [destruct Hk | destruct Hk as [<- | []]].
      exact (eliminate_ok x _ _ H1 H2).
Qed.

Lemma orient_cons_ok : forall k, cons_ok F rs k -> cons_ok F rs (orient_cons k).
Proof.
 intros [[ts lo] hi] H. unfold orient_cons.
 destruct ts as [|[x c] ts]; [exact H |].
 destruct (c <? 0); [| exact H]. cbn [cons_ok] in *.
 rewrite tsum_tscale. lia.
Qed.

(* The greatest common divisor of the coefficients divides each of them. *)
Lemma gcd_fold_acc : forall (ts : terms) a, (fold_left (fun a u => Z.gcd a (snd u)) ts a | a).
Proof.
  induction ts as [| u ts IH]; intros a; cbn [fold_left]; [apply Z.divide_refl |].
  apply Z.divide_trans with (Z.gcd a (snd u)); [apply IH | apply Z.gcd_divide_l].
Qed.

Lemma gcd_fold_in : forall (ts : terms) a u, In u ts -> (fold_left (fun a u => Z.gcd a (snd u)) ts a | snd u).
Proof.
  induction ts as [| v ts IH]; intros a u Hu; [destruct Hu |].
  cbn [fold_left]. destruct Hu as [-> | Hu].
  - apply Z.divide_trans with (Z.gcd a (snd u)); [apply gcd_fold_acc | apply Z.gcd_divide_r].
  - apply IH. exact Hu.
Qed.

Lemma tsum_div : forall g (ts : terms), g <> 0 -> (forall u, In u ts -> (g | snd u)) ->
    g * tsum (map (fun u => (fst u, snd u / g)) ts) = tsum ts.
Proof.
  intros g ts Hg Hd. induction ts as [| [x c] ts IH]; cbn [map tsum fst snd]; [lia |].
  rewrite Z.mul_add_distr_l, IH by (intros u Hu; apply Hd; right; exact Hu).
  destruct (Hd (x, c) (or_introl eq_refl)) as [q Hq]. cbn [snd] in Hq. subst c.
  rewrite Z.div_mul by exact Hg. ring.
Qed.

Lemma reduce_cons_ok : forall k, cons_ok F rs k -> cons_ok F rs (reduce_cons k).
Proof.
  intros [[ts lo] hi] H. unfold reduce_cons. cbv beta iota zeta.
  set (g := fold_left (fun a u => Z.gcd a (snd u)) ts 0).
  destruct (Z.leb_spec g 1) as [| Hg]; [exact H |]. cbn [cons_ok] in *.
  rewrite <- (tsum_div g ts ltac:(lia) (fun u Hu => gcd_fold_in ts 0 u Hu)) in H.
  set (v := tsum (map (fun u => (fst u, snd u / g)) ts)) in *.
  split.
  - apply Z.opp_le_mono. rewrite Z.opp_involutive. apply Z.div_le_lower_bound; [lia | rewrite Z.mul_opp_r; lia].
  - apply Z.div_le_lower_bound; lia.
Qed.

Lemma compact_projection_bounds : forall ts cs lo hi,
 lo <= tsum ts <= hi ->
 Forall (cons_ok F rs) cs ->
 Forall (fun k : constraint => terms_eqb ts (fst (fst k)) = true) cs ->
 let '(lo',hi') := fold_left (fun acc k => let '(l,h) := acc in let '(_,l',h') := k in
    (Z.max l l',Z.min h h')) cs (lo,hi) in lo' <= tsum ts <= hi'.
Proof.
 intros ts cs. induction cs as [| [[terms low] high] cs IH]; intros lo hi H Hcs Hts.
 - exact H.
 - inversion Hcs as [| k ks Hk Hrest]; subst. inversion Hts as [| k ks Ht Hts']; subst.
   cbn [fst] in Ht. apply terms_eqb_eq in Ht. subst terms.
   cbn [fold_left]. apply IH; [cbn [cons_ok] in Hk; lia | exact Hrest | exact Hts'].
Qed.

Lemma compact_projection_ok : forall cs,
 Forall (cons_ok F rs) cs -> Forall (cons_ok F rs) (compact_projection cs).
Proof.
 intros cs H. induction H as [| k cs Hk Hcs IH].
 - constructor.
 - cbn [compact_projection fold_right].
   pose proof (reduce_cons_ok _ (orient_cons_ok k Hk)) as Ho.
   destruct (reduce_cons (orient_cons k)) as [[ts lo] hi] eqn:Eo. cbn [cons_ok] in Ho.
   set (rest := compact_projection cs) in *.
   set (same := fun k' : constraint => let '(ts',_,_) := k' in terms_eqb ts ts').
   assert (Hfiltered : Forall (cons_ok F rs) (filter same rest)).
   { apply Forall_forall. intros k' Hk'. apply filter_In in Hk'.
     exact (proj1 (Forall_forall _ _) IH _ (proj1 Hk')). }
   assert (Hterms : Forall (fun k' : constraint => terms_eqb ts (fst (fst k')) = true) (filter same rest)).
   { apply Forall_forall. intros [[ts' lo'] hi'] Hk'. apply filter_In in Hk'. exact (proj2 Hk'). }
   pose proof (compact_projection_bounds ts (filter same rest) lo hi Ho Hfiltered Hterms) as Hb.
   destruct (fold_left (fun acc (k' : constraint) => let '(l,h) := acc in let '(_,l',h') := k' in
      (Z.max l l',Z.min h h')) (filter same rest) (lo,hi)) as [lo' hi'] eqn:Eb.
   change (Forall (cons_ok F rs)
     (let '(l,h) := fold_left (fun acc (k' : constraint) => let '(l,h) := acc in let '(_,l',h') := k' in
        (Z.max l l',Z.min h h')) (filter same rest) (lo,hi) in
      (ts,l,h) :: filter (fun k' => negb (same k')) rest)).
   change (let '(l,h) := fold_left (fun acc (k' : constraint) => let '(l,h) := acc in let '(_,l',h') := k' in
      (Z.max l l',Z.min h h')) (filter same rest) (lo,hi) in l <= tsum ts <= h) in Hb.
   rewrite Eb in Hb. rewrite Eb. constructor; [exact Hb |].
   apply Forall_forall. intros k' Hk'. apply filter_In in Hk'.
   exact (proj1 (Forall_forall _ _) IH _ (proj1 Hk')).
Qed.

Lemma project_range_sound : forall st extra f a b e,
    state_ok F t st rs -> Forall (cons_ok F rs) extra ->
    project_range F st extra f = Some (a, b) -> flo f <= e <= fhi f ->
    a <= tsum (ft f) + e <= b.
Proof.
  intros st extra f a b e Hst Hextra H He. unfold project_range, project_range_with, project_range_plain in H.
  set (ranges := cache_rf (rft_on F (cons_index (acons st)) st (term_regs (projection_terms st extra f)))
                   (projection_regs F st extra f)) in *.
  assert (Hranges : rfuns_ok F rs ranges).
  { destruct (rft_on_sound F E HE t st rs Hst Hrs (term_regs (projection_terms st extra f))) as [Hr Hs].
    split; intros r a' b' Hx; unfold ranges in Hx.
    - rewrite (proj1 (cache_rf_same _ _ r)) in Hx. exact (Hr r a' b' Hx).
    - rewrite (proj2 (cache_rf_same _ _ r)) in Hx. exact (Hs r a' b' Hx). }
  set (ar0 := atom_range_with F ranges) in *.
  set (ar := fun x => inter (ar0 x) (natural_range F x)) in *.
  assert (Har : forall x a b, ar x = Some (a, b) -> a <= aval x <= b).
  { intros x a' b' Hx. apply (inter_ok (ar0 x) (natural_range F x) (aval x)); [| | exact Hx].
    - intros l h Hr. exact (atom_range_with_sound F E HE rs Hrs _ x l h Hranges Hr).
    - intros l h Hr. exact (natural_range_sound F E HE rs Hrs x l h Hr). }
  set (g := {| fm := fm f; ft := canon_terms_with F ranges (ft f); flo := flo f; fhi := fhi f |}) in *.
  set (cs := map (canon_cons_with F ranges) (near_constraints F st (acons st ++ extra) (ft f))) in *.
  assert (Hcs : Forall (cons_ok F rs) cs).
  { unfold cs. apply Forall_forall. intros k Hk. apply in_map_iff in Hk. destruct Hk as [k' [<- Hin]].
    unfold near_constraints in Hin. apply filter_In in Hin. apply proj1 in Hin. apply in_app_or in Hin. apply (canon_cons_with_ok ranges k' Hranges).
    destruct Hin as [Hin | Hin]; [exact (proj1 (Forall_forall _ _) (proj2 Hst) _ Hin) | exact (proj1 (Forall_forall _ _) Hextra _ Hin)]. }
  set (xs := atoms_nodup (map fst (ft g) ++ flat_map (fun k => let '(ts, _, _) := k in map fst ts) cs) []) in *.
  set (units := flat_map (fun x => match ar x with Some (lo, hi) => [([(x, 1)], lo, hi)] | None => [] end) xs) in *.
  assert (Hun : Forall (cons_ok F rs) units).
  { unfold units. apply Forall_forall. intros k Hk. apply in_flat_map in Hk. destruct Hk as [x [_ Hk]].
    destruct (ar x) as [[lo hi] |] eqn:Ex; [| destruct Hk]. destruct Hk as [<- | []].
    cbn [cons_ok tsum]. pose proof (Har x lo hi Ex). lia. }
  (* a register's value is its exact form's integer value *)
  assert (Heq : Forall (cons_ok F rs) (map (canon_cons_with F ranges) (flat_map (form_eq_with F ranges st) xs))).
  { apply Forall_forall. intros k' Hk'. apply in_map_iff in Hk'. destruct Hk' as [k [<- Hk]].
    apply (canon_cons_with_ok ranges k Hranges). apply in_flat_map in Hk. destruct Hk as [[r ex] [_ Hk]].
    unfold form_eq_with in Hk.
    assert (Hx : forall h al vr, get st r = Some (FForm h al vr) -> atom_val wdn rs (r, ex) = rs r ->
              In k (if Z.eqb (fm h) 64 && Z.eqb (wd F r) 64 && Z.eqb (flo h) (fhi h) && negb (null (ft h)) then
                      match inter (f_rr ranges r) (natural_range F (r,EX)), form_range_with F st (atom_range_with F ranges) h with
                      | Some (lo, hi), Some (p, q) =>
                          if (- 2 ^ 64 <? lo - q) && (hi - p <? 2 ^ 64)
                          then [(sorted_terms ([((r, ex), 1)] ++ tscale (-1) (ft h)), flo h, flo h)] else []
                      | _, _ => []
                      end
                    else []) -> cons_ok F rs k).
    { intros h al vr Eg Hav Hin.
      destruct (Z.eqb (fm h) 64 && Z.eqb (wd F r) 64 && Z.eqb (flo h) (fhi h) && negb (null (ft h))) eqn:Ec;
        [| destruct Hin].
      rewrite !andb_true_iff in Ec. destruct Ec as [[[Hm Hw] Hl] _].
      apply Z.eqb_eq in Hm, Hw, Hl.
      destruct (inter (f_rr ranges r) (natural_range F (r,EX))) as [[lo hi] |] eqn:Erg; [| destruct Hin].
      destruct (form_range_with F st (atom_range_with F ranges) h) as [[p q] |] eqn:Er; [| destruct Hin].
      destruct ((- 2 ^ 64 <? lo - q) && (hi - p <? 2 ^ 64)) eqn:Epq; [| destruct Hin].
      destruct Hin as [<- | []]. apply andb_true_iff in Epq. destruct Epq as [Hp Hq].
      apply Z.ltb_lt in Hp, Hq.
      pose proof (proj1 Hst r _ Eg) as Hh. unfold fact_ok in Hh. cbn [fact_okv] in Hh.
      destruct Hh as [Hwf [[e1 [He1 Hv1]] _]].
      assert (Hr1 : p <= tsum (ft h) + e1 <= q).
      { apply (form_range_with_sound F E t st rs Hst Hrs (atom_range_with F ranges) h p q e1); [| exact Er | exact He1].
        intros x l u Hx. exact (atom_range_with_sound F E HE rs Hrs ranges x l u Hranges Hx). }
      assert (Hrr : lo <= rs r <= hi).
      { apply (inter_ok (f_rr ranges r) (natural_range F (r,EX)) (rs r)); [| | exact Erg].
        - intros l u Hr. exact (proj1 Hranges r l u Hr).
        - intros l u Hr. exact (natural_range_sound F E HE rs Hrs (r,EX) l u Hr). }
      rewrite Hm in Hv1.
      assert (Hval : rs r = tsum (ft h) + e1) by (apply (eqm_close 64); [lia | exact Hv1 | lia]).
      cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app, tsum_tscale. cbn [tsum].
      rewrite Hav. lia. }
    destruct ex; try destruct Hk.
    - destruct (get st r) as [[h al vr | | |] |] eqn:Eg; try destruct Hk. exact (Hx h al vr eq_refl eq_refl Hk).
    - destruct (get st r) as [[h al vr | | |] |] eqn:Eg; try destruct Hk. exact (Hx h al vr eq_refl eq_refl Hk). }
  (* a register is 2^k times its value shifted right by k plus its low k bits *)
  assert (Hlinks : Forall (cons_ok F rs) (shift_links xs)).
  { apply Forall_forall. intros c Hc. unfold shift_links in Hc. apply in_flat_map in Hc.
    destruct Hc as [[r ex] [_ Hc]]. destruct ex as [ | | | k | offset n]; try destruct Hc.
    - destruct ((0 <=? k) && existsb (atom_eqb (r, EX)) xs) eqn:Hk0; [| destruct Hc].
      apply andb_true_iff in Hk0. destruct Hk0 as [Hk _]. apply Z.leb_le in Hk. destruct Hc as [<- | []].
      cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE t rs Hrs). cbn [tsum atom_val].
      assert (Hp : 0 < 2 ^ k) by (apply Z.pow_pos_nonneg; lia).
      pose proof (Z.div_mod (rs r) (2 ^ k) ltac:(lia)). pose proof (Z.mod_pos_bound (rs r) (2 ^ k) Hp). nia.
    - destruct (Z.eqb offset 0 && (0 <=? n) &&
         (existsb (atom_eqb (r,EX)) xs || existsb (atom_eqb (r,EZ)) xs)) eqn:Ebit; [|destruct Hc].
      rewrite !andb_true_iff in Ebit. destruct Ebit as [[Ea Hn] _]. apply Z.eqb_eq in Ea. subst offset.
      apply Z.leb_le in Hn. destruct Hc as [<-|[]].
      cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE t rs Hrs).
      destruct (existsb (atom_eqb (r,EX)) xs); cbn [tsum atom_val]; rewrite Z.div_1_r.
      all: pose proof (rs_range F E rs Hrs r) as Hr;
           pose proof (wd_range F E HE r) as Hw;
           change (1<=wd F r<=64) in Hw;
           assert (Hp:0<2^n) by (apply Z.pow_pos_nonneg;lia);
           pose proof (Z.mod_pos_bound (rs r) (2^n) Hp) as Hmod;
           assert (Hle:rs r mod 2^n<=rs r) by (apply Z.mod_le;lia);
           assert (H64:2^wd F r<=2^64) by (apply Z.pow_le_mono_r;lia);
           change (0<=rs r<2^wd F r) in Hr;lia. }
  assert (Hfold : forall l acc, Forall (cons_ok F rs) acc ->
            Forall (cons_ok F rs) (fold_left (fun acc x => compact_projection (fm_step x acc)) l acc)).
  { intros l. induction l as [| x l IH]; intros acc Hacc; cbn [fold_left]; [exact Hacc |].
    apply IH. apply compact_projection_ok. apply fm_step_ok. exact Hacc. }
  set (kept := projection_slice F (ft g)
    (cs ++ units ++ map (canon_cons_with F ranges) (flat_map (form_eq_with F ranges st) xs) ++ shift_links xs)) in *.
  assert (Hkept : Forall (cons_ok F rs) kept).
  { unfold kept, projection_slice. apply Forall_forall. intros k Hk.
    apply filter_In in Hk. destruct Hk as [Hk _].
    apply in_app_or in Hk as [Hk | Hk].
    - exact (proj1 (Forall_forall _ _) Hcs _ Hk).
    - apply in_app_or in Hk as [Hk | Hk].
      + exact (proj1 (Forall_forall _ _) Hun _ Hk).
      + apply in_app_or in Hk as [Hk | Hk].
        * exact (proj1 (Forall_forall _ _) Heq _ Hk).
        * exact (proj1 (Forall_forall _ _) Hlinks _ Hk). }
  pose proof (Hfold
    (filter (fun x => if tmem (ft g) x then false
                     else short_any (atom_eqb x) (projection_members kept))
      (atoms_nodup (flat_map (fun k => let '(ts,_,_) := k in map fst ts)
        (map (canon_cons_with F ranges) (flat_map (form_eq_with F ranges st) xs))) xs))
    (compact_projection kept) (compact_projection_ok kept Hkept)) as Hcs'.
  assert (Hg : tsum (ft g) = tsum (ft f)) by (cbn [ft g]; exact (canon_terms_with_tsum ranges (ft f) Hranges)).
  pose proof (projected_cons_fold_sound F E rs Hrs ar g _ (plain_range F ar g) e Har Hcs' He) as Hf.
  rewrite Hg in Hf. apply (Hf ltac:(intros a' b' Hp; rewrite <- Hg;
                                    apply (plain_range_sound F E rs Hrs ar g);
                                    [intros x a0 b0 Hx; exact (Har x a0 b0 Hx) | exact Hp | exact He])).
  exact H.
Qed.

(* A residue form's register equals the form's integer value when that
   value, projected, lies within 2^m of every point of the register's
   window: the two agree modulo 2^m. A signed reading that is at least 0
   is the register's value. *)
Lemma residue_eq_sound : forall st x k,
    state_ok F t st rs -> In k (residue_eq F (project_range F st []) st x) -> cons_ok F rs k.
Proof.
  intros st [r ex] k Hst Hk. unfold residue_eq in Hk.
  destruct ex; try destruct Hk.
  destruct (get st r) as [[h al vr | | |] |] eqn:Eg; try destruct Hk.
  destruct (Z.eqb (wd F r) 64 && (0 <? fm h) && (fm h <? 64) && Z.eqb (flo h) (fhi h) && negb (null (ft h))) eqn:Ec;
    [| destruct Hk].
  rewrite !andb_true_iff in Ec. destruct Ec as [[[[Hw Hm0] Hm1] Hl] _].
  apply Z.eqb_eq in Hw, Hl. apply Z.ltb_lt in Hm0, Hm1.
  destruct (residue_window st r (FForm h al vr)) as [[[lo hi] signed] |] eqn:Ew; [| destruct Hk].
  destruct (project_range F st [] {| fm := 64; ft := ft h; flo := flo h; fhi := fhi h |}) as [[p q] |] eqn:Ep;
    [| destruct Hk].
  destruct ((q - lo <? 2 ^ fm h) && (hi - p <? 2 ^ fm h) && (negb signed || (0 <=? p))) eqn:Eb; [| destruct Hk].
  destruct Hk as [<- | []].
  rewrite !andb_true_iff in Eb. destruct Eb as [[Hq Hp] Hs]. apply Z.ltb_lt in Hq, Hp.
  pose proof (proj1 Hst r _ Eg) as Hh. unfold fact_ok in Hh. cbn [fact_okv] in Hh.
  destruct Hh as [Hwf [[e1 [He1 Hv1]] [_ Hvr]]].
  assert (He : e1 = flo h) by lia. subst e1.
  pose proof (project_range_sound st [] _ p q (flo h) Hst (Forall_nil _) Ep ltac:(cbn [flo fhi]; lia)) as Hpq.
  cbn [ft] in Hpq.
  pose proof (rs_range F E rs Hrs r) as Hr. change (0 <= rs r < 2 ^ wd F r) in Hr. rewrite Hw in Hr.
  assert (Hval : rs r = tsum (ft h) + flo h).
  { unfold residue_window in Ew.
    destruct vr as [[lo' hi'] |].
    - injection Ew as E1 E2 E3. subst lo' hi' signed. specialize (Hvr lo hi eq_refl).
      apply (eqm_close (fm h)); [lia | exact Hv1 | lia].
    - destruct (find (fun k => let '(ts, _, _) := k in terms_eqb ts [((r, ES), 1)]) (acons st))
        as [[[ts lo'] hi'] |] eqn:Ef; [| discriminate].
      injection Ew as E1 E2 E3. subst lo' hi' signed.
      apply find_some in Ef. destruct Ef as [Hin Ht]. apply terms_eqb_eq in Ht. subst ts.
      pose proof (proj1 (Forall_forall _ _) (proj2 Hst) _ Hin) as Hc. cbn [cons_ok tsum atom_val] in Hc.
      assert (Hdn : wdn r = 64%nat) by (unfold wd in Hw; lia). rewrite Hdn in Hc.
      cbn [negb orb] in Hs. apply Z.leb_le in Hs.
      assert (Hse : sext 64 (rs r) = tsum (ft h) + flo h).
      { apply (eqm_close (fm h)); [lia | | lia].
        eapply eqm_trans; [apply (eqm_weaken 64); [lia | apply sext_eqm] | exact Hv1]. }
      rewrite <- Hse. symmetry. apply sext_nonneg; [lia | exact Hr | lia]. }
  cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app, tsum_tscale. cbn [tsum atom_val].
  rewrite Hval. lia.
Qed.

Lemma check_range_sound : forall st f a b e,
    state_ok F t st rs -> check_range F st f = Some (a, b) -> flo f <= e <= fhi f ->
    a <= tsum (ft f) + e <= b.
Proof.
  intros st f a b e Hst H He. unfold check_range in H.
  eapply (project_range_sound st _ f a b e Hst); [| exact H | exact He].
  apply Forall_forall. intros k Hk. apply in_flat_map in Hk. destruct Hk as [x [_ Hk]].
  exact (residue_eq_sound st x k Hst Hk).
Qed.

Lemma check_sat_range : forall st f v L H,
    state_ok F t st rs -> fm f = 64 -> sat f v -> check_range F st f = Some (L, H) ->
    exists X, L <= X <= H /\ eqm 64 v X.
Proof.
  intros st f v L H Hst Hm [e [He Hv]] Hr. exists (tsum (ft f) + e).
  pose proof (check_range_sound st f L H e Hst Hr He). rewrite Hm in Hv. split; [lia | exact Hv].
Qed.

Lemma check_access_sat_range : forall st f v L H,
    state_ok F t st rs -> fm f = 64 -> sat f v -> check_range F st f = Some (L, H) ->
    exists X, L <= X <= H /\ eqm 64 v X.
Proof.
  intros st f v L H Hst Hm [e [He Hv]] Hr. exists (tsum (ft f) + e).
  pose proof (check_range_sound st f L H e Hst Hr He). rewrite Hm in Hv. split; [lia | exact Hv].
Qed.

Lemma inside_sound : forall st r rest ext R width,
    state_ok F t st rs -> 0 <= width -> wf rest -> fm rest = 64 -> sat rest R -> bound_form F ext = true ->
    (forall x y, r = Some (x, y) -> exists X, x <= X <= y /\ eqm 64 R X) ->
    inside F st r rest ext width = true ->
    exists X, 0 <= X /\ X + width <= bval F E ext /\ eqm 64 R X.
Proof.
  intros st r rest ext R width Hst Hw Hwr Hmr HsR Hb Hr H. unfold inside in H.
  destruct r as [[x y] |]; [| discriminate].
  destruct (Hr x y eq_refl) as [X [HX Hm]].
  apply andb_true_iff in H. destruct H as [H0 Hc]. apply Z.leb_le in H0.
  destruct (bound_sat ext Hb) as [Hwe [Hme [Hlh Hse]]].
  apply orb_true_iff in Hc. destruct Hc as [Hd | Ht].
  - (* a constant extent *)
    apply andb_true_iff in Hd. destruct Hd as [Hn Hy]. apply Z.leb_le in Hy.
    exists X. split; [lia |]. split; [| exact Hm].
    rewrite <- (bound_value ext Hb). destruct (ft ext); [| discriminate]. cbn [tsum]. lia.
  - (* through the constraints: ext - rest is at least width *)
    destruct (form_range_near F st (fadd ext (fscale rest (-1) 64))) as [[xd yd] |] eqn:Ed; [| discriminate].
    destruct (form_range_near F st ext) as [[xe ye] |] eqn:Ee; [| discriminate].
    rewrite !andb_true_iff in Ht. destruct Ht as [[[Hwd Hs] He0] He1].
    apply Z.leb_le in Hwd, He0. apply Z.ltb_lt in Hs, He1.
    assert (Hwc : wf (fscale rest (-1) 64)) by (apply wf_fscale; [exact Hwr | lia]).
    assert (HwD : wf (fadd ext (fscale rest (-1) 64))) by (apply wf_fadd; assumption).
    assert (HmD : fm (fadd ext (fscale rest (-1) 64)) = 64).
    { unfold fadd. rewrite mk_fm, Hme, (fm_fscale64 rest (-1) Hmr). reflexivity. }
    assert (HsD : sat (fadd ext (fscale rest (-1) 64)) (bval F E ext + -1 * R)).
    { apply sat_fadd; [exact Hwe | exact Hwc | exact Hse | apply sat_fscale; [exact Hwr | lia | exact HsR]]. }
    destruct (near_acc_sat_range st _ _ xd yd Hst HmD HsD Ed) as [Y [HY [HmY _]]].
    destruct (near_acc_sat_range st ext (bval F E ext) xe ye Hst Hme Hse Ee) as [XE [HXE [_ HexE]]].
    rewrite (HexE Hlh), (bound_value ext Hb) in HXE.
    (* X + Y and the extent agree modulo 2^64 and both lie in [0, 2^64) *)
    assert (Hsum : eqm 64 (bval F E ext) (X + Y)).
    { apply (eqm_trans _ _ (R + (bval F E ext + -1 * R))); [replace (R + (bval F E ext + -1 * R)) with (bval F E ext) by ring; apply eqm_refl |].
      apply eqm_add; [lia | exact Hm | exact HmY]. }
    assert (Heq : bval F E ext = X + Y) by (apply (eqm_small_eq 64 64); [lia | lia | lia | exact Hsum]).
    exists X. split; [lia | split; [lia | exact Hm]].
Qed.

Lemma inside_deep_sound : forall st rest ext R width,
    state_ok F t st rs -> 0 <= width -> wf rest -> fm rest = 64 -> sat rest R -> bound_form F ext = true ->
    inside_deep F st rest ext width = true ->
    exists X, 0 <= X /\ X + width <= bval F E ext /\ eqm 64 R X.
Proof.
  intros st rest ext R width Hst Hw Hwr Hmr HsR Hb H. unfold inside_deep in H.
  destruct (check_range F st rest) as [[x y] |] eqn:Er; [| discriminate].
  destruct (check_access_sat_range st rest R x y Hst Hmr HsR Er) as [X [HX Hm]].
  destruct (check_range F st (fadd ext (fscale rest (-1) 64))) as [[xd yd] |] eqn:Ed; [| discriminate].
  destruct (form_range_near F st ext) as [[xe ye] |] eqn:Ee; [| discriminate].
  rewrite !andb_true_iff in H. destruct H as [[[[H0 Hwd] Hs] He0] He1].
  apply Z.leb_le in H0, Hwd, He0. apply Z.ltb_lt in Hs, He1.
  destruct (bound_sat ext Hb) as [Hwe [Hme [Hlh Hse]]].
  assert (Hwc : wf (fscale rest (-1) 64)) by (apply wf_fscale; [exact Hwr | lia]).
  assert (HmD : fm (fadd ext (fscale rest (-1) 64)) = 64).
  { unfold fadd. rewrite mk_fm, Hme, (fm_fscale64 rest (-1) Hmr). reflexivity. }
  assert (HsD : sat (fadd ext (fscale rest (-1) 64)) (bval F E ext + -1 * R)).
  { apply sat_fadd; [exact Hwe | exact Hwc | exact Hse | apply sat_fscale; [exact Hwr | lia | exact HsR]]. }
  destruct (check_access_sat_range st _ _ xd yd Hst HmD HsD Ed) as [Y [HY HmY]].
  destruct (near_acc_sat_range st ext (bval F E ext) xe ye Hst Hme Hse Ee) as [XE [HXE [_ HexE]]].
  rewrite (HexE Hlh), (bound_value ext Hb) in HXE.
  assert (Hsum : eqm 64 (bval F E ext) (X + Y)).
  { apply (eqm_trans _ _ (R + (bval F E ext + -1 * R))); [replace (R + (bval F E ext + -1 * R)) with (bval F E ext) by ring; apply eqm_refl |].
    apply eqm_add; [lia | exact Hm | exact HmY]. }
  assert (Heq : bval F E ext = X + Y) by (apply (eqm_small_eq 64 64); [lia | lia | lia | exact Hsum]).
  exists X. split; [lia | split; [lia | exact Hm]].
Qed.

Lemma span_of_access_sound : forall st g a off width P sp rest,
    state_ok F t st rs -> holds g rs = true -> 0 <= width ->
    span_of_access F st g a off width = Some (P, sp, rest) ->
    role_of F P = Some sp /\
    exists e, 0 <= e /\ e + width <= bval F E (sp_ext sp) /\
              trunc 64 (rs a + off) = trunc 64 (e_init E P + e) /\
              aligned (GateDomain.rest_align F st a off P rest) e.
Proof.
  intros st g a off width P sp rest Hst Hg Hw H. unfold span_of_access in H.
  destruct (addr_sound st a off Hst) as [HwA [HmA [HnA HsA]]].
  set (A := fshift (normalize F st (as_form F st a EX)) off off) in *.
  set (v := rs a + off) in *.
  rewrite HmA, Z.eqb_refl in H. cbn [negb] in H.
  destruct (base_params F A) as [| P' [| P2 ps]] eqn:Eb; try discriminate.
  destruct (base_rest A P' v HmA HnA HsA Eb) as [[sp' Erl] Hrest].
  rewrite Erl in H.
  set (rest' := mk 64 (without (ft A) (P', EX)) (flo A) (fhi A)) in *.
  assert (Hmr : fm rest' = 64) by (unfold rest'; apply mk_fm).
  assert (Hwr : wf rest') by (unfold wf; lia).
  assert (HP : rs P' = e_init E P') by exact (role_rs P' sp' Erl).
  pose proof (ok_spans F E HE P' sp' Erl) as Hsp. unfold span_ok in Hsp. apply andb_true_iff in Hsp.
  destruct Hsp as [Hext _].
  assert (Hal : forall X, eqm 64 (v - rs P') X -> aligned (GateDomain.rest_align F st a off P' rest') X)
    by (intros X HX; exact (rest_align_sound st a off P' rest' X Hst Hmr Hrest HX)).
  set (al := GateDomain.rest_align F st a off P' rest') in *.
  assert (Hin : exists X, 0 <= X /\ X + width <= bval F E (sp_ext sp') /\ eqm 64 (v - rs P') X).
  { destruct (inside F st (form_range_near F st rest') rest' (sp_ext sp') width) eqn:E1.
    - apply (inside_sound st (form_range_near F st rest') rest' _ (v - rs P') width Hst Hw Hwr Hmr Hrest Hext); [| exact E1].
      intros x y Er. destruct (near_acc_sat_range st rest' (v - rs P') x y Hst Hmr Hrest Er) as [X [HX [Hv _]]].
      exists X. split; [exact HX | exact Hv].
    - cbn [orb] in H.
      destruct (inside F st (guard_bound F st g rest' al) rest' (sp_ext sp') width) eqn:E2.
      + apply (inside_sound st (guard_bound F st g rest' al) rest' _ (v - rs P') width Hst Hw Hwr Hmr Hrest Hext);
          [| exact E2].
        intros x y Er. exact (guard_bound_sound st g rest' (v - rs P') al x y Hst Hg Hmr Hrest Hal Er).
      + cbn [orb] in H.
        destruct (inside_deep F st rest' (sp_ext sp') width) eqn:E3; [| discriminate].
        exact (inside_deep_sound st rest' _ (v - rs P') width Hst Hw Hwr Hmr Hrest Hext E3). }
  destruct Hin as [X [HX0 [HXw HmX]]].
  assert (Hsome : Some (P', sp', rest') = Some (P, sp, rest)).
  { destruct (if inside F st (form_range_near F st rest') rest' (sp_ext sp') width then true else
              if inside F st (guard_bound F st g rest' al) rest' (sp_ext sp') width then true else
              inside_deep F st rest' (sp_ext sp') width); [exact H | discriminate]. }
  injection Hsome as <- <- <-.
  split; [exact Erl |]. exists X. split; [exact HX0 |]. split; [exact HXw |]. split; [| exact (Hal X HmX)].
  apply trunc64_eqm. rewrite <- HP. apply eqm_base. exact HmX.
Qed.

(* ------------------------------------------------------------------ *)
(* Accesses inside a variable.                                         *)

Lemma nodup_map_filter : forall (A B : Type) (f : A -> B) (p : A -> bool) l,
    NoDup (map f l) -> NoDup (map f (filter p l)).
Proof.
  intros A B f p l. induction l as [| x l IH]; intros Hn; cbn; [constructor |].
  inversion Hn as [| ? ? Hx Hl]; subst. destruct (p x); [| exact (IH Hl)].
  cbn. constructor; [| exact (IH Hl)].
  intros Hin. apply Hx. apply in_map_iff in Hin. destruct Hin as [y [Hy Hin]].
  apply filter_In in Hin. apply in_map_iff. exists y. split; [exact Hy | exact (proj1 Hin)].
Qed.

Lemma has_unit_in : forall ts x, has_unit ts x = true -> In ((x, EX), 1) ts.
Proof.
  intros ts x H. unfold has_unit in H. apply existsb_exists in H. destruct H as [[y c] [Hin H]].
  apply andb_true_iff in H. destruct H as [Hy Hc]. apply atom_eqb_eq in Hy. apply Z.eqb_eq in Hc.
  cbn [fst snd] in Hy, Hc. subst y c. exact Hin.
Qed.

Lemma access_range_sound : forall st f lower upper a b e,
 state_ok F t st rs -> access_range F st f lower upper = Some (a,b) ->
 flo f <= e <= fhi f -> a <= tsum (ft f) + e <= b.
Proof.
 intros st f lower upper a b e Hst H He. unfold access_range in H.
 destruct (bounded_form_range F st f lower upper) as [[p q]|] eqn:Ef.
 - destruct ((lower <=? p) && (q <=? upper)).
   + injection H as <- <-. exact (bounded_form_range_sound F E HE t st rs Hst Hrs f lower upper p q e Ef He).
   + exact (check_range_sound st f a b e Hst H He).
 - exact (check_range_sound st f a b e Hst H He).
Qed.

Lemma access_sat_range : forall st f lower upper v L H,
 state_ok F t st rs -> fm f = 64 -> sat f v ->
 access_range F st f lower upper = Some (L,H) ->
 exists X, L <= X <= H /\ eqm 64 v X.
Proof.
 intros st f lower upper v L H Hst Hm [e [He Hv]] Hr.
 exists (tsum (ft f) + e). split.
 - exact (access_range_sound st f lower upper L H e Hst Hr He).
 - rewrite Hm in Hv. exact Hv.
Qed.

(* The allocation extent and access are checked in the same launch state. *)
Lemma var_extent_sound : forall st rest bytes upper width al R,
 state_ok F t st rs -> 0 <= width -> wf rest -> fm rest = 64 -> sat rest R ->
 al <= 64 -> aligned al R ->
 bound_form F bytes = true -> var_extent F st rest bytes upper width al = true ->
 bval F E bytes + upper <= window_size /\
 exists X, 0 <= X /\ X + width <= bval F E bytes /\ eqm 64 R X.
Proof.
 intros st rest bytes upper width al R Hst Hw Hwr Hmr Hsat Hal HR Hb H.
 unfold var_extent in H.
 destruct (bound_sat bytes Hb) as [Hwb [Hmb [Hlh Hsb]]].
 destruct (null (ft bytes)) eqn:Hnull.
 - destruct (access_range F st rest 0 (flo bytes - width)) as [[x y]|] eqn:Er; [|discriminate].
   destruct (access_sat_range st rest 0 (flo bytes-width) R x y Hst Hmr Hsat Er) as [X [HX HM]].
   assert (HA : aligned al X).
   { apply (aligned_eqm 64 al X R); [lia|exact Hal|apply eqm_sym; exact HM|exact HR]. }
   pose proof (round_range_sound al x y X HX HA) as Hround.
   destruct (round al (x,y)) as [a b] eqn:Eround. cbn [fst snd] in Hround.
   rewrite !andb_true_iff in H. destruct H as [[Hu Hx] Hy]. apply Z.leb_le in Hu,Hx,Hy.
   assert (Hbytes : bval F E bytes = flo bytes).
   { unfold bval. destruct (ft bytes); [reflexivity|discriminate]. }
   rewrite Hbytes. split; [lia|].
   exists X. split; [lia|]. split; [lia|exact HM].
 - destruct (form_range F st bytes) as [[lo hi]|] eqn:Er; [|discriminate].
   apply andb_true_iff in H. destruct H as [Hu Hinside]. apply Z.leb_le in Hu.
   pose proof (form_range_sound F E HE t st rs Hst Hrs bytes lo hi (flo bytes) Er ltac:(lia)) as Hr.
   rewrite (bound_value bytes Hb) in Hr. split; [lia|].
   exact (inside_deep_sound st rest bytes R width Hst Hw Hwr Hmr Hsat Hb Hinside).
Qed.

(* Strip the variable base, then check the complete allocation extent. *)
Lemma var_tail : forall vars st T lo0 hi0 v e b width (aligns : reg -> Z),
    state_ok F t st rs -> NoDup (map fst T) -> lo0 <= e <= hi0 -> eqm 64 (v - b) (tsum T + e) ->
    0 <= b -> b + window_size < 2 ^ 64 -> 0 <= width ->
    (forall u, aligns u <= 64 /\ aligned (aligns u) (v-b-rs u)) ->
    existsb (fun e0 =>
               let '(u, bytes) := e0 in
               is_fixed F u &&
               match find (unit_of u) T with
               | Some (xa, _) =>
                   match lookup (lf_bounds F) u with
                   | Some (lo, hi) => (0 <=? lo) && bound_form F bytes &&
                       var_extent F st (mk 64 (without T xa) lo0 hi0) bytes hi width (aligns u)
                   | None => false end
               | None => false end) vars = true ->
    exists u bytes X, In (u, bytes) vars /\ 0 <= e_init E u /\ e_init E u + bval F E bytes <= window_size /\
      0 <= X /\ X + width <= bval F E bytes /\ trunc 64 v = b + e_init E u + X.
Proof.
  intros vars st T lo0 hi0 v e b width aligns Hst Hn He Hv Hb0 Hbw Hw Hal H.
  apply existsb_exists in H. destruct H as [[u bytes] [Hin H]].
  apply andb_true_iff in H. destruct H as [Hfix Hr].
  destruct (find (unit_of u) T) as [[xa c] |] eqn:Ef; [| discriminate].
  destruct (find_some _ _ Ef) as [HinT Hu]. unfold unit_of in Hu. cbn [fst snd] in Hu.
  rewrite !andb_true_iff in Hu. destruct Hu as [[Hxu Hex] Hc]. apply Z.eqb_eq in Hc. subst c.
  assert (Hval : aval xa = rs u).
  { destruct xa as [r ex]. cbn [fst] in Hxu. apply Pos.eqb_eq in Hxu. subst r.
    cbn [snd] in Hex. destruct ex; try discriminate; reflexivity. }
  destruct (lookup (lf_bounds F) u) as [[lo hi] |] eqn:El; [| discriminate].
  rewrite !andb_true_iff in Hr. destruct Hr as [[Hlo Hbound] Hextent]. apply Z.leb_le in Hlo.
  set (rest := mk 64 (without T xa) lo0 hi0) in *.
  set (Tf := {| fm := 64; ft := T; flo := lo0; fhi := hi0 |}).
  assert (Hs : sat Tf (v - b)) by (exists e; cbn; split; [exact He | exact Hv]).
  pose proof (sat_strip Tf xa 1 (v - b) eq_refl Hn HinT Hs) as Hsu.
  cbn [ft flo fhi] in Hsu. rewrite Hval in Hsu. fold rest in Hsu.
  assert (Hmr : fm rest = 64) by (unfold rest; apply mk_fm).
  assert (Hwr : wf rest) by (unfold rest; apply wf_mk; lia).
  destruct (Hal u) as [Hal64 Halu].
  replace (v-b-rs u) with (v-b-1*rs u) in Halu by ring.
  destruct (var_extent_sound st rest bytes hi width (aligns u) (v-b-1*rs u) Hst Hw Hwr Hmr Hsu Hal64 Halu Hbound Hextent)
    as [Hsize [X [HX0 [HXw HvX]]]].
  pose proof (ok_bounds F E HE u lo hi Hfix El) as Hbu.
  assert (Hru : rs u = e_init E u) by exact (fixed_rs u Hfix).
  exists u, bytes, X. split; [exact Hin |]. split; [lia |]. split; [lia |]. split; [lia |]. split; [lia |].
  replace (v - b - 1 * rs u) with (v - (b + rs u)) in HvX by ring.
  rewrite (trunc64_eqm v (b + rs u + X) (eqm_base v (b + rs u) X HvX)).
  unfold trunc. change (Z.of_nat 64) with 64. rewrite Hru. apply Z.mod_small.
  unfold window_size in *. lia.
Qed.

Lemma private_align_sound : forall st base a u off,
 state_ok F t st rs ->
 private_align F st base a u off <= 64 /\
 aligned (private_align F st base a u off)
   (rs a + off - (match base with Some w => rs w | None => 0 end) - rs u).
Proof.
 intros st base a u off Hst. split; [unfold private_align; lia|].
 set (al := private_align F st base a u off).
 replace (rs a + off - (match base with Some w => rs w | None => 0 end) - rs u)
   with ((rs a + off) + - (match base with Some w => rs w | None => 0 end) + - rs u) by ring.
 apply aligned_add; [apply aligned_add; [apply aligned_add|]|].
 - apply (aligned_mono (reg_align F st a)); [unfold al, private_align; lia|exact (reg_align_sound F E HE t st rs Hst Hrs a)].
 - apply (aligned_mono (v2 off)); [unfold al, private_align; lia|apply aligned_v2].
 - apply aligned_opp. destruct base as [w|]; [|apply aligned_zero].
   apply (aligned_mono (reg_align F st w)); [unfold al, private_align; lia|exact (reg_align_sound F E HE t st rs Hst Hrs w)].
 - apply aligned_opp. apply (aligned_mono (reg_align F st u)); [unfold al, private_align; lia|exact (reg_align_sound F E HE t st rs Hst Hrs u)].
Qed.

Definition base_value (base : option reg) : Z := match base with Some w => e_init E w | None => 0 end.

(* An access that var_of places inside a variable lies inside it. *)
Lemma var_of_sound : forall vars base st a off width,
    state_ok F t st rs -> 0 <= width -> var_of F vars base st a off width = true ->
    exists u bytes X, In (u, bytes) vars /\ 0 <= e_init E u /\ e_init E u + bval F E bytes <= window_size /\
      0 <= X /\ X + width <= bval F E bytes /\ trunc 64 (rs a + off) = base_value base + e_init E u + X /\
      0 <= base_value base /\ base_value base + window_size < 2 ^ 64.
Proof.
  intros vars base st a off width Hst Hw H.
  destruct (addr_sound_ext st a off (value_ext F a) (value_ext_exact a) Hst) as [_ [HmA [HnA HsA]]].
  unfold var_of in H.
  set (A := fshift (normalize F st (as_form F st a (value_ext F a))) off off) in *.
  set (v := rs a + off) in *.
  rewrite HmA, Z.eqb_refl in H. cbn [andb] in H.
  destruct HsA as [e [He Hv]]. rewrite HmA in Hv.
  apply andb_true_iff in H. destruct H as [Hbase Hvars].
  destruct base as [w |]; cbn [base_value].
  - (* a generic address: the window's base, then a variable *)
    rewrite !andb_true_iff in Hbase. destruct Hbase as [[Hfix Hunit] Hb].
    destruct (lookup (lf_bounds F) w) as [[lo hi] |] eqn:El; [| discriminate].
    rewrite andb_true_iff in Hb. destruct Hb as [Hlo Hhi]. apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi.
    pose proof (ok_bounds F E HE w lo hi Hfix El) as Hbw.
    assert (Hrw : rs w = e_init E w) by exact (fixed_rs w Hfix).
    pose proof (tsum_without (ft A) (w, EX) 1 HnA (has_unit_in (ft A) w Hunit)) as Hsplit.
    cbn [atom_val] in Hsplit.
    assert (Hv' : eqm 64 (v - rs w) (tsum (without (ft A) (w, EX)) + e)).
    { rewrite Hsplit in Hv. pose proof (eqm_add 64 _ _ (- rs w) (- rs w) ltac:(lia) Hv (eqm_refl 64 _)) as Hm.
      replace (v - rs w) with (v + - rs w) by ring.
      replace (tsum (without (ft A) (w, EX)) + e) with (tsum (without (ft A) (w, EX)) + 1 * rs w + e + - rs w)
        by ring.
      exact Hm. }
    rewrite Hrw in Hv'.
    assert (Hal : forall u, private_align F st (Some w) a u off <= 64 /\
      aligned (private_align F st (Some w) a u off) (v-e_init E w-rs u)).
    { intro u. pose proof (private_align_sound st (Some w) a u off Hst) as Ha.
      cbn in Ha. rewrite Hrw in Ha. exact Ha. }
    destruct (var_tail vars st (without (ft A) (w, EX)) (flo A) (fhi A) v e (e_init E w) width (fun u => private_align F st (Some w) a u off) Hst
                (nodup_map_filter _ _ fst _ (ft A) HnA) He Hv' ltac:(lia) ltac:(lia) Hw Hal Hvars)
      as [u [bytes [X [Hin [Hu0 [Hub [HX0 [HXw Hx]]]]]]]].
    exists u, bytes, X. repeat (split; [assumption |]). split; lia.
  - (* an address of shared or local memory: a variable *)
    assert (Hv0 : eqm 64 (v - 0) (tsum (ft A) + e)) by (replace (v - 0) with v by ring; exact Hv).
    assert (Hal : forall u, private_align F st None a u off <= 64 /\
      aligned (private_align F st None a u off) (v-0-rs u)).
    { intro u. exact (private_align_sound st None a u off Hst). }
    destruct (var_tail vars st (ft A) (flo A) (fhi A) v e 0 width (fun u => private_align F st None a u off) Hst HnA He Hv0 ltac:(lia)
                ltac:(unfold window_size; lia) Hw Hal Hvars)
      as [u [bytes [X [Hin [Hu0 [Hub [HX0 [HXw Hx]]]]]]]].
    exists u, bytes, X. repeat (split; [assumption |]). unfold window_size. split; lia.
Qed.

(* A generic address inside a variable in a window lies in the window. *)
Lemma in_private_window : forall x width, in_private F E x width -> in_window F E x width = true.
Proof.
  intros x width [[u [bytes [_ [Hu0 [Hub [Hlo Hhi]]]]]] | [u [bytes [_ [Hu0 [Hub [Hlo Hhi]]]]]]];
    unfold in_window; apply existsb_exists;
    [exists (lf_shared_window F) | exists (lf_local_window F)];
    (split; [cbn; tauto |]); apply andb_true_iff; split; apply Z.leb_le; lia.
Qed.

(* A value whose integer value vf gives, which form_in_range checks, lies in
   the index range. *)
Lemma form_in_range_sound : forall st vf ix e0 V,
    state_ok F t st rs -> bound_form F (ix_lo ix) = true -> bound_form F (ix_hi ix) = true ->
    flo vf <= e0 <= fhi vf -> tsum (ft vf) + e0 = V -> form_in_range F st vf ix = true ->
    bval F E (ix_lo ix) <= V <= bval F E (ix_hi ix).
Proof.
  intros st vf ix e0 V Hst Hlo Hhi He0 Hv H. unfold form_in_range in H.
  (* the difference of the value and a bound, read through the constraints *)
  assert (Hd : forall b x y, bound_form F b = true ->
            check_range F st {| fm := 64; ft := sorted_terms (ft vf ++ tscale (-1) (ft b));
                                flo := flo vf - flo b; fhi := fhi vf - flo b |} = Some (x, y) ->
            x <= V - bval F E b <= y).
  { intros b x y Hb Hcr.
    pose proof (bound_value b Hb) as Hbv.
    pose proof (check_range_sound st _ x y (e0 - flo b) Hst Hcr ltac:(cbn [flo fhi]; lia)) as Hr.
    cbn [ft] in Hr. rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app, tsum_tscale in Hr. lia. }
  destruct (check_range F st {| fm := 64; ft := sorted_terms (ft vf ++ tscale (-1) (ft (ix_lo ix)));
                                flo := flo vf - flo (ix_lo ix); fhi := fhi vf - flo (ix_lo ix) |})
    as [[x1 y1] |] eqn:E1; [| discriminate].
  destruct (check_range F st {| fm := 64; ft := sorted_terms (ft vf ++ tscale (-1) (ft (ix_hi ix)));
                                flo := flo vf - flo (ix_hi ix); fhi := fhi vf - flo (ix_hi ix) |})
    as [[x2 y2] |] eqn:E2; [| discriminate].
  apply andb_true_iff in H. destruct H as [H1 H2]. apply Z.leb_le in H1, H2.
  pose proof (Hd _ _ _ Hlo E1). pose proof (Hd _ _ _ Hhi E2). lia.
Qed.

(* A value in_range checks lies in the index range. *)
Lemma in_range_sound : forall st o ix,
    state_ok F t st rs -> bound_form F (ix_lo ix) = true -> bound_form F (ix_hi ix) = true ->
    1 <= 8 * ix_elem ix <= 64 -> in_range F st o ix = true ->
    bval F E (ix_lo ix) <= extend (Z.to_nat (8 * ix_elem ix)) (ix_signed ix) (opv rs o) <= bval F E (ix_hi ix).
Proof.
  intros st o ix Hst Hlo Hhi Hw H. unfold in_range in H.
  set (w := 8 * ix_elem ix) in *. set (sg := ix_signed ix) in *.
  lazymatch type of H with (match ?m with _ => _ end) = true => destruct m as [[wr sgr] same] eqn:Em end.
  cbv beta iota zeta in H. apply orb_true_iff in H. destruct H as [H | H].
  2: { (* a register of the element's width whose bits lie below the sign bit *)
       destruct o as [r | z]; [| discriminate].
       apply andb_true_iff in H. destruct H as [Hwr H]. apply Z.eqb_eq in Hwr.
       set (bits := {| fm := 64; ft := [((r, EZ), 1)]; flo := 0; fhi := 0 |}) in *.
       destruct (check_range F st bits) as [[x0 y0] |] eqn:Ec; [| discriminate].
       rewrite !andb_true_iff in H. destruct H as [[Hx0 Hy0] H]. apply Z.leb_le in Hx0. apply Z.ltb_lt in Hy0.
       assert (Ha : tsum (ft bits) + 0 = rs r) by (cbn [bits ft tsum atom_val]; ring).
       assert (Hlh : flo bits <= 0 <= fhi bits) by (cbn [bits flo fhi]; lia).
       pose proof (check_range_sound st _ x0 y0 0 Hst Ec Hlh) as Hr. rewrite Ha in Hr.
       assert (Hext : extend (Z.to_nat w) sg (opv rs (OReg r)) = rs r).
       { cbn [opv]. assert (Hwn : Z.of_nat (Z.to_nat w) = w) by lia.
         destruct sg; cbn [extend].
         - apply sext_small; [lia | rewrite Hwn; lia].
         - apply trunc_small. rewrite Hwn. assert (2 ^ (w - 1) <= 2 ^ w) by (apply Z.pow_le_mono_r; lia). lia. }
       rewrite Hext. exact (form_in_range_sound st _ ix 0 (rs r) Hst Hlo Hhi Hlh Ha H). }
  destruct (cmp_range F st o wr sgr) as [[x0 y0] |] eqn:Er; [| discriminate].
  apply andb_true_iff in H. destruct H as [Hs H].
  (* the value read at width wr is the element's value *)
  assert (Hread : exists e0, flo (cmp_form F st o wr sgr) <= e0 <= fhi (cmp_form F st o wr sgr) /\
                             tsum (ft (cmp_form F st o wr sgr)) + e0 = extend (Z.to_nat w) sg (opv rs o)).
  { destruct o as [r | z].
    - destruct (Z.eqb_spec (wd F r) w) as [Hwr | Hwr]; injection Em as <- <- <-.
      + assert (Ho : owidth F (OReg r) w) by (intros r' Hr'; injection Hr' as <-; exact Hwr).
        exact (cmp_exact F E HE t rs Hrs st (OReg r) w sg x0 y0 Hst Hw Ho Er).
      + pose proof (wd_range F E HE r) as Hwd. pose proof (rs_range F E rs Hrs r) as Hrr.
        assert (Ho : owidth F (OReg r) (wd F r)) by (intros r' Hr'; injection Hr' as <-; reflexivity).
        assert (Hw' : 1 <= wd F r <= 64) by (unfold wd; lia).
        destruct (cmp_exact F E HE t rs Hrs st (OReg r) (wd F r) false x0 y0 Hst Hw' Ho Er) as [e0 [He0 Hv]].
        pose proof (cmp_range_sound F E HE t rs Hrs st (OReg r) (wd F r) false x0 y0 Hst Hw' Ho Er) as Hxy.
        cbn [orb] in Hs. rewrite andb_true_iff, Z.leb_le, Z.ltb_lt in Hs.
        unfold cval in Hv, Hxy. cbn [opv extend] in Hv, Hxy |- *.
        assert (Hz : trunc (Z.to_nat (wd F r)) (rs r) = rs r)
          by (apply trunc_small; unfold wd; rewrite Nat2Z.id; exact Hrr).
        rewrite Hz in Hv, Hxy. exists e0. split; [exact He0 | rewrite Hv].
        assert (Hwn : Z.of_nat (Z.to_nat w) = w) by lia.
        destruct sg; cbn [extend].
        * symmetry. apply sext_small; [lia | rewrite Hwn; lia].
        * symmetry. apply trunc_small. rewrite Hwn.
          assert (2 ^ (w - 1) <= 2 ^ w) by (apply Z.pow_le_mono_r; lia). lia.
    - injection Em as <- <- <-.
      assert (Ho : owidth F (OImm z) w) by (intros r' Hr'; discriminate).
      exact (cmp_exact F E HE t rs Hrs st (OImm z) w sg x0 y0 Hst Hw Ho Er). }
  destruct Hread as [e0 [He0 Hv]].
  exact (form_in_range_sound st _ ix e0 _ Hst Hlo Hhi He0 Hv H).
Qed.

(* An element size is the power of two of its valuation. *)
Lemma elem_v2 : forall elem, elem = 1 \/ elem = 2 \/ elem = 4 \/ elem = 8 -> 2 ^ v2 elem = elem.
Proof. intros elem [-> | [-> | [-> | ->]]]; reflexivity. Qed.

Lemma refine_origins_sound : forall st xs,
 state_ok F t st rs -> exists out, refine_origins F st xs = Some out /\ state_ok F t out rs.
Proof.
 intros st xs Hst.
 assert (Hone : forall s p, state_ok F t s rs -> exists out,
   (match check_range F s (atom64 p EX) with
    | Some (lo,hi) => if 1 <=? lo then refine F s p true else if hi <=? 0 then refine F s p false else Some s
    | None => Some s end) = Some out /\ state_ok F t out rs).
 { intros s p Hs. destruct (check_range F s (atom64 p EX)) as [[lo hi] |] eqn:Er; [| exists s; auto].
   pose proof (check_range_sound s (atom64 p EX) lo hi 0 Hs Er ltac:(change (0 <= 0 <= 0); lia)) as Hr.
   change (lo <= 1 * rs p + 0 + 0 <= hi) in Hr.
   destruct (1 <=? lo) eqn:El.
   - apply Z.leb_le in El. apply (refine_sound F E HE t rs Hrs s p true Hs).
     apply negb_true_iff. apply Z.eqb_neq. lia.
   - destruct (hi <=? 0) eqn:Eh; [| exists s; auto].
     apply Z.leb_le in Eh. pose proof (rs_range F E rs Hrs p) as Hp.
     apply (refine_sound F E HE t rs Hrs s p false Hs).
     apply negb_false_iff. apply Z.eqb_eq. lia. }
 unfold refine_origins. revert st Hst.
 induction xs as [| [p e] xs IH]; intros st Hst; cbn [fold_left fst]; [exists st; auto |].
 destruct (Hone st p Hst) as [out [Eout Hout]]. rewrite Eout. apply IH. exact Hout.
Qed.

Lemma access_guard_sound : forall st refined g,
 state_ok F t refined rs ->
 exists out, access_guard F st refined g = Some out /\ state_ok F t out rs.
Proof.
 intros st refined [[neg p]|] Hst; [|exists refined;auto].
 unfold access_guard. apply refine_origins_sound. exact Hst.
Qed.

Lemma access_state_ok_sound : forall st g k gen a off width v,
    state_ok F t st rs -> holds g rs = true -> 0 <= width -> access_state_ok F st g k gen a off width v = true ->
    region F E t k (trunc 64 (rs a + off)) width (match k, v with KStore, Some o => Some (opv rs o) | _, _ => None end) \/
    (gen = true /\ in_private F E (trunc 64 (rs a + off)) width).
Proof.
  intros st g k gen a off width v Hst Hg Hw H.
  unfold access_state_ok in H.
  destruct (span_of_access F st g a off width) as [[[P sp] rest] |] eqn:Es.
  - destruct (span_of_access_sound st g a off width P sp rest Hst Hg Hw Es) as [Erl [e [He [Hew [Haddr Hal]]]]].
    apply andb_true_iff in H. destruct H as [Hdir Hval].
    unfold value_ok in Hval. apply andb_true_iff in Hval. destruct Hval as [Hflag Hval].
    left. rewrite Haddr. apply (RgBuffer F E t k P sp e width _ Erl Hdir He Hew).
    + (* an access to a span with an index range reads or writes one whole element *)
      intros ix Eix. rewrite Eix in Hval.
      pose proof (ok_spans F E HE P sp Erl) as Hsp. unfold span_ok in Hsp. rewrite Eix in Hsp.
      rewrite !andb_true_iff in Hsp. destruct Hsp as [_ [[[Hel _] _] _]].
      assert (Helem : ix_elem ix = 1 \/ ix_elem ix = 2 \/ ix_elem ix = 4 \/ ix_elem ix = 8).
      { rewrite !orb_true_iff, !Z.eqb_eq in Hel. tauto. }
      assert (Hwhole : Z.eqb width (ix_elem ix) && (v2 (ix_elem ix) <=? GateDomain.rest_align F st a off P rest) = true).
      { destruct k; [exact Hval | | discriminate Hval].
        destruct v as [o |]; [| discriminate Hval]. apply andb_true_iff in Hval. exact (proj1 Hval). }
      apply andb_true_iff in Hwhole. destruct Hwhole as [Hwd Hra].
      apply Z.eqb_eq in Hwd. apply Z.leb_le in Hra.
      split; [exact Hwd |].
      apply Z.mod_divide; [destruct Helem as [-> | [-> | [-> | ->]]]; lia |].
      rewrite <- (elem_v2 _ Helem).
      pose proof (aligned_mono _ (v2 (ix_elem ix)) e Hra Hal) as Hd. unfold aligned in Hd.
      rewrite Z.max_r in Hd by apply v2_nonneg. exact Hd.
    + (* a write into a span with an index range *)
      intros ix Eix Hk. rewrite Eix in Hval.
      pose proof (ok_spans F E HE P sp Erl) as Hsp. unfold span_ok in Hsp. rewrite Eix in Hsp.
      rewrite !andb_true_iff in Hsp. destruct Hsp as [_ [[[Hel Hblo] Hbhi] _]].
      assert (Helem : ix_elem ix = 1 \/ ix_elem ix = 2 \/ ix_elem ix = 4 \/ ix_elem ix = 8).
      { rewrite !orb_true_iff, !Z.eqb_eq in Hel. tauto. }
      destruct k; [contradiction | | discriminate].
      destruct v as [o |]; [| discriminate].
      rewrite !andb_true_iff in Hval. destruct Hval as [[Hwd Hra] Hin].
      apply Z.eqb_eq in Hwd. apply Z.leb_le in Hra.
      split; [reflexivity |]. split; [exact Hwd |]. split.
      * apply Z.mod_divide; [destruct Helem as [-> | [-> | [-> | ->]]]; lia |].
        rewrite <- (elem_v2 _ Helem).
        pose proof (aligned_mono _ (v2 (ix_elem ix)) e Hra Hal) as Hd. unfold aligned in Hd.
        rewrite Z.max_r in Hd by apply v2_nonneg. exact Hd.
      * exists (opv rs o). split; [reflexivity |].
        apply (in_range_sound st o ix Hst Hblo Hbhi); [destruct Helem as [-> | [-> | [-> | ->]]]; lia | exact Hin].
    + (* a write into a flag span stores 1 into one whole word *)
      intros Hf. rewrite Hf in Hflag. cbn [negb orb] in Hflag.
      destruct k; try discriminate Hflag.
      destruct v as [[r | z] |]; try discriminate Hflag.
      rewrite !andb_true_iff in Hflag. destruct Hflag as [[Hwd Hra] Hz].
      apply Z.eqb_eq in Hwd, Hz. apply Z.leb_le in Hra. subst width z.
      split; [reflexivity |]. split; [reflexivity |]. split; [| reflexivity].
      apply Z.mod_divide; [lia |].
      pose proof (aligned_mono _ 2 e Hra Hal) as Hd. unfold aligned in Hd. exact Hd.
  - apply andb_true_iff in H. destruct H as [Hgen Hwin]. right.
    split; [exact Hgen |]. unfold in_private.
    apply orb_true_iff in Hwin. destruct Hwin as [Hwin | Hwin]; [left | right; apply andb_true_iff in Hwin; destruct Hwin as [_ Hwin]];
      destruct (var_of_sound _ _ st a off width Hst Hw Hwin) as [u [bytes [X [Hin [Hu0 [Hub [HX0 [HXw [Hx _]]]]]]]]];
      exists u, bytes; cbn [base_value] in Hx; rewrite Hx; (split; [exact Hin |]); lia.
Qed.

Theorem access_ok_sound : forall st g k gen a off width v,
    state_ok F t st rs -> holds g rs = true -> 0 <= width -> access_ok F st g k gen a off width v = true ->
    region F E t k (trunc 64 (rs a + off)) width (match k, v with KStore, Some o => Some (opv rs o) | _, _ => None end) \/
    (gen = true /\ in_private F E (trunc 64 (rs a + off)) width).
Proof.
 intros st g k gen a off width v Hst Hg Hw H.
 destruct (refine_guard_sound F E HE t rs Hrs st g Hst Hg) as [sg [Esg Hsg]].
 unfold access_ok in H. rewrite Esg in H.
 destruct (access_state_ok F sg g k gen a off width v) eqn:Eb.
 - exact (access_state_ok_sound sg g k gen a off width v Hsg Hg Hw Eb).
 - destruct g as [[neg p]|]; [|discriminate].
   destruct (access_guard_sound st sg (Some(neg,p)) Hsg) as [out [Eout Hout]].
   rewrite Eout in H. exact (access_state_ok_sound out (Some(neg,p)) k gen a off width v Hout Hg Hw H).
Qed.


(* ------------------------------------------------------------------ *)
(* Alignment.                                                          *)

(* A form whose constant is a range gives no alignment. *)
Lemma form_align_range : forall st f, flo f <> fhi f -> form_align F st f <= 0.
Proof.
  intros st f Hlh. unfold form_align, form_align_with.
  assert (Hc : const_align f = 0) by (unfold const_align; destruct (Z.eqb_spec (flo f) (fhi f)); [contradiction | reflexivity]).
  rewrite Hc.
  assert (Hfold : forall ts k, k <= 0 ->
            fold_left (fun k0 '(x, c) => Z.min k0 (v2 c + match snd x with EX | EZ | ES => ra F DEPTH st (fst x) | _ => 0 end))
                      ts k <= 0).
  { intros ts. induction ts as [| [x c] ts IH]; intros k Hk; cbn [fold_left]; [exact Hk |]. apply IH. lia. }
  exact (Hfold (ft f) 0 (Z.le_refl 0)).
Qed.

(* The address a + off is a multiple of 2^k for each k up to its alignment
   and up to 64. *)
Lemma addr_align_sound : forall st a off k,
    state_ok F t st rs -> k <= addr_align F st a off -> k <= 64 -> aligned k (rs a + off).
Proof.
  intros st a off k Hst Hk H64. unfold addr_align in Hk.
  destruct (Z.max_spec (form_align F st (addr_form F st a off)) (Z.min (reg_align F st a) (v2 off)))
    as [[_ Hm] | [_ Hm]]; rewrite Hm in Hk.
  - pose proof (reg_align_sound F E HE t st rs Hst Hrs a) as Ha.
    apply aligned_add.
    + apply (aligned_mono (reg_align F st a)); [lia | exact Ha].
    + apply (aligned_mono (v2 off)); [lia | apply aligned_v2].
  - destruct (addr_sound st a off Hst) as [_ [HmA [_ [e [He Hs]]]]].
    unfold addr_form in Hk. set (A := fshift (normalize F st (as_form F st a EX)) off off) in *.
    destruct (Z.eqb_spec (flo A) (fhi A)) as [Hlh | Hlh]; [| apply aligned_nonpos; pose proof (form_align_range st A Hlh); lia].
    assert (e = flo A) by lia. subst e. rewrite HmA in Hs.
    apply (aligned_eqm 64 k (rs a + off) (tsum (ft A) + flo A)); [lia | exact H64 | exact Hs |].
    apply (aligned_mono (form_align F st A)); [exact Hk |].
    apply (form_align_sound F E HE t st rs Hst Hrs); [exact Hlh | lia].
Qed.

End Access.

(* A projection reads the ranges of the registers of its goal and of its
   constraints, and through the exact forms of these registers the ranges
   of the forms' registers. Range functions that agree on such registers
   give one projection. *)
Definition cons_in (P : reg -> Prop) (cs : list constraint) : Prop := forall k, In k cs -> terms_in P (fst (fst k)).

Lemma cons_in_app : forall P a b, cons_in P a -> cons_in P b -> cons_in P (a ++ b).
Proof. intros P a b Ha Hb k Hk. apply in_app_or in Hk. destruct Hk as [Hk|Hk];[exact (Ha k Hk)|exact (Hb k Hk)]. Qed.
Lemma cons_in_filter : forall P (g : constraint -> bool) cs, cons_in P cs -> cons_in P (filter g cs).
Proof. intros P g cs H k Hk. apply filter_In in Hk. exact (H k (proj1 Hk)). Qed.
Lemma flat_map_ext_in : forall (A B : Type) (f g : A -> list B) l,
 (forall x, In x l -> f x = g x) -> flat_map f l = flat_map g l.
Proof.
 intros A B f g l H. induction l as [|x l IH];[reflexivity|]. cbn [flat_map].
 rewrite (H x (or_introl eq_refl)),IH;[reflexivity|]. intros y Hy. apply H. right. exact Hy.
Qed.
Lemma terms_in_sorted : forall P ts, terms_in P ts -> terms_in P (sorted_terms ts).
Proof. intros P ts H. unfold sorted_terms. apply terms_in_filter,terms_in_tsort,H. Qed.
Lemma terms_in_canon : forall F P a ts, terms_in P ts -> terms_in P (canon_terms_with F a ts).
Proof.
 intros F P a ts H. unfold canon_terms_with. apply terms_in_sorted.
 intros t Ht. apply in_map_iff in Ht. destruct Ht as [[[r e] c] [<- Hin]].
 pose proof (H _ Hin) as Hr. cbn [fst] in *. destruct e;cbn [canon_atom_with];try exact Hr.
 destruct (f_rr a r) as [[lo hi]|];[destruct (_ && _)|];exact Hr.
Qed.
Lemma atoms_nodup_in : forall xs acc x, In x (atoms_nodup xs acc) -> In x xs \/ In x acc.
Proof.
 intros xs. induction xs as [|y xs IH];intros acc x H;cbn [atoms_nodup] in H.
 - right. apply in_rev. exact H.
 - destruct (existsb (atom_eqb y) acc).
   + destruct (IH acc x H) as [Hx|Hx];[left;right;exact Hx|right;exact Hx].
   + destruct (IH (y::acc) x H) as [Hx|[<-|Hx]];[left;right;exact Hx|left;left;reflexivity|right;exact Hx].
Qed.
Lemma eliminate_in : forall P x k1 k2, terms_in P (fst (fst k1)) -> terms_in P (fst (fst k2)) ->
 terms_in P (fst (fst (eliminate x k1 k2))).
Proof.
 intros P x [[t1 l1] h1] [[t2 l2] h2] H1 H2. unfold eliminate,cons_lin. cbn [fst] in *.
 apply terms_in_sorted,terms_in_app;apply terms_in_tscale;assumption.
Qed.
Lemma fm_step_in : forall P x cs, cons_in P cs -> cons_in P (fm_step x cs).
Proof.
 intros P x cs H. unfold fm_step. cbv zeta. apply cons_in_app;[apply cons_in_filter,H|].
 destruct (find _ _) as [ke |] eqn:Ef.
 - apply find_some in Ef. destruct Ef as [Hke _]. apply filter_In in Hke.
   pose proof (H ke (proj1 Hke)) as He.
   destruct ke as [[te le] he]. intros k Hk.
   apply in_flat_map in Hk. destruct Hk as [k1 [Hk1 Hk]]. apply filter_In in Hk1.
   pose proof (H k1 (proj1 Hk1)) as H1.
   destruct k1 as [[t1 l1] h1].
   destruct (terms_eqb t1 te);[destruct Hk|destruct Hk as [<- | []]].
   exact (eliminate_in P x _ _ H1 He).
 - rewrite flat_map_budget_pair_prefix.
   intros k Hk. apply in_firstn_in in Hk.
   apply in_flat_map in Hk. destruct Hk as [k1 [Hk1 Hk]]. apply in_flat_map in Hk. destruct Hk as [k2 [Hk2 Hk]].
   apply filter_In in Hk1,Hk2.
   pose proof (H k1 (proj1 Hk1)) as H1. pose proof (H k2 (proj1 Hk2)) as H2.
   destruct k1 as [[t1 l1] h1]. destruct k2 as [[t2 l2] h2].
   destruct (terms_eqb t1 t2);[destruct Hk|destruct Hk as [<- | []]].
   exact (eliminate_in P x _ _ H1 H2).
Qed.
Lemma compact_projection_in : forall P cs, cons_in P cs -> cons_in P (compact_projection cs).
Proof.
 intros P cs H. induction cs as [|k cs IH];[intros k' []|].
 assert (Hk : terms_in P (fst (fst (reduce_cons (orient_cons k))))).
 { pose proof (H k (or_introl eq_refl)) as Hk0. destruct k as [[ts lo] hi]. cbn [fst] in Hk0.
   assert (Ho : terms_in P (fst (fst (orient_cons (ts,lo,hi))))).
   { unfold orient_cons. destruct ts as [|[x c] ts'];[exact Hk0|].
     destruct (c <? 0);[apply terms_in_tscale|];exact Hk0. }
   destruct (orient_cons (ts,lo,hi)) as [[ts1 lo1] hi1]. cbn [fst] in Ho. unfold reduce_cons.
   destruct (fold_left (fun a u => Z.gcd a (snd u)) ts1 0 <=? 1);[exact Ho|].
   cbn [fst]. intros t Ht. apply in_map_iff in Ht. destruct Ht as [u [<- Hu]]. exact (Ho u Hu). }
 assert (Hrest : cons_in P (compact_projection cs)) by (apply IH;intros k' Hk';apply H;right;exact Hk').
 cbn [compact_projection fold_right]. fold (compact_projection cs).
 destruct (reduce_cons (orient_cons k)) as [[ts lo] hi]. cbn [fst] in Hk.
 match goal with |- cons_in P (let '(_,_) := ?e in _) => destruct e as [lo' hi'] end.
 intros k' [<-|Hk'];[exact Hk|exact (cons_in_filter _ _ _ Hrest k' Hk')].
Qed.

Lemma canon_atom_local : forall F P a b x, rf_agree P a b -> P (fst x) -> canon_atom_with F a x = canon_atom_with F b x.
Proof.
 intros F P a b [r e] H Hr. cbn [fst] in Hr. destruct (H r Hr) as [Hrr _].
 destruct e;cbn [canon_atom_with];try reflexivity;rewrite Hrr;reflexivity.
Qed.
Lemma canon_terms_local : forall F P a b ts, rf_agree P a b -> terms_in P ts ->
 canon_terms_with F a ts = canon_terms_with F b ts.
Proof.
 intros F P a b ts H Hts. unfold canon_terms_with. f_equal. apply map_ext_in. intros [x c] Hin. cbn [fst snd].
 rewrite (canon_atom_local F P a b x H (Hts (x,c) Hin)). reflexivity.
Qed.
Lemma canon_cons_local : forall F P a b k, rf_agree P a b -> terms_in P (fst (fst k)) ->
 canon_cons_with F a k = canon_cons_with F b k.
Proof. intros F P a b [[ts lo] hi] H Hts. unfold canon_cons_with. rewrite (canon_terms_local F P a b ts H Hts). reflexivity. Qed.
(* The exact form of a register of the projection reads the ranges of the
   form's registers and of the constraints' registers. *)
Lemma form_eq_local : forall F P a b st x, rf_agree P a b -> P (fst x) -> cons_in P (acons st) ->
 (forall h al vr, get st (fst x) = Some (FForm h al vr) -> wd F (fst x) <= fm h -> terms_in P (ft h)) ->
 form_eq_with F a st x = form_eq_with F b st x /\ cons_in P (form_eq_with F b st x).
Proof.
 intros F P a b st [r e] H Hr Hacons Hform. cbn [fst] in *. unfold form_eq_with.
 destruct (get st r) as [[h al vr| | |]|] eqn:Hg;
   destruct e;try (split;[reflexivity|intros ? []]);
   (destruct (Z.eqb (fm h) 64 && Z.eqb (wd F r) 64 && Z.eqb (flo h) (fhi h) && negb (null (ft h))) eqn:Ec;
    [|split;[reflexivity|intros ? []]]);
   rewrite !andb_true_iff in Ec;destruct Ec as [[[Hm Hw] _] _];apply Z.eqb_eq in Hm,Hw;
   pose proof (Hform h al vr eq_refl ltac:(lia)) as Hh;
   unfold form_range_with;
   (rewrite (proj1 (H r Hr)),(form_range_in_local F P (acons st) _ _ h (atom_range_local F P a b H) Hh);
    [|intros ts lo hi Hin _;exact (Hacons _ Hin)]);
   (split;[reflexivity|]);
   (destruct (inter _ _) as [[lo hi]|];[|intros ? []]);
   (destruct (form_range_in _ _ _ _) as [[p q]|];[|intros ? []]);
   (destruct (_ && _);[|intros ? []]);
   intros ? [<-|[]];cbn [fst];apply terms_in_sorted,terms_in_app;
   [intros t [<-|[]];exact Hr|apply terms_in_tscale,Hh|intros t [<-|[]];exact Hr|apply terms_in_tscale,Hh].
Qed.
Lemma shift_links_in : forall (P : reg -> Prop) (xs : list atom), (forall x, In x xs -> P (fst x)) -> cons_in P (shift_links xs).
Proof.
 intros P xs H k Hk. unfold shift_links in Hk. apply in_flat_map in Hk. destruct Hk as [[r e] [Hx Hk]].
 pose proof (H _ Hx) as Hr. cbn [fst] in Hr.
 destruct e;cbv beta iota zeta in Hk;try destruct Hk;
   (match type of Hk with In _ (if ?b then _ else _) => destruct b;[|destruct Hk] end);
   destruct Hk as [<-|[]];cbn [fst];apply terms_in_sorted;
   intros t Ht;destruct Ht as [<-|[<-|[]]];exact Hr.
Qed.
Lemma projection_at_local : forall F P ar br f k c d, ranges_agree P ar br -> terms_in P (ft f) ->
 terms_in P (fst (fst k)) -> projection_at F ar f k c d = projection_at F br f k c d.
Proof.
 intros F P ar br f [[ts lo]hi] c d H Hf Hk. unfold projection_at. cbn [fst] in Hk.
 destruct (Z.eqb d 0);[reflexivity|].
 rewrite (plain_range_local F P ar br _ H);[reflexivity|]. cbn [ft].
 apply terms_in_filter,terms_in_tsort,terms_in_app;apply terms_in_tscale;assumption.
Qed.
Lemma projected_cons_local : forall F P ar br f k, ranges_agree P ar br -> terms_in P (ft f) ->
 terms_in P (fst (fst k)) -> projected_cons_range F ar f k = projected_cons_range F br f k.
Proof.
 intros F P ar br f [[ts lo]hi] H Hf Hk. unfold projected_cons_range. cbn zeta.
 match goal with |- fold_left _ ?ps _ = _ => generalize ps as pivots end.
 intros pivots. generalize(@None range) as acc.
 induction pivots as [|[c d] ps IH];intros acc;cbn [fold_left];[reflexivity|].
 rewrite (projection_at_local F P ar br f _ _ _ H Hf Hk). apply IH.
Qed.
Lemma project_range_plain_local : forall F P a b st extra f, rf_agree P a b ->
 terms_in P (projection_terms st extra f) ->
 (forall r h al vr, P r -> get st r = Some (FForm h al vr) -> wd F r <= fm h -> terms_in P (ft h)) ->
 project_range_plain F a st extra f = project_range_plain F b st extra f.
Proof.
 intros F P a b st extra f H Hterms Hform. unfold projection_terms in Hterms.
 assert (Hf : terms_in P (ft f)) by (intros t Ht;apply Hterms,in_or_app;left;exact Ht).
 assert (Hall : cons_in P (acons st ++ extra)).
 { intros k Hk t Ht. apply Hterms,in_or_app. right. apply in_flat_map. exists k. split;[exact Hk|].
   destruct k as [[ts lo] hi]. exact Ht. }
 assert (Hacons : cons_in P (acons st)) by (intros k Hk;apply Hall,in_or_app;left;exact Hk).
 unfold project_range_plain.
 assert(Har:ranges_agree P (fun x=>inter(atom_range_with F a x)(natural_range F x))
   (fun x=>inter(atom_range_with F b x)(natural_range F x))).
 {intros x Hx. cbv beta. rewrite(atom_range_local F P a b H x Hx). reflexivity. }
 rewrite(canon_terms_local F P a b (ft f) H Hf).
 assert(Hnear:cons_in P (near_constraints F st(acons st++extra)(ft f))).
 {intros k Hk. unfold near_constraints in Hk. apply filter_In in Hk. exact (Hall k (proj1 Hk)). }
 assert(Hcs:map(canon_cons_with F a)(near_constraints F st(acons st++extra)(ft f))=map(canon_cons_with F b)(near_constraints F st(acons st++extra)(ft f))).
 {apply map_ext_in. intros k Hk. exact (canon_cons_local F P a b k H (Hnear k Hk)). } rewrite Hcs.
 set(g:={|fm:=fm f;ft:=canon_terms_with F b(ft f);flo:=flo f;fhi:=fhi f|}).
 assert(Hg:terms_in P (ft g)) by exact (terms_in_canon F P b (ft f) Hf).
 set(cs:=map(canon_cons_with F b)(near_constraints F st(acons st++extra)(ft f))).
 assert(Hcsin:cons_in P cs).
 {intros k Hk. unfold cs in Hk. apply in_map_iff in Hk. destruct Hk as [[[ts lo] hi] [<- Hin]].
  exact (terms_in_canon F P b ts (Hnear _ Hin)). }
 set(xs:=atoms_nodup(map fst(ft g)++flat_map(fun k=>let '(ts,_,_):=k in map fst ts)cs)[]).
 assert(Hxs:forall x,In x xs -> P (fst x)).
 {intros x Hx. destruct (atoms_nodup_in _ [] x Hx) as [Hx'|[]]. apply in_app_or in Hx'. destruct Hx' as [Hx'|Hx'].
  - apply in_map_iff in Hx'. destruct Hx' as [t [<- Ht]]. exact (Hg t Ht).
  - apply in_flat_map in Hx'. destruct Hx' as [[[ts lo] hi] [Hk Hx']]. apply in_map_iff in Hx'.
    destruct Hx' as [t [<- Ht]]. exact (Hcsin _ Hk t Ht). }
 assert(Hunits:flat_map(fun x=>match inter(atom_range_with F a x)(natural_range F x)with Some(lo,hi)=>[([(x,1)],lo,hi)]|None=>[]end)xs=flat_map(fun x=>match inter(atom_range_with F b x)(natural_range F x)with Some(lo,hi)=>[([(x,1)],lo,hi)]|None=>[]end)xs).
 {apply flat_map_ext_in. intros x Hx. pose proof (Har x (Hxs x Hx)) as Hx'. cbv beta in Hx'. rewrite Hx'. reflexivity. }
 rewrite Hunits.
 assert(Hxform:forall x,In x xs -> forall h al vr, get st (fst x) = Some (FForm h al vr) -> wd F (fst x) <= fm h -> terms_in P (ft h)).
 {intros x Hx h al vr Hgx Hw. exact (Hform (fst x) h al vr (Hxs x Hx) Hgx Hw). }
 assert(Hinner:flat_map(form_eq_with F a st)xs=flat_map(form_eq_with F b st)xs).
 {apply flat_map_ext_in. intros x Hx. exact (proj1 (form_eq_local F P a b st x H (Hxs x Hx) Hacons (Hxform x Hx))). }
 assert(Heqin:cons_in P (flat_map(form_eq_with F b st)xs)).
 {intros k Hk. apply in_flat_map in Hk. destruct Hk as [x [Hx Hk]].
  exact (proj2 (form_eq_local F P a b st x H (Hxs x Hx) Hacons (Hxform x Hx)) k Hk). }
 assert(Heqs:map(canon_cons_with F a)(flat_map(form_eq_with F a st)xs)=map(canon_cons_with F b)(flat_map(form_eq_with F b st)xs)).
 {rewrite Hinner. apply map_ext_in. intros k Hk. exact (canon_cons_local F P a b k H (Heqin k Hk)). }
 rewrite Heqs.
 rewrite(plain_range_local F P _ _ g Har Hg).
 set(units:=flat_map(fun x=>match inter(atom_range_with F b x)(natural_range F x)with Some(lo,hi)=>[([(x,1)],lo,hi)]|None=>[]end)xs).
 assert(Hunin:cons_in P units).
 {intros k Hk. unfold units in Hk. apply in_flat_map in Hk. destruct Hk as [x [Hx Hk]].
  destruct (inter _ _) as [[lo hi]|];[|destruct Hk]. destruct Hk as [<-|[]]. intros t [<-|[]]. exact (Hxs x Hx). }
 set(eqs:=map(canon_cons_with F b)(flat_map(form_eq_with F b st)xs)).
 assert(Heqsin:cons_in P eqs).
 {intros k Hk. unfold eqs in Hk. apply in_map_iff in Hk. destruct Hk as [[[ts lo] hi] [<- Hin]].
  exact (terms_in_canon F P b ts (Heqin _ Hin)). }
 match goal with |- fold_left _ ?ks _ = _ => assert (Hks : cons_in P ks); [|revert Hks; generalize ks as projected] end.
 - set(kept:=projection_slice F(ft g)(cs++units++eqs++shift_links xs)).
   assert(Hkept:cons_in P kept).
   {unfold kept,projection_slice. apply cons_in_filter.
    apply cons_in_app;[exact Hcsin|apply cons_in_app;[exact Hunin|apply cons_in_app;[exact Heqsin|apply shift_links_in,Hxs]]]. }
   generalize (compact_projection_in P kept Hkept).
   match goal with |- _ -> cons_in P (fold_left _ ?l _) => generalize l end.
   intros l. generalize (compact_projection kept). induction l as [|x l IH];intros init Hinit;cbn [fold_left];[exact Hinit|].
   apply IH. apply compact_projection_in,fm_step_in,Hinit.
 - intros projected Hks. generalize(plain_range F (fun x=>inter(atom_range_with F b x)(natural_range F x)) g) as acc.
   induction projected as [|k ks IH];intros acc;cbn [fold_left];[reflexivity|].
   rewrite(projected_cons_local F P _ _ g k Har Hg (Hks k (or_introl eq_refl))).
   apply IH. intros k' Hk'. apply Hks. right. exact Hk'.
Qed.
(* The cache of a projection's ranges gives the projection of the ranges. *)
Lemma project_range_with_plain : forall F a st extra f,
 project_range_with F a st extra f = project_range_plain F a st extra f.
Proof.
 intros F a st extra f. unfold project_range_with.
 apply (project_range_plain_local F (fun _ => True));
 [intros r _;exact (cache_rf_same a _ r)|intros t _;exact I|intros r h al vr _ _ _ t _;exact I].
Qed.
Lemma project_range_local : forall F P a b st extra f, rf_agree P a b ->
 terms_in P (projection_terms st extra f) ->
 (forall r h al vr, P r -> get st r = Some (FForm h al vr) -> wd F r <= fm h -> terms_in P (ft h)) ->
 project_range_with F a st extra f = project_range_with F b st extra f.
Proof.
 intros F P a b st extra f H Hterms Hform. rewrite !project_range_with_plain.
 exact (project_range_plain_local F P a b st extra f H Hterms Hform).
Qed.
Lemma project_range_extensional : forall F a b,rf_equivalent a b->forall st extra f,
 project_range_with F a st extra f=project_range_with F b st extra f.
Proof.
 intros F a b H st extra f. apply (project_range_local F (fun _ => True));
 [intros r _;exact (H r)|intros t _;exact I|intros r h al vr _ _ _ t _;exact I].
Qed.
Lemma residue_eq_extensional : forall F a b,(forall f,a f=b f)->forall st x,
 residue_eq F a st x=residue_eq F b st x.
Proof. intros F a b H st [r e]. unfold residue_eq. destruct e;try reflexivity.
 destruct(get st r) as [[f al vr| | |]|];try reflexivity.
 destruct(Z.eqb(wd F r)64 && (0 <? fm f) && (fm f <? 64) && Z.eqb(flo f)(fhi f) && negb(null(ft f)));try reflexivity.
 destruct(residue_window st r (FForm f al vr)) as [[[lo hi]sg]|];try reflexivity.
 rewrite H. reflexivity. Qed.

(* The cached ranges of a projection give the projection of the uncached
   layers: they agree on every register that the projection reads. *)
Theorem project_range_exact : forall F st extra f,
 project_range F st extra f = project_range_with F (plain_ranges F st) st extra f.
Proof.
 intros F st extra f. unfold project_range.
 set (ix := cons_index (acons st)). set (regs := term_regs (projection_terms st extra f)).
 apply (project_range_local F (covered F st (reach_regs F ix st regs)));[exact (rft_on_agree F ix st regs)| |].
 - intros t Ht. apply reach_seeds. apply term_regs_in. exact Ht.
 - intros r h al vr Hr Hg Hw t Ht.
   assert (Hn : layer_node F st r = true).
   { unfold layer_node. rewrite Hg. apply andb_true_iff. split;[apply Z.leb_le;exact Hw|].
     destruct (ft h);[destruct Ht|reflexivity]. }
   apply (reach_closed F ix st regs r);[exact (Hr Hn)|]. unfold layer_reads. rewrite Hg.
   rewrite Z.geb_leb. apply Z.leb_le in Hw. rewrite Hw. unfold range_reads. apply in_or_app. left.
   apply term_regs_in. exact Ht.
Qed.
