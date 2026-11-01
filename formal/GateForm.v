(*
  Linear forms over register atoms, modulo a power of two.

  An atom reads one register: its value, its value read as a signed integer
  of the register's width, its value shifted right, or a bit field of its
  value. A form is a list of atoms with integer coefficients, a modulus 2^m,
  and an interval [lo, hi] for its constant: a value v satisfies the form in
  a register state when v = sum (c * atom) + e modulo 2^m for some e in
  [lo, hi]. The operations on forms (sum, scaling, shifting the constant,
  lowering the modulus) are proved to keep every value that satisfied their
  inputs.
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import ZArith.Znumtheory.
From Coq Require Import PArith.PArith.
From Coq Require Import Bool.Bool.
From Coq Require Import Lia.
From Coq Require Import Lists.List.
Import ListNotations.

Require Import OpGate.GateIsa.

Local Open Scope Z_scope.

(* ------------------------------------------------------------------ *)
(* Arithmetic.                                                         *)

(* Equal modulo 2^m. *)
Definition eqm (m a b : Z) : Prop := a mod 2 ^ m = b mod 2 ^ m.

Lemma pow_pos : forall m, 0 <= m -> 0 < 2 ^ m.
Proof. intros m Hm. apply Z.pow_pos_nonneg; lia. Qed.

Lemma eqm_refl : forall m a, eqm m a a.
Proof. unfold eqm. reflexivity. Qed.

Lemma eqm_sym : forall m a b, eqm m a b -> eqm m b a.
Proof. unfold eqm. intros. congruence. Qed.

Lemma eqm_trans : forall m a b c, eqm m a b -> eqm m b c -> eqm m a c.
Proof. unfold eqm. intros. congruence. Qed.

Lemma eqm_add : forall m a b c d, 0 <= m -> eqm m a b -> eqm m c d -> eqm m (a + c) (b + d).
Proof.
  unfold eqm. intros m a b c d Hm H1 H2.
  rewrite Z.add_mod, H1, H2, <- Z.add_mod by (pose proof (pow_pos m Hm); lia). reflexivity.
Qed.

Lemma eqm_mul : forall m a b k, 0 <= m -> eqm m a b -> eqm m (k * a) (k * b).
Proof.
  unfold eqm. intros m a b k Hm H.
  rewrite Z.mul_mod, H, <- Z.mul_mod by (pose proof (pow_pos m Hm); lia). reflexivity.
Qed.

Lemma eqm_opp : forall m a b, 0 <= m -> eqm m a b -> eqm m (- a) (- b).
Proof.
  intros m a b Hm H. replace (- a) with (-1 * a) by lia. replace (- b) with (-1 * b) by lia.
  apply eqm_mul; assumption.
Qed.

Lemma eqm_mod : forall m a, 0 <= m -> eqm m (a mod 2 ^ m) a.
Proof. unfold eqm. intros m a Hm. apply Z.mod_mod. pose proof (pow_pos m Hm). lia. Qed.

(* Equal modulo 2^m implies equal modulo 2^m' for m' <= m. *)
Lemma eqm_weaken : forall m m' a b, 0 <= m' <= m -> eqm m a b -> eqm m' a b.
Proof.
  unfold eqm. intros m m' a b Hm H.
  assert (Hd : (2 ^ m' | 2 ^ m)).
  { exists (2 ^ (m - m')). rewrite <- Z.pow_add_r by lia. f_equal. lia. }
  assert (Hp : 0 < 2 ^ m') by (apply pow_pos; lia).
  assert (Hq : 0 < 2 ^ m) by (apply pow_pos; lia).
  rewrite (Zmod_div_mod (2 ^ m') (2 ^ m) a), (Zmod_div_mod (2 ^ m') (2 ^ m) b) by assumption.
  rewrite H. reflexivity.
Qed.

(* Two values of one congruence class modulo 2^m less than 2^m apart are
   equal. *)
Lemma eqm_close : forall m a b, 0 <= m -> eqm m a b -> - 2 ^ m < a - b < 2 ^ m -> a = b.
Proof.
  intros m a b Hm0 H Hd. unfold eqm in H.
  assert (Hp : 0 < 2 ^ m) by (apply pow_pos; lia).
  assert (Hm : (a - b) mod 2 ^ m = 0) by (rewrite Zminus_mod, H, Z.sub_diag; apply Z.mod_0_l; lia).
  pose proof (Z.div_mod (a - b) (2 ^ m) ltac:(lia)) as Hq. rewrite Hm, Z.add_0_r in Hq.
  assert (Hq0 : (a - b) / 2 ^ m = 0) by nia. rewrite Hq0 in Hq. lia.
Qed.

(* Scaling by k raises the modulus by k's 2-adic valuation. *)
Lemma eqm_scale : forall m j a b k,
    0 <= m -> 0 <= j -> (2 ^ j | k) -> eqm m a b -> eqm (m + j) (k * a) (k * b).
Proof.
  unfold eqm. intros m j a b k Hm Hj [q Hq] H. subst k.
  assert (Hp : 0 < 2 ^ m) by (apply pow_pos; lia).
  assert (Hpj : 0 < 2 ^ j) by (apply pow_pos; lia).
  rewrite Z.pow_add_r by lia.
  (* k * a = q * 2^j * a and a = 2^m * (a / 2^m) + a mod 2^m *)
  rewrite (Z.div_mod a (2 ^ m)) by lia. rewrite (Z.div_mod b (2 ^ m)) by lia.
  rewrite H.
  replace (q * 2 ^ j * (2 ^ m * (a / 2 ^ m) + b mod 2 ^ m))
    with (q * 2 ^ j * (b mod 2 ^ m) + (q * (a / 2 ^ m)) * (2 ^ m * 2 ^ j)) by ring.
  replace (q * 2 ^ j * (2 ^ m * (b / 2 ^ m) + b mod 2 ^ m))
    with (q * 2 ^ j * (b mod 2 ^ m) + (q * (b / 2 ^ m)) * (2 ^ m * 2 ^ j)) by ring.
  rewrite !Z.mod_add by nia. reflexivity.
Qed.

(* 2^m, for the common widths without iterating the multiplication. *)
Definition pow2 (m : Z) : Z :=
  match m with
  | 64 => 18446744073709551616 | 63 => 9223372036854775808
  | 32 => 4294967296 | 31 => 2147483648
  | 16 => 65536 | 15 => 32768 | 8 => 256 | 7 => 128
  | _ => 2 ^ m
  end.
Lemma pow2_eq : forall m, pow2 m = 2 ^ m.
Proof. intros m. unfold pow2. repeat (match goal with |- context [match ?x with _ => _ end] => destruct x end); reflexivity. Qed.

(* The representative of c modulo 2^m in (-2^(m-1), 2^(m-1)]. *)
Definition rep (m c : Z) : Z :=
  let r := c mod pow2 m in if Z.ltb (pow2 (m - 1)) r then r - pow2 m else r.

Lemma rep_eqm : forall m c, 0 <= m -> eqm m (rep m c) c.
Proof.
  intros m c Hm. unfold rep, eqm. rewrite !pow2_eq. assert (Hp : 0 < 2 ^ m) by (apply pow_pos; lia).
  destruct (Z.ltb (2 ^ (m - 1)) (c mod 2 ^ m)).
  - replace (c mod 2 ^ m - 2 ^ m) with (c mod 2 ^ m + (-1) * 2 ^ m) by ring.
    rewrite Z.mod_add by lia. apply Z.mod_mod. lia.
  - apply Z.mod_mod. lia.
Qed.

(* The 2-adic valuation, 64 for zero. *)
Fixpoint v2_fuel (n : nat) (z : Z) : Z :=
  match n with
  | O => 0
  | S n' => if Z.eqb (z mod 2) 0 then 1 + v2_fuel n' (z / 2) else 0
  end.

Definition v2 (z : Z) : Z := if Z.eqb z 0 then 64 else v2_fuel 64 z.

Lemma v2_fuel_nonneg : forall n z, 0 <= v2_fuel n z.
Proof.
  induction n as [| n IH]; intros z; cbn [v2_fuel]; [lia |].
  destruct (Z.eqb (z mod 2) 0); [pose proof (IH (z / 2)); lia | lia].
Qed.

Lemma v2_fuel_divides : forall n z, (2 ^ v2_fuel n z | z).
Proof.
  induction n; intros z; cbn [v2_fuel]; [exists z; lia |].
  destruct (Z.eqb (z mod 2) 0) eqn:E; [| exists z; lia].
  apply Z.eqb_eq in E. destruct (IHn (z / 2)) as [q Hq].
  pose proof (v2_fuel_nonneg n (z / 2)) as Hv.
  set (v := v2_fuel n (z / 2)) in *.
  assert (Hz : z = 2 * (z / 2)) by (rewrite (Z.div_mod z 2) at 1 by lia; rewrite E; lia).
  exists q. rewrite Hz at 1. rewrite Hq. rewrite Z.pow_add_r by lia. ring.
Qed.

Lemma v2_nonneg : forall z, 0 <= v2 z.
Proof. intros z. unfold v2. destruct (Z.eqb z 0); [lia | apply v2_fuel_nonneg]. Qed.

Lemma v2_divides : forall z, (2 ^ v2 z | z).
Proof.
  intros z. unfold v2. destruct (Z.eqb z 0) eqn:E.
  - apply Z.eqb_eq in E. subst. exists 0. lia.
  - apply v2_fuel_divides.
Qed.

(* ------------------------------------------------------------------ *)
(* Atoms.                                                              *)

Inductive ext :=
| EX                      (* the value *)
| EZ                      (* the value, of a register narrower than 64 bits *)
| ES                      (* the value read as a signed integer of its width *)
| EShr (k : Z)            (* the value shifted right by k *)
| EBf (a n : Z).          (* bits [a, a + n) of the value *)

Definition atom := (reg * ext)%type.

Definition ext_key (e : ext) : Z * Z * Z :=
  match e with
  | EX => (0, 0, 0)
  | EZ => (1, 0, 0)
  | ES => (2, 0, 0)
  | EShr k => (3, k, 0)
  | EBf a n => (4, a, n)
  end.

Definition ext_eqb (e f : ext) : bool :=
  match e, f with
  | EX, EX | EZ, EZ | ES, ES => true
  | EShr k, EShr k' => Z.eqb k k'
  | EBf a n, EBf a' n' => Z.eqb a a' && Z.eqb n n'
  | _, _ => false
  end.

Lemma ext_eqb_eq : forall e f, ext_eqb e f = true <-> e = f.
Proof.
  intros e f. destruct e, f; cbn [ext_eqb];
    try (split; intros H; [discriminate H || reflexivity | discriminate H || reflexivity]).
  - rewrite Z.eqb_eq. split; [intros -> | intros H; inversion H]; reflexivity.
  - rewrite andb_true_iff, !Z.eqb_eq.
    split; [intros [-> ->] | intros H; inversion H; split]; reflexivity.
Qed.

Definition atom_eqb (x y : atom) : bool := Pos.eqb (fst x) (fst y) && ext_eqb (snd x) (snd y).

Lemma atom_eqb_eq : forall x y, atom_eqb x y = true <-> x = y.
Proof.
  intros [r e] [r' e']. unfold atom_eqb. cbn [fst snd]. rewrite andb_true_iff, Pos.eqb_eq, ext_eqb_eq.
  split; [intros [-> ->]; reflexivity | intros H; inversion H; auto].
Qed.

(* A total order on atoms, so that equal forms have equal term lists. *)
Definition key_ltb (x y : Z * Z * Z) : bool :=
  let '(a, b, c) := x in let '(a', b', c') := y in
  Z.ltb a a' || (Z.eqb a a' && (Z.ltb b b' || (Z.eqb b b' && Z.ltb c c'))).

Definition atom_ltb (x y : atom) : bool :=
  Pos.ltb (fst x) (fst y) || (Pos.eqb (fst x) (fst y) && key_ltb (ext_key (snd x)) (ext_key (snd y))).

Section Atoms.

Variable wd : reg -> nat.

Definition atom_val (rs : reg -> Z) (x : atom) : Z :=
  let (r, e) := x in
  match e with
  | EX | EZ => rs r
  | ES => sext (wd r) (rs r)
  | EShr k => rs r / 2 ^ k
  | EBf a n => (rs r / 2 ^ a) mod 2 ^ n
  end.

(* ------------------------------------------------------------------ *)
(* Term lists.                                                         *)

Definition terms := list (atom * Z).

Fixpoint tsum (rs : reg -> Z) (ts : terms) : Z :=
  match ts with
  | [] => 0
  | (x, c) :: ts' => c * atom_val rs x + tsum rs ts'
  end.

Lemma tsum_app : forall rs t u, tsum rs (t ++ u) = tsum rs t + tsum rs u.
Proof. intros rs t u. induction t as [| [x c] t IH]; cbn [tsum app]; lia. Qed.

(* Add c * x to a sorted term list. *)
Fixpoint tinsert (x : atom) (c : Z) (ts : terms) : terms :=
  match ts with
  | [] => [(x, c)]
  | (y, d) :: ts' =>
      if atom_eqb x y then (y, c + d) :: ts'
      else if atom_ltb x y then (x, c) :: ts
      else (y, d) :: tinsert x c ts'
  end.

Lemma tsum_tinsert : forall rs x c ts, tsum rs (tinsert x c ts) = c * atom_val rs x + tsum rs ts.
Proof.
  intros rs x c ts. induction ts as [| [y d] ts IH]; cbn [tinsert tsum]; [lia |].
  destruct (atom_eqb x y) eqn:E.
  - apply atom_eqb_eq in E. subst. cbn [tsum]. lia.
  - destruct (atom_ltb x y); cbn [tsum]; [lia | rewrite IH; lia].
Qed.

Fixpoint tsort (ts : terms) : terms :=
  match ts with
  | [] => []
  | (x, c) :: ts' => tinsert x c (tsort ts')
  end.

Lemma tsum_tsort : forall rs ts, tsum rs (tsort ts) = tsum rs ts.
Proof. intros rs ts. induction ts as [| [x c] ts IH]; cbn [tsort tsum]; [lia | rewrite tsum_tinsert, IH; lia]. Qed.

(* Coefficients reduced modulo 2^m, and those that vanish dropped. *)
Fixpoint treduce (m : Z) (ts : terms) : terms :=
  match ts with
  | [] => []
  | (x, c) :: ts' =>
      let c' := rep m c in
      if Z.eqb (c' mod pow2 m) 0 then treduce m ts' else (x, c') :: treduce m ts'
  end.

Lemma tsum_treduce : forall rs m ts, 0 <= m -> eqm m (tsum rs (treduce m ts)) (tsum rs ts).
Proof.
  intros rs m ts Hm. assert (Hp : 0 < 2 ^ m) by (apply pow_pos; lia).
  induction ts as [| [x c] ts IH]; cbn [treduce tsum]; [apply eqm_refl |]. rewrite pow2_eq.
  destruct (Z.eqb (rep m c mod 2 ^ m) 0) eqn:E.
  - apply Z.eqb_eq in E. pose proof (rep_eqm m c Hm) as Hr. unfold eqm in *.
    rewrite Z.add_mod by lia. rewrite (Z.mul_mod c) by lia. rewrite <- Hr, E.
    rewrite Z.mul_0_l, Z.mod_0_l, Z.add_0_l by lia. rewrite Z.mod_mod by lia.
    exact IH.
  - cbn [tsum]. apply eqm_add; [exact Hm | | exact IH].
    unfold eqm. rewrite (Z.mul_mod (rep m c)), (Z.mul_mod c) by lia.
    rewrite (rep_eqm m c Hm). reflexivity.
Qed.

Definition canon (m : Z) (ts : terms) : terms := treduce m (tsort ts).

Lemma tsum_canon : forall rs m ts, 0 <= m -> eqm m (tsum rs (canon m ts)) (tsum rs ts).
Proof.
  intros rs m ts Hm. unfold canon. eapply eqm_trans; [apply tsum_treduce; exact Hm |].
  rewrite tsum_tsort. apply eqm_refl.
Qed.

Definition tscale (k : Z) (ts : terms) : terms := map (fun '(x, c) => (x, c * k)) ts.

Lemma tsum_tscale : forall rs k ts, tsum rs (tscale k ts) = k * tsum rs ts.
Proof.
  intros rs k ts. unfold tscale. induction ts as [| [x c] ts IH]; cbn [map tsum]; [lia | rewrite IH; ring].
Qed.

(* ------------------------------------------------------------------ *)
(* Forms.                                                              *)

Record form := { fm : Z; ft : terms; flo : Z; fhi : Z }.

(* v satisfies f in register state rs. *)
Definition sat (rs : reg -> Z) (f : form) (v : Z) : Prop :=
  exists e, flo f <= e <= fhi f /\ eqm (fm f) v (tsum rs (ft f) + e).

Definition wf (f : form) : Prop := 0 <= fm f <= 64.

Definition mk (m : Z) (ts : terms) (lo hi : Z) : form :=
  let t := canon m ts in
  match t with
  | [] => {| fm := m; ft := t; flo := lo; fhi := hi |}
  | _ :: _ => if Z.eqb lo hi then {| fm := m; ft := t; flo := rep m lo; fhi := rep m lo |}
              else {| fm := m; ft := t; flo := lo; fhi := hi |}
  end.

Lemma sat_mk : forall rs m ts lo hi v e,
    0 <= m -> lo <= e <= hi -> eqm m v (tsum rs ts + e) -> sat rs (mk m ts lo hi) v.
Proof.
  intros rs m ts lo hi v e Hm He Hv. unfold mk.
  pose proof (tsum_canon rs m ts Hm) as Hc.
  destruct (canon m ts) as [| y t'] eqn:Ec.
  - exists e. cbn [fm ft flo fhi]. split; [exact He |]. eapply eqm_trans; [exact Hv |].
    apply eqm_add; [exact Hm | apply eqm_sym; exact Hc | apply eqm_refl].
  - destruct (Z.eqb lo hi) eqn:El.
    + apply Z.eqb_eq in El. subst hi. assert (e = lo) by lia. subst e.
      exists (rep m lo). cbn [fm ft flo fhi]. split; [lia |]. eapply eqm_trans; [exact Hv |].
      apply eqm_add; [exact Hm | apply eqm_sym; exact Hc |].
      apply eqm_sym, rep_eqm, Hm.
    + exists e. cbn [fm ft flo fhi]. split; [exact He |]. eapply eqm_trans; [exact Hv |].
      apply eqm_add; [exact Hm | apply eqm_sym; exact Hc | apply eqm_refl].
Qed.

(* mk allows exactly the values its arguments describe. *)
Lemma sat_mk_inv : forall rs m ts lo hi v,
    0 <= m -> sat rs (mk m ts lo hi) v -> exists e, lo <= e <= hi /\ eqm m v (tsum rs ts + e).
Proof.
  intros rs m ts lo hi v Hm [e [He Hv]]. unfold mk in *.
  pose proof (tsum_canon rs m ts Hm) as Hc.
  destruct (canon m ts) as [| y t'] eqn:Ec; cbn [fm ft flo fhi] in *.
  - exists e. split; [exact He |]. eapply eqm_trans; [exact Hv |].
    apply eqm_add; [exact Hm | exact Hc | apply eqm_refl].
  - destruct (Z.eqb lo hi) eqn:El; cbn [fm ft flo fhi] in *.
    + apply Z.eqb_eq in El. subst hi. assert (e = rep m lo) by lia. subst e.
      exists lo. split; [lia |]. eapply eqm_trans; [exact Hv |].
      apply eqm_add; [exact Hm | exact Hc | apply rep_eqm; exact Hm].
    + exists e. split; [exact He |]. eapply eqm_trans; [exact Hv |].
      apply eqm_add; [exact Hm | exact Hc | apply eqm_refl].
Qed.

Lemma mk_fm : forall m ts lo hi, fm (mk m ts lo hi) = m.
Proof.
  intros. unfold mk. destruct (canon m ts); [reflexivity |]. destruct (Z.eqb lo hi); reflexivity.
Qed.

Definition fconst (m z : Z) : form := {| fm := m; ft := []; flo := z; fhi := z |}.

Lemma sat_fconst : forall rs m z v, eqm m v z -> sat rs (fconst m z) v.
Proof. intros rs m z v H. exists z. unfold fconst. cbn [fm ft flo fhi tsum]. split; [lia | rewrite Z.add_0_l; exact H]. Qed.

Definition fatom (m : Z) (x : atom) (c : Z) : form := mk m [(x, c)] 0 0.

Lemma sat_fatom : forall rs m x c, 0 <= m -> sat rs (fatom m x c) (c * atom_val rs x).
Proof.
  intros rs m x c Hm. unfold fatom. apply (sat_mk rs m _ 0 0 _ 0 Hm); [lia |].
  cbn [tsum]. rewrite !Z.add_0_r. apply eqm_refl.
Qed.

Definition fadd (f g : form) : form :=
  mk (Z.min (fm f) (fm g)) (ft f ++ ft g) (flo f + flo g) (fhi f + fhi g).

Lemma sat_fadd : forall rs f g a b,
    wf f -> wf g -> sat rs f a -> sat rs g b -> sat rs (fadd f g) (a + b).
Proof.
  intros rs f g a b Hf Hg [e [He Ha]] [e' [He' Hb]]. unfold wf in *.
  unfold fadd. apply (sat_mk rs _ _ _ _ _ (e + e')); [lia | lia |].
  rewrite tsum_app.
  replace (tsum rs (ft f) + tsum rs (ft g) + (e + e')) with ((tsum rs (ft f) + e) + (tsum rs (ft g) + e')) by ring.
  apply eqm_add; [lia | |].
  - eapply eqm_weaken; [| exact Ha]. lia.
  - eapply eqm_weaken; [| exact Hb]. lia.
Qed.

Definition fscale (f : form) (k cap : Z) : form :=
  let m := if Z.eqb k 0 then cap else Z.min cap (fm f + v2 k) in
  mk m (tscale k (ft f)) (Z.min (flo f * k) (fhi f * k)) (Z.max (flo f * k) (fhi f * k)).

Lemma sat_fscale : forall rs f k cap a,
    wf f -> 0 <= cap -> sat rs f a -> sat rs (fscale f k cap) (k * a).
Proof.
  intros rs f k cap a Hf Hc [e [He Ha]]. unfold wf in Hf. unfold fscale.
  apply (sat_mk rs _ _ _ _ _ (e * k)).
  - destruct (Z.eqb k 0); [lia | pose proof (v2_nonneg k); lia].
  - destruct (Z.le_ge_cases 0 k).
    + split; [apply Z.min_le_iff; left | apply Z.max_le_iff; right]; nia.
    + split; [apply Z.min_le_iff; right | apply Z.max_le_iff; left]; nia.
  - rewrite tsum_tscale. replace (k * tsum rs (ft f) + e * k) with (k * (tsum rs (ft f) + e)) by ring.
    destruct (Z.eqb k 0) eqn:Ek.
    + apply Z.eqb_eq in Ek. subst k. rewrite !Z.mul_0_l. apply eqm_refl.
    + eapply eqm_weaken; [| apply (eqm_scale (fm f) (v2 k)); [lia | apply v2_nonneg | apply v2_divides | exact Ha]].
      pose proof (v2_nonneg k). lia.
Qed.

Definition fshift (f : form) (lo hi : Z) : form :=
  mk (fm f) (ft f) (flo f + lo) (fhi f + hi).

Lemma sat_fshift : forall rs f lo hi a d,
    wf f -> lo <= d <= hi -> sat rs f a -> sat rs (fshift f lo hi) (a + d).
Proof.
  intros rs f lo hi a d Hf Hd [e [He Ha]]. unfold wf in Hf. unfold fshift.
  apply (sat_mk rs _ _ _ _ _ (e + d)); [lia | lia |].
  replace (tsum rs (ft f) + (e + d)) with ((tsum rs (ft f) + e) + d) by ring.
  apply eqm_add; [lia | exact Ha | apply eqm_refl].
Qed.

Definition fat (f : form) (m : Z) : form := mk (Z.min m (fm f)) (ft f) (flo f) (fhi f).

Lemma sat_fat : forall rs f m a, wf f -> 0 <= m -> sat rs f a -> sat rs (fat f m) a.
Proof.
  intros rs f m a Hf Hm [e [He Ha]]. unfold wf in Hf. unfold fat.
  apply (sat_mk rs _ _ _ _ _ e); [lia | exact He |].
  eapply eqm_weaken; [| exact Ha]. lia.
Qed.

Lemma wf_mk : forall m ts lo hi, 0 <= m <= 64 -> wf (mk m ts lo hi).
Proof. intros. unfold wf. rewrite mk_fm. assumption. Qed.

Lemma wf_fadd : forall f g, wf f -> wf g -> wf (fadd f g).
Proof. intros f g Hf Hg. unfold wf in *. unfold fadd. rewrite mk_fm. lia. Qed.

Lemma wf_fscale : forall f k cap, wf f -> 0 <= cap <= 64 -> wf (fscale f k cap).
Proof.
  intros f k cap Hf Hc. unfold wf in *. unfold fscale. rewrite mk_fm.
  destruct (Z.eqb k 0); [lia | pose proof (v2_nonneg k); lia].
Qed.

Lemma wf_fshift : forall f lo hi, wf f -> wf (fshift f lo hi).
Proof. intros f lo hi Hf. unfold wf in *. unfold fshift. rewrite mk_fm. lia. Qed.

Lemma wf_fat : forall f m, wf f -> 0 <= m -> wf (fat f m).
Proof. intros f m Hf Hm. unfold wf in *. unfold fat. rewrite mk_fm. lia. Qed.

End Atoms.
