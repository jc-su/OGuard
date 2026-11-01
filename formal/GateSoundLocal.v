(*
  Soundness of the addresses of rule R8, local memory cells: the local
  address that `local_at` and `local_of_generic` name is the address the
  instruction uses, a store at a local variable's offset writes, leaves, or
  overlaps a cell as `cell_op_of` says, and a load of a cell's address and
  width reads that cell. Distinct local variables do not overlap (env_ok),
  and every cell lies inside its variable (lf_ok).
*)

From Coq Require Import ZArith.ZArith.
From Coq Require Import ZArith.Znumtheory.
From Coq Require Import PArith.PArith.
From Coq Require Import Arith.PeanoNat.
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
Require Import OpGate.GateSoundRefine.
Require Import OpGate.GateSoundAccess.

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

Lemma lookup_in : forall {A} (l : list (reg * A)) r v, lookup l r = Some v -> In (r, v) l.
Proof.
  intros A l r v. induction l as [| [x w] l IH]; cbn [lookup]; [discriminate |].
  destruct (Pos.eqb_spec x r) as [-> | _]; [intros H; injection H as ->; left; reflexivity |].
  intros H. right. exact (IH H).
Qed.

Lemma find_ext : forall {A} (f g : A -> bool) l, (forall x, In x l -> f x = g x) -> find f l = find g l.
Proof.
  intros A f g l H. induction l as [| x l IH]; [reflexivity |]. cbn [find].
  rewrite (H x (or_introl eq_refl)). destruct (g x); [reflexivity |].
  apply IH. intros y Hy. apply H. right. exact Hy.
Qed.

Lemma tget_in : forall ts x, tget ts x <> 0 -> In (x, tget ts x) ts.
Proof.
  induction ts as [| [y d] ts IH]; intros x H; cbn [tget] in *; [lia |].
  destruct (atom_eqb x y) eqn:Ex.
  - apply atom_eqb_eq in Ex. subst. left. reflexivity.
  - right. apply IH. exact H.
Qed.

(* Two frames on the call chain of one frame: one calls the other. *)
Lemma prefix_related : forall a b fr, prefix a fr = true -> prefix b fr = true -> prefix a b = true \/ prefix b a = true.
Proof.
  induction a as [| x a IH]; intros b fr Ha Hb; [left; reflexivity |].
  destruct b as [| y b]; [right; reflexivity |].
  destruct fr as [| z fr]; [discriminate |]. cbn [prefix] in *.
  apply andb_true_iff in Ha, Hb. destruct Ha as [Hx Ha]. destruct Hb as [Hy Hb].
  apply N.eqb_eq in Hx, Hy. subst. rewrite N.eqb_refl. cbn [andb]. exact (IH b fr Ha Hb).
Qed.

Section Local.

Set Default Proof Using "All".

Variable F : lfacts.
Variable E : env.
Hypothesis HE : env_ok F E.
Variable t : nat.
Variable rs : reg -> Z.
Hypothesis Hrs : regs_ok F E rs.

Notation wdn := (wdn F).
Notation init := (e_init E).

(* A cell is a register that is not fixed, of 4 or 8 bytes, inside its
   local variable. *)
Lemma cell_in : forall p u c, In (p, u, c) (lf_cells F) ->
    is_fixed F p = false /\ (wd F p = 32 \/ wd F p = 64) /\
    exists s f, lookup (lf_locals F) u = Some (s, f) /\ 0 <= c /\ c + wd F p / 8 <= s.
Proof.
  intros p u c Hin. pose proof (ok_cells F E HE _ Hin) as H. unfold cell_ok in H.
  rewrite !andb_true_iff in H. destruct H as [[Hf Hw] Hl].
  split; [apply negb_true_iff; exact Hf |].
  split; [rewrite orb_true_iff, !Z.eqb_eq in Hw; exact Hw |].
  destruct (lookup (lf_locals F) u) as [[s f] |]; [| discriminate].
  apply andb_true_iff in Hl. destruct Hl as [H1 H2]. apply Z.leb_le in H1, H2. exists s, f. auto.
Qed.

(* Two distinct live variables do not overlap. *)
Lemma live_disjoint : forall fr u s f u' s' f',
    lookup (lf_locals F) u = Some (s, f) -> lookup (lf_locals F) u' = Some (s', f') -> u <> u' ->
    live (lf_locals F) fr u = true -> live (lf_locals F) fr u' = true ->
    init u + s <= init u' \/ init u' + s' <= init u.
Proof.
  intros fr u s f u' s' f' Hl Hl' Hne Hu Hu'. unfold live in Hu, Hu'. rewrite Hl in Hu. rewrite Hl' in Hu'.
  apply lookup_in in Hl, Hl'.
  destruct (prefix_related f f' fr Hu Hu') as [Hp | Hp].
  - exact (ok_disjoint F E HE u s f u' s' f' Hl Hl' Hne Hp).
  - destruct (ok_disjoint F E HE u' s' f' u s f Hl' Hl (fun H => Hne (eq_sym H)) Hp); [right | left]; exact H.
Qed.

(* A local variable's register holds its launch value, below 2^32. *)
Lemma local_reg : forall u s, lookup (lf_locals F) u = Some s -> rs u = init u /\ 0 <= init u < 2 ^ 32.
Proof.
  intros u s H. apply lookup_in in H. destruct (ok_locals F E HE u s H) as [Hf [Hw _]].
  pose proof (proj2 Hrs u Hf) as Hu. split; [exact Hu |].
  pose proof (proj1 Hrs u) as Hr. unfold wdz in Hr. rewrite Hw, Hu in Hr. exact Hr.
Qed.

(* The form local_at reads: one local variable's register and a constant. *)
Lemma local_at_sum : forall A u c, local_at F A = Some (u, c) ->
    fm A = 64 /\ flo A = c /\ fhi A = c /\ 0 <= c <= 2 ^ 32 /\
    (exists s, lookup (lf_locals F) u = Some s) /\ tsum wdn rs (ft A) = rs u.
Proof.
  intros A u c H. unfold local_at in H.
  destruct (Z.eqb (fm A) 64 && Z.eqb (flo A) (fhi A) && (0 <=? flo A) && (flo A <=? 2 ^ 32)) eqn:Ec;
    cbn [negb] in H; [| discriminate].
  rewrite !andb_true_iff in Ec. destruct Ec as [[[Hm Hlh] H0] H1].
  apply Z.eqb_eq in Hm, Hlh. apply Z.leb_le in H0, H1.
  destruct (ft A) as [| [[u' e] k] rest] eqn:Et; [discriminate |].
  destruct e; destruct k as [| [k | k |] | k]; destruct rest; cbn in H; try discriminate.
  all: destruct (lookup (lf_locals F) u') as [s |] eqn:El; cbn [is_some] in H; [| discriminate].
  all: injection H as <- <-.
  all: split; [exact Hm |]; split; [reflexivity |]; split; [lia |]; split; [lia |].
  all: split; [exists s; exact El |]; cbn [tsum atom_val]; lia.
Qed.

(* The local address of a load or store that local_at names. *)
Lemma local_at_sound : forall st a off u c, state_ok F t st rs ->
    local_at F (addr_form F st a off) = Some (u, c) ->
    (exists s, lookup (lf_locals F) u = Some s) /\ 0 <= c <= 2 ^ 32 /\ trunc 64 (rs a + off) = init u + c.
Proof.
  intros st a off u c Hst H.
  destruct (addr_sound F E HE t rs Hrs st a off Hst) as [_ [HmA [_ HsA]]].
  unfold addr_form in H.
  destruct (local_at_sum _ u c H) as [_ [Hlo [Hhi [Hc [[s Hs] Hsum]]]]].
  destruct (local_reg u s Hs) as [Hu Hi].
  split; [exists s; exact Hs |]. split; [exact Hc |].
  destruct HsA as [e [He Hm]]. rewrite HmA, Hsum, Hu in Hm. assert (e = c) by lia. subst e.
  rewrite <- (trunc_small 64 (init u + c)) by (change (Z.of_nat 64) with 64; lia).
  apply trunc64_eqm. exact Hm.
Qed.

(* A value of a form that local_span reads is, modulo 2^64, one local
   variable's register and an offset in the span. *)
Lemma local_span_sat : forall st A width u lo hi v, state_ok F t st rs ->
    fm A = 64 -> NoDup (map fst (ft A)) -> sat wdn rs A v ->
    local_span F st A width = Some (u, lo, hi) ->
    (exists s, lookup (lf_locals F) u = Some s) /\ 0 <= lo /\ hi <= 2 ^ 32 /\
    exists e, lo <= e <= hi /\ eqm 64 v (rs u + e).
Proof.
  intros st A width u lo hi v Hst Hm Hn Hs H. unfold local_span in H.
  destruct (local_at F A) as [[u' c] |] eqn:El.
  - (* a constant offset *)
    injection H as <- <- <-.
    destruct (local_at_sum _ u' c El) as [_ [Hlo [Hhi [Hc [Hs' Hsum]]]]].
    split; [exact Hs' |]. split; [lia |]. split; [lia |].
    destruct Hs as [e [He Hv]]. rewrite Hm, Hsum in Hv. rewrite Hlo, Hhi in He.
    exists c. split; [lia |]. replace c with e by lia. exact Hv.
  - (* a variable's register and a rest whose range the state bounds *)
    rewrite Hm, Z.eqb_refl in H. cbn [negb] in H.
    match type of H with context [find ?f (ft A)] => destruct (find f (ft A)) as [[[u0 e0] c0] |] eqn:Ef end;
      [| discriminate].
    destruct (find_some _ _ Ef) as [Hin Hp]. cbv beta iota in Hp.
    rewrite !andb_true_iff in Hp. destruct Hp as [[Hc1 He0] Hl]. apply Z.eqb_eq in Hc1. subst c0.
    cbn [fst] in H.
    destruct (access_range F st (mk 64 (without (ft A) (u0, e0)) (flo A) (fhi A)) 0 (local_size F u0 - width))
      as [[lo' hi'] |] eqn:Er; [| discriminate].
    destruct ((0 <=? lo') && (hi' <=? 2 ^ 32)) eqn:Eb; [| discriminate].
    injection H as -> -> ->. apply andb_true_iff in Eb. destruct Eb as [Hb0 Hb1]. apply Z.leb_le in Hb0, Hb1.
    destruct (lookup (lf_locals F) u) as [s |] eqn:Es; cbn [is_some] in Hl; [| discriminate].
    split; [exists s; reflexivity |]. split; [exact Hb0 |]. split; [exact Hb1 |].
    pose proof (sat_strip F E HE t rs Hrs A (u, e0) 1 v Hm Hn Hin Hs) as Hsu.
    assert (Hval : atom_val wdn rs (u, e0) = rs u) by (destruct e0; try discriminate; reflexivity).
    rewrite Hval in Hsu.
    destruct (access_sat_range F E HE t rs Hrs st _ 0 (local_size F u - width) (v - 1 * rs u) lo hi Hst
                (mk_fm _ _ _ _) Hsu Er) as [X [HX HM]].
    exists X. split; [exact HX |].
    replace (v - 1 * rs u) with (v - rs u) in HM by ring. exact (eqm_base F E HE t rs Hrs v (rs u) X HM).
Qed.

(* The local address of a store that local_span names: the variable's
   address and an offset in the span. *)
Lemma local_span_sound : forall st a off width u lo hi, state_ok F t st rs ->
    local_span F st (addr_form F st a off) width = Some (u, lo, hi) ->
    (exists s, lookup (lf_locals F) u = Some s) /\ 0 <= lo /\ hi <= 2 ^ 32 /\
    exists e, lo <= e <= hi /\ trunc 64 (rs a + off) = init u + e.
Proof.
  intros st a off width u lo hi Hst H.
  destruct (addr_sound F E HE t rs Hrs st a off Hst) as [_ [HmA [HnA HsA]]].
  unfold addr_form in H.
  destruct (local_span_sat st _ width u lo hi _ Hst HmA HnA HsA H) as [[s Hs] [Hlo [Hhi [e [He Hm]]]]].
  destruct (local_reg u s Hs) as [Hu Hi].
  split; [exists s; exact Hs |]. split; [exact Hlo |]. split; [exact Hhi |].
  exists e. split; [exact He |]. rewrite Hu in Hm.
  rewrite <- (trunc_small 64 (init u + e)) by (change (Z.of_nat 64) with 64; lia).
  apply trunc64_eqm. exact Hm.
Qed.

(* The generic address of a store that local_of_generic names lies in the
   window of local memory, at the variable's address and an offset in the
   span. *)
Lemma local_of_generic_sound : forall st a off width u lo hi, state_ok F t st rs -> 0 <= width ->
    local_of_generic F st (addr_form F st a off) width = Some (u, lo, hi) ->
    (exists s, lookup (lf_locals F) u = Some s) /\ 0 <= lo /\
    init (lf_local_window F) <= trunc 64 (rs a + off) /\
    trunc 64 (rs a + off) + width <= init (lf_local_window F) + window_size /\
    exists e, lo <= e <= hi /\ trunc 64 (rs a + off) - init (lf_local_window F) = init u + e.
Proof.
  intros st a off width u lo hi Hst Hw H.
  destruct (addr_sound F E HE t rs Hrs st a off Hst) as [_ [HmA [HnA HsA]]].
  unfold local_of_generic, addr_form in H.
  set (A := fshift (normalize F st (as_form F st a EX)) off off) in *. set (w := lf_local_window F) in *.
  destruct (Z.eqb (tget (ft A) (w, EX)) 1) eqn:Et; cbn [negb] in H; [| discriminate].
  apply Z.eqb_eq in Et.
  destruct (reg_range F st w) as [[lw hw] |] eqn:Ew; [| discriminate].
  set (A' := {| fm := fm A; ft := without (ft A) (w, EX); flo := flo A; fhi := fhi A |}) in *.
  destruct (local_span F st A' width) as [[[u' lo'] hi'] |] eqn:El; [| discriminate].
  destruct (reg_range F st u') as [[lu hu] |] eqn:Eu; [| discriminate].
  destruct ((hw + window_size <? 2 ^ 64) && (hu + hi' + width <=? window_size)) eqn:Eb; [| discriminate].
  injection H as -> -> ->. apply andb_true_iff in Eb. destruct Eb as [Hb1 Hb2].
  apply Z.ltb_lt in Hb1. apply Z.leb_le in Hb2.
  (* the address less the window's base satisfies the form without it *)
  assert (Hin : In ((w, EX), 1) (ft A)) by (rewrite <- Et; apply tget_in; lia).
  pose proof (tsum_without F E HE t rs Hrs (ft A) (w, EX) 1 HnA Hin) as Hsplit. cbn [atom_val] in Hsplit.
  assert (HsA' : sat wdn rs A' (rs a + off - rs w)).
  { destruct HsA as [e [He Hm]]. exists e. unfold A'. cbn [fm ft flo fhi]. split; [exact He |].
    rewrite HmA in *. rewrite Hsplit in Hm.
    pose proof (eqm_add 64 _ _ (- rs w) (- rs w) ltac:(lia) Hm (eqm_refl 64 _)) as Hm'.
    replace (rs a + off - rs w) with (rs a + off + - rs w) by ring.
    replace (tsum wdn rs (without (ft A) (w, EX)) + e)
      with (tsum wdn rs (without (ft A) (w, EX)) + 1 * rs w + e + - rs w) by ring.
    exact Hm'. }
  assert (HmA' : fm A' = 64) by exact HmA.
  assert (HnA' : NoDup (map fst (ft A'))) by (unfold A'; cbn [ft]; exact (nodup_map_filter F E HE t rs Hrs _ _ fst _ (ft A) HnA)).
  destruct (local_span_sat st A' width u lo hi _ Hst HmA' HnA' HsA' El) as [[s Hs] [Hlo [Hhi [e [He Hm]]]]].
  destruct (local_reg u s Hs) as [Hu Hi].
  pose proof (reg_range_sound F E HE t st rs Hst Hrs w lw hw Ew) as Hrw.
  pose proof (reg_range_sound F E HE t st rs Hst Hrs u lu hu Eu) as Hru.
  assert (Hwv : rs w = init w) by exact (proj2 Hrs w (proj2 (ok_windows F E HE))).
  pose proof (proj1 Hrs w) as Hw0. pose proof (proj1 Hrs u) as Hu0.
  assert (Hx : trunc 64 (rs a + off) = rs u + rs w + e).
  { rewrite <- (trunc_small 64 (rs u + rs w + e)) by (change (Z.of_nat 64) with 64; unfold window_size in *; lia).
    apply trunc64_eqm. replace (rs u + rs w + e) with (rs w + (rs u + e)) by ring.
    exact (eqm_base F E HE t rs Hrs _ (rs w) (rs u + e) Hm). }
  rewrite Hx, Hwv, Hu. rewrite Hwv in Hrw. rewrite Hu in Hru.
  split; [exists s; exact Hs |]. split; [exact Hlo |].
  split; [unfold window_size in *; lia |]. split; [unfold window_size in *; lia |].
  exists e. split; [exact He | lia].
Qed.

(* A store of width bytes at an offset e in [lo, hi] of local variable u
   writes, leaves, or may overlap cell (p, u', c') as cell_op_of says. *)
Lemma cell_op_sound : forall fr u s f lo hi e width p u' c',
    lookup (lf_locals F) u = Some (s, f) -> 0 <= lo -> lo <= e <= hi -> 0 < width -> In (p, u', c') (lf_cells F) ->
    (cell_op_of F fr u lo hi width (p, u', c') = CPut ->
       init u' + c' = init u + e /\ Z.of_nat (wdn p) = 8 * width) /\
    (cell_op_of F fr u lo hi width (p, u', c') = CKeep ->
       (Z.eqb (init u' + c') (init u + e) && Z.eqb (Z.of_nat (wdn p)) (8 * width)) = false /\
       ((init u' + c' <? init u + e + width) && (init u + e <? init u' + c' + Z.of_nat (wdn p) / 8)) = false).
Proof.
  intros fr u s f lo hi e width p u' c' Hs Hlo He Hw Hin.
  destruct (cell_in p u' c' Hin) as [_ [Hwp [s' [f' [Hs' [Hc0 Hcs]]]]]].
  fold (wd F p). assert (H4 : 4 <= wd F p / 8) by (destruct Hwp as [-> | ->]; cbn; lia).
  unfold cell_op_of. destruct (Pos.eqb_spec u' u) as [-> | Hne].
  - destruct (Z.eqb lo hi && Z.eqb c' lo && Z.eqb (wd F p) (8 * width)) eqn:Ep.
    + rewrite !andb_true_iff in Ep. destruct Ep as [[E0 E1] E2]. apply Z.eqb_eq in E0, E1, E2. subst hi c'.
      replace e with lo by lia.
      split; [intros _; split; [reflexivity | exact E2] | discriminate].
    + split; [destruct ((hi + width <=? c') || (c' + wd F p / 8 <=? lo)); discriminate |].
      destruct ((hi + width <=? c') || (c' + wd F p / 8 <=? lo)) eqn:Ek; [| discriminate].
      intros _. apply orb_true_iff in Ek. rewrite !Z.leb_le in Ek.
      split; apply andb_false_iff; rewrite ?Z.eqb_neq, ?Z.ltb_ge; lia.
  - destruct (live (lf_locals F) fr u && live (lf_locals F) fr u' && (hi + width <=? local_size F u)) eqn:Ek;
      [| split; discriminate].
    split; [discriminate |]. intros _. rewrite !andb_true_iff in Ek. destruct Ek as [[Hu Hu'] Ek].
    unfold local_size in Ek. rewrite Hs in Ek. apply Z.leb_le in Ek.
    destruct (live_disjoint fr u s f u' s' f' Hs Hs' (fun H => Hne (eq_sym H)) Hu Hu') as [Hd | Hd];
      split; apply andb_false_iff; rewrite ?Z.eqb_neq, ?Z.ltb_ge; lia.
Qed.

(* A load of bits at offset c of live local variable u, where the first cell
   of that variable, offset, and width is p, reads cell p. *)
Lemma cell_at_sound : forall fr u s f c bits p u0 c0,
    lookup (lf_locals F) u = Some (s, f) -> 0 <= c -> live (lf_locals F) fr u = true ->
    find (fun k => let '(p, u', c') := k in Pos.eqb u' u && Z.eqb c' c && Z.eqb (wd F p) (Z.of_nat bits))
         (lf_cells F) = Some (p, u0, c0) ->
    cell_at wdn init (lf_cells F) (lf_locals F) fr (init u + c) bits = Some p.
Proof.
  intros fr u s f c bits p u0 c0 Hs Hc Hlive Hf.
  destruct (find_some _ _ Hf) as [Hin Hp]. cbv beta iota in Hp. rewrite !andb_true_iff in Hp.
  destruct Hp as [[Hu Hc'] Hb]. apply Pos.eqb_eq in Hu. apply Z.eqb_eq in Hc', Hb. subst u0 c0.
  destruct (cell_in p u c Hin) as [_ [Hwp [s0 [f0 [Hs0 [_ Hcs]]]]]]. rewrite Hs in Hs0. injection Hs0 as <- <-.
  unfold cell_at.
  rewrite (find_ext _ (fun k => let '(p, u', c') := k in Pos.eqb u' u && Z.eqb c' c && Z.eqb (wd F p) (Z.of_nat bits))).
  - rewrite Hf. reflexivity.
  - intros [[p' u'] c'] Hin'. destruct (cell_in p' u' c' Hin') as [_ [Hwp' [s' [f' [Hs' [Hc0' Hcs']]]]]].
    assert (Hwb : Nat.eqb (wdn p') bits = Z.eqb (wd F p') (Z.of_nat bits)).
    { unfold wd. destruct (Nat.eqb_spec (wdn p') bits) as [-> | Hn]; symmetry;
        [apply Z.eqb_eq; reflexivity | apply Z.eqb_neq; lia]. }
    rewrite Hwb. destruct (Pos.eqb_spec u' u) as [-> | Hne].
    + rewrite Hlive. cbn [andb]. f_equal. apply eq_true_iff_eq. rewrite !Z.eqb_eq. lia.
    + cbn [andb]. destruct (live (lf_locals F) fr u') eqn:Hu'; [cbn [andb] | reflexivity].
      destruct (Z.eqb_spec (wd F p') (Z.of_nat bits)) as [Hw' |]; [| rewrite andb_false_r; reflexivity].
      rewrite andb_true_r. apply Z.eqb_neq. intros Heq.
      assert (H4 : 4 <= wd F p / 8) by (destruct Hwp as [-> | ->]; cbn; lia).
      assert (H4' : 4 <= wd F p' / 8) by (destruct Hwp' as [-> | ->]; cbn; lia).
      destruct (live_disjoint fr u s f u' s' f' Hs Hs' (fun H => Hne (eq_sym H)) Hlive Hu') as [Hd | Hd]; lia.
Qed.

(* Every global access an instruction the check accepts makes stays in its
   region: a generic access that the check places in a window makes none. *)
Theorem access_of_sound : forall st i k addr width v,
    state_ok F t st rs -> wf_instr F i = true -> access_of F st i = true ->
    access_at (in_window F E) rs i = Some (k, addr, width, v) -> region F E t k addr width v.
Proof.
  intros st i k addr width v Hst Hwf H Hat. unfold access_at in Hat.
  unfold access_of in H. apply andb_true_iff in H. destruct H as [H _].
  destruct (holds (guard_of i) rs) eqn:Hg; cbn [negb] in Hat; [| discriminate].
  destruct i; try discriminate; cbv zeta in Hat; cbn [guard_of] in Hg; unfold region_of in H;
    unfold wf_instr in Hwf; rewrite !andb_true_iff in Hwf.
  - (* a tracked load of bits / 8 bytes *)
    destruct (gen && in_window F E (trunc 64 (rs a + off)) (Z.of_nat bits / 8)) eqn:Ew; [discriminate |].
    injection Hat as <- <- <- <-.
    destruct Hwf as [[_ [[[_ _] Hb8] _]] _]. apply Z.leb_le in Hb8.
    assert (H8 : 0 <= Z.of_nat bits / 8) by (apply Z.div_pos; lia).
    destruct (access_ok_sound F E HE t rs Hrs st g KLoad gen a off _ None Hst Hg H8 H) as [Hr | [-> Hin]];
      [exact Hr | rewrite (in_private_window F E HE t rs Hrs _ _ Hin) in Ew; discriminate].
  - (* any other access, of a positive width *)
    destruct (gen && in_window F E (trunc 64 (rs a + off)) width0) eqn:Ew; [discriminate |].
    injection Hat as <- <- <- <-.
    destruct Hwf as [[_ [_ Hw]] _]. apply Z.ltb_lt in Hw.
    assert (Hw0 : 0 <= width0) by lia.
    apply orb_true_iff in H. destruct H as [H | H].
    + destruct (access_ok_sound F E HE t rs Hrs st g k0 gen a off width0 v0 Hst Hg Hw0 H) as [Hr | [-> Hin]];
        [exact Hr | rewrite (in_private_window F E HE t rs Hrs _ _ Hin) in Ew; discriminate].
    + (* a store in a live local variable, in the window of local memory *)
      exfalso. rewrite !andb_true_iff in H. destruct H as [[Hgen _] Hk1]. subst gen.
      unfold local_store_ok in Hk1.
      destruct (local_of_generic F st (addr_form F st a off) width0) as [[[u lo] hi] |] eqn:El; [| discriminate].
      destruct (local_of_generic_sound st a off width0 u lo hi Hst Hw0 El) as [_ [_ [Hlo [Hhi _]]]].
      assert (Hin : in_window F E (trunc 64 (rs a + off)) width0 = true).
      { unfold in_window. apply existsb_exists. exists (lf_local_window F).
        split; [right; left; reflexivity |]. apply andb_true_iff. split; apply Z.leb_le; lia. }
      rewrite Hin in Ew. discriminate.
Qed.

(* An access inside a buffer lies in no window: no buffer lies in one. *)
Lemma region_apart : forall k x width v, region F E t k x width v -> 0 < width -> in_window F E x width = false.
Proof.
  intros k x width v Hr Hw. inversion Hr as [k' P sp e width' v' Erl _ He Hew _ _ _]; subst.
  pose proof (ok_init F E HE P) as HP. unfold wdz in HP.
  assert (Hwd : 1 <= wd F P <= 64) by (pose proof (ok_widths F E HE P) as H; unfold wdz in H; exact H).
  assert (HP64 : 0 <= init P < 2 ^ 64).
  { split; [lia |]. apply (Z.lt_le_trans _ (2 ^ wd F P)); [lia | apply Z.pow_le_mono_r; lia]. }
  assert (Hcase : forall w, (w = lf_shared_window F \/ w = lf_local_window F) ->
            ((init w <=? trunc 64 (init P + e)) && (trunc 64 (init P + e) + width <=? init w + window_size)) = false).
  { intros w Hwin. pose proof (ok_init F E HE w) as Hw0. unfold wdz in Hw0.
    assert (Hw64 : 0 <= init w < 2 ^ 64).
    { pose proof (ok_widths F E HE w) as H. unfold wdz in H.
      split; [lia |]. apply (Z.lt_le_trans _ (2 ^ wd F w)); [lia | apply Z.pow_le_mono_r; lia]. }
    destruct (ok_apart F E HE P sp w Erl Hwin) as [Hb | [Hb Hb']].
    - assert (Hx : trunc 64 (init P + e) = init P + e).
      { unfold trunc. change (Z.of_nat 64) with 64. apply Z.mod_small. lia. }
      rewrite Hx. apply andb_false_iff. left. apply Z.leb_gt. lia.
    - assert (Hx : trunc 64 (init P + e) = init P + e).
      { unfold trunc. change (Z.of_nat 64) with 64. apply Z.mod_small. lia. }
      rewrite Hx. apply andb_false_iff. right. apply Z.leb_gt. lia. }
  unfold in_window. cbn [existsb].
  rewrite (Hcase _ (or_introl eq_refl)), (Hcase _ (or_intror eq_refl)). reflexivity.
Qed.

(* An address below 2^32 modulo 2^64 is the same modulo 2^32. *)
Lemma trunc32_of_64 : forall v, trunc 64 v < 2 ^ 32 -> trunc 32 v = trunc 64 v.
Proof.
  intros v H. unfold trunc in *. change (Z.of_nat 64) with 64 in *. change (Z.of_nat 32) with 32.
  rewrite (Zmod_div_mod (2 ^ 32) (2 ^ 64) v) by (try lia; exists (2 ^ 32); reflexivity).
  apply Z.mod_small. pose proof (Z.mod_pos_bound v (2 ^ 64) ltac:(lia)). lia.
Qed.

(* A local variable's register and bytes are one of local_vars. *)
Lemma local_var_in : forall u s f, lookup (lf_locals F) u = Some (s, f) -> In (u, fconst 64 s) (local_vars F).
Proof.
  intros u s f H. apply lookup_in in H. unfold local_vars. apply in_map_iff.
  exists (u, (s, f)). split; [reflexivity | exact H].
Qed.

(* Every access to shared or local memory that an instruction the check
   accepts makes lies inside a variable of that memory: at an address of
   shared or local memory, or at a generic address in a window. *)
Theorem private_of_sound : forall st i p,
    state_ok F t st rs -> wf_instr F i = true -> access_of F st i = true ->
    private_access (in_window F E) rs i = Some p -> private_in F E p.
Proof.
  intros st i p Hst Hwf H Hp. unfold private_access in Hp.
  unfold access_of in H. apply andb_true_iff in H. destruct H as [H _].
  destruct (holds (guard_of i) rs) eqn:Hg; cbn [negb] in Hp; [| discriminate].
  destruct i; try discriminate; cbv zeta in Hp; cbn [guard_of] in Hg; unfold region_of in H;
    unfold wf_instr in Hwf; rewrite !andb_true_iff in Hwf.
  - (* a generic load *)
    destruct (gen && in_window F E (trunc 64 (rs a + off)) (Z.of_nat bits / 8)) eqn:Ew; [| discriminate].
    injection Hp as <-. cbn [private_in].
    destruct Hwf as [[_ [[[_ _] Hb8] _]] _]. apply Z.leb_le in Hb8.
    assert (H8 : 0 < Z.of_nat bits / 8) by (apply Z.div_str_pos; lia).
    destruct (access_ok_sound F E HE t rs Hrs st g KLoad gen a off _ None Hst Hg (Z.lt_le_incl _ _ H8) H)
      as [Hr | [_ Hin]]; [| exact Hin].
    exfalso. apply andb_true_iff in Ew. rewrite (region_apart _ _ _ _ Hr H8) in Ew. destruct Ew; discriminate.
  - (* a generic store or atomic *)
    destruct (gen && in_window F E (trunc 64 (rs a + off)) width) eqn:Ew; [| discriminate].
    injection Hp as <-. cbn [private_in].
    destruct Hwf as [[_ [_ Hw]] _]. apply Z.ltb_lt in Hw.
    apply orb_true_iff in H. destruct H as [H | H].
    + destruct (access_ok_sound F E HE t rs Hrs st g k gen a off width v Hst Hg (Z.lt_le_incl _ _ Hw) H)
        as [Hr | [_ Hin]]; [| exact Hin].
      exfalso. apply andb_true_iff in Ew. rewrite (region_apart _ _ _ _ Hr Hw) in Ew. destruct Ew; discriminate.
    + (* in a live local variable, in the window of local memory *)
      rewrite !andb_true_iff in H. destruct H as [_ Hk1]. unfold local_store_ok in Hk1.
      destruct (local_of_generic F st (addr_form F st a off) width) as [[[u lo] hi] |] eqn:El; [| discriminate].
      destruct (local_of_generic_sound st a off width u lo hi Hst (Z.lt_le_incl _ _ Hw) El)
        as [[[s f] Hs] [Hlo [_ [_ [e [He Hx]]]]]].
      rewrite !andb_true_iff in Hk1. destruct Hk1 as [_ Hsz]. unfold local_size in Hsz. rewrite Hs in Hsz.
      apply Z.leb_le in Hsz.
      destruct (ok_locals F E HE u (s, f) (lookup_in _ _ _ Hs)) as [_ [_ [Hu0 Hub]]]. cbn [fst] in Hub.
      right. exists u, (fconst 64 s). split; [exact (local_var_in u s f Hs) |]. change (0 <= init u /\ init u + s <= window_size /\ init u <= trunc 64 (rs a + off) - init (lf_local_window F) /\ trunc 64 (rs a + off) - init (lf_local_window F) + width <= init u + s). lia.
  - (* a load of shared memory *)
    injection Hp as <-. cbn [private_in]. unfold private_ok in H.
    destruct (refine_guard_sound F E HE t rs Hrs st g Hst Hg) as [sg [Esg Hsg]]. rewrite Esg in H.
    pose proof (Z.div_pos (Z.of_nat bits) 8 (Nat2Z.is_nonneg bits) ltac:(lia)) as Hw8.
    destruct (var_of_sound F E HE t rs Hrs _ None sg a off _ Hsg Hw8 H)
      as [u [bytes [X [Hin [Hu0 [Hub [HX0 [HXw [Hx _]]]]]]]]].
    cbn [base_value] in Hx.
    assert (Hlt : trunc 64 (rs a + off) < 2 ^ 32) by (rewrite Hx; unfold window_size in Hub; lia).
    rewrite (trunc32_of_64 _ Hlt), Hx.
    exists u, bytes. split; [exact Hin |]. lia.
  - (* a load of local memory *)
    injection Hp as <-. cbn [private_in]. unfold private_ok in H.
    destruct (refine_guard_sound F E HE t rs Hrs st g Hst Hg) as [sg [Esg Hsg]]. rewrite Esg in H.
    destruct (var_of_sound F E HE t rs Hrs _ None sg a off _ Hsg (Z.div_pos (Z.of_nat bits) 8 (Nat2Z.is_nonneg bits) ltac:(lia)) H)
      as [u [bytes [X [Hin [Hu0 [Hub [HX0 [HXw [Hx _]]]]]]]]].
    cbn [base_value] in Hx. rewrite Hx. exists u, bytes. split; [exact Hin |]. lia.
  - (* a store to local memory, inside a live local variable *)
    injection Hp as <-. cbn [private_in]. unfold local_store_ok in H.
    destruct (local_span F st (addr_form F st a off) width) as [[[u lo] hi] |] eqn:El; [| discriminate].
    destruct (local_span_sound st a off width u lo hi Hst El) as [[[s f] Hs] [Hlo [_ [e [He Hx]]]]].
    rewrite !andb_true_iff in H. destruct H as [_ Hsz]. unfold local_size in Hsz. rewrite Hs in Hsz.
    apply Z.leb_le in Hsz.
    destruct (ok_locals F E HE u (s, f) (lookup_in _ _ _ Hs)) as [_ [_ [Hu0 Hub]]]. cbn [fst] in Hub.
    exists u, (fconst 64 s). split; [exact (local_var_in u s f Hs) |]. change (0 <= init u /\ init u + s <= window_size /\ init u <= trunc 64 (rs a + off) /\ trunc 64 (rs a + off) + width <= init u + s). rewrite Hx. lia.
  - (* an access to shared memory *)
    injection Hp as <-. cbn [private_in]. unfold private_ok in H.
    destruct Hwf as [[_ Hw0] _]. apply Z.ltb_lt in Hw0.
    destruct (refine_guard_sound F E HE t rs Hrs st g Hst Hg) as [sg [Esg Hsg]]. rewrite Esg in H.
    destruct (var_of_sound F E HE t rs Hrs _ None sg a off _ Hsg (Z.lt_le_incl _ _ Hw0) H)
      as [u [bytes [X [Hin [Hu0 [Hub [HX0 [HXw [Hx _]]]]]]]]].
    cbn [base_value] in Hx.
    assert (Hlt : trunc 64 (rs a + off) < 2 ^ 32) by (rewrite Hx; unfold window_size in Hub; lia).
    rewrite (trunc32_of_64 _ Hlt), Hx.
    exists u, bytes. split; [exact Hin |]. lia.
Qed.

(* An address the check aligns keeps its alignment modulo 2^n. *)
Lemma aligned_trunc : forall n st a off w,
    state_ok F t st rs -> (32 <= n)%nat -> need w <= addr_align F st a off ->
    aligned (need w) (trunc n (rs a + off)).
Proof.
  intros n st a off w Hst Hn H.
  assert (Hw : need w <= 5) by (unfold need; lia).
  apply (aligned_eqm (Z.of_nat n) _ _ (rs a + off)); [lia | lia | apply trunc_eqm |].
  apply (addr_align_sound F E HE t rs Hrs st a off _ Hst H). lia.
Qed.

(* Every memory access an instruction the check accepts makes, in any state
   space, is aligned to what its width needs. *)
Theorem aligned_of_sound : forall st i addr width,
    state_ok F t st rs -> access_of F st i = true ->
    mem_access rs i = Some (addr, width) -> aligned (need width) addr.
Proof.
  intros st i addr width Hst H Hm. unfold access_of in H. apply andb_true_iff in H. destruct H as [_ H].
  unfold mem_access in Hm.
  destruct (holds (guard_of i) rs); cbn [negb] in Hm; [| discriminate].
  destruct i; try discriminate; unfold aligned_ok in H; apply Z.leb_le in H; injection Hm as <- <-;
    apply (aligned_trunc _ st a off); first [exact Hst | lia | exact H].
Qed.

End Local.
