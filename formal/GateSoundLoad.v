(*
  Soundness of rule R3, the facts of a value loaded from an index input:
  the constraints that index_load adds hold of the registers after the load,
  because every whole element of an index input lies in the input's range
  (env_ok), and the load's address is one such element.
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

Local Open Scope Z_scope.

Strategy expand [reg_range sreg_range atom_range form_range reg_align form_align].
Strategy 1000 [rf ra DEPTH].

Lemma sext_trunc : forall n v, sext n (trunc n v) = sext n v.
Proof. intros n v. unfold sext. rewrite (trunc_small n (trunc n v) (trunc_range n v)). reflexivity. Qed.

Lemma extend_trunc : forall n sg v, extend n sg (trunc n v) = extend n sg v.
Proof.
  intros n [] v; cbn [extend]; [apply sext_trunc | apply trunc_small; apply trunc_range].
Qed.

(* An element size is the power of two of its valuation. *)
Lemma elem_pow : forall elem, elem = 1 \/ elem = 2 \/ elem = 4 \/ elem = 8 -> 2 ^ v2 elem = elem.
Proof. intros elem [-> | [-> | [-> | ->]]]; reflexivity. Qed.

Section Load.

Set Default Proof Using "All".

Variable F : lfacts.
Variable E : env.
Hypothesis HE : env_ok F E.

Lemma v_trunc : forall d bits v, Z.of_nat bits <= wd F d -> trunc (wdn F d) (trunc bits v) = trunc bits v.
Proof.
  intros d bits v H. apply trunc_small. pose proof (trunc_range bits v) as Hr. split; [lia |].
  eapply Z.lt_le_trans; [apply Hr |]. apply Z.pow_le_mono_r; [lia | exact H].
Qed.

(* A bound constraint holds when the atom's value lies at the stated
   distance from the bound form's value. *)
Lemma bound_cons_ok : forall rs x f lo hi,
    regs_ok F E rs -> bound_form F f = true ->
    lo <= atom_val (wdn F) rs x - bval F E f + flo f <= hi ->
    cons_ok F rs (bound_cons x f lo hi).
Proof.
  intros rs x f lo hi Hrs Hb Hv. unfold bound_cons. cbn [cons_ok].
  rewrite (tsum_sorted_terms F E HE 0 rs Hrs), tsum_app, tsum_tscale. cbn [tsum].
  pose proof (bound_value F E HE 0 rs Hrs f Hb) as Hbv. lia.
Qed.

Lemma add_index_cons : forall rs st atoms lo hi xl yh V,
    regs_ok F E rs -> bound_form F lo = true -> bound_form F hi = true ->
    bval F E lo <= V <= bval F E hi -> xl <= bval F E lo -> bval F E hi <= yh ->
    (forall x, In x atoms -> atom_val (wdn F) rs x = V) ->
    forall t, state_ok F t st rs ->
    state_ok F t (fold_left (fun s x => add_cons (add_cons s (bound_cons x lo (flo lo) (yh - xl + flo lo)))
                                                 (bound_cons x hi (xl - yh + flo hi) (flo hi)))
                            atoms st) rs.
Proof.
  intros rs st atoms lo hi xl yh V Hrs Hlo Hhi HV Hxl Hyh Hat t.
  revert st. induction atoms as [| x atoms IH]; intros st Hst; cbn [fold_left]; [exact Hst |].
  apply IH; [intros y Hy; apply Hat; right; exact Hy |].
  pose proof (Hat x (or_introl eq_refl)) as Hx.
  apply (add_cons_ok F E HE t rs Hrs); [apply (add_cons_ok F E HE t rs Hrs); [exact Hst |] |].
  - apply bound_cons_ok; [exact Hrs | exact Hlo | rewrite Hx; lia].
  - apply bound_cons_ok; [exact Hrs | exact Hhi | rewrite Hx; lia].
Qed.

(* After an unguarded load writes d, the constraints of index_load hold. *)
Theorem index_load_sound : forall t st st1 gen bits d a off rs rs',
    state_ok F t st rs -> regs_ok F E rs -> wf_instr F (ILoad None gen bits d a off) = true ->
    rs' = upd (wdn F) rs d (trunc bits (e_mem E t (trunc 64 (rs a + off)))) ->
    regs_ok F E rs' -> state_ok F (S t) st1 rs' ->
    state_ok F (S t) (index_load F st st1 (Z.of_nat bits) d a off) rs'.
Proof.
  intros t st st1 gen bits d a off rs rs' Hst Hrs Hwf Hrs'e Hrs' Hst1.
  unfold wf_instr in Hwf. cbn [dest guard_of] in Hwf. rewrite !andb_true_iff in Hwf.
  destruct Hwf as [[_ [[[Hbd Ha] Hb8] Hbm]] _].
  apply Z.leb_le in Hbd, Hb8. apply Z.eqb_eq in Ha, Hbm.
  unfold index_load.
  destruct (negb (index_width F (Z.of_nat bits))) eqn:Ewidth; [exact Hst1 |].
  destruct (span_of_access F st None a off (Z.of_nat bits / 8)) as [[[P sp] rest] |] eqn:Es; [| exact Hst1].
  destruct (sp_index sp) as [ix |] eqn:Eix; [| exact Hst1].
  destruct (negb (Z.eqb (Z.of_nat bits) (8 * ix_elem ix)) || (rest_align F st a off P rest <? v2 (ix_elem ix)))
    eqn:Ec; [exact Hst1 |].
  rewrite !orb_false_iff in Ec. destruct Ec as [Eb Eal].
  apply negb_false_iff, Z.eqb_eq in Eb. apply Z.ltb_ge in Eal.
  destruct (form_range F st (ix_lo ix)) as [[xl yl] |] eqn:El; [| exact Hst1].
  destruct (form_range F st (ix_hi ix)) as [[xh yh] |] eqn:Eh; [| exact Hst1].
  destruct (ix_signed ix && negb (Z.eqb (wd F d) (Z.of_nat bits)) && (xl <? 0)) eqn:Ez; [exact Hst1 |].
  (* the element the load reads *)
  assert (Hw8 : Z.of_nat bits / 8 = ix_elem ix) by (rewrite Eb, Z.mul_comm; apply Z.div_mul; lia).
  destruct (span_of_access_sound F E HE t rs Hrs st None a off (Z.of_nat bits / 8) P sp rest Hst eq_refl
              ltac:(rewrite Hw8; lia) Es) as [Erl [e [He [Hew [Haddr Hal]]]]].
  pose proof (ok_spans F E HE P sp Erl) as Hsp. unfold span_ok in Hsp. rewrite Eix in Hsp.
  rewrite !andb_true_iff in Hsp. destruct Hsp as [Hext [[[Hel Hblo] Hbhi] _]].
  assert (Helem : ix_elem ix = 1 \/ ix_elem ix = 2 \/ ix_elem ix = 4 \/ ix_elem ix = 8).
  { rewrite !orb_true_iff, !Z.eqb_eq in Hel. tauto. }
  assert (Hmod : e mod ix_elem ix = 0).
  { apply Z.mod_divide; [lia |]. rewrite <- (elem_pow _ Helem).
    pose proof (aligned_mono _ (v2 (ix_elem ix)) e Eal Hal) as Hd. unfold aligned in Hd.
    rewrite Z.max_r in Hd by apply v2_nonneg. exact Hd. }
  rewrite Hw8 in Hew.
  pose proof (ok_index F E HE P sp ix t e Erl Eix He Hmod Hew) as Hread.
  (* the loaded value, read as the input's integer type *)
  set (V := ix_read E ix P t e) in *.
  assert (Hbits : Z.to_nat (8 * ix_elem ix) = bits) by (rewrite <- Eb; apply Nat2Z.id).
  assert (Hd : rs' d = trunc bits (e_mem E t (trunc 64 (e_init E P + e)))).
  { rewrite Hrs'e. unfold upd. rewrite Pos.eqb_refl, Haddr. apply v_trunc. exact Hbd. }
  assert (HV : V = extend bits (ix_signed ix) (rs' d)).
  { unfold V, ix_read. rewrite Hbits, Hd. rewrite extend_trunc. reflexivity. }
  (* the ranges of the bounds, at the launch values *)
  destruct (bound_sat F E HE t rs Hrs (ix_lo ix) Hblo) as [_ [_ [Hlolh _]]].
  destruct (bound_sat F E HE t rs Hrs (ix_hi ix) Hbhi) as [_ [_ [Hhilh _]]].
  pose proof (form_range_sound F E HE t st rs Hst Hrs (ix_lo ix) xl yl (flo (ix_lo ix)) El ltac:(lia)) as Hrl.
  pose proof (form_range_sound F E HE t st rs Hst Hrs (ix_hi ix) xh yh (flo (ix_hi ix)) Eh ltac:(lia)) as Hrh.
  rewrite (bound_value F E HE t rs Hrs _ Hblo) in Hrl. rewrite (bound_value F E HE t rs Hrs _ Hbhi) in Hrh.
  (* the value of each atom the constraints mention *)
  pose proof (wd_range F E HE d) as Hwd.
  assert (Hatoms : forall x, In x (if ix_signed ix && Z.eqb (wd F d) (Z.of_nat bits) then [(d, ES)]
                                   else if Z.eqb (wd F d) 64 then [(d, EX)] else [(d, EX); (d, EZ)]) ->
                   atom_val (wdn F) rs' x = V).
  { assert (Htr : trunc bits (rs' d) = rs' d) by (rewrite Hd; apply trunc_small; apply trunc_range).
    (* the atoms of the zero-extended value *)
    assert (Hz : V = rs' d -> forall x, In x (if Z.eqb (wd F d) 64 then [(d, EX)] else [(d, EX); (d, EZ)]) ->
                 atom_val (wdn F) rs' x = V).
    { intros HVz x Hx. rewrite HVz. destruct (Z.eqb (wd F d) 64).
      - destruct Hx as [<- | []]. reflexivity.
      - destruct Hx as [<- | [<- | []]]; reflexivity. }
    intros x Hx. destruct (ix_signed ix) eqn:Esgn; destruct (Z.eqb (wd F d) (Z.of_nat bits)) eqn:Ew; cbn [andb] in Hx.
    - destruct Hx as [<- | []]. cbn [atom_val]. rewrite HV. cbn [extend].
      apply Z.eqb_eq in Ew. unfold wd in Ew. apply Nat2Z.inj in Ew. rewrite Ew. reflexivity.
    - (* the value is not negative, so its sign extension is itself *)
      refine (Hz _ x Hx). cbn [andb negb] in Ez. apply Z.ltb_ge in Ez.
      assert (Hnn : 0 <= V) by lia.
      rewrite HV in Hnn |- *. cbn [extend] in Hnn |- *.
      apply sext_nonneg; [lia | rewrite Hd; apply trunc_range | exact Hnn].
    - refine (Hz _ x Hx). rewrite HV. cbn [extend]. exact Htr.
    - refine (Hz _ x Hx). rewrite HV. cbn [extend]. exact Htr. }
  set (st2 := fold_left _ _ st1).
  assert (Hst2 : state_ok F (S t) st2 rs').
  { apply (add_index_cons rs' st1 _ (ix_lo ix) (ix_hi ix) xl yh V Hrs' Hblo Hbhi Hread ltac:(lia) ltac:(lia) Hatoms (S t) Hst1). }
  (* the hull of the range as d's fact *)
  set (ex := if ix_signed ix && Z.eqb (wd F d) (Z.of_nat bits) then ES else if Z.eqb (wd F d) 64 then EX else EZ).
  assert (Hval : xl <= atom_val (wdn F) rs' (d, ex) <= yh).
  { rewrite Hatoms; [lia |]. unfold ex. destruct (ix_signed ix && Z.eqb (wd F d) (Z.of_nat bits)); [left; reflexivity |].
    destruct (Z.eqb (wd F d) 64); [left; reflexivity | right; left; reflexivity]. }
  destruct (tighten_reg_sound F E HE (S t) rs' Hrs' st2 d ex xl yh Hst2 Hval) as [s [Eti Hs]].
  rewrite Eti. exact Hs.
Qed.

End Load.
