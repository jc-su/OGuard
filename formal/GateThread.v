(*
  The checker's theorem for one thread: if the executable check accepts a
  kernel's invariants, then in every run of a thread that starts from
  launch facts the runtime establishes, every global memory access stays in
  its region. `check_block_sound` extends it to every thread of every
  block.

  `launch_ok` lists the facts the runtime establishes at launch. The facts
  that are properties of the declared launch facts alone, `lf_ok`, the
  check decides itself.
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
Require Import OpGate.GateSoundRefine.
Require Import OpGate.GateSoundJoin.
Require Import OpGate.GateSoundAccess.
Require Import OpGate.GateSoundLocal.
Require Import OpGate.GateSoundTransfer.

Local Open Scope Z_scope.

Section Address.

Set Default Proof Using "All".

Variable F : lfacts.
Variable prog : program.
Variable E : env.

(* ------------------------------------------------------------------ *)
(* Launch facts.                                                       *)

(* The facts the runtime establishes at launch. *)
Record launch_ok : Prop := {
  (* registers hold values of their width; fixed registers lie in their
     declared bounds and are multiples of their declared alignments *)
  lk_init : forall r, 0 <= e_init E r < 2 ^ wdz F r;
  lk_bounds : forall r lo hi, is_fixed F r = true -> lookup (lf_bounds F) r = Some (lo, hi) ->
              lo <= e_init E r <= hi;
  lk_aligns : forall r k, is_fixed F r = true -> lookup (lf_aligns F) r = Some k ->
              (2 ^ Z.max 0 k | e_init E r);
  (* every whole element of an index input lies in its range at every step,
     because no thread writes an index input during the launch *)
  lk_index : forall P sp ix t e, role_of F P = Some sp -> sp_index sp = Some ix ->
             0 <= e -> e mod ix_elem ix = 0 -> e + ix_elem ix <= bval F E (sp_ext sp) ->
             bval F E (ix_lo ix) <= ix_read E ix P t e <= bval F E (ix_hi ix);
  (* distinct local variables of one frame, or of a frame and a frame it
     calls, do not overlap: the compiler lays out a function's variables in
     its frame, and a call's frame below its caller's *)
  lk_disjoint : forall u1 s1 f1 u2 s2 f2, In (u1, (s1, f1)) (lf_locals F) -> In (u2, (s2, f2)) (lf_locals F) ->
                u1 <> u2 -> prefix f1 f2 = true ->
                e_init E u1 + s1 <= e_init E u2 \/ e_init E u2 + s2 <= e_init E u1;
  (* no buffer lies in the window of shared or local memory, and none wraps
     past 2^64: the driver maps no allocation there *)
  lk_apart : forall P sp w, role_of F P = Some sp -> (w = lf_shared_window F \/ w = lf_local_window F) ->
             e_init E P + bval F E (sp_ext sp) <= e_init E w \/
             (e_init E w + window_size <= e_init E P /\ e_init E P + bval F E (sp_ext sp) <= 2 ^ 64);
  lk_derived : Forall (cons_ok F (e_init E)) (lf_derived F)
}.

(* The declared launch facts that the check decides, with the runtime's,
   give every fact the soundness proofs assume. *)
Theorem env_of_launch : lf_ok F = true -> launch_ok -> env_ok F E.
Proof.
  intros Hlf Hk. unfold lf_ok in Hlf. rewrite !andb_true_iff in Hlf.
  destruct Hlf as [[[[[Hw Hro] Hsw] Hlw] Hloc] Hcell].
  rewrite forallb_forall in Hw, Hro, Hloc, Hcell.
  refine {| ok_init := lk_init Hk; ok_bounds := lk_bounds Hk; ok_aligns := lk_aligns Hk;
            ok_index := lk_index Hk; ok_disjoint := lk_disjoint Hk; ok_cells := Hcell;
            ok_apart := lk_apart Hk |}.
  - intros r. unfold wdz, wd, wdn. destruct (PM.find r (lf_widths F)) as [n |] eqn:Ef; [| lia].
    apply PM.elements_correct in Ef. specialize (Hw _ Ef). cbn [snd] in Hw.
    apply andb_true_iff in Hw. destruct Hw as [H1 H2]. apply Nat.leb_le in H1, H2. lia.
  - intros P sp H. apply lookup_in in H. specialize (Hro _ H). cbn [fst snd] in Hro.
    apply andb_true_iff in Hro. exact (proj1 Hro).
  - intros P sp H. apply lookup_in in H. specialize (Hro _ H). cbn [fst snd] in Hro.
    apply andb_true_iff in Hro. exact (proj2 Hro).
  - split; [exact Hsw | exact Hlw].
  - intros u s H. specialize (Hloc _ H). cbn [fst snd] in Hloc. rewrite !andb_true_iff in Hloc.
    destruct Hloc as [[H1 H2] H3].
    destruct (lookup (lf_bounds F) u) as [[lo hi] |] eqn:El; [| discriminate].
    rewrite andb_true_iff in H3. destruct H3 as [H3 H4]. apply Z.leb_le in H3, H4.
    pose proof (lk_bounds Hk u lo hi H1 El) as Hb.
    split; [exact H1 |]. split; [apply Z.eqb_eq; exact H2 |]. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* Blocks.                                                             *)

(* A plan is its block's body, with the registers each trim keeps. *)
Lemma plan_from_body : forall left keep out body,
    map fst (snd (plan_from left keep out body)) = body.
Proof.
  intros left keep out body. revert left. induction body as [| ins body IH]; intros left; [reflexivity |].
  cbn [plan_from]. specialize (IH (match left with S (S l) => S l | _ => TRIM_EVERY end)).
  destruct (plan_from (match left with S (S l) => S l | _ => TRIM_EVERY end) keep out body)
    as [after plan].
  cbn [snd map fst] in *. rewrite IH. reflexivity.
Qed.

Lemma case_plan_body : forall inv b, map fst (case_plan F prog inv b) = block_body prog b.
Proof. intros inv b. unfold case_plan, block_plan. apply plan_from_body. Qed.

Lemma plan_nth : forall (plan : walk_plan) body i ins, map fst plan = body ->
    nth_error body i = Some ins -> exists keep, nth_error plan i = Some (ins, keep).
Proof.
  intros plan body i ins Hp H. rewrite <- Hp, nth_error_map in H.
  destruct (nth_error plan i) as [[x keep] |]; cbn [option_map fst] in H; [| discriminate].
  injection H as ->. exists keep. reflexivity.
Qed.

(* The walk without trims is the walk of the plan that keeps every fact. *)
Lemma untrimmed_body : forall body : list instr, map fst (map (fun ins => (ins, @None (PM.t unit))) body) = body.
Proof. intros body. rewrite map_map. apply map_id. Qed.

Lemma untrimmed_walk : forall st (body : list instr) i,
    plan_walk F st (firstn i (map (fun ins => (ins, @None (PM.t unit))) body)) = walk F st (firstn i body).
Proof.
  intros st body i. revert st body. induction i as [| i IH]; intros st [| ins body]; try reflexivity.
  cbn [firstn map plan_walk walk trim_after]. apply IH.
Qed.

Lemma plan_walk_bottom : forall st plan, is_bottom st = true -> is_bottom (plan_walk F st plan) = true.
Proof.
  intros st plan. revert st. induction plan as [| [ins keep] plan IH]; intros st Hb; [exact Hb |].
  cbn [plan_walk]. apply IH. apply trim_after_bottom. unfold transfer at 1. rewrite Hb. exact Hb.
Qed.

(* A thread is never in a block whose invariant is bottom. *)
Lemma not_bottom : forall t st plan rs, state_ok F t (plan_walk F st plan) rs -> is_bottom st = false.
Proof.
  intros t st plan rs H. destruct (is_bottom st) eqn:Eb; [| reflexivity].
  exfalso. exact (bottom_not_ok F t _ rs (plan_walk_bottom st plan Eb) H).
Qed.

Lemma plan_walk_step : forall st plan i ins keep,
    nth_error plan i = Some (ins, keep) ->
    plan_walk F st (firstn (S i) plan) = trim_after F keep (transfer F (plan_walk F st (firstn i plan)) ins).
Proof.
  intros st plan. revert st. induction plan as [| [x k] plan IH]; intros st i ins keep H.
  - destruct i; discriminate.
  - destruct i as [| i]; cbn [nth_error] in H.
    + injection H as -> ->. reflexivity.
    + cbn [firstn plan_walk]. exact (IH _ i ins keep H).
Qed.

Lemma checked_walk_out : forall st plan out, checked_walk F st plan = Some out -> out = plan_walk F st plan.
Proof.
  intros st plan. revert st. induction plan as [| [ins keep] plan IH]; intros st out H.
  - injection H as <-. reflexivity.
  - cbn [checked_walk] in H. cbn [plan_walk].
    destruct (wf_instr F ins); [| discriminate]. destruct (access_of F st ins); [| discriminate].
    exact (IH _ _ H).
Qed.

Lemma checked_walk_at : forall st plan out i ins keep,
    checked_walk F st plan = Some out -> nth_error plan i = Some (ins, keep) ->
    wf_instr F ins = true /\ access_of F (plan_walk F st (firstn i plan)) ins = true.
Proof.
  intros st plan. revert st. induction plan as [| [x k] plan IH]; intros st out i ins keep Hw H.
  - destruct i; discriminate.
  - cbn [checked_walk] in Hw.
    destruct (wf_instr F x) eqn:Hwf; [| discriminate].
    destruct (access_of F st x) eqn:Hacc; [| discriminate].
    destruct i as [| i]; cbn [nth_error] in H.
    + injection H as -> ->. split; assumption.
    + cbn [firstn plan_walk]. exact (IH _ _ i ins keep Hw H).
Qed.

Lemma nth_removelast : forall {A} (l : list A) i x,
    nth_error l i = Some x -> (S i < length l)%nat -> In x (removelast l).
Proof.
  intros A l. induction l as [| y l IH]; intros i x H Hl; [destruct i; discriminate |].
  destruct l as [| z l]; cbn [length] in Hl; [lia |].
  destruct i as [| i]; cbn [nth_error] in H.
  - injection H as ->. cbn [removelast]. left. reflexivity.
  - change (removelast (y :: z :: l)) with (y :: removelast (z :: l)). right.
    apply (IH i x H). cbn [length] in *. lia.
Qed.

Lemma last_of_nth : forall l i x, nth_error l i = Some x -> S i = length l -> last_instr l = Some x.
Proof.
  intros l i x H Hl. unfold last_instr.
  assert (Hsplit : l = firstn i l ++ [x]).
  { rewrite <- (firstn_skipn i l) at 1. f_equal.
    pose proof (nth_error_split l i H) as [l1 [l2 [Hl' Hlen]]].
    rewrite Hl' in *. rewrite skipn_app. rewrite app_length in Hl. cbn [length] in Hl.
    replace (i - length l1)%nat with 0%nat by lia. rewrite skipn_all2 by lia. cbn [app skipn].
    destruct l2; [reflexivity | cbn [length] in Hl; lia]. }
  rewrite Hsplit, rev_app_distr. reflexivity.
Qed.

Lemma structured_block : forall b,
    structured prog = true -> (b < length prog)%nat ->
    (0 < block_len prog b)%nat /\
    (forall i ins, nth_error (block_body prog b) i = Some ins -> is_jump ins = true ->
                   S i = length (block_body prog b)).
Proof.
  intros b Hs Hb. unfold structured in Hs. rewrite forallb_forall in Hs.
  unfold block_len, block_body. destruct (nth_error prog b) as [body |] eqn:Eb.
  2: { apply nth_error_None in Eb. lia. }
  specialize (Hs body (nth_error_In prog b Eb)). apply andb_true_iff in Hs. destruct Hs as [Hne Hj].
  apply negb_true_iff, Nat.eqb_neq in Hne. split; [lia |].
  intros i ins Hi Hjump. rewrite forallb_forall in Hj.
  pose proof (nth_error_Some body i) as Hlt. assert (i < length body)%nat by (apply Hlt; rewrite Hi; discriminate).
  destruct (Nat.eq_dec (S i) (length body)) as [| Hne']; [assumption |].
  pose proof (Hj ins (nth_removelast body i ins Hi ltac:(lia))) as Hn. rewrite Hjump in Hn. discriminate.
Qed.

Lemma fetch_block : forall b i ins, fetch prog b i = Some ins -> (b < length prog)%nat /\ (i < block_len prog b)%nat.
Proof.
  intros b i ins H. unfold fetch, block_len, block_body in *.
  destruct (nth_error prog b) as [body |] eqn:Eb; [| destruct i; discriminate].
  split; [apply nth_error_Some; rewrite Eb; discriminate |].
  apply nth_error_Some. rewrite H. discriminate.
Qed.

(* ------------------------------------------------------------------ *)
(* Runs.                                                               *)

(* Where a step goes. *)
Lemma step_pos : forall t s s' ins,
    fetch prog (blk s) (idx s) = Some ins -> step prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F) t s = Some s' ->
    ((blk s', idx s') = next_pos prog (blk s) (idx s) /\
     (forall g tg, ins = IBra g tg -> holds g (regs s) = false) /\
     (forall g, ins = IExit g -> holds g (regs s) = false)) \/
    (exists g tg, ins = IBra g tg /\ holds g (regs s) = true /\ (blk s', idx s') = (tg, 0%nat)) \/
    (exists g, ins = IExit g /\ holds g (regs s) = true /\ (blk s', idx s') = (length prog, 0%nat)).
Proof.
  intros t s s' ins Hf H. unfold step in H. rewrite Hf in H. injection H as <-.
  destruct (holds (guard_of ins) (regs s)) eqn:Hg; cbn [negb].
  - destruct ins; cbn [guard_of] in Hg;
      try (left; split; [symmetry; apply surjective_pairing | split; intros; discriminate]).
    + right. left. eexists _, _. split; [reflexivity | split; [exact Hg | reflexivity]].
    + right. right. eexists _. split; [reflexivity | split; [exact Hg | reflexivity]].
  - left. split; [symmetry; apply surjective_pairing |].
    split; [intros g tg -> | intros g ->]; cbn [guard_of] in Hg; exact Hg.
Qed.

Lemma jump_regs : forall t s s' ins,
    fetch prog (blk s) (idx s) = Some ins -> is_jump ins = true ->
    step prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F) t s = Some s' -> regs s' = regs s.
Proof.
  intros t s s' ins Hf Hj H. unfold step in H. rewrite Hf in H. injection H as <-.
  destruct (holds (guard_of ins) (regs s)); cbn [negb]; [| reflexivity].
  destruct ins; cbn [is_jump] in Hj; try discriminate; reflexivity.
Qed.

(* The state a thread is in at step t agrees with a case of the checked
   invariant of its block, advanced to its position by the walk of every
   plan of the block: with the trims of the check, or without trims. *)
Definition inv_ok (inv : list (list astate)) (t : nat) (s : state) : Prop :=
 regs_ok F E (regs s) /\
 ((blk s < length prog)%nat ->
  (idx s < block_len prog (blk s))%nat /\
  exists st, In st (cases_at inv (blk s)) /\
   forall plan, map fst plan = block_body prog (blk s) ->
   state_ok F t (plan_walk F st (firstn (idx s) plan)) (regs s)).

Lemma inv_walk : forall st b i t rs,
    (forall plan, map fst plan = block_body prog b -> state_ok F t (plan_walk F st (firstn i plan)) rs) ->
    state_ok F t (walk F st (firstn i (block_body prog b))) rs.
Proof.
  intros st b i t rs H. rewrite <- untrimmed_walk. apply H. apply untrimmed_body.
Qed.

Lemma check_facts_of_check_cases : forall inv,
 check_cases F prog inv = true -> check_case_facts F prog inv = true.
Proof.
 intros inv H. unfold check_cases, check_case_facts in *. cbv zeta in *. rewrite !andb_true_iff in *.
 destruct H as [H0 Hb]. split; [exact H0|]. rewrite forallb_forall in *.
 intros b Hin. specialize (Hb b Hin). rewrite forallb_forall in *.
 intros st Hst. specialize (Hb st Hst). unfold case_block_ok in Hb. unfold case_facts_ok.
 destruct (is_bottom st); [rewrite Hb; reflexivity|].
 destruct (checked_walk F st (case_plan F prog inv b)) as [out |] eqn:Hw; [| discriminate].
 rewrite <- (checked_walk_out _ _ _ Hw), Hb, orb_true_r, andb_true_r.
 rewrite forallb_forall. intros ins Hins. destruct (In_nth_error _ _ Hins) as [i Hi].
 destruct (plan_nth _ _ i ins (case_plan_body inv b) Hi) as [keep Hk].
 exact (proj1 (checked_walk_at _ _ _ i ins keep Hw Hk)).
Qed.

Lemma run_inv : forall inv,
    check_case_facts F prog inv = true -> launch_ok ->
    forall t s, run' F prog E t = Some s -> inv_ok inv t s.
Proof.
  intros inv Hc Hk. pose proof Hc as Hc'. unfold check_case_facts in Hc'. cbv zeta in Hc'.
  rewrite !andb_true_iff in Hc'.
  destruct Hc' as [[[Hlf Hstr] Hentry] Hblocks].
  apply existsb_exists in Hentry. destruct Hentry as [entry [Hentry He]].
  unfold entry_case in He. apply andb_true_iff in He. destruct He as [Hemp Hnil].
  pose proof (env_of_launch Hlf Hk) as HE.
  rewrite forallb_forall in Hblocks.
  assert (Hbok : forall b st, (b < length prog)%nat -> In st (cases_at inv b) ->
    case_facts_ok F prog inv (case_plan F prog inv b) b st = true).
  { intros b st Hb Hin. specialize (Hblocks b ltac:(apply in_seq; lia)).
    rewrite forallb_forall in Hblocks. exact (Hblocks st Hin). }
  induction t as [| t IH]; intros s Hrun.
  - unfold run' in Hrun. cbn [run] in Hrun. injection Hrun as <-. split.
    + split; [exact (lk_init Hk) | intros r _; reflexivity].
    + intros Hb. cbn [blk idx]. destruct (structured_block 0 Hstr Hb) as [Hlen _]. split; [exact Hlen |].
      exists entry. split; [exact Hentry|]. intros plan _. cbn [firstn plan_walk regs]. split.
      * intros r x Hx. unfold get in Hx. apply PM.is_empty_2 in Hemp. exfalso. exact (Hemp r x (PM.find_2 _ _ Hx)).
      * apply Forall_forall. intros k Hin.
        rewrite forallb_forall in Hnil. specialize (Hnil k Hin).
        apply existsb_exists in Hnil. destruct Hnil as [k0 [Hk0 Heq]].
        destruct k as [[ts lo] hi], k0 as [[us l] h].
        unfold constraint_eqb in Heq. rewrite !andb_true_iff in Heq.
        destruct Heq as [[Ht Hl] Hh]. apply terms_eqb_eq in Ht.
        apply Z.eqb_eq in Hl, Hh. subst us l h.
        pose proof (lk_derived Hk) as Hd. rewrite Forall_forall in Hd.
        exact (Hd _ Hk0).
  - unfold run' in Hrun, IH. cbn [run] in Hrun.
    destruct (run prog (wdn F) (e_mem E) (e_arb E) (e_shm E) (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F) t)
      as [s0 |] eqn:E0; [| discriminate].
    destruct (fetch prog (blk s0) (idx s0)) as [ins |] eqn:Ef.
    2: { unfold step in Hrun. rewrite Ef in Hrun. discriminate. }
    destruct (IH s0 eq_refl) as [Hrs0 Hin0].
    destruct (fetch_block _ _ _ Ef) as [Hb Hi].
    destruct (Hin0 Hb) as [_ [entry0 [Hentry0 Hall0]]].
    set (b := blk s0) in *. set (i := idx s0) in *. set (body := block_body prog b) in *.
    set (plan := case_plan F prog inv b) in *.
    assert (Hplan : map fst plan = body) by exact (case_plan_body inv b).
    pose proof (Hall0 plan Hplan) as Hst0.
    pose proof (Hbok b entry0 Hb Hentry0) as Hblk. unfold case_facts_ok in Hblk. apply andb_true_iff in Hblk.
    destruct Hblk as [Hwfs Hedges].
    rewrite (not_bottom t entry0 (firstn i plan) (regs s0) Hst0) in Hedges.
    cbn [orb] in Hedges. unfold case_edges_ok in Hedges. fold body in Hwfs, Hedges.
    assert (Hnth : nth_error body i = Some ins) by exact Ef.
    assert (Hwf : wf_instr F ins = true)
      by (rewrite forallb_forall in Hwfs; exact (Hwfs ins (nth_error_In _ _ Hnth))).
    destruct (transfer_sound F prog E HE t s0 s _ ins Ef Hrun Hst0 Hrs0 Hwf) as [_ Hrs1].
    (* every plan of block b holds one step further *)
    assert (Hall1 : forall q, map fst q = body ->
                      state_ok F (S t) (plan_walk F entry0 (firstn (S i) q)) (regs s)).
    { intros q Hq. destruct (plan_nth q body i ins Hq Hnth) as [keep Hkeep].
      rewrite (plan_walk_step _ _ _ _ _ Hkeep). apply trim_after_sound.
      exact (proj1 (transfer_sound F prog E HE t s0 s _ ins Ef Hrun (Hall0 q Hq) Hrs0 Hwf)). }
    pose proof (Hall1 plan Hplan) as Hst1.
    split; [exact Hrs1 |]. intros Hb'.
    (* the edges of block b *)
    assert (Hedge : forall v s1, In (v, Some s1) (edge_states F prog b (plan_walk F entry0 plan)) ->
                      (v < length prog)%nat -> state_ok F (S t) s1 (regs s) ->
                      (0 < block_len prog v)%nat /\
                      exists target, In target (cases_at inv v) /\ state_ok F (S t) target (regs s)).
    { intros v s1 Hin Hv Hs1. rewrite forallb_forall in Hedges. specialize (Hedges _ Hin). cbv beta iota in Hedges.
      destruct (Nat.ltb_spec v (length prog)); [|lia].
      apply existsb_exists in Hedges. destruct Hedges as [target [HinT Hle]].
      split; [exact (proj1 (structured_block v Hstr Hv))|].
      exists target. split; [exact HinT|].
      exact (state_le_sound F E HE (S t) (regs s) Hrs1 s1 target Hs1 Hle). }

    (* at the block's end the walk is the block's out state *)
    assert (Hend : S i = length body -> state_ok F (S t) (plan_walk F entry0 plan) (regs s)).
    { intros Hl. rewrite firstn_all2 in Hst1 by (rewrite <- Hplan, map_length in Hl; lia).
      exact Hst1. }
    assert (Hjl : is_jump ins = true -> S i = length body) by (exact (proj2 (structured_block b Hstr Hb) i ins Hnth)).
    assert (Hat : forall v, (blk s, idx s) = (v, 0%nat) -> (v < length prog)%nat ->
                   (0 < block_len prog v)%nat /\
                   (exists target, In target (cases_at inv v) /\ state_ok F (S t) target (regs s)) ->
                   (idx s < block_len prog (blk s))%nat /\
                   exists target, In target (cases_at inv (blk s)) /\
                     forall q, map fst q = block_body prog (blk s) ->
                     state_ok F (S t) (plan_walk F target (firstn (idx s) q)) (regs s)).
    { intros v Hv _ [H1 [target [HinT H2]]]. injection Hv as -> ->.
      split; [exact H1|]. exists target. split; [exact HinT|]. intros q _. cbn [firstn plan_walk]. exact H2. }
    destruct (step_pos t s0 s ins Ef Hrun) as [[Hn [Hbra Hexit]] | [[g [tg [-> [Hg Hp]]]] | [g [-> [Hg Hp]]]]].
    + (* the next position *)
      unfold next_pos in Hn. fold b i in Hn.
      destruct (Nat.ltb_spec (S i) (block_len prog b)) as [Hlt | Hge].
      * injection Hn as -> ->. split; [exact Hlt|]. exists entry0. split; [exact Hentry0|exact Hall1].
      * injection Hn as Hbs His. rewrite Hbs in Hb'.
        assert (Hl : S i = length body) by (unfold block_len in Hge, Hi; fold body in Hge, Hi; lia).
        apply (Hat (S b)); [rewrite Hbs, His; reflexivity | exact Hb' |].
        pose proof (last_of_nth body i ins Hnth Hl) as Hlast.
        unfold edge_states in Hedge. fold body in Hedge. rewrite Hlast in Hedge.
        destruct ins as [| | | | | | | | | | | | | | | g tg | g | | | | | | | |];
          try (apply (Hedge (S b) _ (or_introl eq_refl) Hb' (Hend Hl))).
        -- (* a branch whose guard fails *)
           specialize (Hbra g tg eq_refl). rewrite (jump_regs t s0 s _ Ef eq_refl Hrun) in *.
           destruct g as [[neg p] |]; [| discriminate].
           cbn [holds] in Hbra.
           destruct (refine_sound F E HE (S t) (regs s0) Hrs0 _ p neg (Hend Hl)) as [s1 [Er Hs1]].
           { destruct neg, (Z.eqb (regs s0 p) 0); cbn in Hbra |- *; congruence. }
           apply (Hedge (S b) s1); [rewrite Er; right; left; reflexivity | exact Hb' | exact Hs1].
        -- (* an exit whose guard fails *)
           specialize (Hexit g eq_refl). rewrite (jump_regs t s0 s _ Ef eq_refl Hrun) in *.
           destruct g as [[neg p] |]; [| discriminate].
           cbn [holds] in Hexit.
           destruct (refine_sound F E HE (S t) (regs s0) Hrs0 _ p neg (Hend Hl)) as [s1 [Er Hs1]].
           { destruct neg, (Z.eqb (regs s0 p) 0); cbn in Hexit |- *; congruence. }
           apply (Hedge (S b) s1); [rewrite Er; left; reflexivity | exact Hb' | exact Hs1].
    + (* a taken branch *)
      pose proof Hp as Hp'. injection Hp' as Hbs _. rewrite Hbs in Hb'.
      apply (Hat tg Hp Hb').
      pose proof (Hjl eq_refl) as Hl.
      pose proof (last_of_nth body i _ Hnth Hl) as Hlast.
      rewrite (jump_regs t s0 s _ Ef eq_refl Hrun) in *.
      unfold edge_states in Hedge. fold body in Hedge. rewrite Hlast in Hedge.
      destruct g as [[neg p] |].
      * cbn [holds] in Hg.
        destruct (refine_sound F E HE (S t) (regs s0) Hrs0 _ p (negb neg) (Hend Hl)) as [s1 [Er Hs1]].
        { destruct neg, (Z.eqb (regs s0 p) 0); cbn in Hg |- *; congruence. }
        apply (Hedge tg s1); [rewrite Er; left; reflexivity | exact Hb' | exact Hs1].
      * apply (Hedge tg _ (or_introl eq_refl) Hb' (Hend Hl)).
    + (* an exit leaves the program *)
      injection Hp as Hbs _. rewrite Hbs in Hb'. lia.
Qed.

(* ------------------------------------------------------------------ *)
(* The theorem.                                                        *)

(* If the check accepts the program with invariants inv, every global
   access of every run that starts from the launch facts stays in its
   region. *)
Theorem check_cases_sound : forall inv,
    check_cases F prog inv = true -> launch_ok ->
    forall t s i, run' F prog E t = Some s -> fetch prog (blk s) (idx s) = Some i ->
      (forall k addr width v, access_at (in_window F E) (regs s) i = Some (k, addr, width, v) ->
         region F E t k addr width v) /\
      (forall addr width, mem_access (regs s) i = Some (addr, width) -> aligned (need width) addr) /\
      (forall p, private_access (in_window F E) (regs s) i = Some p -> private_in F E p) /\
      wf_instr F i = true.
Proof.
  intros inv Hc Hk t s i Hrun Hf.
  pose proof Hc as Hc'. unfold check_cases in Hc'. cbv zeta in Hc'. rewrite !andb_true_iff in Hc'.
  destruct Hc' as [[[Hlf Hstr] _] Hblocks].
  pose proof (env_of_launch Hlf Hk) as HE.
  destruct (run_inv inv (check_facts_of_check_cases inv Hc) Hk t s Hrun) as [Hrs Hin].
  destruct (fetch_block _ _ _ Hf) as [Hb _]. destruct (Hin Hb) as [_ [entry [Hentry Hall]]].
  pose proof (Hall _ (case_plan_body inv (blk s))) as Hst.
  rewrite forallb_forall in Hblocks.
  specialize (Hblocks (blk s) ltac:(apply in_seq; lia)). rewrite forallb_forall in Hblocks.
  pose proof (Hblocks entry Hentry) as Hblk. unfold case_block_ok in Hblk.
  rewrite (not_bottom t entry _ (regs s) Hst) in Hblk.
  destruct (checked_walk F entry (case_plan F prog inv (blk s))) as [out |] eqn:Hwalk; [| discriminate].
  destruct (plan_nth _ _ (idx s) i (case_plan_body inv (blk s)) Hf) as [keep Hkeep].
  destruct (checked_walk_at _ _ _ (idx s) i keep Hwalk Hkeep) as [Hwf Hacc].
  split; [| split; [| split]].
  - intros k addr width v Ha. exact (access_of_sound F E HE t (regs s) Hrs _ i k addr width v Hst Hwf Hacc Ha).
  - intros addr width Ha. exact (aligned_of_sound F E HE t (regs s) Hrs _ i addr width Hst Hacc Ha).
  - intros p Hp. exact (private_of_sound F E HE t (regs s) Hrs _ i p Hst Hwf Hacc Hp).
  - exact Hwf.
Qed.

Theorem check_thread_sound : forall inv,
 check_thread F prog inv = true -> launch_ok ->
 forall t s i, run' F prog E t = Some s -> fetch prog (blk s) (idx s) = Some i ->
 (forall k addr width v, access_at (in_window F E) (regs s) i = Some (k,addr,width,v) -> region F E t k addr width v) /\
 (forall addr width, mem_access (regs s) i = Some (addr,width) -> aligned (need width) addr) /\
 (forall p, private_access (in_window F E) (regs s) i = Some p -> private_in F E p) /\ wf_instr F i = true.
Proof. intros inv H. exact (check_cases_sound _ H). Qed.

End Address.

(* ------------------------------------------------------------------ *)
(* Every thread of every block.                                        *)

(* In every run that starts from the launch facts, every global access
   stays in its region, every memory access, in any state space, is
   aligned to what its width needs, and every access to shared or local
   memory lies inside a variable of that memory, which the block's or the
   thread's memory holds: a trap, a breakpoint, and undefined behavior are
   stores to address 0, which no region contains, so none of them happens.
   Every instruction a thread reaches is well formed: its registers have
   the widths it needs, and a load reads whole bytes.

   This covers every thread of every block of a launch. A thread runs the
   one-thread semantics in its own environment: its launch values differ
   from those of the other threads only in the thread registers, and each
   load from global or shared memory and each unmodeled result gives the
   value that the environment chooses for that step. So the run of a thread
   in any interleaving of the threads of a block is a run in some
   environment, and launch_ok holds for the environment of every thread. *)
Definition kernel_safe (F : lfacts) (prog : program) : Prop :=
  forall E, launch_ok F E ->
  forall t s i, run' F prog E t = Some s -> fetch prog (blk s) (idx s) = Some i ->
    (forall k addr width v, access_at (in_window F E) (regs s) i = Some (k, addr, width, v) ->
       region F E t k addr width v) /\
    (forall addr width, mem_access (regs s) i = Some (addr, width) -> aligned (need width) addr) /\
    (forall p, private_access (in_window F E) (regs s) i = Some p -> private_in F E p) /\
    wf_instr F i = true.

Lemma split_lookup : forall bounds r cut low u,
 lookup (map (split_bound r cut low) bounds) u =
 option_map (fun pair => let '(lo,hi) := pair in
   if Pos.eqb u r then if low then (lo,Z.min hi (cut-1)) else (Z.max lo cut,hi) else (lo,hi))
 (lookup bounds u).
Proof.
 intros bounds r cut low u. induction bounds as [|[v [lo hi]] bounds IH]; [reflexivity|].
 cbn [map split_bound]. destruct (Pos.eqb v r) eqn:Evr; cbn [lookup];
 destruct (Pos.eqb v u) eqn:Evu; cbn [option_map]; try exact IH.
 - apply Pos.eqb_eq in Evu. subst v. rewrite Evr. reflexivity.
 - apply Pos.eqb_eq in Evu. subst v. rewrite Evr. reflexivity.
Qed.

Lemma launch_split : forall F E r cut (low : bool),
 launch_ok F E ->
 (if low then e_init E r < cut else cut <= e_init E r) ->
 launch_ok (split_launch F r cut low) E.
Proof.
 intros F E r cut low H Hside.
 constructor.
 - exact (lk_init F E H).
 - intros u lo hi Hfixed Hb. cbn [split_launch lf_bounds] in Hb. rewrite split_lookup in Hb.
   destruct (lookup (lf_bounds F) u) as [[a b]|] eqn:Hu; [|discriminate].
   pose proof (lk_bounds F E H u a b Hfixed Hu) as HR. cbn [option_map] in Hb.
   destruct (Pos.eqb u r) eqn:Eur.
   + apply Pos.eqb_eq in Eur. subst u. destruct low; injection Hb as <- <-; lia.
   + injection Hb as <- <-. exact HR.
 - exact (lk_aligns F E H).
 - exact (lk_index F E H).
 - exact (lk_disjoint F E H).
 - exact (lk_apart F E H).
 - exact (lk_derived F E H).
Qed.

Lemma kernel_safe_split : forall F prog r cut,
 kernel_safe (split_launch F r cut true) prog ->
 kernel_safe (split_launch F r cut false) prog -> kernel_safe F prog.
Proof.
 intros F prog r cut Hl Hh E Hk.
 destruct (Z.lt_ge_cases (e_init E r) cut) as [Hlow|Hhigh].
 - pose proof (Hl E (launch_split F E r cut true Hk Hlow)) as HS.
   intros t s i Hrun Hfetch. destruct (HS t s i Hrun Hfetch) as [Hg Hrest].
   split; [|exact Hrest]. intros k addr width v Ha. specialize (Hg k addr width v Ha).
   inversion Hg; subst. econstructor; eassumption.
 - pose proof (Hh E (launch_split F E r cut false Hk ltac:(cbn; lia))) as HS.
   intros t s i Hrun Hfetch. destruct (HS t s i Hrun Hfetch) as [Hg Hrest].
   split; [|exact Hrest]. intros k addr width v Ha. specialize (Hg k addr width v Ha).
   inversion Hg; subst. econstructor; eassumption.
Qed.

Lemma check_partition_sound : forall F prog plan,
 check_partition F prog plan = true -> kernel_safe F prog.
Proof.
 intros F prog plan H E Hk. unfold check_partition in H.
 destruct (propose_cases F prog plan) as [inv|]; [|discriminate].
 exact (check_cases_sound F prog E inv H Hk).
Qed.

Lemma check_block_lf_ok : forall F prog, check_block F prog = true -> lf_ok F = true.
Proof. intros F prog H. unfold check_block in H. destruct (lf_ok F); [reflexivity|discriminate]. Qed.

Theorem check_block_sound : forall F prog, check_block F prog = true -> kernel_safe F prog.
Proof.
 intros F prog H. unfold check_block in H. destruct (lf_ok F); [|discriminate].
 destruct (split_key F prog) as [[r cut]|].
 - apply andb_true_iff in H. destruct H as [Hl Hh].
   exact (kernel_safe_split F prog r cut
     (check_partition_sound (split_launch F r cut true) prog (loop_plan prog) Hl)
     (check_partition_sound (split_launch F r cut false) prog (loop_plan prog) Hh)).
 - exact (check_partition_sound F prog (independent_plan (conditional_loop_plan prog)) H).
Qed.


Lemma check_case_certificate_sound : forall F prog cert,
 check_case_certificate F prog cert = true -> kernel_safe F prog.
Proof.
 intros F prog [inv|] H; [|discriminate].
 exact (fun E Hk => check_cases_sound F prog E inv H Hk).
Qed.

Lemma check_block_certificate_lf_ok : forall F prog cert,
 check_block_certificate F prog cert = true -> lf_ok F = true.
Proof.
 intros F prog cert H. unfold check_block_certificate in H.
 apply andb_true_iff in H. exact (proj1 H).
Qed.

Theorem check_block_certificate_sound : forall F prog cert,
 check_block_certificate F prog cert = true -> kernel_safe F prog.
Proof.
 intros F prog cert H. unfold check_block_certificate in H.
 apply andb_true_iff in H. destruct H as [_ H].
 destruct cert as [cases|r cut low high].
 - exact (check_case_certificate_sound F prog cases H).
 - apply andb_true_iff in H. destruct H as [Hl Hh].
   exact (kernel_safe_split F prog r cut
     (check_case_certificate_sound _ _ _ Hl)
     (check_case_certificate_sound _ _ _ Hh)).
Qed.
