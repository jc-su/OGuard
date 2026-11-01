(* Finite progress certificates for a single CTA synchronization event.
   The control slice selects facts to retain; its accuracy is not trusted.
   Every omitted fact is forgotten soundly, and every guarded concrete
   successor is checked. Spatial accesses remain the ordinary checker's
   responsibility. This module does not model other blocking collectives. *)

From Coq Require Import List ZArith Bool Arith Lia.
Require Import OpGate.GateIsa OpGate.GateForm OpGate.GateDomain
 OpGate.GateMeaning OpGate.GateSoundRange OpGate.GateSoundRefine
 OpGate.GateSoundTransfer OpGate.GateThread.
Import ListNotations.
Local Open Scope Z_scope.

Definition phase_op_regs := opnd_inputs.

(* The progress slice historically omits plain store payloads. Preserve
   that selector while ordinary analysis uses the complete input list. *)
Definition phase_required_reads (i : instr) : list reg :=
  match i with
  | IMem g KStore false a _ _ _ _ [] =>
      match g with Some (_, p) => [p; a] | None => [a] end
  | _ => instruction_inputs i
  end.

Definition phase_mark := mark_regs.

Definition phase_trim (keep : PM.t unit) (st : astate) : astate :=
  {| amap := filter_fact_map keep (amap st); acons := [] |}.

Definition phase_slice_step (body : list instr) (keep : PM.t unit) :=
 fold_left (fun m i => match dest i with
  | Some d => if PM.mem d m then phase_mark (phase_required_reads i) m else m
  | None => m end) body keep.

(* Stop only at an exact structural fixed point. The following theorem
   proves this computes the same marks as all n original passes. *)
Definition phase_marks_eqb := register_marks_eqb.
Lemma phase_marks_eqb_eq : forall x y,
  phase_marks_eqb x y = true -> x = y.
Proof. exact register_marks_eqb_eq. Qed.
Fixpoint phase_slice (n : nat) (body : list instr) (keep : PM.t unit) :=
 match n with
 | O => keep
 | S n => let next := phase_slice_step body keep in
   if phase_marks_eqb next keep then keep else phase_slice n body next
 end.
Lemma phase_slice_iter_fixed : forall n body keep,
 phase_slice_step body keep = keep ->
 Nat.iter n (phase_slice_step body) keep = keep.
Proof.
 induction n as [|n IH]; intros body keep H; [reflexivity|].
 change (phase_slice_step body (Nat.iter n (phase_slice_step body) keep)=keep).
 rewrite IH by exact H. exact H.
Qed.
Lemma phase_slice_iter_shift : forall n body keep,
 Nat.iter n (phase_slice_step body) (phase_slice_step body keep) =
 Nat.iter (S n) (phase_slice_step body) keep.
Proof.
 induction n as [|n IH]; intros body keep; [reflexivity|].
 change (phase_slice_step body (Nat.iter n (phase_slice_step body) (phase_slice_step body keep))=
 phase_slice_step body (Nat.iter (S n) (phase_slice_step body) keep)).
 rewrite IH. reflexivity.
Qed.
Theorem phase_slice_exact : forall n body keep,
 phase_slice n body keep = Nat.iter n (phase_slice_step body) keep.
Proof.
 induction n as [|n IH]; intros body keep; cbn [phase_slice]; [reflexivity|].
 destruct (phase_marks_eqb (phase_slice_step body keep) keep) eqn:Hsame.
 - apply phase_marks_eqb_eq in Hsame. symmetry. apply phase_slice_iter_fixed. exact Hsame.
 - rewrite IH. apply phase_slice_iter_shift.
Qed.

Definition phase_control_regs (body : list instr) :=
 phase_slice (List.length body) body
 (fold_left (fun m i => if is_jump i then match guard_of i with
  | Some (_, p) => PM.add p tt m | None => m end else m) body (PM.empty unit)).

Definition pc := (nat * nat)%type.

Definition phase_guard_edges (F : lfacts) (out : astate) (g : guard) (yes no : pc)
 : list (pc * option astate) :=
 match g with
 | None => [(yes, Some out)]
 | Some (neg, p) => [(yes, refine F out p (negb neg)); (no, refine F out p neg)]
 end.

Definition phase_successors (keep : PM.t unit) (F : lfacts) (prog : program) (pos : pc) (st : astate) (i : instr)
 : list (pc * option astate) :=
 let out := phase_trim keep (transfer F st i) in
 let next := next_pos prog (fst pos) (snd pos) in
 match i with
 | IBra g target => phase_guard_edges F out g (target, 0%nat) next
 | IExit g => phase_guard_edges F out g (List.length prog, 0%nat) next
 | _ => [(next, Some out)]
 end.
(* Fuel counts all pending goals, including both sides of a branch. The
   checker therefore performs at most fuel instruction expansions in total,
   rather than allowing an exponential tree under a per-path depth bound. *)

Fixpoint phase_all (keep : PM.t unit) (fuel : nat) (F : lfacts) (prog : program)
 (event : pc -> bool) (pending : list (pc * option astate)) : bool :=
 match pending with
 | [] => true
 | (pos, ost)::tail => match fuel with
   | O => false
   | S rest => match ost with
     | None => phase_all keep rest F prog event tail
     | Some st => if is_bottom st then phase_all keep rest F prog event tail else
       if event pos then phase_all keep rest F prog event tail else
       match fetch prog (fst pos) (snd pos) with
       | None => false
       | Some i => if wf_instr F i then
          phase_all keep rest F prog event
            (phase_successors keep F prog pos st i ++ tail)
         else false end
     end
   end
 end.

Definition phase_check (keep : PM.t unit) (fuel : nat) (F : lfacts) (prog : program)
 (event : pc -> bool) (pos : pc) (st : astate) : bool :=
 phase_all keep fuel F prog event [(pos, Some st)].

Lemma phase_trim_lookup : forall keep st r f,
  get (phase_trim keep st) r = Some f -> get st r = Some f.
Proof.
  intros keep st r f H. unfold get, phase_trim in H; cbn [amap] in H.
  exact (filter_fact_map_lookup keep (amap st) r f H).
Qed.

Lemma phase_trim_sound : forall F t keep st rs,
 state_ok F t st rs -> state_ok F t (phase_trim keep st) rs.
Proof.
 intros F t keep st rs [Hf Hc]. split.
 - intros r f H. apply Hf. eapply phase_trim_lookup; exact H.
 - constructor.
Qed.

Section Progress.
Variable F : lfacts.
Variable prog : program.
Variable E : env.
Hypothesis HE : env_ok F E.

Definition phase_step := step prog (wdn F) (e_mem E) (e_arb E) (e_shm E)
 (e_init E) (lf_cells F) (lf_locals F) (lf_local_window F).

Definition pc_of (s : state) := (blk s, idx s).

Lemma phase_guard_complete : forall t rs out g yes no,
 state_ok F t out rs -> regs_ok F E rs ->
 exists st, In ((if holds g rs then yes else no), Some st)
   (phase_guard_edges F out g yes no) /\ state_ok F t st rs.
Proof.
 intros t rs out [[neg p]|] yes no Hst Hrs.
 - destruct (refine_sound F E HE t rs Hrs out p
     (negb (Z.eqb (rs p) 0)) Hst eq_refl) as [st [Hr Hs]].
   exists st. split; [|exact Hs].
   cbn [phase_guard_edges holds].
   destruct neg, (Z.eqb (rs p) 0); cbn [negb] in Hr |- *; rewrite Hr; cbn; auto.
 - exists out. split; [left; reflexivity|exact Hst].
Qed.

Lemma phase_successors_complete : forall keep t s s' st i,
 fetch prog (blk s) (idx s) = Some i -> phase_step t s = Some s' ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) -> wf_instr F i = true ->
 exists out, In (pc_of s', Some out) (phase_successors keep F prog (pc_of s) st i) /\
 state_ok F (S t) out (regs s') /\ regs_ok F E (regs s').
Proof.
 intros keep t s s' st i Hf Hstep Hst Hrs Hwf.
 destruct (transfer_sound F prog E HE t s s' st i Hf Hstep Hst Hrs Hwf) as [Ho Hr].
 pose proof (phase_trim_sound F (S t) keep _ _ Ho) as Htrim.
 pose proof (step_pos F prog E t s s' i Hf Hstep) as Hp.
 assert (Hbranch : forall g target, i = IBra g target ->
   pc_of s' = if holds g (regs s') then (target, 0%nat) else next_pos prog (blk s) (idx s)).
 { intros g target Hi. subst i.
   pose proof (jump_regs F prog E t s s' _ Hf eq_refl Hstep) as Hreg.
   rewrite Hreg. unfold pc_of.
   destruct Hp as [[Hn [Hb He]]|[[g' [t' [Hi [Hg Hn]]]]|[g' [Hi [Hg Hn]]]]].
   - rewrite (Hb g target eq_refl). exact Hn.
   - injection Hi as <- <-. rewrite Hg. exact Hn.
   - discriminate. }
 assert (Hexit : forall g, i = IExit g ->
   pc_of s' = if holds g (regs s') then (List.length prog, 0%nat) else next_pos prog (blk s) (idx s)).
 { intros g Hi. subst i.
   pose proof (jump_regs F prog E t s s' _ Hf eq_refl Hstep) as Hreg.
   rewrite Hreg. unfold pc_of.
   destruct Hp as [[Hn [Hb He]]|[[g' [t' [Hi [Hg Hn]]]]|[g' [Hi [Hg Hn]]]]].
   - rewrite (He g eq_refl). exact Hn.
   - discriminate.
   - injection Hi as <-. rewrite Hg. exact Hn. }
 destruct i; cbn [phase_successors pc_of fst snd];
   try (eexists; split; [|split; [exact Htrim|exact Hr]];
        left; destruct Hp as [[Hn _]|[[g0 [tg0 [Hbad _]]]|[g0 [Hbad _]]]];
        [unfold pc_of; rewrite Hn; reflexivity|discriminate|discriminate]).
 - destruct (phase_guard_complete (S t) (regs s') _ g (target, 0%nat)
     (next_pos prog (blk s) (idx s)) Htrim Hr) as [out [Hin Hout]].
   exists out. split; [|split; assumption]. rewrite (Hbranch _ _ eq_refl). exact Hin.
 - destruct (phase_guard_complete (S t) (regs s') _ g (List.length prog, 0%nat)
     (next_pos prog (blk s) (idx s)) Htrim Hr) as [out [Hin Hout]].
   exists out. split; [|split; assumption]. rewrite (Hexit _ eq_refl). exact Hin.
Qed.

Inductive phase_reaches (event : pc -> bool) : nat -> state -> Prop :=
| Phase_here : forall t s, event (pc_of s) = true -> phase_reaches event t s
| Phase_next : forall t s s', event (pc_of s) = false ->
    phase_step t s = Some s' -> phase_reaches event (S t) s' -> phase_reaches event t s.

Lemma phase_all_progress : forall fuel keep event pending pos st t s,
 phase_all keep fuel F prog event pending = true ->
 In (pos, Some st) pending -> pos = pc_of s ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) -> phase_reaches event t s.
Proof.
 induction fuel as [|fuel IH]; intros keep event pending pos st t s Hcheck Hin Hpos Hst Hrs;
   destruct pending as [|[q ost] tail]; [contradiction|discriminate|contradiction|].
 cbn [phase_all] in Hcheck. destruct ost as [current|].
 - destruct (is_bottom current) eqn : Hbottom.
   + destruct Hin as [Heq|Hin].
     * injection Heq as <- <-. exfalso. exact (bottom_not_ok F t current _ Hbottom Hst).
     * eapply IH; eassumption.
   + destruct (event q) eqn : Hevent.
     * destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. apply Phase_here. rewrite <-Hpos. exact Hevent.
       -- eapply IH; eassumption.
     * destruct (fetch prog (fst q) (snd q)) as [i|] eqn : Hf; [|discriminate].
       destruct (wf_instr F i) eqn : Hwf; [|discriminate].
       destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. subst q.
          assert (Hex : exists s', phase_step t s = Some s').
          { unfold phase_step, step. change (fetch prog (blk s) (idx s) = Some i) in Hf.
            rewrite Hf. eexists; reflexivity. }
          destruct Hex as [s' Hstep].
          destruct (phase_successors_complete keep t s s' current i Hf Hstep Hst Hrs Hwf)
            as [out [Hchild [Hout Hrs']]].
          eapply Phase_next; [exact Hevent|exact Hstep|].
          eapply IH; [exact Hcheck| |reflexivity|exact Hout|exact Hrs'].
          apply in_or_app. left. exact Hchild.
       -- eapply IH; [exact Hcheck| |exact Hpos|exact Hst|exact Hrs].
          apply in_or_app. right. exact Hin.
 - destruct Hin as [Heq|Hin]; [discriminate|]. eapply IH; eassumption.
Qed.

Theorem phase_check_progress : forall fuel keep event pos st t s,
 phase_check keep fuel F prog event pos st = true -> pos = pc_of s ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) -> phase_reaches event t s.
Proof.
 intros fuel keep event pos st t s Hcheck Hpos Hst Hrs.
 eapply phase_all_progress; [exact Hcheck|left; reflexivity|exact Hpos|exact Hst|exact Hrs].
Qed.

End Progress.

Definition phase_after (event pos : pc) : bool :=
 Nat.ltb (fst event) (fst pos) ||
 (Nat.eqb (fst event) (fst pos) && Nat.ltb (snd event) (snd pos)).

(* Traverse the actual instructions once. Only branches can violate closure;
   indexed fetch of every instruction would repeatedly walk each block. *)
Definition phase_closed (prog : program) (event : pc) : bool :=
 short_all (fun block => let '(b,body) := block in
   short_all (fun item => let '(i,ins) := item in
     match ins with
     | IBra _ target => if phase_after event (b,i) then Nat.ltb (fst event) target else true
     | _ => true end)
     (combine (seq 0 (List.length body)) body))
   (combine (seq 0 (List.length prog)) prog).

Local Lemma phase_scan_enumeration : forall (A:Type) (xs:list A) (d:A),
 combine (seq 0 (List.length xs)) xs =
 map (fun i=>(i,nth i xs d)) (seq 0 (List.length xs)).
Proof.
 intros A xs d.
 apply nth_ext with (d:=(0%nat,d)) (d':=(0%nat,nth 0 xs d)).
 - rewrite combine_length,map_length,!seq_length,Nat.min_id. reflexivity.
 - intros n Hn. rewrite combine_length,seq_length,Nat.min_id in Hn.
   rewrite combine_nth by apply seq_length.
   rewrite (map_nth (fun i=>(i,nth i xs d)) (seq 0 (List.length xs)) 0%nat n).
   rewrite seq_nth by exact Hn. reflexivity.
Qed.
Local Lemma phase_scan_all_map : forall A B (p:B->bool) (f:A->B) xs,
 short_all p (map f xs)=short_all (fun x=>p(f x)) xs.
Proof. intros A B p f xs;induction xs;cbn;[reflexivity|]. destruct(p(f a));cbn;auto. Qed.
Local Lemma phase_scan_all_ext : forall A (p q:A->bool) xs,
 (forall x,In x xs->p x=q x)->short_all p xs=short_all q xs.
Proof.
 intros A p q xs;induction xs;intros H;cbn;[reflexivity|].
 rewrite (H a (or_introl eq_refl)). destruct(q a);[|reflexivity].
 apply IHxs. intros x Hx. apply H. right;exact Hx.
Qed.
Theorem phase_closed_exact : forall prog event,
 phase_closed prog event=
 short_all (fun b => short_all (fun i =>
  if phase_after event (b, i) then
   match fetch prog b i with
   | Some (IBra _ target) => Nat.ltb (fst event) target
   | _ => true end
  else true) (seq 0 (block_len prog b))) (seq 0 (List.length prog)).
Proof.
 intros prog event. unfold phase_closed.
 rewrite (phase_scan_enumeration _ prog []),phase_scan_all_map.
 apply phase_scan_all_ext. intros b Hb. apply in_seq in Hb.
 assert (Hbody:nth_error prog b=Some(nth b prog [])).
 { apply nth_error_nth'. lia. }
 cbn beta iota zeta.
 unfold block_len,block_body. rewrite Hbody.
 rewrite (phase_scan_enumeration _ (nth b prog []) INop),phase_scan_all_map.
 apply phase_scan_all_ext. intros i Hi. apply in_seq in Hi.
 cbn beta iota zeta. unfold fetch,block_body. rewrite Hbody.
 rewrite (@nth_error_nth' instr (nth b prog []) i INop) by lia.
 destruct (nth i (nth b prog []) INop);destruct(phase_after event(b,i));reflexivity.
Qed.

Lemma phase_after_iff : forall e p,
 phase_after e p = true <->
 (fst e < fst p \/ (fst e = fst p /\ snd e < snd p))%nat.
Proof.
 intros e p. unfold phase_after. rewrite orb_true_iff, andb_true_iff,
 !Nat.ltb_lt, Nat.eqb_eq. reflexivity.
Qed.

Lemma phase_after_irrefl : forall e, phase_after e e = false.
Proof. intros e. destruct (phase_after e e) eqn : H; [|reflexivity]. apply phase_after_iff in H. lia. Qed.

Lemma phase_next_after : forall prog e p,
 phase_after e p = true ->
 phase_after e (next_pos prog (fst p) (snd p)) = true.
Proof.
 intros prog [eb ei] [b i] H. apply phase_after_iff in H.
 apply phase_after_iff. unfold next_pos; cbn [fst snd] in *.
 destruct (Nat.ltb (S i) (block_len prog b)); cbn [fst snd]; lia.
Qed.

Lemma phase_next_self : forall prog p,
 phase_after p (next_pos prog (fst p) (snd p)) = true.
Proof.
 intros prog [b i]. apply phase_after_iff. unfold next_pos; cbn [fst snd].
 destruct (Nat.ltb (S i) (block_len prog b)); cbn [fst snd]; lia.
Qed.

Lemma phase_fetch_bounds : forall prog b i ins,
 fetch prog b i = Some ins -> (b < List.length prog /\ i < block_len prog b)%nat.
Proof.
 intros prog b i ins H. unfold fetch, block_len, block_body in *.
 destruct (nth_error prog b) as [body|] eqn : Hb; [|destruct i; discriminate].
 split; apply nth_error_Some; [rewrite Hb|rewrite H]; discriminate.
Qed.

Lemma phase_closed_branch : forall prog e b i g target,
 phase_closed prog e = true -> phase_after e (b, i) = true ->
 fetch prog b i = Some (IBra g target) -> (fst e < target)%nat.
Proof.
 intros prog e b i g target Hc Ha Hf.
 rewrite phase_closed_exact in Hc. rewrite short_all_same, forallb_forall in Hc.
 destruct (phase_fetch_bounds prog b i _ Hf) as [Hb Hi].
 specialize (Hc b ltac : (apply in_seq; lia)).
 rewrite short_all_same, forallb_forall in Hc.
 specialize (Hc i ltac : (apply in_seq; lia)).
 rewrite Ha, Hf in Hc. apply Nat.ltb_lt. exact Hc.
Qed.

(* Event markers retain the original instruction: a CTA barrier is a nop,
   while a checked full-mask warp collective writes arbitrary result registers.
   Both have one unconditional fallthrough edge. *)
Definition phase_marker (i : instr) : bool :=
 match i with INop | IDef None _ => true | _ => false end.
Definition phase_event_at (prog : program) (event : pc) : bool :=
 match fetch prog (fst event) (snd event) with Some i => phase_marker i | None => false end.
Lemma phase_event_at_fetch : forall prog event,
 phase_event_at prog event=true ->
 exists i,fetch prog (fst event) (snd event)=Some i /\ phase_marker i=true.
Proof.
 intros prog event H. unfold phase_event_at in H.
 destruct (fetch prog (fst event) (snd event)) as [i|] eqn:Hf;[|discriminate].
 exists i. auto.
Qed.
Lemma phase_marker_not_control : forall i,
 phase_marker i=true ->
 (forall g target,i<>IBra g target) /\ (forall g,i<>IExit g).
Proof.
 intros i H. split.
 - intros g target Heq. subst i. discriminate.
 - intros g Heq. subst i. discriminate.
Qed.

Section Closed.
Variable F : lfacts.
Variable prog : program.
Variable E : env.

Lemma phase_after_step : forall event t s s' ins,
 phase_event_at prog event = true ->
 phase_closed prog event = true -> phase_after event (pc_of s) = true ->
 fetch prog (blk s) (idx s) = Some ins ->
 phase_step F prog E t s = Some s' -> phase_after event (pc_of s') = true.
Proof.
 intros event t s s' ins He Hc Ha Hf Hstep.
 pose proof (step_pos F prog E t s s' ins Hf Hstep) as Hp.
 destruct Hp as [[Hnext _]|[[g [target [Hi [Hg Htarget]]]]|[g [Hi [Hg Hexit]]]]].
 - unfold pc_of. rewrite Hnext. exact (phase_next_after prog event (pc_of s) Ha).
 - subst ins. unfold pc_of. rewrite Htarget. apply phase_after_iff. left.
   eapply phase_closed_branch; eassumption.
 - unfold pc_of. rewrite Hexit. apply phase_after_iff. left.
   destruct (phase_event_at_fetch prog event He) as [marker [Hm _]].
   exact (proj1 (phase_fetch_bounds prog _ _ _ Hm)).
Qed.

Lemma phase_event_step : forall event t s s',
 phase_event_at prog event = true ->
 pc_of s = event -> phase_step F prog E t s = Some s' ->
 phase_after event (pc_of s') = true.
Proof.
 intros event t s s' He Hpos Hstep.
 destruct (phase_event_at_fetch prog event He) as [marker [Hm Hmarker]].
 assert (Hf : fetch prog (blk s) (idx s)=Some marker).
 { change (fetch prog (fst (pc_of s)) (snd (pc_of s))=Some marker). rewrite Hpos. exact Hm. }
 pose proof (step_pos F prog E t s s' marker Hf Hstep) as Hp.
 destruct Hp as [[Hnext _]|[[g [target [Hbad _]]]|[g [Hbad _]]]].
 - unfold pc_of. rewrite Hnext. rewrite <-Hpos. exact (phase_next_self prog (pc_of s)).
 - exfalso. exact (proj1 (phase_marker_not_control marker Hmarker) g target Hbad).
 - exfalso. exact (proj2 (phase_marker_not_control marker Hmarker) g Hbad).
Qed.

Fixpoint phase_run (n t : nat) (s : state) : option state :=
 match n with O => Some s | S n => match phase_step F prog E t s with
  | None => None | Some s' => phase_run n (S t) s' end end.

Theorem phase_stays_after : forall n event t s s',
 phase_event_at prog event = true ->
 phase_closed prog event = true -> phase_after event (pc_of s) = true ->
 phase_run n t s = Some s' -> phase_after event (pc_of s') = true.
Proof.
 induction n as [|n IH]; intros event t s s' He Hc Ha Hrun.
 - cbn [phase_run] in Hrun. injection Hrun as <-. exact Ha.
 - cbn [phase_run] in Hrun.
   destruct (phase_step F prog E t s) as [next|] eqn : Hstep; [|discriminate].
   assert (Hex : exists ins, fetch prog (blk s) (idx s)=Some ins).
   { unfold phase_step, step in Hstep.
     destruct (fetch prog (blk s) (idx s)) as [ins|]; [eexists; reflexivity|discriminate]. }
   destruct Hex as [ins Hf].
   eapply IH; [exact He|exact Hc| |exact Hrun].
   eapply phase_after_step; eassumption.
Qed.

Corollary phase_never_returns : forall n event t s s',
 phase_event_at prog event = true ->
 phase_closed prog event = true -> phase_after event (pc_of s) = true ->
 phase_run n t s = Some s' -> pc_of s' <> event.
Proof.
 intros n event t s s' He Hc Ha Hr Hsame.
 pose proof (phase_stays_after n event t s s' He Hc Ha Hr) as Hafter.
 rewrite Hsame, phase_after_irrefl in Hafter. discriminate.
Qed.

End Closed.

Definition phase_event (event pos : pc) : bool :=
 Nat.eqb (fst event) (fst pos) && Nat.eqb (snd event) (snd pos).

Lemma phase_event_eq : forall event pos,
 phase_event event pos = true <-> pos = event.
Proof.
 intros [eb ei] [b i]. unfold phase_event; cbn [fst snd].
 rewrite andb_true_iff, !Nat.eqb_eq. split; [intros [-> ->]; reflexivity|intros H; injection H; auto].
Qed.

Definition phase_initial (E : env) : state :=
 {|blk:=0%nat; idx:=0%nat; regs:=e_init E|}.

Definition phase_positions (prog : program) : list pc :=
 flat_map (fun b => map (fun i => (b, i)) (seq 0 (block_len prog b)))
 (seq 0 (List.length prog)).

(* The exact lexical instruction prefix avoids enumerating and fetching
   every program position for each event. The equality below includes empty
   blocks and arbitrary event coordinates; no event validity is assumed. *)
Local Lemma phase_prefix_flat_map_map : forall A B C (f:B->list C) (g:A->B) xs,
 flat_map f (map g xs)=flat_map (fun x=>f(g x)) xs.
Proof. intros A B C f g xs;induction xs;cbn;[reflexivity|rewrite IHxs;reflexivity]. Qed.
Local Lemma phase_prefix_flat_map_nil : forall A B (xs:list A),
 flat_map (fun _=>@nil B) xs=[].
Proof. intros A B xs;induction xs;cbn;auto. Qed.
Local Lemma phase_prefix_flat_map_ext : forall A B (f g:A->list B) xs,
 (forall x,In x xs->f x=g x)->flat_map f xs=flat_map g xs.
Proof.
 intros A B f g xs;induction xs;intros H;cbn;[reflexivity|].
 rewrite (H a (or_introl eq_refl)),IHxs;[reflexivity|].
 intros x Hin. apply H. right;exact Hin.
Qed.

Local Lemma phase_prefix_flat_map_flat : forall A B C (f:B->list C)(g:A->list B)xs,
 flat_map f(flat_map g xs)=flat_map(fun x=>flat_map f(g x))xs.
Proof. intros A B C f g xs;induction xs;cbn;[reflexivity|rewrite flat_map_app,IHxs;reflexivity]. Qed.

Local Lemma phase_prefix_take : forall A (body:list A) d stop,
 flat_map (fun i=>if Nat.ltb i stop then [nth i body d] else [])
   (seq 0 (List.length body))=firstn stop body.
Proof.
 intros A body. induction body as [|a body IH];intros d stop.
 - destruct stop;reflexivity.
 - cbn [List.length seq flat_map]. rewrite <-seq_shift,phase_prefix_flat_map_map.
   destruct stop as [|stop].
   + cbn [Nat.ltb Nat.leb nth firstn]. apply phase_prefix_flat_map_nil.
   + change (a::flat_map (fun i=>if Nat.ltb i stop then [nth i body d] else [])
       (seq 0 (List.length body))=a::firstn stop body).
     rewrite IH. reflexivity.
Qed.
Local Lemma phase_prefix_all : forall A (body:list A) d,
 flat_map (fun i=>[nth i body d]) (seq 0 (List.length body))=body.
Proof.
 intros A body. induction body as [|a body IH];intros d;[reflexivity|].
 cbn [List.length seq flat_map nth]. rewrite <-seq_shift,phase_prefix_flat_map_map.
 change (a::flat_map (fun i=>[nth i body d]) (seq 0 (List.length body))=a::body).
 rewrite IH. reflexivity.
Qed.
Local Lemma phase_prefix_blocks : forall A (prog:list(list A)) block stop,
 flat_map (fun b=>if Nat.ltb b block then nth b prog []
   else if Nat.eqb b block then firstn stop (nth b prog []) else [])
   (seq 0 (List.length prog))=
 List.concat(firstn block prog)++firstn stop(nth block prog []).
Proof.
 intros A prog. induction prog as [|body prog IH];intros block stop.
 - destruct block,stop;reflexivity.
 - cbn [List.length seq flat_map]. rewrite <-seq_shift,phase_prefix_flat_map_map.
   destruct block as [|block].
   + cbn [Nat.ltb Nat.leb Nat.eqb nth firstn List.concat].
     rewrite phase_prefix_flat_map_nil,app_nil_r. reflexivity.
   + change (body++flat_map (fun b=>if Nat.ltb b block then nth b prog []
       else if Nat.eqb b block then firstn stop(nth b prog []) else [])
       (seq 0 (List.length prog))=
       (body++List.concat(firstn block prog))++firstn stop(nth block prog [])).
     rewrite IH,app_assoc. reflexivity.
Qed.

Definition phase_prefix (prog:program) (event:pc) : list instr :=
 List.concat(firstn (fst event) prog)++firstn (snd event)(nth (fst event) prog []).
Theorem phase_prefix_exact : forall prog event,
 phase_prefix prog event=
 flat_map (fun pos=>if phase_after pos event then
    match fetch prog (fst pos)(snd pos) with Some i=>[i]|None=>[]end
   else [])(phase_positions prog).
Proof.
 intros prog [block stop]. unfold phase_prefix,phase_positions.
 cbn [fst snd]. symmetry.
 rewrite phase_prefix_flat_map_flat.
 transitivity (flat_map (fun b=>if Nat.ltb b block then nth b prog []
   else if Nat.eqb b block then firstn stop(nth b prog []) else [])
   (seq 0(List.length prog)));[|apply phase_prefix_blocks].
 apply phase_prefix_flat_map_ext. intros b Hb. apply in_seq in Hb.
 assert (Hbody:nth_error prog b=Some(nth b prog [])).
 { apply nth_error_nth'. lia. }
 rewrite phase_prefix_flat_map_map. cbn [fst snd].
 unfold block_len,block_body. rewrite Hbody.
 transitivity (flat_map (fun i=>if phase_after (b,i)(block,stop)
   then [nth i(nth b prog []) INop] else [])(seq 0(List.length(nth b prog [])))).
 - apply phase_prefix_flat_map_ext. intros i Hi. apply in_seq in Hi.
   cbn [fst snd]. unfold fetch,block_body. rewrite Hbody.
   rewrite (@nth_error_nth' instr (nth b prog []) i INop) by lia.
   reflexivity.
 - unfold phase_after. cbn [fst snd].
   destruct (Nat.ltb b block);cbn [orb];[apply phase_prefix_all|].
   destruct (Nat.eqb b block);cbn [andb orb];[apply phase_prefix_take|apply phase_prefix_flat_map_nil].
Qed.

Definition phase_controls (prog : program) (event : pc) : PM.t unit :=
 phase_control_regs (phase_prefix prog event).
Theorem phase_controls_exact : forall prog event,
 phase_controls prog event=
 phase_control_regs (flat_map (fun pos =>
   if phase_after pos event then
     match fetch prog (fst pos) (snd pos) with Some i => [i]|None => [] end
   else []) (phase_positions prog)).
Proof. intros prog event. unfold phase_controls. rewrite phase_prefix_exact. reflexivity. Qed.
(* This is a proof budget, not an assumed loop bound. Exhaustion refuses. *)

Definition phase_fuel : nat := Nat.pow 2 13.

Definition single_phase_check (F : lfacts) (prog : program) (event : pc) : bool :=
 if lf_ok F then
 if phase_event_at prog event then
 if phase_closed prog event then
 phase_check (phase_controls prog event) phase_fuel F prog (phase_event event)
   (0%nat, 0%nat) (empty_state F)
 else false else false else false.

Theorem single_phase_check_exact : forall F prog event,
 single_phase_check F prog event =
 (lf_ok F && phase_event_at prog event && phase_closed prog event &&
  phase_check (phase_controls prog event) phase_fuel F prog (phase_event event)
    (0%nat, 0%nat) (empty_state F)).
Proof.
 intros F prog event. unfold single_phase_check.
 destruct (lf_ok F), (phase_event_at prog event), (phase_closed prog event); reflexivity.
Qed.

Definition single_phase_safe (F : lfacts) (prog : program) (event : pc) : Prop :=
 phase_event_at prog event = true /\
 phase_closed prog event = true /\
 forall E, launch_ok F E ->
 phase_reaches F prog E (phase_event event) 0 (phase_initial E).

Theorem single_phase_check_sound : forall F prog event,
 single_phase_check F prog event = true -> single_phase_safe F prog event.
Proof.
 intros F prog event H. rewrite single_phase_check_exact in H.
 rewrite !andb_true_iff in H. destruct H as [[[Hlf He] Hclosed] Hcheck].
 split; [exact He|]. split; [exact Hclosed|].
 intros E Hlaunch. eapply phase_check_progress.
 - exact (env_of_launch F prog E Hlf Hlaunch).
 - exact Hcheck.
 - reflexivity.
 - split.
   + intros r f Hf. unfold get, empty_state in Hf; cbn [amap] in Hf.
     rewrite PM.gempty in Hf. discriminate.
   + exact (lk_derived F E Hlaunch).
 - split; [exact (lk_init F E Hlaunch)|]. intros r Hfixed; reflexivity.
Qed.
