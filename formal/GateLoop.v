(* Checked symbolic progress for one straight-line loop trace.  Descriptors
   and invariant proposals are untrusted; every instruction, edge, and
   arithmetic premise is checked against the original program. *)
From Coq Require Import List ZArith Bool Arith Lia Wf_nat.
Require Import OpGate.GateIsa OpGate.GateForm OpGate.GateDomain OpGate.GateMeaning
 OpGate.GateSoundNorm OpGate.GateSoundAccess OpGate.GateSoundOp OpGate.GateSoundRange OpGate.GateSoundRefine OpGate.GateSoundJoin OpGate.GateSoundTransfer OpGate.GateThread
 OpGate.GateBarrier.
Import ListNotations.
Local Open Scope Z_scope.

Definition loop_writes (i : instr) : list reg :=
 match i with
 | IDef _ ds | IMem _ _ _ _ _ _ _ _ ds => ds
 | ILdL _ _ d _ _ _ => [d]
 | _ => match dest i with Some d=>[d]|None=>[] end
 end.
Definition loop_preserves (r : reg) (i : instr) : bool :=
 negb (existsb (Pos.eqb r) (loop_writes i)).
Definition loop_next (prog : program) (pos : pc) (i : instr) : option pc :=
 match i with
 | IBra None target => Some (target,0%nat)
 | IBra (Some _) _ | IExit _ => None
 | _ => Some (next_pos prog (fst pos) (snd pos))
 end.

Lemma loop_write_all_preserves : forall wd arb ds rs t r,
 existsb (Pos.eqb r) ds = false -> write_all wd arb rs t ds r = rs r.
Proof.
 intros wd arb ds. induction ds as [|d ds IH]; intros rs t r H; cbn [existsb write_all] in *; [reflexivity|].
 apply orb_false_iff in H. destruct H as [Hd Hds]. rewrite IH by exact Hds.
 unfold upd. rewrite Hd. reflexivity.
Qed.

Lemma cells_fold_preserves : forall (f : (reg -> Z) -> reg * reg * Z -> reg -> Z) cells r,
 (forall rs p u off, f rs (p,u,off) r = rs r \/ exists z, f rs (p,u,off) r = z /\ r = p) ->
 existsb (fun k => let '(p,_,_) := k in Pos.eqb p r) cells = false ->
 forall rs, fold_left f cells rs r = rs r.
Proof.
 intros f cells r Hf. induction cells as [|[[p u] off] cells IH]; intros Hcells rs; cbn [fold_left]; [reflexivity|].
 cbn [existsb] in Hcells. apply orb_false_iff in Hcells. destruct Hcells as [Hp Hcells].
 rewrite IH by exact Hcells. destruct (Hf rs p u off) as [H|[z [H Hr]]]; [exact H|].
 subst p. rewrite Pos.eqb_refl in Hp. discriminate.
Qed.

Lemma lstore_noncells_preserves : forall F E t rs x width v r,
 is_cell F r = false ->
 lstore (wdn F) (e_arb E) (e_init E) (lf_cells F) t rs x width v r = rs r.
Proof.
 intros F E t rs x width v r Hcell. unfold lstore.
 apply cells_fold_preserves; [|exact Hcell].
 intros rs0 p u off. unfold lstore_cell.
 destruct (_ && _)%bool; [|destruct (_ && _)%bool]; try (left;reflexivity).
 all: unfold upd; destruct (Pos.eqb_spec r p) as [Heq|Hneq];
 [right; eexists; split; [reflexivity|exact Heq]|left;reflexivity].
Qed.

Lemma lframe_noncells_preserves : forall F E t rs vs r,
 is_cell F r = false ->
 lframe (wdn F) (e_arb E) (lf_cells F) t rs vs r = rs r.
Proof.
 intros F E t rs vs r Hcell. unfold lframe.
 apply cells_fold_preserves; [|exact Hcell].
 intros rs0 p u off. unfold lframe_cell.
 destruct (existsb (Pos.eqb u) vs); [|left;reflexivity].
 unfold upd; destruct (Pos.eqb_spec r p) as [Heq|Hneq];
 [right; eexists; split; [reflexivity|exact Heq]|left;reflexivity].
Qed.

Lemma loop_step_preserves : forall F prog E t s s' i r,
 is_cell F r = false -> fetch prog (blk s) (idx s) = Some i ->
 phase_step F prog E t s = Some s' -> loop_preserves r i = true ->
 regs s' r = regs s r.
Proof.
 intros F prog E t s s' i r Hcell Hfetch Hstep Hkeep.
 unfold phase_step, step in Hstep. rewrite Hfetch in Hstep. injection Hstep as <-.
 destruct (holds (guard_of i) (regs s)) eqn:Hg; cbn [negb]; [|reflexivity].
 unfold loop_preserves in Hkeep. apply negb_true_iff in Hkeep.
 destruct i; cbn [loop_writes dest existsb] in Hkeep;
 cbn [regs goto dest]; try reflexivity;
 try (rewrite orb_false_r in Hkeep; unfold upd; rewrite Hkeep; reflexivity).
 - unfold gstore.
   destruct k, gen; cbn; try (apply loop_write_all_preserves; exact Hkeep).
   all: destruct (_ && _)%bool;
     try (apply loop_write_all_preserves; exact Hkeep).
   all: rewrite lstore_noncells_preserves by exact Hcell;
     apply loop_write_all_preserves; exact Hkeep.
 - apply loop_write_all_preserves. exact Hkeep.
 - apply lstore_noncells_preserves. exact Hcell.
 - apply lframe_noncells_preserves. exact Hcell.
Qed.

Lemma loop_next_sound : forall F prog E t s s' i pos,
 fetch prog (blk s) (idx s) = Some i -> phase_step F prog E t s = Some s' ->
 loop_next prog (pc_of s) i = Some pos -> pc_of s' = pos.
Proof.
 intros F prog E t s s' i pos Hf Hs Hn.
 pose proof (step_pos F prog E t s s' i Hf Hs) as H.
 destruct i; cbn [loop_next] in Hn; try discriminate;
 try (injection Hn as <-; destruct H as [[H _]|[[g0 [tg0 [H _]]]|[g0 [H _]]]];
      [exact H|discriminate|discriminate]).
 destruct g as [[neg p]|]; [discriminate|]. injection Hn as <-.
 destruct H as [[H [Hbra _]]|[[g' [tg [Hi [_ H]]]]|[g' [Hi _]]]].
 - specialize (Hbra None target eq_refl). discriminate.
 - injection Hi as <- <-. exact H.
 - discriminate.
Qed.

Definition loop_protected (counter stride : reg) (i : instr) : bool :=
 loop_preserves counter i && loop_preserves stride i.
Fixpoint loop_linear (fuel : nat) (F : lfacts) (prog : program)
 (counter stride : reg) (pos : pc) (st : astate)
 : option (pc * astate) :=
 match fuel with
 | O => Some (pos,st)
 | S rest => match fetch prog (fst pos) (snd pos) with
   | None => None
   | Some i => if wf_instr F i && loop_protected counter stride i then
     match loop_next prog pos i with
     | None => None
     | Some next => loop_linear rest F prog counter stride next (transfer F st i)
     end else None
   end
 end.

Lemma loop_linear_sound : forall n F prog E counter stride pos st out os t s,
 env_ok F E -> is_cell F counter = false -> is_cell F stride = false ->
 loop_linear n F prog counter stride pos st = Some (out,os) ->
 pos = pc_of s -> state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 exists s', phase_run F prog E n t s = Some s' /\ pc_of s'=out /\
 state_ok F (n+t)%nat os (regs s') /\ regs_ok F E (regs s') /\
 regs s' counter=regs s counter /\ regs s' stride=regs s stride.
Proof.
 induction n as [|n IH]; intros F prog E counter stride pos st out os t s HE Hcounter_cell Hstride_cell H Hpos Hst Hrs.
 - cbn [loop_linear] in H. injection H as <- <-. exists s. split; [reflexivity|]. split; [symmetry;exact Hpos|]. split; [exact Hst|]. split; [exact Hrs|]. repeat split; reflexivity.
 - cbn [loop_linear] in H.
   destruct (fetch prog (fst pos) (snd pos)) as [i|] eqn:Hfetch; [|discriminate].
   destruct (wf_instr F i && loop_protected counter stride i) eqn:Hcheck; [|discriminate].
   destruct (loop_next prog pos i) as [next|] eqn:Hnext; [|discriminate].
   apply andb_true_iff in Hcheck. destruct Hcheck as [Hwf Hprotected].
   unfold loop_protected in Hprotected. rewrite !andb_true_iff in Hprotected.
   destruct Hprotected as [Hcounter Hstride].
   assert (Hf : fetch prog (blk s) (idx s)=Some i) by (rewrite Hpos in Hfetch; exact Hfetch).
   assert (Hs : exists s1, phase_step F prog E t s=Some s1).
   { unfold phase_step,step. rewrite Hf. eexists; reflexivity. }
   destruct Hs as [s1 Hstep].
   destruct (transfer_sound F prog E HE t s s1 st i Hf Hstep Hst Hrs Hwf) as [Hout Hrs1].
   assert (Hpc : pc_of s1=next) by (eapply loop_next_sound; [exact Hf|exact Hstep|rewrite <-Hpos;exact Hnext]).
   destruct (IH F prog E counter stride next (transfer F st i) out os (S t) s1 HE Hcounter_cell Hstride_cell H (eq_sym Hpc) Hout Hrs1)
     as [s' [Hrun [Hend [Hstate [Hregs [Hc Hs]]]]]].
   exists s'. split; [cbn [phase_run]; rewrite Hstep;exact Hrun|].
   split; [exact Hend|]. split; [replace (S n+t)%nat with (n+S t)%nat by lia;exact Hstate|].
   split; [exact Hregs|]. repeat split; rewrite ?Hc,?Hs;
   eapply loop_step_preserves; eassumption.
Qed.

Definition loop_instr_eq_dec (x y : instr) : {x=y}+{x<>y}.
Proof. repeat decide equality. Defined.
Definition loop_fetch_is (prog : program) (pos : pc) (i : instr) : bool :=
 match fetch prog (fst pos) (snd pos) with
 | Some j => if loop_instr_eq_dec j i then true else false
 | None => false end.
Lemma loop_fetch_is_sound : forall prog pos i,
 loop_fetch_is prog pos i=true -> fetch prog (fst pos) (snd pos)=Some i.
Proof.
 intros prog pos i H. unfold loop_fetch_is in H.
 destruct (fetch prog (fst pos) (snd pos)) as [j|] eqn:Hf; [|discriminate].
 destruct (loop_instr_eq_dec j i); [subst;reflexivity|discriminate].
Qed.

Record loop_info := { li_prefix : nat; li_suffix : nat; li_counter : reg;
 li_source : reg; li_stride : reg; li_predicate : reg; li_negated : bool }.
Definition loop_counter_step (d : loop_info) :=
 IBin None BAdd 64 (li_counter d) (li_source d) (OReg (li_stride d)).
Definition loop_branch (d : loop_info) (head : pc) :=
 IBra (Some (li_negated d,li_predicate d)) (fst head).
Definition loop_alias (d : loop_info) :=
 FForm (fatom 64 (li_counter d,EX) 1) 0 None.
Definition loop_source_equal (F : lfacts) (st : astate) (d : loop_info) :=
 if Pos.eqb (li_source d) (li_counter d) then true
 else register_fact_le F st (li_source d) (loop_alias d).
Definition loop_increment_ok (F : lfacts) (prog : program) (pos : pc) (st : astate) (d : loop_info) :=
 loop_fetch_is prog pos (loop_counter_step d) && wf_instr F (loop_counter_step d) &&
 Nat.eqb (wdn F (li_counter d)) 64 && Nat.eqb (wdn F (li_source d)) 64 &&
 Nat.eqb (wdn F (li_stride d)) 64 && negb (Pos.eqb (li_counter d) (li_stride d)) &&
 loop_source_equal F st d &&
 match reg_range F st (li_counter d),reg_range F st (li_stride d) with
 | Some (cl,ch),Some (sl,sh)=>(0 <=? cl) && (0 <? sl) && (ch+sh <? 2^64)
 | _,_=>false end.
Definition loop_branch_ok (F : lfacts) (prog : program) (head pos : pc) (d : loop_info) :=
 Nat.eqb (snd head) 0 && loop_fetch_is prog pos (loop_branch d head) && wf_instr F (loop_branch d head).
Definition loop_branches (F : lfacts) (st : astate) (d : loop_info) (head : pc) :=
 let after:=transfer F st (loop_branch d head) in
 (refine F after (li_predicate d) (negb (li_negated d)),
  refine F after (li_predicate d) (li_negated d)).
Definition loop_iteration (F : lfacts) (prog : program) (head : pc) (d : loop_info) (inv : astate)
 : option (pc * option astate * option astate) :=
 if is_cell F (li_counter d) || is_cell F (li_stride d) then None else
 match loop_linear (li_prefix d) F prog (li_counter d) (li_stride d) head inv with
 | None=>None | Some (pos,st)=>if loop_increment_ok F prog pos st d then
   let next:=next_pos prog (fst pos) (snd pos) in
   match loop_linear (li_suffix d) F prog (li_counter d) (li_stride d)
     next (transfer F st (loop_counter_step d)) with
   | None=>None | Some (jump,after)=>if loop_branch_ok F prog head jump d then
     let '(back,leave):=loop_branches F after d head in
     Some (next_pos prog (fst jump) (snd jump),back,leave)
     else None end else None end.
Definition loop_validate (F : lfacts) (prog : program) (head : pc) (d : loop_info) (inv : astate)
 : option (pc * option astate) :=
 match reg_range F inv (li_counter d),loop_iteration F prog head d inv with
 | Some _,Some (exit,back,leave)=>
   if match back with None=>true|Some st=>state_le F st inv end then Some (exit,leave) else None
 | _,_=>None end.

Lemma loop_alias_sound : forall F E t st rs d,
 env_ok F E -> state_ok F t st rs -> regs_ok F E rs ->
 wdn F (li_counter d)=64%nat -> wdn F (li_source d)=64%nat ->
 loop_source_equal F st d=true -> rs (li_source d)=rs (li_counter d).
Proof.
 intros F E t st rs d HE Hst Hrs Hc Ha H.
 unfold loop_source_equal in H. destruct (Pos.eqb (li_source d) (li_counter d)) eqn:Hequal.
 { apply Pos.eqb_eq in Hequal. rewrite Hequal. reflexivity. }
 pose proof (register_fact_le_sound F E HE t rs Hrs st (li_source d) (loop_alias d) Hst H) as Hfact.
 unfold loop_alias,fact_okv in Hfact. destruct Hfact as [_ [Hsat _]].
 unfold fatom in Hsat. apply sat_mk_inv in Hsat;[|lia]. destruct Hsat as [k [Hk Heq]].
 cbn [tsum atom_val] in Heq. assert (k=0) by lia. subst k.
 replace (1 * rs (li_counter d) + 0 + 0) with (rs (li_counter d)) in Heq by ring.
 pose proof (rs_range F E rs Hrs (li_counter d)) as Hcr.
 pose proof (rs_range F E rs Hrs (li_source d)) as Har.
 rewrite Hc in Hcr. rewrite Ha in Har.
 eapply eqm_small_eq;[|exact Har|exact Hcr|exact Heq]. lia.
Qed.

Lemma loop_increment_sound : forall F prog E pos st d t s,
 env_ok F E -> loop_increment_ok F prog pos st d=true -> pos=pc_of s ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 exists s', phase_step F prog E t s=Some s' /\
 pc_of s'=next_pos prog (fst pos) (snd pos) /\
 state_ok F (S t) (transfer F st (loop_counter_step d)) (regs s') /\ regs_ok F E (regs s') /\
 regs s' (li_counter d)=regs s (li_counter d)+regs s (li_stride d) /\
 regs s' (li_stride d)=regs s (li_stride d) /\ 0<regs s (li_stride d).
Proof.
 intros F prog E pos st d t s HE Hcheck Hpos Hst Hrs.
 unfold loop_increment_ok in Hcheck. rewrite !andb_true_iff in Hcheck.
 assert (Hf : fetch prog (fst pos) (snd pos)=Some (loop_counter_step d)) by (apply loop_fetch_is_sound;tauto).
 assert (Hwf : wf_instr F (loop_counter_step d)=true) by tauto.
 assert (Hc : wdn F (li_counter d)=64%nat) by (apply Nat.eqb_eq;tauto).
 assert (Ha : wdn F (li_source d)=64%nat) by (apply Nat.eqb_eq;tauto).
 assert (Hcs : Pos.eqb (li_counter d) (li_stride d)=false) by (apply negb_true_iff;tauto).
 assert (Halias : regs s (li_source d)=regs s (li_counter d)).
 { eapply loop_alias_sound;try eassumption;tauto. }
 destruct (reg_range F st (li_counter d)) as [[cl ch]|] eqn:Hcr;[|intuition discriminate].
 destruct (reg_range F st (li_stride d)) as [[sl sh]|] eqn:Hsr;[|intuition discriminate].
 rewrite !andb_true_iff,!Z.leb_le,!Z.ltb_lt in Hcheck.
 pose proof (reg_range_sound F E HE t st (regs s) Hst Hrs _ _ _ Hcr) as Hcb.
 pose proof (reg_range_sound F E HE t st (regs s) Hst Hrs _ _ _ Hsr) as Hsb.
 set (rs1:=upd (wdn F) (regs s) (li_counter d) (regs s (li_counter d)+regs s (li_stride d))).
 set (s1:=goto (next_pos prog (fst pos) (snd pos)) rs1).
 assert (Hstep : phase_step F prog E t s=Some s1).
 { unfold phase_step,step. rewrite Hpos in Hf. change (fetch prog (blk s) (idx s)=Some (loop_counter_step d)) in Hf.
   rewrite Hf. unfold loop_counter_step. cbn [guard_of holds negb dest result binop_eval opv].
   rewrite Halias. unfold s1,rs1. rewrite Hpos. reflexivity. }
 assert (Hvalue : rs1 (li_counter d)=regs s (li_counter d)+regs s (li_stride d)).
 { unfold rs1,upd. rewrite Pos.eqb_refl,Hc. apply trunc_small. cbn. tauto || lia. }
 assert (Hstride : rs1 (li_stride d)=regs s (li_stride d)).
 { unfold rs1,upd. rewrite Pos.eqb_sym,Hcs. reflexivity. }
 rewrite Hpos in Hf.
 destruct (transfer_sound F prog E HE t s s1 st (loop_counter_step d) Hf Hstep Hst Hrs Hwf) as [Hout Hregs].
 exists s1. split;[exact Hstep|]. split;[unfold s1,pc_of;cbn [goto blk idx];symmetry;apply surjective_pairing|]. split;[exact Hout|]. split;[exact Hregs|].
 split;[exact Hvalue|]. split;[exact Hstride|]. lia.
Qed.

Lemma loop_branch_sound : forall F prog E head pos st d t s,
 env_ok F E -> loop_branch_ok F prog head pos d=true -> pos=pc_of s ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 exists (s' : state) (out : astate) (taken : bool), phase_step F prog E t s=Some s' /\
 (if taken then fst (loop_branches F st d head) else snd (loop_branches F st d head))=Some out /\
 pc_of s'=(if taken then head else next_pos prog (fst pos) (snd pos)) /\
 state_ok F (S t) out (regs s') /\ regs_ok F E (regs s') /\ regs s'=regs s.
Proof.
 intros F prog E head pos st d t s HE Hcheck Hpos Hst Hrs.
 unfold loop_branch_ok in Hcheck. rewrite !andb_true_iff in Hcheck.
 assert (Hhead : snd head=0%nat) by (apply Nat.eqb_eq;tauto).
 assert (Hf : fetch prog (fst pos) (snd pos)=Some (loop_branch d head)) by (apply loop_fetch_is_sound;tauto).
 assert (Hwf : wf_instr F (loop_branch d head)=true) by tauto.
 set (truth:=negb (Z.eqb (regs s (li_predicate d)) 0)).
 set (taken:=if li_negated d then negb truth else truth).
 set (s1:=goto (if taken then head else next_pos prog (fst pos) (snd pos)) (regs s)).
 assert (Hstep : phase_step F prog E t s=Some s1).
 { unfold phase_step,step. rewrite Hpos in Hf. change (fetch prog (blk s) (idx s)=Some (loop_branch d head)) in Hf.
   rewrite Hf. unfold loop_branch. cbn [guard_of holds].
   unfold taken,truth,s1. destruct (li_negated d), (Z.eqb (regs s (li_predicate d)) 0);cbn;
   rewrite ?Hhead,?Hpos;try reflexivity;rewrite <-Hhead,(surjective_pairing head);reflexivity. }
 rewrite Hpos in Hf.
 destruct (transfer_sound F prog E HE t s s1 st (loop_branch d head) Hf Hstep Hst Hrs Hwf) as [Hout Hregs].
 destruct (refine_sound F E HE (S t) (regs s1) Hregs _ (li_predicate d) truth Hout eq_refl)
  as [out [Href Hgood]].
 exists s1,out,taken. split;[exact Hstep|]. split.
 - unfold loop_branches. unfold taken. destruct (li_negated d),truth;exact Href.
 - split;[unfold s1,pc_of;destruct taken;cbn [goto blk idx];symmetry;apply surjective_pairing|]. split;[exact Hgood|]. split;[exact Hregs|reflexivity].
Qed.

Lemma loop_run_app : forall F prog E n m t s,
 phase_run F prog E (n+m)%nat t s=
 match phase_run F prog E n t s with
 | None=>None | Some mid=>phase_run F prog E m (n+t)%nat mid end.
Proof.
 intros F prog E n. induction n as [|n IH];intros m t s;[reflexivity|].
 cbn [phase_run Nat.add]. destruct (phase_step F prog E t s);[|reflexivity].
 rewrite IH. replace (n+S t)%nat with (S (n+t)) by lia. reflexivity.
Qed.
Definition loop_length (d : loop_info) := (li_prefix d+1+li_suffix d+1)%nat.
Lemma loop_iteration_sound : forall F prog E head d inv exit back leave t s,
 env_ok F E -> loop_iteration F prog head d inv=Some (exit,back,leave) ->
 head=pc_of s -> state_ok F t inv (regs s) -> regs_ok F E (regs s) ->
 exists (s' : state) (out : astate) (taken : bool), phase_run F prog E (loop_length d) t s=Some s' /\
 (if taken then back else leave)=Some out /\ pc_of s'=(if taken then head else exit) /\
 state_ok F (loop_length d+t)%nat out (regs s') /\ regs_ok F E (regs s') /\
 regs s (li_counter d)<regs s' (li_counter d).
Proof.
 intros F prog E head d inv exit back leave t s HE Hiter Hpos Hst Hrs.
 unfold loop_iteration in Hiter.
 destruct (is_cell F (li_counter d) || is_cell F (li_stride d)) eqn:Hcells;[discriminate|].
 apply orb_false_iff in Hcells. destruct Hcells as [Hcounter_cell Hstride_cell].
 destruct (loop_linear (li_prefix d) F prog (li_counter d) (li_stride d) head inv)
 as [[pos st]|] eqn:Hprefix;[|discriminate].
 destruct (loop_increment_ok F prog pos st d) eqn:Hinc;[|discriminate].
 destruct (loop_linear (li_suffix d) F prog (li_counter d) (li_stride d)
 (next_pos prog (fst pos) (snd pos)) (transfer F st (loop_counter_step d)))
 as [[jump after]|] eqn:Hsuffix;[|discriminate].
 destruct (loop_branch_ok F prog head jump d) eqn:Hbranch;[|discriminate].
 destruct (loop_branches F after d head) as [b l] eqn:Hbranches. injection Hiter as <- <- <-.
 destruct (loop_linear_sound (li_prefix d) F prog E (li_counter d) (li_stride d)
  head inv pos st t s HE Hcounter_cell Hstride_cell Hprefix Hpos Hst Hrs)
 as [s1 [Hrun1 [Hpc1 [Hst1 [Hrs1 [Hc1 Hs1]]]]]].
 destruct (loop_increment_sound F prog E pos st d (li_prefix d+t)%nat s1 HE Hinc (eq_sym Hpc1) Hst1 Hrs1)
 as [s2 [Hstep2 [Hpc2 [Hst2 [Hrs2 [Hc2 [Hs2 Hpositive]]]]]]].
 destruct (loop_linear_sound (li_suffix d) F prog E (li_counter d) (li_stride d)
  (next_pos prog (fst pos) (snd pos)) (transfer F st (loop_counter_step d)) jump after
  (S (li_prefix d+t)) s2 HE Hcounter_cell Hstride_cell Hsuffix (eq_sym Hpc2) Hst2 Hrs2)
 as [s3 [Hrun3 [Hpc3 [Hst3 [Hrs3 [Hc3 Hs3]]]]]].
 destruct (loop_branch_sound F prog E head jump after d (li_suffix d+S (li_prefix d+t))%nat s3 HE Hbranch (eq_sym Hpc3) Hst3 Hrs3)
 as [s4 [out [taken [Hstep4 [Hout [Hpc4 [Hst4 [Hrs4 Hregs4]]]]]]]].
 exists s4,out,taken. split.
 - unfold loop_length. rewrite loop_run_app.
   assert (Hpre : phase_run F prog E (li_prefix d+1+li_suffix d)%nat t s=Some s3).
   { rewrite loop_run_app. assert (Hi : phase_run F prog E (li_prefix d+1)%nat t s=Some s2).
     { rewrite loop_run_app,Hrun1. cbn [phase_run]. rewrite Hstep2. reflexivity. }
     rewrite Hi. replace (li_prefix d+1+t)%nat with (S (li_prefix d+t)) by lia. exact Hrun3. }
   rewrite Hpre. cbn [phase_run].
   replace (li_prefix d+1+li_suffix d+t)%nat with (li_suffix d+S (li_prefix d+t))%nat by lia.
   rewrite Hstep4. reflexivity.
 - rewrite Hbranches in Hout. cbn [fst snd] in Hout. split;[exact Hout|].
   split;[exact Hpc4|]. split.
   + unfold loop_length. replace (li_prefix d+1+li_suffix d+1+t)%nat with (S (li_suffix d+S (li_prefix d+t))) by lia. exact Hst4.
   + split;[exact Hrs4|]. rewrite Hregs4,Hc3,Hc2,Hc1. lia.
Qed.

Definition loop_rank (upper : Z) (d : loop_info) (rs : reg -> Z) : nat :=
 Z.to_nat (Z.max 0 (upper-rs (li_counter d))).
Lemma loop_rank_decreases : forall upper d rs rs',
 rs (li_counter d)<rs' (li_counter d) -> rs' (li_counter d)<=upper ->
 (loop_rank upper d rs'<loop_rank upper d rs)%nat.
Proof.
 intros upper d rs rs' Hinc Hbound. unfold loop_rank. rewrite !Z.max_r by lia.
 apply Z2Nat.inj_lt;lia.
Qed.
Theorem loop_validate_progress : forall F prog E head d inv exit ost,
 env_ok F E -> loop_validate F prog head d inv=Some (exit,ost) ->
 forall t s, head=pc_of s -> state_ok F t inv (regs s) -> regs_ok F E (regs s) ->
 exists n s' out, phase_run F prog E n t s=Some s' /\
 pc_of s'=exit /\ ost=Some out /\ state_ok F (n+t)%nat out (regs s') /\ regs_ok F E (regs s').
Proof.
 intros F prog E head d inv exit ost HE Hval. unfold loop_validate in Hval.
 destruct (reg_range F inv (li_counter d)) as [[lo hi]|] eqn:Hbound;[|discriminate].
 destruct (loop_iteration F prog head d inv) as [[[next back] leave]|] eqn:Hiter;[|discriminate].
 destruct (match back with None=>true|Some st=>state_le F st inv end) eqn:Hback;[|discriminate].
 injection Hval as <- <-.
 assert (Hind : forall rank t s,
 loop_rank hi d (regs s)=rank -> head=pc_of s -> state_ok F t inv (regs s) -> regs_ok F E (regs s) ->
 exists n s' out, phase_run F prog E n t s=Some s' /\
 pc_of s'=next /\ leave=Some out /\ state_ok F (n+t)%nat out (regs s') /\ regs_ok F E (regs s')).
 { intro rank. induction rank using lt_wf_ind. intros t s Hrank Hpos Hst Hrs.
   destruct (loop_iteration_sound F prog E head d inv next back leave t s HE Hiter Hpos Hst Hrs)
    as [s1 [out [taken [Hrun [Hout [Hpc [Hst1 [Hrs1 Hinc]]]]]]]].
   destruct taken.
   - destruct back as [back|];[|discriminate]. injection Hout as <-.
     assert (Hinv : state_ok F (loop_length d+t)%nat inv (regs s1)).
     { eapply state_le_sound;[exact HE|exact Hrs1|exact Hst1|exact Hback]. }
     pose proof (reg_range_sound F E HE (loop_length d+t)%nat inv (regs s1) Hinv Hrs1 _ _ _ Hbound) as Hb.
     assert (Hdec : (loop_rank hi d (regs s1)<rank)%nat).
     { rewrite <-Hrank. apply loop_rank_decreases;[exact Hinc|lia]. }
     destruct (H _ Hdec (loop_length d+t)%nat s1 eq_refl (eq_sym Hpc) Hinv Hrs1)
      as [n [last [output [Hlast [Hlastpc [Hleave [Hfinal Hregsfinal]]]]]]].
     exists (loop_length d+n)%nat,last,output. split;[rewrite loop_run_app,Hrun;exact Hlast|].
     split;[exact Hlastpc|]. split;[exact Hleave|]. split;[|exact Hregsfinal].
     replace (loop_length d+n+t)%nat with (n+(loop_length d+t))%nat by lia. exact Hfinal.
   - exists (loop_length d),s1,out. split;[exact Hrun|]. split;[exact Hpc|]. split;[exact Hout|]. split;assumption.
 }
 intros t s Hpos Hst Hrs. exact (Hind (loop_rank hi d (regs s)) t s eq_refl Hpos Hst Hrs).
Qed.

(* Descriptor discovery proposes the last register-stride addition on a
   straight-line self-backedge trace.  Validation checks this one proposal;
   failure never retries another descriptor or skips an obligation. *)
Fixpoint loop_discover (fuel prefix : nat) (prog : program) (head pos : pc)
 (candidate : option loop_info) : option loop_info :=
 match fuel with
 | O=>None
 | S rest=>match fetch prog (fst pos) (snd pos) with
   | None=>None
   | Some (IBra (Some (neg,p)) target)=>
     if Nat.eqb target (fst head) && Nat.eqb (snd head) 0 then
       match candidate with None=>None|Some d=>
       Some {|li_prefix:=li_prefix d;li_suffix:=(prefix-li_prefix d-1)%nat;
         li_counter:=li_counter d;li_source:=li_source d;li_stride:=li_stride d;
         li_predicate:=p;li_negated:=neg|} end else None
   | Some i=>match loop_next prog pos i with
     | None=>None|Some next=>
       let candidate:=match i with
       | IBin None BAdd 64 c a (OReg stride)=>
         Some {|li_prefix:=prefix;li_suffix:=0;li_counter:=c;li_source:=a;
          li_stride:=stride;li_predicate:=c;li_negated:=false|}
       | _=>candidate end in
       loop_discover rest (S prefix) prog head next candidate
     end end end.
Definition loop_seek (fuel prefix : nat) (prog : program) (head pos : pc) :=
 loop_discover fuel prefix prog head pos None.
Definition loop_at (prog : program) (pos : pc) :=
 if Nat.eqb (snd pos) 0 then
 loop_seek (List.length (List.concat prog)) 0 prog pos pos else None.
Fixpoint loop_trace (n : nat) (prog : program) (pos : pc) : list instr :=
 match n with O=>[]|S n=>match fetch prog (fst pos) (snd pos) with
 | None=>[]|Some i=>i::match loop_next prog pos i with
   | None=>[]|Some next=>loop_trace n prog next end end end.
Definition loop_propose (F : lfacts) (prog : program) (head : pc) (d : loop_info) (st : astate) : option astate :=
 match reg_range F st (li_stride d) with
 | Some (_,sh)=>
   let body:=loop_trace (loop_length d) prog head in
   let forgotten:=fold_left (fun s r=>assign F s r None None) (flat_map loop_writes body) st in
   Some {|amap:=PM.add (li_counter d) (FForm (mk 64 [] 0 (2^63-1-sh)) 0 None) (amap forgotten);
          acons:=acons forgotten|}
 | None=>None end.
Lemma loop_run_reaches : forall F prog E n event t s last,
 phase_run F prog E n t s=Some last -> phase_reaches F prog E event (n+t)%nat last ->
 phase_reaches F prog E event t s.
Proof.
 intros F prog E n. induction n as [|n IH];intros event t s last Hr He.
 - cbn [phase_run] in Hr. injection Hr as <-. exact He.
 - destruct (event (pc_of s)) eqn:Here;[apply Phase_here;exact Here|].
   cbn [phase_run] in Hr. destruct (phase_step F prog E t s) as [next|] eqn:Hs;[|discriminate].
   eapply Phase_next;[exact Here|exact Hs|]. eapply IH;[exact Hr|].
   replace (n+S t)%nat with (S n+t)%nat by lia. exact He.
Qed.

Fixpoint ranked_phase_all (keep : PM.t unit) (fuel : nat) (F : lfacts) (prog : program)
 (event : pc -> bool) (pending : list (pc * option astate)) : bool :=
 match pending with
 | []=>true
 | (pos,ost)::tail=>match fuel with O=>false|S rest=>match ost with
   | None=>ranked_phase_all keep rest F prog event tail
   | Some st=>if is_bottom st then ranked_phase_all keep rest F prog event tail else
     if event pos then ranked_phase_all keep rest F prog event tail else
     match loop_at prog pos with
     | Some d=>match loop_propose F prog pos d st with
       | None=>false
       | Some inv=>if state_le F st inv then match loop_validate F prog pos d inv with
         | None=>false | Some edge=>ranked_phase_all keep rest F prog event (edge::tail) end else false
       end
     | None=>match fetch prog (fst pos) (snd pos) with
       | None=>false|Some i=>if wf_instr F i then
         ranked_phase_all keep rest F prog event (phase_successors keep F prog pos st i++tail)
         else false end
     end end end end.

Theorem ranked_phase_all_progress : forall F prog E fuel keep event pending pos st t s,
 env_ok F E -> ranked_phase_all keep fuel F prog event pending=true ->
 In (pos,Some st) pending -> pos=pc_of s -> state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 phase_reaches F prog E event t s.
Proof.
 intros F prog E fuel. induction fuel as [|fuel IH];intros keep event pending pos st t s HE Hcheck Hin Hpos Hst Hrs;
 destruct pending as [|[q ost] tail];[contradiction|discriminate|contradiction|].
 cbn [ranked_phase_all] in Hcheck. destruct ost as [current|].
 - destruct (is_bottom current) eqn:Hbottom.
   + destruct Hin as [Heq|Hin].
     * injection Heq as <- <-. exfalso. exact (bottom_not_ok F t current _ Hbottom Hst).
     * eapply IH;eassumption.
   + destruct (event q) eqn:Hevent.
     * destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. apply Phase_here. rewrite <-Hpos. exact Hevent.
       -- eapply IH;eassumption.
     * destruct (loop_at prog q) as [d|] eqn:Hloop.
       -- destruct (loop_propose F prog q d current) as [inv|];[|discriminate].
          destruct (state_le F current inv) eqn:Hle;[|discriminate].
          destruct (loop_validate F prog q d inv) as [[exit os]|] eqn:Hval;[|discriminate].
          destruct Hin as [Heq|Hin].
          ++ injection Heq as <- <-.
             assert (Hinv : state_ok F t inv (regs s)) by (eapply state_le_sound;eassumption).
             destruct (loop_validate_progress F prog E q d inv exit os HE Hval t s Hpos Hinv Hrs)
               as [n [last [out [Hrun [Hpc [Hos [Hout Hregs]]]]]]].
             eapply loop_run_reaches;[exact Hrun|]. eapply IH;[exact HE|exact Hcheck| |symmetry;exact Hpc|exact Hout|exact Hregs].
             rewrite Hos. left;reflexivity.
          ++ eapply IH;[exact HE|exact Hcheck|right;exact Hin|exact Hpos|exact Hst|exact Hrs].
       -- destruct (fetch prog (fst q) (snd q)) as [i|] eqn:Hfetch;[|discriminate].
          destruct (wf_instr F i) eqn:Hwf;[|discriminate].
          destruct Hin as [Heq|Hin].
          ++ injection Heq as <- <-. subst q.
             assert (Hf : fetch prog (blk s) (idx s)=Some i) by exact Hfetch.
             assert (Hex : exists s',phase_step F prog E t s=Some s').
             { unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
             destruct Hex as [s' Hstep].
             destruct (phase_successors_complete F prog E HE keep t s s' current i Hf Hstep Hst Hrs Hwf)
               as [out [Hchild [Hout Hregs]]].
             eapply Phase_next;[exact Hevent|exact Hstep|].
             eapply IH;[exact HE|exact Hcheck| |reflexivity|exact Hout|exact Hregs].
             apply in_or_app;left;exact Hchild.
          ++ eapply IH;[exact HE|exact Hcheck| |exact Hpos|exact Hst|exact Hrs]. apply in_or_app;right;exact Hin.
 - destruct Hin as [Heq|Hin];[discriminate|]. eapply IH;eassumption.
Qed.

(* Decide cheap structural prerequisites before evaluating phase progress. *)
Definition ranked_phase_ready (F : lfacts) (prog : program) (event : pc) : bool :=
 if lf_ok F then if phase_event_at prog event then phase_closed prog event else false else false.
Definition ranked_phase_progress (F : lfacts) (prog : program) (event : pc) : bool :=
 ranked_phase_all (phase_controls prog event) phase_fuel F prog (phase_event event)
 [((0%nat,0%nat),Some (empty_state F))].
Definition ranked_phase_check (F : lfacts) (prog : program) (event : pc) : bool :=
 if ranked_phase_ready F prog event then ranked_phase_progress F prog event else false.
Theorem ranked_phase_check_exact : forall F prog event,
 ranked_phase_check F prog event =
 (lf_ok F && phase_event_at prog event && phase_closed prog event &&
  ranked_phase_all (phase_controls prog event) phase_fuel F prog (phase_event event)
   [((0%nat,0%nat),Some (empty_state F))]).
Proof.
 intros F prog event. unfold ranked_phase_check,ranked_phase_ready,ranked_phase_progress.
 destruct (lf_ok F), (phase_event_at prog event), (phase_closed prog event); reflexivity.
Qed.

Theorem ranked_phase_check_sound : forall F prog event,
 ranked_phase_check F prog event=true -> single_phase_safe F prog event.
Proof.
 intros F prog event H. rewrite ranked_phase_check_exact in H. rewrite !andb_true_iff in H.
 destruct H as [[[Hlf He] Hclosed] Hcheck].
 split;[exact He|]. split;[exact Hclosed|]. intros E Hlaunch.
 eapply ranked_phase_all_progress.
 - exact (env_of_launch F prog E Hlf Hlaunch).
 - exact Hcheck.
 - left;reflexivity.
 - reflexivity.
 - split.
   + intros r f Hf. unfold get,empty_state in Hf;cbn [amap] in Hf. rewrite PM.gempty in Hf. discriminate.
   + exact (lk_derived F E Hlaunch).
 - split;[exact (lk_init F E Hlaunch)|]. intros r Hfixed;reflexivity.
Qed.

(* A one-block trace has a structural progress proof when all instructions
   fall through to one final unconditional exit. Data guards do not change
   this control property; all memory and value obligations remain checked
   by the ordinary instruction checker. *)
Fixpoint straight_event_body (body : list instr) : bool :=
 match body with
 | [] => false
 | i :: tail => if is_jump i then
     match i,tail with IExit None,[] => true | _,_ => false end
   else straight_event_body tail
 end.
Definition straight_event_program (prog : program) : bool :=
 match prog with [body] => straight_event_body body | _ => false end.

Lemma straight_event_body_shape : forall body,
 straight_event_body body=true ->
 exists prefix,body=prefix++[IExit None] /\
 Forall (fun i=>is_jump i=false) prefix.
Proof.
 induction body as [|i tail IH];intros H;[discriminate|].
 cbn [straight_event_body] in H.
 destruct (is_jump i) eqn:Hjump.
 - destruct i;try discriminate. destruct g;[discriminate|].
   destruct tail;[|discriminate]. exists [];split;[reflexivity|constructor].
 - destruct (IH H) as [prefix [Heq Hprefix]].
   exists (i::prefix). split;[cbn;rewrite Heq;reflexivity|constructor;assumption].
Qed.

Lemma straight_event_no_branch : forall prefix event,
 Forall (fun i=>is_jump i=false) prefix ->
 phase_closed [prefix++[IExit None]] event=true.
Proof.
 intros prefix event Hprefix. rewrite phase_closed_exact.
 rewrite short_all_same,forallb_forall. intros b Hb.
 rewrite short_all_same,forallb_forall. intros n Hn.
 destruct (phase_after event (b,n));[|reflexivity].
 destruct (fetch [prefix++[IExit None]] b n) as [i|] eqn:Hfetch;[|reflexivity].
 destruct i;try reflexivity.
 exfalso. unfold fetch,block_body in Hfetch.
 destruct b;[|destruct b,n;discriminate]. cbn [nth_error] in Hfetch.
 apply nth_error_In in Hfetch. apply in_app_or in Hfetch.
 destruct Hfetch as [Hin|[Heq|[]]];[|discriminate].
 rewrite Forall_forall in Hprefix. specialize (Hprefix _ Hin). discriminate.
Qed.

Lemma straight_event_step : forall F E prefix n t s,
 Forall (fun i=>is_jump i=false) prefix ->
 (n<List.length prefix)%nat -> pc_of s=(0%nat,n) ->
 exists s',phase_step F [prefix++[IExit None]] E t s=Some s' /\
 pc_of s'=(0%nat,S n).
Proof.
 intros F E prefix n t s Hprefix Hn Hpos.
 assert (Hex : exists i,nth_error prefix n=Some i).
 { destruct (nth_error prefix n) as [i|] eqn:Hnth;[eauto|].
   apply nth_error_None in Hnth. lia. }
 destruct Hex as [i Hnth].
 assert (Hjump : is_jump i=false).
 { rewrite Forall_forall in Hprefix. apply Hprefix. eapply nth_error_In;exact Hnth. }
 assert (Hfetch : fetch [prefix++[IExit None]] (blk s) (idx s)=Some i).
 { unfold pc_of in Hpos. injection Hpos as Hb Hi. rewrite Hb,Hi.
   unfold fetch,block_body. cbn [nth_error]. rewrite nth_error_app1 by exact Hn. exact Hnth. }
 assert (Hnext : next_pos [prefix++[IExit None]] (blk s) (idx s)=(0%nat,S n)).
 { unfold pc_of in Hpos. injection Hpos as Hb Hi. rewrite Hb,Hi.
   unfold next_pos,block_len,block_body. cbn [nth_error]. rewrite app_length.
   cbn [List.length]. assert (Hlt : (S n <? List.length prefix + 1)%nat=true) by (apply Nat.ltb_lt;lia). rewrite Hlt. reflexivity. }
 assert (Hstep : exists s',phase_step F [prefix++[IExit None]] E t s=Some s').
 { unfold phase_step,step. rewrite Hfetch. eexists;reflexivity. }
 destruct Hstep as [s' Hstep]. exists s'. split;[exact Hstep|].
 pose proof (step_pos F [prefix++[IExit None]] E t s s' i Hfetch Hstep) as Hp.
 destruct Hp as [[Hp _]|[[g [target [Heq _]]]|[g [Heq _]]]].
 - unfold pc_of. rewrite Hp. exact Hnext.
 - subst i. discriminate.
 - subst i. discriminate.
Qed.

Lemma straight_event_run : forall F E prefix n,
 Forall (fun i=>is_jump i=false) prefix -> (n<=List.length prefix)%nat ->
 exists s,phase_run F [prefix++[IExit None]] E n 0 (phase_initial E)=Some s /\
 pc_of s=(0%nat,n).
Proof.
 intros F E prefix n. induction n as [|n IH];intros Hprefix Hn.
 - exists (phase_initial E). split;reflexivity.
 - destruct (IH Hprefix ltac:(lia)) as [mid [Hrun Hpos]].
   destruct (straight_event_step F E prefix n n mid Hprefix ltac:(lia) Hpos)
    as [last [Hstep Hlast]]. exists last. split;[|exact Hlast].
   replace (S n) with (n+1)%nat by lia. rewrite loop_run_app,Hrun.
   cbn [phase_run]. rewrite Nat.add_0_r,Hstep. reflexivity.
Qed.

Lemma straight_event_safe : forall F prog event,
 straight_event_program prog=true -> phase_event_at prog event=true ->
 single_phase_safe F prog event.
Proof.
 intros F prog event Hstraight Hevent.
 destruct prog as [|body [|other tail]];try discriminate.
 cbn [straight_event_program] in Hstraight.
 destruct (straight_event_body_shape body Hstraight) as [prefix [-> Hprefix]].
 split;[exact Hevent|]. split;[apply straight_event_no_branch;exact Hprefix|].
 intros E Hlaunch.
 destruct (phase_event_at_fetch _ _ Hevent) as [i [Hfetch Hmarker]].
 destruct (phase_fetch_bounds _ _ _ _ Hfetch) as [Hb Hi].
 destruct event as [b n]. cbn [fst snd List.length] in Hb,Hi,Hfetch.
 assert (Hb0 : b=0%nat) by lia. subst b.
 unfold block_len,block_body in Hi. cbn [nth_error] in Hi.
 rewrite app_length in Hi. cbn [List.length] in Hi.
 destruct (straight_event_run F E prefix n Hprefix ltac:(lia)) as [last [Hrun Hpos]].
 eapply loop_run_reaches;[exact Hrun|].
 apply Phase_here. rewrite Hpos. apply phase_event_eq. reflexivity.
Qed.

(* Every checked event is reached by every modeled launched thread. Its
   no-return property fixes the order in which any two distinct events can
   occur; the check requires that same strict order for the source list. *)
Fixpoint loop_events_ordered (events : list pc) : bool :=
 match events with []=>true|e::tail=>short_all (phase_after e) tail && loop_events_ordered tail end.
(* Adjacent comparisons suffice for the same strict lexicographic order.
   Retain the complete pairwise predicate in the public safety statement. *)
Fixpoint loop_events_after (previous:pc) (events:list pc) : bool :=
 match events with
 | []=>true
 | event::tail=>if phase_after previous event then loop_events_after event tail else false
 end.
Definition loop_events_adjacent (events:list pc) : bool :=
 match events with []=>true|first::tail=>loop_events_after first tail end.
Lemma loop_phase_after_transitive : forall a b c,
 phase_after a b=true -> phase_after b c=true -> phase_after a c=true.
Proof.
 intros a b c Hab Hbc. apply phase_after_iff in Hab,Hbc.
 apply phase_after_iff. lia.
Qed.
Lemma loop_events_after_exact : forall events first,
 loop_events_after first events =
 short_all (phase_after first) events && loop_events_ordered events.
Proof.
 induction events as [|next tail IH];intros first;[reflexivity|].
 cbn [loop_events_after short_all loop_events_ordered]. rewrite IH.
 destruct (phase_after first next) eqn:Hnext;cbn;[|reflexivity].
 destruct (short_all (phase_after next) tail) eqn:Htail;cbn.
 - assert (Hfirst:short_all (phase_after first) tail=true).
   { rewrite short_all_same,forallb_forall in Htail |- *.
     intros last Hin. eapply loop_phase_after_transitive;[exact Hnext|exact (Htail last Hin)]. }
   rewrite Hfirst. reflexivity.
 - destruct (short_all (phase_after first) tail);reflexivity.
Qed.
Theorem loop_events_adjacent_exact : forall events,
 loop_events_adjacent events=loop_events_ordered events.
Proof.
 intros [|first tail];[reflexivity|]. cbn [loop_events_adjacent loop_events_ordered].
 apply loop_events_after_exact.
Qed.

(* Hoist all structural obligations together: no expensive phase is useful
   when another event already fails its necessary structural premise. *)
Definition loop_events_check (F : lfacts) (prog : program) (events : list pc) : bool :=
 if loop_events_adjacent events then
   if straight_event_program prog then
     if lf_ok F then short_all (phase_event_at prog) events else false
   else if short_all (ranked_phase_ready F prog) events then
     short_all (ranked_phase_progress F prog) events else false
 else false.

Lemma short_all_prerequisites : forall A (p q:A->bool) xs,
 (if short_all p xs then short_all q xs else false) =
 short_all (fun x=>p x && q x) xs.
Proof.
 intros A p q xs. induction xs as [|x xs IH]; [reflexivity|].
 cbn [short_all]. destruct (p x), (q x); cbn; try reflexivity.
 - exact IH.
 - destruct (short_all p xs); reflexivity.
Qed.
Theorem loop_events_check_exact : forall F prog events,
 loop_events_check F prog events =
 (loop_events_ordered events &&
  if straight_event_program prog then
    lf_ok F && short_all (phase_event_at prog) events
  else short_all (ranked_phase_check F prog) events).
Proof.
 intros F prog events. unfold loop_events_check.
 rewrite loop_events_adjacent_exact.
 destruct (loop_events_ordered events); [|reflexivity].
 destruct (straight_event_program prog); cbv beta iota delta [andb].
 - destruct (lf_ok F); reflexivity.
 - rewrite short_all_prerequisites.
   induction events as [|e events IH]; [reflexivity|].
   cbn [short_all]. unfold ranked_phase_check at 1.
   destruct (ranked_phase_ready F prog e); cbv beta iota delta [andb]; [|reflexivity].
   destruct (ranked_phase_progress F prog e); [exact IH|reflexivity].
Qed.

Lemma loop_reaches_run : forall F prog E event t s,
 phase_reaches F prog E (phase_event event) t s ->
 exists n last,phase_run F prog E n t s=Some last /\ pc_of last=event.
Proof.
 intros F prog E event t s H. induction H as [t s He|t s s' He Hs Hr IH].
 - exists 0%nat,s. split;[reflexivity|]. apply phase_event_eq;exact He.
 - destruct IH as [n [last [Hrun Hpc]]]. exists (S n),last. split;[|exact Hpc].
   cbn [phase_run]. rewrite Hs. exact Hrun.
Qed.

Lemma loop_run_after_event : forall F prog E n event t s last,
 phase_event_at prog event=true -> phase_closed prog event=true ->
 pc_of s=event -> phase_run F prog E (S n) t s=Some last ->
 phase_after event (pc_of last)=true.
Proof.
 intros F prog E n event t s last Hf Hc Hp Hr.
 cbn [phase_run] in Hr. destruct (phase_step F prog E t s) as [next|] eqn:Hs;[|discriminate].
 eapply phase_stays_after;[exact Hf|exact Hc| |exact Hr].
 eapply phase_event_step;eassumption.
Qed.

Lemma loop_hit_order : forall F prog E e1 e2 n1 n2 t s s1 s2,
 phase_event_at prog e2=true -> phase_closed prog e2=true ->
 phase_after e1 e2=true ->
 phase_run F prog E n1 t s=Some s1 -> pc_of s1=e1 ->
 phase_run F prog E n2 t s=Some s2 -> pc_of s2=e2 -> (n1<n2)%nat.
Proof.
 intros F prog E e1 e2 n1 n2 t s s1 s2 Hf Hc Horder H1 Hp1 H2 Hp2.
 destruct (Nat.lt_trichotomy n1 n2) as [Hlt|[Heq|Hgt]];[exact Hlt| |].
 - subst n2. rewrite H1 in H2. injection H2 as <-. rewrite <-Hp1,<-Hp2,phase_after_irrefl in Horder. discriminate.
 - assert (Hex : exists k,n1=(n2+S k)%nat) by (exists (n1-n2-1)%nat;lia).
   destruct Hex as [k ->]. rewrite loop_run_app,H2 in H1.
   pose proof (loop_run_after_event F prog E k e2 (n2+t)%nat s2 s1 Hf Hc Hp2 H1) as Ha.
   rewrite Hp1 in Ha. apply phase_after_iff in Ha,Horder. lia.
Qed.

Lemma loop_hit_unique : forall F prog E event n1 n2 t s s1 s2,
 phase_event_at prog event=true -> phase_closed prog event=true ->
 phase_run F prog E n1 t s=Some s1 -> pc_of s1=event ->
 phase_run F prog E n2 t s=Some s2 -> pc_of s2=event -> n1=n2.
Proof.
 intros F prog E event n1 n2 t s s1 s2 Hf Hc H1 Hp1 H2 Hp2.
 assert (Hnot : forall a b first last,
 (a<b)%nat -> phase_run F prog E a t s=Some first -> pc_of first=event ->
 phase_run F prog E b t s=Some last -> pc_of last=event -> False).
 { intros a b first last Hlt Ha Hpa Hb Hpb.
   assert (Hex : exists k,b=(a+S k)%nat) by (exists (b-a-1)%nat;lia).
   destruct Hex as [k ->]. rewrite loop_run_app,Ha in Hb.
   pose proof (loop_run_after_event F prog E k event (a+t)%nat first last Hf Hc Hpa Hb) as Hafter.
   rewrite Hpb,phase_after_irrefl in Hafter. discriminate. }
 destruct (Nat.lt_trichotomy n1 n2) as [Hlt|[Heq|Hgt]];[|exact Heq|];exfalso;eapply Hnot;eassumption.
Qed.

Definition loop_events_safe (F : lfacts) (prog : program) (events : list pc) : Prop :=
 loop_events_ordered events=true /\ Forall (single_phase_safe F prog) events /\
 forall E,launch_ok F E ->
 (forall event,In event events -> exists n s,
   phase_run F prog E n 0 (phase_initial E)=Some s /\ pc_of s=event) /\
 (forall e1 e2 n1 n2 s1 s2,In e1 events -> In e2 events -> phase_after e1 e2=true ->
   phase_run F prog E n1 0 (phase_initial E)=Some s1 -> pc_of s1=e1 ->
   phase_run F prog E n2 0 (phase_initial E)=Some s2 -> pc_of s2=e2 -> (n1<n2)%nat) /\
 (forall event n1 n2 s1 s2,In event events ->
   phase_run F prog E n1 0 (phase_initial E)=Some s1 -> pc_of s1=event ->
   phase_run F prog E n2 0 (phase_initial E)=Some s2 -> pc_of s2=event -> n1=n2).

Theorem loop_events_check_sound : forall F prog events,
 loop_events_check F prog events=true -> loop_events_safe F prog events.
Proof.
 intros F prog events H. rewrite loop_events_check_exact in H. apply andb_true_iff in H.
 destruct H as [Horder Hchecks].
 assert (Hsafe : forall event,In event events -> single_phase_safe F prog event).
 { destruct (straight_event_program prog) eqn:Hstraight.
   - first [ discriminate
     | apply andb_true_iff in Hchecks; destruct Hchecks as [_ Hchecks];
       rewrite short_all_same,forallb_forall in Hchecks;
       intros event Hin; apply straight_event_safe;
       [exact Hstraight|exact (Hchecks event Hin)] ].
   - rewrite short_all_same,forallb_forall in Hchecks.
     intros event Hin. apply ranked_phase_check_sound. exact (Hchecks event Hin). }
 split;[exact Horder|]. split;[apply Forall_forall;exact Hsafe|].
 intros E Hlaunch. split.
 - intros event Hin. apply loop_reaches_run. exact (proj2 (proj2 (Hsafe event Hin)) E Hlaunch).
 - split.
   + intros e1 e2 n1 n2 s1 s2 H1 H2 Hord Hr1 Hp1 Hr2 Hp2.
     destruct (Hsafe e2 H2) as [Hf [Hc _]]. eapply loop_hit_order;eassumption.
   + intros event n1 n2 s1 s2 Hin Hr1 Hp1 Hr2 Hp2.
     destruct (Hsafe event Hin) as [Hf [Hc _]]. eapply loop_hit_unique;eassumption.
Qed.

Lemma loop_events_ordered_at : forall events i j e1 e2,
 loop_events_ordered events=true -> nth_error events i=Some e1 -> nth_error events j=Some e2 ->
 (i<j)%nat -> phase_after e1 e2=true.
Proof.
 induction events as [|e tail IH];intros i j e1 e2 Horder H1 H2 Hij.
 - destruct i;discriminate.
 - cbn [loop_events_ordered] in Horder. apply andb_true_iff in Horder.
   destruct Horder as [Hhead Htail]. destruct i,j;cbn [nth_error] in H1,H2;try lia.
   + injection H1 as <-. rewrite short_all_same,forallb_forall in Hhead.
     apply Hhead. apply nth_error_In in H2. exact H2.
   + eapply IH;eassumption || lia.
Qed.
Corollary loop_events_in_order : forall F prog events E i j e1 e2 n1 n2 s1 s2,
 loop_events_safe F prog events -> launch_ok F E ->
 nth_error events i=Some e1 -> nth_error events j=Some e2 -> (i<j)%nat ->
 phase_run F prog E n1 0 (phase_initial E)=Some s1 -> pc_of s1=e1 ->
 phase_run F prog E n2 0 (phase_initial E)=Some s2 -> pc_of s2=e2 -> (n1<n2)%nat.
Proof.
 intros F prog events E i j e1 e2 n1 n2 s1 s2 [Horder [Hsafe Hruns]] Hlaunch H1 H2 Hij Hr1 Hp1 Hr2 Hp2.
 destruct (Hruns E Hlaunch) as [_ [Hordered _]].
 eapply Hordered;try eassumption.
 - apply nth_error_In in H1;exact H1.
 - apply nth_error_In in H2;exact H2.
 - eapply loop_events_ordered_at;eassumption.
Qed.

(* A checked event tail shares one ranked prefix proof. All subsequent
   events lie in the same basic block, and every intervening instruction
   falls through. Event markers, strict order and no-return obligations
   remain the ordinary ones, including for the first and last events. *)
Definition event_tail_shape (prog:program) (events:list pc) : bool :=
 match events with
 | [] => false
 | first::tail =>
   let final := last tail first in
   if short_all (fun e=>Nat.eqb (fst e) (fst first) &&
      Nat.leb (snd first) (snd e) && Nat.leb (snd e) (snd final)) events then
   if Nat.ltb (snd final) (block_len prog (fst first)) then
   short_all (fun i=>negb (is_jump i))
     (firstn (snd final-snd first) (skipn (snd first) (block_body prog (fst first))))
   else false else false
 end.
Definition event_tail_case (prog:program) (events:list pc) : bool :=
 if straight_event_program prog then false else event_tail_shape prog events.
Definition event_tail_check (F:lfacts) (prog:program) (events:list pc) : bool :=
 if loop_events_adjacent events then
 if short_all (ranked_phase_ready F prog) events then
 match events with first::_=>ranked_phase_progress F prog first | []=>false end
 else false else false.

Lemma event_tail_nth_skipn : forall A (xs:list A) n k,
 nth_error (skipn n xs) k=nth_error xs (n+k).
Proof.
 intros A xs n. revert xs. induction n as [|n IH];intros xs k;[reflexivity|].
 destruct xs as [|x xs];[destruct k;reflexivity|]. cbn [skipn Nat.add nth_error]. apply IH.
Qed.
Lemma event_tail_nth_firstn : forall A (xs:list A) n k x,
 (k<n)%nat -> nth_error xs k=Some x -> In x (firstn n xs).
Proof.
 intros A xs n. revert xs. induction n as [|n IH];intros xs k x Hk Hnth;[lia|].
 destruct xs as [|y xs];[destruct k;discriminate|].
 destruct k as [|k];cbn [nth_error] in Hnth;cbn [firstn].
 - injection Hnth as <-. left;reflexivity.
 - right. eapply IH with (k:=k);[lia|exact Hnth].
Qed.
Lemma event_tail_shape_sound : forall prog first tail event,
 event_tail_shape prog (first::tail)=true -> In event (first::tail) ->
 fst event=fst first /\ (snd first<=snd event)%nat /\
 (snd event<block_len prog (fst first))%nat /\
 forall n,(snd first<=n<snd event)%nat ->
 exists i,fetch prog (fst first) n=Some i /\ is_jump i=false.
Proof.
 intros prog first tail event Hshape Hin. unfold event_tail_shape in Hshape.
 destruct (short_all (fun e=>Nat.eqb (fst e) (fst first) && Nat.leb (snd first) (snd e) && Nat.leb (snd e) (snd (last tail first))) (first::tail)) eqn:Hall;[|discriminate].
 destruct (Nat.ltb (snd (last tail first)) (block_len prog (fst first))) eqn:Hbound;[|discriminate].
 apply Nat.ltb_lt in Hbound.
 rewrite short_all_same,forallb_forall in Hall,Hshape.
 specialize (Hall event Hin). rewrite !andb_true_iff in Hall.
 destruct Hall as [[Hb Hlo] Hhi]. apply Nat.eqb_eq in Hb.
 apply Nat.leb_le in Hlo,Hhi. split;[exact Hb|]. split;[exact Hlo|]. split;[lia|].
 intros n Hn. assert (Hnlen:(n<block_len prog (fst first))%nat) by lia.
 destruct (fetch prog (fst first) n) as [i|] eqn:Hfetch.
 - exists i. split;[reflexivity|]. apply negb_true_iff.
   apply Hshape. eapply event_tail_nth_firstn with (k:=(n-snd first)%nat).
   + lia.
   + rewrite event_tail_nth_skipn. replace (snd first+(n-snd first))%nat with n by lia.
     exact Hfetch.
 - unfold fetch in Hfetch. apply nth_error_None in Hfetch.
   unfold block_len in Hnlen. lia.
Qed.

Lemma event_tail_step : forall F prog E b n t s i,
 fetch prog b n=Some i -> is_jump i=false ->
 (S n<block_len prog b)%nat -> pc_of s=(b,n) ->
 exists s',phase_step F prog E t s=Some s' /\ pc_of s'=(b,S n).
Proof.
 intros F prog E b n t s i Hfetch Hjump Hlen Hpos.
 assert (Hf:fetch prog (blk s) (idx s)=Some i).
 { unfold pc_of in Hpos. injection Hpos as Hb Hi. rewrite Hb,Hi. exact Hfetch. }
 assert (Hstep:exists s',phase_step F prog E t s=Some s').
 { unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
 destruct Hstep as [s' Hstep]. exists s'. split;[exact Hstep|].
 pose proof (step_pos F prog E t s s' i Hf Hstep) as Hp.
 destruct Hp as [[Hp _]|[[g [target [Heq _]]]|[g [Heq _]]]].
 - unfold pc_of. rewrite Hp. unfold pc_of in Hpos. injection Hpos as Hb Hi.
   rewrite Hb,Hi. unfold next_pos. apply Nat.ltb_lt in Hlen. rewrite Hlen. reflexivity.
 - subst i. discriminate.
 - subst i. discriminate.
Qed.
Lemma event_tail_reaches : forall F prog E distance b start t s,
 pc_of s=(b,start) -> (start+distance<block_len prog b)%nat ->
 (forall n,(start<=n<start+distance)%nat ->
   exists i,fetch prog b n=Some i /\ is_jump i=false) ->
 phase_reaches F prog E (phase_event (b,(start+distance)%nat)) t s.
Proof.
 intros F prog E distance. induction distance as [|distance IH];intros b start t s Hpos Hlen Hbody.
 - rewrite Nat.add_0_r. apply Phase_here. rewrite Hpos. apply phase_event_eq;reflexivity.
 - destruct (Hbody start ltac:(lia)) as [i [Hfetch Hjump]].
   destruct (event_tail_step F prog E b start t s i Hfetch Hjump ltac:(lia) Hpos)
    as [s' [Hstep Hnext]].
   eapply Phase_next.
   + rewrite Hpos. unfold phase_event. cbn [fst snd]. rewrite Nat.eqb_refl. cbn.
     apply Nat.eqb_neq. lia.
   + exact Hstep.
   + replace (start+S distance)%nat with (S start+distance)%nat by lia.
     apply IH;[exact Hnext|lia|]. intros n Hn. apply Hbody. lia.
Qed.
Lemma event_tail_safe : forall F prog first tail event,
 event_tail_shape prog (first::tail)=true ->
 single_phase_safe F prog first -> In event (first::tail) ->
 phase_event_at prog event=true -> phase_closed prog event=true ->
 single_phase_safe F prog event.
Proof.
 intros F prog first tail event Hshape Hfirst Hin Hmarker Hclosed.
 split;[exact Hmarker|]. split;[exact Hclosed|]. intros E Hlaunch.
 destruct (event_tail_shape_sound prog first tail event Hshape Hin)
  as [Hb [Hlo [Hlen Hbody]]].
 destruct (loop_reaches_run F prog E first 0 (phase_initial E)
  (proj2 (proj2 Hfirst) E Hlaunch)) as [n [mid [Hrun Hpos]]].
 eapply loop_run_reaches;[exact Hrun|].
 replace event with (fst first,snd event) by (destruct event,first;cbn in *;subst;reflexivity).
 replace (snd event) with (snd first+(snd event-snd first))%nat by lia.
 apply event_tail_reaches;[destruct first;exact Hpos|lia|].
 intros j Hj. apply Hbody. lia.
Qed.
Theorem event_tail_check_sound : forall F prog events,
 event_tail_shape prog events=true -> event_tail_check F prog events=true ->
 loop_events_safe F prog events.
Proof.
 intros F prog events Hshape Hcheck. destruct events as [|first tail];[discriminate|].
 unfold event_tail_check in Hcheck.
 destruct (loop_events_adjacent (first::tail)) eqn:Horder;[|discriminate].
 destruct (short_all (ranked_phase_ready F prog) (first::tail)) eqn:Hready;[|discriminate].
 rewrite loop_events_adjacent_exact in Horder.
 rewrite short_all_same,forallb_forall in Hready.
 assert (Hfirst:single_phase_safe F prog first).
 { apply ranked_phase_check_sound. unfold ranked_phase_check.
   rewrite (Hready first (or_introl eq_refl)). exact Hcheck. }
 assert (Hsafe:forall event,In event (first::tail)->single_phase_safe F prog event).
 { intros event Hin. specialize (Hready event Hin). unfold ranked_phase_ready in Hready.
   destruct (lf_ok F);[|discriminate]. destruct (phase_event_at prog event) eqn:Hmarker;[|discriminate].
   eapply event_tail_safe;eassumption. }
 split;[exact Horder|]. split;[apply Forall_forall;exact Hsafe|].
 intros E Hlaunch. split.
 - intros event Hin. apply loop_reaches_run. exact (proj2 (proj2 (Hsafe event Hin)) E Hlaunch).
 - split.
   + intros e1 e2 n1 n2 s1 s2 H1 H2 Hord Hr1 Hp1 Hr2 Hp2.
     destruct (Hsafe e2 H2) as [Hf [Hc _]]. eapply loop_hit_order;eassumption.
   + intros event n1 n2 s1 s2 Hin Hr1 Hp1 Hr2 Hp2.
     destruct (Hsafe event Hin) as [Hf [Hc _]]. eapply loop_hit_unique;eassumption.
Qed.
Theorem event_tail_preserves : forall F prog events,
 event_tail_case prog events=true -> loop_events_check F prog events=true ->
 event_tail_check F prog events=true.
Proof.
 intros F prog events Hcase Hcheck. unfold event_tail_case in Hcase.
 destruct (straight_event_program prog) eqn:Hstraight;[discriminate|].
 unfold loop_events_check in Hcheck. unfold event_tail_check.
 destruct (loop_events_adjacent events);[|discriminate]. rewrite Hstraight in Hcheck.
 destruct (short_all (ranked_phase_ready F prog) events);[|discriminate].
 destruct events as [|first tail];[discriminate|]. cbn [short_all] in Hcheck.
 destruct (ranked_phase_progress F prog first);[reflexivity|discriminate].
Qed.

(* Finite occurrence schedules permit repeated visits to a checked event. *)
Definition occurrence_member (events:list pc) (pos:pc) : bool :=
 existsb (fun e=>phase_event e pos) events.

Inductive loop_event_free_run (F:lfacts) (prog:program) (E:env) (events:list pc)
 : nat->nat->state->state->Prop :=
 | Loop_free_done : forall t s,loop_event_free_run F prog E events 0 t s s
 | Loop_free_step : forall n t s mid last,
   occurrence_member events (pc_of s)=false ->
   phase_step F prog E t s=Some mid ->
   loop_event_free_run F prog E events n (S t) mid last ->
   loop_event_free_run F prog E events (S n) t s last.
Lemma loop_event_free_actual_run : forall F prog E events n t s last,
 loop_event_free_run F prog E events n t s last -> phase_run F prog E n t s=Some last.
Proof.
 intros F prog E events n t s last H. induction H;[reflexivity|].
 cbn [phase_run]. rewrite H0. exact IHloop_event_free_run.
Qed.
Lemma loop_event_free_prefix : forall F prog E events n t s last,
 loop_event_free_run F prog E events n t s last ->
 forall k mid,(k<n)%nat -> phase_run F prog E k t s=Some mid ->
 occurrence_member events (pc_of mid)=false.
Proof.
 intros F prog E events n t s last H. induction H;intros k mid' Hk Hrun;[lia|].
 destruct k as [|k].
 - cbn [phase_run] in Hrun. injection Hrun as <-. exact H.
 - cbn [phase_run] in Hrun. rewrite H0 in Hrun. eapply IHloop_event_free_run with (k:=k);[lia|exact Hrun].
Qed.

Lemma loop_event_free_continuation : forall F prog E events n t s last
 (P:nat->state->Prop),
 loop_event_free_run F prog E events n t s last ->
 (forall u v next,occurrence_member events(pc_of v)=false ->
   phase_step F prog E u v=Some next -> P(S u)next -> P u v) ->
 P(n+t)%nat last -> P t s.
Proof.
 intros F prog E events n t s last P Hrun Hnext. induction Hrun;intros HP.
 - exact HP.
 - eapply Hnext;[exact H|exact H0|]. apply IHHrun.
   replace (n+S t)%nat with (S n+t)%nat by lia. exact HP.
Qed.

Inductive occurrence_summary_result :=
 | No_loop_summary
 | Rejected_loop_summary
 | Checked_loop_summary (endpoint:pc) (post:astate).
Definition no_loop_summary (_:pc) (_:astate) := No_loop_summary.
Definition occurrence_summary_ok (F:lfacts) (prog:program) (E:env) (events:list pc)
 (summarize:pc->astate->occurrence_summary_result) :=
 forall pos st endpoint post,summarize pos st=Checked_loop_summary endpoint post ->
 forall t s,pos=pc_of s -> state_ok F t st(regs s) -> regs_ok F E(regs s) ->
 exists n last,loop_event_free_run F prog E events n t s last /\
 pc_of last=endpoint /\ state_ok F(n+t)%nat post(regs last) /\ regs_ok F E(regs last).
Lemma no_loop_summary_ok : forall F prog E events,
 occurrence_summary_ok F prog E events no_loop_summary.
Proof. intros F prog E events pos st endpoint post H. discriminate. Qed.

(* A finite segment ends at its FIRST event or in the checked final suffix.
   It carries all arrival states and the unspent common exploration budget. *)
Fixpoint occurrence_collect_with (summarize:pc->astate->occurrence_summary_result)
 (fuel:nat) (keep:PM.t unit) (F:lfacts) (prog:program)
 (events:list pc) (cut:pc) (pending:list (pc*option astate))
 : option (nat * list (pc*astate)) :=
 match pending with
 | []=>Some (fuel,[])
 | (pos,ost)::tail=>match fuel with
   | O=>None
   | S rest=>match ost with
     | None=>occurrence_collect_with summarize rest keep F prog events cut tail
     | Some st=>if is_bottom st then occurrence_collect_with summarize rest keep F prog events cut tail else
       if occurrence_member events pos || phase_after cut pos then
         match occurrence_collect_with summarize rest keep F prog events cut tail with
         | Some (remaining,arrivals)=>Some (remaining,(pos,st)::arrivals)|None=>None end
       else match summarize pos st with
       | Rejected_loop_summary=>None
       | Checked_loop_summary endpoint post=>
         occurrence_collect_with summarize rest keep F prog events cut ((endpoint,Some post)::tail)
       | No_loop_summary=>match fetch prog (fst pos) (snd pos) with
         | None=>None
         | Some i=>if wf_instr F i then
           occurrence_collect_with summarize rest keep F prog events cut
             (phase_successors keep F prog pos st i++tail)
           else None end
       end
     end
   end
 end.
Definition occurrence_collect := occurrence_collect_with no_loop_summary.

Definition occurrence_join (F:lfacts) (first:astate) (tail:list (pc*astate)) :=
 fold_left (fun acc entry=>join_states F acc (snd entry) false false) tail first.

(* Each nonterminal round consumes one marker and uses the SAME remaining
   exploration budget. The outer structural fuel also bounds recursion. *)
Fixpoint occurrence_word_with (summarize:pc->astate->occurrence_summary_result) (rounds fuel:nat) (keep:PM.t unit) (F:lfacts) (prog:program)
 (events:list pc) (cut pos:pc) (st:astate) : option (list pc) :=
 match rounds with
 | O=>None
 | S rounds=>match occurrence_collect_with summarize fuel keep F prog events cut [(pos,Some st)] with
   | Some (remaining,(event,first)::tail)=>
     if occurrence_member events event then
       let inv:=occurrence_join F first tail in
       if short_all (fun entry=>phase_event event (fst entry) && state_le F (snd entry) inv)
                    ((event,first)::tail) then
         match remaining,fetch prog (fst event) (snd event) with
         | S remaining,Some i=>if phase_marker i && wf_instr F i then
             match occurrence_word_with summarize rounds remaining keep F prog events cut
                 (next_pos prog (fst event) (snd event))
                 (phase_trim keep (transfer F inv i)) with
             | Some word=>Some (event::word)|None=>None end
             else None
         | _,_=>None end
       else None
     else if short_all (fun entry=>phase_after cut (fst entry)) ((event,first)::tail)
       then Some [] else None
   | _=>None end
 end.

Definition occurrence_word := occurrence_word_with no_loop_summary.

Definition occurrence_cut (prog:program) (events:list pc) : pc :=
 match rev events with
 | []=>(0%nat,0%nat)
 | last::_=>(fst last,Nat.pred (block_len prog (fst last))) end.

Definition occurrence_cfg_next (prog:program) (pos:pc) : list pc :=
 let next:=next_pos prog (fst pos) (snd pos) in
 match fetch prog (fst pos) (snd pos) with
 | None|Some (IExit None)=>[]
 | Some (IBra None target)=>[(target,0%nat)]
 | Some (IBra (Some _) target)=>[(target,0%nat);next]
 | Some _=>[next]
 end.

(* This optional structural eligibility pass rejects event-free CFG cycles
   before symbolic exploration. It cannot justify an acceptance: the full
   occurrence collector and its soundness theorem remain required. *)
Definition occurrence_pc_index := PM.t (PM.t unit).

Definition occurrence_pc_member (done : occurrence_pc_index) (pos : pc) : bool :=
  match PM.find (Pos.of_succ_nat (fst pos)) done with
  | Some block =>
      match PM.find (Pos.of_succ_nat (snd pos)) block with
      | Some _ => true
      | None => false
      end
  | None => false
  end.

Definition occurrence_pc_add (pos : pc) (done : occurrence_pc_index)
    : occurrence_pc_index :=
  let b := Pos.of_succ_nat (fst pos) in
  let i := Pos.of_succ_nat (snd pos) in
  let block := match PM.find b done with
               | Some block => block
               | None => PM.empty unit
               end in
  PM.add b (PM.add i tt block) done.

Lemma occurrence_pc_empty : forall pos,
  occurrence_pc_member (PM.empty (PM.t unit)) pos = false.
Proof. intros. unfold occurrence_pc_member. rewrite PM.gempty. reflexivity. Qed.

Lemma occurrence_pc_added : forall done pos query,
  occurrence_pc_member (occurrence_pc_add pos done) query =
  (phase_event pos query || occurrence_pc_member done query)%bool.
Proof.
  intros done [b i] [c j]. unfold occurrence_pc_add, occurrence_pc_member, phase_event.
  cbn [fst snd]. destruct (Nat.eq_dec b c) as [-> | Hbc].
  - rewrite PM.gss, Nat.eqb_refl. cbn [andb].
    destruct (Nat.eq_dec i j) as [-> | Hij].
    + rewrite PM.gss, Nat.eqb_refl. reflexivity.
    + rewrite PM.gso by (intro Heq; apply Hij; now apply SuccNat2Pos.inj in Heq).
      rewrite (proj2 (Nat.eqb_neq i j) Hij). cbn [orb].
      destruct (PM.find (Pos.of_succ_nat c) done); [reflexivity |].
      rewrite PM.gempty. reflexivity.
  - rewrite PM.gso by (intro Heq; apply Hbc; now apply SuccNat2Pos.inj in Heq).
    rewrite (proj2 (Nat.eqb_neq b c) Hbc). reflexivity.
Qed.

Fixpoint occurrence_pc_of_list (done : list pc) : occurrence_pc_index :=
  match done with
  | [] => PM.empty (PM.t unit)
  | pos :: tail => occurrence_pc_add pos (occurrence_pc_of_list tail)
  end.

Lemma occurrence_pc_list_exact : forall done pos,
  occurrence_pc_member (occurrence_pc_of_list done) pos = occurrence_member done pos.
Proof.
  induction done as [|head tail IH]; intros pos.
  - apply occurrence_pc_empty.
  - cbn [occurrence_pc_of_list]. rewrite occurrence_pc_added, IH. reflexivity.
Qed.

(* Every stack frame remembers the membership index of its parent stack.
   Restoring that snapshot also handles duplicate entries in an arbitrary
   initial stack; no uniqueness condition or lossy set removal is needed. *)
Record occurrence_active_index := {
  occurrence_active_members : occurrence_pc_index;
  occurrence_active_frames : list (pc * occurrence_pc_index)
}.

Definition occurrence_active_member (active : occurrence_active_index) (pos : pc) : bool :=
  occurrence_pc_member (occurrence_active_members active) pos.

Definition occurrence_active_push (pos : pc) (active : occurrence_active_index)
    : occurrence_active_index :=
  {| occurrence_active_members := occurrence_pc_add pos (occurrence_active_members active);
     occurrence_active_frames := (pos, occurrence_active_members active) :: occurrence_active_frames active |}.

Definition occurrence_active_pop (active : occurrence_active_index)
    : option (pc * occurrence_active_index) :=
  match occurrence_active_frames active with
  | [] => None
  | (pos, parent) :: rest =>
      Some (pos, {| occurrence_active_members := parent; occurrence_active_frames := rest |})
  end.

Fixpoint occurrence_active_of_list (active : list pc) : occurrence_active_index :=
  match active with
  | [] => {| occurrence_active_members := PM.empty (PM.t unit); occurrence_active_frames := [] |}
  | pos :: rest => occurrence_active_push pos (occurrence_active_of_list rest)
  end.

Definition occurrence_list_pop (active : list pc) : option (pc * list pc) :=
  match active with [] => None | pos :: rest => Some (pos, rest) end.

Lemma occurrence_active_list_exact : forall active pos,
  occurrence_active_member (occurrence_active_of_list active) pos = occurrence_member active pos.
Proof.
  induction active as [|head tail IH]; intros pos.
  - apply occurrence_pc_empty.
  - change (occurrence_pc_member
      (occurrence_pc_add head (occurrence_active_members (occurrence_active_of_list tail))) pos =
      (phase_event head pos || occurrence_member tail pos)%bool).
    rewrite occurrence_pc_added. fold (occurrence_active_member (occurrence_active_of_list tail) pos).
    rewrite IH. reflexivity.
Qed.

Fixpoint occurrence_graph_walk_member (A D : Type)
    (active_member : A -> pc -> bool) (push : pc -> A -> A) (pop : A -> option (pc * A))
    (done_member : D -> pc -> bool) (add : pc -> D -> D)
    (next : pc -> list pc) (fuel : nat) (is_event : pc -> bool) (cut : pc)
    (pending : list (bool * pc)) (active : A) (done : D) : bool :=
  match pending with
  | [] => true
  | (enter, pos) :: tail =>
      match fuel with
      | O => false
      | S rest =>
          if enter then
            if is_event pos || phase_after cut pos then
              occurrence_graph_walk_member A D active_member push pop done_member add
                next rest is_event cut tail active done
            else if done_member done pos then
              occurrence_graph_walk_member A D active_member push pop done_member add
                next rest is_event cut tail active done
            else if active_member active pos then false
            else occurrence_graph_walk_member A D active_member push pop done_member add
              next rest is_event cut
              (map (fun child => (true, child)) (next pos) ++ (false, pos) :: tail)
              (push pos active) done
          else
            match pop active with
            | Some (current, parents) =>
                if phase_event current pos then
                  occurrence_graph_walk_member A D active_member push pop done_member add
                    next rest is_event cut tail parents (add pos done)
                else false
            | None => false
            end
      end
  end.

(* The list interface remains available for exactness statements. The single
   recursive core is parameterized only by the event-membership decision. *)
Definition occurrence_graph_walk_state (A D : Type)
    (active_member : A -> pc -> bool) (push : pc -> A -> A) (pop : A -> option (pc * A))
    (done_member : D -> pc -> bool) (add : pc -> D -> D)
    (next : pc -> list pc) (fuel : nat) (events : list pc) (cut : pc)
    (pending : list (bool * pc)) (active : A) (done : D) : bool :=
  occurrence_graph_walk_member A D active_member push pop done_member add next fuel
    (occurrence_member events) cut pending active done.

Lemma occurrence_graph_walk_member_exact : forall A D active_member push pop done_member add fuel
    (member1 member2 : pc -> bool),
    (forall pos, member1 pos = member2 pos) ->
    forall next cut pending (active : A) (done : D),
  occurrence_graph_walk_member A D active_member push pop done_member add next fuel
    member1 cut pending active done =
  occurrence_graph_walk_member A D active_member push pop done_member add next fuel
    member2 cut pending active done.
Proof.
  intros A D active_member push pop done_member add fuel.
  induction fuel as [|fuel IH]; intros member1 member2 Hevents next cut pending active done;
    destruct pending as [|[enter pos] tail]; try reflexivity.
  cbn [occurrence_graph_walk_member]. destruct enter.
  - rewrite Hevents. destruct (member2 pos || phase_after cut pos); [apply IH; exact Hevents |].
    destruct (done_member done pos); [apply IH; exact Hevents |].
    destruct (active_member active pos); [reflexivity | apply IH; exact Hevents].
  - destruct (pop active) as [[current parents]|]; [|reflexivity].
    destruct (phase_event current pos); [apply IH; exact Hevents | reflexivity].
Qed.

Definition occurrence_pop_related {A B} (R : A -> B -> Prop)
    (left : option (pc * A)) (right : option (pc * B)) : Prop :=
  match left, right with
  | None, None => True
  | Some (p, a), Some (q, b) => p = q /\ R a b
  | _, _ => False
  end.

Lemma occurrence_graph_walk_state_ext : forall A1 A2 D1 D2
    (active1 : A1 -> pc -> bool) (active2 : A2 -> pc -> bool)
    (push1 : pc -> A1 -> A1) (push2 : pc -> A2 -> A2)
    (pop1 : A1 -> option (pc * A1)) (pop2 : A2 -> option (pc * A2))
    (member1 : D1 -> pc -> bool) (member2 : D2 -> pc -> bool)
    (add1 : pc -> D1 -> D1) (add2 : pc -> D2 -> D2)
    (RA : A1 -> A2 -> Prop) (RD : D1 -> D2 -> Prop),
  (forall a1 a2, RA a1 a2 -> forall pos, active1 a1 pos = active2 a2 pos) ->
  (forall a1 a2, RA a1 a2 -> forall pos, RA (push1 pos a1) (push2 pos a2)) ->
  (forall a1 a2, RA a1 a2 -> occurrence_pop_related RA (pop1 a1) (pop2 a2)) ->
  (forall d1 d2, RD d1 d2 -> forall pos, member1 d1 pos = member2 d2 pos) ->
  (forall d1 d2, RD d1 d2 -> forall pos, RD (add1 pos d1) (add2 pos d2)) ->
  forall fuel next events cut pending a1 a2 d1 d2, RA a1 a2 -> RD d1 d2 ->
    occurrence_graph_walk_state A1 D1 active1 push1 pop1 member1 add1
      next fuel events cut pending a1 d1 =
    occurrence_graph_walk_state A2 D2 active2 push2 pop2 member2 add2
      next fuel events cut pending a2 d2.
Proof.
  intros A1 A2 D1 D2 active1 active2 push1 push2 pop1 pop2 member1 member2 add1 add2
    RA RD Hactive Hpush Hpop Hmember Hadd fuel.
  unfold occurrence_graph_walk_state.
  induction fuel as [|fuel IH]; intros next events cut pending a1 a2 d1 d2 HA HD;
    destruct pending as [|[enter pos] tail]; try reflexivity.
  cbn [occurrence_graph_walk_member]. destruct enter.
  - destruct (occurrence_member events pos || phase_after cut pos); [apply IH; assumption |].
    rewrite (Hmember _ _ HD pos). destruct (member2 d2 pos); [apply IH; assumption |].
    rewrite (Hactive _ _ HA pos). destruct (active2 a2 pos); [reflexivity |].
    apply IH; [apply Hpush |]; assumption.
  - pose proof (Hpop _ _ HA) as HP.
    destruct (pop1 a1) as [[p parent1]|]; destruct (pop2 a2) as [[q parent2]|];
      cbn [occurrence_pop_related] in HP; try contradiction; try reflexivity.
    destruct HP as [-> HR]. destruct (phase_event q pos); [|reflexivity].
    apply IH; [exact HR | apply Hadd; exact HD].
Qed.

Definition occurrence_graph_walk_with (next : pc -> list pc) (fuel : nat)
    (events : list pc) (cut : pc) (pending : list (bool * pc)) (active done : list pc) : bool :=
  occurrence_graph_walk_member occurrence_active_index occurrence_pc_index
    occurrence_active_member occurrence_active_push occurrence_active_pop
    occurrence_pc_member occurrence_pc_add next fuel
    (occurrence_pc_member (occurrence_pc_of_list events)) cut pending
    (occurrence_active_of_list active) (occurrence_pc_of_list done).

Theorem occurrence_graph_walk_with_exact : forall next fuel events cut pending active done,
  occurrence_graph_walk_with next fuel events cut pending active done =
  occurrence_graph_walk_state (list pc) (list pc)
    occurrence_member (fun pos rest => pos :: rest) occurrence_list_pop
    occurrence_member (fun pos rest => pos :: rest)
    next fuel events cut pending active done.
Proof.
  intros. unfold occurrence_graph_walk_with.
  transitivity (occurrence_graph_walk_state occurrence_active_index occurrence_pc_index
    occurrence_active_member occurrence_active_push occurrence_active_pop
    occurrence_pc_member occurrence_pc_add next fuel events cut pending
    (occurrence_active_of_list active) (occurrence_pc_of_list done)).
  - unfold occurrence_graph_walk_state. apply occurrence_graph_walk_member_exact.
    apply occurrence_pc_list_exact.
  -
    eapply occurrence_graph_walk_state_ext with
    (RA := fun index nodes => index = occurrence_active_of_list nodes)
    (RD := fun index nodes => forall pos, occurrence_pc_member index pos = occurrence_member nodes pos).
    + intros index nodes -> pos. apply occurrence_active_list_exact.
    + intros index nodes -> pos. reflexivity.
    + intros index nodes ->. destruct nodes as [|pos nodes]; [exact I |].
    split; [reflexivity |]. destruct (occurrence_active_of_list nodes). reflexivity.
    + intros index nodes HR pos. apply HR.
    + intros index nodes HR pos query. rewrite occurrence_pc_added, HR. reflexivity.
    + reflexivity.
    + apply occurrence_pc_list_exact.
Qed.


Definition occurrence_graph_walk (fuel:nat) (prog:program) (events:list pc) (cut:pc)
 (pending:list(bool*pc)) (active done:list pc) : bool :=
 occurrence_graph_walk_with (occurrence_cfg_next prog) fuel events cut pending active done.

Definition occurrence_segments_acyclic (prog:program) (events:list pc) (cut:pc) :=
 let positions:=List.length (List.concat prog) in
 occurrence_graph_walk (4*positions+2*List.length events+4)%nat prog events cut
 ((true,(0%nat,0%nat))::map (fun event=>(true,next_pos prog (fst event) (snd event))) events)
 [] [].

Definition occurrence_checked_word (summarize:pc->astate->occurrence_summary_result) (F:lfacts) (prog:program) (events:list pc) :=
 let cut:=occurrence_cut prog events in
 if lf_ok F && Nat.ltb (fst cut) (List.length prog) && phase_closed prog cut &&
    short_all (fun e=>phase_event_at prog e && negb (phase_after cut e)) events then
   let keep:=phase_control_regs (List.concat prog) in
   match occurrence_word_with summarize phase_fuel phase_fuel keep F prog events cut (0%nat,0%nat) (empty_state F) with
   | Some word=>if short_all (fun e=>occurrence_member word e) events then Some word else None
   | None=>None end
 else None.

Definition occurrence_program_word (F:lfacts) (prog:program) (events:list pc) :=
 if occurrence_segments_acyclic prog events (occurrence_cut prog events)
 then occurrence_checked_word no_loop_summary F prog events else None.

Lemma occurrence_member_in : forall events pos,
 occurrence_member events pos=true <-> In pos events.
Proof.
 intros events pos. unfold occurrence_member. rewrite existsb_exists.
 split.
 - intros [event [Hin Heq]]. apply phase_event_eq in Heq. subst;exact Hin.
 - intros Hin. exists pos. split;[exact Hin|apply phase_event_eq;reflexivity].
Qed.

Section OccurrenceSemantics.
Variable F:lfacts.
Variable prog:program.
Variable E:env.
Hypothesis HE:env_ok F E.
Variable events:list pc.

(* Every actual event is consumed in order. The final suffix may diverge,
   but cannot execute another event. The word is independent of E. *)
Inductive occurrence_trace : list pc -> nat -> state -> Prop :=
| Occurrence_done : forall t s,
    (forall n last,phase_run F prog E n t s=Some last ->
       occurrence_member events (pc_of last)=false) ->
    occurrence_trace [] t s
| Occurrence_now : forall event word t s next,
    occurrence_member events (pc_of s)=true -> pc_of s=event ->
    phase_step F prog E t s=Some next ->
    occurrence_trace word (S t) next -> occurrence_trace (event::word) t s
| Occurrence_later : forall word t s next,
    occurrence_member events (pc_of s)=false ->
    phase_step F prog E t s=Some next ->
    occurrence_trace word (S t) next -> occurrence_trace word t s.

Lemma occurrence_cut_step : forall cut t s next i,
 (fst cut<List.length prog)%nat -> phase_closed prog cut=true ->
 phase_after cut (pc_of s)=true -> fetch prog (blk s) (idx s)=Some i ->
 phase_step F prog E t s=Some next -> phase_after cut (pc_of next)=true.
Proof.
 intros cut t s next i Hbound Hclosed Hafter Hfetch Hstep.
 pose proof (step_pos F prog E t s next i Hfetch Hstep) as Hpos.
 destruct Hpos as [[Hnext _]|[[g [target [Hi [_ Htarget]]]]|[g [Hi [_ Hexit]]]]].
 - unfold pc_of. rewrite Hnext. exact (phase_next_after prog cut (pc_of s) Hafter).
 - subst i. unfold pc_of. rewrite Htarget. apply phase_after_iff. left.
   eapply phase_closed_branch;eassumption.
 - unfold pc_of. rewrite Hexit. apply phase_after_iff. left;exact Hbound.
Qed.
Lemma occurrence_cut_stays : forall n cut t s last,
 (fst cut<List.length prog)%nat -> phase_closed prog cut=true ->
 phase_after cut (pc_of s)=true -> phase_run F prog E n t s=Some last ->
 phase_after cut (pc_of last)=true.
Proof.
 induction n as [|n IH];intros cut t s last Hb Hc Ha Hr.
 - cbn [phase_run] in Hr. injection Hr as <-. exact Ha.
 - cbn [phase_run] in Hr.
   destruct (phase_step F prog E t s) as [next|] eqn:Hs;[|discriminate].
   assert (Hfetch:exists i,fetch prog (blk s) (idx s)=Some i).
   { unfold phase_step,step in Hs.
     destruct (fetch prog (blk s) (idx s)) as [i|];[eexists;reflexivity|discriminate]. }
   destruct Hfetch as [i Hf]. eapply IH;[exact Hb|exact Hc| |exact Hr].
   eapply occurrence_cut_step;eassumption.
Qed.
Lemma occurrence_cut_trace : forall cut t s,
 (fst cut<List.length prog)%nat -> phase_closed prog cut=true ->
 (forall e,In e events -> phase_after cut e=false) ->
 phase_after cut (pc_of s)=true -> occurrence_trace [] t s.
Proof.
 intros cut t s Hb Hc Hbefore Ha. constructor. intros n last Hr.
 pose proof (occurrence_cut_stays n cut t s last Hb Hc Ha Hr) as Hlast.
 destruct (occurrence_member events (pc_of last)) eqn:He;[|reflexivity].
 apply occurrence_member_in in He. specialize (Hbefore _ He).
 rewrite Hbefore in Hlast. discriminate.
Qed.

(* The collector supports any backward-closed concrete continuation. Thus
   its abstract endpoint list does not become an unproved execution trace. *)
Lemma occurrence_collect_with_continuation : forall summarize,
 occurrence_summary_ok F prog E events summarize -> forall fuel keep cut pending remaining arrivals
 (P:nat->state->Prop),
 occurrence_collect_with summarize fuel keep F prog events cut pending=Some (remaining,arrivals) ->
 (forall t s next,occurrence_member events (pc_of s)=false ->
   phase_step F prog E t s=Some next -> P (S t) next -> P t s) ->
 (forall pos st,In (pos,st) arrivals -> forall t s,pos=pc_of s ->
   state_ok F t st (regs s) -> regs_ok F E (regs s) -> P t s) ->
 forall pos st,In (pos,Some st) pending -> forall t s,pos=pc_of s ->
   state_ok F t st (regs s) -> regs_ok F E (regs s) -> P t s.
Proof.
 intros summarize Hsummary. induction fuel as [|fuel IH];intros keep cut pending remaining arrivals P Hcheck Hnext Hend
 pos st Hin t s Hpos Hst Hrs;
 destruct pending as [|[q ost] tail];[contradiction|discriminate|contradiction|].
 cbn [occurrence_collect_with] in Hcheck. destruct ost as [current|].
 - destruct (is_bottom current) eqn:Hbottom.
   + destruct Hin as [Heq|Hin].
     * injection Heq as <- <-. exfalso. exact (bottom_not_ok F t current _ Hbottom Hst).
     * eapply IH;eassumption.
   + destruct (occurrence_member events q || phase_after cut q) eqn:Hstop.
     * destruct (occurrence_collect_with summarize fuel keep F prog events cut tail) as [[rest ends]|] eqn:Htail;
       [|discriminate]. injection Hcheck as <- <-.
       destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. eapply Hend;[left;reflexivity|exact Hpos|exact Hst|exact Hrs].
       -- eapply IH;[exact Htail|exact Hnext| |exact Hin|exact Hpos|exact Hst|exact Hrs].
          intros p a Ha u v Hp Hv Hrv. eapply Hend;[right;exact Ha|exact Hp|exact Hv|exact Hrv].
     * destruct (summarize q current) as [| |endpoint post] eqn:Hsummary_step;[|discriminate|].
       { destruct (fetch prog (fst q) (snd q)) as [i|] eqn:Hfetch;[|discriminate].
       destruct (wf_instr F i) eqn:Hwf;[|discriminate].
       destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. subst q.
          assert (Hf:fetch prog (blk s) (idx s)=Some i) by exact Hfetch.
          assert (Hex:exists next,phase_step F prog E t s=Some next).
          { unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
          destruct Hex as [next Hstep].
          destruct (phase_successors_complete F prog E HE keep t s next current i Hf Hstep Hst Hrs Hwf)
          as [out [Hchild [Hout Hregs]]].
          eapply Hnext;[apply orb_false_iff in Hstop;exact (proj1 Hstop)|exact Hstep|].
          eapply IH;[exact Hcheck|exact Hnext|exact Hend| |reflexivity|exact Hout|exact Hregs].
          apply in_or_app;left;exact Hchild.
       -- eapply IH;[exact Hcheck|exact Hnext|exact Hend| |exact Hpos|exact Hst|exact Hrs].
          apply in_or_app;right;exact Hin. }
       { destruct Hin as [Heq|Hin].
         - injection Heq as <- <-.
           destruct (Hsummary q current endpoint post Hsummary_step t s Hpos Hst Hrs)
             as [n [last [Hrun [Hpc [Hout Hregs]]]]].
           eapply loop_event_free_continuation;[exact Hrun|exact Hnext|].
           eapply IH;[exact Hcheck|exact Hnext|exact Hend|left;reflexivity|symmetry;exact Hpc|exact Hout|exact Hregs].
         - eapply IH;[exact Hcheck|exact Hnext|exact Hend|right;exact Hin|exact Hpos|exact Hst|exact Hrs]. }

 - destruct Hin as [Heq|Hin];[discriminate|]. eapply IH;eassumption.
Qed.

Lemma occurrence_collect_continuation : forall fuel keep cut pending remaining arrivals
 (P:nat->state->Prop),
 occurrence_collect fuel keep F prog events cut pending=Some (remaining,arrivals) ->
 (forall t s next,occurrence_member events (pc_of s)=false ->
   phase_step F prog E t s=Some next -> P (S t) next -> P t s) ->
 (forall pos st,In (pos,st) arrivals -> forall t s,pos=pc_of s ->
   state_ok F t st (regs s) -> regs_ok F E (regs s) -> P t s) ->
 forall pos st,In (pos,Some st) pending -> forall t s,pos=pc_of s ->
   state_ok F t st (regs s) -> regs_ok F E (regs s) -> P t s.
Proof. intros. eapply occurrence_collect_with_continuation; eauto using no_loop_summary_ok. Qed.

Theorem occurrence_word_with_sound : forall summarize,
 occurrence_summary_ok F prog E events summarize -> forall rounds fuel keep cut pos st word,
 (fst cut<List.length prog)%nat -> phase_closed prog cut=true ->
 (forall e,In e events -> phase_after cut e=false) ->
 occurrence_word_with summarize rounds fuel keep F prog events cut pos st=Some word ->
 forall t s,pos=pc_of s -> state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 occurrence_trace word t s.
Proof.
 intros summarize Hsummary. induction rounds as [|rounds IH];intros fuel keep cut pos st word Hb Hc Hbefore Hcheck t s Hpos Hst Hrs;
 [discriminate|]. cbn [occurrence_word_with] in Hcheck.
 destruct (occurrence_collect_with summarize fuel keep F prog events cut [(pos,Some st)])
   as [[remaining arrivals]|] eqn:Hcollect;[|discriminate].
 destruct arrivals as [|[event first] tail];[discriminate|].
 destruct (occurrence_member events event) eqn:Hevent.
 - set (inv:=occurrence_join F first tail) in *.
   destruct (short_all (fun entry=>phase_event event (fst entry) && state_le F (snd entry) inv)
     ((event,first)::tail)) eqn:Hall;[|discriminate].
   destruct remaining as [|remaining];[discriminate|].
   destruct (fetch prog (fst event) (snd event)) as [i|] eqn:Hfetch;[|discriminate].
   destruct (phase_marker i && wf_instr F i) eqn:Hmarker;[|discriminate].
   destruct (occurrence_word_with summarize rounds remaining keep F prog events cut
     (next_pos prog (fst event) (snd event)) (phase_trim keep (transfer F inv i)))
     as [rest|] eqn:Hrest;[|discriminate]. injection Hcheck as <-.
   apply andb_true_iff in Hmarker. destruct Hmarker as [Hmark Hwf].
   rewrite short_all_same,forallb_forall in Hall.
   eapply occurrence_collect_with_continuation;[exact Hsummary|exact Hcollect| | |left;reflexivity|exact Hpos|exact Hst|exact Hrs].
   + intros u v next Hnot Hstep Htrace. eapply Occurrence_later;eassumption.
   + intros p a Hin u v Hp Ha Hr.
     specialize (Hall _ Hin). cbn [fst snd] in Hall. apply andb_true_iff in Hall.
     destruct Hall as [Hsame Hle]. apply phase_event_eq in Hsame. subst p.
     assert (Hi:state_ok F u inv (regs v)).
     { eapply state_le_sound;eassumption. }
     assert (Hf:fetch prog (blk v) (idx v)=Some i) by (rewrite Hp in Hfetch;exact Hfetch).
     assert (Hex:exists next,phase_step F prog E u v=Some next).
     { unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
     destruct Hex as [next Hstep].
     destruct (transfer_sound F prog E HE u v next inv i Hf Hstep Hi Hr Hwf) as [Ho Hregs].
     assert (Hnext:pc_of next=next_pos prog (fst event) (snd event)).
     { pose proof (step_pos F prog E u v next i Hf Hstep) as H.
       destruct H as [[H _]|[[g [target [Hbad _]]]|[g [Hbad _]]]].
       - change (pc_of next=next_pos prog (fst (pc_of v)) (snd (pc_of v))) in H. rewrite <-Hp in H. exact H.
       - exfalso. exact (proj1 (phase_marker_not_control i Hmark) g target Hbad).
       - exfalso. exact (proj2 (phase_marker_not_control i Hmark) g Hbad). }
     eapply Occurrence_now;[rewrite <-Hp;exact Hevent|symmetry;exact Hp|exact Hstep|].
     eapply IH;[exact Hb|exact Hc|exact Hbefore|exact Hrest|symmetry;exact Hnext| |exact Hregs].
     apply phase_trim_sound. exact Ho.
 - destruct (short_all (fun entry=>phase_after cut (fst entry)) ((event,first)::tail)) eqn:Hall;
   [|discriminate]. injection Hcheck as <-.
   rewrite short_all_same,forallb_forall in Hall.
   eapply occurrence_collect_with_continuation;[exact Hsummary|exact Hcollect| | |left;reflexivity|exact Hpos|exact Hst|exact Hrs].
   + intros u v next Hnot Hstep Htrace. eapply Occurrence_later;eassumption.
   + intros p a Hin u v Hp Ha Hr. eapply occurrence_cut_trace;[exact Hb|exact Hc|exact Hbefore|].
     rewrite <-Hp. exact (Hall _ Hin).
Qed.
Theorem occurrence_word_sound : forall rounds fuel keep cut pos st word,
 (fst cut<List.length prog)%nat -> phase_closed prog cut=true ->
 (forall e,In e events -> phase_after cut e=false) ->
 occurrence_word rounds fuel keep F prog events cut pos st=Some word ->
 forall t s,pos=pc_of s -> state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 occurrence_trace word t s.
Proof. intros. eapply occurrence_word_with_sound; eauto using no_loop_summary_ok. Qed.
End OccurrenceSemantics.

(* This observation includes precisely those event positions whose marker
   is actually executed in the prefix. It does not assume termination. *)
Fixpoint occurrence_observe (F:lfacts) (prog:program) (E:env) (events:list pc)
 (n t:nat) (s:state) : option (list pc*state) :=
 match n with
 | O=>Some ([],s)
 | S n=>match phase_step F prog E t s with
   | None=>None
   | Some next=>match occurrence_observe F prog E events n (S t) next with
     | Some (word,last)=>Some ((if occurrence_member events (pc_of s)
         then pc_of s::word else word),last)
     | None=>None end end
 end.
Theorem occurrence_trace_realizes : forall F prog E events word t s,
 occurrence_trace F prog E events word t s ->
 exists n last, occurrence_observe F prog E events n t s=Some (word,last) /\
 (forall k final,phase_run F prog E k (n+t)%nat last=Some final ->
   occurrence_member events (pc_of final)=false).
Proof.
 intros F prog E events word t s Htrace. induction Htrace as
 [t s Hnone|event word t s next Hmember Hpc Hstep Htrace [n [last [Hobs Hnone]]]
 |word t s next Hmember Hstep Htrace [n [last [Hobs Hnone]]]].
 - exists 0%nat,s. split;[reflexivity|exact Hnone].
 - exists (S n),last. split.
   + cbn [occurrence_observe]. rewrite Hstep,Hobs,Hmember,Hpc. reflexivity.
   + replace (S n+t)%nat with (n+S t)%nat by lia. exact Hnone.
 - exists (S n),last. split.
   + cbn [occurrence_observe]. rewrite Hstep,Hobs,Hmember. reflexivity.
   + replace (S n+t)%nat with (n+S t)%nat by lia. exact Hnone.
Qed.

Definition occurrence_schedule_safe (F:lfacts) (prog:program) (events:list pc) : Prop :=
 (forall event,In event events -> phase_event_at prog event=true) /\
 exists word,(forall event,In event events -> In event word) /\
 forall E,launch_ok F E -> occurrence_trace F prog E events word 0 (phase_initial E).

Theorem occurrence_checked_word_sound : forall F prog events word summarize,
 (forall E,env_ok F E -> occurrence_summary_ok F prog E events summarize) ->
 occurrence_checked_word summarize F prog events=Some word -> occurrence_schedule_safe F prog events.
Proof.
 intros F prog events word summarize Hsummary Hcheck. unfold occurrence_checked_word in Hcheck.
 set (cut:=occurrence_cut prog events) in *.
 destruct (lf_ok F && Nat.ltb (fst cut) (List.length prog) && phase_closed prog cut &&
   short_all (fun e=>phase_event_at prog e && negb (phase_after cut e)) events) eqn:Hbase;
 [|discriminate].
 destruct (occurrence_word_with summarize phase_fuel phase_fuel (phase_control_regs (List.concat prog))
   F prog events cut (0%nat,0%nat) (empty_state F)) as [found|] eqn:Hword;[|discriminate].
 destruct (short_all (fun e=>occurrence_member found e) events) eqn:Hcover;[|discriminate].
 injection Hcheck as <-. rewrite !andb_true_iff in Hbase.
 destruct Hbase as [[[Hlf Hbound] Hclosed] Hevents].
 apply Nat.ltb_lt in Hbound. rewrite short_all_same,forallb_forall in Hevents,Hcover.
 split.
 - intros event Hin. specialize (Hevents _ Hin). apply andb_true_iff in Hevents. exact (proj1 Hevents).
 - exists found. split.
   + intros event Hin. apply occurrence_member_in. exact (Hcover _ Hin).
   + intros E Hlaunch. eapply occurrence_word_with_sound.
     * exact (env_of_launch F prog E Hlf Hlaunch).
     * apply Hsummary. exact (env_of_launch F prog E Hlf Hlaunch).
     * exact Hbound.
     * exact Hclosed.
     * intros event Hin. specialize (Hevents _ Hin). apply andb_true_iff in Hevents.
       apply negb_true_iff. exact (proj2 Hevents).
     * exact Hword.
     * reflexivity.
     * split.
       -- intros r f Hf. unfold get,empty_state in Hf;cbn [amap] in Hf. rewrite PM.gempty in Hf. discriminate.
       -- exact (lk_derived F E Hlaunch).
     * split;[exact (lk_init F E Hlaunch)|]. intros r Hfixed;reflexivity.
Qed.


Theorem occurrence_program_word_sound : forall F prog events word,
 occurrence_program_word F prog events=Some word -> occurrence_schedule_safe F prog events.
Proof.
 intros F prog events word H. unfold occurrence_program_word in H.
 destruct(occurrence_segments_acyclic prog events(occurrence_cut prog events));[|discriminate].
 eapply occurrence_checked_word_sound;[intros;apply no_loop_summary_ok|exact H].
Qed.

(* Choose the proof form from the actual IR before analysis. No unsuccessful
   phase checker is retried through the occurrence checker. *)
Definition occurrence_case (prog:program) (events:list pc) : bool :=
 if straight_event_program prog then false else
 negb (short_all (phase_closed prog) events).
Definition event_progress_nonocc (F:lfacts) (prog:program) (events:list pc) : bool :=
 if event_tail_case prog events then event_tail_check F prog events
 else loop_events_check F prog events.
Definition event_progress_check (F:lfacts) (prog:program) (events:list pc) : bool :=
 if occurrence_case prog events then
   match occurrence_program_word F prog events with Some _=>true|None=>false end
 else event_progress_nonocc F prog events.
Definition event_progress_safe (F:lfacts) (prog:program) (events:list pc) : Prop :=
 if occurrence_case prog events then occurrence_schedule_safe F prog events
 else loop_events_safe F prog events.
Theorem event_progress_check_sound : forall F prog events,
 event_progress_check F prog events=true -> event_progress_safe F prog events.
Proof.
 intros F prog events H. unfold event_progress_check,event_progress_safe,event_progress_nonocc in *.
 destruct (occurrence_case prog events).
 - first [discriminate |
     destruct (occurrence_program_word F prog events) as [word|] eqn:Hw;[|discriminate];
     eapply occurrence_program_word_sound;exact Hw].
 - destruct (event_tail_case prog events) eqn:Htail.
   + apply event_tail_check_sound;[|exact H]. unfold event_tail_case in Htail.
     destruct (straight_event_program prog);[discriminate|exact Htail].
   + apply loop_events_check_sound;exact H.
Qed.


(* Every old successful multiple-event check remains accepted with its
   original safety predicate. Straight one-block and occurrence schedules
   keep their existing proof routes; eligible tails share one prefix. *)
Theorem event_progress_preserves : forall F prog events,
 loop_events_check F prog events=true -> event_progress_check F prog events=true.
Proof.
 intros F prog events Hcheck.
 pose proof (loop_events_check_sound F prog events Hcheck) as [_ [Hsafe _]].
 assert (Hclosed:short_all (phase_closed prog) events=true).
 { rewrite short_all_same,forallb_forall. intros event Hin.
   rewrite Forall_forall in Hsafe. specialize (Hsafe event Hin).
   exact (proj1 (proj2 Hsafe)). }
 unfold event_progress_check,event_progress_nonocc,occurrence_case.
 destruct (straight_event_program prog) eqn:Hstraight.
 - unfold event_tail_case. rewrite Hstraight. exact Hcheck.
 - rewrite Hclosed. destruct (event_tail_case prog events) eqn:Htail.
   + eapply event_tail_preserves;eassumption.
   + exact Hcheck.
Qed.

(* A bounded certificate for an event-free forward-branch segment. The
   endpoint abstract state is a proposal: every arriving state is checked
   against it. Both polarities of every guarded branch are propagated. *)
Definition loop_dag_successors (F:lfacts) (prog:program) (pos:pc)
 (st:astate) (i:instr) : list (pc*option astate) :=
 let out:=transfer F st i in
 let next:=next_pos prog (fst pos) (snd pos) in
 match i with
 | IBra g target=>phase_guard_edges F out g (target,0%nat) next
 | IExit g=>phase_guard_edges F out g (List.length prog,0%nat) next
 | _=>[(next,Some out)]
 end.
Definition loop_dag_forward (prog:program) (pos:pc) (i:instr) : bool :=
 match i with
 | IExit _=>false
 | IBra None target=>phase_after pos (target,0%nat)
 | IBra (Some _) target=>phase_after pos (target,0%nat) &&
     phase_after pos (next_pos prog (fst pos) (snd pos))
 | _=>phase_after pos (next_pos prog (fst pos) (snd pos))
 end.
Fixpoint loop_dag_all (fuel:nat) (F:lfacts) (prog:program)
 (protected:list reg) (events:list pc) (endpoint:pc) (post:astate)
 (pending:list (pc*option astate)) : bool :=
 match pending with
 | []=>true
 | (pos,ost)::tail=>match fuel with
   | O=>false
   | S rest=>match ost with
     | None=>loop_dag_all rest F prog protected events endpoint post tail
     | Some st=>if is_bottom st then
         loop_dag_all rest F prog protected events endpoint post tail
       else if phase_event endpoint pos then
         if state_le F st post then
           loop_dag_all rest F prog protected events endpoint post tail else false
       else if occurrence_member events pos then false
       else match fetch prog (fst pos) (snd pos) with
         | None=>false
         | Some i=>if wf_instr F i && short_all (fun r=>loop_preserves r i) protected && loop_dag_forward prog pos i then
           loop_dag_all rest F prog protected events endpoint post
             (loop_dag_successors F prog pos st i++tail)
           else false end
     end
   end
 end.
Definition loop_dag_check (fuel:nat) (F:lfacts) (prog:program)
 (protected:list reg) (events:list pc) (start endpoint:pc) (entry post:astate) :=
 if short_all (fun r=>negb(is_cell F r)) protected then
   loop_dag_all fuel F prog protected events endpoint post [(start,Some entry)]
 else false.

Lemma loop_dag_successors_complete : forall F prog E t s s' st i,
 env_ok F E -> fetch prog (blk s) (idx s)=Some i -> phase_step F prog E t s=Some s' ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) -> wf_instr F i=true ->
 exists out,In (pc_of s',Some out) (loop_dag_successors F prog (pc_of s) st i) /\
 state_ok F (S t) out (regs s') /\ regs_ok F E (regs s').
Proof.
 intros F prog E t s s' st i HE Hf Hstep Hst Hrs Hwf.
 destruct (transfer_sound F prog E HE t s s' st i Hf Hstep Hst Hrs Hwf) as [Ho Hr].
 pose proof (step_pos F prog E t s s' i Hf Hstep) as Hp.
 assert (Hbranch : forall g target,i=IBra g target ->
   pc_of s'=if holds g (regs s') then (target,0%nat) else next_pos prog (blk s) (idx s)).
 { intros g target Hi. subst i.
   pose proof (jump_regs F prog E t s s' _ Hf eq_refl Hstep) as Hreg.
   rewrite Hreg. unfold pc_of.
   destruct Hp as [[Hn [Hb He]]|[[g' [t' [Hi [Hg Hn]]]]|[g' [Hi [Hg Hn]]]]].
   - rewrite (Hb g target eq_refl). exact Hn.
   - injection Hi as <- <-. rewrite Hg. exact Hn.
   - discriminate. }
 assert (Hexit : forall g,i=IExit g ->
   pc_of s'=if holds g (regs s') then (List.length prog,0%nat) else next_pos prog (blk s) (idx s)).
 { intros g Hi. subst i.
   pose proof (jump_regs F prog E t s s' _ Hf eq_refl Hstep) as Hreg.
   rewrite Hreg. unfold pc_of.
   destruct Hp as [[Hn [Hb He]]|[[g' [t' [Hi [Hg Hn]]]]|[g' [Hi [Hg Hn]]]]].
   - rewrite (He g eq_refl). exact Hn.
   - discriminate.
   - injection Hi as <-. rewrite Hg. exact Hn. }
 destruct i; cbn [loop_dag_successors pc_of fst snd];
   try (eexists; split;[|split;[exact Ho|exact Hr]];
        left;destruct Hp as [[Hn _]|[[g0 [tg0 [Hbad _]]]|[g0 [Hbad _]]]];
        [unfold pc_of;rewrite Hn;reflexivity|discriminate|discriminate]).
 - destruct (phase_guard_complete F E HE (S t) (regs s') _ g (target,0%nat)
     (next_pos prog (blk s) (idx s)) Ho Hr) as [out [Hin Hout]].
   exists out. split;[|split;assumption]. rewrite (Hbranch _ _ eq_refl). exact Hin.
 - destruct (phase_guard_complete F E HE (S t) (regs s') _ g (List.length prog,0%nat)
     (next_pos prog (blk s) (idx s)) Ho Hr) as [out [Hin Hout]].
   exists out. split;[|split;assumption]. rewrite (Hexit _ eq_refl). exact Hin.
Qed.

Theorem loop_dag_all_sound : forall fuel F prog E protected events endpoint post pending pos st t s,
 env_ok F E -> (forall r, In r protected -> is_cell F r=false) ->
 loop_dag_all fuel F prog protected events endpoint post pending=true ->
 In (pos,Some st) pending -> pos=pc_of s ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 exists n last,loop_event_free_run F prog E events n t s last /\ pc_of last=endpoint /\
 state_ok F (n+t)%nat post (regs last) /\ regs_ok F E (regs last) /\
 forall r,In r protected -> regs last r=regs s r.
Proof.
 induction fuel as [|fuel IH];intros F prog E protected events endpoint post pending pos st t s
 HE Hcells Hcheck Hin Hpos Hst Hrs;
 destruct pending as [|[q ost] tail];[contradiction|discriminate|contradiction|].
 cbn [loop_dag_all] in Hcheck. destruct ost as [current|].
 - destruct (is_bottom current) eqn:Hbottom.
   + destruct Hin as [Heq|Hin].
     * injection Heq as <- <-. exfalso. exact (bottom_not_ok F t current _ Hbottom Hst).
     * eapply IH;eassumption.
   + destruct (phase_event endpoint q) eqn:Hend.
     * destruct (state_le F current post) eqn:Hle;[|discriminate].
       destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. exists 0%nat,s. split;[constructor|]. split.
          ++ apply phase_event_eq in Hend. rewrite <-Hpos. exact Hend.
          ++ split;[eapply state_le_sound;eassumption|]. split;[exact Hrs|]. intros r Hr;reflexivity.
       -- eapply IH;eassumption.
     * destruct (occurrence_member events q) eqn:Hfree;[discriminate|].
       destruct (fetch prog (fst q) (snd q)) as [i|] eqn:Hfetch;[|discriminate].
       destruct (wf_instr F i && short_all (fun r=>loop_preserves r i) protected && loop_dag_forward prog q i)
         eqn:Hlocal;[|discriminate].
       rewrite !andb_true_iff in Hlocal. destruct Hlocal as [[Hwf Hprotected] Hforward].
       rewrite short_all_same,forallb_forall in Hprotected.
       destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. subst q.
          assert (Hf:fetch prog (blk s) (idx s)=Some i) by exact Hfetch.
          assert (Hex:exists mid,phase_step F prog E t s=Some mid).
          { unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
          destruct Hex as [mid Hstep].
          destruct (loop_dag_successors_complete F prog E t s mid current i HE Hf Hstep Hst Hrs Hwf)
            as [child [Hchild [Hout Hregs]]].
          destruct (IH F prog E protected events endpoint post
            (loop_dag_successors F prog (pc_of s) current i++tail) (pc_of mid) child (S t) mid
            HE Hcells Hcheck (in_or_app _ _ _ (or_introl Hchild)) eq_refl Hout Hregs)
            as [n [last [Hrun [Hlast [Hpost [Hrs' Hpres]]]]]].
          exists (S n),last. split;[econstructor;eassumption|]. split;[exact Hlast|].
          split;[replace (S n+t)%nat with (n+S t)%nat by lia;exact Hpost|].
          split;[exact Hrs'|]. intros r Hr. rewrite (Hpres r Hr).
          eapply loop_step_preserves;[exact (Hcells r Hr)|exact Hf|exact Hstep|exact (Hprotected r Hr)].
       -- eapply IH;[exact HE|exact Hcells|exact Hcheck| |exact Hpos|exact Hst|exact Hrs].
          apply in_or_app;right;exact Hin.
 - destruct Hin as [Heq|Hin];[discriminate|]. eapply IH;eassumption.
Qed.
Theorem loop_dag_check_sound : forall fuel F prog E protected events start endpoint entry post t s,
 env_ok F E -> loop_dag_check fuel F prog protected events start endpoint entry post=true ->
 start=pc_of s -> state_ok F t entry (regs s) -> regs_ok F E (regs s) ->
 exists n last,loop_event_free_run F prog E events n t s last /\ pc_of last=endpoint /\
 state_ok F (n+t)%nat post (regs last) /\ regs_ok F E (regs last) /\
 forall r,In r protected -> regs last r=regs s r.
Proof.
 intros fuel F prog E protected events start endpoint entry post t s HE Hcheck Hpos Hst Hrs.
 unfold loop_dag_check in Hcheck.
 destruct (short_all (fun r=>negb(is_cell F r)) protected) eqn:Hcells;[|discriminate].
 rewrite short_all_same,forallb_forall in Hcells.
 eapply loop_dag_all_sound;[exact HE| |exact Hcheck|left;reflexivity|exact Hpos|exact Hst|exact Hrs].
 intros r Hr. apply negb_true_iff. exact (Hcells r Hr).
Qed.

Lemma loop_event_free_app : forall F prog E events n m t s mid last,
 loop_event_free_run F prog E events n t s mid ->
 loop_event_free_run F prog E events m (n+t)%nat mid last ->
 loop_event_free_run F prog E events (n+m)%nat t s last.
Proof.
 intros F prog E events n m t s mid last H. revert m last.
 induction H;intros m final Htail;[exact Htail|].
 cbn [Nat.add]. econstructor;[exact H|exact H0|].
 apply IHloop_event_free_run. replace (n+S t)%nat with (S n+t)%nat by lia. exact Htail.
Qed.

(* The arithmetic descriptor is reused at its actual increment and branch.
   Prefix and suffix certificates protect the counter. Stride may be
   computed in the prefix: positivity and nonwrap are checked at increment. *)
Record loop_dag_certificate := {
 ldc_arithmetic : loop_info;
 ldc_increment : pc;
 ldc_branch : pc;
 ldc_before : astate;
 ldc_after : astate
}.
Definition loop_dag_iteration (fuel:nat) (F:lfacts) (prog:program)
 (events:list pc) (head:pc) (cert:loop_dag_certificate) (inv:astate)
 : option (pc*option astate*option astate) :=
 let d:=ldc_arithmetic cert in
 let inc:=ldc_increment cert in
 let jump:=ldc_branch cert in
 let before:=ldc_before cert in
 let after:=ldc_after cert in
 if loop_dag_check fuel F prog [li_counter d] events head inc inv before then
 if occurrence_member events inc then None else
 if loop_increment_ok F prog inc before d then
 if loop_dag_check fuel F prog [li_counter d] events
    (next_pos prog (fst inc) (snd inc)) jump
    (transfer F before (loop_counter_step d)) after then
 if occurrence_member events jump then None else
 if loop_branch_ok F prog head jump d then
   let '(back,leave):=loop_branches F after d head in
   Some(next_pos prog (fst jump) (snd jump),back,leave)
 else None else None else None else None.
Definition loop_dag_validate (fuel:nat) (F:lfacts) (prog:program)
 (events:list pc) (head:pc) (cert:loop_dag_certificate) (inv:astate)
 : option (pc*option astate) :=
 match reg_range F inv (li_counter (ldc_arithmetic cert)),
       loop_dag_iteration fuel F prog events head cert inv with
 | Some _,Some(exit,back,leave)=>
   if match back with None=>true|Some st=>state_le F st inv end
   then Some(exit,leave) else None
 | _,_=>None end.

Theorem loop_dag_iteration_sound : forall fuel F prog E events head cert inv exit back leave t s,
 env_ok F E -> loop_dag_iteration fuel F prog events head cert inv=Some(exit,back,leave) ->
 head=pc_of s -> state_ok F t inv (regs s) -> regs_ok F E (regs s) ->
 exists (n:nat) (last:state) (out:astate) (taken:bool),
 (0<n)%nat /\ loop_event_free_run F prog E events n t s last /\
 (if taken then back else leave)=Some out /\ pc_of last=(if taken then head else exit) /\
 state_ok F (n+t)%nat out (regs last) /\ regs_ok F E (regs last) /\
 regs s (li_counter (ldc_arithmetic cert))<regs last (li_counter (ldc_arithmetic cert)).
Proof.
 intros fuel F prog E events head [d inc jump before after] inv exit back leave t s HE Hiter Hpos Hst Hrs.
 unfold loop_dag_iteration in Hiter.
 cbn [ldc_arithmetic ldc_increment ldc_branch ldc_before ldc_after] in Hiter |- *.
 destruct (loop_dag_check fuel F prog [li_counter d] events head inc inv before) eqn:Hprefix;[|discriminate].
 destruct (occurrence_member events inc) eqn:Hincfree;[discriminate|].
 destruct (loop_increment_ok F prog inc before d) eqn:Hinc;[|discriminate].
 destruct (loop_dag_check fuel F prog [li_counter d] events
  (next_pos prog (fst inc) (snd inc)) jump (transfer F before (loop_counter_step d)) after)
  eqn:Hsuffix;[|discriminate].
 destruct (occurrence_member events jump) eqn:Hjumpfree;[discriminate|].
 destruct (loop_branch_ok F prog head jump d) eqn:Hbranch;[|discriminate].
 destruct (loop_branches F after d head) as [b l] eqn:Hbranches. injection Hiter as <- <- <-.
 destruct (loop_dag_check_sound fuel F prog E [li_counter d] events head inc inv before t s
   HE Hprefix Hpos Hst Hrs) as [n1 [s1 [Hrun1 [Hpc1 [Hst1 [Hrs1 Hpres1]]]]]].
 destruct (loop_increment_sound F prog E inc before d (n1+t)%nat s1
   HE Hinc (eq_sym Hpc1) Hst1 Hrs1)
   as [s2 [Hstep2 [Hpc2 [Hst2 [Hrs2 [Hc2 [Hs2 Hpositive]]]]]]].
 destruct (loop_dag_check_sound fuel F prog E [li_counter d] events
   (next_pos prog (fst inc) (snd inc)) jump (transfer F before (loop_counter_step d)) after
   (S(n1+t)) s2 HE Hsuffix (eq_sym Hpc2) Hst2 Hrs2)
   as [n3 [s3 [Hrun3 [Hpc3 [Hst3 [Hrs3 Hpres3]]]]]].
 destruct (loop_branch_sound F prog E head jump after d (n3+S(n1+t))%nat s3
   HE Hbranch (eq_sym Hpc3) Hst3 Hrs3)
   as [s4 [out [taken [Hstep4 [Hout [Hpc4 [Hst4 [Hrs4 Hregs4]]]]]]]].
 exists (n1+1+n3+1)%nat,s4,out,taken. split;[lia|]. split.
 - apply loop_event_free_app with (mid:=s3).
   + apply loop_event_free_app with (mid:=s2).
     * apply loop_event_free_app with (mid:=s1);[exact Hrun1|].
       econstructor;[rewrite Hpc1;exact Hincfree|exact Hstep2|constructor].
     * replace (n1+1+t)%nat with (S(n1+t)) by lia. exact Hrun3.
   + replace (n1+1+n3+t)%nat with (n3+S(n1+t))%nat by lia.
     econstructor;[rewrite Hpc3;exact Hjumpfree|exact Hstep4|constructor].
 - rewrite Hbranches in Hout. cbn [fst snd] in Hout. split;[exact Hout|].
   split;[exact Hpc4|]. split.
   + replace (n1+1+n3+1+t)%nat with (S(n3+S(n1+t))) by lia. exact Hst4.
   + split;[exact Hrs4|]. rewrite Hregs4,(Hpres3 _ (or_introl eq_refl)),Hc2,
       (Hpres1 _ (or_introl eq_refl)). lia.
Qed.

Theorem loop_dag_validate_progress : forall fuel F prog E events head cert inv exit ost,
 env_ok F E -> loop_dag_validate fuel F prog events head cert inv=Some(exit,ost) ->
 forall t s,head=pc_of s -> state_ok F t inv (regs s) -> regs_ok F E (regs s) ->
 exists n last out,loop_event_free_run F prog E events n t s last /\
 pc_of last=exit /\ ost=Some out /\ state_ok F (n+t)%nat out (regs last) /\ regs_ok F E (regs last).
Proof.
 intros fuel F prog E events head cert inv exit ost HE Hval. unfold loop_dag_validate in Hval.
 destruct (reg_range F inv (li_counter (ldc_arithmetic cert))) as [[lo hi]|] eqn:Hbound;[|discriminate].
 destruct (loop_dag_iteration fuel F prog events head cert inv) as [[[next back] leave]|] eqn:Hiter;[|discriminate].
 destruct (match back with None=>true|Some st=>state_le F st inv end) eqn:Hback;[|discriminate].
 injection Hval as <- <-.
 assert (Hind : forall rank t s,
 loop_rank hi (ldc_arithmetic cert) (regs s)=rank -> head=pc_of s ->
 state_ok F t inv (regs s) -> regs_ok F E (regs s) ->
 exists n last out,loop_event_free_run F prog E events n t s last /\
 pc_of last=next /\ leave=Some out /\ state_ok F (n+t)%nat out (regs last) /\ regs_ok F E (regs last)).
 { intro rank. induction rank using lt_wf_ind. intros t s Hrank Hpos Hst Hrs.
   destruct (loop_dag_iteration_sound fuel F prog E events head cert inv next back leave t s
     HE Hiter Hpos Hst Hrs)
    as [n1 [s1 [out [taken [Hpositive [Hrun [Hout [Hpc [Hst1 [Hrs1 Hinc]]]]]]]]]].
   destruct taken.
   - destruct back as [back|];[|discriminate]. injection Hout as <-.
     assert (Hinv:state_ok F (n1+t)%nat inv (regs s1)).
     { eapply state_le_sound;[exact HE|exact Hrs1|exact Hst1|exact Hback]. }
     pose proof (reg_range_sound F E HE (n1+t)%nat inv (regs s1) Hinv Hrs1 _ _ _ Hbound) as Hb.
     assert (Hdec:(loop_rank hi (ldc_arithmetic cert) (regs s1)<rank)%nat).
     { rewrite <-Hrank. apply loop_rank_decreases;[exact Hinc|lia]. }
     destruct (H _ Hdec (n1+t)%nat s1 eq_refl (eq_sym Hpc) Hinv Hrs1)
       as [n [last [output [Hlast [Hlastpc [Hleave [Hfinal Hregsfinal]]]]]]].
     exists (n1+n)%nat,last,output. split;[eapply loop_event_free_app;eassumption|].
     split;[exact Hlastpc|]. split;[exact Hleave|]. split;[|exact Hregsfinal].
     replace (n1+n+t)%nat with (n+(n1+t))%nat by lia. exact Hfinal.
   - exists n1,s1,out. split;[exact Hrun|]. split;[exact Hpc|]. split;[exact Hout|]. split;assumption.
 }
 intros t s Hpos Hst Hrs. exact (Hind (loop_rank hi (ldc_arithmetic cert) (regs s)) t s eq_refl Hpos Hst Hrs).
Qed.

(* Descriptor choice is structural. Existing straight descriptors retain
   their validator. A discovered DAG descriptor is never retried through a
   different validator after a failed obligation. All proposals are checked. *)
Record loop_dag_shape := {
 lds_arithmetic:loop_info;
 lds_increment:pc;
 lds_branch:pc;
 lds_limit:opnd;
 lds_signed:bool
}.
Fixpoint loop_dag_increment_at (counter:reg) (block index:nat) (body:list instr)
 : option (pc*reg*reg) :=
 match body with
 | []=>None
 | i::tail=>match i with
   | IBin None BAdd 64 dst src (OReg stride)=>
     if Pos.eqb dst counter then Some((block,index),src,stride)
     else loop_dag_increment_at counter block (S index) tail
   | _=>loop_dag_increment_at counter block (S index) tail end
 end.
Definition loop_dag_shape_block (head:pc) (block:nat) (body:list instr) : option loop_dag_shape :=
 match rev body with
 | IBra (Some(false,p)) target::ISetp None CLt 64 sg q counter limit::_=>
   if Nat.eqb target (fst head) && Nat.eqb (snd head) 0 && Pos.eqb p q then
   match loop_dag_increment_at counter block 0 body with
   | Some(inc,src,stride)=>Some {|lds_arithmetic:={|li_prefix:=0;li_suffix:=0;
       li_counter:=counter;li_source:=src;li_stride:=stride;li_predicate:=p;li_negated:=false|};
       lds_increment:=inc;lds_branch:=(block,Nat.pred(List.length body));lds_limit:=limit;lds_signed:=sg|}
   | None=>None end else None
 | _=>None end.
Fixpoint loop_dag_shape_scan (head:pc) (block:nat) (bodies:program) : option loop_dag_shape :=
 match bodies with
 | []=>None
 | body::tail=>match loop_dag_shape_block head block body with
   | Some d=>Some d
   | None=>loop_dag_shape_scan head (S block) tail end
 end.
Definition loop_dag_at (prog:program) (head:pc) :=
 if Nat.eqb (snd head) 0 then loop_dag_shape_scan head 0 prog else None.
Inductive loop_summary_descriptor :=
 | Loop_straight_descriptor (d:loop_info)
 | Loop_dag_descriptor (d:loop_dag_shape).
(* A direct final comparison identifies the induction value relevant to the
   backedge. Keep the existing straight descriptor when it already uses
   either compared operand; otherwise choose an available checked DAG shape
   before proposing or validating anything. *)
Definition loop_counter_compared (prog:program) (head:pc) (d:loop_info) : bool :=
 match last (loop_trace (li_prefix d+li_suffix d+1) prog head) INop with
 | ISetp None _ 64 _ p a b=>
   if Pos.eqb p (li_predicate d) then
     Pos.eqb a (li_counter d) ||
       match b with OReg r=>Pos.eqb r (li_counter d)|_=>false end
   else true
 | _=>true end.
Definition loop_summary_at (prog:program) (head:pc) : option loop_summary_descriptor :=
 match loop_at prog head with
 | Some d=>if loop_counter_compared prog head d then Some(Loop_straight_descriptor d)
   else match loop_dag_at prog head with
     | Some shape=>Some(Loop_dag_descriptor shape)
     | None=>Some(Loop_straight_descriptor d) end
 | None=>match loop_dag_at prog head with Some d=>Some(Loop_dag_descriptor d)|None=>None end
 end.
Theorem loop_summary_straight_preserved : forall prog head d,
 loop_at prog head=Some d -> loop_counter_compared prog head d=true ->
 loop_summary_at prog head=Some(Loop_straight_descriptor d).
Proof. intros prog head d H Hcompared. unfold loop_summary_at. rewrite H,Hcompared. reflexivity. Qed.
Theorem loop_summary_compared_selected : forall prog head old shape,
 loop_at prog head=Some old -> loop_counter_compared prog head old=false ->
 loop_dag_at prog head=Some shape ->
 loop_summary_at prog head=Some(Loop_dag_descriptor shape).
Proof. intros prog head old shape Hold Hcounter Hshape.
 unfold loop_summary_at. rewrite Hold,Hcounter,Hshape. reflexivity. Qed.
Definition loop_dag_selected (prog:program) :=
 existsb(fun b=>match loop_summary_at prog (b,0%nat) with
   Some(Loop_dag_descriptor _)=>true|_=>false end)(seq 0(List.length prog)).

(* Inference is deliberately separate from authority. A finite proposal
   traversal collects potential endpoint states; the DAG checker traverses
   again and validates every proposed endpoint containment. *)
(* Stop proposal at the same non-forward edges that its DAG validator
   rejects, before spending the exploration budget inside a nested cycle. *)
Fixpoint loop_dag_collect (fuel:nat)(F:lfacts)(prog:program)(endpoint:pc)
 (pending:list(pc*option astate)) : option(list astate) :=
 match pending with
 | []=>Some []
 | (pos,ost)::tail=>match fuel with
   | O=>None
   | S rest=>match ost with
     | None=>loop_dag_collect rest F prog endpoint tail
     | Some st=>if is_bottom st then loop_dag_collect rest F prog endpoint tail else
       if phase_event endpoint pos then
         match loop_dag_collect rest F prog endpoint tail with Some rest=>Some(st::rest)|None=>None end
       else match fetch prog(fst pos)(snd pos)with
         | Some i=>if loop_dag_forward prog pos i then
             loop_dag_collect rest F prog endpoint(loop_dag_successors F prog pos st i++tail)
           else None
         | None=>None end
     end
   end
 end.
Definition loop_dag_post F prog endpoint start st :=
 match loop_dag_collect phase_fuel F prog endpoint[(start,Some st)] with
 | Some(first::tail)=>Some(fold_left(fun out st=>join_states F out st false false)tail first)
 | _=>None end.
Definition loop_dag_propose F prog head (shape:loop_dag_shape) st
 : option(astate*loop_dag_certificate) :=
 let d:=lds_arithmetic shape in
 match cmp_range F st(lds_limit shape)64(lds_signed shape)with
 | None=>None
 | Some(_,hi)=>
   let body:=List.concat(firstn(S(fst(lds_branch shape))-fst head)(skipn(fst head)prog))in
   let forgotten:=fold_left(fun s r=>assign F s r None None)(flat_map loop_writes body)st in
   let inv:={|amap:=PM.add(li_counter d)(FForm(range_form 64 0(Z.max 0 hi))0 None)
     (amap forgotten);acons:=acons forgotten|}in
   let inc:=lds_increment shape in
   match loop_dag_post F prog inc head inv with
   | None=>None
   | Some before=>match loop_dag_post F prog(lds_branch shape)(next_pos prog(fst inc)(snd inc))
       (transfer F before(loop_counter_step d))with
     | None=>None
     | Some after=>Some(inv,{|ldc_arithmetic:=d;ldc_increment:=inc;ldc_branch:=lds_branch shape;
                            ldc_before:=before;ldc_after:=after|})end
   end
 end.
Inductive loop_summary_certificate :=
 | Loop_straight_certificate (d:loop_info)
 | Loop_dag_certificate (cert:loop_dag_certificate).
Definition loop_summary_propose F prog head descriptor st
 : option(astate*loop_summary_certificate) :=
 match descriptor with
 | Loop_straight_descriptor d=>match loop_propose F prog head d st with
   | Some inv=>Some(inv,Loop_straight_certificate d)|None=>None end
 | Loop_dag_descriptor d=>match loop_dag_propose F prog head d st with
   | Some(inv,cert)=>Some(inv,Loop_dag_certificate cert)|None=>None end
 end.
Definition loop_summary_validate F prog events head certificate inv :=
 match certificate with
 | Loop_straight_certificate d=>loop_validate F prog head d inv
 | Loop_dag_certificate cert=>loop_dag_validate phase_fuel F prog events head cert inv
 end.
Theorem loop_summary_legacy_validator : forall F prog events head d inv,
 loop_summary_validate F prog events head(Loop_straight_certificate d)inv=
 loop_validate F prog head d inv.
Proof. reflexivity. Qed.
Theorem loop_summary_validate_progress : forall F prog E events head cert inv exit ost,
 env_ok F E -> loop_summary_validate F prog events head cert inv=Some(exit,ost) ->
 forall t s,head=pc_of s -> state_ok F t inv(regs s) -> regs_ok F E(regs s) ->
 exists n last out,phase_run F prog E n t s=Some last /\ pc_of last=exit /\
 ost=Some out /\ state_ok F(n+t)%nat out(regs last) /\ regs_ok F E(regs last).
Proof.
 intros F prog E events head [d|cert] inv exit ost HE Hval t s Hpos Hst Hrs.
 - eapply loop_validate_progress;eassumption.
 - destruct (loop_dag_validate_progress phase_fuel F prog E events head cert inv exit ost
    HE Hval t s Hpos Hst Hrs) as [n [last [out [Hrun Hrest]]]].
   exists n,last,out. split;[eapply loop_event_free_actual_run;exact Hrun|exact Hrest].
Qed.

Fixpoint ranked_summary_all (events:list pc) (keep : PM.t unit) (fuel : nat) (F : lfacts) (prog : program)
 (event : pc -> bool) (pending : list (pc * option astate)) : bool :=
 match pending with
 | []=>true
 | (pos,ost)::tail=>match fuel with O=>false|S rest=>match ost with
   | None=>ranked_summary_all events keep rest F prog event tail
   | Some st=>if is_bottom st then ranked_summary_all events keep rest F prog event tail else
     if event pos then ranked_summary_all events keep rest F prog event tail else
     match loop_summary_at prog pos with
     | Some d=>match loop_summary_propose F prog pos d st with
       | None=>false
       | Some (inv,certificate)=>if state_le F st inv then match loop_summary_validate F prog events pos certificate inv with
         | None=>false | Some edge=>ranked_summary_all events keep rest F prog event (edge::tail) end else false
       end
     | None=>match fetch prog (fst pos) (snd pos) with
       | None=>false|Some i=>if wf_instr F i then
         ranked_summary_all events keep rest F prog event (phase_successors keep F prog pos st i++tail)
         else false end
     end end end end.

Theorem ranked_summary_all_progress : forall F prog E fuel events keep event pending pos st t s,
 env_ok F E -> ranked_summary_all events keep fuel F prog event pending=true ->
 In (pos,Some st) pending -> pos=pc_of s -> state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 phase_reaches F prog E event t s.
Proof.
 intros F prog E fuel. induction fuel as [|fuel IH];intros events keep event pending pos st t s HE Hcheck Hin Hpos Hst Hrs;
 destruct pending as [|[q ost] tail];[contradiction|discriminate|contradiction|].
 cbn [ranked_summary_all] in Hcheck. destruct ost as [current|].
 - destruct (is_bottom current) eqn:Hbottom.
   + destruct Hin as [Heq|Hin].
     * injection Heq as <- <-. exfalso. exact (bottom_not_ok F t current _ Hbottom Hst).
     * eapply IH;eassumption.
   + destruct (event q) eqn:Hevent.
     * destruct Hin as [Heq|Hin].
       -- injection Heq as <- <-. apply Phase_here. rewrite <-Hpos. exact Hevent.
       -- eapply IH;eassumption.
     * destruct (loop_summary_at prog q) as [d|] eqn:Hloop.
       -- destruct (loop_summary_propose F prog q d current) as [[inv certificate]|];[|discriminate].
          destruct (state_le F current inv) eqn:Hle;[|discriminate].
          destruct (loop_summary_validate F prog events q certificate inv) as [[exit os]|] eqn:Hval;[|discriminate].
          destruct Hin as [Heq|Hin].
          ++ injection Heq as <- <-.
             assert (Hinv : state_ok F t inv (regs s)) by (eapply state_le_sound;eassumption).
             destruct (loop_summary_validate_progress F prog E events q certificate inv exit os HE Hval t s Hpos Hinv Hrs)
               as [n [last [out [Hrun [Hpc [Hos [Hout Hregs]]]]]]].
             eapply loop_run_reaches;[exact Hrun|]. eapply IH;[exact HE|exact Hcheck| |symmetry;exact Hpc|exact Hout|exact Hregs].
             rewrite Hos. left;reflexivity.
          ++ eapply IH;[exact HE|exact Hcheck|right;exact Hin|exact Hpos|exact Hst|exact Hrs].
       -- destruct (fetch prog (fst q) (snd q)) as [i|] eqn:Hfetch;[|discriminate].
          destruct (wf_instr F i) eqn:Hwf;[|discriminate].
          destruct Hin as [Heq|Hin].
          ++ injection Heq as <- <-. subst q.
             assert (Hf : fetch prog (blk s) (idx s)=Some i) by exact Hfetch.
             assert (Hex : exists s',phase_step F prog E t s=Some s').
             { unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
             destruct Hex as [s' Hstep].
             destruct (phase_successors_complete F prog E HE keep t s s' current i Hf Hstep Hst Hrs Hwf)
               as [out [Hchild [Hout Hregs]]].
             eapply Phase_next;[exact Hevent|exact Hstep|].
             eapply IH;[exact HE|exact Hcheck| |reflexivity|exact Hout|exact Hregs].
             apply in_or_app;left;exact Hchild.
          ++ eapply IH;[exact HE|exact Hcheck| |exact Hpos|exact Hst|exact Hrs]. apply in_or_app;right;exact Hin.
 - destruct Hin as [Heq|Hin];[discriminate|]. eapply IH;eassumption.
Qed.


Definition ranked_summary_progress F prog events event :=
 ranked_summary_all events (phase_controls prog event) phase_fuel F prog (phase_event event)
 [((0%nat,0%nat),Some(empty_state F))].
Lemma ranked_summary_safe : forall F prog events event,
 ranked_phase_ready F prog event=true -> ranked_summary_progress F prog events event=true ->
 single_phase_safe F prog event.
Proof.
 intros F prog events event Hready Hcheck. unfold ranked_phase_ready in Hready.
 destruct (lf_ok F) eqn:Hlf;[|discriminate].
 destruct (phase_event_at prog event) eqn:Hmarker;[|discriminate].
 split;[exact Hmarker|]. split;[exact Hready|]. intros E Hlaunch.
 unfold ranked_summary_progress in Hcheck. eapply ranked_summary_all_progress.
 - exact(env_of_launch F prog E Hlf Hlaunch).
 - exact Hcheck.
 - left;reflexivity.
 - reflexivity.
 - split.
   + intros r f Hf. unfold get,empty_state in Hf;cbn [amap] in Hf. rewrite PM.gempty in Hf. discriminate.
   + exact(lk_derived F E Hlaunch).
 - split;[exact(lk_init F E Hlaunch)|]. intros r Hfixed;reflexivity.
Qed.
Definition loop_summaries_check F prog events :=
 if loop_events_adjacent events then
 if short_all(ranked_phase_ready F prog)events then
 if event_tail_case prog events then
   match events with first::_=>ranked_summary_progress F prog events first|[]=>false end
 else short_all(ranked_summary_progress F prog events)events
 else false else false.
Theorem loop_summaries_check_sound : forall F prog events,
 loop_summaries_check F prog events=true -> loop_events_safe F prog events.
Proof.
 intros F prog events Hcheck. unfold loop_summaries_check in Hcheck.
 destruct(loop_events_adjacent events)eqn:Horder;[|discriminate].
 destruct(short_all(ranked_phase_ready F prog)events)eqn:Hready;[|discriminate].
 rewrite loop_events_adjacent_exact in Horder.
 rewrite short_all_same,forallb_forall in Hready.
 assert(Hsafe:forall event,In event events->single_phase_safe F prog event).
 { destruct(event_tail_case prog events)eqn:Htail.
   - unfold event_tail_case in Htail. destruct(straight_event_program prog);[discriminate|].
     destruct events as[|first tail];[discriminate|].
     assert(Hfirst:single_phase_safe F prog first).
     { eapply ranked_summary_safe;[exact(Hready first(or_introl eq_refl))|exact Hcheck]. }
     intros event Hin. specialize(Hready event Hin). unfold ranked_phase_ready in Hready.
     destruct(lf_ok F);[|discriminate]. destruct(phase_event_at prog event)eqn:Hmarker;[|discriminate].
     eapply event_tail_safe;eassumption.
   - rewrite short_all_same,forallb_forall in Hcheck. intros event Hin.
     eapply ranked_summary_safe;[exact(Hready event Hin)|exact(Hcheck event Hin)]. }
 split;[exact Horder|]. split;[apply Forall_forall;exact Hsafe|].
 intros E Hlaunch. split.
 - intros event Hin. apply loop_reaches_run. exact(proj2(proj2(Hsafe event Hin))E Hlaunch).
 - split.
   + intros e1 e2 n1 n2 s1 s2 H1 H2 Hord Hr1 Hp1 Hr2 Hp2.
     destruct(Hsafe e2 H2)as[Hf[Hc _]]. eapply loop_hit_order;eassumption.
   + intros event n1 n2 s1 s2 Hin Hr1 Hp1 Hr2 Hp2.
     destruct(Hsafe event Hin)as[Hf[Hc _]]. eapply loop_hit_unique;eassumption.
Qed.

(* A structural descriptor commits to its checked certificate. A failed
   proposal, entry containment, or event-free proof cannot fall through to
   ordinary instruction exploration. *)
(* The certificate chooses an equality modulus. The checker requires it to
   cover the full source width; older descriptors retain their 64-bit witness. *)
Record signed_loop_step := {
 sl_width:nat; sl_alias_width:nat; sl_counter:reg; sl_source:reg; sl_operand:opnd; sl_down:bool
}.
Definition signed_loop_instr d := IBin None BAdd (sl_width d) (sl_counter d) (sl_source d) (sl_operand d).
Definition signed_loop_value d (rs:reg->Z) :=
 let v:=sext(sl_width d)(rs(sl_counter d)) in if sl_down d then -v else v.
Definition signed_loop_alias d := FForm(fatom (Z.of_nat(sl_alias_width d)) (sl_counter d,if Nat.eqb(sl_alias_width d)64 then EX else EZ)1)0 None.
Definition signed_loop_source_equal F st d :=
 if Pos.eqb(sl_source d)(sl_counter d) then true
 else register_fact_le F st(sl_source d)(signed_loop_alias d).
Definition signed_loop_step_ok F prog pos st d :=
 loop_fetch_is prog pos(signed_loop_instr d) && wf_instr F(signed_loop_instr d) &&
 (1 <=? Z.of_nat(sl_width d)) && (Z.of_nat(sl_width d) <=? 64) &&
 Nat.eqb(wdn F(sl_counter d))(sl_width d) &&
 Nat.eqb(wdn F(sl_source d))(sl_width d) &&
 opnd_width_ok F(sl_operand d)(Z.of_nat(sl_width d)) &&
 (Z.of_nat(sl_width d) <=? Z.of_nat(sl_alias_width d)) && (Z.of_nat(sl_alias_width d) <=? 64) &&
 signed_loop_source_equal F st d &&
 match cmp_range F st(OReg(sl_counter d))(Z.of_nat(sl_width d))true,
       cmp_range F st(sl_operand d)(Z.of_nat(sl_width d))true with
 | Some(cl,ch),Some(sl,sh)=>
   (if sl_down d then sh <? 0 else 0 <? sl) &&
   (-2^(Z.of_nat(sl_width d)-1) <=? cl+sl) &&
   (ch+sh <? 2^(Z.of_nat(sl_width d)-1))
 | _,_=>false end.
Lemma signed_loop_source_sound : forall F E t st rs d,
 env_ok F E -> state_ok F t st rs -> regs_ok F E rs ->
 (1<=sl_width d<=sl_alias_width d)%nat ->
 wdn F(sl_counter d)=sl_width d -> wdn F(sl_source d)=sl_width d ->
 signed_loop_source_equal F st d=true -> rs(sl_source d)=rs(sl_counter d).
Proof.
 intros F E t st rs d HE Hst Hrs Hw Hc Ha H.
 unfold signed_loop_source_equal in H.
 destruct(Pos.eqb(sl_source d)(sl_counter d))eqn:Heq.
 {apply Pos.eqb_eq in Heq. now rewrite Heq. }
 pose proof(register_fact_le_sound F E HE t rs Hrs st(sl_source d)(signed_loop_alias d)Hst H)as Hfact.
 unfold signed_loop_alias,fact_okv in Hfact. destruct Hfact as[_[Hsat _]].
 unfold fatom in Hsat. apply sat_mk_inv in Hsat;[|lia]. destruct Hsat as[k[Hk Heqm]].
 assert(Hatom:atom_val(wdn F)rs(sl_counter d,if Nat.eqb(sl_alias_width d)64 then EX else EZ)=rs(sl_counter d))
 by(destruct(Nat.eqb(sl_alias_width d)64);reflexivity).
 unfold tsum in Heqm. cbn [fold_right fst snd] in Heqm. rewrite Hatom in Heqm.
 assert(k=0)by lia. subst k.
 replace(1*rs(sl_counter d)+0+0)with(rs(sl_counter d))in Heqm by ring.
 pose proof(rs_range F E rs Hrs(sl_counter d))as Hcr.
 pose proof(rs_range F E rs Hrs(sl_source d))as Har. rewrite Hc in Hcr. rewrite Ha in Har.
 eapply eqm_small_eq;[|exact Har|exact Hcr|exact Heqm]. lia.
Qed.
(* The checked signed addition gives a tighter counter interval than the
   generic modular transfer can retain when the interval crosses zero. *)
Definition signed_loop_step_base F st d :=
 let out:=transfer F st(signed_loop_instr d)in
 match cmp_range F st(OReg(sl_counter d))(Z.of_nat(sl_width d))true,
       cmp_range F st(sl_operand d)(Z.of_nat(sl_width d))true with
 | Some(cl,ch),Some(sl,sh)=>put out(sl_counter d)
   (FForm(range_form(Z.of_nat(sl_width d))(cl+sl)(ch+sh))
     (reg_align F out(sl_counter d))None)
 | _,_=>out end.
(* The checked nonwrapping addition also relates surviving operands to
   the new signed counter. A write to either operand later kills this fact. *)
Definition signed_loop_step_relation st d:=
 if Pos.eqb(sl_source d)(sl_counter d)then st else
 match sl_operand d with
 |OReg r=>if Pos.eqb r(sl_counter d)then st else
   add_cons st(sorted_terms[((sl_counter d,ES),1);((sl_source d,ES),-1);((r,ES),-1)],0,0)
 |OImm z=>let step:=sext(sl_width d)z in
   add_cons st(sorted_terms[((sl_counter d,ES),1);((sl_source d,ES),-1)],step,step)
 end.
Definition signed_loop_step_post (conjunction:bool) F st d:=
 let out:=signed_loop_step_base F st d in
 if conjunction then signed_loop_step_relation out d else out.
Lemma signed_loop_step_sound : forall conjunction F prog E pos st d t s,
 env_ok F E -> signed_loop_step_ok F prog pos st d=true -> pos=pc_of s ->
 state_ok F t st(regs s) -> regs_ok F E(regs s) ->
 exists s',phase_step F prog E t s=Some s' /\
 pc_of s'=next_pos prog(fst pos)(snd pos) /\
 state_ok F(S t)(signed_loop_step_post conjunction F st d)(regs s') /\ regs_ok F E(regs s') /\
 signed_loop_value d(regs s)<signed_loop_value d(regs s').
Proof.
 intros conjunction F prog E pos st d t s HE Hcheck Hpos Hst Hrs.
 unfold signed_loop_step_ok in Hcheck. rewrite !andb_true_iff in Hcheck.
 assert(Hf:fetch prog(fst pos)(snd pos)=Some(signed_loop_instr d))by(apply loop_fetch_is_sound;tauto).
 assert(Hwf:wf_instr F(signed_loop_instr d)=true)by tauto.
 assert(Hw:1<=Z.of_nat(sl_width d)<=64)by(rewrite <-!Z.leb_le;tauto).
 assert(Hc:wdn F(sl_counter d)=sl_width d)by(apply Nat.eqb_eq;tauto).
 assert(Ha:wdn F(sl_source d)=sl_width d)by(apply Nat.eqb_eq;tauto).
 assert(Ho:owidth F(sl_operand d)(Z.of_nat(sl_width d))).
 {unfold owidth. intros r Hr.
  assert(Hwidth:opnd_width_ok F(sl_operand d)(Z.of_nat(sl_width d))=true)by tauto.
  rewrite Hr in Hwidth. unfold opnd_width_ok in Hwidth. apply Z.eqb_eq. exact Hwidth. }
 assert(Hco:owidth F(OReg(sl_counter d))(Z.of_nat(sl_width d))).
 {unfold owidth. intros r Hr. injection Hr as <-. unfold wd. now rewrite Hc. }
 assert(Halias:regs s(sl_source d)=regs s(sl_counter d)).
 {eapply signed_loop_source_sound;eauto;[lia|tauto]. }
 destruct(cmp_range F st(OReg(sl_counter d))(Z.of_nat(sl_width d))true)as[[cl ch]|]eqn:Hcr;[|intuition discriminate].
 destruct(cmp_range F st(sl_operand d)(Z.of_nat(sl_width d))true)as[[sl sh]|]eqn:Hsr;[|intuition discriminate].
 pose proof(cmp_range_sound F E HE t(regs s)Hrs st _ _ _ _ _ Hst Hw Hco Hcr)as Hcb.
 pose proof(cmp_range_sound F E HE t(regs s)Hrs st _ _ _ _ _ Hst Hw Ho Hsr)as Hsb.
 unfold cval,extend in Hcb,Hsb. rewrite Nat2Z.id in Hcb,Hsb. cbn[opv]in Hcb.
 rewrite !andb_true_iff in Hcheck. rewrite !Z.leb_le,!Z.ltb_lt in Hcheck.
 set(rs1:=upd(wdn F)(regs s)(sl_counter d)(regs s(sl_counter d)+opv(regs s)(sl_operand d))).
 set(s1:=goto(next_pos prog(fst pos)(snd pos))rs1).
 assert(Hstep:phase_step F prog E t s=Some s1).
 {unfold phase_step,step. rewrite Hpos in Hf.
  change(fetch prog(blk s)(idx s)=Some(signed_loop_instr d))in Hf. rewrite Hf.
  unfold signed_loop_instr. cbn[guard_of holds negb dest result binop_eval].
  rewrite Halias. unfold s1,rs1. rewrite Hpos. reflexivity. }
 assert(Hvalue:sext(sl_width d)(rs1(sl_counter d))=
   sext(sl_width d)(regs s(sl_counter d))+sext(sl_width d)(opv(regs s)(sl_operand d))).
 {unfold rs1,upd. rewrite Pos.eqb_refl,Hc.
  eapply (eqm_sext_eq (Z.of_nat(sl_width d))).
  - lia.
  - lia.
  - apply trunc_range.
  - split;lia.
  - eapply eqm_trans;[apply trunc_eqm|].
    apply eqm_add;[lia|apply eqm_sym,sext_eqm|apply eqm_sym,sext_eqm]. }
 rewrite Hpos in Hf.
 destruct(transfer_sound F prog E HE t s s1 st(signed_loop_instr d)Hf Hstep Hst Hrs Hwf)as[Hout Hregs].
 assert(Hbase:state_ok F(S t)(signed_loop_step_base F st d)(regs s1)).
 {unfold signed_loop_step_base. rewrite Hcr,Hsr.
  apply(put_ok F E HE(S t)(regs s1)Hregs);[exact Hout|].
  unfold fact_ok,fact_okv. split.
  - unfold wf,range_form;cbn[fm];lia.
  - split.
    + apply(sat_range(wdn F)(regs s1)(Z.of_nat(sl_width d))(Z.of_nat(sl_width d))
       (cl+sl)(ch+sh)(regs s1(sl_counter d))(sext(sl_width d)(regs s1(sl_counter d)))).
      * lia.
      * change(regs s1)with rs1. rewrite Hvalue. lia.
      * apply eqm_sym,sext_eqm.
    + split;[eapply reg_align_sound;eassumption|discriminate]. }
 assert(Hpost:state_ok F(S t)(signed_loop_step_post conjunction F st d)(regs s1)).
 {unfold signed_loop_step_post. destruct conjunction;[|exact Hbase].
  unfold signed_loop_step_relation. destruct(Pos.eqb(sl_source d)(sl_counter d))eqn:Hsource;[exact Hbase|].
  assert(Hsource_value:rs1(sl_source d)=regs s(sl_source d))by(unfold rs1,upd;rewrite Hsource;reflexivity).
  destruct(sl_operand d)as[r|z]eqn:Hop.
  - destruct(Pos.eqb r(sl_counter d))eqn:Hoperand;[exact Hbase|].
    assert(Hoperand_value:rs1 r=regs s r)by(unfold rs1,upd;rewrite Hoperand;reflexivity).
    assert(Hoperand_width:wdn F r=sl_width d).
    {pose proof(Ho r eq_refl)as Hwidth. unfold wd in Hwidth. lia. }
    apply(add_cons_ok F E HE(S t)(regs s1)Hregs);[exact Hbase|].
    unfold cons_ok. rewrite(tsum_sorted_terms F E HE(S t)(regs s1)Hregs).
    cbn[tsum atom_val fst snd]. change(regs s1)with rs1.
    rewrite Hc,Ha,Hoperand_width,Hsource_value,Hoperand_value,Hvalue. cbn[opv]. rewrite Halias. lia.
  - apply(add_cons_ok F E HE(S t)(regs s1)Hregs);[exact Hbase|].
    unfold cons_ok. rewrite(tsum_sorted_terms F E HE(S t)(regs s1)Hregs).
    cbn[tsum atom_val fst snd]. change(regs s1)with rs1.
    rewrite Hc,Ha,Hsource_value,Hvalue. cbn[opv]. rewrite Halias. lia. }
 exists s1. split;[exact Hstep|]. split.
 - unfold s1,pc_of;cbn[goto blk idx]. symmetry;apply surjective_pairing.
 - split;[exact Hpost|]. split;[exact Hregs|].
   unfold signed_loop_value. change(regs s1)with rs1. rewrite Hvalue.
   destruct(sl_down d)eqn:Hdown;rewrite !Z.ltb_lt in Hcheck;lia.
Qed.


Record signed_loop_certificate := {
 slc_step:signed_loop_step; slc_conjunction:bool; slc_predicate:reg; slc_negated:bool;
 slc_increment:pc; slc_branch:pc; slc_before:astate; slc_after:astate
}.
(* The existing branch checker depends only on the predicate/polarity.
   Arithmetic and protected registers are checked separately below. *)
Definition signed_loop_control cert : loop_info :=
 {|li_prefix:=0;li_suffix:=0;li_counter:=sl_counter(slc_step cert);
   li_source:=sl_source(slc_step cert);li_stride:=sl_counter(slc_step cert);
   li_predicate:=slc_predicate cert;li_negated:=slc_negated cert|}.
(* An exact signed singleton comparison is refined in its original signed
   interpretation. Canonical unsigned equality can discard a useful modular
   interval at zero; the raw refinement theorem covers either interpretation. *)
Definition signed_loop_refine F st p truth : option astate :=
 match get st p with
 | Some(FPred [k] true)=>if csg k then
     refine_cmp_raw F st (if truth then k else negate k)
   else refine F st p truth
 | _=>refine F st p truth
 end.
Lemma signed_loop_refine_sound : forall F E t rs st p truth,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 negb(Z.eqb(rs p)0)=truth ->
 exists out,signed_loop_refine F st p truth=Some out /\ state_ok F t out rs.
Proof.
 intros F E t rs st p truth HE Hr Hst Htruth.
 assert(Hordinary:exists out,refine F st p truth=Some out /\ state_ok F t out rs)
 by(eapply refine_sound;eassumption).
 unfold signed_loop_refine.
 destruct(get st p)as[fact|]eqn:Hget;[|exact Hordinary].
 destruct fact as[f al vr|al|cs exact|cs exact];try exact Hordinary.
 destruct cs as[|k tail];[exact Hordinary|].
 destruct tail;[|exact Hordinary]. destruct exact;[|exact Hordinary].
 destruct(csg k);[|exact Hordinary].
 pose proof(proj1 Hst p _ Hget)as Hfact.
 unfold fact_ok,fact_okv in Hfact.
 apply(refine_cmp_raw_sound F E HE t rs Hr st);[exact Hst|].
 destruct truth.
 - apply negb_true_iff,Z.eqb_neq in Htruth.
   pose proof(proj1 Hfact Htruth)as Hk. inversion Hk;assumption.
 - apply(proj2(negate_holds rs k)). intro Hk.
   assert(Hks:Forall(cmp_holds rs)[k])by(constructor;[exact Hk|constructor]).
   pose proof(proj2 Hfact eq_refl Hks)as Hnz.
   apply negb_false_iff,Z.eqb_eq in Htruth. contradiction.
Qed.
(* A true predicate fact implies every stored comparison. Refining a
   false conjunction cannot negate all clauses, so that path retains the
   existing refinement. This mode is selected before validation. *)
Fixpoint signed_refine_all F st cs:option astate:=
 match cs with []=>Some st|k::tail=>match refine_cmp_raw F st k with
 |Some next=>signed_refine_all F next tail|None=>None end end.
Lemma signed_refine_all_sound : forall F E t rs st cs,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs -> Forall(cmp_holds rs)cs ->
 exists out,signed_refine_all F st cs=Some out /\ state_ok F t out rs.
Proof.
 intros F E t rs st cs HE Hr;revert st. induction cs as[|k cs IH];intros st Hst Hall;cbn[signed_refine_all].
 - exists st;auto.
 - inversion Hall as[|k' cs' Hk Hcs];subst.
   destruct(refine_cmp_raw_sound F E HE t rs Hr st k Hst Hk)as[next[Hnext Hgood]].
   rewrite Hnext. apply IH;assumption.
Qed.
Definition signed_loop_refine_clauses (conjunction:bool) F st p truth:=
 if conjunction && truth then
 match get st p with
 |Some(FPred cs _)=>signed_refine_all F st cs
 |_=>signed_loop_refine F st p truth end
 else signed_loop_refine F st p truth.
Lemma signed_loop_refine_clauses_sound : forall conjunction F E t rs st p truth,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 negb(Z.eqb(rs p)0)=truth ->
 exists out,signed_loop_refine_clauses conjunction F st p truth=Some out /\ state_ok F t out rs.
Proof.
 intros conjunction F E t rs st p truth HE Hr Hst Htruth.
 assert(Hordinary:exists out,signed_loop_refine F st p truth=Some out /\ state_ok F t out rs)
 by(eapply signed_loop_refine_sound;eassumption).
 unfold signed_loop_refine_clauses. destruct(conjunction && truth)eqn:Hmode;[|exact Hordinary].
 apply andb_true_iff in Hmode. destruct Hmode as[_ ->].
 destruct(get st p)as[f|]eqn:Hget;[|exact Hordinary].
 destruct f as[f al vr|al|cs exact|cs exact];try exact Hordinary.
 eapply signed_refine_all_sound;[exact HE|exact Hr|exact Hst|].
 pose proof(proj1 Hst p _ Hget)as Hfact. unfold fact_ok,fact_okv in Hfact.
 apply(proj1 Hfact). apply negb_true_iff,Z.eqb_neq in Htruth. exact Htruth.
Qed.
(* Materialize the checked signed counter range after clause refinement.
   This retains bounds obtained through a surviving signed-add relation. *)
Definition signed_loop_refresh F st r:option astate:=
 match check_range F st(form_of_terms[((r,ES),1)])with
 |Some(lo,hi)=>tighten_reg F st r ES lo hi
 |None=>Some st end.
Lemma signed_loop_refresh_sound : forall F E t rs st r,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 exists out,signed_loop_refresh F st r=Some out /\ state_ok F t out rs.
Proof.
 intros F E t rs st r HE Hr Hst. unfold signed_loop_refresh.
 destruct(check_range F st(form_of_terms[((r,ES),1)]))as[[lo hi]|]eqn:Hrange;[|exists st;auto].
 pose proof(check_range_sound F E HE t rs Hr st(form_of_terms[((r,ES),1)])lo hi 0 Hst Hrange ltac:(cbn;lia))as Hvalue.
 cbn[ft form_of_terms tsum fold_right atom_val fst snd]in Hvalue.
 apply(tighten_reg_sound F E HE t rs Hr st r ES lo hi Hst). cbn[atom_val]. lia.
Qed.
Definition signed_loop_refine_mode (conjunction:bool) F st r p truth:=
 match signed_loop_refine_clauses conjunction F st p truth with
 |None=>None
 |Some out=>if conjunction then signed_loop_refresh F out r else Some out end.
Lemma signed_loop_refine_mode_sound : forall conjunction F E t rs st r p truth,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 negb(Z.eqb(rs p)0)=truth ->
 exists out,signed_loop_refine_mode conjunction F st r p truth=Some out /\ state_ok F t out rs.
Proof.
 intros conjunction F E t rs st r p truth HE Hr Hst Htruth.
 destruct(signed_loop_refine_clauses_sound conjunction F E t rs st p truth HE Hr Hst Htruth)as[out[Href Hout]].
 unfold signed_loop_refine_mode. rewrite Href. destruct conjunction;[|exists out;auto].
 eapply signed_loop_refresh_sound;eassumption.
Qed.
Definition signed_loop_branches (conjunction:bool)(F:lfacts)(st:astate)(d:loop_info)(head:pc):=
 let after:=transfer F st(loop_branch d head)in
 (signed_loop_refine_mode conjunction F after(li_counter d)(li_predicate d)(negb(li_negated d)),
  signed_loop_refine_mode conjunction F after(li_counter d)(li_predicate d)(li_negated d)).
Lemma signed_loop_branch_sound : forall conjunction F prog E head pos st d t s,
 env_ok F E -> loop_branch_ok F prog head pos d=true -> pos=pc_of s ->
 state_ok F t st (regs s) -> regs_ok F E (regs s) ->
 exists (s' : state) (out : astate) (taken : bool), phase_step F prog E t s=Some s' /\
 (if taken then fst (signed_loop_branches conjunction F st d head) else snd (signed_loop_branches conjunction F st d head))=Some out /\
 pc_of s'=(if taken then head else next_pos prog (fst pos) (snd pos)) /\
 state_ok F (S t) out (regs s') /\ regs_ok F E (regs s') /\ regs s'=regs s.
Proof.
 intros conjunction F prog E head pos st d t s HE Hcheck Hpos Hst Hrs.
 unfold loop_branch_ok in Hcheck. rewrite !andb_true_iff in Hcheck.
 assert (Hhead : snd head=0%nat) by (apply Nat.eqb_eq;tauto).
 assert (Hf : fetch prog (fst pos) (snd pos)=Some (loop_branch d head)) by (apply loop_fetch_is_sound;tauto).
 assert (Hwf : wf_instr F (loop_branch d head)=true) by tauto.
 set (truth:=negb (Z.eqb (regs s (li_predicate d)) 0)).
 set (taken:=if li_negated d then negb truth else truth).
 set (s1:=goto (if taken then head else next_pos prog (fst pos) (snd pos)) (regs s)).
 assert (Hstep : phase_step F prog E t s=Some s1).
 { unfold phase_step,step. rewrite Hpos in Hf. change (fetch prog (blk s) (idx s)=Some (loop_branch d head)) in Hf.
   rewrite Hf. unfold loop_branch. cbn [guard_of holds].
   unfold taken,truth,s1. destruct (li_negated d), (Z.eqb (regs s (li_predicate d)) 0);cbn;
   rewrite ?Hhead,?Hpos;try reflexivity;rewrite <-Hhead,(surjective_pairing head);reflexivity. }
 rewrite Hpos in Hf.
 destruct (transfer_sound F prog E HE t s s1 st (loop_branch d head) Hf Hstep Hst Hrs Hwf) as [Hout Hregs].
 destruct (signed_loop_refine_mode_sound conjunction F E (S t) (regs s1) _ (li_counter d) (li_predicate d) truth HE Hregs Hout eq_refl)
  as [out [Href Hgood]].
 exists s1,out,taken. split;[exact Hstep|]. split.
 - unfold signed_loop_branches. unfold taken. destruct (li_negated d),truth;exact Href.
 - split;[unfold s1,pc_of;destruct taken;cbn [goto blk idx];symmetry;apply surjective_pairing|]. split;[exact Hgood|]. split;[exact Hregs|reflexivity].
Qed.

Definition signed_loop_iteration fuel F prog events head cert inv
 : option(pc*option astate*option astate) :=
 let d:=slc_step cert in let inc:=slc_increment cert in let jump:=slc_branch cert in
 let before:=slc_before cert in let after:=slc_after cert in
 if loop_dag_check fuel F prog[sl_counter d]events head inc inv before then
 if occurrence_member events inc then None else
 if signed_loop_step_ok F prog inc before d then
 if loop_dag_check fuel F prog[sl_counter d]events
   (next_pos prog(fst inc)(snd inc))jump(signed_loop_step_post(slc_conjunction cert)F before d)after then
 if occurrence_member events jump then None else
 if loop_branch_ok F prog head jump(signed_loop_control cert)then
   let '(back,leave):=signed_loop_branches (slc_conjunction cert) F after(signed_loop_control cert)head in
   Some(next_pos prog(fst jump)(snd jump),back,leave)
 else None else None else None else None.
Theorem signed_loop_iteration_sound : forall fuel F prog E events head cert inv exit back leave t s,
 env_ok F E -> signed_loop_iteration fuel F prog events head cert inv=Some(exit,back,leave) ->
 head=pc_of s -> state_ok F t inv(regs s) -> regs_ok F E(regs s) ->
 exists(n:nat)(last:state)(out:astate)(taken:bool),
 (0<n)%nat /\ loop_event_free_run F prog E events n t s last /\
 (if taken then back else leave)=Some out /\ pc_of last=(if taken then head else exit) /\
 state_ok F(n+t)%nat out(regs last) /\ regs_ok F E(regs last) /\
 signed_loop_value(slc_step cert)(regs s)<signed_loop_value(slc_step cert)(regs last).
Proof.
 intros fuel F prog E events head cert inv exit back leave t s HE Hiter Hpos Hst Hrs.
 unfold signed_loop_iteration in Hiter.
 set(d:=slc_step cert)in *. set(inc:=slc_increment cert)in *. set(jump:=slc_branch cert)in *.
 set(before:=slc_before cert)in *. set(after:=slc_after cert)in *.
 destruct(loop_dag_check fuel F prog[sl_counter d]events head inc inv before)eqn:Hprefix;[|discriminate].
 destruct(occurrence_member events inc)eqn:Hincfree;[discriminate|].
 destruct(signed_loop_step_ok F prog inc before d)eqn:Hinc;[|discriminate].
 destruct(loop_dag_check fuel F prog[sl_counter d]events
  (next_pos prog(fst inc)(snd inc))jump(signed_loop_step_post(slc_conjunction cert)F before d)after)eqn:Hsuffix;[|discriminate].
 destruct(occurrence_member events jump)eqn:Hjumpfree;[discriminate|].
 destruct(loop_branch_ok F prog head jump(signed_loop_control cert))eqn:Hbranch;[|discriminate].
 destruct(signed_loop_branches (slc_conjunction cert) F after(signed_loop_control cert)head)as[b l]eqn:Hbranches. injection Hiter as <- <- <-.
 destruct(loop_dag_check_sound fuel F prog E[sl_counter d]events head inc inv before t s
  HE Hprefix Hpos Hst Hrs)as[n1[s1[Hrun1[Hpc1[Hst1[Hrs1 Hpres1]]]]]].
 destruct(signed_loop_step_sound(slc_conjunction cert)F prog E inc before d(n1+t)%nat s1
  HE Hinc(eq_sym Hpc1)Hst1 Hrs1)as[s2[Hstep2[Hpc2[Hst2[Hrs2 Hincrease]]]]].
 destruct(loop_dag_check_sound fuel F prog E[sl_counter d]events
  (next_pos prog(fst inc)(snd inc))jump(signed_loop_step_post(slc_conjunction cert)F before d)after
  (S(n1+t))s2 HE Hsuffix(eq_sym Hpc2)Hst2 Hrs2)as[n3[s3[Hrun3[Hpc3[Hst3[Hrs3 Hpres3]]]]]].
 destruct(signed_loop_branch_sound (slc_conjunction cert) F prog E head jump after(signed_loop_control cert)(n3+S(n1+t))%nat s3
  HE Hbranch(eq_sym Hpc3)Hst3 Hrs3)as[s4[out[taken[Hstep4[Hout[Hpc4[Hst4[Hrs4 Hregs4]]]]]]]].
 exists(n1+1+n3+1)%nat,s4,out,taken. split;[lia|]. split.
 - apply loop_event_free_app with(mid:=s3).
   + apply loop_event_free_app with(mid:=s2).
     * apply loop_event_free_app with(mid:=s1);[exact Hrun1|].
       econstructor;[rewrite Hpc1;exact Hincfree|exact Hstep2|constructor].
     * replace(n1+1+t)%nat with(S(n1+t))by lia. exact Hrun3.
   + replace(n1+1+n3+t)%nat with(n3+S(n1+t))%nat by lia.
     econstructor;[rewrite Hpc3;exact Hjumpfree|exact Hstep4|constructor].
 - rewrite Hbranches in Hout. cbn[fst snd]in Hout. split;[exact Hout|].
   split;[exact Hpc4|]. split.
   + replace(n1+1+n3+1+t)%nat with(S(n3+S(n1+t)))by lia. exact Hst4.
   + split;[exact Hrs4|]. unfold signed_loop_value in *.
     rewrite Hregs4,(Hpres3 _ (or_introl eq_refl)).
     rewrite(Hpres1 _ (or_introl eq_refl))in Hincrease. exact Hincrease.
Qed.
Definition signed_loop_validate fuel F prog events head cert inv
 : option(pc*option astate) :=
 let d:=slc_step cert in let w:=sl_width d in
 if (1 <=? Z.of_nat w)&&(Z.of_nat w <=? 64)&&Nat.eqb(wdn F(sl_counter d))w then
 match cmp_range F inv(OReg(sl_counter d))(Z.of_nat w)true,
       signed_loop_iteration fuel F prog events head cert inv with
 | Some _,Some(exit,back,leave)=>
   if match back with None=>true|Some st=>state_le F st inv end then Some(exit,leave)else None
 | _,_=>None end else None.
Definition signed_loop_rank upper d rs :=Z.to_nat(Z.max 0(upper-signed_loop_value d rs)).
Lemma signed_loop_rank_decreases : forall upper d rs rs',
 signed_loop_value d rs<signed_loop_value d rs' -> signed_loop_value d rs'<=upper ->
 (signed_loop_rank upper d rs'<signed_loop_rank upper d rs)%nat.
Proof. intros. unfold signed_loop_rank. rewrite !Z.max_r by lia. apply Z2Nat.inj_lt;lia. Qed.
Theorem signed_loop_validate_progress : forall fuel F prog E events head cert inv exit ost,
 env_ok F E -> signed_loop_validate fuel F prog events head cert inv=Some(exit,ost) ->
 forall t s,head=pc_of s -> state_ok F t inv(regs s) -> regs_ok F E(regs s) ->
 exists n last out,loop_event_free_run F prog E events n t s last /\
 pc_of last=exit /\ ost=Some out /\ state_ok F(n+t)%nat out(regs last) /\ regs_ok F E(regs last).
Proof.
 intros fuel F prog E events head cert inv exit ost HE Hval.
 unfold signed_loop_validate in Hval. cbv zeta in Hval.
 destruct((1 <=? Z.of_nat(sl_width(slc_step cert))) && (Z.of_nat(sl_width(slc_step cert)) <=? 64) &&
  Nat.eqb(wdn F(sl_counter(slc_step cert)))(sl_width(slc_step cert)))eqn:Hwidth;[|discriminate].
 rewrite !andb_true_iff,!Z.leb_le,Nat.eqb_eq in Hwidth.
 destruct(cmp_range F inv(OReg(sl_counter(slc_step cert)))(Z.of_nat(sl_width(slc_step cert)))true)as[[lo hi]|]eqn:Hbound;[|discriminate].
 destruct(signed_loop_iteration fuel F prog events head cert inv)as[[[next back]leave]|]eqn:Hiter;[|discriminate].
 destruct(match back with None=>true|Some st=>state_le F st inv end)eqn:Hback;[|discriminate].
 injection Hval as <- <-.
 set(upper:=if sl_down(slc_step cert)then -lo else hi).
 assert(Hupper:forall t rs,state_ok F t inv rs -> regs_ok F E rs ->signed_loop_value(slc_step cert)rs<=upper).
 {intros t rs Hst Hrs.
  assert(Ho:owidth F(OReg(sl_counter(slc_step cert)))(Z.of_nat(sl_width(slc_step cert)))).
  {unfold owidth. intros r Hr. injection Hr as <-. unfold wd. now rewrite(proj2 Hwidth). }
  pose proof(cmp_range_sound F E HE t rs Hrs inv _ _ _ _ _ Hst(proj1 Hwidth)Ho Hbound)as Hb.
  unfold cval,extend in Hb. rewrite Nat2Z.id in Hb. cbn[opv]in Hb.
  unfold upper,signed_loop_value. destruct(sl_down(slc_step cert));lia. }
 assert(Hind:forall rank t s,signed_loop_rank upper(slc_step cert)(regs s)=rank ->
 head=pc_of s ->state_ok F t inv(regs s)->regs_ok F E(regs s)->
 exists n last out,loop_event_free_run F prog E events n t s last /\
 pc_of last=next /\ leave=Some out /\state_ok F(n+t)%nat out(regs last)/\regs_ok F E(regs last)).
 {intro rank. induction rank using lt_wf_ind. intros t s Hrank Hpos Hst Hrs.
  destruct(signed_loop_iteration_sound fuel F prog E events head cert inv next back leave t s
   HE Hiter Hpos Hst Hrs)as[n1[s1[out[taken[Hpositive[Hrun[Hout[Hpc[Hst1[Hrs1 Hinc]]]]]]]]]].
  destruct taken.
  - destruct back as[back|];[|discriminate]. injection Hout as <-.
    assert(Hinv:state_ok F(n1+t)%nat inv(regs s1))by(eapply state_le_sound;eassumption).
    assert(Hdec:(signed_loop_rank upper(slc_step cert)(regs s1)<rank)%nat).
    {rewrite <-Hrank. apply signed_loop_rank_decreases;[exact Hinc|eapply Hupper;eassumption]. }
    destruct(H _ Hdec(n1+t)%nat s1 eq_refl(eq_sym Hpc)Hinv Hrs1)
      as[n[last[output[Hlast[Hlastpc[Hleave[Hfinal Hregsfinal]]]]]]].
    exists(n1+n)%nat,last,output. split;[eapply loop_event_free_app;eassumption|].
    split;[exact Hlastpc|]. split;[exact Hleave|]. split;[|exact Hregsfinal].
    replace(n1+n+t)%nat with(n+(n1+t))%nat by lia. exact Hfinal.
  - exists n1,s1,out. split;[exact Hrun|]. split;[exact Hpc|]. split;[exact Hout|]. split;assumption.
 }
 intros t s Hpos Hst Hrs. exact(Hind(signed_loop_rank upper(slc_step cert)(regs s))t s eq_refl Hpos Hst Hrs).
Qed.

Record signed_loop_shape := {
 sls_step:signed_loop_step; sls_predicate:reg; sls_negated:bool;
 sls_increment:pc; sls_branch:pc; sls_limit:opnd; sls_comparison:cmp; sls_signed:bool
}.
Fixpoint signed_loop_increment_at(counter:reg)(width block index:nat)(body:list instr)
 :option(pc*reg*opnd):=
 match body with
 | []=>None
 | i::tail=>match i with
   | IBin None BAdd w dst src stride=>
     if Nat.eqb w width && Pos.eqb dst counter then Some((block,index),src,stride)
     else signed_loop_increment_at counter width block(S index)tail
   | _=>signed_loop_increment_at counter width block(S index)tail end
 end.
Definition signed_loop_shape_block(head:pc)(block:nat)(body:list instr):option signed_loop_shape:=
 match rev body with
 | IBra(Some(neg,p))target::ISetp None comparison width sg q counter limit::_=>
   if Nat.eqb target(fst head) && Nat.eqb(snd head)0 && Pos.eqb p q then
   match signed_loop_increment_at counter width block 0 body with
   | Some(inc,src,stride)=>
     let down:=match stride with OImm z=>sext width z <? 0 | OReg _=>false end in
     Some {|sls_step:={|sl_width:=width;sl_alias_width:=64;sl_counter:=counter;sl_source:=src;sl_operand:=stride;sl_down:=down|};
       sls_predicate:=p;sls_negated:=neg;sls_increment:=inc;
       sls_branch:=(block,Nat.pred(List.length body));sls_limit:=limit;
       sls_comparison:=comparison;sls_signed:=sg|}
   | None=>None end else None
 | _=>None end.
Fixpoint signed_loop_shape_scan(head:pc)(block:nat)(bodies:program):option signed_loop_shape:=
 match bodies with
 | []=>None
 | body::tail=>match signed_loop_shape_block head block body with
   | Some d=>Some d|None=>signed_loop_shape_scan head(S block)tail end
 end.
Definition signed_loop_at prog head:=
 if Nat.eqb(snd head)0 then signed_loop_shape_scan head 0 prog else None.
(* Select only event-free source spans before invoking the certificate.
   Event-bearing loops remain with the complete occurrence-word traversal. *)
Definition signed_loop_at_event_free prog events head:=
 match signed_loop_at prog head with
 | Some shape=>if short_all(fun event=>phase_after event head ||
     phase_after(sls_branch shape)event)events then Some shape else None
 | None=>None end.
Definition signed_loop_limit cmp neg down lo hi :=
 match cmp,neg,down with
 | CLt,false,false|CGe,true,false|CNe,false,false|CEq,true,false=>hi-1
 | CLe,false,false|CGt,true,false=>hi
 | CGt,false,true|CLe,true,true|CNe,false,true|CEq,true,true=>lo+1
 | CGe,false,true|CLt,true,true=>lo
 | _,_,_=>if down then lo else hi
 end.
Definition signed_loop_propose_offset (conjunction:bool) F prog head shape st (offset:Z):option(astate*signed_loop_certificate):=
 let d:=sls_step shape in let w:=Z.of_nat(sl_width d)in
 match cmp_range F st(OReg(sl_counter d))w true,
       cmp_range F st(sls_limit shape)w(sls_signed shape)with
 | Some(cl,ch),Some(ll,lh)=>
   let boundary:=signed_loop_limit(sls_comparison shape)(sls_negated shape)(sl_down d)ll lh + offset in
   let lo:=if sl_down d then Z.min cl boundary else cl in
   let hi:=if sl_down d then ch else Z.max ch boundary in
   let body:=List.concat(firstn(S(fst(sls_branch shape))-fst head)(skipn(fst head)prog))in
   let forgotten:=fold_left(fun s r=>assign F s r None None)(flat_map loop_writes body)st in
   let alignment:=Z.min(reg_align F st(sl_counter d))(opnd_align F st(sl_operand d))in
   let inv:={|amap:=PM.add(sl_counter d)(FForm(range_form w lo hi)alignment None)(amap forgotten);
              acons:=acons forgotten|}in
   let inc:=sls_increment shape in
   match loop_dag_post F prog inc head inv with
   | None=>None
   | Some before=>match loop_dag_post F prog(sls_branch shape)(next_pos prog(fst inc)(snd inc))
       (signed_loop_step_post conjunction F before d)with
     | None=>None
     | Some after=>Some(inv,{|slc_step:=d;slc_conjunction:=conjunction;slc_predicate:=sls_predicate shape;
         slc_negated:=sls_negated shape;slc_increment:=inc;slc_branch:=sls_branch shape;
         slc_before:=before;slc_after:=after|})end
   end
 | _,_=>None end.
Definition signed_loop_propose F prog head shape st:=
 signed_loop_propose_offset false F prog head shape st 0.
Definition signed_summary_check F prog events pos st
 (proposal:option(astate*signed_loop_certificate)):occurrence_summary_result:=
 match proposal with
 | None=>Rejected_loop_summary
 | Some(inv,cert)=>if state_le F st inv then
   match signed_loop_validate phase_fuel F prog events pos cert inv with
   | Some(endpoint,Some post)=>Checked_loop_summary endpoint post
   | _=>Rejected_loop_summary end else Rejected_loop_summary end.
Lemma signed_summary_check_sound:forall F prog E events pos st proposal endpoint post,
 env_ok F E -> signed_summary_check F prog events pos st proposal=Checked_loop_summary endpoint post ->
 forall t s,pos=pc_of s -> state_ok F t st(regs s) -> regs_ok F E(regs s) ->
 exists n last,loop_event_free_run F prog E events n t s last /\ pc_of last=endpoint /\
 state_ok F(n+t)%nat post(regs last) /\ regs_ok F E(regs last).
Proof.
 intros F prog E events pos st proposal endpoint post HE H t s Hpos Hst Hrs.
 unfold signed_summary_check in H.
 destruct proposal as[[inv cert]|];[|discriminate].
 destruct(state_le F st inv)eqn:Hle;[|discriminate].
 destruct(signed_loop_validate phase_fuel F prog events pos cert inv)as[[exit ost]|]eqn:Hval;[|discriminate].
 destruct ost as[out|];[|discriminate]. injection H as <- <-.
 assert(Hinv:state_ok F t inv(regs s))by(eapply state_le_sound;eassumption).
 destruct(signed_loop_validate_progress phase_fuel F prog E events pos cert inv exit(Some out)
  HE Hval t s Hpos Hinv Hrs)as[n[last[actual[Hrun[Hpc[Heq[Hout Hr]]]]]]].
 injection Heq as <-. exists n,last. split;[exact Hrun|]. split;[exact Hpc|]. split;assumption.
Qed.
Definition signed_occurrence_summary F prog events pos st:occurrence_summary_result:=
 match signed_loop_at_event_free prog events pos with
 | None=>No_loop_summary
 | Some shape=>signed_summary_check F prog events pos st(signed_loop_propose F prog pos shape st)
 end.
Theorem signed_occurrence_summary_sound:forall F prog E events,
 env_ok F E->occurrence_summary_ok F prog E events(signed_occurrence_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H.
 unfold signed_occurrence_summary in H.
 destruct(signed_loop_at_event_free prog events pos)as[shape|];[|discriminate].
 eapply signed_summary_check_sound;eassumption.
Qed.

Definition occurrence_loop_summary F prog events pos st : occurrence_summary_result :=
 match loop_dag_at prog pos with
 | None=>No_loop_summary
 | Some shape=>match loop_dag_propose F prog pos shape st with
   | None=>Rejected_loop_summary
   | Some(inv,cert)=>if state_le F st inv then
     match loop_dag_validate phase_fuel F prog events pos cert inv with
     | Some(endpoint,Some post)=>Checked_loop_summary endpoint post
     | _=>Rejected_loop_summary end
     else Rejected_loop_summary
   end
 end.
Theorem occurrence_loop_summary_sound : forall F prog E events,
 env_ok F E -> occurrence_summary_ok F prog E events(occurrence_loop_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H t s Hpos Hst Hrs.
 unfold occurrence_loop_summary in H.
 destruct(loop_dag_at prog pos)as[shape|];[|discriminate].
 destruct(loop_dag_propose F prog pos shape st)as[[inv cert]|];[|discriminate].
 destruct(state_le F st inv)eqn:Hle;[|discriminate].
 destruct(loop_dag_validate phase_fuel F prog events pos cert inv)as[[exit ost]|]eqn:Hval;[|discriminate].
 destruct ost as[out|];[|discriminate]. injection H as <- <-.
 assert(Hinv:state_ok F t inv(regs s))by(eapply state_le_sound;eassumption).
 destruct(loop_dag_validate_progress phase_fuel F prog E events pos cert inv exit(Some out)
   HE Hval t s Hpos Hinv Hrs)as[n[last[actual[Hrun[Hpc[Heq[Hout Hr]]]]]]].
 injection Heq as <-. exists n,last. split;[exact Hrun|]. split;[exact Hpc|]. split;assumption.
Qed.
Definition occurrence_signed_loop_summary F prog events pos st : occurrence_summary_result :=
 match loop_dag_at prog pos with
 | Some _=>occurrence_loop_summary F prog events pos st
 | None=>signed_occurrence_summary F prog events pos st end.
Theorem occurrence_signed_loop_summary_sound : forall F prog E events,
 env_ok F E -> occurrence_summary_ok F prog E events(occurrence_signed_loop_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H t s Hpos Hst Hrs.
 unfold occurrence_signed_loop_summary in H.
 destruct(loop_dag_at prog pos).
 - eapply(occurrence_loop_summary_sound F prog E events HE pos st endpoint post H t s);eassumption.
 - eapply(signed_occurrence_summary_sound F prog E events HE pos st endpoint post H t s);eassumption.
Qed.

Definition occurrence_program_word_general F prog events :=
 if occurrence_segments_acyclic prog events(occurrence_cut prog events)
 then occurrence_checked_word no_loop_summary F prog events
 else occurrence_checked_word(occurrence_loop_summary F prog events)F prog events.
Theorem occurrence_program_word_general_sound : forall F prog events word,
 occurrence_program_word_general F prog events=Some word -> occurrence_schedule_safe F prog events.
Proof.
 intros F prog events word H. unfold occurrence_program_word_general in H.
 destruct(occurrence_segments_acyclic prog events(occurrence_cut prog events)).
 - eapply occurrence_checked_word_sound;[intros;apply no_loop_summary_ok|exact H].
 - eapply occurrence_checked_word_sound;[intros;apply occurrence_loop_summary_sound;assumption|exact H].
Qed.
Theorem occurrence_program_word_preserves : forall F prog events word,
 occurrence_program_word F prog events=Some word -> occurrence_program_word_general F prog events=Some word.
Proof.
 intros F prog events word H. unfold occurrence_program_word in H.
 unfold occurrence_program_word_general.
 destruct(occurrence_segments_acyclic prog events(occurrence_cut prog events));[exact H|discriminate].
Qed.

(* Select a certificate family before running it. Acyclic occurrence
   segments retain their old path; cyclic segments may use a checked
   event-free loop summary. No failed certificate is retried. *)
Definition event_progress_general F prog events :=
 if occurrence_case prog events then
   match occurrence_program_word_general F prog events with Some _=>true|None=>false end
 else if straight_event_program prog then event_progress_nonocc F prog events else
 if loop_dag_selected prog then loop_summaries_check F prog events
 else event_progress_nonocc F prog events.
Theorem event_progress_general_sound : forall F prog events,
 event_progress_general F prog events=true -> event_progress_safe F prog events.
Proof.
 intros F prog events H. unfold event_progress_general in H.
 destruct(occurrence_case prog events)eqn:Hcase.
 - destruct(occurrence_program_word_general F prog events)as[word|]eqn:Hword;[|discriminate].
   unfold event_progress_safe. rewrite Hcase. eapply occurrence_program_word_general_sound;exact Hword.
 - assert(Hold:event_progress_nonocc F prog events=true -> event_progress_safe F prog events).
   { intros Hnon. apply event_progress_check_sound. unfold event_progress_check. rewrite Hcase. exact Hnon. }
   destruct(straight_event_program prog);[apply Hold;exact H|].
   destruct(loop_dag_selected prog);[|apply Hold;exact H].
   unfold event_progress_safe. rewrite Hcase. apply loop_summaries_check_sound;exact H.
Qed.
Theorem event_progress_general_legacy : forall F prog events,
 loop_dag_selected prog=false ->
 (occurrence_case prog events=false \/
  occurrence_segments_acyclic prog events(occurrence_cut prog events)=true) ->
 event_progress_general F prog events=event_progress_check F prog events.
Proof.
 intros F prog events Hdag Heligible. unfold event_progress_general,event_progress_check.
 destruct(occurrence_case prog events)eqn:Hcase.
 - assert(Hacyclic:occurrence_segments_acyclic prog events(occurrence_cut prog events)=true).
   { destruct Heligible;[congruence|assumption]. }
   unfold occurrence_program_word_general,occurrence_program_word. rewrite Hacyclic. reflexivity.
 - rewrite Hdag. destruct(straight_event_program prog);reflexivity.
Qed.
Theorem event_progress_general_preserves : forall F prog events,
 (if occurrence_case prog events then event_progress_check F prog events else
  if straight_event_program prog then event_progress_nonocc F prog events else
  if loop_dag_selected prog then loop_summaries_check F prog events else
  event_progress_nonocc F prog events)=true ->
 event_progress_general F prog events=true.
Proof.
 intros F prog events H. unfold event_progress_general.
 destruct(occurrence_case prog events)eqn:Hcase;[|exact H].
 unfold event_progress_check in H. rewrite Hcase in H.
 destruct(occurrence_program_word F prog events)as[word|]eqn:Hword;[|discriminate].
 rewrite(occurrence_program_word_preserves F prog events word Hword). reflexivity.
Qed.

Definition warp_facts (F:lfacts) (r:reg) (w:nat) : lfacts :=
 split_launch (split_launch F r (32*Z.of_nat w) false)
              r (32*(Z.of_nat w+1)) true.

Lemma warp_launch : forall F E r w,
 launch_ok F E -> 32*Z.of_nat w <= e_init E r < 32*(Z.of_nat w+1) ->
 launch_ok (warp_facts F r w) E.
Proof.
 intros F E r w H Hrange. unfold warp_facts.
 eapply (launch_split (split_launch F r (32*Z.of_nat w) false) E r (32*(Z.of_nat w+1)) true).
 - exact (launch_split F E r (32*Z.of_nat w) false H (proj1 Hrange)).
 - exact (proj2 Hrange).
Qed.

Lemma phase_run_split_same : forall F prog E r cut low n t s,
 phase_run (split_launch F r cut low) prog E n t s = phase_run F prog E n t s.
Proof.
 intros F prog E r cut low n. induction n;intros t s;[reflexivity|].
 cbn [phase_run]. change
 (match phase_step F prog E t s with
  | Some next=>phase_run (split_launch F r cut low) prog E n (S t) next
  | None=>None end =
 match phase_step F prog E t s with
 | Some next=>phase_run F prog E n (S t) next
 | None=>None end).
 destruct (phase_step F prog E t s);[apply IHn|reflexivity].
Qed.

Lemma occurrence_trace_split_same : forall F prog E events word t s r cut low,
 occurrence_trace (split_launch F r cut low) prog E events word t s ->
 occurrence_trace F prog E events word t s.
Proof.
 intros F prog E events word t s r cut low H.
 induction H.
 - constructor. intros n final Hrun. apply (H n final).
   rewrite phase_run_split_same. exact Hrun.
 - econstructor 2;[exact H|exact H0|exact H1|exact IHoccurrence_trace].
 - econstructor 3;[exact H|exact H0|exact IHoccurrence_trace].
Qed.

Lemma occurrence_trace_warp_same : forall F prog E events word t s r w,
 occurrence_trace (warp_facts F r w) prog E events word t s ->
 occurrence_trace F prog E events word t s.
Proof.
 intros. unfold warp_facts in H.
 apply occurrence_trace_split_same in H. eapply occurrence_trace_split_same;exact H.
Qed.

Definition event_projection (selected:list pc) (word:list pc) : list pc :=
 filter (occurrence_member selected) word.

Fixpoint event_word_eqb (a b:list pc) : bool :=
 match a,b with
 | [],[]=>true
 | x::xs,y::ys=>if phase_event x y then event_word_eqb xs ys else false
 | _,_=>false
 end.
Lemma event_word_eqb_eq : forall a b,event_word_eqb a b=true -> a=b.
Proof.
 induction a as[|x xs IH];intros [|y ys] H;try discriminate;[reflexivity|].
 cbn [event_word_eqb] in H. destruct(phase_event x y)eqn:Hxy;[|discriminate].
 apply phase_event_eq in Hxy. f_equal;[symmetry;exact Hxy|apply IH;exact H].
Qed.

Section Scoped.
Variable word_check : lfacts -> program -> list pc -> option(list pc).
Hypothesis word_check_sound : forall F prog events word,
 word_check F prog events=Some word -> forall E,launch_ok F E ->
 occurrence_trace F prog E events word 0 (phase_initial E).

Fixpoint warp_words_from (F:lfacts) (prog:program) (events:list pc) (r:reg)
 (first count:nat) : option(list(nat*list pc)) :=
 match count with
 | O=>Some []
 | S rest=>match word_check (warp_facts F r first) prog events with
   | None=>None
   | Some word=>match warp_words_from F prog events r (S first) rest with
     | Some words=>Some((first,word)::words)|None=>None end
   end
 end.

Lemma warp_words_from_indices : forall count F prog events r first words,
 warp_words_from F prog events r first count=Some words ->
 map fst words=seq first count.
Proof.
 induction count;intros F prog events r first words H;cbn [warp_words_from] in H.
 - injection H as <-. reflexivity.
 - destruct(word_check (warp_facts F r first) prog events) as[word|];[|discriminate].
   destruct(warp_words_from F prog events r (S first) count) as[tail|]eqn:Htail;[|discriminate].
   injection H as <-. cbn. f_equal. eapply IHcount;exact Htail.
Qed.

Lemma warp_words_from_trace : forall count F prog events r first words,
 warp_words_from F prog events r first count=Some words ->
 forall w word,In(w,word)words -> forall E,
 launch_ok F E -> 32*Z.of_nat w<=e_init E r<32*(Z.of_nat w+1) ->
 occurrence_trace F prog E events word 0 (phase_initial E).
Proof.
 induction count;intros F prog events r first words H w word Hin E Hlaunch Hrange;
 cbn [warp_words_from] in H.
 - injection H as <-. contradiction.
 - destruct(word_check (warp_facts F r first) prog events) as[found|]eqn:Hfound;[|discriminate].
   destruct(warp_words_from F prog events r (S first) count) as[tail|]eqn:Htail;[|discriminate].
   injection H as <-. destruct Hin as[Heq|Hin].
   + inversion Heq;subst. eapply occurrence_trace_warp_same.
     eapply word_check_sound;[exact Hfound|apply warp_launch;assumption].
   + eapply IHcount;eassumption.
Qed.

Definition scoped_words_check (F:lfacts) (prog:program) (events cta:list pc)
 (r:reg) (count:nat) : option(list(nat*list pc)) :=
 match warp_words_from F prog events r 0 count with
 | Some((w,word)::tail)=>
   if short_all(fun pair=>event_word_eqb (event_projection cta (snd pair))
                            (event_projection cta word))tail
   then Some((w,word)::tail) else None
 | _=>None
 end.

Definition scoped_schedule_safe (F:lfacts) (prog:program) (events cta:list pc)
 (r:reg) (count:nat) : Prop :=
 exists words common,
 map fst words=seq 0 count /\
 (forall w word,In(w,word)words -> event_projection cta word=common /\
   forall E,launch_ok F E -> 32*Z.of_nat w<=e_init E r<32*(Z.of_nat w+1) ->
   occurrence_trace F prog E events word 0 (phase_initial E)).

Theorem scoped_words_check_sound : forall F prog events cta r count words,
 scoped_words_check F prog events cta r count=Some words ->
 scoped_schedule_safe F prog events cta r count.
Proof.
 intros F prog events cta r count words H. unfold scoped_words_check in H.
 destruct(warp_words_from F prog events r 0 count) as[[|[w word]tail]|]eqn:Hwords;try discriminate.
 destruct(short_all(fun pair=>event_word_eqb(event_projection cta(snd pair))(event_projection cta word))tail)eqn:Hsame;[|discriminate].
 injection H as <-. exists((w,word)::tail),(event_projection cta word). split.
 - eapply warp_words_from_indices;exact Hwords.
 - intros u found Hin. split.
   + destruct Hin as[Heq|Hin];[inversion Heq;reflexivity|].
     rewrite short_all_same,forallb_forall in Hsame.
     specialize(Hsame (u,found) Hin). apply event_word_eqb_eq. exact Hsame.
   + eapply warp_words_from_trace;[exact Hwords|exact Hin].
Qed.
End Scoped.


Definition scoped_cut (prog:program) (events:list pc) : pc :=
 let first:=fold_left(fun cut event=>if phase_after cut event then event else cut)events(0%nat,0%nat)in
 let placed:=flat_map(fun pair=>let '(b,body):=pair in
   map(fun item=>let '(i,ins):=item in ((b,i),ins))
      (combine(seq 0(List.length body))body))
   (combine(seq 0(List.length prog))prog)in
 fold_left(fun cut pair=>let '(pos,ins):=pair in
   match ins with
   | IBra _ target=>if phase_after cut pos && negb(phase_after cut(target,0%nat))
                   then pos else cut
   | _=>cut end)placed first.

(* A supported event-free segment may use a checked loop descriptor. The
   structural pass contracts its declared exit edge and rejects remaining
   cycles; it never authorizes an execution or replaces certificate checking. *)
Definition occurrence_summary_next (prog:program) (pos:pc) : list pc :=
 match loop_dag_at prog pos with
 | Some shape=>[next_pos prog (fst(lds_branch shape))(snd(lds_branch shape))]
 | None=>occurrence_cfg_next prog pos
 end.
Definition occurrence_segments_supported (prog:program) (events:list pc) (cut:pc) : bool :=
 let positions:=List.length(List.concat prog) in
 occurrence_graph_walk_with (occurrence_summary_next prog)
   (4*positions+2*List.length events+4)%nat events cut
   ((true,(0%nat,0%nat))::map(fun event=>(true,next_pos prog(fst event)(snd event)))events)
   [] [].

Definition occurrence_signed_summary_next (prog:program)(events:list pc)(pos:pc):list pc:=
 match loop_dag_at prog pos with
 | Some shape=>[next_pos prog(fst(lds_branch shape))(snd(lds_branch shape))]
 | None=>match signed_loop_at_event_free prog events pos with
   | Some shape=>[next_pos prog(fst(sls_branch shape))(snd(sls_branch shape))]
   | None=>occurrence_cfg_next prog pos end
 end.
Definition occurrence_signed_segments_supported (prog:program)(events:list pc)(cut:pc):bool:=
 let positions:=List.length(List.concat prog)in
 occurrence_graph_walk_with(occurrence_signed_summary_next prog events)
   (4*positions+2*List.length events+4)%nat events cut
   ((true,(0%nat,0%nat))::map(fun event=>(true,next_pos prog(fst event)(snd event)))events)[][].

Definition compound_has_reg (r:reg)(rs:list reg):=existsb(Pos.eqb r)rs.
Definition compound_drop_written i rs := filter(fun r=>negb(compound_has_reg r(loop_writes i)))rs.
Fixpoint compound_comparisons (pending:list reg)(body:list(nat*instr)) : list(nat*cmpf):=
 match body with
 | []=>[]
 | (index,i)::tail=>
   let rest:=compound_drop_written i pending in
   match i with
   | IAndp None d a b=>compound_comparisons(if compound_has_reg d pending then a::b::rest else rest)tail
   | IMov None d a=>compound_comparisons(if compound_has_reg d pending then a::rest else rest)tail
   | ISetp None c w sg d a b=>
     if compound_has_reg d pending then
       (index,{|cc:=c;cw:=w;csg:=sg;ca:=a;cb:=b|})::compound_comparisons rest tail
     else compound_comparisons rest tail
   | _=>compound_comparisons rest tail end
 end.
Record compound_shape:={compound_base:signed_loop_shape;compound_before:bool}.
Definition compound_candidate(block length:nat)(p:reg)(comparison:nat*cmpf)(ii:nat*instr):option compound_shape:=
 let '(ci,k):=comparison in let '(inc,i):=ii in
 match i with
 | IBin None BAdd width counter source operand=>
   if Nat.eqb width(cw k) && Nat.leb 1 width && Nat.leb width 64 &&
      (match cc k with CLt|CLe=>true|_=>false end)&&
      (Pos.eqb(ca k)counter || Pos.eqb(ca k)source) then
     Some{|compound_base:={|sls_step:={|sl_width:=width;sl_alias_width:=width;sl_counter:=counter;sl_source:=source;sl_operand:=operand;sl_down:=false|};
       sls_predicate:=p;sls_negated:=false;sls_increment:=(block,inc);sls_branch:=(block,Nat.pred length);
       sls_limit:=cb k;sls_comparison:=cc k;sls_signed:=csg k|};
       compound_before:=negb(Pos.eqb(ca k)counter)||Nat.ltb ci inc|}
   else None
 | _=>None end.
Definition compound_block(head:pc)(block:nat)(body:list instr):list compound_shape:=
 match rev(combine(seq 0(List.length body))body)with
 | (_,IBra(Some(false,p))target)::tail=>
   if Nat.eqb target(fst head)&&Nat.eqb(snd head)0&&Nat.leb(fst head)block then
   flat_map(fun comparison=>flat_map(fun ii=>match compound_candidate block(List.length body)p comparison ii with Some d=>[d]|None=>[]end)(combine(seq 0(List.length body))body))
     (compound_comparisons[p]tail)
   else []
 | _=>[]end.
Definition compound_at prog head:=
 match flat_map(fun indexed=>compound_block head(fst indexed)(snd indexed))(combine(seq 0(List.length prog))prog)with
 | [shape]=>
   let base:=compound_base shape in let step:=sls_step base in
   let body:=List.concat(firstn(S(fst(sls_branch base))-fst head)(skipn(fst head)prog))in
   if Pos.eqb(sl_counter step)(sl_source step)||existsb(fun i=>match i with IMov None a b=>Pos.eqb a(sl_source step)&&Pos.eqb b(sl_counter step)|_=>false end)body
   then Some shape else None
 | _=>None end.
Definition compound_event_free prog events head:=
 match compound_at prog head with
 | Some shape=>if short_all(fun event=>phase_after event head||phase_after(sls_branch(compound_base shape))event)events then Some shape else None
 | None=>None end.
Definition compound_propose F prog head shape st:option(astate*signed_loop_certificate):=
 let base:=compound_base shape in let d:=sls_step base in
 if compound_before shape then
 match loop_dag_post F prog(sls_increment base)head st with
 | Some trial_before=>match cmp_range F trial_before(sl_operand d)(Z.of_nat(sl_width d))true with
   | Some(_,sh)=>signed_loop_propose_offset true F prog head base st sh
   | None=>None end
 | None=>None end
 else signed_loop_propose_offset true F prog head base st 0.
Inductive compound_descriptor:=
 | Compound_unsigned(shape:loop_dag_shape)
 | Compound_signed(shape:signed_loop_shape)
 | Compound_prior(shape:compound_shape).
(* This route is used only after the three old structural capabilities are
   absent. True backedges avoid mistaking a forward branch for a loop.
   No failed certificate is retried through another descriptor. *)
Definition compound_descriptor_at prog events pos:option compound_descriptor:=
 match loop_dag_at prog pos with
 | Some shape=>if Nat.leb(fst pos)(fst(lds_branch shape))then Some(Compound_unsigned shape)else None
 | None=>match signed_loop_at_event_free prog events pos with
   | Some shape=>if Nat.leb(fst pos)(fst(sls_branch shape))then Some(Compound_signed shape)else None
   | None=>option_map Compound_prior(compound_event_free prog events pos)
   end
 end.
Definition compound_occurrence_summary F prog events pos st:occurrence_summary_result:=
 match compound_descriptor_at prog events pos with
 | None=>No_loop_summary
 | Some(Compound_unsigned _)=>occurrence_loop_summary F prog events pos st
 | Some(Compound_signed _)=>signed_occurrence_summary F prog events pos st
 | Some(Compound_prior shape)=>signed_summary_check F prog events pos st(compound_propose F prog pos shape st)
 end.
Theorem compound_occurrence_summary_sound:forall F prog E events,
 env_ok F E -> occurrence_summary_ok F prog E events(compound_occurrence_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H.
 unfold compound_occurrence_summary in H.
 destruct(compound_descriptor_at prog events pos)as[[unsigned|signed|prior]|];[ | | |discriminate].
 - exact(occurrence_loop_summary_sound F prog E events HE pos st endpoint post H).
 - exact(signed_occurrence_summary_sound F prog E events HE pos st endpoint post H).
 - eapply signed_summary_check_sound;eassumption.
Qed.
Definition occurrence_compound_next prog events pos:list pc:=
 match compound_descriptor_at prog events pos with
 | Some(Compound_unsigned shape)=>[next_pos prog(fst(lds_branch shape))(snd(lds_branch shape))]
 | Some(Compound_signed shape)=>[next_pos prog(fst(sls_branch shape))(snd(sls_branch shape))]
 | Some(Compound_prior shape)=>[next_pos prog(fst(sls_branch(compound_base shape)))(snd(sls_branch(compound_base shape)))]
 | None=>occurrence_cfg_next prog pos end.
Definition occurrence_compound_segments_supported prog events cut:=
 let positions:=List.length(List.concat prog)in
 occurrence_graph_walk_with(occurrence_compound_next prog events)
  (4*positions+2*List.length events+4)%nat events cut
  ((true,(0%nat,0%nat))::map(fun event=>(true,next_pos prog(fst event)(snd event)))events)[][].

(* UnrolledLoops: a loop whose every iteration the abstract states decide.
   Numba's range loops step by `selp q,1,0,p` on the back edge's predicate,
   often after a copy of the counter that the comparison reads; with a
   constant start and bound, the states at the latch decide the back edge
   in each iteration, and the loop is walked one iteration at a time. *)
Definition unroll_rounds : nat := 64.

(* The latch: the one block whose last instruction is a conditional branch
   back to the head; every event lies outside the blocks from the head to
   the latch. *)
Definition unroll_latch(head:pc)(block:nat)(body:list instr):option(pc*loop_info):=
 match rev body with
 | IBra(Some(neg,p))target::_=>
   if Nat.eqb target(fst head) && Nat.eqb(snd head)0 && Nat.leb(fst head)block then
     Some((block,Nat.pred(List.length body)),
       {|li_prefix:=0;li_suffix:=0;li_counter:=p;li_source:=p;li_stride:=p;li_predicate:=p;li_negated:=neg|})
   else None
 | _=>None end.
Definition unroll_shape(prog:program)(events:list pc)(head:pc):option(pc*loop_info):=
 match flat_map(fun indexed=>match unroll_latch head(fst indexed)(snd indexed)with Some x=>[x]|None=>[]end)
   (combine(seq 0(List.length prog))prog)with
 | [(jump,d)]=>if short_all(fun e=>phase_after e head||phase_after jump e)events then Some(jump,d)else None
 | _=>None end.

(* A compound shape whose step register is a selection of 1 or 0. *)
Definition compound_selp_step(prog:program)(head:pc)(shape:compound_shape):bool:=
 let base:=compound_base shape in
 let body:=List.concat(firstn(S(fst(sls_branch base))-fst head)(skipn(fst head)prog))in
 match sl_operand(sls_step base)with
 | OReg q=>existsb(fun i=>match i with ISelp None _ r(OImm 1)(OImm 0)_=>Pos.eqb r q|_=>false end)body
 | OImm _=>false end.

(* Each round walks the body from the head to the latch, and the branch's
   predicate must be decided: taken, the next round starts at the head with
   the refined state; not taken, the loop ends. *)
Fixpoint unroll_iterate(rounds:nat)(F:lfacts)(prog:program)(events:list pc)(head jump:pc)(d:loop_info)
 (st:astate):option(pc*astate):=
 match rounds with
 | O=>None
 | S rest=>match loop_dag_post F prog jump head st with
   | None=>None
   | Some after=>
     if loop_dag_check phase_fuel F prog[]events head jump st after &&
        negb(occurrence_member events jump) && loop_branch_ok F prog head jump d then
     match signed_loop_branches false F after d head with
     | (None,Some out)=>Some(next_pos prog(fst jump)(snd jump),out)
     | (Some next,None)=>unroll_iterate rest F prog events head jump d next
     | _=>None end
     else None
   end
 end.

Lemma unroll_iterate_sound : forall rounds F prog E events head jump d st endpoint post,
 env_ok F E -> unroll_iterate rounds F prog events head jump d st=Some(endpoint,post) ->
 forall t s,head=pc_of s -> state_ok F t st(regs s) -> regs_ok F E(regs s) ->
 exists n last,loop_event_free_run F prog E events n t s last /\ pc_of last=endpoint /\
 state_ok F(n+t)%nat post(regs last) /\ regs_ok F E(regs last).
Proof.
 induction rounds as[|rounds IH];intros F prog E events head jump d st endpoint post HE H t s Hpos Hst Hrs;
   [discriminate|].
 cbn[unroll_iterate]in H.
 destruct(loop_dag_post F prog jump head st)as[after|];[|discriminate].
 destruct(loop_dag_check phase_fuel F prog[]events head jump st after &&
   negb(occurrence_member events jump) && loop_branch_ok F prog head jump d)eqn:Hconds;[|discriminate].
 rewrite !andb_true_iff,negb_true_iff in Hconds. destruct Hconds as[[Hdag Hfree]Hbranch].
 destruct(loop_dag_check_sound phase_fuel F prog E[]events head jump st after t s HE Hdag Hpos Hst Hrs)
   as[n1[s1[Hrun1[Hpc1[Hst1[Hrs1 _]]]]]].
 destruct(signed_loop_branch_sound false F prog E head jump after d(n1+t)%nat s1
   HE Hbranch(eq_sym Hpc1)Hst1 Hrs1)as[s2[out[taken[Hstep2[Hout[Hpc2[Hst2[Hrs2 _]]]]]]]].
 assert(Hrun2:loop_event_free_run F prog E events(n1+1)%nat t s s2).
 { apply loop_event_free_app with(mid:=s1);[exact Hrun1|].
   econstructor;[rewrite Hpc1;exact Hfree|exact Hstep2|constructor]. }
 destruct(signed_loop_branches false F after d head)as[[next|][leave|]];try discriminate.
 - destruct taken;[|cbn[snd]in Hout;discriminate]. cbn[fst]in Hout. injection Hout as <-.
   destruct(IH F prog E events head jump d next endpoint post HE H(n1+1+t)%nat s2(eq_sym Hpc2)
     ltac:(replace(n1+1+t)%nat with(S(n1+t))by lia;exact Hst2)Hrs2)as[n3[last[Hrun3[Hpc3[Hst3 Hrs3]]]]].
   exists(n1+1+n3)%nat,last. split;[eapply loop_event_free_app;[exact Hrun2|exact Hrun3]|].
   split;[exact Hpc3|]. split;[|exact Hrs3]. replace(n1+1+n3+t)%nat with(n3+(n1+1+t))%nat by lia. exact Hst3.
 - injection H as <- <-. destruct taken;[cbn[fst]in Hout;discriminate|]. cbn[snd]in Hout. injection Hout as <-.
   exists(n1+1)%nat,s2. split;[exact Hrun2|]. split;[exact Hpc2|]. split;[|exact Hrs2].
   replace(n1+1+t)%nat with(S(n1+t))by lia. exact Hst2.
Qed.

(* A Numba range loop is walked by rounds; any other loop keeps its compound
   descriptor. This selection precedes every certificate check. *)
Definition unrolled_loop_at(prog:program)(events:list pc)(pos:pc):option(pc*loop_info):=
 match compound_descriptor_at prog events pos with
 | Some(Compound_prior shape)=>if compound_selp_step prog pos shape then unroll_shape prog events pos else None
 | Some _=>None
 | None=>unroll_shape prog events pos end.
Definition unrolled_occurrence_summary F prog events pos st:occurrence_summary_result:=
 match unrolled_loop_at prog events pos with
 | Some(jump,d)=>match unroll_iterate unroll_rounds F prog events pos jump d st with
   | Some(endpoint,post)=>Checked_loop_summary endpoint post
   | None=>Rejected_loop_summary end
 | None=>compound_occurrence_summary F prog events pos st end.
Theorem unrolled_occurrence_summary_sound:forall F prog E events,
 env_ok F E -> occurrence_summary_ok F prog E events(unrolled_occurrence_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H. unfold unrolled_occurrence_summary in H.
 destruct(unrolled_loop_at prog events pos)as[[jump d]|].
 - destruct(unroll_iterate unroll_rounds F prog events pos jump d st)as[[e out]|]eqn:Hiter;[|discriminate].
   injection H as <- <-. intros t s Hpos Hst Hrs.
   exact(unroll_iterate_sound unroll_rounds F prog E events pos jump d st e out HE Hiter t s Hpos Hst Hrs).
 - exact(compound_occurrence_summary_sound F prog E events HE pos st endpoint post H).
Qed.
Definition occurrence_unrolled_next prog events pos:list pc:=
 match unrolled_loop_at prog events pos with
 | Some(jump,_)=>[next_pos prog(fst jump)(snd jump)]
 | None=>occurrence_compound_next prog events pos end.
Definition occurrence_unrolled_segments_supported prog events cut:=
 let positions:=List.length(List.concat prog)in
 occurrence_graph_walk_with(occurrence_unrolled_next prog events)
  (4*positions+2*List.length events+4)%nat events cut
  ((true,(0%nat,0%nat))::map(fun event=>(true,next_pos prog(fst event)(snd event)))events)[][].

(* Capability dispatch precedes certificate checking. Every previously
   supported program keeps exactly its original callback and word checker. *)
Definition scoped_summary_select F prog events cut : option(pc->astate->occurrence_summary_result):=
 if occurrence_segments_acyclic prog events cut then Some no_loop_summary else
 if occurrence_segments_supported prog events cut then Some(occurrence_loop_summary F prog events)else
 if occurrence_signed_segments_supported prog events cut then Some(occurrence_signed_loop_summary F prog events)else
 if occurrence_compound_segments_supported prog events cut then Some(compound_occurrence_summary F prog events)else
 if occurrence_unrolled_segments_supported prog events cut then Some(unrolled_occurrence_summary F prog events)
 else None.
Lemma scoped_summary_select_sound : forall F prog E events cut summarize,
 env_ok F E -> scoped_summary_select F prog events cut=Some summarize ->
 occurrence_summary_ok F prog E events summarize.
Proof.
 intros F prog E events cut summarize HE H. unfold scoped_summary_select in H.
 destruct(occurrence_segments_acyclic prog events cut).
 - injection H as <-. apply no_loop_summary_ok.
 - destruct(occurrence_segments_supported prog events cut).
   + injection H as <-. apply occurrence_loop_summary_sound;exact HE.
   + destruct(occurrence_signed_segments_supported prog events cut).
     * injection H as <-. apply occurrence_signed_loop_summary_sound;exact HE.
     * destruct(occurrence_compound_segments_supported prog events cut).
       -- injection H as <-. apply compound_occurrence_summary_sound;exact HE.
       -- destruct(occurrence_unrolled_segments_supported prog events cut);[|discriminate].
          injection H as <-. apply unrolled_occurrence_summary_sound;exact HE.
Qed.

Definition scoped_complete_word (F:lfacts)(prog:program)(events:list pc):option(list pc):=
 let cut:=scoped_cut prog events in
 if lf_ok F && Nat.ltb(fst cut)(List.length prog) && phase_closed prog cut &&
    short_all(fun e=>phase_event_at prog e && negb(phase_after cut e))events then
 match scoped_summary_select F prog events cut with
 | Some summarize=>occurrence_word_with summarize phase_fuel phase_fuel(phase_control_regs(List.concat prog))
     F prog events cut(0%nat,0%nat)(empty_state F)
 | None=>None end
 else None.

(* The antecedent is the previous complete-word definition verbatim. *)
Theorem scoped_complete_word_preserves : forall (F:lfacts)(prog:program)(events word:list pc),
 (let cut:=scoped_cut prog events in
 if lf_ok F && Nat.ltb(fst cut)(List.length prog) && phase_closed prog cut &&
    short_all(fun e=>phase_event_at prog e && negb(phase_after cut e))events then
 let acyclic:=occurrence_segments_acyclic prog events cut in
 if (if acyclic then true else occurrence_segments_supported prog events cut) then
 let summarize:=if acyclic
                then no_loop_summary else occurrence_loop_summary F prog events in
 occurrence_word_with summarize phase_fuel phase_fuel(phase_control_regs(List.concat prog))
   F prog events cut(0%nat,0%nat)(empty_state F)
 else None
 else None)=Some word -> scoped_complete_word F prog events=Some word.
Proof.
 intros F prog events word H. unfold scoped_complete_word,scoped_summary_select.
 cbv zeta in H |- *.
 destruct(lf_ok F && Nat.ltb(fst(scoped_cut prog events))(List.length prog) &&
  phase_closed prog(scoped_cut prog events) &&
  short_all(fun e=>phase_event_at prog e && negb(phase_after(scoped_cut prog events)e))events);[|discriminate].
 destruct(occurrence_segments_acyclic prog events(scoped_cut prog events));[exact H|].
 destruct(occurrence_segments_supported prog events(scoped_cut prog events));[exact H|discriminate].
Qed.

(* Preserve the complete previous three-capability computation as well.
   The compound capability is selected only when that computation had no
   supported callback, before any certificate collection begins. *)
Theorem scoped_signed_complete_word_preserves : forall (F:lfacts)(prog:program)(events word:list pc),
 (let cut:=scoped_cut prog events in
 if lf_ok F && Nat.ltb(fst cut)(List.length prog) && phase_closed prog cut &&
    short_all(fun e=>phase_event_at prog e && negb(phase_after cut e))events then
 let selected:=
   if occurrence_segments_acyclic prog events cut then Some no_loop_summary else
   if occurrence_segments_supported prog events cut then Some(occurrence_loop_summary F prog events)else
   if occurrence_signed_segments_supported prog events cut then Some(occurrence_signed_loop_summary F prog events)
   else None in
 match selected with
 | Some summarize=>occurrence_word_with summarize phase_fuel phase_fuel(phase_control_regs(List.concat prog))
     F prog events cut(0%nat,0%nat)(empty_state F)
 | None=>None end
 else None)=Some word -> scoped_complete_word F prog events=Some word.
Proof.
 intros F prog events word H. unfold scoped_complete_word,scoped_summary_select.
 cbv zeta in H |- *.
 destruct(lf_ok F && Nat.ltb(fst(scoped_cut prog events))(List.length prog) &&
  phase_closed prog(scoped_cut prog events) &&
  short_all(fun e=>phase_event_at prog e && negb(phase_after(scoped_cut prog events)e))events);[|discriminate].
 destruct(occurrence_segments_acyclic prog events(scoped_cut prog events));[exact H|].
 destruct(occurrence_segments_supported prog events(scoped_cut prog events));[exact H|].
 destruct(occurrence_signed_segments_supported prog events(scoped_cut prog events));[exact H|discriminate].
Qed.

Theorem scoped_complete_word_sound : forall F prog events word,
 scoped_complete_word F prog events=Some word -> forall E,launch_ok F E ->
 occurrence_trace F prog E events word 0(phase_initial E).
Proof.
 intros F prog events word H E Hlaunch. unfold scoped_complete_word in H.
 set(cut:=scoped_cut prog events)in *.
 destruct(lf_ok F && Nat.ltb(fst cut)(List.length prog) && phase_closed prog cut &&
   short_all(fun e=>phase_event_at prog e && negb(phase_after cut e))events)eqn:Hbase;[|discriminate].
 destruct(scoped_summary_select F prog events cut)as[summarize|]eqn:Hsupport;[|discriminate].
 rewrite !andb_true_iff in Hbase. destruct Hbase as[[[Hlf Hbound]Hclosed]Hevents].
 apply Nat.ltb_lt in Hbound. rewrite short_all_same,forallb_forall in Hevents.
 eapply occurrence_word_with_sound with (summarize:=summarize).
 - exact(env_of_launch F prog E Hlf Hlaunch).
 - eapply scoped_summary_select_sound;[exact(env_of_launch F prog E Hlf Hlaunch)|exact Hsupport].
 - exact Hbound.
 - exact Hclosed.
 - intros event Hin. specialize(Hevents event Hin). apply andb_true_iff in Hevents.
   apply negb_true_iff. exact(proj2 Hevents).
 - exact H.
 - reflexivity.
 - split.
   + intros r f Hf. unfold get,empty_state in Hf;cbn[amap]in Hf. rewrite PM.gempty in Hf. discriminate.
   + exact(lk_derived F E Hlaunch).
 - split;[exact(lk_init F E Hlaunch)|]. intros r Hfixed;reflexivity.
Qed.

Definition warp_scoped_check := scoped_words_check scoped_complete_word.
Theorem warp_scoped_check_sound : forall F prog events cta r count words,
 warp_scoped_check F prog events cta r count=Some words ->
 scoped_schedule_safe F prog events cta r count.
Proof.
 intros. eapply scoped_words_check_sound;[exact scoped_complete_word_sound|exact H].
Qed.

