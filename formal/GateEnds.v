(* Barrier participation through loops that GateTerm proves end.

   The occurrence word of a kernel needs, for each event-free loop before
   a waiting site, that every thread leaves the loop. GateTerm proves that
   every thread of an accepted kernel ends (`kernel_ends`). A thread that
   ends and that enters an event-free loop whose ways out all lead to one
   position leaves the loop at that position, so such a loop is summarized
   by the ranges of the registers that it does not write, and its ending
   is the termination check's obligation. The guarantee of this route is stated for
   a kernel that ends; the certificate check proves that it ends.

   The file also selects one occurrence word, instead of a progress proof
   for each event, for a program whose segments between events are acyclic. *)

From Coq Require Import List Arith ZArith Bool Lia.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning GateSoundRange GateSoundJoin GateSoundTransfer
  GateThread GateBarrier GateLoop GateTerm GateContext.
Import ListNotations.
Local Open Scope Z_scope.

(* EndsRun *)
(* The run from state s at step t reaches a position without an instruction. *)
Definition phase_ends(F:lfacts)(prog:program)(E:env)(t:nat)(s:state):Prop:=
 exists m last,phase_run F prog E m t s=Some last /\ fetch prog(blk last)(idx last)=None.

Lemma phase_ends_step:forall F prog E t s next,
 phase_ends F prog E t s ->phase_step F prog E t s=Some next ->phase_ends F prog E(S t)next.
Proof.
 intros F prog E t s next[m[last[Hrun Hend]]]Hstep. destruct m as[|m].
 - cbn[phase_run]in Hrun. injection Hrun as <-. unfold phase_step,step in Hstep.
   rewrite Hend in Hstep. discriminate.
 - cbn[phase_run]in Hrun. rewrite Hstep in Hrun. exists m,last. split;assumption.
Qed.

Lemma phase_ends_run:forall F prog E events n t s last,
 loop_event_free_run F prog E events n t s last ->phase_ends F prog E t s ->phase_ends F prog E(n+t)last.
Proof.
 intros F prog E events n t s last H. induction H as[t s|n t s mid last Hfree Hstep Hrest IH];intros Hends.
 - exact Hends.
 - replace(S n+t)%nat with(n+S t)%nat by lia. apply IH. eapply phase_ends_step;eassumption.
Qed.

Lemma ends_run_initial:forall F prog E t s,
 run' F prog E t=Some s ->phase_run F prog E t 0(phase_initial E)=Some s.
Proof.
 intros F prog E t. induction t as[|t IH];intros s H.
 - exact H.
 - rewrite term_run_succ in H. rewrite warp_run_last,Nat.add_0_r.
   destruct(run' F prog E t)as[s1|]eqn:Hs1;[|discriminate]. rewrite(IH s1 eq_refl). exact H.
Qed.

Lemma kernel_ends_initial:forall F prog,kernel_ends F prog ->
 forall E,launch_ok F E ->phase_ends F prog E 0(phase_initial E).
Proof.
 intros F prog Hends E Hlaunch. destruct(Hends E Hlaunch)as[t[s[Hrun Hfetch]]].
 exists t,s. split;[exact(ends_run_initial F prog E t s Hrun)|exact Hfetch].
Qed.

(* EndsWord *)
Section EndsWord.
Variable F:lfacts.
Variable prog:program.
Variable E:env.
Hypothesis HE:env_ok F E.
Variable events:list pc.

(* A loop summary whose run exists for every thread that ends. *)
Definition occurrence_summary_ends_ok(summarize:pc->astate->occurrence_summary_result):Prop:=
 forall pos st endpoint post,summarize pos st=Checked_loop_summary endpoint post ->
 forall t s,pos=pc_of s ->state_ok F t st(regs s)->regs_ok F E(regs s)->phase_ends F prog E t s ->
 exists n last,loop_event_free_run F prog E events n t s last /\ pc_of last=endpoint /\
 state_ok F(n+t)%nat post(regs last)/\regs_ok F E(regs last).

Lemma ends_collect_continuation:forall summarize,
 occurrence_summary_ends_ok summarize ->forall fuel keep cut pending remaining arrivals
 (P:nat->state->Prop),
 occurrence_collect_with summarize fuel keep F prog events cut pending=Some(remaining,arrivals) ->
 (forall t s next,occurrence_member events(pc_of s)=false ->
   phase_step F prog E t s=Some next ->P(S t)next ->P t s) ->
 (forall pos st,In(pos,st)arrivals ->forall t s,pos=pc_of s ->
   state_ok F t st(regs s)->regs_ok F E(regs s)->phase_ends F prog E t s ->P t s) ->
 forall pos st,In(pos,Some st)pending ->forall t s,pos=pc_of s ->
   state_ok F t st(regs s)->regs_ok F E(regs s)->phase_ends F prog E t s ->P t s.
Proof.
 intros summarize Hsummary. induction fuel as[|fuel IH];intros keep cut pending remaining arrivals P Hcheck Hnext Hend
 pos st Hin t s Hpos Hst Hrs Hends;
 destruct pending as[|[q ost]tail];[contradiction|discriminate|contradiction|].
 cbn[occurrence_collect_with]in Hcheck. destruct ost as[current|].
 - destruct(is_bottom current)eqn:Hbottom.
   + destruct Hin as[Heq|Hin].
     * injection Heq as <- <-. exfalso. exact(bottom_not_ok F t current _ Hbottom Hst).
     * eapply IH;eassumption.
   + destruct(occurrence_member events q||phase_after cut q)eqn:Hstop.
     * destruct(occurrence_collect_with summarize fuel keep F prog events cut tail)as[[rest ends]|]eqn:Htail;
       [|discriminate]. injection Hcheck as <- <-.
       destruct Hin as[Heq|Hin].
       -- injection Heq as <- <-. eapply Hend;[left;reflexivity|exact Hpos|exact Hst|exact Hrs|exact Hends].
       -- eapply IH;[exact Htail|exact Hnext| |exact Hin|exact Hpos|exact Hst|exact Hrs|exact Hends].
          intros p a Ha u v Hp Hv Hrv Hev. eapply Hend;[right;exact Ha|exact Hp|exact Hv|exact Hrv|exact Hev].
     * destruct(summarize q current)as[| |endpoint post]eqn:Hsummary_step;[|discriminate|].
       { destruct(fetch prog(fst q)(snd q))as[i|]eqn:Hfetch;[|discriminate].
       destruct(wf_instr F i)eqn:Hwf;[|discriminate].
       destruct Hin as[Heq|Hin].
       -- injection Heq as <- <-. subst q.
          assert(Hf:fetch prog(blk s)(idx s)=Some i)by exact Hfetch.
          assert(Hex:exists next,phase_step F prog E t s=Some next).
          {unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
          destruct Hex as[next Hstep].
          destruct(phase_successors_complete F prog E HE keep t s next current i Hf Hstep Hst Hrs Hwf)
          as[out[Hchild[Hout Hregs]]].
          eapply Hnext;[apply orb_false_iff in Hstop;exact(proj1 Hstop)|exact Hstep|].
          eapply IH;[exact Hcheck|exact Hnext|exact Hend| |reflexivity|exact Hout|exact Hregs|].
          ++ apply in_or_app;left;exact Hchild.
          ++ eapply phase_ends_step;eassumption.
       -- eapply IH;[exact Hcheck|exact Hnext|exact Hend| |exact Hpos|exact Hst|exact Hrs|exact Hends].
          apply in_or_app;right;exact Hin. }
       { destruct Hin as[Heq|Hin].
         - injection Heq as <- <-.
           destruct(Hsummary q current endpoint post Hsummary_step t s Hpos Hst Hrs Hends)
             as[n[last[Hrun[Hpc[Hout Hregs]]]]].
           eapply loop_event_free_continuation;[exact Hrun|exact Hnext|].
           eapply IH;[exact Hcheck|exact Hnext|exact Hend|left;reflexivity|symmetry;exact Hpc|exact Hout|exact Hregs|].
           eapply phase_ends_run;eassumption.
         - eapply IH;[exact Hcheck|exact Hnext|exact Hend|right;exact Hin|exact Hpos|exact Hst|exact Hrs|exact Hends]. }
 - destruct Hin as[Heq|Hin];[discriminate|]. eapply IH;eassumption.
Qed.

Theorem ends_word_sound:forall summarize,
 occurrence_summary_ends_ok summarize ->forall rounds fuel keep cut pos st word,
 (fst cut<List.length prog)%nat ->phase_closed prog cut=true ->
 (forall e,In e events ->phase_after cut e=false) ->
 occurrence_word_with summarize rounds fuel keep F prog events cut pos st=Some word ->
 forall t s,pos=pc_of s ->state_ok F t st(regs s)->regs_ok F E(regs s)->phase_ends F prog E t s ->
 occurrence_trace F prog E events word t s.
Proof.
 intros summarize Hsummary. induction rounds as[|rounds IH];intros fuel keep cut pos st word Hb Hc Hbefore Hcheck t s Hpos Hst Hrs Hends;
 [discriminate|]. cbn[occurrence_word_with]in Hcheck.
 destruct(occurrence_collect_with summarize fuel keep F prog events cut[(pos,Some st)])
   as[[remaining arrivals]|]eqn:Hcollect;[|discriminate].
 destruct arrivals as[|[event first]tail];[discriminate|].
 destruct(occurrence_member events event)eqn:Hevent.
 - set(inv:=occurrence_join F first tail)in *.
   destruct(short_all(fun entry=>phase_event event(fst entry)&&state_le F(snd entry)inv)
     ((event,first)::tail))eqn:Hall;[|discriminate].
   destruct remaining as[|remaining];[discriminate|].
   destruct(fetch prog(fst event)(snd event))as[i|]eqn:Hfetch;[|discriminate].
   destruct(phase_marker i&&wf_instr F i)eqn:Hmarker;[|discriminate].
   destruct(occurrence_word_with summarize rounds remaining keep F prog events cut
     (next_pos prog(fst event)(snd event))(phase_trim keep(transfer F inv i)))
     as[rest|]eqn:Hrest;[|discriminate]. injection Hcheck as <-.
   apply andb_true_iff in Hmarker. destruct Hmarker as[Hmark Hwf].
   rewrite short_all_same,forallb_forall in Hall.
   eapply ends_collect_continuation;[exact Hsummary|exact Hcollect| | |left;reflexivity|exact Hpos|exact Hst|exact Hrs|exact Hends].
   + intros u v next Hnot Hstep Htrace. eapply Occurrence_later;eassumption.
   + intros p a Hin u v Hp Ha Hr Hv.
     specialize(Hall _ Hin). cbn[fst snd]in Hall. apply andb_true_iff in Hall.
     destruct Hall as[Hsame Hle]. apply phase_event_eq in Hsame. subst p.
     assert(Hi:state_ok F u inv(regs v)).
     {eapply state_le_sound;eassumption. }
     assert(Hf:fetch prog(blk v)(idx v)=Some i)by(rewrite Hp in Hfetch;exact Hfetch).
     assert(Hex:exists next,phase_step F prog E u v=Some next).
     {unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
     destruct Hex as[next Hstep].
     destruct(transfer_sound F prog E HE u v next inv i Hf Hstep Hi Hr Hwf)as[Ho Hregs].
     assert(Hnext:pc_of next=next_pos prog(fst event)(snd event)).
     {pose proof(step_pos F prog E u v next i Hf Hstep)as H.
      destruct H as[[H _]|[[g[target[Hbad _]]]|[g[Hbad _]]]].
      - change(pc_of next=next_pos prog(fst(pc_of v))(snd(pc_of v)))in H. rewrite <-Hp in H. exact H.
      - exfalso. exact(proj1(phase_marker_not_control i Hmark)g target Hbad).
      - exfalso. exact(proj2(phase_marker_not_control i Hmark)g Hbad). }
     eapply Occurrence_now;[rewrite <-Hp;exact Hevent|symmetry;exact Hp|exact Hstep|].
     eapply IH;[exact Hb|exact Hc|exact Hbefore|exact Hrest|symmetry;exact Hnext| |exact Hregs|].
     * apply phase_trim_sound. exact Ho.
     * eapply phase_ends_step;eassumption.
 - destruct(short_all(fun entry=>phase_after cut(fst entry))((event,first)::tail))eqn:Hall;
   [|discriminate]. injection Hcheck as <-.
   rewrite short_all_same,forallb_forall in Hall.
   eapply ends_collect_continuation;[exact Hsummary|exact Hcollect| | |left;reflexivity|exact Hpos|exact Hst|exact Hrs|exact Hends].
   + intros u v next Hnot Hstep Htrace. eapply Occurrence_later;eassumption.
   + intros p a Hin u v Hp Ha Hr Hv. eapply occurrence_cut_trace;[exact Hb|exact Hc|exact Hbefore|].
     rewrite <-Hp. exact(Hall _ Hin).
Qed.
End EndsWord.

(* EndsRegion *)
(* The loop of head block h: its blocks h to its one latch, every way out of
   them leading to one position, and neither an event nor an exit inside. *)
Definition ends_latches(prog:program)(h:nat):list nat:=
 flat_map(fun b=>match fetch prog b(block_len prog b-1)with
   |Some(IBra _ t)=>if Nat.eqb t h&&Nat.leb h b then[b]else[]
   |_=>[]end)(seq h(List.length prog-h)).
Definition ends_positions(prog:program)(h l:nat):list pc:=
 flat_map(fun b=>map(fun i=>(b,i))(seq 0(block_len prog b)))(seq h(S l-h)).
Definition ends_region(prog:program)(events:list pc)(head:pc):option(pc*list pc):=
 if negb(Nat.eqb(snd head)0)then None else
 match ends_latches prog(fst head)with
 |[l]=>
   let region:=ends_positions prog(fst head)l in
   let outs:=flat_map(fun pos=>filter(fun q=>negb(occurrence_member region q))(occurrence_cfg_next prog pos))region in
   match outs with
   |exit::rest=>
     if occurrence_member region head&&
        short_all(fun q=>phase_event exit q)rest&&
        short_all(fun pos=>negb(occurrence_member events pos)&&
          match fetch prog(fst pos)(snd pos)with Some i=>context_not_exit i|None=>false end)region
     then Some(exit,region)else None
   |[]=>None end
 |_=>None end.
Definition ends_written(prog:program)(region:list pc):list reg:=
 flat_map(fun pos=>match fetch prog(fst pos)(snd pos)with Some i=>loop_writes i|None=>[]end)region.
(* The registers that decide control flow and that the loop does not write. *)
Definition ends_kept(F:lfacts)(prog:program)(region:list pc):list reg:=
 filter(fun r=>negb(existsb(Pos.eqb r)(ends_written prog region))&&negb(is_cell F r))
   (map fst(PM.elements(phase_control_regs(List.concat prog)))).
(* An event-free loop with one way out keeps the ranges of the registers
   that it does not write; any other loop keeps the occurrence route's
   summary. *)
Definition ends_region_summary(F:lfacts)(prog:program)(events:list pc)(pos:pc)(st:astate):occurrence_summary_result:=
 match ends_region prog events pos with
 |Some(exit,region)=>
   if short_all(fun q=>match fetch prog(fst q)(snd q)with Some i=>wf_instr F i|None=>false end)region
   then Checked_loop_summary exit(context_range_state(mixed_ranges F st(ends_kept F prog region)))
   else Rejected_loop_summary
 |None=>No_loop_summary end.
Definition ends_summary(F:lfacts)(prog:program)(events:list pc)(pos:pc)(st:astate):occurrence_summary_result:=
 match ends_region prog events pos with
 |Some _=>ends_region_summary F prog events pos st
 |None=>occurrence_loop_summary F prog events pos st end.

Lemma ends_region_spec:forall prog events head exit region,
 ends_region prog events head=Some(exit,region) ->
 In head region /\
 forall pos,In pos region ->occurrence_member events pos=false /\
   exists i,fetch prog(fst pos)(snd pos)=Some i /\ context_not_exit i=true /\
   forall q,In q(occurrence_cfg_next prog pos)->In q region \/ q=exit.
Proof.
 intros prog events head exit region H. unfold ends_region in H.
 destruct(negb(Nat.eqb(snd head)0));[discriminate|].
 destruct(ends_latches prog(fst head))as[|l[|]];try discriminate.
 set(region0:=ends_positions prog(fst head)l)in *.
 set(outs:=flat_map(fun pos=>filter(fun q=>negb(occurrence_member region0 q))(occurrence_cfg_next prog pos))region0)in *.
 destruct outs as[|e rest]eqn:Houts;[discriminate|].
 destruct(occurrence_member region0 head&&short_all(fun q=>phase_event e q)rest&&
   short_all(fun pos=>negb(occurrence_member events pos)&&
   match fetch prog(fst pos)(snd pos)with Some i=>context_not_exit i|None=>false end)region0)eqn:Hc;[|discriminate].
 injection H as <- <-. rewrite !andb_true_iff in Hc. destruct Hc as[[Hhead Hsame]Hall].
 rewrite short_all_same,forallb_forall in Hsame,Hall.
 assert(Hout:forall q,In q outs ->q=e).
 {rewrite Houts. intros q[Hq|Hq];[symmetry;exact Hq|]. specialize(Hsame q Hq). apply phase_event_eq in Hsame. exact Hsame. }
 split;[apply occurrence_member_in;exact Hhead|].
 intros pos Hin. specialize(Hall pos Hin). apply andb_true_iff in Hall. destruct Hall as[Hfree Hfetch].
   split;[now apply negb_true_iff|].
   destruct(fetch prog(fst pos)(snd pos))as[i|];[|discriminate]. exists i. split;[reflexivity|]. split;[exact Hfetch|].
   intros q Hq. destruct(occurrence_member region0 q)eqn:Hm;[left;now apply occurrence_member_in|].
   right. apply Hout. unfold outs. apply in_flat_map. exists pos. split;[exact Hin|].
   apply filter_In. split;[exact Hq|]. now rewrite Hm.
Qed.

Section EndsRegionRun.
Variable F:lfacts.
Variable prog:program.
Variable E:env.
Hypothesis HE:env_ok F E.
Variable events:list pc.
Variables exit:pc.
Variable region:list pc.
Hypothesis Hinside:forall pos,In pos region ->occurrence_member events pos=false /\
   exists i,fetch prog(fst pos)(snd pos)=Some i /\ context_not_exit i=true /\
   forall q,In q(occurrence_cfg_next prog pos)->In q region \/ q=exit.
Hypothesis Hwf:forall pos i,In pos region ->fetch prog(fst pos)(snd pos)=Some i ->wf_instr F i=true.

Lemma ends_kept_preserves:forall pos i r,In pos region ->fetch prog(fst pos)(snd pos)=Some i ->
 In r(ends_kept F prog region)->is_cell F r=false /\ loop_preserves r i=true.
Proof.
 intros pos i r Hin Hf Hr. unfold ends_kept in Hr. apply filter_In in Hr. destruct Hr as[_ Hr].
 apply andb_true_iff in Hr. destruct Hr as[Hw Hc]. apply negb_true_iff in Hw,Hc. split;[exact Hc|].
 unfold loop_preserves. apply negb_true_iff. destruct(existsb(Pos.eqb r)(loop_writes i))eqn:Hx;[|reflexivity].
 exfalso. assert(Hy:existsb(Pos.eqb r)(ends_written prog region)=true).
 {apply existsb_exists in Hx. destruct Hx as[x[Hx Heq]]. apply existsb_exists.
  exists x. split;[|exact Heq]. unfold ends_written. apply in_flat_map. exists pos. split;[exact Hin|]. now rewrite Hf. }
 congruence.
Qed.

Lemma ends_region_run:forall m t s last,In(pc_of s)region ->regs_ok F E(regs s)->
 phase_run F prog E m t s=Some last ->fetch prog(blk last)(idx last)=None ->
 exists n out,loop_event_free_run F prog E events n t s out /\ pc_of out=exit /\
 regs_ok F E(regs out)/\forall r,In r(ends_kept F prog region)->regs out r=regs s r.
Proof.
 induction m as[|m IH];intros t s last Hin Hrs Hrun Hend.
 - cbn[phase_run]in Hrun. injection Hrun as <-. destruct(Hinside _ Hin)as[_[i[Hf _]]].
   unfold pc_of in Hf. cbn[fst snd]in Hf. congruence.
 - destruct(Hinside _ Hin)as[Hfree[i[Hf[Hnot Hsucc]]]].
   assert(Hfs:fetch prog(blk s)(idx s)=Some i)by exact Hf.
   cbn[phase_run]in Hrun. destruct(phase_step F prog E t s)as[next|]eqn:Hstep;[|discriminate].
   assert(Hnrs:regs_ok F E(regs next)).
   {eapply context_step_regs;[exact HE|exact Hfs|exact Hstep|exact(Hwf _ _ Hin Hf)|exact Hrs]. }
   assert(Hkeep:forall r,In r(ends_kept F prog region)->regs next r=regs s r).
   {intros r Hr. destruct(ends_kept_preserves _ _ _ Hin Hf Hr)as[Hc Hp].
    exact(loop_step_preserves F prog E t s next i r Hc Hfs Hstep Hp). }
   pose proof(context_cfg_complete F prog E t s next i Hnot Hfs Hstep)as Hcfg.
   destruct(Hsucc _ Hcfg)as[Hin'|Hat].
   + destruct(IH(S t)next last Hin' Hnrs Hrun Hend)as[n[out[Hrest[Hpc[Hors Hokeep]]]]].
     exists(S n),out. split;[econstructor;[exact Hfree|exact Hstep|exact Hrest]|].
     split;[exact Hpc|]. split;[exact Hors|]. intros r Hr. rewrite(Hokeep r Hr). exact(Hkeep r Hr).
   + exists 1%nat,next. split;[econstructor;[exact Hfree|exact Hstep|constructor]|].
     split;[exact Hat|]. split;[exact Hnrs|]. exact Hkeep.
Qed.
End EndsRegionRun.

Lemma mixed_ranges_in:forall F st registers r lo hi,In(r,(lo,hi))(mixed_ranges F st registers)->
 In r registers /\ counted_range_check F st r(wdn F r)lo hi=true.
Proof.
 intros F st registers r lo hi H. unfold mixed_ranges in H. apply in_flat_map in H.
 destruct H as[x[Hx Hin]]. destruct(cmp_range F st(OReg x)(Z.of_nat(wdn F x))true)as[[a b]|];[|contradiction].
 destruct(counted_range_check F st x(wdn F x)a b)eqn:Hc;[|contradiction].
 destruct Hin as[Heq|[]]. injection Heq as -> -> ->. split;assumption.
Qed.

Lemma ends_region_summary_ok:forall F prog E events,env_ok F E ->
 occurrence_summary_ends_ok F prog E events(ends_region_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H t s Hpos Hst Hrs[m[last[Hrun Hend]]].
 unfold ends_region_summary in H.
 destruct(ends_region prog events pos)as[[exit region]|]eqn:Hregion;[|discriminate].
 destruct(short_all(fun q=>match fetch prog(fst q)(snd q)with Some i=>wf_instr F i|None=>false end)region)eqn:Hwf;
 [|discriminate]. injection H as <- <-.
 destruct(ends_region_spec prog events pos exit region Hregion)as[Hhead Hinside].
 assert(Hwf':forall q i,In q region ->fetch prog(fst q)(snd q)=Some i ->wf_instr F i=true).
 {intros q i Hq Hf. rewrite short_all_same,forallb_forall in Hwf. specialize(Hwf q Hq). now rewrite Hf in Hwf. }
 destruct(ends_region_run F prog E HE events exit region Hinside Hwf' m t s last
   ltac:(rewrite <-Hpos;exact Hhead)Hrs Hrun Hend)as[n[out[Hfree[Hpc[Hors Hkeep]]]]].
 exists n,out. split;[exact Hfree|]. split;[exact Hpc|]. split;[|exact Hors].
 apply context_range_state_sound. intros r lo hi Hr.
 destruct(mixed_ranges_in F st _ r lo hi Hr)as[Hk Hc]. rewrite(Hkeep r Hk).
 eapply(counted_range_sound F E st t(regs s)r(wdn F r)lo hi);eassumption.
Qed.

Theorem ends_summary_ok:forall F prog E events,env_ok F E ->
 occurrence_summary_ends_ok F prog E events(ends_summary F prog events).
Proof.
 intros F prog E events HE pos st endpoint post H t s Hpos Hst Hrs Hends.
 unfold ends_summary in H. destruct(ends_region prog events pos).
 - exact(ends_region_summary_ok F prog E events HE pos st endpoint post H t s Hpos Hst Hrs Hends).
 - destruct(occurrence_loop_summary_sound F prog E events HE pos st endpoint post H t s Hpos Hst Hrs)
     as[n[last Hrest]]. exists n,last. exact Hrest.
Qed.

(* EndsRoute *)
(* Selected for a program that the occurrence route takes and whose segments
   between events hold an event-free cycle. *)
Definition ends_route_selected(prog:program)(events:list pc):bool:=
 if occurrence_case prog events
 then negb(occurrence_segments_acyclic prog events(occurrence_cut prog events))else false.
Definition ends_progress_check(F:lfacts)(prog:program)(events:list pc):bool:=
 match occurrence_checked_word(ends_summary F prog events)F prog events with Some _=>true|None=>false end.

Theorem ends_progress_check_sound:forall F prog events,
 ends_progress_check F prog events=true ->kernel_ends F prog ->occurrence_schedule_safe F prog events.
Proof.
 intros F prog events Hcheck Hends. unfold ends_progress_check,occurrence_checked_word in Hcheck.
 set(cut:=occurrence_cut prog events)in *.
 destruct(lf_ok F&&Nat.ltb(fst cut)(List.length prog)&&phase_closed prog cut&&
   short_all(fun e=>phase_event_at prog e&&negb(phase_after cut e))events)eqn:Hbase;[|discriminate].
 destruct(occurrence_word_with(ends_summary F prog events)phase_fuel phase_fuel(phase_control_regs(List.concat prog))
   F prog events cut(0%nat,0%nat)(empty_state F))as[found|]eqn:Hword;[|discriminate].
 destruct(short_all(fun e=>occurrence_member found e)events)eqn:Hcover;[|discriminate].
 rewrite !andb_true_iff in Hbase. destruct Hbase as[[[Hlf Hbound]Hclosed]Hevents].
 apply Nat.ltb_lt in Hbound. rewrite short_all_same,forallb_forall in Hevents,Hcover.
 split.
 - intros event Hin. specialize(Hevents _ Hin). apply andb_true_iff in Hevents. exact(proj1 Hevents).
 - exists found. split.
   + intros event Hin. apply occurrence_member_in. exact(Hcover _ Hin).
   + intros E Hlaunch. pose proof(env_of_launch F prog E Hlf Hlaunch)as HE.
     eapply(ends_word_sound F prog E HE events(ends_summary F prog events)(ends_summary_ok F prog E events HE)).
     * exact Hbound.
     * exact Hclosed.
     * intros event Hin. specialize(Hevents _ Hin). apply andb_true_iff in Hevents.
       apply negb_true_iff. exact(proj2 Hevents).
     * exact Hword.
     * reflexivity.
     * split.
       -- intros r f Hf. unfold get,empty_state in Hf;cbn[amap]in Hf. rewrite PM.gempty in Hf. discriminate.
       -- exact(lk_derived F E Hlaunch).
     * split;[exact(lk_init F E Hlaunch)|]. intros r Hfixed;reflexivity.
     * exact(kernel_ends_initial F prog Hends E Hlaunch).
Qed.

(* A program whose events lie outside loops, outside the shapes of the
   straight and tail checks, and whose segments between events are acyclic:
   one occurrence word replaces a separate progress proof for each event. *)
Definition closed_route_selected(prog:program)(events:list pc):bool:=
 if straight_event_program prog then false else
 if occurrence_case prog events then false else
 if event_tail_case prog events then false else
 occurrence_segments_acyclic prog events(occurrence_cut prog events).
Definition closed_progress_check(F:lfacts)(prog:program)(events:list pc):bool:=
 match occurrence_checked_word no_loop_summary F prog events with Some _=>true|None=>false end.

(* The progress check of the fixed route: the occurrence word through
   summarized loops, for a kernel that ends, the occurrence word of a
   program with acyclic segments, or the event progress check. *)
Definition fixed_progress_check(F:lfacts)(prog:program)(events:list pc):bool:=
 if ends_route_selected prog events then ends_progress_check F prog events
 else if closed_route_selected prog events then closed_progress_check F prog events
 else event_progress_general F prog events.
Definition fixed_progress_safe(F:lfacts)(prog:program)(events:list pc):Prop:=
 event_progress_safe F prog events \/ (kernel_ends F prog ->occurrence_schedule_safe F prog events).
Theorem fixed_progress_check_sound:forall F prog events,
 fixed_progress_check F prog events=true ->fixed_progress_safe F prog events.
Proof.
 intros F prog events H. unfold fixed_progress_check in H. unfold fixed_progress_safe.
 destruct(ends_route_selected prog events).
 - right. intros Hends. exact(ends_progress_check_sound F prog events H Hends).
 - destruct(closed_route_selected prog events).
   + right. intros _. unfold closed_progress_check in H.
     destruct(occurrence_checked_word no_loop_summary F prog events)as[word|]eqn:Hword;[|discriminate].
     eapply occurrence_checked_word_sound;[|exact Hword]. intros E HE. apply no_loop_summary_ok.
   + left. exact(event_progress_general_sound F prog events H).
Qed.
