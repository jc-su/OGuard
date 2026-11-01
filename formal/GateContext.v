From Coq Require Import List Arith ZArith Bool Lia.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning GateSoundOp GateSoundRefine GateSoundRange GateSoundJoin GateSoundTransfer GateThread GateBarrier GateLoop.
Import ListNotations.
Local Open Scope Z_scope.

(* ContextLoop *)
(* Complete observation prefixes compose with a checked suffix. *)
Lemma observed_prefix_trace : forall F prog E events n t s word last tail,
 occurrence_observe F prog E events n t s=Some(word,last) ->
 occurrence_trace F prog E events tail (n+t)%nat last ->
 occurrence_trace F prog E events (word++tail) t s.
Proof.
 intros F prog E events n. induction n as[|n IH];intros t s word last tail Hobs Htail.
 - cbn[occurrence_observe]in Hobs. injection Hobs as <- <-. exact Htail.
 - cbn[occurrence_observe]in Hobs.
   destruct(phase_step F prog E t s)as[next|]eqn:Hstep;[|discriminate].
   destruct(occurrence_observe F prog E events n (S t)next)as[[found final]|]eqn:Hrest;[|discriminate].
   injection Hobs as <- <-.
   replace(S n+t)%nat with(n+S t)%nat in Htail by lia.
   specialize(IH (S t)next found final tail Hrest Htail).
   destruct(occurrence_member events(pc_of s))eqn:Hmember;cbn.
   + econstructor 2;[exact Hmember|reflexivity|exact Hstep|exact IH].
   + econstructor 3;[exact Hmember|exact Hstep|exact IH].
Qed.

Section ContextLoop.
Variable F:lfacts.
Variable prog:program.
Variable events body_word:list pc.
Variable Q:Type.
Variable proceed:Q->bool.
Variable advance:Q->Q.
Variable rank:Q->nat.
Variable invariant:env->Q->nat->state->Prop.
Hypothesis decreases:forall q,proceed q=true -> (rank(advance q)<rank q)%nat.
Hypothesis body:forall E q t s,invariant E q t s -> proceed q=true ->
 exists n last,occurrence_observe F prog E events n t s=Some(body_word,last) /\
 invariant E (advance q)(n+t)%nat last.
Hypothesis done:forall E q t s,invariant E q t s -> proceed q=false ->
 occurrence_trace F prog E events [] t s.

Fixpoint context_word(fuel:nat)(q:Q):list pc:=
 match fuel with
 |O=>[]
 |S rest=>if proceed q then body_word++context_word rest(advance q)else[]
 end.
Lemma context_word_complete:forall fuel q E t s,
 (rank q<fuel)%nat -> invariant E q t s ->
 occurrence_trace F prog E events (context_word fuel q) t s.
Proof.
 induction fuel as[|fuel IH];intros q E t s Hrank Hinv;[lia|].
 cbn[context_word]. destruct(proceed q)eqn:Hgo.
 - destruct(body E q t s Hinv Hgo)as[n[last[Hobs Hnext]]].
   eapply observed_prefix_trace;[exact Hobs|].
   eapply IH;[pose proof(decreases q Hgo);lia|exact Hnext].
 - eapply done;eassumption.
Qed.

(* A common context fixes ONE word before choosing either thread or oracle. *)
Theorem context_pair_complete:forall q E1 t1 s1 E2 t2 s2,
 invariant E1 q t1 s1 -> invariant E2 q t2 s2 ->
 let word:=context_word(S(rank q))q in
 occurrence_trace F prog E1 events word t1 s1 /\
 occurrence_trace F prog E2 events word t2 s2.
Proof.
 intros q E1 t1 s1 E2 t2 s2 H1 H2. split;apply context_word_complete;try lia;assumption.
Qed.
Lemma context_word_repeats:forall fuel q,exists k,
 context_word fuel q=List.concat(List.repeat body_word k).
Proof.
 induction fuel as[|fuel IH];intros q;cbn[context_word].
 - exists 0%nat;reflexivity.
 - destruct(proceed q).
   + destruct(IH(advance q))as[k Hk]. exists(S k). cbn[List.repeat List.concat]. now rewrite Hk.
   + exists 0%nat;reflexivity.
Qed.
End ContextLoop.

(* ContextRegion *)
Definition context_not_exit(i:instr):bool:=match i with IExit _=>false|_=>true end.
Lemma context_cfg_complete:forall F prog E t s next i,
 context_not_exit i=true -> fetch prog(blk s)(idx s)=Some i ->
 phase_step F prog E t s=Some next ->
 In(pc_of next)(occurrence_cfg_next prog(pc_of s)).
Proof.
 intros F prog E t s next i Hexit Hfetch Hstep.
 pose proof(step_pos F prog E t s next i Hfetch Hstep)as Hpos.
 unfold occurrence_cfg_next,pc_of. cbn[fst snd]. rewrite Hfetch.
 destruct Hpos as[[Hnext[Hbra Hexit2]]|[[g[target[Hi[Hg Hnext]]]]|[g[Hi[Hg Hnext]]]]].
 - rewrite Hnext. destruct i;cbn;auto.
   + destruct g as[[neg p]|];cbn;auto. specialize(Hbra None target eq_refl). discriminate.
   + discriminate.
 - subst i. rewrite Hnext. destruct g;cbn;auto.
 - subst i. discriminate.
Qed.
Definition context_empty:astate:={|amap:=PM.empty fact;acons:=[]|}.
Lemma context_empty_ok:forall F t rs,state_ok F t context_empty rs.
Proof.
 intros. split;[|constructor]. intros r f Hf. unfold get,context_empty in Hf;cbn[amap]in Hf.
 rewrite PM.gempty in Hf. discriminate.
Qed.
Lemma context_step_regs:forall F prog E t s next i,
 env_ok F E -> fetch prog(blk s)(idx s)=Some i ->
 phase_step F prog E t s=Some next ->wf_instr F i=true ->
 regs_ok F E(regs s)->regs_ok F E(regs next).
Proof.
 intros. destruct(transfer_sound F prog E H t s next context_empty i H0 H1
  (context_empty_ok F t(regs s))H3 H2)as[_ Hrs]. exact Hrs.
Qed.

Fixpoint context_region(F:lfacts)(prog:program)(events:list pc)
 (endpoint:pc)(protected:list reg)(nodes:list pc):bool:=
 match nodes with
 |[]=>true
 |pos::tail=>match fetch prog(fst pos)(snd pos)with
 |None=>false
 |Some i=>wf_instr F i && context_not_exit i && negb(occurrence_member events pos) &&
   short_all(fun r=>negb(is_cell F r)&&loop_preserves r i)protected &&
   short_all(fun child=>phase_event endpoint child||occurrence_member tail child)
     (occurrence_cfg_next prog pos) && context_region F prog events endpoint protected tail
 end end.

Theorem context_region_sound:forall nodes F prog E events endpoint protected,
 env_ok F E -> context_region F prog events endpoint protected nodes=true ->
 forall t s,In(pc_of s)nodes -> regs_ok F E(regs s) ->
 exists n last,loop_event_free_run F prog E events n t s last /\
 pc_of last=endpoint /\ regs_ok F E(regs last) /\
 forall r,In r protected ->regs last r=regs s r.
Proof.
 induction nodes as[|pos tail IH];intros F prog E events endpoint protected HE Hcheck t s Hin Hrs;
 [contradiction|]. cbn[context_region]in Hcheck.
 destruct(fetch prog(fst pos)(snd pos))as[i|]eqn:Hfetch;[|discriminate].
 rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[[[Hwf Hexit]Hfree]Hkeep]Hnext]Htail].
 destruct Hin as[Hpos|Hin].
 - subst pos. assert(Hf:fetch prog(blk s)(idx s)=Some i)by exact Hfetch.
   assert(Hex:exists mid,phase_step F prog E t s=Some mid).
   {unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
   destruct Hex as[mid Hstep].
   pose proof(context_cfg_complete F prog E t s mid i Hexit Hf Hstep)as Hchild.
   pose proof(context_step_regs F prog E t s mid i HE Hf Hstep Hwf Hrs)as Hmid.
   rewrite short_all_same,forallb_forall in Hnext,Hkeep. specialize(Hnext _ Hchild).
   apply orb_true_iff in Hnext.
   assert(Hpres:forall r,In r protected ->regs mid r=regs s r).
   {intros r Hr. specialize(Hkeep r Hr). apply andb_true_iff in Hkeep.
    destruct Hkeep as[Hcell Hkeep]. apply negb_true_iff in Hcell.
    eapply loop_step_preserves;eassumption. }
   apply negb_true_iff in Hfree.
   destruct Hnext as[Hend|Hnext].
   + apply phase_event_eq in Hend. exists 1%nat,mid. split.
     * econstructor;[exact Hfree|exact Hstep|constructor].
     * split;[exact Hend|]. split;assumption.
   + apply occurrence_member_in in Hnext.
     destruct(IH F prog E events endpoint protected HE Htail(S t)mid Hnext Hmid)
       as[n[last[Hrun[Hend[Hlast Hsaved]]]]].
     exists(S n),last. split;[econstructor;eassumption|]. split;[exact Hend|].
     split;[exact Hlast|]. intros r Hr. rewrite(Hsaved r Hr). exact(Hpres r Hr).
 - eapply IH;eassumption.
Qed.

Lemma event_free_observed:forall F prog E events n t s last,
 loop_event_free_run F prog E events n t s last ->
 occurrence_observe F prog E events n t s=Some([],last).
Proof.
 intros F prog E events n t s last H. induction H;[reflexivity|].
 cbn[occurrence_observe]. rewrite H0,IHloop_event_free_run,H. reflexivity.
Qed.

(* ContextEvents *)
Definition context_segment F prog events endpoint protected start nodes:=
 (phase_event endpoint start || occurrence_member nodes start) &&
 context_region F prog events endpoint protected nodes.
Lemma context_segment_sound:forall F prog E events endpoint protected start nodes t s,
 env_ok F E -> context_segment F prog events endpoint protected start nodes=true ->
 pc_of s=start -> regs_ok F E(regs s) ->
 exists n last,occurrence_observe F prog E events n t s=Some([],last) /\
 pc_of last=endpoint /\ regs_ok F E(regs last) /\
 forall r,In r protected ->regs last r=regs s r.
Proof.
 intros F prog E events endpoint protected start nodes t s HE Hcheck Hpc Hrs.
 unfold context_segment in Hcheck. apply andb_true_iff in Hcheck. destruct Hcheck as[Hstart Hregion].
 apply orb_true_iff in Hstart. destruct Hstart as[Hend|Hin].
 - apply phase_event_eq in Hend. exists 0%nat,s. split;[reflexivity|].
   split;[congruence|]. split;[exact Hrs|]. intros;reflexivity.
 - apply occurrence_member_in in Hin. rewrite <-Hpc in Hin.
   destruct(context_region_sound nodes F prog E events endpoint protected HE Hregion t s Hin Hrs)
    as[n[last[Hrun[Hend[Hlast Hpres]]]]].
   exists n,last. split;[apply event_free_observed;exact Hrun|]. auto.
Qed.
Lemma observed_append:forall F prog E events n t s a mid m b last,
 occurrence_observe F prog E events n t s=Some(a,mid) ->
 occurrence_observe F prog E events m (n+t)%nat mid=Some(b,last) ->
 occurrence_observe F prog E events (n+m)%nat t s=Some(a++b,last).
Proof.
 intros F prog E events n. induction n as[|n IH];intros t s a mid m b last H1 H2.
 - cbn[occurrence_observe]in H1. injection H1 as <- <-. exact H2.
 - cbn[occurrence_observe]in H1.
   destruct(phase_step F prog E t s)as[next|]eqn:Hstep;[|discriminate].
   destruct(occurrence_observe F prog E events n(S t)next)as[[found final]|]eqn:Hrest;[|discriminate].
   injection H1 as <- <-. cbn[Nat.add occurrence_observe]. rewrite Hstep.
   replace(S n+t)%nat with(n+S t)%nat in H2 by lia.
   rewrite(IH (S t)next found final m b last Hrest H2).
   destruct(occurrence_member events(pc_of s));reflexivity.
Qed.

Fixpoint context_events F prog events protected start finish
 (plans:list(list pc*pc)):bool:=
 match plans with
 |[]=>phase_event finish start
 |(nodes,event)::tail=>
  context_segment F prog events event protected start nodes &&
  occurrence_member events event && loop_fetch_is prog event INop &&
  context_events F prog events protected (next_pos prog(fst event)(snd event))finish tail
 end.
Theorem context_events_sound:forall plans F prog E events protected start finish t s,
 env_ok F E -> context_events F prog events protected start finish plans=true ->
 pc_of s=start -> regs_ok F E(regs s) ->
 exists n last,occurrence_observe F prog E events n t s=Some(map snd plans,last) /\
 pc_of last=finish /\ regs_ok F E(regs last) /\
 forall r,In r protected ->regs last r=regs s r.
Proof.
 induction plans as[|[nodes event]tail IH];intros F prog E events protected start finish t s HE Hcheck Hpc Hrs.
 - cbn[context_events]in Hcheck. apply phase_event_eq in Hcheck.
   exists 0%nat,s. split;[reflexivity|]. split;[congruence|]. split;[exact Hrs|]. intros;reflexivity.
 - cbn[context_events]in Hcheck. rewrite !andb_true_iff in Hcheck.
   destruct Hcheck as[[[Hsegment Hmember]Hmarker]Htail].
   destruct(context_segment_sound F prog E events event protected start nodes t s HE Hsegment Hpc Hrs)
    as[n[mid[Hprefix[Hmidpc[Hmidrs Hmidpres]]]]].
   set(next:=goto(next_pos prog(fst event)(snd event))(regs mid)).
   assert(Hf:fetch prog(blk mid)(idx mid)=Some INop).
   {apply loop_fetch_is_sound in Hmarker. rewrite <-Hmidpc in Hmarker. exact Hmarker. }
   assert(Hstep:phase_step F prog E(n+t)%nat mid=Some next).
   {unfold phase_step,step. rewrite Hf. cbn[guard_of holds]. unfold next. rewrite <-Hmidpc. reflexivity. }
   assert(Hnextpc:pc_of next=next_pos prog(fst event)(snd event)).
   {unfold next,pc_of,goto. destruct(next_pos prog(fst event)(snd event));reflexivity. }
   assert(Hnexteq:regs next=regs mid).
   {unfold next,goto. destruct(next_pos prog(fst event)(snd event));reflexivity. }
   assert(Hnextrs:regs_ok F E(regs next))by(now rewrite Hnexteq).
   destruct(IH F prog E events protected (next_pos prog(fst event)(snd event))finish
    (S(n+t))next HE Htail Hnextpc Hnextrs)
    as[m[last[Hsuffix[Hend[Hlast Hpres]]]]].
   exists(n+S m)%nat,last. split.
   + change(map snd((nodes,event)::tail))with(event::map snd tail).
     replace(event::map snd tail)with([]++(event::map snd tail))by reflexivity.
     eapply observed_append;[exact Hprefix|].
     cbn[occurrence_observe]. rewrite Hstep,Hsuffix,Hmidpc,Hmember. reflexivity.
   + split;[exact Hend|]. split;[exact Hlast|]. intros r Hr.
     rewrite(Hpres r Hr),Hnexteq. exact(Hmidpres r Hr).
Qed.

(* ContextArithmetic *)
Definition context_range_map(bounds:list(reg*(Z*Z))):PM.t fact:=
 fold_right(fun pair m=>PM.add(fst pair)(FForm(range_form 64(fst(snd pair))(snd(snd pair)))0 None)m)(PM.empty fact)bounds.
Definition context_range_state bounds:astate:={|amap:=context_range_map bounds;acons:=[]|}.
Lemma context_range_state_sound:forall F t rs bounds,
 (forall r lo hi,In(r,(lo,hi))bounds ->lo<=rs r<=hi)->
 state_ok F t(context_range_state bounds)rs.
Proof.
 intros F t rs bounds Hbounds. split;[|constructor].
 intros r f Hfind. unfold get,context_range_state in Hfind;cbn[amap]in Hfind.
 induction bounds as[|[q[lo hi]]tail IH];cbn[context_range_map fold_right fst snd]in Hfind.
 - rewrite PM.gempty in Hfind. discriminate.
 - destruct(Pos.eq_dec r q)as[Heq|Hne].
   + subst q. rewrite PM.gss in Hfind. injection Hfind as <-. unfold fact_ok,fact_okv.
     split;[unfold wf,range_form;cbn;lia|]. split.
     * unfold sat,range_form. cbn[fm ft flo fhi tsum fold_right].
       exists(rs r). split;[apply(Hbounds r lo hi);left;reflexivity|].
       replace(0+rs r)with(rs r)by lia. apply eqm_refl.
     * split;[apply aligned_nonpos;lia|discriminate].
   + rewrite PM.gso in Hfind by exact Hne. apply IH;[|exact Hfind]. intros r' lo' hi' Hin. apply(Hbounds r' lo' hi');right;exact Hin.
Qed.

Definition context_increment counter stride:signed_loop_step:=
 {|sl_width:=64;sl_alias_width:=64;sl_counter:=counter;sl_source:=counter;
   sl_operand:=OReg stride;sl_down:=false|}.
Lemma context_increment_sound:forall F prog E events pos st counter stride t s,
 env_ok F E -> signed_loop_step_ok F prog pos st(context_increment counter stride)=true ->
 occurrence_member events pos=false ->pc_of s=pos ->state_ok F t st(regs s) ->regs_ok F E(regs s)->
 0<=regs s counter+regs s stride<2^63 ->
 exists next,occurrence_observe F prog E events 1 t s=Some([],next) /\
 pc_of next=next_pos prog(fst pos)(snd pos) /\ regs_ok F E(regs next) /\
 regs next counter=regs s counter+regs s stride /\
 forall r,r<>counter ->regs next r=regs s r.
Proof.
 intros F prog E events pos st counter stride t s HE Hcheck Hfree Hpc Hst Hrs Hsum.
 destruct(signed_loop_step_sound false F prog E pos st(context_increment counter stride)t s
  HE Hcheck(eq_sym Hpc)Hst Hrs)as[next[Hstep[Hnext[Hpost[Hregs Hinc]]]]].
 assert(Hshape:fetch prog(blk s)(idx s)=Some(signed_loop_instr(context_increment counter stride))).
 {unfold signed_loop_step_ok in Hcheck. rewrite !andb_true_iff in Hcheck.
  assert(Hf:loop_fetch_is prog pos(signed_loop_instr(context_increment counter stride))=true)by tauto.
  apply loop_fetch_is_sound in Hf. rewrite <-Hpc in Hf. exact Hf. }
 assert(Hwidth:wdn F counter=64%nat).
 {unfold signed_loop_step_ok in Hcheck. rewrite !andb_true_iff in Hcheck.
  apply Nat.eqb_eq. change(Nat.eqb(wdn F(sl_counter(context_increment counter stride)))(sl_width(context_increment counter stride))=true). tauto. }
 pose proof(step_regs_dest F prog E HE t s next(signed_loop_instr(context_increment counter stride))counter Hshape eq_refl Hstep)as Hupdate.
 cbn[signed_loop_instr context_increment guard_of holds result binop_eval opv]in Hupdate.
 exists next. split.
 - cbn[occurrence_observe]. rewrite Hstep,Hpc,Hfree. reflexivity.
 - split;[exact Hnext|]. split;[exact Hregs|]. split.
   + rewrite Hupdate. unfold upd. rewrite Pos.eqb_refl,Hwidth. apply trunc_small. cbn;lia.
   + intros r Hr. rewrite Hupdate. unfold upd. assert(Hneq:Pos.eqb r counter=false)by(apply Pos.eqb_neq;exact Hr). now rewrite Hneq.
Qed.
Lemma context_rank_decreases:forall c stride limit,
 0<stride ->c<limit ->
 (Z.to_nat(Z.max 0(limit-(c+stride)))<Z.to_nat(Z.max 0(limit-c)))%nat.
Proof.
 intros. assert(Hrange:0<=Z.max 0(limit-(c+stride))<limit-c).
 {split;[apply Z.le_max_l|]. apply Z.max_lub_lt;lia. }
 rewrite(Z.max_r 0(limit-c))by lia. apply Z2Nat.inj_lt;lia.
Qed.

(* ContextTest *)
Definition context_test_polarity(c:cmp)(neg:bool):bool:=
 match c,neg with CGe,false|CLt,true=>true|_,_=>false end.
Lemma context_test_polarity_sound:forall comparison neg c limit,
 context_test_polarity comparison neg=true ->
 (if neg then Z.eqb(bool_z(cmp_eval comparison c limit))0
  else negb(Z.eqb(bool_z(cmp_eval comparison c limit))0))=negb(Z.ltb c limit).
Proof.
 intros comparison neg c limit H. destruct comparison,neg;try discriminate;cbn[cmp_eval bool_z];
 destruct(Z.ltb_spec c limit);try reflexivity.
 - assert(Hge:(limit<=?c)=false)by(apply Z.leb_gt;lia). now rewrite Hge.
 - assert(Hge:(limit<=?c)=true)by(apply Z.leb_le;lia). now rewrite Hge.
Qed.
Definition reverse_cmp(c:cmp):cmp:=match c with
 |CLt=>CGt|CLe=>CGe|CGt=>CLt|CGe=>CLe|CEq=>CEq|CNe=>CNe end.
Lemma reverse_cmp_exact:forall c a b,cmp_eval c b a=cmp_eval(reverse_cmp c)a b.
Proof. intros c a b;destruct c;cbn;try reflexivity;now rewrite Z.eqb_sym. Qed.
Definition context_comparison(reversed:bool)(c:cmp):cmp:=if reversed then reverse_cmp c else c.
Definition context_compare_instr (reversed:bool) comparison predicate counter limit:=
 ISetp None comparison 64 true predicate
 (if reversed then limit else counter)(OReg(if reversed then counter else limit)).
Lemma context_comparison_exact:forall (rev:bool) c a b,
 cmp_eval c(if rev then b else a)(if rev then a else b)=cmp_eval(context_comparison rev c)a b.
Proof. intros [] c a b;cbn[context_comparison];[apply reverse_cmp_exact|reflexivity]. Qed.
Definition context_test reversed F prog events protected compare_pc branch_pc nodes counter limit predicate comparison neg target:=
 context_test_polarity(context_comparison reversed comparison)neg &&
 Nat.eqb(wdn F predicate)1 &&
 short_all(fun r=>negb(Pos.eqb r predicate))protected &&
 loop_fetch_is prog compare_pc(context_compare_instr reversed comparison predicate counter limit) &&
 wf_instr F(context_compare_instr reversed comparison predicate counter limit) &&
 negb(occurrence_member events compare_pc) &&
 context_segment F prog events branch_pc(predicate::protected)
  (next_pos prog(fst compare_pc)(snd compare_pc))nodes &&
 loop_fetch_is prog branch_pc(IBra(Some(neg,predicate))target) &&
 negb(occurrence_member events branch_pc).
Theorem context_test_sound:forall reversed F prog E events protected compare_pc branch_pc nodes counter limit predicate comparison neg target t s,
 env_ok F E ->
 context_test reversed F prog events protected compare_pc branch_pc nodes counter limit predicate comparison neg target=true ->
 pc_of s=compare_pc ->regs_ok F E(regs s)->
 0<=regs s counter<2^63 ->0<=regs s limit<2^63 ->
 exists n last,occurrence_observe F prog E events n t s=Some([],last) /\
 pc_of last=(if regs s counter <? regs s limit then next_pos prog(fst branch_pc)(snd branch_pc)else(target,0%nat)) /\
 regs_ok F E(regs last) /\ forall r,In r protected ->regs last r=regs s r.
Proof.
 intros reversed F prog E events protected compare_pc branch_pc nodes counter limit predicate comparison neg target t s HE Hcheck Hpc Hrs Hc Hl.
 unfold context_test in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[[[[[[Hpol Hwidth]Hprotect]Hcmp]Hwf]Hfree]Hsegment]Hbranch]Hbranchfree].
 apply Nat.eqb_eq in Hwidth. rewrite short_all_same,forallb_forall in Hprotect.
 apply negb_true_iff in Hfree,Hbranchfree. apply loop_fetch_is_sound in Hcmp,Hbranch.
 set(value:=bool_z(cmp_eval(context_comparison reversed comparison)(regs s counter)(regs s limit))).
 set(mid:=goto(next_pos prog(fst compare_pc)(snd compare_pc))(upd(wdn F)(regs s)predicate value)).
 assert(Hmidpc:pc_of mid=next_pos prog(fst compare_pc)(snd compare_pc)).
 {unfold mid,pc_of,goto. destruct(next_pos prog(fst compare_pc)(snd compare_pc));reflexivity. }
 assert(Hmideq:regs mid=upd(wdn F)(regs s)predicate value).
 {unfold mid,goto. destruct(next_pos prog(fst compare_pc)(snd compare_pc));reflexivity. }
 assert(Hcmpstep:phase_step F prog E t s=Some mid).
 {unfold phase_step,step. rewrite <-Hpc in Hcmp. cbn[pc_of fst snd]in Hcmp. rewrite Hcmp.
  unfold context_compare_instr. cbn[guard_of holds dest result opv extend].
  assert(Hcsmall:sext 64(regs s counter)=regs s counter)by(apply sext_small;cbn;lia).
  assert(Hlsmall:sext 64(regs s limit)=regs s limit)by(apply sext_small;cbn;lia).
  destruct reversed;cbn in *;rewrite ?Hcsmall,?Hlsmall;
  unfold mid,value;rewrite <-Hpc;[rewrite reverse_cmp_exact|];reflexivity. }
 assert(Hmidrs:regs_ok F E(regs mid)).
 {eapply context_step_regs;[exact HE| |exact Hcmpstep|exact Hwf|exact Hrs].
  rewrite <-Hpc in Hcmp. exact Hcmp. }
 assert(Hpred:regs mid predicate=value).
 {rewrite Hmideq. unfold upd. rewrite Pos.eqb_refl,Hwidth.
  unfold value. destruct(cmp_eval(context_comparison reversed comparison)(regs s counter)(regs s limit));reflexivity. }
 assert(Hsaved:forall r,In r protected->regs mid r=regs s r).
 {intros r Hr. rewrite Hmideq. unfold upd. specialize(Hprotect r Hr).
  apply negb_true_iff in Hprotect. now rewrite Hprotect. }
 destruct(context_segment_sound F prog E events branch_pc(predicate::protected)
  (next_pos prog(fst compare_pc)(snd compare_pc))nodes(S t)mid HE Hsegment Hmidpc Hmidrs)
  as[n[atbranch[Hrun[Hbranchpc[Hbranchrs Hbranchsaved]]]]].
 set(last:=goto(if regs s counter <? regs s limit then next_pos prog(fst branch_pc)(snd branch_pc)else(target,0%nat))(regs atbranch)).
 assert(Hlasteq:regs last=regs atbranch).
 {unfold last,goto. destruct(regs s counter <? regs s limit);[destruct(next_pos prog(fst branch_pc)(snd branch_pc))|];reflexivity. }
 assert(Hlastpc:pc_of last=(if regs s counter <? regs s limit then next_pos prog(fst branch_pc)(snd branch_pc)else(target,0%nat))).
 {unfold last,pc_of,goto. destruct(regs s counter <? regs s limit);[destruct(next_pos prog(fst branch_pc)(snd branch_pc))|];reflexivity. }
 assert(Hbranchstep:phase_step F prog E(n+S t)%nat atbranch=Some last).
 {unfold phase_step,step. rewrite <-Hbranchpc in Hbranch.
  cbn[pc_of fst snd]in Hbranch. rewrite Hbranch.
  cbn[guard_of holds]. rewrite(Hbranchsaved predicate(or_introl eq_refl)),Hpred.
  assert(Hguard:xorb neg(negb(Z.eqb value 0))=negb(regs s counter <? regs s limit)).
  {pose proof(context_test_polarity_sound(context_comparison reversed comparison)neg(regs s counter)(regs s limit)Hpol)as Hp.
   fold value in Hp. destruct neg,(Z.eqb value 0);exact Hp. }
  rewrite Hguard. unfold last. rewrite <-Hbranchpc.
  destruct(regs s counter <? regs s limit);reflexivity. }
 assert(Hbranchobs:occurrence_observe F prog E events 1(n+S t)%nat atbranch=Some([],last)).
 {cbn[occurrence_observe]. rewrite Hbranchstep,Hbranchpc,Hbranchfree. reflexivity. }
 pose proof(observed_append F prog E events n(S t)mid []atbranch 1 []last Hrun Hbranchobs)as Hwhole.
 exists(S(n+1))%nat,last. split.
 - cbn[occurrence_observe]. rewrite Hcmpstep,Hwhole,Hpc,Hfree. reflexivity.
 - split;[exact Hlastpc|]. split;[now rewrite Hlasteq|].
   intros r Hr. rewrite Hlasteq,(Hbranchsaved r(or_intror Hr)). exact(Hsaved r Hr).
Qed.



(* ContextPrefix *)
Definition context_prefix fuel keep F prog events start endpoint entry post:bool:=
 match occurrence_collect_with no_loop_summary fuel keep F prog(endpoint::events)
   (List.length prog,0%nat)[(start,Some entry)]with
 |Some(_,arrivals)=>short_all(fun pair=>phase_event endpoint(fst pair)&&state_le F(snd pair)post)arrivals
 |None=>false end.
Theorem context_prefix_sound:forall fuel keep F prog E events start endpoint entry post t s,
 env_ok F E -> context_prefix fuel keep F prog events start endpoint entry post=true ->
 pc_of s=start ->state_ok F t entry(regs s)->regs_ok F E(regs s)->
 exists n last,occurrence_observe F prog E events n t s=Some([],last) /\
 pc_of last=endpoint /\ state_ok F(n+t)%nat post(regs last) /\ regs_ok F E(regs last).
Proof.
 intros fuel keep F prog E events start endpoint entry post t s HE Hcheck Hpc Hst Hrs.
 unfold context_prefix in Hcheck.
 destruct(occurrence_collect_with no_loop_summary fuel keep F prog(endpoint::events)
 (List.length prog,0%nat)[(start,Some entry)])as[[remaining arrivals]|]eqn:Hcollect;[|discriminate].
 rewrite short_all_same,forallb_forall in Hcheck.
 eapply(occurrence_collect_with_continuation F prog E HE(endpoint::events)no_loop_summary
   (no_loop_summary_ok F prog E(endpoint::events))fuel keep(List.length prog,0%nat)
   [(start,Some entry)]remaining arrivals
   (fun u v=>exists n last,occurrence_observe F prog E events n u v=Some([],last) /\
    pc_of last=endpoint /\ state_ok F(n+u)%nat post(regs last) /\ regs_ok F E(regs last)) Hcollect).
 - intros u v next Hfree Hstep[n[last[Hobs[Hend[Hlast Hr]]]]].
   exists(S n),last. split.
   + cbn[occurrence_member existsb]in Hfree. apply orb_false_iff in Hfree.
     destruct Hfree as[_ Hfree]. change(occurrence_member events(pc_of v)=false)in Hfree.
     cbn[occurrence_observe]. rewrite Hstep,Hobs,Hfree. reflexivity.
   + split;[exact Hend|]. split;[|exact Hr].
     replace(S n+u)%nat with(n+S u)%nat by lia. exact Hlast.
 - intros pos st Hin u v Hpos Hv Hrv. specialize(Hcheck(pos,st)Hin).
   apply andb_true_iff in Hcheck. destruct Hcheck as[Hend Hle].
   cbn[fst snd]in Hend,Hle. apply phase_event_eq in Hend.
   exists 0%nat,v. split;[reflexivity|]. split;[congruence|].
   split;[eapply state_le_sound;eassumption|exact Hrv].
 - left;reflexivity.
 - symmetry;exact Hpc.
 - exact Hst.
 - exact Hrs.
Qed.

(* Only unconditional deterministic arithmetic is eligible. Loads, arbitrary
   results and division remain unmarked even with uniform addresses. *)
Definition uniform_pure (i:instr):bool:=
 match i with
 | IMov None _ _ | IImm None _ _ | IBin None _ _ _ _ _
 | IMad None _ _ _ _ _ | IWide None _ _ _ _ _
 | ICvt None _ _ _ _ _ | IMinMax None _ _ _ _ _ _ => true
 | _=>false end.
Definition uniform_value (i:instr)(rs:reg->Z):Z:=
 result (fun _ _=>0)(fun _ _=>0)(fun _ _=>0) 0 rs i.
Definition uniform_value_step(F:lfacts)(i:instr)(rs:reg->Z):reg->Z:=
 if uniform_pure i then match dest i with
 |Some d=>upd(wdn F)rs d(uniform_value i rs)|None=>rs end else rs.
Definition uniform_marks_step(F:lfacts)(i:instr)(marks:reg->bool):reg->bool:=
 let keep r:=marks r && negb(is_cell F r) && loop_preserves r i in
 if uniform_pure i && forallb marks(instruction_inputs i) then
 match dest i with
 |Some d=>fun r=>if Pos.eqb r d then negb(is_cell F d) else keep r
 |None=>keep end else keep.
Definition uniform_agree(marks:reg->bool)(rs vs:reg->Z):Prop:=
 forall r,marks r=true ->rs r=vs r.

Lemma uniform_pure_guard:forall i,uniform_pure i=true ->guard_of i=None.
Proof. intros i. destruct i;cbn;try discriminate;destruct g;cbn;congruence. Qed.
Lemma uniform_pure_dest:forall i,uniform_pure i=true ->exists d,dest i=Some d.
Proof. intros i. destruct i;cbn;try discriminate;destruct g;cbn;try discriminate;eauto. Qed.
Lemma uniform_pure_writes:forall i d,uniform_pure i=true ->dest i=Some d ->loop_writes i=[d].
Proof. intros i d Hp Hd. destruct i;cbn in *;try discriminate;injection Hd as <-;reflexivity. Qed.
Lemma uniform_opnd_ext:forall a x y,
 (forall r,In r(opnd_inputs a)->x r=y r)->opv x a=opv y a.
Proof. intros [r|z] x y H;cbn;[apply H;cbn;auto|reflexivity]. Qed.
Lemma uniform_result_ext:forall i x y mem arb shm t,
 uniform_pure i=true ->
 (forall r,In r(instruction_inputs i)->x r=y r)->
 result mem arb shm t x i=uniform_value i y.
Proof.
 intros i x y mem arb shm t Hp Hxy.
 destruct i;cbn[uniform_pure]in Hp;try discriminate;
 destruct g;try discriminate;cbn[instruction_inputs guard_of]in Hxy;
 unfold uniform_value;cbn[result].
 - apply Hxy;cbn;auto.
 - reflexivity.
 - rewrite(Hxy a)by(cbn;auto).
   rewrite(uniform_opnd_ext b x y)by(intros;apply Hxy;cbn;auto). reflexivity.
 - rewrite(Hxy a)by(cbn;auto).
   rewrite(uniform_opnd_ext b x y)by(intros;apply Hxy;right;apply in_or_app;auto).
   rewrite(uniform_opnd_ext c x y)by(intros;apply Hxy;right;apply in_or_app;auto). reflexivity.
 - rewrite(Hxy a)by(cbn;auto).
   rewrite(uniform_opnd_ext b x y)by(intros;apply Hxy;right;apply in_or_app;auto).
   destruct c as[r|];[rewrite(Hxy r)by(right;apply in_or_app;right;cbn;auto)|];reflexivity.
 - rewrite(Hxy s)by(cbn;auto). reflexivity.
 - rewrite(Hxy a)by(cbn;auto).
   rewrite(uniform_opnd_ext b x y)by(intros;apply Hxy;cbn;auto). reflexivity.
Qed.

Lemma uniform_value_keep:forall F i rs r,
 loop_preserves r i=true ->uniform_value_step F i rs r=rs r.
Proof.
 intros F i rs r H. unfold uniform_value_step.
 destruct(uniform_pure i)eqn:Hp;[|reflexivity].
 destruct(uniform_pure_dest i Hp)as[d Hd]. rewrite Hd.
 unfold loop_preserves in H. rewrite(uniform_pure_writes i d Hp Hd)in H.
 cbn[existsb]in H. rewrite orb_false_r in H. apply negb_true_iff in H.
 unfold upd. rewrite H. reflexivity.
Qed.
Lemma uniform_value_step_agree:forall F i marks x y,
 uniform_agree marks x y ->
 uniform_agree(uniform_marks_step F i marks)
   (uniform_value_step F i x)(uniform_value_step F i y).
Proof.
 intros F i marks x y H r Hr. unfold uniform_marks_step in Hr.
 destruct(uniform_pure i && forallb marks(instruction_inputs i))eqn:Hchecked.
 - apply andb_true_iff in Hchecked. destruct Hchecked as[Hp Hall].
   destruct(uniform_pure_dest i Hp)as[d Hd]. rewrite Hd in Hr.
   destruct(Pos.eqb_spec r d)as[->|Hneq].
   + unfold uniform_value_step. rewrite Hp,Hd. unfold upd. rewrite Pos.eqb_refl.
     f_equal. unfold uniform_value at 1. apply uniform_result_ext;[exact Hp|].
     intros q Hq. apply H. apply forallb_forall with(x:=q)in Hall;assumption.
   + rewrite !andb_true_iff in Hr. destruct Hr as[[Hm Hcell]Hkeep].
     rewrite !uniform_value_keep by exact Hkeep. apply H;exact Hm.
 - rewrite !andb_true_iff in Hr. destruct Hr as[[Hm Hcell]Hkeep].
   rewrite !uniform_value_keep by exact Hkeep. apply H;exact Hm.
Qed.

Lemma uniform_step_agree:forall F prog E t s s' i marks values,
 env_ok F E ->fetch prog(blk s)(idx s)=Some i ->phase_step F prog E t s=Some s' ->
 uniform_agree marks(regs s)values ->
 uniform_agree(uniform_marks_step F i marks)(regs s')(uniform_value_step F i values).
Proof.
 intros F prog E t s s' i marks values HE Hfetch Hstep Hagree r Hr.
 unfold uniform_marks_step in Hr.
 destruct(uniform_pure i && forallb marks(instruction_inputs i))eqn:Hchecked.
 - apply andb_true_iff in Hchecked. destruct Hchecked as[Hp Hall].
   destruct(uniform_pure_dest i Hp)as[d Hd]. rewrite Hd in Hr.
   destruct(Pos.eqb_spec r d)as[->|Hneq].
   + pose proof(step_regs_dest F prog E HE t s s' i d Hfetch Hd Hstep)as Hregs.
     rewrite(uniform_pure_guard i Hp)in Hregs. cbn[holds]in Hregs.
     rewrite Hregs. unfold uniform_value_step. rewrite Hp,Hd.
     unfold upd. rewrite Pos.eqb_refl. f_equal.
     apply uniform_result_ext;[exact Hp|]. intros q Hq. apply Hagree.
     apply forallb_forall with(x:=q)in Hall;assumption.
   + rewrite !andb_true_iff in Hr. destruct Hr as[[Hm Hcell]Hkeep].
     apply negb_true_iff in Hcell.
     rewrite(loop_step_preserves F prog E t s s' i r Hcell Hfetch Hstep Hkeep).
     rewrite uniform_value_keep by exact Hkeep. apply Hagree;exact Hm.
 - rewrite !andb_true_iff in Hr. destruct Hr as[[Hm Hcell]Hkeep].
   apply negb_true_iff in Hcell.
   rewrite(loop_step_preserves F prog E t s s' i r Hcell Hfetch Hstep Hkeep).
   rewrite uniform_value_keep by exact Hkeep. apply Hagree;exact Hm.
Qed.

Fixpoint uniform_marks_run(F:lfacts)(code:list instr)(marks:reg->bool):reg->bool:=
 match code with []=>marks|i::rest=>uniform_marks_run F rest(uniform_marks_step F i marks)end.
Fixpoint uniform_values_run(F:lfacts)(code:list instr)(values:reg->Z):reg->Z:=
 match code with []=>values|i::rest=>uniform_values_run F rest(uniform_value_step F i values)end.
Lemma uniform_values_run_agree:forall code F marks x y,
 uniform_agree marks x y ->
 uniform_agree(uniform_marks_run F code marks)
 (uniform_values_run F code x)(uniform_values_run F code y).
Proof.
 induction code as[|i rest IH];intros F marks x y H;cbn;[exact H|].
 apply IH. apply uniform_value_step_agree. exact H.
Qed.

(* The collected prefix is checked against actual fetches, the full event
   alphabet, and fixed successors. Conditional branches are refused. *)
Fixpoint uniform_prefix_collect(fuel:nat)(F:lfacts)(prog:program)
 (events:list pc)(endpoint pos:pc):option(list instr):=
 if phase_event endpoint pos then Some [] else
 match fuel with
 |O=>None
 |S rest=>match fetch prog(fst pos)(snd pos)with
 |None=>None
 |Some i=>if wf_instr F i && negb(occurrence_member events pos)then
   match loop_next prog pos i with
   |None=>None
   |Some next=>match uniform_prefix_collect rest F prog events endpoint next with
    |Some code=>Some(i::code)|None=>None end
   end else None end end.

Theorem uniform_prefix_sound:forall fuel F prog E events endpoint pos code marks values t s,
 env_ok F E ->uniform_prefix_collect fuel F prog events endpoint pos=Some code ->
 pos=pc_of s ->regs_ok F E(regs s) ->uniform_agree marks(regs s)values ->
 exists last,
 occurrence_observe F prog E events(length code)t s=Some([],last) /\
 pc_of last=endpoint /\ regs_ok F E(regs last) /\
 uniform_agree(uniform_marks_run F code marks)(regs last)(uniform_values_run F code values).
Proof.
 induction fuel as[|fuel IH];intros F prog E events endpoint pos code marks values t s HE Hcollect Hpos Hrs Hagree;
 cbn[uniform_prefix_collect]in Hcollect;
 destruct(phase_event endpoint pos)eqn:Hend.
 - apply phase_event_eq in Hend. injection Hcollect as <-. exists s. cbn.
   split;[reflexivity|]. split;[congruence|]. split;assumption.
 - discriminate.
 - apply phase_event_eq in Hend. injection Hcollect as <-. exists s. cbn.
   split;[reflexivity|]. split;[congruence|]. split;assumption.
 - destruct(fetch prog(fst pos)(snd pos))as[i|]eqn:Hfetch;[|discriminate].
   destruct(wf_instr F i && negb(occurrence_member events pos))eqn:Hcheck;[|discriminate].
   destruct(loop_next prog pos i)as[next|]eqn:Hnext;[|discriminate].
   destruct(uniform_prefix_collect fuel F prog events endpoint next)as[tail|]eqn:Htail;[|discriminate].
   injection Hcollect as <-. apply andb_true_iff in Hcheck. destruct Hcheck as[Hwf Hfree].
   apply negb_true_iff in Hfree.
   assert(Hf:fetch prog(blk s)(idx s)=Some i)by(rewrite Hpos in Hfetch;exact Hfetch).
   assert(Hex:exists mid,phase_step F prog E t s=Some mid).
   {unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
   destruct Hex as[mid Hstep].
   assert(Hpc:pc_of mid=next)by(eapply loop_next_sound;[exact Hf|exact Hstep|rewrite <-Hpos;exact Hnext]).
   pose proof(context_step_regs F prog E t s mid i HE Hf Hstep Hwf Hrs)as Hmid.
   pose proof(uniform_step_agree F prog E t s mid i marks values HE Hf Hstep Hagree)as Hma.
   destruct(IH F prog E events endpoint next tail
    (uniform_marks_step F i marks)(uniform_value_step F i values)(S t)mid
    HE Htail(eq_sym Hpc)Hmid Hma)as[last[Hobs[Hend2[Hlast Hlastagree]]]].
   exists last. cbn[uniform_marks_run uniform_values_run length].
   split;[|auto]. cbn[occurrence_observe]. rewrite Hstep,Hobs,<-Hpos,Hfree. reflexivity.
Qed.

Theorem uniform_prefix_pair:forall fuel F prog E1 E2 events endpoint pos code marks t1 s1 t2 s2,
 env_ok F E1 ->env_ok F E2 ->
 uniform_prefix_collect fuel F prog events endpoint pos=Some code ->
 pos=pc_of s1 ->pos=pc_of s2 ->regs_ok F E1(regs s1)->regs_ok F E2(regs s2)->
 uniform_agree marks(regs s1)(regs s2) ->
 exists last1 last2,
 occurrence_observe F prog E1 events(length code)t1 s1=Some([],last1) /\
 occurrence_observe F prog E2 events(length code)t2 s2=Some([],last2) /\
 pc_of last1=endpoint /\pc_of last2=endpoint /\
 uniform_agree(uniform_marks_run F code marks)(regs last1)(regs last2).
Proof.
 intros fuel F prog E1 E2 events endpoint pos code marks t1 s1 t2 s2 HE1 HE2 Hcode Hpos1 Hpos2 Hrs1 Hrs2 Hagree.
 assert(Hid:forall marks values,uniform_agree marks values values)by(intros m v r Hr;reflexivity).
 destruct(uniform_prefix_sound fuel F prog E1 events endpoint pos code marks(regs s1)t1 s1
 HE1 Hcode Hpos1 Hrs1(Hid marks(regs s1)))as[l1[Hobs1[Hend1[Hr1 Ha1]]]].
 destruct(uniform_prefix_sound fuel F prog E2 events endpoint pos code marks(regs s2)t2 s2
 HE2 Hcode Hpos2 Hrs2(Hid marks(regs s2)))as[l2[Hobs2[Hend2[Hr2 Ha2]]]].
 exists l1,l2. repeat split;try assumption.
 intros r Hr. rewrite(Ha1 r Hr),(Ha2 r Hr).
 exact(uniform_values_run_agree code F marks(regs s1)(regs s2)Hagree r Hr).
Qed.


Fixpoint uniform_states_run(F:lfacts)(code:list instr)(st:astate):astate:=
 match code with []=>st|i::rest=>uniform_states_run F rest(transfer F st i)end.
Theorem uniform_prefix_state_sound:forall fuel F prog E events endpoint pos code st t s,
 env_ok F E ->uniform_prefix_collect fuel F prog events endpoint pos=Some code ->
 pos=pc_of s ->regs_ok F E(regs s) ->state_ok F t st(regs s) ->
 exists last,
 occurrence_observe F prog E events(length code)t s=Some([],last) /\
 pc_of last=endpoint /\ regs_ok F E(regs last) /\
 state_ok F(length code+t)%nat(uniform_states_run F code st)(regs last).
Proof.
 induction fuel as[|fuel IH];intros F prog E events endpoint pos code st t s HE Hcollect Hpos Hrs Hst;
 cbn[uniform_prefix_collect]in Hcollect;
 destruct(phase_event endpoint pos)eqn:Hend.
 - apply phase_event_eq in Hend. injection Hcollect as <-. exists s. cbn.
   split;[reflexivity|]. split;[congruence|]. split;assumption.
 - discriminate.
 - apply phase_event_eq in Hend. injection Hcollect as <-. exists s. cbn.
   split;[reflexivity|]. split;[congruence|]. split;assumption.
 - destruct(fetch prog(fst pos)(snd pos))as[i|]eqn:Hfetch;[|discriminate].
   destruct(wf_instr F i && negb(occurrence_member events pos))eqn:Hcheck;[|discriminate].
   destruct(loop_next prog pos i)as[next|]eqn:Hnext;[|discriminate].
   destruct(uniform_prefix_collect fuel F prog events endpoint next)as[tail|]eqn:Htail;[|discriminate].
   injection Hcollect as <-. apply andb_true_iff in Hcheck. destruct Hcheck as[Hwf Hfree].
   apply negb_true_iff in Hfree.
   assert(Hf:fetch prog(blk s)(idx s)=Some i)by(rewrite Hpos in Hfetch;exact Hfetch).
   assert(Hex:exists mid,phase_step F prog E t s=Some mid).
   {unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
   destruct Hex as[mid Hstep].
   assert(Hpc:pc_of mid=next)by(eapply loop_next_sound;[exact Hf|exact Hstep|rewrite <-Hpos;exact Hnext]).
   destruct(transfer_sound F prog E HE t s mid st i Hf Hstep Hst Hrs Hwf)as[Hmidstate Hmidregs].
   destruct(IH F prog E events endpoint next tail(transfer F st i)(S t)mid
    HE Htail(eq_sym Hpc)Hmidregs Hmidstate)as[last[Hobs[Hend2[Hlast Hlaststate]]]].
   exists last. cbn[uniform_states_run length]. split.
   + cbn[occurrence_observe]. rewrite Hstep,Hobs,<-Hpos,Hfree. reflexivity.
   + split;[exact Hend2|]. split;[exact Hlast|].
     replace(S(length tail)+t)%nat with(length tail+S t)%nat by lia. exact Hlaststate.
Qed.

Lemma uniform_values_preserves:forall code F r values,
 forallb(loop_preserves r)code=true ->uniform_values_run F code values r=values r.
Proof.
 induction code as[|i code IH];intros F r values H;cbn in *;[reflexivity|].
 apply andb_true_iff in H. destruct H as[Hi Hcode]. rewrite IH by exact Hcode.
 apply uniform_value_keep. exact Hi.
Qed.
Lemma uniform_prefix_preserves:forall fuel F prog E events endpoint pos code r t s last,
 is_cell F r=false ->uniform_prefix_collect fuel F prog events endpoint pos=Some code ->
 forallb(loop_preserves r)code=true ->pos=pc_of s ->
 occurrence_observe F prog E events(length code)t s=Some([],last)->regs last r=regs s r.
Proof.
 induction fuel as[|fuel IH];intros F prog E events endpoint pos code r t s last Hcell Hcollect Hkeep Hpos Hobs;
 cbn[uniform_prefix_collect]in Hcollect;
 destruct(phase_event endpoint pos)eqn:Hend.
 - injection Hcollect as <-. cbn in Hobs. injection Hobs as <-. reflexivity.
 - discriminate.
 - injection Hcollect as <-. cbn in Hobs. injection Hobs as <-. reflexivity.
 - destruct(fetch prog(fst pos)(snd pos))as[i|]eqn:Hfetch;[|discriminate].
   destruct(wf_instr F i && negb(occurrence_member events pos))eqn:Hcheck;[|discriminate].
   destruct(loop_next prog pos i)as[next|]eqn:Hnext;[|discriminate].
   destruct(uniform_prefix_collect fuel F prog events endpoint next)as[tail|]eqn:Htail;[|discriminate].
   injection Hcollect as <-. cbn[forallb]in Hkeep. apply andb_true_iff in Hkeep. destruct Hkeep as[Hi Hkeep].
   cbn[length occurrence_observe]in Hobs.
   destruct(phase_step F prog E t s)as[mid|]eqn:Hstep;[|discriminate].
   apply andb_true_iff in Hcheck. destruct Hcheck as[_ Hfree]. apply negb_true_iff in Hfree.
   rewrite <-Hpos,Hfree in Hobs.
   destruct(occurrence_observe F prog E events(length tail)(S t)mid)as[[found final]|]eqn:Hrest;[|discriminate].
   injection Hobs as Hfound Hfinal. subst found final.
   assert(Hf:fetch prog(blk s)(idx s)=Some i)by(rewrite Hpos in Hfetch;exact Hfetch).
   assert(Hpc:pc_of mid=next)by(eapply loop_next_sound;[exact Hf|exact Hstep|rewrite <-Hpos;exact Hnext]).
   rewrite(IH F prog E events endpoint next tail r(S t)mid last Hcell Htail Hkeep(eq_sym Hpc)Hrest).
   eapply loop_step_preserves;eassumption.
Qed.
Definition uniform_prefix_check fuel F prog events endpoint pos code:=
 match uniform_prefix_collect fuel F prog events endpoint pos with
 |Some actual=>if list_eq_dec loop_instr_eq_dec actual code then true else false
 |None=>false end.
Lemma uniform_prefix_check_sound:forall fuel F prog events endpoint pos code,
 uniform_prefix_check fuel F prog events endpoint pos code=true ->
 uniform_prefix_collect fuel F prog events endpoint pos=Some code.
Proof.
 intros. unfold uniform_prefix_check in H.
 destruct(uniform_prefix_collect fuel F prog events endpoint pos)as[actual|];[|discriminate].
 destruct(list_eq_dec loop_instr_eq_dec actual code);[congruence|discriminate].
Qed.
Theorem uniform_prefix_full_sound:forall fuel F prog E events endpoint pos code st marks values t s,
 env_ok F E ->uniform_prefix_collect fuel F prog events endpoint pos=Some code ->
 pos=pc_of s ->regs_ok F E(regs s) ->state_ok F t st(regs s) ->
 uniform_agree marks(regs s)values ->
 exists last,
 occurrence_observe F prog E events(length code)t s=Some([],last) /\
 pc_of last=endpoint /\ regs_ok F E(regs last) /\
 state_ok F(length code+t)%nat(uniform_states_run F code st)(regs last) /\
 uniform_agree(uniform_marks_run F code marks)(regs last)(uniform_values_run F code values).
Proof.
 intros fuel F prog E events endpoint pos code st marks values t s HE Hcode Hpos Hregs Hstate Hagree.
 destruct(uniform_prefix_state_sound fuel F prog E events endpoint pos code st t s HE Hcode Hpos Hregs Hstate)
 as[last[Hobs[Hpc[Hrs Hst]]]].
 destruct(uniform_prefix_sound fuel F prog E events endpoint pos code marks values t s HE Hcode Hpos Hregs Hagree)
 as[other[Hobs2[Hpc2[Hrs2 Ha]]]].
 rewrite Hobs in Hobs2. injection Hobs2 as <-.
 exists last. split;[exact Hobs|]. split;[exact Hpc|]. split;[exact Hrs|]. split;assumption.
Qed.

(* ContextAgreement *)
Definition context_initial_agree (uniform:reg->bool) (E1 E2:env):Prop:=
 forall r,uniform r=true ->e_init E1 r=e_init E2 r.
Definition context_form_ok F uniform f:=
 Z.eqb(fm f)64 && Z.eqb(flo f)(fhi f) &&
 short_all(fun term=>is_fixed F(fst(fst term))&&uniform(fst(fst term)))(ft f).
Definition context_form_value F E f:=trunc 64(tsum(wdn F)(e_init E)(ft f)+flo f).
Lemma context_terms_agree:forall F ts rs1 rs2,
 (forall r e k,In((r,e),k)ts ->rs1 r=rs2 r)->
 tsum(wdn F)rs1 ts=tsum(wdn F)rs2 ts.
Proof.
 intros F ts. induction ts as[|[[r e]k]tail IH];intros rs1 rs2 Hagree;[reflexivity|].
 cbn[tsum fold_right fst snd].
 assert(Hr:rs1 r=rs2 r)by(apply(Hagree r e k);left;reflexivity).
 assert(Htail:tsum(wdn F)rs1 tail=tsum(wdn F)rs2 tail).
 {apply IH. intros r' e' k' Hin. apply(Hagree r' e' k');right;exact Hin. }
 change(k*atom_val(wdn F)rs1(r,e)+tsum(wdn F)rs1 tail=
        k*atom_val(wdn F)rs2(r,e)+tsum(wdn F)rs2 tail).
 rewrite Htail. destruct e;cbn[atom_val];rewrite Hr;reflexivity.
Qed.
Lemma context_form_agree:forall F uniform E1 E2 f,
 context_form_ok F uniform f=true ->context_initial_agree uniform E1 E2 ->
 context_form_value F E1 f=context_form_value F E2 f.
Proof.
 intros F uniform E1 E2 f Hform Hagree. unfold context_form_ok in Hform.
 rewrite !andb_true_iff in Hform. destruct Hform as[[_ _]Hterms].
 rewrite short_all_same,forallb_forall in Hterms. unfold context_form_value.
 f_equal. f_equal. apply context_terms_agree. intros r e k Hin.
 specialize(Hterms((r,e),k)Hin). apply andb_true_iff in Hterms.
 apply Hagree. exact(proj2 Hterms).
Qed.
Lemma context_form_exact:forall F uniform E st t rs r f al vr,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->
 context_form_ok F uniform f=true ->get st r=Some(FForm f al vr)->wdn F r=64%nat ->
 rs r=context_form_value F E f.
Proof.
 intros F uniform E st t rs r f al vr HE Hrs Hst Hform Hget Hw.
 unfold context_form_ok in Hform. rewrite !andb_true_iff in Hform.
 destruct Hform as[[Hfm Hsame]Hterms]. apply Z.eqb_eq in Hfm,Hsame.
 rewrite short_all_same,forallb_forall in Hterms.
 pose proof(proj1 Hst r(FForm f al vr)Hget)as Hfact.
 unfold fact_ok,fact_okv in Hfact. destruct Hfact as[_[[offset[Hbounds Heqm]]_]].
 assert(Hoff:offset=flo f)by lia. subst offset. rewrite Hfm in Heqm.
 assert(Hsum:tsum(wdn F)rs(ft f)=tsum(wdn F)(e_init E)(ft f)).
 {apply context_terms_agree. intros r' e k Hin. specialize(Hterms((r',e),k)Hin).
  apply andb_true_iff in Hterms. apply(proj2 Hrs). exact(proj1 Hterms). }
 rewrite Hsum in Heqm. unfold context_form_value.
 pose proof(rs_range F E rs Hrs r)as Hrange. rewrite Hw in Hrange.
 eapply(GateSoundRange.eqm_small_eq 64 64);[lia|exact Hrange|apply trunc_range|].
 eapply eqm_trans;[exact Heqm|]. apply eqm_sym,trunc_eqm.
Qed.

(* ContextValues *)
Definition context_unsigned_range F st r lo hi:=
 Nat.eqb(wdn F r)64 &&
 match cmp_range F st(OReg r)64 true with
 |Some(a,b)=>(0<=?a)&&(b<?2^63)&&(lo<=?a)&&(b<=?hi)
 |None=>false end.
Lemma context_unsigned_range_sound:forall F E st t rs r lo hi,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->
 context_unsigned_range F st r lo hi=true ->lo<=rs r<=hi.
Proof.
 intros F E st t rs r lo hi HE Hrs Hst Hcheck.
 unfold context_unsigned_range in Hcheck. apply andb_true_iff in Hcheck.
 destruct Hcheck as[Hw Hbounds]. apply Nat.eqb_eq in Hw.
 destruct(cmp_range F st(OReg r)64 true)as[[a b]|]eqn:Hrange;[|discriminate].
 rewrite !andb_true_iff,!Z.leb_le,!Z.ltb_lt in Hbounds.
 assert(Hwidth:owidth F(OReg r)64).
 {intros r' Hr'. injection Hr' as <-. unfold wd. now rewrite Hw. }
 pose proof(cmp_range_sound F E HE t rs Hrs st(OReg r)64 true a b Hst ltac:(lia)Hwidth Hrange)as Hv.
 unfold cval,extend in Hv. cbn[opv]in Hv.
 pose proof(rs_range F E rs Hrs r)as Hraw. rewrite Hw in Hraw.
 change(a<=sext 64(rs r)<=b)in Hv. unfold sext in Hv.
 rewrite GateSoundRange.trunc_small in Hv by exact Hraw.
 destruct(Z.ltb_spec(rs r)(2^(Z.of_nat 64-1)));cbn in *;lia.
Qed.

(* A value claim is independently checked against the actual abstract state. *)
Definition context_value_check F uniform st r f:=context_form_ok F uniform f &&
 Nat.eqb(wdn F r)64 &&register_fact_le F st r(FForm f 0 None).
Lemma context_value_sound:forall F uniform E st t rs r f,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->
 context_value_check F uniform st r f=true ->rs r=context_form_value F E f.
Proof.
 intros F uniform E st t rs r f HE Hrs Hst Hcheck.
 unfold context_value_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[Hform Hw]Hle]. apply Nat.eqb_eq in Hw.
 pose proof(register_fact_le_sound F E HE t rs Hrs st r(FForm f 0 None)Hst Hle)as Hfact.
 set(one:={|amap:=PM.add r(FForm f 0 None)(PM.empty fact);acons:=[]|}).
 assert(Hone:state_ok F t one rs).
 {split;[|constructor]. intros r' f' Hget. unfold get,one in Hget;cbn[amap]in Hget.
  destruct(Pos.eq_dec r' r)as[->|Hne].
  - rewrite PM.gss in Hget. injection Hget as <-. exact Hfact.
  - rewrite PM.gso,PM.gempty in Hget by exact Hne. discriminate. }
 eapply(context_form_exact F uniform E one t rs r f 0 None);try eassumption.
 unfold get,one;cbn[amap]. apply PM.gss.
Qed.

Lemma context_form_from_value:forall F uniform E t rs r f,
 regs_ok F E rs ->context_form_ok F uniform f=true ->rs r=context_form_value F E f ->
 fact_ok F t rs r(FForm f 0 None).
Proof.
 intros F uniform E t rs r f Hrs Hform Hvalue.
 unfold context_form_ok in Hform. rewrite !andb_true_iff in Hform.
 destruct Hform as[[Hfm Hsame]Hterms]. apply Z.eqb_eq in Hfm,Hsame.
 rewrite short_all_same,forallb_forall in Hterms.
 assert(Hsum:tsum(wdn F)rs(ft f)=tsum(wdn F)(e_init E)(ft f)).
 {apply context_terms_agree. intros r' e k Hin. specialize(Hterms((r',e),k)Hin).
  apply andb_true_iff in Hterms. apply(proj2 Hrs). exact(proj1 Hterms). }
 unfold fact_ok,fact_okv. split;[unfold wf;rewrite Hfm;lia|]. split.
 - unfold sat. exists(flo f). split;[lia|]. rewrite Hfm,Hsum,Hvalue.
   unfold context_form_value. apply trunc_eqm.
 - split;[apply aligned_nonpos;lia|discriminate].
Qed.
Definition context_exact_map(bindings:list(reg*form)):=
 fold_right(fun pair m=>PM.add(fst pair)(FForm(snd pair)0 None)m)(PM.empty fact)bindings.
Definition context_exact_state bindings:astate:={|amap:=context_exact_map bindings;acons:=[]|}.
Lemma context_exact_state_sound:forall F uniform E t rs bindings,
 regs_ok F E rs ->
 (forall r f,In(r,f)bindings ->context_form_ok F uniform f=true /\rs r=context_form_value F E f)->
 state_ok F t(context_exact_state bindings)rs.
Proof.
 intros F uniform E t rs bindings Hrs Hvalues. split;[|constructor]. intros r f Hfind.
 unfold get,context_exact_state in Hfind;cbn[amap]in Hfind.
 induction bindings as[|[q form]tail IH];cbn[context_exact_map fold_right fst snd]in Hfind.
 - rewrite PM.gempty in Hfind. discriminate.
 - destruct(Pos.eq_dec r q)as[->|Hne].
   + rewrite PM.gss in Hfind. injection Hfind as <-.
     destruct(Hvalues q form(or_introl eq_refl))as[Hform Hvalue].
     eapply context_form_from_value;eassumption.
   + rewrite PM.gso in Hfind by exact Hne. apply IH;[|exact Hfind].
     intros r' f' Hin. apply(Hvalues r' f');right;exact Hin.
Qed.

(* ContextController *)
Lemma context_increment_segment_sound:forall F prog E events pos target nodes protected st counter stride t s,
 env_ok F E -> signed_loop_step_ok F prog pos st(context_increment counter stride)=true ->
 occurrence_member events pos=false ->
 context_segment F prog events target protected(next_pos prog(fst pos)(snd pos))nodes=true ->
 In counter protected ->pc_of s=pos ->state_ok F t st(regs s) ->regs_ok F E(regs s)->
 0<=regs s counter+regs s stride<2^63 ->
 exists n last,occurrence_observe F prog E events n t s=Some([],last) /\
 pc_of last=target /\regs_ok F E(regs last) /\
 regs last counter=regs s counter+regs s stride /\
 forall r,In r protected ->r<>counter ->regs last r=regs s r.
Proof.
 intros F prog E events pos target nodes protected st counter stride t s HE Hstep Hfree Hgap Hcounter Hpc Hst Hrs Hsum.
 destruct(context_increment_sound F prog E events pos st counter stride t s HE Hstep Hfree Hpc Hst Hrs Hsum)
 as[mid[Hobs[Hmidpc[Hmidrs[Hinc Hsaved]]]]].
 destruct(context_segment_sound F prog E events target protected(next_pos prog(fst pos)(snd pos))nodes
 (1+t)%nat mid HE Hgap Hmidpc Hmidrs)as[n[last[Htail[Hend[Hlast Hkeep]]]]].
 exists(1+n)%nat,last. split.
 - exact(observed_append F prog E events 1 t s []mid n []last Hobs Htail).
 - split;[exact Hend|]. split;[exact Hlast|]. split.
   + rewrite(Hkeep counter Hcounter). exact Hinc.
   + intros r Hr Hne. rewrite(Hkeep r Hr). apply Hsaved;exact Hne.
Qed.
Record context_controller:= {
 ct_counter:reg;ct_stride:reg;ct_limit:reg;ct_predicate:reg;
 ct_head:pc;ct_body_end:pc;ct_increment:pc;ct_test:pc;ct_branch:pc;ct_exit:nat;
 ct_body:list(list pc*pc);ct_tail:list pc;ct_increment_gap:list pc;ct_test_gap:list pc;
 ct_limit_max:Z;ct_stride_max:Z;ct_cut:pc
}.
Definition controller_protected d:=[ct_counter d;ct_stride d;ct_limit d].
Definition controller_bounds d:=[(ct_counter d,(0,ct_limit_max d-1));
 (ct_stride d,(1,ct_stride_max d));(ct_limit d,(0,ct_limit_max d))].
Definition controller_check F prog events d:=
 (0<=?ct_limit_max d)&&(1<=?ct_stride_max d)&&
 (ct_limit_max d+ct_stride_max d <? 2^63)&&
 negb(Pos.eqb(ct_stride d)(ct_counter d))&&negb(Pos.eqb(ct_limit d)(ct_counter d))&&
 context_events F prog events(controller_protected d)(ct_head d)(ct_body_end d)(ct_body d)&&
 context_segment F prog events(ct_increment d)(controller_protected d)(ct_body_end d)(ct_tail d)&&
 signed_loop_step_ok F prog(ct_increment d)(context_range_state(controller_bounds d))
  (context_increment(ct_counter d)(ct_stride d))&&
 negb(occurrence_member events(ct_increment d))&&
 context_segment F prog events(ct_test d)(controller_protected d)
  (next_pos prog(fst(ct_increment d))(snd(ct_increment d)))(ct_increment_gap d)&&
 context_test false F prog events(controller_protected d)(ct_test d)(ct_branch d)(ct_test_gap d)
  (ct_counter d)(ct_limit d)(ct_predicate d)CLt true(ct_exit d)&&
 phase_event(ct_head d)(next_pos prog(fst(ct_branch d))(snd(ct_branch d)))&&
 Nat.ltb(fst(ct_cut d))(List.length prog)&&phase_closed prog(ct_cut d)&&
 phase_after(ct_cut d)(ct_exit d,0%nat)&&
 short_all(fun event=>negb(phase_after(ct_cut d)event)&&occurrence_member(map snd(ct_body d))event)events.
Record context_count:={cq_counter:Z;cq_stride:Z;cq_limit:Z}.
Definition context_proceed q:=(0<?cq_stride q)&&(cq_counter q<?cq_limit q).
Definition context_advance q:={|cq_counter:=cq_counter q+cq_stride q;cq_stride:=cq_stride q;cq_limit:=cq_limit q|}.
Definition context_rank q:=Z.to_nat(Z.max 0(cq_limit q-cq_counter q)).
Lemma controller_rank_decreases:forall q,context_proceed q=true ->
 (context_rank(context_advance q)<context_rank q)%nat.
Proof.
 intros q H. unfold context_proceed in H. apply andb_true_iff in H.
 destruct H as[Hs Hc]. apply Z.ltb_lt in Hs,Hc. apply context_rank_decreases;assumption.
Qed.
Definition controller_invariant F d E q (t:nat) s:Prop:=
 env_ok F E /\regs_ok F E(regs s)/\
 if context_proceed q then
 pc_of s=ct_head d /\ 0<=cq_counter q /\ 1<=cq_stride q<=ct_stride_max d /\
 0<=cq_limit q<=ct_limit_max d /\
 regs s(ct_counter d)=cq_counter q /\regs s(ct_stride d)=cq_stride q /\regs s(ct_limit d)=cq_limit q
 else pc_of s=(ct_exit d,0%nat).

Lemma controller_done:forall F prog events d,
 controller_check F prog events d=true ->forall E q t s,
 controller_invariant F d E q t s ->context_proceed q=false ->
 occurrence_trace F prog E events [] t s.
Proof.
 intros F prog events d Hcheck E q t s[HE[Hrs Hinv]]Hgo.
 unfold controller_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 rewrite Hgo in Hinv.
 eapply(occurrence_cut_trace F prog E events(ct_cut d)t s).
 - apply Nat.ltb_lt;tauto.
 - tauto.
 - assert(Hall:short_all(fun event=>negb(phase_after(ct_cut d)event)&&occurrence_member(map snd(ct_body d))event)events=true)by tauto.
   rewrite short_all_same,forallb_forall in Hall. intros event Hin. specialize(Hall event Hin).
   apply andb_true_iff in Hall. apply negb_true_iff. exact(proj1 Hall).
 - rewrite Hinv. tauto.
Qed.

Lemma controller_body:forall F prog events d,
 controller_check F prog events d=true ->forall E q t s,
 controller_invariant F d E q t s ->context_proceed q=true ->
 exists n last,occurrence_observe F prog E events n t s=Some(map snd(ct_body d),last) /\
 controller_invariant F d E(context_advance q)(n+t)%nat last.
Proof.
 intros F prog events d Hcheck E q t s[HE[Hrs Hinv]]Hgo.
 unfold controller_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 rewrite Hgo in Hinv. destruct Hinv as[Hhead[Hc[Hs[Hl[Hcv[Hsv Hlv]]]]]].
 assert(Hpos:0<?cq_stride q=true)by(apply Z.ltb_lt;lia).
 assert(Hless:cq_counter q<cq_limit q).
 {unfold context_proceed in Hgo. apply andb_true_iff in Hgo. apply Z.ltb_lt. exact(proj2 Hgo). }
 assert(Hcap:ct_limit_max d+ct_stride_max d<2^63)by(apply Z.ltb_lt;tauto).
 assert(Hbody:context_events F prog events(controller_protected d)(ct_head d)(ct_body_end d)(ct_body d)=true)by tauto.
 destruct(context_events_sound(ct_body d)F prog E events(controller_protected d)
  (ct_head d)(ct_body_end d)t s HE Hbody Hhead Hrs)
  as[nb[sb[Hb[Hbp[Hbr Hbs]]]]].
 assert(Htail:context_segment F prog events(ct_increment d)(controller_protected d)(ct_body_end d)(ct_tail d)=true)by tauto.
 destruct(context_segment_sound F prog E events(ct_increment d)(controller_protected d)
  (ct_body_end d)(ct_tail d)(nb+t)%nat sb HE Htail Hbp Hbr)
  as[ns[si[Hi[Hip[Hir His]]]]].
 assert(Hpres:forall r,In r(controller_protected d)->regs si r=regs s r).
 {intros r Hr. rewrite(His r Hr). exact(Hbs r Hr). }
 assert(Hic:regs si(ct_counter d)=cq_counter q)by(rewrite Hpres;[exact Hcv|left;reflexivity]).
 assert(Hisv:regs si(ct_stride d)=cq_stride q)by(rewrite Hpres;[exact Hsv|right;left;reflexivity]).
 assert(Hil:regs si(ct_limit d)=cq_limit q)by(rewrite Hpres;[exact Hlv|right;right;left;reflexivity]).
 assert(Hnumeric:state_ok F(ns+(nb+t))%nat(context_range_state(controller_bounds d))(regs si)).
 {apply context_range_state_sound. intros r lo hi Hin. unfold controller_bounds in Hin.
  destruct Hin as[Hwhich|[Hwhich|[Hwhich|[]]]];injection Hwhich as <- <- <-;rewrite ?Hic,?Hisv,?Hil;lia. }
 assert(Hinc:signed_loop_step_ok F prog(ct_increment d)(context_range_state(controller_bounds d))
   (context_increment(ct_counter d)(ct_stride d))=true)by tauto.
 assert(Hincfree:occurrence_member events(ct_increment d)=false)by(apply negb_true_iff;tauto).
 assert(Hsum:0<=regs si(ct_counter d)+regs si(ct_stride d)<2^63)by(rewrite Hic,Hisv;lia).
 assert(Hgap:context_segment F prog events(ct_test d)(controller_protected d)
  (next_pos prog(fst(ct_increment d))(snd(ct_increment d)))(ct_increment_gap d)=true)by tauto.
 destruct(context_increment_segment_sound F prog E events(ct_increment d)(ct_test d)(ct_increment_gap d)
  (controller_protected d)(context_range_state(controller_bounds d))
  (ct_counter d)(ct_stride d)(ns+(nb+t))%nat si HE Hinc Hincfree Hgap (or_introl eq_refl) Hip Hnumeric Hir Hsum)
  as[na[sa[Ha[Hap[Har[Hac Has]]]]]].
 assert(Halias_s:ct_stride d<>ct_counter d)by(apply Pos.eqb_neq,negb_true_iff;tauto).
 assert(Halias_l:ct_limit d<>ct_counter d)by(apply Pos.eqb_neq,negb_true_iff;tauto).
 assert(Hacv:regs sa(ct_counter d)=cq_counter q+cq_stride q)by(rewrite Hac,Hic,Hisv;reflexivity).
 assert(Hasv:regs sa(ct_stride d)=cq_stride q)by(rewrite(Has _ (or_intror(or_introl eq_refl)) Halias_s);exact Hisv).
 assert(Halv:regs sa(ct_limit d)=cq_limit q)by(rewrite(Has _ (or_intror(or_intror(or_introl eq_refl))) Halias_l);exact Hil).
 assert(Htest:context_test false F prog events(controller_protected d)(ct_test d)(ct_branch d)(ct_test_gap d)
   (ct_counter d)(ct_limit d)(ct_predicate d)CLt true(ct_exit d)=true)by tauto.
 destruct(context_test_sound false F prog E events(controller_protected d)(ct_test d)(ct_branch d)(ct_test_gap d)
   (ct_counter d)(ct_limit d)(ct_predicate d)CLt true(ct_exit d)(na+(ns+(nb+t)))%nat sa
   HE Htest Hap Har ltac:(rewrite Hacv;lia)ltac:(rewrite Halv;lia))
  as[nt[last[Ht[Htp[Htr Hts]]]]].
 assert(Hheadnext:next_pos prog(fst(ct_branch d))(snd(ct_branch d))=ct_head d)by(apply phase_event_eq;tauto).
 rewrite Hheadnext,Hacv,Halv in Htp.
 assert(Hobs:occurrence_observe F prog E events(nb+(ns+(na+nt)))%nat t s=Some(map snd(ct_body d),last)).
 {pose proof(observed_append F prog E events na(ns+(nb+t))%nat si []sa nt []last Ha Ht)as H1.
  cbn[List.app]in H1.
  pose proof(observed_append F prog E events ns(nb+t)%nat sb []si(na+nt)%nat []last Hi H1)as H2.
  cbn[List.app]in H2.
  pose proof(observed_append F prog E events nb t s(map snd(ct_body d))sb(ns+(na+nt))%nat []last Hb H2)as H3.
  now rewrite app_nil_r in H3. }
 exists(nb+(ns+(na+nt)))%nat,last. split;[exact Hobs|].
 unfold controller_invariant. split;[exact HE|]. split;[exact Htr|].
 assert(Hnextgo:context_proceed(context_advance q)=(cq_counter q+cq_stride q <?cq_limit q)).
 {change(((0<?cq_stride q)&&(cq_counter q+cq_stride q <?cq_limit q))=(cq_counter q+cq_stride q <?cq_limit q)). now rewrite Hpos. }
 rewrite Hnextgo. cbn[context_advance cq_counter cq_stride cq_limit].
 destruct(cq_counter q+cq_stride q <? cq_limit q)eqn:Hmore.
 - apply Z.ltb_lt in Hmore. repeat split;try lia;try exact Htp.
   + rewrite(Hts _ (or_introl eq_refl));exact Hacv.
   + rewrite(Hts _ (or_intror(or_introl eq_refl)));exact Hasv.
   + rewrite(Hts _ (or_intror(or_intror(or_introl eq_refl))));exact Halv.
 - exact Htp.
Qed.

Definition controller_word d q:=context_word(map snd(ct_body d))context_count
 context_proceed context_advance(S(context_rank q))q.
Theorem controller_pair_sound:forall F prog events d,
 controller_check F prog events d=true ->forall q E1 t1 s1 E2 t2 s2,
 controller_invariant F d E1 q t1 s1 ->controller_invariant F d E2 q t2 s2 ->
 occurrence_trace F prog E1 events(controller_word d q)t1 s1 /\
 occurrence_trace F prog E2 events(controller_word d q)t2 s2.
Proof.
 intros F prog events d Hcheck q E1 t1 s1 E2 t2 s2 H1 H2.
 eapply(context_pair_complete F prog events(map snd(ct_body d))context_count
  context_proceed context_advance context_rank(controller_invariant F d)
  controller_rank_decreases(controller_body F prog events d Hcheck)
  (controller_done F prog events d Hcheck));eassumption.
Qed.



(* ContextEntry *)
Record context_entry:={
 ce_counter:form;ce_stride:form;ce_limit:form;
 ce_test:pc;ce_branch:pc;ce_gap:list pc;ce_predicate:reg;ce_comparison:cmp;ce_reversed:bool;ce_negated:bool;ce_fuel:nat
}.
Definition entry_early_bindings d entry:=[(ct_counter d,ce_counter entry);(ct_limit d,ce_limit entry)].
Definition entry_ready_bindings d entry:=(ct_stride d,ce_stride entry)::entry_early_bindings d entry.
Definition entry_early d entry:=context_exact_state(entry_early_bindings d entry).
Definition entry_ready d entry:=context_exact_state(entry_ready_bindings d entry).
Definition entry_context F E entry:context_count:={|
 cq_counter:=context_form_value F E(ce_counter entry);
 cq_stride:=context_form_value F E(ce_stride entry);
 cq_limit:=context_form_value F E(ce_limit entry)|}.
Definition context_entry_check F uniform prog events d entry:=
 let keep:=phase_control_regs(List.concat prog)in
 lf_ok F &&controller_check F prog events d&&
 context_prefix(ce_fuel entry)keep F prog events(0%nat,0%nat)(ce_test entry)(empty_state F)(entry_early d entry)&&
 context_value_check F uniform(entry_early d entry)(ct_counter d)(ce_counter entry)&&
 context_value_check F uniform(entry_early d entry)(ct_limit d)(ce_limit entry)&&
 context_unsigned_range F(entry_early d entry)(ct_counter d)0(2^63-1)&&
 context_unsigned_range F(entry_early d entry)(ct_limit d)0(ct_limit_max d)&&
 context_test(ce_reversed entry)F prog events(controller_protected d)(ce_test entry)(ce_branch entry)(ce_gap entry)
  (ct_counter d)(ct_limit d)(ce_predicate entry)(ce_comparison entry)(ce_negated entry)(ct_exit d)&&
 context_prefix(ce_fuel entry)keep F prog events(next_pos prog(fst(ce_branch entry))(snd(ce_branch entry)))
  (ct_head d)(entry_early d entry)(entry_ready d entry)&&
 context_value_check F uniform(entry_ready d entry)(ct_counter d)(ce_counter entry)&&
 context_value_check F uniform(entry_ready d entry)(ct_limit d)(ce_limit entry)&&
 context_value_check F uniform(entry_ready d entry)(ct_stride d)(ce_stride entry)&&
 context_unsigned_range F(entry_ready d entry)(ct_stride d)1(ct_stride_max d).
Lemma context_value_form:forall F uniform st r f,
 context_value_check F uniform st r f=true ->context_form_ok F uniform f=true.
Proof. intros. unfold context_value_check in H. rewrite !andb_true_iff in H. tauto. Qed.
Lemma context_entry_agree:forall F uniform prog events d entry E1 E2,
 context_entry_check F uniform prog events d entry=true ->context_initial_agree uniform E1 E2 ->
 entry_context F E1 entry=entry_context F E2 entry.
Proof.
 intros F uniform prog events d entry E1 E2 Hcheck Hagree.
 unfold context_entry_check in Hcheck;cbv zeta in Hcheck. rewrite !andb_true_iff in Hcheck.
 unfold entry_context. f_equal.
 - apply(context_form_agree F uniform E1 E2);[|exact Hagree].
   eapply(context_value_form F uniform(entry_ready d entry)(ct_counter d));tauto.
 - apply(context_form_agree F uniform E1 E2);[|exact Hagree].
   eapply(context_value_form F uniform(entry_ready d entry)(ct_stride d));tauto.
 - apply(context_form_agree F uniform E1 E2);[|exact Hagree].
   eapply(context_value_form F uniform(entry_ready d entry)(ct_limit d));tauto.
Qed.
Lemma context_controller_one:forall F prog events d E q t s,
 controller_check F prog events d=true ->controller_invariant F d E q t s ->
 occurrence_trace F prog E events(controller_word d q)t s.
Proof. intros. exact(proj1(controller_pair_sound F prog events d H q E t s E t s H0 H0)). Qed.
Theorem context_entry_sound:forall F uniform prog events d entry,
 context_entry_check F uniform prog events d entry=true ->forall E,
 launch_ok F E ->occurrence_trace F prog E events(controller_word d(entry_context F E entry))0(phase_initial E).
Proof.
 intros F uniform prog events d entry Hcheck E Hlaunch.
 unfold context_entry_check in Hcheck;cbv zeta in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as [[[[[[[[[[[[Hlf Hcontroller] Hearly] Hvc] Hvl] Hrc] Hrl] Htest] Hready] Hreadyc] Hreadyl] Hreadys] Hreadysrange].
 assert(HE:env_ok F E)by(exact(env_of_launch F prog E Hlf Hlaunch)).
 assert(Hinitial:state_ok F 0(empty_state F)(regs(phase_initial E))).
 {split;[|exact(lk_derived F E Hlaunch)]. intros r f Hf. unfold get,empty_state in Hf;cbn[amap]in Hf.
  rewrite PM.gempty in Hf. discriminate. }
 assert(Hinitialregs:regs_ok F E(regs(phase_initial E))).
 {split;[exact(lk_init F E Hlaunch)|]. intros;reflexivity. }
 destruct(context_prefix_sound(ce_fuel entry)(phase_control_regs(List.concat prog))F prog E events
   (0%nat,0%nat)(ce_test entry)(empty_state F)(entry_early d entry)0(phase_initial E)
   HE Hearly eq_refl Hinitial Hinitialregs)
  as[n[at_test[Hprefix[Htestpc[Hteststate Htestrs]]]]].
 assert(Hcv:regs at_test(ct_counter d)=context_form_value F E(ce_counter entry))
  by(eapply(context_value_sound F uniform E(entry_early d entry)(n+0)%nat(regs at_test)(ct_counter d)(ce_counter entry));[exact HE|exact Htestrs|exact Hteststate|exact Hvc]).
 assert(Hlv:regs at_test(ct_limit d)=context_form_value F E(ce_limit entry))
  by(eapply(context_value_sound F uniform E(entry_early d entry)(n+0)%nat(regs at_test)(ct_limit d)(ce_limit entry));[exact HE|exact Htestrs|exact Hteststate|exact Hvl]).
 assert(Hcb:0<=regs at_test(ct_counter d)<=2^63-1)
  by(eapply(context_unsigned_range_sound F E(entry_early d entry)(n+0)%nat(regs at_test)(ct_counter d)0(2^63-1));[exact HE|exact Htestrs|exact Hteststate|exact Hrc]).
 assert(Hlb:0<=regs at_test(ct_limit d)<=ct_limit_max d)
  by(eapply(context_unsigned_range_sound F E(entry_early d entry)(n+0)%nat(regs at_test)(ct_limit d)0(ct_limit_max d));[exact HE|exact Htestrs|exact Hteststate|exact Hrl]).
 assert(Hcap:ct_limit_max d+ct_stride_max d<2^63 /\1<=ct_stride_max d).
 {unfold controller_check in Hcontroller. rewrite !andb_true_iff in Hcontroller.
  split;[apply Z.ltb_lt|apply Z.leb_le];tauto. }
 destruct(context_test_sound(ce_reversed entry)F prog E events(controller_protected d)(ce_test entry)(ce_branch entry)(ce_gap entry)
  (ct_counter d)(ct_limit d)(ce_predicate entry)(ce_comparison entry)(ce_negated entry)(ct_exit d)(n+0)%nat at_test
  HE Htest Htestpc Htestrs ltac:(lia)ltac:(lia))
  as[nt[after_test[Htestobs[Hafterpc[Hafterrs Haftersaved]]]]].
 apply(observed_prefix_trace F prog E events n 0(phase_initial E)[] at_test (controller_word d(entry_context F E entry))Hprefix).
 apply(observed_prefix_trace F prog E events nt(n+0)%nat at_test [] after_test (controller_word d(entry_context F E entry))Htestobs).
 destruct(regs at_test(ct_counter d) <?regs at_test(ct_limit d))eqn:Hcontinue.
 - assert(Hearlypost:state_ok F(nt+(n+0))%nat(entry_early d entry)(regs after_test)).
   {apply context_exact_state_sound with(uniform:=uniform)(E:=E);[exact Hafterrs|].
    intros r f Hin. unfold entry_early_bindings in Hin.
    destruct Hin as[Hwhich|[Hwhich|[]]];injection Hwhich as <- <-;split.
    - exact(context_value_form F uniform(entry_early d entry)(ct_counter d)(ce_counter entry)Hvc).
    - rewrite(Haftersaved _ (or_introl eq_refl));exact Hcv.
    - exact(context_value_form F uniform(entry_early d entry)(ct_limit d)(ce_limit entry)Hvl).
    - rewrite(Haftersaved _ (or_intror(or_intror(or_introl eq_refl))));exact Hlv. }
   destruct(context_prefix_sound(ce_fuel entry)(phase_control_regs(List.concat prog))F prog E events
    (next_pos prog(fst(ce_branch entry))(snd(ce_branch entry)))(ct_head d)(entry_early d entry)(entry_ready d entry)
    (nt+(n+0))%nat after_test HE Hready Hafterpc Hearlypost Hafterrs)
    as[ni[head[Hinit[Hheadpc[Hheadstate Hheadrs]]]]].
   apply(observed_prefix_trace F prog E events ni(nt+(n+0))%nat after_test [] head (controller_word d(entry_context F E entry))Hinit).
   apply context_controller_one;[exact Hcontroller|].
   assert(Hheadc:regs head(ct_counter d)=context_form_value F E(ce_counter entry))
    by(eapply(context_value_sound F uniform E(entry_ready d entry)(ni+(nt+(n+0)))%nat(regs head)(ct_counter d)(ce_counter entry));[exact HE|exact Hheadrs|exact Hheadstate|exact Hreadyc]).
   assert(Hheadl:regs head(ct_limit d)=context_form_value F E(ce_limit entry))
    by(eapply(context_value_sound F uniform E(entry_ready d entry)(ni+(nt+(n+0)))%nat(regs head)(ct_limit d)(ce_limit entry));[exact HE|exact Hheadrs|exact Hheadstate|exact Hreadyl]).
   assert(Hheads:regs head(ct_stride d)=context_form_value F E(ce_stride entry))
    by(eapply(context_value_sound F uniform E(entry_ready d entry)(ni+(nt+(n+0)))%nat(regs head)(ct_stride d)(ce_stride entry));[exact HE|exact Hheadrs|exact Hheadstate|exact Hreadys]).
   assert(Hsb:1<=regs head(ct_stride d)<=ct_stride_max d)
    by(eapply(context_unsigned_range_sound F E(entry_ready d entry)(ni+(nt+(n+0)))%nat(regs head)(ct_stride d)1(ct_stride_max d));[exact HE|exact Hheadrs|exact Hheadstate|exact Hreadysrange]).
   assert(Hgo:context_proceed(entry_context F E entry)=true).
   {unfold context_proceed,entry_context;cbn[cq_counter cq_stride cq_limit].
    apply andb_true_iff. split;[apply Z.ltb_lt;rewrite <-Hheads;lia|]. now rewrite <-Hcv,<-Hlv. }
   unfold controller_invariant. split;[exact HE|]. split;[exact Hheadrs|]. rewrite Hgo.
   cbn[entry_context cq_counter cq_stride cq_limit]. repeat split;try assumption;
   rewrite ?Hcv,?Hlv in Hcb,Hlb;rewrite ?Hheads in Hsb;lia.
 - apply context_controller_one;[exact Hcontroller|].
   unfold controller_invariant. split;[exact HE|]. split;[exact Hafterrs|].
   assert(Hgo:context_proceed(entry_context F E entry)=false).
   {unfold context_proceed,entry_context;cbn[cq_counter cq_stride cq_limit]. rewrite <-Hcv,<-Hlv,Hcontinue. apply andb_false_r. }
   now rewrite Hgo.
Qed.
Theorem context_entry_pair_sound:forall F uniform prog events d entry,
 context_entry_check F uniform prog events d entry=true ->forall E1 E2,
 launch_ok F E1 ->launch_ok F E2 ->context_initial_agree uniform E1 E2 ->
 let word:=controller_word d(entry_context F E1 entry)in
 occurrence_trace F prog E1 events word 0(phase_initial E1) /\
 occurrence_trace F prog E2 events word 0(phase_initial E2).
Proof.
 intros F uniform prog events d entry Hcheck E1 E2 H1 H2 Hagree. split.
 - eapply context_entry_sound;eassumption.
 - rewrite(context_entry_agree F uniform prog events d entry E1 E2 Hcheck Hagree).
   eapply context_entry_sound;eassumption.
Qed.


Record uniform_entry:={
 ue_test:pc;ue_branch:pc;ue_gap:list pc;ue_predicate:reg;ue_comparison:cmp;
 ue_reversed:bool;ue_negated:bool;ue_fuel:nat;
 ue_early:list instr;ue_ready:list instr
}.
Definition uniform_initial_marks F uniform r:=uniform r && negb(is_cell F r).
Definition uniform_ready_marks F uniform d r:=
 if Pos.eqb r(ct_counter d)||Pos.eqb r(ct_limit d)then negb(is_cell F r)
 else is_fixed F r && uniform r && negb(is_cell F r).
Definition uniform_before_values F E e:=uniform_values_run F(ue_early e)(e_init E).
Definition uniform_ready_values F E d e r:=
 if Pos.eqb r(ct_counter d)then uniform_before_values F E e(ct_counter d)
 else if Pos.eqb r(ct_limit d)then uniform_before_values F E e(ct_limit d)
 else e_init E r.
Definition uniform_ready_state d:=context_range_state
 [(ct_counter d,(0,2^63-1));(ct_limit d,(0,ct_limit_max d))].
Definition uniform_entry_context F E d e:context_count:={|
 cq_counter:=uniform_before_values F E e(ct_counter d);
 cq_stride:=uniform_values_run F(ue_ready e)(uniform_ready_values F E d e)(ct_stride d);
 cq_limit:=uniform_before_values F E e(ct_limit d)|}.
Definition uniform_entry_check F uniform prog events d e:=
 let early_marks:=uniform_marks_run F(ue_early e)(uniform_initial_marks F uniform)in
 let early_state:=uniform_states_run F(ue_early e)(empty_state F)in
 let ready_marks:=uniform_marks_run F(ue_ready e)(uniform_ready_marks F uniform d)in
 let ready_state:=uniform_states_run F(ue_ready e)(uniform_ready_state d)in
 lf_ok F && controller_check F prog events d &&
 uniform_prefix_check(ue_fuel e)F prog events(ue_test e)(0%nat,0%nat)(ue_early e)&&
 early_marks(ct_counter d)&&
 early_marks(ct_limit d)&&
 context_unsigned_range F(early_state)(ct_counter d)0(2^63-1)&&
 context_unsigned_range F(early_state)(ct_limit d)0(ct_limit_max d)&&
 context_test(ue_reversed e)F prog events(controller_protected d)
  (ue_test e)(ue_branch e)(ue_gap e)(ct_counter d)(ct_limit d)(ue_predicate e)(ue_comparison e)(ue_negated e)(ct_exit d)&&
 uniform_prefix_check(ue_fuel e)F prog events(ct_head d)
  (next_pos prog(fst(ue_branch e))(snd(ue_branch e)))(ue_ready e)&&
 negb(is_cell F(ct_counter d))&&negb(is_cell F(ct_limit d))&&
 forallb(loop_preserves(ct_counter d))(ue_ready e)&&
 forallb(loop_preserves(ct_limit d))(ue_ready e)&&
 ready_marks(ct_stride d)&&
 context_unsigned_range F(ready_state)(ct_stride d)1(ct_stride_max d).

Lemma uniform_before_agree:forall F uniform E1 E2 e,
 context_initial_agree uniform E1 E2 ->
 uniform_agree(uniform_marks_run F(ue_early e)(uniform_initial_marks F uniform))
  (uniform_before_values F E1 e)(uniform_before_values F E2 e).
Proof.
 intros. apply uniform_values_run_agree. intros r Hr.
 unfold uniform_initial_marks in Hr. apply andb_true_iff in Hr. apply H;exact(proj1 Hr).
Qed.
Lemma uniform_ready_agree:forall F uniform E1 E2 d e,
 context_initial_agree uniform E1 E2 ->
 uniform_before_values F E1 e(ct_counter d)=uniform_before_values F E2 e(ct_counter d) ->
 uniform_before_values F E1 e(ct_limit d)=uniform_before_values F E2 e(ct_limit d) ->
 uniform_agree(uniform_ready_marks F uniform d)(uniform_ready_values F E1 d e)(uniform_ready_values F E2 d e).
Proof.
 intros F uniform E1 E2 d e Hagree Hcounter Hlimit r Hr.
 unfold uniform_ready_marks in Hr. unfold uniform_ready_values.
 destruct(Pos.eqb r(ct_counter d))eqn:Hc;[exact Hcounter|].
 destruct(Pos.eqb r(ct_limit d))eqn:Hl;[exact Hlimit|].
 cbn in Hr. rewrite !andb_true_iff in Hr. apply Hagree;tauto.
Qed.
Lemma uniform_entry_agree:forall F uniform prog events d e E1 E2,
 uniform_entry_check F uniform prog events d e=true ->context_initial_agree uniform E1 E2 ->
 uniform_entry_context F E1 d e=uniform_entry_context F E2 d e.
Proof.
 intros F uniform prog events d e E1 E2 Hcheck Hagree.
 unfold uniform_entry_check in Hcheck;cbv zeta in Hcheck. rewrite !andb_true_iff in Hcheck.
 pose proof(uniform_before_agree F uniform E1 E2 e Hagree)as Hbefore.
 assert(Hc:uniform_before_values F E1 e(ct_counter d)=uniform_before_values F E2 e(ct_counter d))by(apply Hbefore;tauto).
 assert(Hl:uniform_before_values F E1 e(ct_limit d)=uniform_before_values F E2 e(ct_limit d))by(apply Hbefore;tauto).
 unfold uniform_entry_context. f_equal;try assumption.
 apply(uniform_values_run_agree(ue_ready e)F(uniform_ready_marks F uniform d)
 (uniform_ready_values F E1 d e)(uniform_ready_values F E2 d e));[|tauto].
 apply uniform_ready_agree;assumption.
Qed.
Lemma uniform_ready_initial:forall F uniform E d e s rs,
 regs_ok F E rs ->
 regs s(ct_counter d)=uniform_before_values F E e(ct_counter d) ->
 regs s(ct_limit d)=uniform_before_values F E e(ct_limit d) ->
 rs(ct_counter d)=regs s(ct_counter d) ->rs(ct_limit d)=regs s(ct_limit d) ->
 uniform_agree(uniform_ready_marks F uniform d)rs(uniform_ready_values F E d e).
Proof.
 intros F uniform E d e s rs Hrs Hc Hl Hsavedc Hsavedl r Hr.
 unfold uniform_ready_marks in Hr. unfold uniform_ready_values.
 destruct(Pos.eqb r(ct_counter d))eqn:Hrc.
 - apply Pos.eqb_eq in Hrc. rewrite Hrc. congruence.
 - destruct(Pos.eqb r(ct_limit d))eqn:Hrl.
   + apply Pos.eqb_eq in Hrl. rewrite Hrl. congruence.
   +
 cbn in Hr. rewrite !andb_true_iff in Hr. apply(proj2 Hrs);tauto.
Qed.

Theorem uniform_entry_sound:forall F uniform prog events d e,
 uniform_entry_check F uniform prog events d e=true ->forall E,
 launch_ok F E ->occurrence_trace F prog E events(controller_word d(uniform_entry_context F E d e))0(phase_initial E).
Proof.
 intros F uniform prog events d e Hcheck E Hlaunch.
 unfold uniform_entry_check in Hcheck;cbv zeta in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[[[[[[[[[[[[Hlf Hcontroller]Hearly]Hcm]Hlm]Hcr]Hlr]Htest]Hready]Hccell]Hlcell]Hckeep]Hlkeep]Hsm]Hsr].
 apply uniform_prefix_check_sound in Hearly,Hready.
 apply negb_true_iff in Hccell,Hlcell.
 assert(HE:env_ok F E)by(exact(env_of_launch F prog E Hlf Hlaunch)).
 assert(Hinitial:state_ok F 0(empty_state F)(regs(phase_initial E))).
 {split;[|exact(lk_derived F E Hlaunch)]. intros r f Hf. unfold get,empty_state in Hf;cbn[amap]in Hf.
  rewrite PM.gempty in Hf. discriminate. }
 assert(Hinitialregs:regs_ok F E(regs(phase_initial E))).
 {split;[exact(lk_init F E Hlaunch)|]. intros;reflexivity. }
 assert(Hinitialvalues:uniform_agree(uniform_initial_marks F uniform)(regs(phase_initial E))(e_init E))by(intros r Hr;reflexivity).
 destruct(uniform_prefix_full_sound(ue_fuel e)F prog E events(ue_test e)(0%nat,0%nat)(ue_early e)
 (empty_state F)(uniform_initial_marks F uniform)(e_init E)0(phase_initial E)
 HE Hearly eq_refl Hinitialregs Hinitial Hinitialvalues)
 as[at_test[Hprefix[Htestpc[Htestrs[Hteststate Htestvalues]]]]].
 set(n:=length(ue_early e))in *.
 assert(Hcv:regs at_test(ct_counter d)=uniform_before_values F E e(ct_counter d))by(apply Htestvalues;exact Hcm).
 assert(Hlv:regs at_test(ct_limit d)=uniform_before_values F E e(ct_limit d))by(apply Htestvalues;exact Hlm).
 assert(Hcb:0<=regs at_test(ct_counter d)<=2^63-1)
 by(eapply context_unsigned_range_sound;[exact HE|exact Htestrs|exact Hteststate|exact Hcr]).
 assert(Hlb:0<=regs at_test(ct_limit d)<=ct_limit_max d)
 by(eapply context_unsigned_range_sound;[exact HE|exact Htestrs|exact Hteststate|exact Hlr]).
 assert(Hcap:ct_limit_max d+ct_stride_max d<2^63 /\1<=ct_stride_max d).
 {unfold controller_check in Hcontroller. rewrite !andb_true_iff in Hcontroller.
  split;[apply Z.ltb_lt|apply Z.leb_le];tauto. }
 destruct(context_test_sound(ue_reversed e)F prog E events(controller_protected d)
 (ue_test e)(ue_branch e)(ue_gap e)(ct_counter d)(ct_limit d)(ue_predicate e)(ue_comparison e)(ue_negated e)(ct_exit d)
 n at_test HE Htest Htestpc Htestrs ltac:(lia)ltac:(lia))
 as[nt[after_test[Htestobs[Hafterpc[Hafterrs Haftersaved]]]]].
 apply(observed_prefix_trace F prog E events n 0(phase_initial E)[]at_test _ Hprefix).
 replace(n+0)%nat with n by lia.
 apply(observed_prefix_trace F prog E events nt n at_test []after_test _ Htestobs).
 destruct(regs at_test(ct_counter d)<?regs at_test(ct_limit d))eqn:Hcontinue.
 - assert(Hafterstate:state_ok F(nt+n)%nat(uniform_ready_state d)(regs after_test)).
   {apply context_range_state_sound. intros r lo hi Hin. cbn in Hin.
    destruct Hin as[Hwhich|[Hwhich|[]]];injection Hwhich as <- <- <-;
    rewrite Haftersaved;try assumption;unfold controller_protected;cbn;auto. }
   assert(Haftervalues:uniform_agree(uniform_ready_marks F uniform d)(regs after_test)(uniform_ready_values F E d e)).
   {eapply uniform_ready_initial;try eassumption.
    - apply Haftersaved;left;reflexivity.
    - apply Haftersaved;right;right;left;reflexivity. }
   destruct(uniform_prefix_full_sound(ue_fuel e)F prog E events(ct_head d)
    (next_pos prog(fst(ue_branch e))(snd(ue_branch e)))(ue_ready e)(uniform_ready_state d)
    (uniform_ready_marks F uniform d)(uniform_ready_values F E d e)(nt+n)%nat after_test
    HE Hready(eq_sym Hafterpc)Hafterrs Hafterstate Haftervalues)
   as[head[Hinit[Hheadpc[Hheadrs[Hheadstate Hheadvalues]]]]].
   set(ni:=length(ue_ready e))in *.
   apply(observed_prefix_trace F prog E events ni(nt+n)%nat after_test []head _ Hinit).
   assert(Hheadc:regs head(ct_counter d)=uniform_before_values F E e(ct_counter d)).
   {rewrite(uniform_prefix_preserves(ue_fuel e)F prog E events(ct_head d)
    (next_pos prog(fst(ue_branch e))(snd(ue_branch e)))(ue_ready e)(ct_counter d)(nt+n)%nat after_test head
    Hccell Hready Hckeep(eq_sym Hafterpc)Hinit).
    rewrite(Haftersaved _ (or_introl eq_refl)). exact Hcv. }
   assert(Hheadl:regs head(ct_limit d)=uniform_before_values F E e(ct_limit d)).
   {rewrite(uniform_prefix_preserves(ue_fuel e)F prog E events(ct_head d)
    (next_pos prog(fst(ue_branch e))(snd(ue_branch e)))(ue_ready e)(ct_limit d)(nt+n)%nat after_test head
    Hlcell Hready Hlkeep(eq_sym Hafterpc)Hinit).
    rewrite(Haftersaved _ (or_intror(or_intror(or_introl eq_refl)))). exact Hlv. }
   assert(Hheads:regs head(ct_stride d)=uniform_values_run F(ue_ready e)(uniform_ready_values F E d e)(ct_stride d))by(apply Hheadvalues;exact Hsm).
   assert(Hsb:1<=regs head(ct_stride d)<=ct_stride_max d)
    by(eapply context_unsigned_range_sound;[exact HE|exact Hheadrs|exact Hheadstate|exact Hsr]).
   assert(Hgo:context_proceed(uniform_entry_context F E d e)=true).
   {unfold context_proceed,uniform_entry_context;cbn[cq_counter cq_stride cq_limit].
    apply andb_true_iff. split;[apply Z.ltb_lt;rewrite <-Hheads;lia|]. now rewrite <-Hcv,<-Hlv. }
   apply context_controller_one;[exact Hcontroller|];
   unfold controller_invariant;split;[exact HE|];split;[exact Hheadrs|];rewrite Hgo;
   cbn[uniform_entry_context cq_counter cq_stride cq_limit];repeat split;try assumption;
   rewrite ?Hcv,?Hlv in Hcb,Hlb;rewrite ?Hheads in Hsb;lia.
 - assert(Hgo:context_proceed(uniform_entry_context F E d e)=false).
   {unfold context_proceed,uniform_entry_context;cbn[cq_counter cq_stride cq_limit].
    rewrite <-Hcv,<-Hlv,Hcontinue. apply andb_false_r. }
   apply context_controller_one;[exact Hcontroller|];
   unfold controller_invariant;split;[exact HE|];split;[exact Hafterrs|];now rewrite Hgo.
Qed.

(* Discovery only. Every proposed field is rechecked by context_entry_check. *)
Record context_shape := {
 cs_counter : reg; cs_stride : reg; cs_limit : reg; cs_predicate : reg;
 cs_increment : pc; cs_increment_gap:list pc; cs_test : pc; cs_branch : pc; cs_head : pc; cs_exit : nat
}.
Definition context_items (prog:program) : list(pc*instr) :=
 flat_map(fun indexed=>let '(b,body):=indexed in
   map(fun item=>let '(i,ins):=item in ((b,i),ins))
     (combine(seq 0(List.length body))body))
   (combine(seq 0(List.length prog))prog).
Fixpoint context_find_test(prog:program)(events:list pc)(counter stride:reg)
 (block index:nat)(body:list instr):option(pc*instr):=
 match body with
 |[]=>None
 |i::rest=>match i with
  |ISetp None CLt 64 true _ tested(OReg _)=>
    if Pos.eqb counter tested then Some((block,index),i)else None
  |_=>if negb(occurrence_member events(block,index))&&loop_preserves counter i&&loop_preserves stride i then
    match i with IBra _ _|IExit _=>None
    |_=>context_find_test prog events counter stride block(S index)rest end
    else None end end.
(* The adjacent instruction uses actual CFG fallthrough, including a block boundary. *)
Definition context_test_candidate (allow_gap:bool) prog events counter stride pos:option(pc*instr*list pc):=
 let first:=next_pos prog(fst pos)(snd pos)in
 match fetch prog(fst first)(snd first)with
 |Some(ISetp None CLt 64 true predicate tested(OReg limit))=>
   Some(first,ISetp None CLt 64 true predicate tested(OReg limit),[])
 |_=>if allow_gap then
   match context_find_test prog events counter stride(fst pos)(S(snd pos))
    (skipn(S(snd pos))(nth(fst pos)prog[]))with
   |Some(test,i)=>Some(test,i,map(fun n=>(fst pos,n))(seq(S(snd pos))(snd test-S(snd pos))))
   |None=>None end else None end.
Definition context_shape_at_mode (allow_gap:bool)(prog:program)(events:list pc)(pos:pc)(ins:instr):option context_shape:=
 match ins with
 |IBin None BAdd 64 counter source(OReg stride)=>
  match context_test_candidate allow_gap prog events counter stride pos with
  |Some(test,ISetp None CLt 64 true predicate tested(OReg limit),gap)=>
   let branch:=next_pos prog(fst test)(snd test)in
   match fetch prog(fst branch)(snd branch)with
   |Some(IBra(Some(true,p))exit)=>
    let head:=next_pos prog(fst branch)(snd branch)in
    if Pos.eqb counter source&&Pos.eqb counter tested&&Pos.eqb predicate p&&Nat.ltb(fst head)exit&&
      negb(Nat.eqb(List.length events)0)&&
      short_all(fun e=>negb(phase_after e(fst pos,0%nat))&&phase_after e(exit,0%nat))events
    then Some{|cs_counter:=counter;cs_stride:=stride;cs_limit:=limit;cs_predicate:=predicate;
      cs_increment:=pos;cs_increment_gap:=gap;
      cs_test:=test;cs_branch:=branch;cs_head:=head;cs_exit:=exit|}else None
   |_=>None end
  |_=>None end
 |_=>None end.
Definition context_shape_at:=context_shape_at_mode true.
Definition context_shape_candidates allow_gap prog events:=
 flat_map(fun item=>match context_shape_at_mode allow_gap prog events(fst item)(snd item)with
 |Some d=>[d]|None=>[]end)(context_items prog).
(* Preserve a unique adjacent shape before considering protected-gap shapes.
   This source-only choice precedes all certificate checking. *)
Definition context_shape_select prog events :=
 match context_shape_candidates false prog events with
 |[d]=>Some d
 |[]=>match context_shape_candidates true prog events with [d]=>Some d|_=>None end
 |_=>None end.

(* This worklist never establishes soundness. Ordered region validation later
   checks every actual successor, all event exclusions, and the endpoint. *)
Fixpoint context_region_collect (fuel:nat) (prog:program) (stops:list pc)
 (pending seen hits:list pc) : option (list pc*list pc) :=
 match pending with
 | []=>Some(seen,hits)
 | pos::tail=>match fuel with
   | O=>None
   | S rest=>
     if occurrence_member stops pos then
       context_region_collect rest prog stops tail seen
         (if occurrence_member hits pos then hits else pos::hits)
     else if occurrence_member seen pos then context_region_collect rest prog stops tail seen hits
     else match fetch prog (fst pos) (snd pos) with
     |None=>None
     |Some _=>context_region_collect rest prog stops
       (occurrence_cfg_next prog pos++tail) (pos::seen) hits
     end
   end
 end.
Definition context_region_propose prog stops start : option (list pc*pc) :=
 match context_region_collect phase_fuel prog stops [start] [] [] with
 |Some(seen,[endpoint])=>Some(filter(fun pos=>occurrence_member seen pos)(phase_positions prog),endpoint)
 |_=>None end.
Fixpoint context_body_propose (fuel:nat) prog events increment start visited
 : option (list(list pc*pc)*pc*list pc) :=
 match fuel with
 |O=>None
 |S rest=>match context_region_propose prog (increment::events) start with
   |Some(nodes,endpoint)=>
     if phase_event increment endpoint then Some([],start,nodes)
     else if occurrence_member events endpoint && negb(occurrence_member visited endpoint) then
       match context_body_propose rest prog events increment
         (next_pos prog(fst endpoint)(snd endpoint))(endpoint::visited)with
       |Some(plans,finish,tail)=>Some((nodes,endpoint)::plans,finish,tail)
       |None=>None end
     else None
   |None=>None end
 end.
Record context_entry_shape := {
 ces_test:pc;ces_branch:pc;ces_gap:list pc;ces_predicate:reg;ces_comparison:cmp;ces_reversed:bool;ces_negated:bool
}.
Definition context_initial_at_mode allow_reversed prog shape(test branch:pc):option context_entry_shape:=
 match fetch prog(fst test)(snd test),fetch prog(fst branch)(snd branch)with
 |Some(ISetp None comparison 64 true p lhs(OReg rhs)),Some(IBra(Some(neg,q))exit)=>
   let ordered:=Pos.eqb lhs(cs_counter shape)&&Pos.eqb rhs(cs_limit shape)in
   let swapped:=Pos.eqb lhs(cs_limit shape)&&Pos.eqb rhs(cs_counter shape)in
   let reversed:=negb ordered in
   if (ordered||(allow_reversed&&swapped))&&Pos.eqb p q&&Nat.eqb exit(cs_exit shape)&&
     Nat.eqb(fst test)(fst branch)&&Nat.ltb(snd test)(snd branch)&&phase_after branch(cs_increment shape)&&
     context_test_polarity(context_comparison reversed comparison)neg
   then Some{|ces_test:=test;ces_branch:=branch;
     ces_gap:=map(fun i=>(fst test,i))(seq(S(snd test))(snd branch-S(snd test)));
     ces_predicate:=p;ces_comparison:=comparison;ces_reversed:=reversed;ces_negated:=neg|}
   else None
 |_,_=>None end.
Definition context_initial_at:=context_initial_at_mode true.
Definition context_initial_candidates allow_reversed prog shape:=
 let items:=context_items prog in
 let tests:=map fst(filter(fun item=>match snd item with
   |ISetp None _ 64 true _ counter(OReg limit)=>
     (Pos.eqb counter(cs_counter shape)&&Pos.eqb limit(cs_limit shape))||
     (allow_reversed&&(Pos.eqb counter(cs_limit shape)&&Pos.eqb limit(cs_counter shape)))
   |_=>false end)items)in
 let branches:=map fst(filter(fun item=>match snd item with
   |IBra(Some _)exit=>Nat.eqb exit(cs_exit shape)&&phase_after(fst item)(cs_increment shape)
   |_=>false end)items)in
 flat_map(fun test=>flat_map(fun branch=>
   match context_initial_at_mode allow_reversed prog shape test branch with Some e=>[e]|None=>[]end)branches)tests.
(* Ordered candidates retain priority over newly supported reversed tests. *)
Definition context_initial_select prog shape:=
 match context_initial_candidates false prog shape with
 |[e]=>Some e
 |[]=>match context_initial_candidates true prog shape with [e]=>Some e|_=>None end
 |_=>None end.
Definition context_prefix_propose F prog events start endpoint st :=
 match occurrence_collect_with no_loop_summary phase_fuel (phase_control_regs(List.concat prog))
   F prog(endpoint::events)(List.length prog,0%nat)[(start,Some st)]with
 |Some(_,[(pos,post)])=>if phase_event endpoint pos then Some post else None
 |_=>None end.
Definition context_get_form st r := match get st r with Some(FForm f _ _)=>Some f|_=>None end.
Definition context_propose F prog events shape : option(context_controller*context_entry) :=
 match context_initial_select prog shape,
       context_body_propose(S(List.length events))prog events(cs_increment shape)(cs_head shape)[]with
 |Some first,Some(body,finish,tail)=>
   match context_prefix_propose F prog events(0%nat,0%nat)(ces_test first)(empty_state F)with
   |Some before=>match context_get_form before(cs_counter shape),context_get_form before(cs_limit shape)with
     |Some cf,Some lf=>
       let early:=context_exact_state[(cs_counter shape,cf);(cs_limit shape,lf)]in
       match context_prefix_propose F prog events
         (next_pos prog(fst(ces_branch first))(snd(ces_branch first)))(cs_head shape)early with
       |Some ready=>match context_get_form ready(cs_stride shape),
          cmp_range F early(OReg(cs_limit shape))64 true,
          cmp_range F ready(OReg(cs_stride shape))64 true with
         |Some sf,Some(_,limit_max),Some(_,stride_max)=>
           Some({|ct_counter:=cs_counter shape;ct_stride:=cs_stride shape;ct_limit:=cs_limit shape;
             ct_predicate:=cs_predicate shape;ct_head:=cs_head shape;ct_body_end:=finish;
             ct_increment:=cs_increment shape;ct_test:=cs_test shape;ct_branch:=cs_branch shape;
             ct_exit:=cs_exit shape;ct_body:=body;ct_tail:=tail;ct_increment_gap:=cs_increment_gap shape;ct_test_gap:=[];
             (* A nonempty induction interval also covers a globally empty loop. *)
             ct_limit_max:=Z.max 1 limit_max;ct_stride_max:=stride_max;ct_cut:=scoped_cut prog events|},
             {|ce_counter:=cf;ce_stride:=sf;ce_limit:=lf;ce_test:=ces_test first;
             ce_branch:=ces_branch first;ce_gap:=ces_gap first;ce_predicate:=ces_predicate first;
             ce_comparison:=ces_comparison first;ce_reversed:=ces_reversed first;ce_negated:=ces_negated first;ce_fuel:=phase_fuel|})
         |_,_,_=>None end
       |None=>None end
     |_,_=>None end
   |None=>None end
 |_,_=>None end.
(* Eligibility selects one value-certificate representation before checking.
   Every previously valid exact-form entry selects its unchanged proof route. *)
Definition context_exact_eligible F uniform entry:=
 context_form_ok F uniform(ce_counter entry)&&context_form_ok F uniform(ce_limit entry)&&
 context_form_ok F uniform(ce_stride entry).
Definition context_uniform_propose F prog events d entry:option uniform_entry:=
 match uniform_prefix_collect(ce_fuel entry)F prog events(ce_test entry)(0%nat,0%nat),
       uniform_prefix_collect(ce_fuel entry)F prog events(ct_head d)
        (next_pos prog(fst(ce_branch entry))(snd(ce_branch entry)))with
 |Some early,Some ready=>Some{|ue_test:=ce_test entry;ue_branch:=ce_branch entry;ue_gap:=ce_gap entry;
  ue_predicate:=ce_predicate entry;ue_comparison:=ce_comparison entry;ue_reversed:=ce_reversed entry;
  ue_negated:=ce_negated entry;ue_fuel:=ce_fuel entry;ue_early:=early;ue_ready:=ready|}
 |_,_=>None end.
Definition context_check_entry F uniform prog events d entry:=
 if context_exact_eligible F uniform entry then context_entry_check F uniform prog events d entry
 else match context_uniform_propose F prog events d entry with
 |Some uentry=>uniform_entry_check F uniform prog events d uentry|None=>false end.
Lemma context_exact_entry_preserved:forall F uniform prog events d entry,
 context_entry_check F uniform prog events d entry=true ->
 context_check_entry F uniform prog events d entry=true.
Proof.
 intros F uniform prog events d entry Hcheck.
 assert(Heligible:context_exact_eligible F uniform entry=true).
 {unfold context_entry_check in Hcheck;cbv zeta in Hcheck;rewrite !andb_true_iff in Hcheck.
  destruct Hcheck as [[[[[[[[[[[[Hlf Hcontroller] Hearly] Hvc] Hvl] Hrc] Hrl] Htest] Hready] Hreadyc] Hreadyl] Hreadys] Hreadysrange].
  unfold context_exact_eligible. rewrite !andb_true_iff. split;[split|].
  - exact(context_value_form F uniform(entry_early d entry)(ct_counter d)(ce_counter entry)Hvc).
  - exact(context_value_form F uniform(entry_early d entry)(ct_limit d)(ce_limit entry)Hvl).
  - exact(context_value_form F uniform(entry_ready d entry)(ct_stride d)(ce_stride entry)Hreadys). }
 unfold context_check_entry. now rewrite Heligible.
Qed.
Definition context_check F uniform prog events shape :=
 match context_propose F prog events shape with
 |Some(d,entry)=>context_check_entry F uniform prog events d entry
 |None=>false end.
Definition context_schedule_safe F uniform prog events : Prop :=
 exists word:env->list pc,
 (forall E,launch_ok F E->occurrence_trace F prog E events(word E)0(phase_initial E)) /\
 (forall E1 E2,context_initial_agree uniform E1 E2->word E1=word E2).
Theorem context_check_sound:forall F uniform prog events shape,
 context_check F uniform prog events shape=true ->context_schedule_safe F uniform prog events.
Proof.
 intros F uniform prog events shape Hcheck. unfold context_check in Hcheck.
 destruct(context_propose F prog events shape)as[[d entry]|]eqn:Hproposal;[|discriminate].
 unfold context_check_entry in Hcheck.
 destruct(context_exact_eligible F uniform entry)eqn:Hstrategy.
 - exists(fun E=>controller_word d(entry_context F E entry)). split.
   + intros E Hlaunch. eapply context_entry_sound;eassumption.
   + intros E1 E2 Hagree. now rewrite(context_entry_agree F uniform prog events d entry E1 E2 Hcheck Hagree).
 - destruct(context_uniform_propose F prog events d entry)as[uentry|]eqn:Hu;[|discriminate].
   first [discriminate |
    exists(fun E=>controller_word d(uniform_entry_context F E d uentry));split;
    [intros E Hlaunch;eapply uniform_entry_sound;eassumption|
     intros E1 E2 Hagree;now rewrite(uniform_entry_agree F uniform prog events d uentry E1 E2 Hcheck Hagree)]].
Qed.

(* Segment *)

(* The endpoint is an unexecuted boundary, including when it is a member
   of the complete event alphabet. Summaries must avoid endpoint and events. *)
Fixpoint occurrence_segment_with
 (summarize:pc->astate->occurrence_summary_result) (rounds fuel:nat)
 (keep:PM.t unit) (F:lfacts) (prog:program) (events:list pc)
 (endpoint pos:pc) (st:astate) : option(list pc*astate) :=
 match rounds with
 |O=>None
 |S rounds=>
 match occurrence_collect_with summarize fuel keep F prog (endpoint::events)
   (List.length prog,0%nat) [(pos,Some st)] with
 |Some(remaining,(event,first)::tail)=>
  let inv:=occurrence_join F first tail in
  if short_all(fun entry=>phase_event event(fst entry)&&state_le F(snd entry)inv)
    ((event,first)::tail)then
   if phase_event endpoint event then Some([],inv)
   else if occurrence_member events event then
    match remaining,fetch prog(fst event)(snd event)with
    |S remaining,Some i=>if phase_marker i && wf_instr F i then
      match occurrence_segment_with summarize rounds remaining keep F prog events endpoint
       (next_pos prog(fst event)(snd event))(phase_trim keep(transfer F inv i))with
      |Some(word,post)=>Some(event::word,post)|None=>None end
      else None
    |_,_=>None end
   else None
  else None
 | _=>None end
 end.

Definition occurrence_segment_result F prog E events endpoint post word t s :=
 exists n last,occurrence_observe F prog E events n t s=Some(word,last)/\
 loop_event_free_run F prog E [endpoint] n t s last /\
 pc_of last=endpoint /\state_ok F(n+t)%nat post(regs last)/\regs_ok F E(regs last).

Lemma occurrence_segment_later:forall F prog E events endpoint post word u v next,
 occurrence_member (endpoint::events)(pc_of v)=false ->
 phase_step F prog E u v=Some next ->
 occurrence_segment_result F prog E events endpoint post word(S u)next ->
 occurrence_segment_result F prog E events endpoint post word u v.
Proof.
 intros F prog E events endpoint post word u v next Hnot Hstep [n[last[Hobs[Havoid[Hpc[Hst Hrs]]]]]].
 assert(Hfree:occurrence_member events(pc_of v)=false).
 {unfold occurrence_member in Hnot. cbn[existsb]in Hnot.
  apply orb_false_iff in Hnot. exact(proj2 Hnot). }
 exists(S n),last. split.
 - cbn[occurrence_observe]. rewrite Hstep,Hobs,Hfree. reflexivity.
 - split.
   + econstructor;[|exact Hstep|exact Havoid].
     unfold occurrence_member in Hnot|-*. cbn[existsb]in Hnot|-*.
     apply orb_false_iff in Hnot. now rewrite(proj1 Hnot).
   + split;[exact Hpc|]. replace(S n+u)%nat with(n+S u)%nat by lia. auto.
Qed.

Theorem occurrence_segment_with_sound:forall F prog E events endpoint,
 env_ok F E ->forall summarize,
 occurrence_summary_ok F prog E(endpoint::events)summarize ->
 forall rounds fuel keep pos st word post,
 occurrence_segment_with summarize rounds fuel keep F prog events endpoint pos st=Some(word,post) ->
 forall t s,pos=pc_of s ->state_ok F t st(regs s)->regs_ok F E(regs s)->
 occurrence_segment_result F prog E events endpoint post word t s.
Proof.
 intros F prog E events endpoint HE summarize Hsummary.
 induction rounds as[|rounds IH];intros fuel keep pos st word post Hcheck t s Hpos Hst Hrs;
 [discriminate|]. cbn[occurrence_segment_with]in Hcheck.
 destruct(occurrence_collect_with summarize fuel keep F prog(endpoint::events)
  (List.length prog,0%nat)[(pos,Some st)])as[[remaining arrivals]|]eqn:Hcollect;[|discriminate].
 destruct arrivals as[|[event first]tail];[discriminate|].
 set(inv:=occurrence_join F first tail)in *.
 destruct(short_all(fun entry=>phase_event event(fst entry)&&state_le F(snd entry)inv)
  ((event,first)::tail))eqn:Hall;[|discriminate].
 rewrite short_all_same,forallb_forall in Hall.
 destruct(phase_event endpoint event)eqn:Hendpoint.
 - injection Hcheck as <- <-. apply phase_event_eq in Hendpoint.
   eapply occurrence_collect_with_continuation;[exact HE|exact Hsummary|exact Hcollect| | |left;reflexivity|exact Hpos|exact Hst|exact Hrs].
   + intros. eapply occurrence_segment_later;eassumption.
   + intros p a Hin u v Hp Ha Hr. specialize(Hall _ Hin).
     cbn[fst snd]in Hall. apply andb_true_iff in Hall. destruct Hall as[Hsame Hle].
     apply phase_event_eq in Hsame.
     exists 0%nat,v. split;[reflexivity|]. split;[constructor|]. split;[congruence|]. split;[|exact Hr].
     eapply state_le_sound;eassumption.
 - destruct(occurrence_member events event)eqn:Hevent;[|discriminate].
   destruct remaining as[|remaining];[discriminate|].
   destruct(fetch prog(fst event)(snd event))as[i|]eqn:Hfetch;[|discriminate].
   destruct(phase_marker i && wf_instr F i)eqn:Hmarker;[|discriminate].
   destruct(occurrence_segment_with summarize rounds remaining keep F prog events endpoint
    (next_pos prog(fst event)(snd event))(phase_trim keep(transfer F inv i)))
    as[[rest laststate]|]eqn:Hrest;[|discriminate].
   injection Hcheck as <- <-. apply andb_true_iff in Hmarker. destruct Hmarker as[Hmark Hwf].
   eapply occurrence_collect_with_continuation;[exact HE|exact Hsummary|exact Hcollect| | |left;reflexivity|exact Hpos|exact Hst|exact Hrs].
   + intros. eapply occurrence_segment_later;eassumption.
   + intros p a Hin u v Hp Ha Hr. specialize(Hall _ Hin).
     cbn[fst snd]in Hall. apply andb_true_iff in Hall. destruct Hall as[Hsame Hle].
     apply phase_event_eq in Hsame. subst p.
     assert(Hi:state_ok F u inv(regs v))by(eapply state_le_sound;eassumption).
     assert(Hf:fetch prog(blk v)(idx v)=Some i)by(rewrite Hp in Hfetch;exact Hfetch).
     assert(Hex:exists next,phase_step F prog E u v=Some next).
     {unfold phase_step,step. rewrite Hf. eexists;reflexivity. }
     destruct Hex as[next Hstep].
     destruct(transfer_sound F prog E HE u v next inv i Hf Hstep Hi Hr Hwf)as[Ho Hregs].
     assert(Hnext:pc_of next=next_pos prog(fst event)(snd event)).
     {pose proof(step_pos F prog E u v next i Hf Hstep)as H.
      destruct H as[[H _]|[[g[target[Hbad _]]]|[g[Hbad _]]]].
      - change(pc_of next=next_pos prog(fst(pc_of v))(snd(pc_of v)))in H.
        rewrite <-Hp in H. exact H.
      - exfalso. exact(proj1(phase_marker_not_control i Hmark)g target Hbad).
      - exfalso. exact(proj2(phase_marker_not_control i Hmark)g Hbad). }
     destruct(IH remaining keep(next_pos prog(fst event)(snd event))
      (phase_trim keep(transfer F inv i))rest laststate Hrest(S u)next
      (eq_sym Hnext)(phase_trim_sound F (S u) keep(transfer F inv i)(regs next)Ho)Hregs)
      as[n[last[Hobs[Havoid[Hpc[Hpost Hlast]]]]]].
     exists(S n),last. split.
     * cbn[occurrence_observe]. rewrite Hstep,Hobs,<-Hp,Hevent. reflexivity.
     * split.
       -- econstructor;[|exact Hstep|exact Havoid].
          unfold occurrence_member. cbn[existsb]. rewrite <-Hp,Hendpoint. reflexivity.
       -- split;[exact Hpc|]. replace(S n+u)%nat with(n+S u)%nat by lia. auto.
Qed.

(* SpanTotal *)
Definition span_cfg_next prog pos:=
 match fetch prog(fst pos)(snd pos)with
 |Some(IExit None)=>[(List.length prog,0%nat)]
 |Some(IExit(Some _))=>[(List.length prog,0%nat);next_pos prog(fst pos)(snd pos)]
 |_=>occurrence_cfg_next prog pos end.
Lemma span_cfg_complete:forall F prog E t s next i,
 fetch prog(blk s)(idx s)=Some i ->phase_step F prog E t s=Some next ->
 In(pc_of next)(span_cfg_next prog(pc_of s)).
Proof.
 intros F prog E t s next i Hfetch Hstep.
 pose proof(step_pos F prog E t s next i Hfetch Hstep)as Hpos.
 unfold span_cfg_next,occurrence_cfg_next,pc_of. cbn[fst snd]. rewrite Hfetch.
 destruct Hpos as[[Hnext[Hbra Hexit]]|[[g[target[Hi[Hg Hnext]]]]|[g[Hi[Hg Hnext]]]]].
 - rewrite Hnext. destruct i;cbn;auto.
   + destruct g as[[neg p]|];cbn;auto. specialize(Hbra None target eq_refl). discriminate.
   + destruct g as[[neg p]|];cbn;auto. specialize(Hexit None eq_refl). discriminate.
 - subst i. rewrite Hnext. destruct g;cbn;auto.
 - subst i. rewrite Hnext. destruct g;cbn;auto.
Qed.
Definition total_span_check F prog endpoint protected nodes:=
 short_all(fun pos=>match fetch prog(fst pos)(snd pos)with
 |None=>true
 |Some i=>
  short_all(fun r=>negb(is_cell F r)&&loop_preserves r i)protected &&
  short_all(fun child=>phase_event endpoint child||occurrence_member nodes child)
   (span_cfg_next prog pos)
 end)nodes.
Lemma total_span_step:forall F prog E endpoint protected nodes,
 env_ok F E ->total_span_check F prog endpoint protected nodes=true ->
 forall t s next,In(pc_of s)nodes ->phase_step F prog E t s=Some next ->
 (pc_of next=endpoint \/ In(pc_of next)nodes)/\
 (forall r,In r protected ->regs next r=regs s r).
Proof.
 intros F prog E endpoint protected nodes HE Hcheck t s next Hin Hstep.
 unfold total_span_check in Hcheck. rewrite short_all_same,forallb_forall in Hcheck.
 specialize(Hcheck _ Hin).
 destruct(fetch prog(fst(pc_of s))(snd(pc_of s)))as[i|]eqn:Hfetch.
 2:{unfold phase_step,step in Hstep. change(fetch prog(blk s)(idx s)=None)in Hfetch. rewrite Hfetch in Hstep. discriminate. }
 rewrite !andb_true_iff in Hcheck. destruct Hcheck as[Hkeep Hchildren].
 assert(Hf:fetch prog(blk s)(idx s)=Some i)by exact Hfetch.
 pose proof(span_cfg_complete F prog E t s next i Hf Hstep)as Hchild.
 rewrite short_all_same,forallb_forall in Hchildren,Hkeep.
 specialize(Hchildren _ Hchild). apply orb_true_iff in Hchildren.
 split.
 - destruct Hchildren as[Hend|Hnodes].
   + left. apply phase_event_eq in Hend. exact Hend.
   + right. now apply occurrence_member_in.
 - intros r Hr. specialize(Hkeep _ Hr). apply andb_true_iff in Hkeep.
   destruct Hkeep as[Hcell Hpres]. apply negb_true_iff in Hcell.
   eapply loop_step_preserves;eassumption.
Qed.
Theorem total_span_preserves:forall F prog E endpoint protected nodes n t s last,
 env_ok F E ->total_span_check F prog endpoint protected nodes=true ->
 loop_event_free_run F prog E[endpoint]n t s last ->
 (pc_of s=endpoint \/ In(pc_of s)nodes) ->
 forall r,In r protected ->regs last r=regs s r.
Proof.
 intros F prog E endpoint protected nodes n t s last HE Hcheck Hrun.
 induction Hrun as[t s|n t s mid last Hfree Hstep Htail IH];intros Hstart r Hr;
 [reflexivity|].
 assert(Hin:In(pc_of s)nodes).
 {destruct Hstart as[Hend|Hin];[|exact Hin].
  unfold occurrence_member in Hfree. cbn[existsb]in Hfree.
  rewrite Hend in Hfree.
  assert(Heq:phase_event endpoint endpoint=true)by(apply phase_event_eq;reflexivity).
  rewrite Heq in Hfree. discriminate. }
 destruct(total_span_step F prog E endpoint protected nodes HE Hcheck t s mid Hin Hstep)as[Hnext Hpres].
 rewrite(IH Hnext r Hr). exact(Hpres r Hr).
Qed.
Corollary occurrence_segment_total_protected:forall F prog E events endpoint summarize rounds fuel keep pos st word post protected nodes,
 env_ok F E ->occurrence_summary_ok F prog E(endpoint::events)summarize ->
 occurrence_segment_with summarize rounds fuel keep F prog events endpoint pos st=Some(word,post) ->
 total_span_check F prog endpoint protected nodes=true ->In pos nodes ->
 forall t s,pos=pc_of s ->state_ok F t st(regs s)->regs_ok F E(regs s)->
 exists n last,occurrence_observe F prog E events n t s=Some(word,last)/\
 pc_of last=endpoint /\state_ok F(n+t)%nat post(regs last)/\regs_ok F E(regs last)/\
 forall r,In r protected->regs last r=regs s r.
Proof.
 intros F prog E events endpoint summarize rounds fuel keep pos st word post protected nodes HE Hsummary Hcheck Hspan Hin t s Hpos Hst Hrs.
 destruct(occurrence_segment_with_sound F prog E events endpoint HE summarize Hsummary rounds fuel keep pos st word post Hcheck t s Hpos Hst Hrs)
 as[n[last[Hobs[Havoid[Hpc[Hpost Hregs]]]]]].
 exists n,last. split;[exact Hobs|]. split;[exact Hpc|].
 split;[exact Hpost|]. split;[exact Hregs|]. intros r Hr.
 eapply total_span_preserves;[exact HE|exact Hspan|exact Havoid| |exact Hr].
 right. now rewrite <-Hpos.
Qed.

(* A segment that starts at its endpoint takes no step. *)
Corollary occurrence_segment_total_protected_at:forall F prog E events endpoint summarize rounds fuel keep pos st word post protected nodes,
 env_ok F E ->occurrence_summary_ok F prog E(endpoint::events)summarize ->
 occurrence_segment_with summarize rounds fuel keep F prog events endpoint pos st=Some(word,post) ->
 total_span_check F prog endpoint protected nodes=true ->(In pos nodes \/ pos=endpoint) ->
 forall t s,pos=pc_of s ->state_ok F t st(regs s)->regs_ok F E(regs s)->
 exists n last,occurrence_observe F prog E events n t s=Some(word,last)/\
 pc_of last=endpoint /\state_ok F(n+t)%nat post(regs last)/\regs_ok F E(regs last)/\
 forall r,In r protected->regs last r=regs s r.
Proof.
 intros F prog E events endpoint summarize rounds fuel keep pos st word post protected nodes HE Hsummary Hcheck Hspan Hin t s Hpos Hst Hrs.
 destruct Hin as[Hin|Hat];[eapply occurrence_segment_total_protected;eassumption|].
 destruct(occurrence_segment_with_sound F prog E events endpoint HE summarize Hsummary rounds fuel keep pos st word post Hcheck t s Hpos Hst Hrs)
 as[n[last[Hobs[Havoid[Hpc[Hpost Hregs]]]]]].
 exists n,last. split;[exact Hobs|]. split;[exact Hpc|].
 split;[exact Hpost|]. split;[exact Hregs|]. intros r Hr.
 destruct Havoid as[t0 s0|n0 t0 s0 mid last0 Hfree _ _];[reflexivity|].
 exfalso. rewrite <-Hpos,Hat in Hfree. cbn[occurrence_member existsb]in Hfree.
 unfold phase_event in Hfree. rewrite !Nat.eqb_refl in Hfree. discriminate.
Qed.

(* Counter *)

Definition counted_add_check F prog events pos width counter stride:=
 Nat.eqb(wdn F counter)width &&
 loop_fetch_is prog pos(IBin None BAdd width counter counter(OImm stride)) &&
 wf_instr F(IBin None BAdd width counter counter(OImm stride)) &&
 negb(occurrence_member events pos).
Lemma counted_add_sound:forall F prog E events pos width counter stride t s,
 env_ok F E ->counted_add_check F prog events pos width counter stride=true ->
 pc_of s=pos ->regs_ok F E(regs s)->
 0<=regs s counter+stride<2^Z.of_nat width ->
 exists next,occurrence_observe F prog E events 1 t s=Some([],next)/\
 pc_of next=next_pos prog(fst pos)(snd pos)/\regs_ok F E(regs next)/\
 regs next counter=regs s counter+stride /\
 forall r,r<>counter ->regs next r=regs s r.
Proof.
 intros F prog E events pos width counter stride t s HE Hcheck Hpc Hrs Hsum.
 unfold counted_add_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[Hw Hfetch]Hwf]Hfree]. apply Nat.eqb_eq in Hw.
 apply loop_fetch_is_sound in Hfetch. apply negb_true_iff in Hfree.
 set(next:=goto(next_pos prog(fst pos)(snd pos))
  (upd(wdn F)(regs s)counter(regs s counter+stride))).
 assert(Hnextpc:pc_of next=next_pos prog(fst pos)(snd pos)).
 {unfold next,pc_of,goto. destruct(next_pos prog(fst pos)(snd pos));reflexivity. }
 assert(Hnextregs:regs next=upd(wdn F)(regs s)counter(regs s counter+stride)).
 {unfold next,goto. destruct(next_pos prog(fst pos)(snd pos));reflexivity. }
 assert(Hf:fetch prog(blk s)(idx s)=Some(IBin None BAdd width counter counter(OImm stride))).
 {rewrite <-Hpc in Hfetch. exact Hfetch. }
 assert(Hstep:phase_step F prog E t s=Some next).
 {unfold phase_step,step. rewrite Hf. cbn[guard_of holds dest result binop_eval opv]. unfold next. rewrite <-Hpc. reflexivity. }
 exists next. split.
 - cbn[occurrence_observe]. rewrite Hstep,Hpc,Hfree. reflexivity.
 - split;[exact Hnextpc|]. split.
   + eapply context_step_regs;eassumption.
   + split.
     * rewrite Hnextregs. unfold upd. rewrite Pos.eqb_refl,Hw. apply trunc_small. exact Hsum.
     * intros r Hr. rewrite Hnextregs. unfold upd. apply Pos.eqb_neq in Hr. now rewrite Hr.
Qed.

(* The loop's test: the counter below the limit, or, for a step of 1, the
   counter different from the limit in either operand order. *)
Inductive counted_cmp:=Counted_lt|Counted_ne(swap signed:bool).
Definition counted_test_instr k width predicate counter limit:instr:=
 match k with
 |Counted_lt=>ISetp None CLt width true predicate counter(OReg limit)
 |Counted_ne swap signed=>if swap then ISetp None CNe width signed predicate limit(OReg counter)
                          else ISetp None CNe width signed predicate counter(OReg limit)
 end.
Definition counted_test_check F prog events k test branch width counter limit predicate head:=
 (1<=?Z.of_nat width)&&(Z.of_nat width<=?64)&&
 Nat.eqb(wdn F predicate)1&&
 negb(Pos.eqb predicate counter)&&negb(Pos.eqb predicate limit)&&
 loop_fetch_is prog test(counted_test_instr k width predicate counter limit)&&
 wf_instr F(counted_test_instr k width predicate counter limit)&&
 negb(occurrence_member events test)&&
 phase_event branch(next_pos prog(fst test)(snd test))&&
 loop_fetch_is prog branch(IBra(Some(false,predicate))head)&&
 negb(occurrence_member events branch).
(* A counter at most the limit differs from it exactly when it is below it. *)
Definition counted_kind_ok k(c l:Z):Prop:=match k with Counted_lt=>True|Counted_ne _ _=>c<=l end.
Lemma counted_extend_small:forall w sg v,1<=Z.of_nat w ->0<=v<2^(Z.of_nat w-1)->extend w sg v=v.
Proof.
 intros w sg v Hw Hv.
 assert(Hpow:2^(Z.of_nat w)=2*2^(Z.of_nat w-1)).
 {replace(Z.of_nat w)with(Z.succ(Z.of_nat w-1))at 1 by lia. apply Z.pow_succ_r;lia. }
 destruct sg;cbn[extend].
 - apply sext_small;lia.
 - apply GateSoundRange.trunc_small;lia.
Qed.
Lemma counted_test_value:forall k width predicate counter limit mem arb shm t rs,
 1<=Z.of_nat width ->0<=rs counter<2^(Z.of_nat width-1)->0<=rs limit<2^(Z.of_nat width-1)->
 counted_kind_ok k(rs counter)(rs limit)->
 result mem arb shm t rs(counted_test_instr k width predicate counter limit)=bool_z(rs counter<?rs limit).
Proof.
 intros k width predicate counter limit mem arb shm t rs Hw Hc Hl Hk.
 destruct k as[|swap sg];cbn[counted_test_instr].
 - cbn[result opv cmp_eval]. rewrite !(counted_extend_small width true)by lia. reflexivity.
 - cbn[counted_kind_ok]in Hk.
   destruct swap;cbn[result opv cmp_eval];rewrite !(counted_extend_small width sg)by lia;f_equal;
   destruct(Z.eqb_spec(rs limit)(rs counter)),(Z.eqb_spec(rs counter)(rs limit)),(Z.ltb_spec(rs counter)(rs limit));
   cbn;lia||reflexivity.
Qed.
Lemma counted_test_sound:forall F prog E events k test branch width counter limit predicate head t s,
 env_ok F E ->counted_test_check F prog events k test branch width counter limit predicate head=true ->
 pc_of s=test ->regs_ok F E(regs s)->
 0<=regs s counter<2^(Z.of_nat width-1)->0<=regs s limit<2^(Z.of_nat width-1)->
 counted_kind_ok k(regs s counter)(regs s limit)->
 exists last,occurrence_observe F prog E events 2 t s=Some([],last)/\
 pc_of last=(if regs s counter<?regs s limit then(head,0%nat)else next_pos prog(fst branch)(snd branch))/\
 regs_ok F E(regs last)/\forall r,r<>predicate ->regs last r=regs s r.
Proof.
 intros F prog E events k test branch width counter limit predicate head t s HE Hcheck Hpc Hrs Hc Hl Hk.
 unfold counted_test_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[[[[[[[[Hlo Hhi]Hw]Hpcounter]Hplimit]Hfetch]Hwf]Hfree]Hnext]Hbranch]Hbranchfree].
 apply Nat.eqb_eq in Hw. apply phase_event_eq in Hnext. apply Z.leb_le in Hlo.
 apply loop_fetch_is_sound in Hfetch,Hbranch. apply negb_true_iff in Hfree,Hbranchfree.
 set(value:=bool_z(regs s counter<?regs s limit)).
 set(mid:=goto branch(upd(wdn F)(regs s)predicate value)).
 assert(Hmidpc:pc_of mid=branch)by(unfold mid,pc_of,goto;destruct branch;reflexivity).
 assert(Hmidregs:regs mid=upd(wdn F)(regs s)predicate value)by(unfold mid,goto;destruct branch;reflexivity).
 assert(Hf:fetch prog(blk s)(idx s)=Some(counted_test_instr k width predicate counter limit)).
 {rewrite <-Hpc in Hfetch. exact Hfetch. }
 assert(Hstep:phase_step F prog E t s=Some mid).
 {unfold phase_step,step. rewrite Hf.
  pose proof(counted_test_value k width predicate counter limit(e_mem E)(e_arb E)(e_shm E)t(regs s)Hlo Hc Hl Hk)as Hv.
  rewrite <-Hpc in Hnext. cbn[pc_of fst snd]in Hnext.
  destruct k as[|[] sg];cbn[counted_test_instr guard_of holds negb dest]in Hv|-*;
  rewrite Hv;unfold mid,value;rewrite <-Hnext;reflexivity. }
 assert(Hmidrs:regs_ok F E(regs mid)).
 {eapply context_step_regs;eassumption. }
 assert(Hpred:regs mid predicate=value).
 {rewrite Hmidregs. unfold upd. rewrite Pos.eqb_refl,Hw.
  unfold value. destruct(regs s counter<?regs s limit);reflexivity. }
 set(last:=goto(if regs s counter<?regs s limit then(head,0%nat)else next_pos prog(fst branch)(snd branch))(regs mid)).
 assert(Hlastregs:regs last=regs mid).
 {unfold last,goto. destruct(regs s counter<?regs s limit);[|destruct(next_pos prog(fst branch)(snd branch))];reflexivity. }
 assert(Hlastpc:pc_of last=(if regs s counter<?regs s limit then(head,0%nat)else next_pos prog(fst branch)(snd branch))).
 {unfold last,pc_of,goto. destruct(regs s counter<?regs s limit);[|destruct(next_pos prog(fst branch)(snd branch))];reflexivity. }
 assert(Hbstep:phase_step F prog E(S t)mid=Some last).
 {unfold phase_step,step. rewrite <-Hmidpc in Hbranch. cbn[pc_of fst snd]in Hbranch. rewrite Hbranch.
  cbn[guard_of holds]. rewrite Hpred. unfold last,value. rewrite <-Hmidpc.
  destruct(regs s counter<?regs s limit);reflexivity. }
 exists last. split.
 - cbn[occurrence_observe]. rewrite Hstep,Hbstep,Hpc,Hmidpc,Hfree,Hbranchfree. reflexivity.
 - split;[exact Hlastpc|]. split;[now rewrite Hlastregs|].
   intros r Hr. rewrite Hlastregs,Hmidregs. unfold upd. apply Pos.eqb_neq in Hr. now rewrite Hr.
Qed.

(* Straight-line instructions between the counter's update and its test:
   each keeps every protected register and is not an event. *)
Fixpoint counted_gap_check F prog events(fuel:nat)(pos test:pc)(protected:list reg):bool:=
 if phase_event test pos then true else
 match fuel with
 |O=>false
 |S rest=>match fetch prog(fst pos)(snd pos)with
   |Some i=>negb(occurrence_member events pos)&&negb(is_jump i)&&wf_instr F i&&
     short_all(fun r=>negb(is_cell F r)&&loop_preserves r i)protected&&
     counted_gap_check F prog events rest(next_pos prog(fst pos)(snd pos))test protected
   |None=>false end
 end.
Lemma counted_gap_sound:forall fuel F prog E events pos test protected t s,
 env_ok F E ->counted_gap_check F prog events fuel pos test protected=true ->
 pc_of s=pos ->regs_ok F E(regs s)->
 exists n next,occurrence_observe F prog E events n t s=Some([],next)/\pc_of next=test /\
 regs_ok F E(regs next)/\forall r,In r protected ->regs next r=regs s r.
Proof.
 induction fuel as[|fuel IH];intros F prog E events pos test protected t s HE Hcheck Hpc Hrs;
 cbn[counted_gap_check]in Hcheck;
 (destruct(phase_event test pos)eqn:Hat;
  [apply phase_event_eq in Hat;exists 0%nat,s;split;[reflexivity|];split;[congruence|];split;[exact Hrs|];
   intros;reflexivity|]).
 - discriminate.
 - destruct(fetch prog(fst pos)(snd pos))as[i|]eqn:Hf;[|discriminate].
   rewrite !andb_true_iff in Hcheck. destruct Hcheck as[[[[Hfree Hjump]Hwf]Hkeep]Hrest].
   apply negb_true_iff in Hfree,Hjump.
   assert(Hfs:fetch prog(blk s)(idx s)=Some i)by(rewrite <-Hpc in Hf;exact Hf).
   assert(Hex:exists next,phase_step F prog E t s=Some next).
   {unfold phase_step,step. rewrite Hfs. eexists;reflexivity. }
   destruct Hex as[next Hstep].
   assert(Hnpc:pc_of next=next_pos prog(fst pos)(snd pos)).
   {destruct(step_pos F prog E t s next i Hfs Hstep)as[[Hn _]|[[g[tg[Hi _]]]|[g[Hi _]]]];
    [unfold pc_of;rewrite Hn,<-Hpc;reflexivity|subst i;discriminate|subst i;discriminate]. }
   assert(Hnrs:regs_ok F E(regs next))by(eapply context_step_regs;eassumption).
   destruct(IH F prog E events(next_pos prog(fst pos)(snd pos))test protected(S t)next HE Hrest Hnpc Hnrs)
     as[n[last[Hobs[Hlast[Hlrs Hpres]]]]].
   exists(S n),last. split.
   + cbn[occurrence_observe]. rewrite Hstep,Hobs,Hpc,Hfree. reflexivity.
   + split;[exact Hlast|]. split;[exact Hlrs|]. intros r Hr. rewrite(Hpres r Hr).
     rewrite short_all_same,forallb_forall in Hkeep. specialize(Hkeep r Hr).
     apply andb_true_iff in Hkeep. destruct Hkeep as[Hcell Hp]. apply negb_true_iff in Hcell.
     exact(loop_step_preserves F prog E t s next i r Hcell Hfs Hstep Hp).
Qed.

(* Ranges *)
Definition counted_range_check F st r width lo hi:=
 (1<=?Z.of_nat width)&&(Z.of_nat width<=?64)&&Nat.eqb(wdn F r)width&&
 match cmp_range F st(OReg r)(Z.of_nat width)true with
 |Some(a,b)=>(0<=?a)&&(b<?2^(Z.of_nat width-1))&&(lo<=?a)&&(b<=?hi)
 |None=>false end.
Lemma counted_range_sound:forall F E st t rs r width lo hi,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->
 counted_range_check F st r width lo hi=true ->lo<=rs r<=hi.
Proof.
 intros F E st t rs r width lo hi HE Hrs Hst Hcheck.
 unfold counted_range_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[Hwlo Hwhi]Hw]Hbounds]. apply Z.leb_le in Hwlo,Hwhi. apply Nat.eqb_eq in Hw.
 destruct(cmp_range F st(OReg r)(Z.of_nat width)true)as[[a b]|]eqn:Hrange;[|discriminate].
 rewrite !andb_true_iff,!Z.leb_le,!Z.ltb_lt in Hbounds.
 assert(Hwidth:owidth F(OReg r)(Z.of_nat width)).
 {intros r' Hr'. injection Hr' as <-. unfold wd. now rewrite Hw. }
 pose proof(cmp_range_sound F E HE t rs Hrs st(OReg r)(Z.of_nat width)true a b Hst ltac:(lia)Hwidth Hrange)as Hv.
 unfold cval,extend in Hv. cbn[opv]in Hv. rewrite Nat2Z.id in Hv.
 pose proof(rs_range F E rs Hrs r)as Hraw. rewrite Hw in Hraw.
 unfold sext in Hv. rewrite GateSoundRange.trunc_small in Hv by exact Hraw.
 assert(Hpow:2^(Z.of_nat width)=2*2^(Z.of_nat width-1)).
 {replace(Z.of_nat width)with(Z.succ(Z.of_nat width-1))at 1 by lia. apply Z.pow_succ_r;lia. }
 destruct(Z.ltb_spec(rs r)(2^(Z.of_nat width-1)));lia.
Qed.
Definition counted_bounds_check F st bounds:=
 short_all(fun item=>let '(r,(lo,hi)):=item in counted_range_check F st r(wdn F r)lo hi)bounds.
Theorem counted_bounds_sound:forall F E st t rs bounds,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->counted_bounds_check F st bounds=true ->
 state_ok F t(context_range_state bounds)rs.
Proof.
 intros F E st t rs bounds HE Hrs Hst Hcheck. apply context_range_state_sound.
 unfold counted_bounds_check in Hcheck. rewrite short_all_same,forallb_forall in Hcheck.
 intros r lo hi Hin. specialize(Hcheck(r,(lo,hi))Hin).
 eapply(counted_range_sound F E st t rs r(wdn F r)lo hi);eassumption.
Qed.

(* Branch *)
Definition comparison_value rs k:=cmp_eval(cc k)(extend(cw k)(csg k)(rs(ca k)))(extend(cw k)(csg k)(opv rs(cb k))).
Lemma singleton_predicate_value:forall F t st rs p k,
 state_ok F t st rs ->get st p=Some(FPred[k]true)->
 negb(Z.eqb(rs p)0)=comparison_value rs k.
Proof.
 intros F t st rs p k Hst Hget.
 pose proof(proj1 Hst p(FPred[k]true)Hget)as Hfact.
 unfold fact_ok,fact_okv in Hfact. destruct Hfact as[Hyes Hno].
 destruct(Z.eqb_spec(rs p)0)as[Hz|Hz];cbn;destruct(comparison_value rs k)eqn:Hcmp;try reflexivity.
 - exfalso. apply(Hno eq_refl);[constructor;[exact Hcmp|constructor]|exact Hz].
 - specialize(Hyes Hz). inversion Hyes as[|? ? Hhead Htail];subst.
   unfold cmp_holds in Hhead. change(comparison_value rs k=true)in Hhead. congruence.
Qed.
Lemma observed_branch_refined:forall F prog E events pos st p neg target t s,
 env_ok F E ->fetch prog(fst pos)(snd pos)=Some(IBra(Some(neg,p))target)->
 wf_instr F(IBra(Some(neg,p))target)=true ->occurrence_member events pos=false ->
 pc_of s=pos ->state_ok F t st(regs s)->regs_ok F E(regs s)->
 exists next out,occurrence_observe F prog E events 1 t s=Some([],next)/\
 pc_of next=(if xorb neg(negb(Z.eqb(regs s p)0))then(target,0%nat)else next_pos prog(fst pos)(snd pos))/\
 regs next=regs s /\ regs_ok F E(regs next)/\
 refine F(transfer F st(IBra(Some(neg,p))target))p(negb(Z.eqb(regs s p)0))=Some out/\
 state_ok F(S t)out(regs next).
Proof.
 intros F prog E events pos st p neg target t s HE Hfetch Hwf Hfree Hpc Hst Hrs.
 set(next:=goto(if xorb neg(negb(Z.eqb(regs s p)0))then(target,0%nat)else next_pos prog(fst pos)(snd pos))(regs s)).
 assert(Hnextregs:regs next=regs s).
 {unfold next,goto. destruct(xorb neg(negb(Z.eqb(regs s p)0)));[|destruct(next_pos prog(fst pos)(snd pos))];reflexivity. }
 assert(Hnextpc:pc_of next=(if xorb neg(negb(Z.eqb(regs s p)0))then(target,0%nat)else next_pos prog(fst pos)(snd pos))).
 {unfold next,pc_of,goto. destruct(xorb neg(negb(Z.eqb(regs s p)0)));[|destruct(next_pos prog(fst pos)(snd pos))];reflexivity. }
 assert(Hf:fetch prog(blk s)(idx s)=Some(IBra(Some(neg,p))target))by(rewrite <-Hpc in Hfetch;exact Hfetch).
 assert(Hstep:phase_step F prog E t s=Some next).
 {unfold phase_step,step. rewrite Hf. cbn[guard_of holds]. unfold next. rewrite <-Hpc. destruct(xorb neg(negb(Z.eqb(regs s p)0)));reflexivity. }
 destruct(transfer_sound F prog E HE t s next st(IBra(Some(neg,p))target)Hf Hstep Hst Hrs Hwf)as[Hout Hr].
 destruct(refine_sound F E HE(S t)(regs next)Hr(transfer F st(IBra(Some(neg,p))target))p
  (negb(Z.eqb(regs s p)0))Hout ltac:(now rewrite Hnextregs))as[out[Hrefine Hfinal]].
 exists next,out. split;[cbn[occurrence_observe];now rewrite Hstep,Hpc,Hfree|].
 split;[exact Hnextpc|]. split;[exact Hnextregs|]. split;[exact Hr|]. split;assumption.
Qed.

(* MixedOuter *)
Record mixed_outer_plan := {
 mo_head:nat; mo_add:pc; mo_test:pc; mo_branch:pc; mo_cut:pc;
 mo_width:nat; mo_counter:reg; mo_limit:reg; mo_predicate:reg;
 mo_stride:Z; mo_kind:counted_cmp; mo_limit_max:Z; mo_aux:list(reg*(Z*Z));
 mo_nodes:list pc; mo_keep:PM.t unit; mo_rounds:nat; mo_fuel:nat
}.
Definition mixed_outer_bounds d:=
 (mo_counter d,(0,mo_limit_max d))::(mo_limit d,(1,mo_limit_max d))::mo_aux d.
Definition mixed_outer_protected d:=mo_counter d::mo_limit d::map fst(mo_aux d).
Definition mixed_outer_exit prog d:=next_pos prog(fst(mo_branch d))(snd(mo_branch d)).
Definition mixed_outer_shape_ok F prog events d:=
 (0<?mo_stride d)&&(1<=?mo_limit_max d)&&
 (mo_limit_max d+mo_stride d<?2^(Z.of_nat(mo_width d)-1))&&
 short_all(fun item=>negb(Pos.eqb(fst item)(mo_counter d))&&
                      negb(Pos.eqb(fst item)(mo_predicate d)))(mo_aux d)&&
 counted_add_check F prog events(mo_add d)(mo_width d)(mo_counter d)(mo_stride d)&&
 match mo_kind d with Counted_lt=>true|Counted_ne _ _=>Z.eqb(mo_stride d)1 end&&
 counted_gap_check F prog events(block_len prog(fst(mo_add d)))
   (next_pos prog(fst(mo_add d))(snd(mo_add d)))(mo_test d)(mixed_outer_protected d)&&
 counted_test_check F prog events(mo_kind d)(mo_test d)(mo_branch d)(mo_width d)
   (mo_counter d)(mo_limit d)(mo_predicate d)(mo_head d)&&
 total_span_check F prog(mo_add d)(mixed_outer_protected d)(mo_nodes d)&&
 occurrence_member(mo_nodes d)(mo_head d,0%nat)&&
 Nat.ltb(fst(mo_cut d))(List.length prog)&&phase_closed prog(mo_cut d)&&
 short_all(fun event=>negb(phase_after(mo_cut d)event))events&&
 phase_after(mo_cut d)(mixed_outer_exit prog d).
Definition mixed_outer_prepare F prog events d:=
 if mixed_outer_shape_ok F prog events d then
 occurrence_segment_with no_loop_summary(mo_rounds d)(mo_fuel d)(mo_keep d)
 F prog events(mo_add d)(mo_head d,0%nat)(context_range_state(mixed_outer_bounds d))
 else None.
Definition mixed_outer_invariant F prog d E q (t:nat) s:=
 env_ok F E /\regs_ok F E(regs s)/\cq_stride q=mo_stride d/\
 1<=cq_limit q<=mo_limit_max d/\
 if context_proceed q then
 pc_of s=(mo_head d,0%nat)/\0<=cq_counter q/\
 regs s(mo_counter d)=cq_counter q/\regs s(mo_limit d)=cq_limit q/\
 (forall r lo hi,In(r,(lo,hi))(mo_aux d)->lo<=regs s r<=hi)
 else pc_of s=mixed_outer_exit prog d.
Lemma mixed_outer_body:forall F prog events d word post,
 mixed_outer_prepare F prog events d=Some(word,post)->
 forall E q t s,mixed_outer_invariant F prog d E q t s->context_proceed q=true ->
 exists n last,occurrence_observe F prog E events n t s=Some(word,last)/\
 mixed_outer_invariant F prog d E(context_advance q)(n+t)%nat last.
Proof.
 intros F prog events d word post Hprepared E q t s[HE[Hrs[Hstride[Hlimit Hinv]]]]Hgo.
 unfold mixed_outer_prepare in Hprepared.
 destruct(mixed_outer_shape_ok F prog events d)eqn:Hshape;[|discriminate].
 unfold mixed_outer_shape_ok in Hshape. rewrite !andb_true_iff in Hshape.
 destruct Hshape as[[[[[[[[[[[[[Hstridepos Hmax]Hwrap]Haux]Haddcheck]Hkind]Hgap]Htestcheck]Hspan]Hnodes]Hcutbound]Hclosed]Hbefore]Hafter].
 apply Z.ltb_lt in Hstridepos,Hwrap. apply Z.leb_le in Hmax.
 apply occurrence_member_in in Hnodes.
 rewrite Hgo in Hinv. destruct Hinv as[Hpc[Hc[Hcv[Hlv Hbounds]]]].
 assert(Hlt:cq_counter q<cq_limit q).
 {unfold context_proceed in Hgo. rewrite andb_true_iff,!Z.ltb_lt in Hgo. tauto. }
 assert(Hentry:state_ok F t(context_range_state(mixed_outer_bounds d))(regs s)).
 {apply context_range_state_sound. intros r lo hi Hin. unfold mixed_outer_bounds in Hin.
  destruct Hin as[H|[H|H]];[injection H as <- <- <-;rewrite Hcv;lia|
  injection H as <- <- <-;rewrite Hlv;lia|exact(Hbounds r lo hi H)]. }
 destruct(occurrence_segment_total_protected F prog E events(mo_add d)no_loop_summary
  (mo_rounds d)(mo_fuel d)(mo_keep d)(mo_head d,0%nat)(context_range_state(mixed_outer_bounds d))
  word post(mixed_outer_protected d)(mo_nodes d)HE(no_loop_summary_ok F prog E(mo_add d::events))
  Hprepared Hspan Hnodes t s(eq_sym Hpc)Hentry Hrs)
 as[n[mid[Hobs[Hmidpc[Hmidpost[Hmidrs Hpres]]]]]].
 assert(Hmc:regs mid(mo_counter d)=cq_counter q).
 {rewrite Hpres;[exact Hcv|left;reflexivity]. }
 assert(Hml:regs mid(mo_limit d)=cq_limit q).
 {rewrite Hpres;[exact Hlv|right;left;reflexivity]. }
 assert(Hwidth:1<=Z.of_nat(mo_width d)).
 {unfold counted_test_check in Htestcheck. rewrite !andb_true_iff in Htestcheck.
  apply Z.leb_le. tauto. }
 assert(Hpow:2^(Z.of_nat(mo_width d))=2*2^(Z.of_nat(mo_width d)-1)).
 {replace(Z.of_nat(mo_width d))with(Z.succ(Z.of_nat(mo_width d)-1))at 1 by lia.
  apply Z.pow_succ_r;lia. }
 destruct(counted_add_sound F prog E events(mo_add d)(mo_width d)(mo_counter d)(mo_stride d)
 (n+t)%nat mid HE Haddcheck Hmidpc Hmidrs ltac:(rewrite Hmc;lia))
 as[inc[Hadd[Hincpc[Hincrs[Hinc Hsaved]]]]].
 assert(Hic:regs inc(mo_counter d)=cq_counter q+mo_stride d)by(now rewrite Hinc,Hmc).
 assert(Hdifferent:mo_limit d<>mo_counter d).
 {intro Heq. rewrite Heq,Hcv in Hlv. lia. }
 assert(Hil:regs inc(mo_limit d)=cq_limit q)by(rewrite Hsaved;[exact Hml|exact Hdifferent]).
 assert(Hincaux:forall r lo hi,In(r,(lo,hi))(mo_aux d)->regs inc r=regs mid r).
 {intros r lo hi Hr. rewrite short_all_same,forallb_forall in Haux.
  specialize(Haux(r,(lo,hi))Hr). cbn[fst]in Haux. rewrite andb_true_iff,!negb_true_iff,!Pos.eqb_neq in Haux.
  apply Hsaved. tauto. }
 destruct(counted_gap_sound _ F prog E events _(mo_test d)(mixed_outer_protected d)(S(n+t))inc HE Hgap Hincpc Hincrs)
  as[ng[gap[Hgapobs[Hgappc[Hgaprs Hgappres]]]]].
 assert(Hgc:regs gap(mo_counter d)=cq_counter q+mo_stride d).
 {rewrite Hgappres;[exact Hic|left;reflexivity]. }
 assert(Hgl:regs gap(mo_limit d)=cq_limit q).
 {rewrite Hgappres;[exact Hil|right;left;reflexivity]. }
 assert(Hgk:counted_kind_ok(mo_kind d)(regs gap(mo_counter d))(regs gap(mo_limit d))).
 {destruct(mo_kind d);cbn[counted_kind_ok];[exact I|]. apply Z.eqb_eq in Hkind. rewrite Hgc,Hgl. lia. }
 destruct(counted_test_sound F prog E events(mo_kind d)(mo_test d)(mo_branch d)(mo_width d)
 (mo_counter d)(mo_limit d)(mo_predicate d)(mo_head d)(ng+S(n+t))gap HE Htestcheck Hgappc Hgaprs
 ltac:(rewrite Hgc;lia)ltac:(rewrite Hgl;lia)Hgk)
 as[last[Htest[Hlastpc[Hlastrs Htestpres]]]].
 rewrite Hgc,Hgl in Hlastpc.
 pose proof(observed_append F prog E events ng(S(n+t))inc[]gap 2[]last Hgapobs Htest)as Hgt.
 cbn[List.app]in Hgt.
 pose proof(observed_append F prog E events 1(n+t)%nat mid []inc(ng+2)[]last Hadd Hgt)as Htail.
 cbn[List.app]in Htail.
 pose proof(observed_append F prog E events n t s word mid(1+(ng+2))[]last Hobs Htail)as Hwhole.
 rewrite app_nil_r in Hwhole. exists(n+(1+(ng+2)))%nat,last. split;[exact Hwhole|].
 unfold mixed_outer_invariant. split;[exact HE|]. split;[exact Hlastrs|].
 split;[exact Hstride|]. split;[exact Hlimit|].
 assert(Hnextgo:context_proceed(context_advance q)=(cq_counter q+mo_stride d<?cq_limit q)).
 {unfold context_proceed,context_advance;cbn. rewrite Hstride.
  assert((0<?mo_stride d)=true)by(apply Z.ltb_lt;exact Hstridepos). now rewrite H. }
 rewrite Hnextgo. cbn[context_advance cq_counter cq_limit]. rewrite Hstride.
 destruct(cq_counter q+mo_stride d<?cq_limit q)eqn:Hmore;[|exact Hlastpc].
 split;[exact Hlastpc|]. split;[lia|].
 assert(Hpcounter:mo_counter d<>mo_predicate d).
 {unfold counted_test_check in Htestcheck. rewrite !andb_true_iff in Htestcheck.
  assert(negb(Pos.eqb(mo_predicate d)(mo_counter d))=true)by tauto.
  apply negb_true_iff,Pos.eqb_neq in H. congruence. }
 assert(Hplimit:mo_limit d<>mo_predicate d).
 {unfold counted_test_check in Htestcheck. rewrite !andb_true_iff in Htestcheck.
  assert(negb(Pos.eqb(mo_predicate d)(mo_limit d))=true)by tauto.
  apply negb_true_iff,Pos.eqb_neq in H. congruence. }
 split;[rewrite Htestpres by exact Hpcounter;exact Hgc|].
 split;[rewrite Htestpres by exact Hplimit;exact Hgl|].
 intros r lo hi Hr.
 assert(Hprot:In r(mixed_outer_protected d)).
 {unfold mixed_outer_protected. right;right. apply in_map with(f:=@fst reg(Z*Z))in Hr. exact Hr. }
 rewrite short_all_same,forallb_forall in Haux.
 specialize(Haux(r,(lo,hi))Hr). cbn[fst]in Haux. rewrite andb_true_iff,!negb_true_iff,!Pos.eqb_neq in Haux.
 rewrite Htestpres by tauto. rewrite(Hgappres r Hprot). rewrite Hsaved by tauto. rewrite Hpres.
 - exact(Hbounds r lo hi Hr).
 - unfold mixed_outer_protected. right;right. apply in_map with(f:=@fst reg(Z*Z))in Hr. exact Hr.
Qed.
Lemma mixed_outer_done:forall F prog events d word post,
 mixed_outer_prepare F prog events d=Some(word,post)->
 forall E q t s,mixed_outer_invariant F prog d E q t s->context_proceed q=false ->
 occurrence_trace F prog E events [] t s.
Proof.
 intros F prog events d word post Hprepared E q t s[HE[Hrs[Hstride[Hlimit Hinv]]]]Hgo.
 unfold mixed_outer_prepare in Hprepared. destruct(mixed_outer_shape_ok F prog events d)eqn:Hshape;[|discriminate].
 unfold mixed_outer_shape_ok in Hshape. rewrite !andb_true_iff in Hshape.
 destruct Hshape as[[[[[[[[[[[[[Hs Hm]Hw]Ha]Hadd]Hkind]Hgap]Htest]Hspan]Hnodes]Hcut]Hclosed]Hbefore]Hafter].
 rewrite Hgo in Hinv. eapply(occurrence_cut_trace F prog E events(mo_cut d)t s).
 - now apply Nat.ltb_lt in Hcut.
 - exact Hclosed.
 - rewrite short_all_same,forallb_forall in Hbefore. intros e Hin. apply negb_true_iff. exact(Hbefore e Hin).
 - now rewrite Hinv.
Qed.
Definition mixed_outer_word body q:=context_word body context_count
 context_proceed context_advance(S(context_rank q))q.
Theorem mixed_outer_pair:forall F prog events d word post,
 mixed_outer_prepare F prog events d=Some(word,post)->
 forall q E1 t1 s1 E2 t2 s2,
 mixed_outer_invariant F prog d E1 q t1 s1 ->mixed_outer_invariant F prog d E2 q t2 s2 ->
 occurrence_trace F prog E1 events(mixed_outer_word word q)t1 s1 /\
 occurrence_trace F prog E2 events(mixed_outer_word word q)t2 s2.
Proof.
 intros F prog events d word post Hprepared q E1 t1 s1 E2 t2 s2 H1 H2.
 eapply(context_pair_complete F prog events word context_count
 context_proceed context_advance context_rank(mixed_outer_invariant F prog d)
 controller_rank_decreases(mixed_outer_body F prog events d word post Hprepared)
 (mixed_outer_done F prog events d word post Hprepared));eassumption.
Qed.

(* MixedEntry *)
Record mixed_entry_plan:={
 me_header:list instr; me_branch:pc; me_predicate:reg; me_negated:bool; me_skip:nat;
 me_ready:list instr; me_ready_end:pc; me_bounds:list(reg*(Z*Z));
 me_nodes:list pc; me_keep:PM.t unit; me_rounds:nat; me_fuel:nat
}.
Definition mixed_entry_branch e:=IBra(Some(me_negated e,me_predicate e))(me_skip e).
Definition mixed_entry_header F e:=uniform_states_run F(me_header e)(empty_state F).
Definition mixed_entry_header_marks F uniform e:=uniform_marks_run F(me_header e)uniform.
Definition mixed_entry_ready_marks F uniform e:=
 uniform_marks_run F(me_ready e)(mixed_entry_header_marks F uniform e).
Definition mixed_entry_header_values F E e:=uniform_values_run F(me_header e)(e_init E).
Definition mixed_entry_ready_values F E e:=
 uniform_values_run F(me_ready e)(mixed_entry_header_values F E e).
Definition mixed_comparison_marked marks k:=
 marks(ca k)&&match cb k with OReg r=>marks r|OImm _=>true end.
Definition mixed_entry_final_bounds d:=
 (mo_counter d,(0,0))::(mo_limit d,(1,mo_limit_max d))::mo_aux d.
Record mixed_entry_evidence:={
 mv_comparison:cmpf;mv_false:astate;mv_ready:astate;
 mv_prefix_word:list pc;mv_prefix_post:astate;mv_body_word:list pc;mv_body_post:astate
}.
Definition mixed_entry_prepare F uniform prog events d e:option mixed_entry_evidence:=
 let header:=mixed_entry_header F e in
 match get header(me_predicate e)with
 |Some(FPred[k]true)=>
  if lf_ok F&&
   uniform_prefix_check(me_fuel e)F prog events(me_branch e)(0%nat,0%nat)(me_header e)&&
   mixed_comparison_marked(mixed_entry_header_marks F uniform e)k&&
   loop_fetch_is prog(me_branch e)(mixed_entry_branch e)&&wf_instr F(mixed_entry_branch e)&&
   negb(occurrence_member events(me_branch e))&&phase_after(mo_cut d)(me_skip e,0%nat)
  then match refine F(transfer F header(mixed_entry_branch e))(me_predicate e)(me_negated e)with
  |Some entry=>let ready:=uniform_states_run F(me_ready e)entry in
   if uniform_prefix_check(me_fuel e)F prog events(me_ready_end e)
       (next_pos prog(fst(me_branch e))(snd(me_branch e)))(me_ready e)&&
     mixed_entry_ready_marks F uniform e(mo_limit d)&&counted_bounds_check F ready(me_bounds e)&&
     total_span_check F prog(mo_head d,0%nat)[mo_limit d](me_nodes e)&&
     (occurrence_member(me_nodes e)(me_ready_end e)||phase_event(mo_head d,0%nat)(me_ready_end e))
   then match occurrence_segment_with
     (compound_occurrence_summary F prog((mo_head d,0%nat)::events))(me_rounds e)(me_fuel e)(me_keep e)
     F prog events(mo_head d,0%nat)(me_ready_end e)(context_range_state(me_bounds e))with
   |Some(prefix,post)=>if counted_bounds_check F post(mixed_entry_final_bounds d)then
     match mixed_outer_prepare F prog events d with
     |Some(body,bodypost)=>Some{|mv_comparison:=k;mv_false:=entry;mv_ready:=ready;
       mv_prefix_word:=prefix;mv_prefix_post:=post;mv_body_word:=body;mv_body_post:=bodypost|}
     |None=>None end else None
   |None=>None end else None
  |None=>None end else None
 |_=>None end.
Definition mixed_entry_check F uniform prog events d e:=
 match mixed_entry_prepare F uniform prog events d e with Some _=>true|None=>false end.
Definition mixed_entry_count F E d e:context_count:={|
 cq_counter:=0;cq_stride:=mo_stride d;cq_limit:=mixed_entry_ready_values F E e(mo_limit d)|}.
Definition mixed_entry_word F E d e v:=
 if xorb(me_negated e)(comparison_value(mixed_entry_header_values F E e)(mv_comparison v))then[]else
 mv_prefix_word v++mixed_outer_word(mv_body_word v)(mixed_entry_count F E d e).
Lemma mixed_comparison_agree:forall marks rs values k,
 uniform_agree marks rs values ->mixed_comparison_marked marks k=true ->
 comparison_value rs k=comparison_value values k.
Proof.
 intros marks rs values k Hagree Hmarked. unfold mixed_comparison_marked in Hmarked.
 apply andb_true_iff in Hmarked. destruct Hmarked as[Ha Hb].
 unfold comparison_value. rewrite(Hagree _ Ha). destruct(cb k);cbn[opv];[now rewrite(Hagree _ Hb)|reflexivity].
Qed.
Definition mixed_entry_valid F uniform prog events d e v:Prop:=
 lf_ok F=true /\
 uniform_prefix_collect(me_fuel e)F prog events(me_branch e)(0%nat,0%nat)=Some(me_header e) /\
 get(mixed_entry_header F e)(me_predicate e)=Some(FPred[mv_comparison v]true) /\
 mixed_comparison_marked(mixed_entry_header_marks F uniform e)(mv_comparison v)=true /\
 fetch prog(fst(me_branch e))(snd(me_branch e))=Some(mixed_entry_branch e) /\
 wf_instr F(mixed_entry_branch e)=true /\occurrence_member events(me_branch e)=false /\
 phase_after(mo_cut d)(me_skip e,0%nat)=true /\
 refine F(transfer F(mixed_entry_header F e)(mixed_entry_branch e))(me_predicate e)(me_negated e)=Some(mv_false v) /\
 mv_ready v=uniform_states_run F(me_ready e)(mv_false v) /\
 uniform_prefix_collect(me_fuel e)F prog events(me_ready_end e)
   (next_pos prog(fst(me_branch e))(snd(me_branch e)))=Some(me_ready e) /\
 mixed_entry_ready_marks F uniform e(mo_limit d)=true /\counted_bounds_check F(mv_ready v)(me_bounds e)=true /\
 total_span_check F prog(mo_head d,0%nat)[mo_limit d](me_nodes e)=true /\
 (occurrence_member(me_nodes e)(me_ready_end e)||phase_event(mo_head d,0%nat)(me_ready_end e))=true /\
 occurrence_segment_with(compound_occurrence_summary F prog((mo_head d,0%nat)::events))
 (me_rounds e)(me_fuel e)(me_keep e)F prog events(mo_head d,0%nat)(me_ready_end e)
 (context_range_state(me_bounds e))=Some(mv_prefix_word v,mv_prefix_post v) /\
 counted_bounds_check F(mv_prefix_post v)(mixed_entry_final_bounds d)=true /\
 mixed_outer_prepare F prog events d=Some(mv_body_word v,mv_body_post v).
Lemma mixed_entry_prepare_valid:forall F uniform prog events d e v,
 mixed_entry_prepare F uniform prog events d e=Some v ->mixed_entry_valid F uniform prog events d e v.
Proof.
 intros F uniform prog events d e v H. unfold mixed_entry_prepare in H.
 destruct(get(mixed_entry_header F e)(me_predicate e))as[f|]eqn:Hget;[|discriminate].
 destruct f;try discriminate. destruct cs as[|k[|other rest]];try discriminate. destruct exact;[|discriminate].
 destruct(lf_ok F&&uniform_prefix_check(me_fuel e)F prog events(me_branch e)(0%nat,0%nat)(me_header e)&&
 mixed_comparison_marked(mixed_entry_header_marks F uniform e)k&&loop_fetch_is prog(me_branch e)(mixed_entry_branch e)&&
 wf_instr F(mixed_entry_branch e)&&negb(occurrence_member events(me_branch e))&&phase_after(mo_cut d)(me_skip e,0%nat))eqn:Hhead;[|discriminate].
 destruct(refine F(transfer F(mixed_entry_header F e)(mixed_entry_branch e))(me_predicate e)(me_negated e))as[entry|]eqn:Href;[|discriminate].
 destruct(uniform_prefix_check(me_fuel e)F prog events(me_ready_end e)
  (next_pos prog(fst(me_branch e))(snd(me_branch e)))(me_ready e)&&
  mixed_entry_ready_marks F uniform e(mo_limit d)&&counted_bounds_check F(uniform_states_run F(me_ready e)entry)(me_bounds e)&&
  total_span_check F prog(mo_head d,0%nat)[mo_limit d](me_nodes e)&&(occurrence_member(me_nodes e)(me_ready_end e)||phase_event(mo_head d,0%nat)(me_ready_end e)))eqn:Hready;[|discriminate].
 destruct(occurrence_segment_with(compound_occurrence_summary F prog((mo_head d,0%nat)::events))
  (me_rounds e)(me_fuel e)(me_keep e)F prog events(mo_head d,0%nat)(me_ready_end e)
  (context_range_state(me_bounds e)))as[[prefix post]|]eqn:Hprefix;[|discriminate].
 destruct(counted_bounds_check F post(mixed_entry_final_bounds d))eqn:Hpost;[|discriminate].
 destruct(mixed_outer_prepare F prog events d)as[[body bodypost]|]eqn:Hbody;[|discriminate].
 injection H as <-. unfold mixed_entry_valid;cbn[mv_comparison mv_false mv_ready mv_prefix_word mv_prefix_post mv_body_word mv_body_post].
 apply andb_true_iff in Hhead as[Hhead Hskip].
 apply andb_true_iff in Hhead as[Hhead Hfree].
 apply andb_true_iff in Hhead as[Hhead Hwf].
 apply andb_true_iff in Hhead as[Hhead Hfetch].
 apply andb_true_iff in Hhead as[Hhead Hmarked].
 apply andb_true_iff in Hhead as[Hlf Hheader].
 apply andb_true_iff in Hready as[Hready Hnodes].
 apply andb_true_iff in Hready as[Hready Hspan].
 apply andb_true_iff in Hready as[Hready Hbounds].
 apply andb_true_iff in Hready as[Hcode Hlimit].
 apply uniform_prefix_check_sound in Hheader,Hcode.
 apply loop_fetch_is_sound in Hfetch. apply negb_true_iff in Hfree.
 repeat split;assumption || reflexivity.
Qed.
Definition mixed_initial_count d limit:context_count:={|cq_counter:=0;cq_stride:=mo_stride d;cq_limit:=limit|}.
Lemma mixed_entry_prefix_to_outer:forall F uniform prog events d e v E t s limit,
 mixed_entry_valid F uniform prog events d e v ->env_ok F E ->
 pc_of s=me_ready_end e ->regs_ok F E(regs s)->state_ok F t(mv_ready v)(regs s)->
 regs s(mo_limit d)=limit ->
 exists n head,occurrence_observe F prog E events n t s=Some(mv_prefix_word v,head)/\
 mixed_outer_invariant F prog d E(mixed_initial_count d limit)(n+t)%nat head.
Proof.
 intros F uniform prog events d e v E t s limit Hv HE Hpc Hrs Hst HL.
 destruct Hv as[Hlf[Hheader[Hget[Hmarked[Hfetch[Hwf[Hfree[Hskip[Href[Hreadystate[Hready[Hlimitmark[Hbounds[Hspan[Hnodes[Hprefix[Hpost Hbody]]]]]]]]]]]]]]]]].
 assert(Hnumeric:state_ok F t(context_range_state(me_bounds e))(regs s))
 by(eapply counted_bounds_sound;eassumption).
 assert(Hnodes':In(me_ready_end e)(me_nodes e)\/me_ready_end e=(mo_head d,0%nat)).
 {apply orb_true_iff in Hnodes. destruct Hnodes as[H|H];[left;now apply occurrence_member_in|right;now apply phase_event_eq]. }
 destruct(occurrence_segment_total_protected_at F prog E events(mo_head d,0%nat)
 (compound_occurrence_summary F prog((mo_head d,0%nat)::events))(me_rounds e)(me_fuel e)(me_keep e)
 (me_ready_end e)(context_range_state(me_bounds e))(mv_prefix_word v)(mv_prefix_post v)[mo_limit d](me_nodes e)
 HE(compound_occurrence_summary_sound F prog E((mo_head d,0%nat)::events)HE)
 Hprefix Hspan Hnodes' t s(eq_sym Hpc)Hnumeric Hrs)
 as[n[head[Hobs[Hheadpc[Hheadst[Hheadrs Hpres]]]]]].
 assert(HLfinal:regs head(mo_limit d)=limit)by(rewrite Hpres;[exact HL|left;reflexivity]).
 assert(Hrange:forall r lo hi,In(r,(lo,hi))(mixed_entry_final_bounds d)->lo<=regs head r<=hi).
 {intros r lo hi Hin. unfold counted_bounds_check in Hpost. rewrite short_all_same,forallb_forall in Hpost.
  specialize(Hpost(r,(lo,hi))Hin). eapply(counted_range_sound F E(mv_prefix_post v)(n+t)%nat(regs head)r(wdn F r)lo hi);eassumption. }
 assert(Hcounter:regs head(mo_counter d)=0).
 {pose proof(Hrange(mo_counter d)0 0(or_introl eq_refl));lia. }
 assert(Hlimit:1<=limit<=mo_limit_max d).
 {rewrite <-HLfinal. apply Hrange. right;left;reflexivity. }
 assert(Hstride:0<mo_stride d).
 {unfold mixed_outer_prepare in Hbody. destruct(mixed_outer_shape_ok F prog events d)eqn:Hshape;[|discriminate].
  unfold mixed_outer_shape_ok in Hshape. rewrite !andb_true_iff in Hshape. apply Z.ltb_lt;tauto. }
 exists n,head. split;[exact Hobs|]. unfold mixed_outer_invariant.
 split;[exact HE|]. split;[exact Hheadrs|]. split;[reflexivity|]. split;[exact Hlimit|].
 assert(Hgo:context_proceed(mixed_initial_count d limit)=true).
 {unfold context_proceed,mixed_initial_count;cbn[cq_counter cq_stride cq_limit].
  apply andb_true_iff;split;apply Z.ltb_lt;lia. }
 rewrite Hgo. cbn[mixed_initial_count cq_counter cq_limit].
 split;[exact Hheadpc|]. split;[lia|]. split;[exact Hcounter|]. split;[exact HLfinal|].
 intros r lo hi Hin. apply Hrange. right;right;exact Hin.
Qed.
Lemma mixed_entry_skip_trace:forall F prog events d body post e E t s,
 mixed_outer_prepare F prog events d=Some(body,post)->
 phase_after(mo_cut d)(me_skip e,0%nat)=true ->pc_of s=(me_skip e,0%nat) ->
 occurrence_trace F prog E events[]t s.
Proof.
 intros F prog events d body post e E t s Hbody Hafter Hpc.
 unfold mixed_outer_prepare in Hbody. destruct(mixed_outer_shape_ok F prog events d)eqn:Hshape;[|discriminate].
 unfold mixed_outer_shape_ok in Hshape. rewrite !andb_true_iff in Hshape.
 eapply(occurrence_cut_trace F prog E events(mo_cut d)t s).
 - apply Nat.ltb_lt;tauto.
 - tauto.
 - assert(Hbefore:short_all(fun event=>negb(phase_after(mo_cut d)event))events=true)by tauto.
   rewrite short_all_same,forallb_forall in Hbefore. intros event Hin. apply negb_true_iff. exact(Hbefore event Hin).
 - now rewrite Hpc.
Qed.
Lemma mixed_entry_trace:forall F uniform prog events d e v,
 mixed_entry_valid F uniform prog events d e v ->forall E,launch_ok F E ->
 occurrence_trace F prog E events(mixed_entry_word F E d e v)0(phase_initial E).
Proof.
 intros F uniform prog events d e v Hv E Hlaunch. pose proof Hv as Hparts.
 destruct Hparts as[Hlf[Hheader[Hget[Hmarked[Hfetch[Hwf[Hfree[Hskip[Href[Hreadystate[Hready[Hlimitmark[Hbounds[Hspan[Hnodes[Hprefix[Hpost Hbody]]]]]]]]]]]]]]]]].
 assert(HE:env_ok F E)by(exact(env_of_launch F prog E Hlf Hlaunch)).
 assert(Hrs:regs_ok F E(regs(phase_initial E))).
 {split;[exact(lk_init F E Hlaunch)|]. intros;reflexivity. }
 assert(Hst:state_ok F 0(empty_state F)(regs(phase_initial E))).
 {split;[|exact(lk_derived F E Hlaunch)]. intros r f Hf. unfold get,empty_state in Hf;cbn[amap]in Hf.
  rewrite PM.gempty in Hf. discriminate. }
 destruct(uniform_prefix_full_sound(me_fuel e)F prog E events(me_branch e)(0%nat,0%nat)
  (me_header e)(empty_state F)uniform(e_init E)0(phase_initial E)HE Hheader eq_refl Hrs Hst
  ltac:(intros r Hr;reflexivity))as[atbranch[Hheaderobs[Hpc[Hrs1[Hst1 Hagree1]]]]].
 fold(mixed_entry_header F e)in Hst1.
 pose proof(singleton_predicate_value F(List.length(me_header e)+0)%nat(mixed_entry_header F e)(regs atbranch)
  (me_predicate e)(mv_comparison v)Hst1 Hget)as Hguard.
 rewrite(mixed_comparison_agree _ _ _ _ Hagree1 Hmarked)in Hguard.
 fold(mixed_entry_header_values F E e)in Hguard.
 destruct(observed_branch_refined F prog E events(me_branch e)(mixed_entry_header F e)(me_predicate e)(me_negated e)(me_skip e)
  (List.length(me_header e)+0)%nat atbranch HE Hfetch Hwf Hfree Hpc Hst1 Hrs1)
 as[next[out[Hbranch[Hnextpc[Hregs[Hrs2[Hrefine Hst2]]]]]]].
 rewrite Hguard in Hnextpc,Hrefine.
 fold(mixed_entry_branch e)in Hrefine.
 unfold mixed_entry_word.
 destruct(xorb(me_negated e)(comparison_value(mixed_entry_header_values F E e)(mv_comparison v)))eqn:Hchoice.
 -
   assert(Hdone:occurrence_trace F prog E events[](S(List.length(me_header e)+0))next)
   by(eapply mixed_entry_skip_trace;eassumption).
   apply(observed_prefix_trace F prog E events(List.length(me_header e))0(phase_initial E)[]atbranch[]Hheaderobs).
   apply(observed_prefix_trace F prog E events 1(List.length(me_header e)+0)%nat atbranch[]next[]Hbranch).
   exact Hdone.
 -
   assert(Hpol:comparison_value(mixed_entry_header_values F E e)(mv_comparison v)=me_negated e).
   {destruct(me_negated e), (comparison_value(mixed_entry_header_values F E e)(mv_comparison v));try discriminate;reflexivity. }
   rewrite Hpol,Href in Hrefine. injection Hrefine as <-.
   assert(Hagree2:uniform_agree(mixed_entry_header_marks F uniform e)(regs next)(mixed_entry_header_values F E e)).
   {rewrite Hregs. exact Hagree1. }
   destruct(uniform_prefix_full_sound(me_fuel e)F prog E events(me_ready_end e)
    (next_pos prog(fst(me_branch e))(snd(me_branch e)))(me_ready e)(mv_false v)
    (mixed_entry_header_marks F uniform e)(mixed_entry_header_values F E e)
    (S(List.length(me_header e)+0))next HE Hready(eq_sym Hnextpc)Hrs2 Hst2 Hagree2)
   as[preload[Hreadyobs[Hprepc[Hrs3[Hst3 Hagree3]]]]].
   rewrite <-Hreadystate in Hst3.
   assert(Hlimit:regs preload(mo_limit d)=mixed_entry_ready_values F E e(mo_limit d)).
   {apply Hagree3. exact Hlimitmark. }
   destruct(mixed_entry_prefix_to_outer F uniform prog events d e v E
    (List.length(me_ready e)+S(List.length(me_header e)+0))%nat preload
    (mixed_entry_ready_values F E e(mo_limit d))Hv HE Hprepc Hrs3 Hst3 Hlimit)
   as[n[head[Hprefixobs Hinv]]].
   pose proof(proj1(mixed_outer_pair F prog events d(mv_body_word v)(mv_body_post v)Hbody
    (mixed_initial_count d(mixed_entry_ready_values F E e(mo_limit d))) E _ head E _ head Hinv Hinv))as Htail.
   apply(observed_prefix_trace F prog E events(List.length(me_header e))0(phase_initial E)[]atbranch _ Hheaderobs).
   apply(observed_prefix_trace F prog E events 1(List.length(me_header e)+0)%nat atbranch[]next _ Hbranch).
   apply(observed_prefix_trace F prog E events(List.length(me_ready e))(S(List.length(me_header e)+0))next[]preload _ Hreadyobs).
   eapply observed_prefix_trace;[exact Hprefixobs|exact Htail].
Qed.
Lemma mixed_entry_word_agree:forall F uniform prog events d e v,
 mixed_entry_valid F uniform prog events d e v ->forall E1 E2,
 context_initial_agree uniform E1 E2 ->mixed_entry_word F E1 d e v=mixed_entry_word F E2 d e v.
Proof.
 intros F uniform prog events d e v Hv E1 E2 Hinitial.
 destruct Hv as[Hlf[Hheader[Hget[Hmarked[Hfetch[Hwf[Hfree[Hskip[Href[Hreadystate[Hready[Hlimitmark Hrest]]]]]]]]]]]].
 pose proof(uniform_values_run_agree(me_header e)F uniform(e_init E1)(e_init E2)Hinitial)as Hheaderagree.
 pose proof(mixed_comparison_agree _ _ _ _ Hheaderagree Hmarked)as Hchoice.
 pose proof(uniform_values_run_agree(me_ready e)F(mixed_entry_header_marks F uniform e)
  (mixed_entry_header_values F E1 e)(mixed_entry_header_values F E2 e)Hheaderagree)as Hreadyagree.
 assert(Hlimit:mixed_entry_ready_values F E1 e(mo_limit d)=mixed_entry_ready_values F E2 e(mo_limit d)).
 {apply Hreadyagree;exact Hlimitmark. }
 unfold mixed_entry_word,mixed_entry_count. fold(mixed_entry_header_values F E1 e)in Hchoice.
 fold(mixed_entry_header_values F E2 e)in Hchoice. now rewrite Hchoice,Hlimit.
Qed.
Theorem mixed_entry_check_sound:forall F uniform prog events d e,
 mixed_entry_check F uniform prog events d e=true ->context_schedule_safe F uniform prog events.
Proof.
 intros F uniform prog events d e H. unfold mixed_entry_check in H.
 destruct(mixed_entry_prepare F uniform prog events d e)as[v|]eqn:Hprepared;[|discriminate].
 apply mixed_entry_prepare_valid in Hprepared.
 exists(fun E=>mixed_entry_word F E d e v). split.
 - intros E Hlaunch. eapply mixed_entry_trace;eassumption.
 - intros E1 E2 Hagree. eapply mixed_entry_word_agree;eassumption.
Qed.

(* MixedDiscovery *)
Record mixed_loop_shape := {
 ml_head:nat; ml_add:pc; ml_test:pc; ml_branch:pc;
 ml_width:nat; ml_counter:reg; ml_limit:reg; ml_predicate:reg; ml_stride:Z; ml_kind:counted_cmp
}.
(* The test of a counter: below the limit, or different from it. *)
Definition counted_test_of(width:nat)(counter:reg)(i:instr):option(counted_cmp*reg*reg):=
 match i with
 |ISetp None CLt tw true predicate tested(OReg limit)=>
   if Nat.eqb width tw&&Pos.eqb counter tested then Some(Counted_lt,predicate,limit)else None
 |ISetp None CNe tw signed predicate a(OReg b)=>
   if Nat.eqb width tw then
     if Pos.eqb counter a then Some(Counted_ne false signed,predicate,b)
     else if Pos.eqb counter b then Some(Counted_ne true signed,predicate,a)else None
   else None
 |_=>None end.
(* The latch block ends with the test and the back edge; the instructions
   between the counter's update and the test do not write the counter. *)
Definition mixed_loop_at prog events pos i:option mixed_loop_shape:=
 match i with
 | IBin None BAdd width counter source(OImm stride)=>
   let body:=block_body prog(fst pos)in let len:=List.length body in
   let test:=(fst pos,len-2)%nat in let branch:=(fst pos,len-1)%nat in
   match nth_error body(len-1),nth_error body(len-2)with
   |Some(IBra(Some(false,p))head),Some ti=>
     match counted_test_of width counter ti with
     |Some(kind,predicate,limit)=>
       if Nat.ltb(snd pos)(len-2)&&Pos.eqb counter source&&Pos.eqb predicate p&&(0<?stride)&&
          match kind with Counted_lt=>true|Counted_ne _ _=>Z.eqb stride 1 end&&
          phase_after(head,0%nat)pos&&
          existsb(fun event=>negb(phase_after event(head,0%nat))&&phase_after event pos)events&&
          short_all(fun g=>loop_preserves counter g)(firstn(len-2-S(snd pos))(skipn(S(snd pos))body))
       then Some{|ml_head:=head;ml_add:=pos;ml_test:=test;ml_branch:=branch;
          ml_width:=width;ml_counter:=counter;ml_limit:=limit;ml_predicate:=predicate;
          ml_stride:=stride;ml_kind:=kind|}
       else None
     |None=>None end
   |_,_=>None end
 |_=>None end.
Definition mixed_loop_select prog events:=
 match flat_map(fun item=>match mixed_loop_at prog events(fst item)(snd item)with
  Some d=>[d]|None=>[]end)(context_items prog)with
 |[d]=>Some d|_=>None end.
Definition mixed_test_boundary prog pos i:=
 match i with
 |ISetp _ _ _ _ p _ _=>let next:=next_pos prog(fst pos)(snd pos)in
   match fetch prog(fst next)(snd next)with
   |Some(IBra(Some(_,q))_)=>Pos.eqb p q|_=>false end
 |_=>false end.
(* The scan stops at an event, at the loop's head stop, or, for the ready
   code, before a test that a branch reads. *)
Fixpoint mixed_prefix_scan stop_before_test(stop:pc)fuel prog events pos:option(pc*list instr):=
 match fuel with
 |O=>None
 |S rest=>match fetch prog(fst pos)(snd pos)with
  |None=>None
  |Some i=>if occurrence_member events pos || phase_event stop pos ||
              (stop_before_test && mixed_test_boundary prog pos i) then Some(pos,[])else
    match loop_next prog pos i with
    |None=>Some(pos,[])
    |Some next=>match mixed_prefix_scan stop_before_test stop rest prog events next with
      |Some(endpoint,code)=>Some(endpoint,i::code)|None=>None end
    end
  end
 end.
Record mixed_context_shape := {
 ms_loop:mixed_loop_shape;
 ms_header:list instr;ms_branch:pc;ms_predicate:reg;ms_negated:bool;ms_skip:nat;
 ms_ready:list instr;ms_ready_end:pc
}.
(* A context-count route is selected only when its limit has a source
   dependence on a read-only input. Literal/copy/constant-arithmetic limits
   retain the existing finite-word route, including the zero-iteration case.
   This source heuristic supplies no uniformity or numeric fact to validation. *)
Definition mixed_input_roots prog r:=
 negb(existsb(fun i=>existsb(Pos.eqb r)(loop_writes i))(List.concat prog)).
Definition mixed_input_step i (marks:reg->bool):reg->bool:=
 let keep r:=marks r&&loop_preserves r i in
 if uniform_pure i then match dest i with
 |Some d=>fun r=>if Pos.eqb r d then existsb marks(instruction_inputs i)else keep r
 |None=>keep end else keep.
Definition mixed_input_run code marks:=fold_left(fun m i=>mixed_input_step i m)code marks.
Definition mixed_limit_dependent prog header ready limit:=
 mixed_input_run ready(mixed_input_run header(mixed_input_roots prog))limit.
Definition mixed_context_shape_select prog events:option mixed_context_shape:=
 match mixed_loop_select prog events,
       mixed_prefix_scan false(List.length prog,0%nat)phase_fuel prog events(0%nat,0%nat)with
 |Some loop,Some(branch,header)=>
   match fetch prog(fst branch)(snd branch)with
   |Some(IBra(Some(neg,p))skip)=>
     if phase_after(ml_branch loop)(skip,0%nat)&&
       phase_after branch(ml_head loop,0%nat)then
       match mixed_prefix_scan true(ml_head loop,0%nat)phase_fuel prog events(next_pos prog(fst branch)(snd branch))with
       |Some(ready_end,ready)=>if mixed_limit_dependent prog header ready(ml_limit loop)
        then Some{|ms_loop:=loop;ms_header:=header;ms_branch:=branch;
          ms_predicate:=p;ms_negated:=neg;ms_skip:=skip;ms_ready:=ready;ms_ready_end:=ready_end|}else None
       |None=>None end
     else None
   |_=>None end
 |_,_=>None end.
Definition mixed_span_nodes prog first last:=
 map fst(filter(fun item=>negb(phase_after(fst item)first)&&phase_after(fst item)last)(context_items prog)).
Definition mixed_ranges F st (registers:list reg):list(reg*(Z*Z)):=
 flat_map(fun r=>match cmp_range F st(OReg r)(Z.of_nat(wdn F r))true with
 |Some(lo,hi)=>if counted_range_check F st r(wdn F r)lo hi then[(r,(lo,hi))]else[]
 |None=>[]end)registers.
Definition mixed_preserved_aux F prog loop nodes (bounds:list(reg*(Z*Z))):=
 filter(fun item=>let r:=fst item in
  negb(Pos.eqb r(ml_counter loop))&&negb(Pos.eqb r(ml_limit loop))&&
  negb(Pos.eqb r(ml_predicate loop))&&negb(is_cell F r)&&
  short_all(fun pos=>match fetch prog(fst pos)(snd pos)with
   |Some i=>loop_preserves r i|None=>false end)nodes)bounds.
(* Discovery supplies candidates only; the checked entry/body validators retain
   every actual path, event, uniformity, range and finite-progress obligation. *)

(* MixedProposal *)
Definition mixed_context_propose F prog events shape:
 option(mixed_outer_plan*list(reg*(Z*Z))*list pc):=
 let loop:=ms_loop shape in
 let header:=uniform_states_run F(ms_header shape)(empty_state F)in
 match refine F(transfer F header(IBra(Some(ms_negated shape,ms_predicate shape))(ms_skip shape)))
       (ms_predicate shape)(ms_negated shape)with
 |None=>None
 |Some entry=>
  let ready:=uniform_states_run F(ms_ready shape)entry in
  let keep:=phase_control_regs(List.concat prog)in
  let registers:=map fst(PM.elements keep)in
  let bounds:=mixed_ranges F ready registers in
  let head:=(ml_head loop,0%nat)in
  match cmp_range F ready(OReg(ml_limit loop))(Z.of_nat(ml_width loop))true,
        context_region_collect phase_fuel prog[head][ms_ready_end shape][][]with
  |Some(_,limit_max),Some(nodes,_)=>
   match occurrence_segment_with(compound_occurrence_summary F prog(head::events))
       phase_fuel phase_fuel keep F prog events head(ms_ready_end shape)(context_range_state bounds)with
   |None=>None
   |Some(_,post)=>
    let body_nodes:=mixed_span_nodes prog head(ml_add loop)in
    let aux:=mixed_preserved_aux F prog loop body_nodes(mixed_ranges F post registers)in
    Some({|mo_head:=ml_head loop;mo_add:=ml_add loop;mo_test:=ml_test loop;
       mo_branch:=ml_branch loop;mo_cut:=ml_branch loop;mo_width:=ml_width loop;
       mo_counter:=ml_counter loop;mo_limit:=ml_limit loop;mo_predicate:=ml_predicate loop;
       mo_stride:=ml_stride loop;mo_kind:=ml_kind loop;mo_limit_max:=limit_max;mo_aux:=aux;
       mo_nodes:=body_nodes;mo_keep:=keep;mo_rounds:=phase_fuel;mo_fuel:=phase_fuel|},
      bounds,(List.length prog,0%nat)::nodes)
   end
  |_,_=>None end
 end.

(* MixedTop *)
Definition mixed_entry_of_shape d shape bounds nodes:mixed_entry_plan:={|
 me_header:=ms_header shape;me_branch:=ms_branch shape;me_predicate:=ms_predicate shape;
 me_negated:=ms_negated shape;me_skip:=ms_skip shape;me_ready:=ms_ready shape;
 me_ready_end:=ms_ready_end shape;me_bounds:=bounds;me_nodes:=nodes;me_keep:=mo_keep d;
 me_rounds:=phase_fuel;me_fuel:=phase_fuel|}.
Definition mixed_context_check F uniform prog events shape:=
 match mixed_context_propose F prog events shape with
 |Some((d,bounds),nodes)=>mixed_entry_check F uniform prog events d(mixed_entry_of_shape d shape bounds nodes)
 |None=>false end.
Theorem mixed_context_check_sound:forall F uniform prog events shape,
 mixed_context_check F uniform prog events shape=true ->context_schedule_safe F uniform prog events.
Proof.
 intros F uniform prog events shape H. unfold mixed_context_check in H.
 destruct(mixed_context_propose F prog events shape)as[[[d bounds]nodes]|];[|discriminate].
 eapply mixed_entry_check_sound;eassumption.
Qed.

(* ------------------------------------------------------------------ *)
(* WarpExit: an exit before the warp collectives that every lane of a
   warp takes together.                                                *)

(* The entry block ends with a conditional branch that leaves before the
   collectives; whether the branch is taken is a function of the CTA's
   uniform values and of the warp index. A walk of the entry block keeps,
   for each register, the lowest bit from which its value agrees with a
   reference run whose thread index is the warp's first: the uniform
   values agree in every bit, and the thread index from bit 5 upward. Each
   way out of the branch then needs a complete word for each warp. *)

(* An instruction whose written value is a function of the registers it reads. *)
Definition warp_pure (i:instr):bool:=
 match i with
 | IMov None _ _ | IImm None _ _ | IBin None _ _ _ _ _ | IMad None _ _ _ _ _
 | IWide None _ _ _ _ _ | ICvt None _ _ _ _ _ | IMinMax None _ _ _ _ _ _
 | ISetp None _ _ _ _ _ _ | ISelp None _ _ _ _ _
 | IAndp None _ _ _ | IOrp None _ _ _ | IXorp None _ _ _ => true
 | _=>false end.

(* The reference run of pure code. *)
Definition warp_result(rs:reg->Z)(i:instr):Z:=result(fun _ _=>0)(fun _ _=>0)(fun _ _=>0)0 rs i.
Definition warp_value_step(F:lfacts)(i:instr)(rs:reg->Z):reg->Z:=
 match dest i with
 |Some d=>upd(wdn F)rs d(warp_result rs i)
 |None=>rs end.
Fixpoint warp_values_run(F:lfacts)(code:list instr)(rs:reg->Z):reg->Z:=
 match code with []=>rs|i::rest=>warp_values_run F rest(warp_value_step F i rs)end.

(* Register r at level a agrees with the reference run from bit a upward. *)
Definition warp_agree(L:reg->option nat)(x y:reg->Z):Prop:=
 forall r a,L r=Some a->x r/2^Z.of_nat a=y r/2^Z.of_nat a.

(* A register that agrees in every bit, and one that does and whose value
   in the actual run is a multiple of 2^a. *)
Definition warp_zero(L:reg->option nat)(q:reg):bool:=match L q with Some O=>true|_=>false end.
Definition warp_aligned(F:lfacts)(st:astate)(L:reg->option nat)(q:reg)(a:nat):bool:=
 warp_zero L q && Z.leb(Z.of_nat a)(reg_align F st q).
Definition warp_imm_aligned(z:Z)(a:nat):bool:=Z.eqb(z mod 2^Z.of_nat a)0.
(* x plus or minus an aligned y keeps the level of x. *)
Definition warp_offset_level(F:lfacts)(st:astate)(L:reg->option nat)(x y:reg):option nat:=
 match L x with Some a=>if warp_aligned F st L y a then Some a else None|None=>None end.
Definition warp_const_level(L:reg->option nat)(x:reg)(z:Z):option nat:=
 match L x with Some a=>if warp_imm_aligned z a then Some a else None|None=>None end.

(* The level of the register that i writes, from its inputs' levels and the
   actual run's state st before i. *)
Definition warp_level_of(F:lfacts)(st:astate)(L:reg->option nat)(i:instr):option nat:=
 if forallb(warp_zero L)(instruction_inputs i)then Some O else
 match i with
 | IMov None _ s=>L s
 | ICvt None _ from sg _ s=>
   match L s with Some a=>if negb sg || Nat.ltb a from then Some a else None|None=>None end
 | IBin None BAdd _ _ x (OReg y)=>
   match warp_offset_level F st L x y with Some a=>Some a|None=>warp_offset_level F st L y x end
 | IBin None BAdd _ _ x (OImm z)=>warp_const_level L x z
 | IBin None BSub _ _ x (OReg y)=>warp_offset_level F st L x y
 | IBin None BSub _ _ x (OImm z)=>warp_const_level L x z
 | IBin None BShr w _ x (OImm k)=>
   match L x with Some a=>Some(a-Z.to_nat(shamt w k))%nat|None=>None end
 | IBin None BShrs w _ x (OImm k)=>
   match L x with Some a=>if Nat.ltb a w then Some(a-Z.to_nat(shamt w k))%nat else None|None=>None end
 | IBin None BAnd w _ x (OImm m)=>
   match L x with Some a=>if warp_imm_aligned(trunc w m)a then Some O else Some a|None=>None end
 | _=>None end.

Definition warp_level_step(F:lfacts)(st:astate)(L:reg->option nat)(i:instr):reg->option nat:=
 match dest i with
 |Some d=>fun q=>if Pos.eqb q d then warp_level_of F st L i else L q
 |None=>L end.
(* The levels and the abstract state of the actual run after code. *)
Fixpoint warp_walk(F:lfacts)(code:list instr)(L:reg->option nat)(st:astate):(reg->option nat)*astate:=
 match code with []=>(L,st)|i::rest=>warp_walk F rest(warp_level_step F st L i)(transfer F st i)end.

(* The uniform registers agree in every bit, and the thread index register r
   from bit 5 upward. *)
Definition warp_initial_levels(uniform:reg->bool)(r:reg):reg->option nat:=
 fun q=>if uniform q then Some O else if Pos.eqb q r then Some 5%nat else None.
(* The reference values: the uniform registers' launch values, and the first
   thread index of warp w. *)
Definition warp_reference(uniform:reg->bool)(r:reg)(w:nat)(E:env):reg->Z:=
 fun q=>if uniform q then e_init E q else if Pos.eqb q r then 32*Z.of_nat w else 0.

(* The entry block: the code before its last instruction, the guard and
   target of that conditional branch, and the other blocks. *)
Definition warp_exit_split(prog:program):option(list instr*bool*reg*nat*program):=
 match prog with
 |body::rest=>match rev body with
   |IBra(Some(neg,p))tg::before=>Some(rev before,neg,p,tg,rest)
   |_=>None end
 |[]=>None end.

(* The blocks that a thread may reach from the given ones. *)
Fixpoint warp_reach(prog:program)(fuel:nat)(seen:list nat):list nat:=
 match fuel with
 |O=>seen
 |S f=>let next:=nodup Nat.eq_dec(seen++flat_map(fun b=>filter(fun t=>Nat.ltb t(List.length prog))
     (edge_targets prog b))seen)in
   if Nat.eqb(List.length next)(List.length seen)then seen else warp_reach prog f next end.

(* The structural route: a kernel without CTA barriers whose entry block
   ends with a conditional branch, and neither the entry block nor the blocks
   that the branch's target reaches hold an event. *)
Definition warp_exit_selected(prog:program)(events cta:list pc):bool:=
 match cta,warp_exit_split prog with
 |[],Some(_,_,_,tg,_)=>
   let after:=warp_reach prog(List.length prog)[tg]in
   negb(existsb(fun e=>existsb(Nat.eqb(fst e))(0%nat::after))events)
 |_,_=>false end.

(* Every branch targets a block after the entry block. *)
Definition warp_entry_closed(prog:program):bool:=
 short_all(fun i=>match i with IBra _ O=>false|_=>true end)(concat prog).

(* The words of warp w when the entry branch is taken and when it is not.
   The branch predicate must agree in every bit with the reference run. *)
Definition warp_exit_pair(F:lfacts)(uniform:reg->bool)(before:list instr)(p:reg)(tg:nat)
 (rest:program)(events:list pc)(r:reg)(w:nat):option(list pc*list pc):=
 let Fw:=warp_facts F r w in
 if lf_ok Fw && short_all warp_pure before && short_all(wf_instr Fw)before &&
    warp_zero(fst(warp_walk Fw before(warp_initial_levels uniform r)(empty_state Fw)))p
 then match scoped_complete_word Fw((before++[IBra None tg])::rest)events,
            scoped_complete_word Fw((before++[INop])::rest)events with
   |Some taken,Some fall=>Some(taken,fall)
   |_,_=>None end
 else None.

Definition warp_exit_check(F:lfacts)(uniform:reg->bool)(prog:program)(events cta:list pc)(r:reg)(count:nat):bool:=
 match warp_exit_split prog with
 |Some(before,_,p,tg,rest)=>
   warp_entry_closed prog &&
   let pairs:=map(warp_exit_pair F uniform before p tg rest events r)(seq 0 count)in
   match pairs with
   |Some(taken,_)::_=>
     let common:=event_projection cta taken in
     short_all(fun pair=>match pair with
       |Some(taken,fall)=>event_word_eqb(event_projection cta taken)common &&
          event_word_eqb(event_projection cta fall)common
       |None=>false end)pairs
   |_=>false end
 |None=>false end.

(* WarpExitArithmetic: bits from a upward through the operations of the walk. *)
Lemma warp_lv_div : forall a u, 0 <= a -> u / 2^a = Z.shiftr u a.
Proof. intros a u Ha. rewrite Z.shiftr_div_pow2 by exact Ha. reflexivity. Qed.

Lemma warp_lv_trunc : forall n a u v, 0 <= a -> u / 2^a = v / 2^a -> trunc n u / 2^a = trunc n v / 2^a.
Proof.
 intros n a u v Ha H. unfold trunc. rewrite <- !Z.land_ones by lia.
 rewrite !warp_lv_div by exact Ha. rewrite !Z.shiftr_land. rewrite <- !warp_lv_div by exact Ha.
 rewrite H. reflexivity.
Qed.

Lemma warp_lv_below : forall a k u, 0 <= a -> 0 <= u -> (u < 2^a * k <-> u / 2^a < k).
Proof.
 intros a k u Ha Hu. assert (HP : 0 < 2^a) by (apply Z.pow_pos_nonneg; lia).
 pose proof (Z.div_mod u (2^a) ltac:(lia)) as Hd. pose proof (Z.mod_pos_bound u (2^a) HP) as Hm.
 split; intros H; nia.
Qed.

Lemma warp_lv_sext : forall n a u v, (a < n)%nat -> u / 2^Z.of_nat a = v / 2^Z.of_nat a ->
 sext n u / 2^Z.of_nat a = sext n v / 2^Z.of_nat a.
Proof.
 intros n a u v Han H. pose proof (warp_lv_trunc n (Z.of_nat a) u v ltac:(lia) H) as Ht.
 unfold sext. set (tu := trunc n u) in *. set (tv := trunc n v) in *.
 assert (Hu : 0 <= tu) by apply trunc_range. assert (Hv : 0 <= tv) by apply trunc_range.
 assert (Hh : 2^(Z.of_nat n - 1) = 2^Z.of_nat a * 2^(Z.of_nat n - 1 - Z.of_nat a))
   by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
 assert (Hn : 2^Z.of_nat n = 2^(Z.of_nat n - Z.of_nat a) * 2^Z.of_nat a)
   by (rewrite <- Z.pow_add_r by lia; f_equal; lia).
 assert (Hc : (tu <? 2^(Z.of_nat n - 1)) = (tv <? 2^(Z.of_nat n - 1))).
 { rewrite Hh.
   pose proof (warp_lv_below (Z.of_nat a) (2^(Z.of_nat n - 1 - Z.of_nat a)) tu ltac:(lia) Hu) as E1.
   pose proof (warp_lv_below (Z.of_nat a) (2^(Z.of_nat n - 1 - Z.of_nat a)) tv ltac:(lia) Hv) as E2.
   rewrite Ht in E1.
   destruct (Z.ltb_spec tu (2^Z.of_nat a * 2^(Z.of_nat n - 1 - Z.of_nat a))) as [L1|L1],
     (Z.ltb_spec tv (2^Z.of_nat a * 2^(Z.of_nat n - 1 - Z.of_nat a))) as [L2|L2]; [reflexivity| | |reflexivity].
   - apply E1, E2 in L1. lia.
   - apply E2, E1 in L2. lia. }
 rewrite Hc. destruct (tv <? 2^(Z.of_nat n - 1)); [exact Ht|].
 rewrite Hn. replace (tu - 2^(Z.of_nat n - Z.of_nat a) * 2^Z.of_nat a) with (tu + (- 2^(Z.of_nat n - Z.of_nat a)) * 2^Z.of_nat a) by ring.
 replace (tv - 2^(Z.of_nat n - Z.of_nat a) * 2^Z.of_nat a) with (tv + (- 2^(Z.of_nat n - Z.of_nat a)) * 2^Z.of_nat a) by ring.
 assert (HP : 2^Z.of_nat a <> 0) by (apply Z.pow_nonzero; lia).
 rewrite !Z.div_add by exact HP. rewrite Ht. reflexivity.
Qed.

Lemma warp_lv_add : forall a u v1 v2, (2^a | u) -> 0 <= a -> v1 / 2^a = v2 / 2^a -> (u + v1) / 2^a = (u + v2) / 2^a.
Proof.
 intros a u v1 v2 [k ->] Ha H. assert (HP : 2^a <> 0) by (apply Z.pow_nonzero; lia).
 rewrite !Z.div_add_l by exact HP. rewrite H. reflexivity.
Qed.

Lemma warp_lv_sub : forall a u v1 v2, (2^a | u) -> 0 <= a -> v1 / 2^a = v2 / 2^a -> (v1 - u) / 2^a = (v2 - u) / 2^a.
Proof.
 intros a u v1 v2 [k ->] Ha H. assert (HP : 2^a <> 0) by (apply Z.pow_nonzero; lia).
 replace (v1 - k * 2^a) with (v1 + (-k) * 2^a) by ring. replace (v2 - k * 2^a) with (v2 + (-k) * 2^a) by ring.
 rewrite !Z.div_add by exact HP. rewrite H. reflexivity.
Qed.

(* x / 2^s, at the level a - s of a value at level a. *)
Lemma warp_lv_shift : forall a s x1 x2, 0 <= s -> x1 / 2^Z.of_nat a = x2 / 2^Z.of_nat a ->
 x1 / 2^s / 2^Z.of_nat (a - Z.to_nat s) = x2 / 2^s / 2^Z.of_nat (a - Z.to_nat s).
Proof.
 intros a s x1 x2 Hs H. assert (HS : 2^s <> 0) by (apply Z.pow_nonzero; lia).
 assert (HQ : 0 < 2^Z.of_nat (a - Z.to_nat s)) by (apply Z.pow_pos_nonneg; lia).
 rewrite !Z.div_div by assumption. rewrite <- !Z.pow_add_r by lia.
 destruct (Z.le_gt_cases s (Z.of_nat a)) as [Hle|Hgt].
 - replace (s + Z.of_nat (a - Z.to_nat s)) with (Z.of_nat a) by lia. exact H.
 - replace (s + Z.of_nat (a - Z.to_nat s)) with (Z.of_nat a + (s - Z.of_nat a)) by lia.
   rewrite !Z.pow_add_r by lia. rewrite <- !Z.div_div by (try apply Z.pow_nonzero; try apply Z.pow_pos_nonneg; lia).
   rewrite H. reflexivity.
Qed.

Lemma warp_lv_and : forall w a u v m, u / 2^Z.of_nat a = v / 2^Z.of_nat a ->
 Z.land (trunc w u) (trunc w m) / 2^Z.of_nat a = Z.land (trunc w v) (trunc w m) / 2^Z.of_nat a.
Proof.
 intros w a u v m H. pose proof (warp_lv_trunc w (Z.of_nat a) u v ltac:(lia) H) as Ht.
 rewrite !warp_lv_div by lia. rewrite !Z.shiftr_land. rewrite <- !warp_lv_div by lia. rewrite Ht. reflexivity.
Qed.

Lemma warp_lv_and_low : forall a x m, (m mod 2^Z.of_nat a = 0) -> Z.land x m mod 2^Z.of_nat a = 0.
Proof.
 intros a x m Hm. rewrite <- Z.land_ones by lia. rewrite <- Z.land_assoc, Z.land_ones by lia.
 rewrite Hm. apply Z.land_0_r.
Qed.

Lemma warp_lv_full : forall a x y, 0 <= a -> x mod 2^a = 0 -> y mod 2^a = 0 -> x / 2^a = y / 2^a -> x = y.
Proof.
 intros a x y Ha Hx Hy H. assert (HP : 2^a <> 0) by (apply Z.pow_nonzero; lia).
 rewrite (Z.div_mod x (2^a) HP), (Z.div_mod y (2^a) HP), Hx, Hy, H. reflexivity.
Qed.

(* WarpExitStep *)
Lemma warp_pure_guard : forall i, warp_pure i = true -> guard_of i = None.
Proof. intros i. destruct i; cbn; try discriminate; destruct g; cbn; congruence. Qed.
Lemma warp_pure_dest : forall i, warp_pure i = true -> exists d, dest i = Some d.
Proof. intros i. destruct i; cbn; try discriminate; destruct g; cbn; try discriminate; eauto. Qed.
Lemma warp_pure_result : forall i m1 a1 s1 t1 m2 a2 s2 t2 rs, warp_pure i = true ->
 result m1 a1 s1 t1 rs i = result m2 a2 s2 t2 rs i.
Proof. intros i. destruct i; cbn; intros; try discriminate; reflexivity. Qed.
Lemma warp_result_ext : forall i m a s t x y, warp_pure i = true ->
 (forall r, In r (instruction_inputs i) -> x r = y r) -> result m a s t x i = result m a s t y i.
Proof.
 intros i m a s t x y Hp Hxy.
 destruct i; cbn [warp_pure] in Hp; try discriminate; destruct g; try discriminate;
   cbn [instruction_inputs guard_of] in Hxy; cbn [result].
 - apply Hxy; cbn; auto.
 - reflexivity.
 - rewrite (Hxy a0) by (cbn; auto). rewrite (uniform_opnd_ext b x y) by (intros; apply Hxy; cbn; auto). reflexivity.
 - rewrite (Hxy a0) by (cbn; auto).
   rewrite (uniform_opnd_ext b x y) by (intros; apply Hxy; right; apply in_or_app; auto).
   rewrite (uniform_opnd_ext c x y) by (intros; apply Hxy; right; apply in_or_app; auto). reflexivity.
 - rewrite (Hxy a0) by (cbn; auto). rewrite (uniform_opnd_ext b x y) by (intros; apply Hxy; right; apply in_or_app; auto).
   destruct c as [r|]; [rewrite (Hxy r) by (right; apply in_or_app; right; cbn; auto)|]; reflexivity.
 - rewrite (Hxy s0) by (cbn; auto). reflexivity.
 - rewrite (Hxy a0) by (cbn; auto). rewrite (uniform_opnd_ext b x y) by (intros; apply Hxy; cbn; auto). reflexivity.
 - rewrite (Hxy a0), (Hxy b) by (cbn; auto). reflexivity.
 - rewrite (Hxy a0), (Hxy b) by (cbn; auto). reflexivity.
 - rewrite (Hxy a0) by (cbn; auto). rewrite (uniform_opnd_ext b x y) by (intros; apply Hxy; cbn; auto). reflexivity.
 - rewrite (Hxy p) by (cbn; auto).
   rewrite (uniform_opnd_ext a0 x y) by (intros; apply Hxy; right; apply in_or_app; auto).
   rewrite (uniform_opnd_ext b x y) by (intros; apply Hxy; right; apply in_or_app; auto). reflexivity.
 - rewrite (Hxy a0), (Hxy b) by (cbn; auto). reflexivity.
Qed.

Lemma warp_aligned_div : forall a k v, Z.of_nat a <= k -> aligned k v -> (2^Z.of_nat a | v).
Proof.
 intros a k v Hk Hal. unfold aligned in Hal. apply (Z.divide_trans _ (2^Z.max 0 k)); [|exact Hal].
 exists (2^(Z.max 0 k - Z.of_nat a)). rewrite <- Z.pow_add_r by lia. f_equal; lia.
Qed.

Section WarpLevelStep.
Variable F : lfacts.
Variable E : env.
Hypothesis HE : env_ok F E.
Variable t : nat.
Variable st : astate.
Variables x y : reg -> Z.
Hypothesis Hst : state_ok F t st x.
Hypothesis Hrs : regs_ok F E x.
Variable L : reg -> option nat.
Hypothesis Hag : warp_agree L x y.

Lemma warp_zero_eq : forall q, L q = Some O -> x q = y q.
Proof. intros q Hq. pose proof (Hag q O Hq) as H. cbn in H. rewrite !Z.div_1_r in H. exact H. Qed.

Lemma warp_aligned_facts : forall q a, warp_aligned F st L q a = true -> x q = y q /\ (2^Z.of_nat a | x q).
Proof.
 intros q a H. unfold warp_aligned, warp_zero in H. apply andb_true_iff in H. destruct H as [Hz Hk].
 destruct (L q) as [[|?]|] eqn:Hq; try discriminate. split; [exact (warp_zero_eq q Hq)|].
 apply Z.leb_le in Hk. exact (warp_aligned_div a _ _ Hk (reg_align_sound F E HE t st x Hst Hrs q)).
Qed.

Lemma warp_offset_level_sound : forall u v a, warp_offset_level F st L u v = Some a ->
 x u / 2^Z.of_nat a = y u / 2^Z.of_nat a /\ x v = y v /\ (2^Z.of_nat a | x v).
Proof.
 intros u v a H. unfold warp_offset_level in H. destruct (L u) as [a1|] eqn:Hu; [|discriminate].
 destruct (warp_aligned F st L v a1) eqn:Ha; [|discriminate]. injection H as <-.
 split; [exact (Hag u a1 Hu)|exact (warp_aligned_facts v a1 Ha)].
Qed.

Lemma warp_const_level_sound : forall u z a, warp_const_level L u z = Some a ->
 x u / 2^Z.of_nat a = y u / 2^Z.of_nat a /\ (2^Z.of_nat a | z).
Proof.
 intros u z a H. unfold warp_const_level in H. destruct (L u) as [a1|] eqn:Hu; [|discriminate].
 destruct (warp_imm_aligned z a1) eqn:Hm; [|discriminate]. injection H as <-.
 split; [exact (Hag u a1 Hu)|]. unfold warp_imm_aligned in Hm. apply Z.eqb_eq in Hm.
 apply Z.mod_divide in Hm; [exact Hm|apply Z.pow_nonzero; lia].
Qed.

Lemma warp_level_of_sound : forall i a, warp_pure i = true -> warp_level_of F st L i = Some a ->
 warp_result x i / 2^Z.of_nat a = warp_result y i / 2^Z.of_nat a.
Proof.
 intros i a Hp Hl. unfold warp_level_of in Hl. unfold warp_result.
 destruct (forallb (warp_zero L) (instruction_inputs i)) eqn:Hz.
 { injection Hl as <-. cbn [Z.of_nat]. rewrite !Z.pow_0_r, !Z.div_1_r.
   apply warp_result_ext; [exact Hp|]. intros q Hq. rewrite forallb_forall in Hz. specialize (Hz q Hq).
   unfold warp_zero in Hz. destruct (L q) as [[|?]|] eqn:HL; try discriminate. exact (warp_zero_eq q HL). }
 destruct i as [g d s0|g d z|g op w d x0 o|g w d a0 b0 c0|g sg d a0 b0 c0|g to from sg d s0| | | | | | | | | | | | | | | | | | |];
   cbn [warp_pure] in Hp; try discriminate; destruct g as [[? ?]|]; try discriminate;
   cbn [instruction_inputs guard_of forallb] in Hz; try discriminate.
 - (* copy *) cbn [result]. exact (Hag s0 a Hl).
 - (* arithmetic *) cbn [result binop_eval].
   destruct op; try discriminate; destruct o as [y0|z]; try discriminate; cbn [opv].
   + (* add of two registers *)
     destruct (warp_offset_level F st L x0 y0) as [a1|] eqn:H1.
     * injection Hl as <-. destruct (warp_offset_level_sound _ _ _ H1) as (Hx & Heq & Hd). rewrite <- Heq.
       rewrite !(Z.add_comm _ (x y0)). exact (warp_lv_add _ _ _ _ Hd ltac:(lia) Hx).
     * destruct (warp_offset_level_sound _ _ _ Hl) as (Hy & Heq & Hd). rewrite <- Heq.
       exact (warp_lv_add _ _ _ _ Hd ltac:(lia) Hy).
   + (* add of a constant *)
     destruct (warp_const_level_sound _ _ _ Hl) as [Hx Hd].
     rewrite !(Z.add_comm _ z). exact (warp_lv_add _ _ _ _ Hd ltac:(lia) Hx).
   + (* subtraction of a register *)
     destruct (warp_offset_level_sound _ _ _ Hl) as (Hx & Heq & Hd). rewrite <- Heq.
     exact (warp_lv_sub _ _ _ _ Hd ltac:(lia) Hx).
   + (* subtraction of a constant *)
     destruct (warp_const_level_sound _ _ _ Hl) as [Hx Hd].
     exact (warp_lv_sub _ _ _ _ Hd ltac:(lia) Hx).
   + (* logical shift *)
     destruct (L x0) as [a1|] eqn:Hx; [|discriminate]. injection Hl as <-.
     apply warp_lv_shift; [unfold shamt; pose proof (trunc_range 32 z); lia|].
     exact (warp_lv_trunc w (Z.of_nat a1) _ _ ltac:(lia) (Hag x0 a1 Hx)).
   + (* arithmetic shift *)
     destruct (L x0) as [a1|] eqn:Hx; [|discriminate].
     destruct (Nat.ltb a1 w) eqn:Hw; [|discriminate]. injection Hl as <-. apply Nat.ltb_lt in Hw.
     apply warp_lv_shift; [unfold shamt; pose proof (trunc_range 32 z); lia|].
     exact (warp_lv_sext w a1 _ _ Hw (Hag x0 a1 Hx)).
   + (* and with a constant *)
     destruct (L x0) as [a1|] eqn:Hx; [|discriminate].
     pose proof (warp_lv_and w a1 (x x0) (y x0) z (Hag x0 a1 Hx)) as Hand.
     destruct (warp_imm_aligned (trunc w z) a1) eqn:Hm; injection Hl as <-; [|exact Hand].
     unfold warp_imm_aligned in Hm. apply Z.eqb_eq in Hm. cbn [Z.of_nat]. rewrite !Z.pow_0_r, !Z.div_1_r.
     apply (warp_lv_full (Z.of_nat a1)); [lia|apply warp_lv_and_low; exact Hm|apply warp_lv_and_low; exact Hm|exact Hand].
 - (* conversion *) cbn [result].
   destruct (L s0) as [a1|] eqn:Hs; [|discriminate].
   destruct (negb sg || Nat.ltb a1 from) eqn:Hc; [|discriminate]. injection Hl as <-.
   pose proof (Hag s0 a1 Hs) as H. destruct sg; cbn [extend].
   + apply Nat.ltb_lt in Hc. exact (warp_lv_sext from a1 _ _ Hc H).
   + exact (warp_lv_trunc from (Z.of_nat a1) _ _ ltac:(lia) H).
Qed.

End WarpLevelStep.

(* A pure step keeps the agreement, with the written register at its level. *)
Lemma warp_level_step_sound : forall F prog E t s s' st i L y,
 env_ok F E -> regs_ok F E (regs s) -> state_ok F t st (regs s) ->
 fetch prog (blk s) (idx s) = Some i -> warp_pure i = true -> phase_step F prog E t s = Some s' ->
 warp_agree L (regs s) y -> warp_agree (warp_level_step F st L i) (regs s') (warp_value_step F i y).
Proof.
 intros F prog E t s s' st i L y HE Hrs Hst Hf Hp Hs Hag q a Hq.
 destruct (warp_pure_dest i Hp) as [d Hd].
 pose proof (step_regs_dest F prog E HE t s s' i d Hf Hd Hs) as Hregs.
 rewrite (warp_pure_guard i Hp) in Hregs. cbn [holds] in Hregs.
 unfold warp_level_step in Hq. rewrite Hd in Hq. unfold warp_value_step. rewrite Hd, Hregs. unfold upd.
 destruct (Pos.eqb_spec q d) as [->|Hne].
 - apply warp_lv_trunc; [lia|].
   rewrite (warp_pure_result i (e_mem E) (e_arb E) (e_shm E) t (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => 0) 0 (regs s) Hp).
   exact (warp_level_of_sound F E HE t st (regs s) y Hst Hrs L Hag i a Hp Hq).
 - exact (Hag q a Hq).
Qed.

(* WarpExitRun: a run of a program agrees with a run of a second program
   when the two step alike from every state the second one reaches. *)
Lemma warp_run_agree : forall F prog prog' E n t s,
 (forall m s',phase_run F prog' E m t s=Some s' ->
   phase_step F prog E(m+t)s'=phase_step F prog' E(m+t)s') ->
 phase_run F prog E n t s=phase_run F prog' E n t s.
Proof.
 intros F prog prog' E n. induction n as[|n IH];intros t s H;[reflexivity|].
 pose proof(H 0%nat s eq_refl)as H0. cbn[Nat.add]in H0. cbn[phase_run]. rewrite H0.
 destruct(phase_step F prog' E t s)as[s1|]eqn:Hs;[|reflexivity].
 apply IH. intros m s' Hrun. replace(m+S t)%nat with(S m+t)%nat by lia.
 apply H. cbn[phase_run]. rewrite Hs. exact Hrun.
Qed.

Lemma warp_trace_agree : forall F prog prog' E events word t s,
 (forall m s',phase_run F prog' E m t s=Some s' ->
   phase_step F prog E(m+t)s'=phase_step F prog' E(m+t)s') ->
 occurrence_trace F prog' E events word t s -> occurrence_trace F prog E events word t s.
Proof.
 intros F prog prog' E events word t s Hagree H. revert Hagree.
 induction H as[t s Hdone|event word t s next Hm Hpc Hstep Htr IH|word t s next Hm Hstep Htr IH];intros Hagree.
 - constructor. intros n last Hrun. apply(Hdone n).
   rewrite<-(warp_run_agree F prog prog' E n t s Hagree). exact Hrun.
 - pose proof(Hagree 0%nat s eq_refl)as H0. cbn[Nat.add]in H0.
   eapply Occurrence_now;[exact Hm|exact Hpc|rewrite H0;exact Hstep|].
   apply IH. intros m s' Hrun. replace(m+S t)%nat with(S m+t)%nat by lia.
   apply Hagree. cbn[phase_run]. rewrite Hstep. exact Hrun.
 - pose proof(Hagree 0%nat s eq_refl)as H0. cbn[Nat.add]in H0.
   eapply Occurrence_later;[exact Hm|rewrite H0;exact Hstep|].
   apply IH. intros m s' Hrun. replace(m+S t)%nat with(S m+t)%nat by lia.
   apply Hagree. cbn[phase_run]. rewrite Hstep. exact Hrun.
Qed.

(* WarpExitEntry: two programs that differ in the entry block's last instruction. *)
Section WarpEntry.
Variable before:list instr.
Variable rest:program.

Lemma warp_entry_fetch_last : forall x,
 fetch((before++[x])::rest)0(List.length before)=Some x.
Proof.
 intros x. unfold fetch,block_body. cbn[nth_error].
 rewrite nth_error_app2 by lia. rewrite Nat.sub_diag. reflexivity.
Qed.

Lemma warp_entry_fetch_other : forall x y b i,(b,i)<>(0%nat,List.length before) ->
 fetch((before++[x])::rest)b i=fetch((before++[y])::rest)b i.
Proof.
 intros x y b i Hne. unfold fetch,block_body. destruct b as[|b];cbn[nth_error];[|reflexivity].
 destruct(lt_eq_lt_dec i(List.length before))as[[Hlt|Heq]|Hgt].
 - rewrite !nth_error_app1 by exact Hlt. reflexivity.
 - subst i. contradiction.
 - rewrite !nth_error_app2 by lia. destruct(i-List.length before)%nat as[|k]eqn:Hk;[lia|].
   cbn. destruct k;reflexivity.
Qed.

Lemma warp_entry_next : forall x y b i,
 next_pos((before++[x])::rest)b i=next_pos((before++[y])::rest)b i.
Proof.
 intros x y b i. unfold next_pos,block_len,block_body.
 destruct b as[|b];cbn[nth_error];[|reflexivity]. rewrite !app_length. reflexivity.
Qed.

Lemma warp_entry_step_other : forall F E x y t s,pc_of s<>(0%nat,List.length before) ->
 phase_step F((before++[x])::rest)E t s=phase_step F((before++[y])::rest)E t s.
Proof.
 intros F E x y t s Hne. unfold phase_step,step.
 rewrite(warp_entry_fetch_other x y(blk s)(idx s)Hne),(warp_entry_next x y). reflexivity.
Qed.

Lemma warp_entry_step_taken : forall F E neg p tg t s,pc_of s=(0%nat,List.length before) ->
 holds(Some(neg,p))(regs s)=true ->
 phase_step F((before++[IBra(Some(neg,p))tg])::rest)E t s=phase_step F((before++[IBra None tg])::rest)E t s.
Proof.
 intros F E neg p tg t s Hpc Hholds. destruct s as[b i rs]. unfold pc_of in Hpc. cbn[blk idx]in Hpc. cbn[regs]in Hholds.
 injection Hpc as -> ->. unfold phase_step,step. cbn[blk idx regs].
 rewrite !warp_entry_fetch_last. cbn[guard_of]. rewrite Hholds. reflexivity.
Qed.

Lemma warp_entry_step_fall : forall F E neg p tg t s,pc_of s=(0%nat,List.length before) ->
 holds(Some(neg,p))(regs s)=false ->
 phase_step F((before++[IBra(Some(neg,p))tg])::rest)E t s=phase_step F((before++[INop])::rest)E t s.
Proof.
 intros F E neg p tg t s Hpc Hholds. destruct s as[b i rs]. unfold pc_of in Hpc. cbn[blk idx]in Hpc. cbn[regs]in Hholds.
 injection Hpc as -> ->. unfold phase_step,step. cbn[blk idx regs].
 rewrite !warp_entry_fetch_last. cbn[guard_of]. rewrite Hholds.
 rewrite(warp_entry_next(IBra(Some(neg,p))tg)INop). reflexivity.
Qed.

End WarpEntry.

Lemma warp_run_last : forall F prog E n t s,
 phase_run F prog E(S n)t s=match phase_run F prog E n t s with
  |Some s1=>phase_step F prog E(n+t)s1|None=>None end.
Proof.
 intros F prog E n. induction n as[|n IH];intros t s.
 - cbn[phase_run Nat.add]. destruct(phase_step F prog E t s);reflexivity.
 - change(phase_run F prog E(S(S n))t s)with
     (match phase_step F prog E t s with Some s'=>phase_run F prog E(S n)(S t)s'|None=>None end).
   change(phase_run F prog E(S n)t s)with
     (match phase_step F prog E t s with Some s'=>phase_run F prog E n(S t)s'|None=>None end).
   destruct(phase_step F prog E t s)as[s1|];[|reflexivity].
   rewrite IH. replace(n+S t)%nat with(S n+t)%nat by lia. reflexivity.
Qed.

Lemma warp_fetch_in : forall prog b i x,fetch prog b i=Some x ->In x(concat prog).
Proof.
 intros prog b i x H. unfold fetch,block_body in H.
 destruct(nth_error prog b)as[body|]eqn:Hb;[|destruct i;discriminate].
 apply in_concat. exists body. split;[eapply nth_error_In;exact Hb|eapply nth_error_In;exact H].
Qed.

(* A step from outside the entry block, or from its last position, ends
   outside the entry block when every branch targets a later block. *)
Lemma warp_leave_entry : forall F prog E t s s',
 (forall g tg,In(IBra g tg)(concat prog) ->tg<>0%nat) ->prog<>[] ->
 (blk s<>0%nat \/ (block_len prog(blk s)<=S(idx s))%nat) ->
 phase_step F prog E t s=Some s' ->blk s'<>0%nat.
Proof.
 intros F prog E t s s' Hbranch Hprog Hat Hstep.
 destruct(fetch prog(blk s)(idx s))as[i|]eqn:Hf.
 2:{ unfold phase_step,step in Hstep. rewrite Hf in Hstep. discriminate. }
 destruct(step_pos F prog E t s s' i Hf Hstep)as[[Hnext _]|[[g[tg[Hi[_ Htg]]]]|[g[Hi[_ Hexit]]]]].
 - unfold next_pos in Hnext. destruct(Nat.ltb_spec(S(idx s))(block_len prog(blk s)))as[Hlt|Hge].
   + injection Hnext as Hb _. rewrite Hb. destruct Hat as[Hat|Hat];[exact Hat|lia].
   + injection Hnext as Hb _. rewrite Hb. discriminate.
 - injection Htg as Hb _. rewrite Hb. subst i. exact(Hbranch g tg(warp_fetch_in prog _ _ _ Hf)).
 - injection Hexit as Hb _. rewrite Hb. destruct prog;[contradiction|discriminate].
Qed.

Lemma warp_walk_snoc : forall F code x L st,
 warp_walk F(code++[x])L st=let '(L',st'):=warp_walk F code L st in
   (warp_level_step F st' L' x,transfer F st' x).
Proof.
 intros F code. induction code as[|i code IH];intros x L st;[reflexivity|]. cbn[app warp_walk]. apply IH.
Qed.

Lemma warp_values_snoc : forall F code x rs,
 warp_values_run F(code++[x])rs=warp_value_step F x(warp_values_run F code rs).
Proof.
 intros F code. induction code as[|i code IH];intros x rs;[reflexivity|]. cbn[app warp_values_run]. apply IH.
Qed.

(* The thread index agrees with the warp's first index from bit 5 upward. *)
Lemma warp_initial_agree : forall uniform r w E,
 32*Z.of_nat w<=e_init E r<32*(Z.of_nat w+1) ->
 warp_agree(warp_initial_levels uniform r)(e_init E)(warp_reference uniform r w E).
Proof.
 intros uniform r w E Hrange q a Hq. unfold warp_initial_levels in Hq. unfold warp_reference.
 destruct(uniform q);[reflexivity|].
 destruct(Pos.eqb_spec q r)as[->|];[|discriminate]. injection Hq as <-.
 change(2^Z.of_nat 5)with 32. rewrite Z.mul_comm,Z.div_mul by lia.
 symmetry. apply Z.div_unique with(e_init E r-32*Z.of_nat w);lia.
Qed.

Section WarpExitReach.
Variable F:lfacts.
Variable uniform:reg->bool.
Variable r:reg.
Variable w:nat.
Variable before:list instr.
Variable last:instr.
Variable rest:program.
Variable E:env.
Let Fw:=warp_facts F r w.
Let prog:=(before++[last])::rest.
Hypothesis Hlaunch:launch_ok Fw E.
Hypothesis Hlf:lf_ok Fw=true.
Hypothesis Hrange:32*Z.of_nat w<=e_init E r<32*(Z.of_nat w+1).
Hypothesis Hpure:forallb warp_pure before=true.
Hypothesis Hwf:forallb(wf_instr Fw)before=true.
Hypothesis Hbranch:forall g tg,In(IBra g tg)(concat prog) ->tg<>0%nat.

(* The actual run after the first k instructions of the entry block. *)
Definition warp_exit_at(k:nat)(rs:reg->Z):Prop:=
 exists done todo,before=done++todo /\ List.length done=k /\
 let '(L,st):=warp_walk Fw done(warp_initial_levels uniform r)(empty_state Fw)in
 state_ok Fw k st rs /\ regs_ok Fw E rs /\
 warp_agree L rs(warp_values_run Fw done(warp_reference uniform r w E)).

Lemma warp_exit_reach : forall n s',phase_run F prog E n 0(phase_initial E)=Some s' ->
 blk s'=0%nat ->idx s'=n /\ warp_exit_at n(regs s').
Proof.
 pose proof(env_of_launch Fw prog E Hlf Hlaunch)as HE.
 induction n as[|n IH];intros s' Hrun Hblk.
 - cbn[phase_run]in Hrun. injection Hrun as <-. split;[reflexivity|].
   exists [],before. split;[reflexivity|]. split;[reflexivity|]. cbn[warp_walk warp_values_run].
   split;[|split].
   + split;[|exact(lk_derived Fw E Hlaunch)]. intros q f Hf. unfold get,empty_state in Hf;cbn[amap]in Hf.
     rewrite PM.gempty in Hf. discriminate.
   + split;[exact(lk_init Fw E Hlaunch)|]. intros;reflexivity.
   + exact(warp_initial_agree uniform r w E Hrange).
 - rewrite warp_run_last in Hrun. rewrite Nat.add_0_r in Hrun.
   destruct(phase_run F prog E n 0(phase_initial E))as[s1|]eqn:Hs1;[|discriminate].
   assert(Hprog:prog<>[])by discriminate.
   destruct(Nat.eq_dec(blk s1)0)as[Hb1|Hb1].
   2:{ exfalso. exact(warp_leave_entry F prog E n s1 s' Hbranch Hprog(or_introl Hb1)Hrun Hblk). }
   destruct(IH s1 eq_refl Hb1)as[Hi1[done[todo[Hbefore[Hlen Hat]]]]].
   destruct todo as[|x todo].
   { exfalso. apply(warp_leave_entry F prog E n s1 s' Hbranch Hprog);[|exact Hrun|exact Hblk].
     right. rewrite Hb1,Hi1. unfold block_len,block_body,prog. cbn[nth_error].
     rewrite app_length,Hbefore,app_nil_r,Hlen. cbn[List.length]. lia. }
   assert(Hf:fetch prog(blk s1)(idx s1)=Some x).
   { rewrite Hb1,Hi1. unfold fetch,block_body,prog. cbn[nth_error]. rewrite Hbefore,<-app_assoc.
     rewrite nth_error_app2 by lia. rewrite Hlen,Nat.sub_diag. reflexivity. }
   assert(Hx:In x before)by(rewrite Hbefore;apply in_or_app;right;left;reflexivity).
   rewrite forallb_forall in Hpure,Hwf. specialize(Hpure x Hx). specialize(Hwf x Hx).
   destruct(step_pos F prog E n s1 s' x Hf Hrun)as[[Hnext _]|[[g[tg[Hi _]]]|[g[Hi _]]]];
     [|subst x;discriminate|subst x;discriminate].
   assert(Hidx:idx s'=S n).
   { unfold next_pos in Hnext. rewrite Hb1,Hi1 in Hnext.
     assert(Hl:(S n<block_len prog 0)%nat).
     { unfold block_len,block_body,prog. cbn[nth_error]. rewrite app_length,Hbefore,app_length,Hlen.
       cbn[List.length]. lia. }
     apply Nat.ltb_lt in Hl. rewrite Hl in Hnext. injection Hnext as _ Hi. exact Hi. }
   split;[exact Hidx|]. exists(done++[x]),todo. split;[rewrite<-app_assoc;exact Hbefore|].
   split;[rewrite app_length,Hlen;cbn[List.length];lia|].
   rewrite warp_walk_snoc,warp_values_snoc.
   destruct(warp_walk Fw done(warp_initial_levels uniform r)(empty_state Fw))as[L st]eqn:Hwalk.
   destruct Hat as[Hst[Hrs Hag]].
   assert(Hstep:step prog(wdn Fw)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells Fw)(lf_locals Fw)
     (lf_local_window Fw)n s1=Some s')by exact Hrun.
   destruct(transfer_sound Fw prog E HE n s1 s' st x Hf Hstep Hst Hrs Hwf)as[Hst' Hrs'].
   split;[exact Hst'|]. split;[exact Hrs'|].
   exact(warp_level_step_sound Fw prog E n s1 s' st x L _ HE Hrs Hst Hf Hpure Hstep Hag).
Qed.

End WarpExitReach.

(* WarpExitSound *)
(* Whether a thread of warp w takes the entry branch, by the reference run. *)
Definition warp_exit_taken(F:lfacts)(uniform:reg->bool)(before:list instr)(neg:bool)(p:reg)
 (r:reg)(w:nat)(E:env):bool:=
 holds(Some(neg,p))(warp_values_run(warp_facts F r w)before(warp_reference uniform r w E)).

Lemma warp_values_ext : forall F code x y,forallb warp_pure code=true ->
 (forall q,x q=y q) ->forall q,warp_values_run F code x q=warp_values_run F code y q.
Proof.
 intros F code. induction code as[|i code IH];intros x y Hpure Hxy;[exact Hxy|].
 cbn[forallb]in Hpure. apply andb_true_iff in Hpure. destruct Hpure as[Hi Hpure].
 cbn[warp_values_run]. apply IH;[exact Hpure|]. intros q. unfold warp_value_step.
 destruct(dest i)as[d|];[|exact(Hxy q)]. unfold upd,warp_result.
 rewrite(warp_result_ext i _ _ _ _ x y Hi(fun q' _=>Hxy q')),(Hxy q). reflexivity.
Qed.

Lemma warp_exit_taken_agree : forall F uniform before neg p r w E1 E2,
 forallb warp_pure before=true ->context_initial_agree uniform E1 E2 ->
 warp_exit_taken F uniform before neg p r w E1=warp_exit_taken F uniform before neg p r w E2.
Proof.
 intros F uniform before neg p r w E1 E2 Hpure Hagree. unfold warp_exit_taken,holds.
 rewrite(warp_values_ext(warp_facts F r w)before(warp_reference uniform r w E1)
   (warp_reference uniform r w E2)Hpure);[reflexivity|].
 intros q. unfold warp_reference. destruct(uniform q)eqn:Hu;[exact(Hagree q Hu)|reflexivity].
Qed.

Lemma warp_exit_split_sound : forall prog before neg p tg rest,
 warp_exit_split prog=Some(before,neg,p,tg,rest) ->prog=(before++[IBra(Some(neg,p))tg])::rest.
Proof.
 intros prog before neg p tg rest H. destruct prog as[|body rest0];[discriminate|]. cbn in H.
 destruct(rev body)as[|i before0]eqn:Hrev;[discriminate|].
 destruct i;try discriminate. destruct g as[[neg0 p0]|];[|discriminate].
 injection H as <- <- <- <- <-. f_equal. rewrite<-(rev_involutive body),Hrev. reflexivity.
Qed.

Lemma warp_entry_closed_variant : forall before neg p tg rest y,
 warp_entry_closed((before++[IBra(Some(neg,p))tg])::rest)=true ->(y=IBra None tg \/ y=INop) ->
 forall g t,In(IBra g t)(concat((before++[y])::rest)) ->t<>0%nat.
Proof.
 intros before neg p tg rest y Hclosed Hy g t Hin Ht. subst t.
 unfold warp_entry_closed in Hclosed. rewrite short_all_same,forallb_forall in Hclosed.
 cbn[concat]in Hin,Hclosed. apply in_app_or in Hin. rewrite<-app_assoc in Hclosed.
 destruct Hin as[Hin|Hin].
 - apply in_app_or in Hin. destruct Hin as[Hin|[Hin|[]]].
   + specialize(Hclosed _(in_or_app _ _ _(or_introl Hin))). discriminate.
   + subst y. destruct Hy as[Hy|Hy];[|discriminate]. injection Hy as _ <-.
     assert(Hl:In(IBra(Some(neg,p))0)(before++[IBra(Some(neg,p))0]++concat rest))
       by(apply in_or_app;right;left;reflexivity).
     specialize(Hclosed _ Hl). discriminate.
 - assert(Hl:In(IBra g 0)(before++[IBra(Some(neg,p))tg]++concat rest))
     by(apply in_or_app;right;right;exact Hin).
   specialize(Hclosed _ Hl). discriminate.
Qed.

Lemma warp_exit_trace : forall F uniform r w before neg p tg rest events taken fall,
 warp_entry_closed((before++[IBra(Some(neg,p))tg])::rest)=true ->
 warp_exit_pair F uniform before p tg rest events r w=Some(taken,fall) ->
 forall E,launch_ok F E ->32*Z.of_nat w<=e_init E r<32*(Z.of_nat w+1) ->
 occurrence_trace F((before++[IBra(Some(neg,p))tg])::rest)E events
   (if warp_exit_taken F uniform before neg p r w E then taken else fall)0(phase_initial E).
Proof.
 intros F uniform r w before neg p tg rest events taken fall Hclosed Hpair E Hlaunch Hrange.
 unfold warp_exit_pair in Hpair. set(Fw:=warp_facts F r w)in *.
 destruct(lf_ok Fw && short_all warp_pure before && short_all(wf_instr Fw)before &&
   warp_zero(fst(warp_walk Fw before(warp_initial_levels uniform r)(empty_state Fw)))p)eqn:Hconds;[|discriminate].
 destruct(scoped_complete_word Fw((before++[IBra None tg])::rest)events)as[wt|]eqn:Ht;[|discriminate].
 destruct(scoped_complete_word Fw((before++[INop])::rest)events)as[wf|]eqn:Hf;[|discriminate].
 injection Hpair as <- <-.
 rewrite !andb_true_iff in Hconds. destruct Hconds as[[[Hlf Hpure]Hwf]Hzero].
 rewrite short_all_same in Hpure,Hwf.
 pose proof(warp_launch F E r w Hlaunch Hrange)as HlaunchW. fold Fw in HlaunchW.
 (* At the branch, the actual predicate is the reference one. *)
 assert(Hdecide:forall y n s',(y=IBra None tg \/ y=INop) ->
   phase_run F((before++[y])::rest)E n 0(phase_initial E)=Some s' ->
   pc_of s'=(0%nat,List.length before) ->
   holds(Some(neg,p))(regs s')=warp_exit_taken F uniform before neg p r w E).
 { intros y n s' Hy Hrun Hpc. unfold pc_of in Hpc. injection Hpc as Hb Hi.
   destruct(warp_exit_reach F uniform r w before y rest E HlaunchW Hlf Hrange Hpure Hwf
     (warp_entry_closed_variant before neg p tg rest y Hclosed Hy)n s' Hrun Hb)
     as[Hn[done[todo[Hbefore[Hlen Hat]]]]].
   assert(Htodo:todo=[]).
   { destruct todo;[reflexivity|]. exfalso. apply(f_equal(@List.length instr))in Hbefore.
     rewrite app_length in Hbefore. cbn[List.length]in Hbefore. lia. }
   subst todo. rewrite app_nil_r in Hbefore. subst done. unfold warp_exit_at in Hat. fold Fw in Hat.
   destruct(warp_walk Fw before(warp_initial_levels uniform r)(empty_state Fw))as[L st]eqn:Hwalk.
   cbn[fst]in Hzero. destruct Hat as[_[_ Hag]]. unfold warp_zero in Hzero.
   destruct(L p)as[[|?]|]eqn:Hp;try discriminate. pose proof(Hag p 0%nat Hp)as Heq.
   cbn[Z.of_nat]in Heq. rewrite !Z.pow_0_r,!Z.div_1_r in Heq.
   unfold warp_exit_taken,holds. fold Fw. rewrite Heq. reflexivity. }
 destruct(warp_exit_taken F uniform before neg p r w E)eqn:Hdec.
 - pose proof(scoped_complete_word_sound Fw _ events wt Ht E HlaunchW)as Htrace.
   apply occurrence_trace_split_same in Htrace. apply occurrence_trace_split_same in Htrace.
   eapply warp_trace_agree;[|exact Htrace]. intros m s' Hrun.
   destruct(Nat.eq_dec(blk s')0)as[Hb|Hb];[destruct(Nat.eq_dec(idx s')(List.length before))as[Hi|Hi]|].
   + assert(Hpc:pc_of s'=(0%nat,List.length before))by(unfold pc_of;rewrite Hb,Hi;reflexivity).
     apply warp_entry_step_taken;[exact Hpc|].
     exact(Hdecide _ m s'(or_introl eq_refl)Hrun Hpc).
   + apply warp_entry_step_other. unfold pc_of. intros H. injection H as _ H. contradiction.
   + apply warp_entry_step_other. unfold pc_of. intros H. injection H as H _. contradiction.
 - pose proof(scoped_complete_word_sound Fw _ events wf Hf E HlaunchW)as Htrace.
   apply occurrence_trace_split_same in Htrace. apply occurrence_trace_split_same in Htrace.
   eapply warp_trace_agree;[|exact Htrace]. intros m s' Hrun.
   destruct(Nat.eq_dec(blk s')0)as[Hb|Hb];[destruct(Nat.eq_dec(idx s')(List.length before))as[Hi|Hi]|].
   + assert(Hpc:pc_of s'=(0%nat,List.length before))by(unfold pc_of;rewrite Hb,Hi;reflexivity).
     apply warp_entry_step_fall;[exact Hpc|].
     exact(Hdecide _ m s'(or_intror eq_refl)Hrun Hpc).
   + apply warp_entry_step_other. unfold pc_of. intros H. injection H as _ H. contradiction.
   + apply warp_entry_step_other. unfold pc_of. intros H. injection H as H _. contradiction.
Qed.

(* Every lane of warp w follows the same word in a CTA, and every word has
   the same projection onto the CTA barriers. *)
Definition warp_exit_schedule_safe(F:lfacts)(uniform:reg->bool)(prog:program)(events cta:list pc)
 (r:reg)(count:nat):Prop:=
 exists common,forall w,(w<count)%nat ->exists choose:env->list pc,
 (forall E1 E2,context_initial_agree uniform E1 E2 ->choose E1=choose E2) /\
 (forall E,event_projection cta(choose E)=common) /\
 (forall E,launch_ok F E ->32*Z.of_nat w<=e_init E r<32*(Z.of_nat w+1) ->
   occurrence_trace F prog E events(choose E)0(phase_initial E)).

Theorem warp_exit_check_sound : forall F uniform prog events cta r count,
 warp_exit_check F uniform prog events cta r count=true ->
 warp_exit_schedule_safe F uniform prog events cta r count.
Proof.
 intros F uniform prog events cta r count H. unfold warp_exit_check in H.
 destruct(warp_exit_split prog)as[[[[[before neg]p]tg]rest]|]eqn:Hsplit;[|discriminate].
 pose proof(warp_exit_split_sound _ _ _ _ _ _ Hsplit)as Hprog.
 apply andb_true_iff in H. destruct H as[Hclosed H]. rewrite Hprog in Hclosed.
 destruct(map(warp_exit_pair F uniform before p tg rest events r)(seq 0 count))
   as[|[[taken0 fall0]|]tail]eqn:Hpairs;try discriminate.
 exists(event_projection cta taken0). intros w Hw.
 assert(Hin:In(warp_exit_pair F uniform before p tg rest events r w)(Some(taken0,fall0)::tail)).
 { rewrite<-Hpairs. apply in_map. apply in_seq. lia. }
 rewrite short_all_same,forallb_forall in H. specialize(H _ Hin).
 destruct(warp_exit_pair F uniform before p tg rest events r w)as[[taken fall]|]eqn:Hpair;[|discriminate].
 apply andb_true_iff in H. destruct H as[Ht Hf].
 apply event_word_eqb_eq in Ht. apply event_word_eqb_eq in Hf.
 assert(Hpure:forallb warp_pure before=true).
 { unfold warp_exit_pair in Hpair.
   destruct(lf_ok(warp_facts F r w))eqn:Hlf;[|discriminate].
   destruct(short_all warp_pure before)eqn:Hp;[exact Hp|discriminate]. }
 exists(fun E=>if warp_exit_taken F uniform before neg p r w E then taken else fall).
 split;[|split].
 - intros E1 E2 Hagree. rewrite(warp_exit_taken_agree F uniform before neg p r w E1 E2 Hpure Hagree).
   reflexivity.
 - intros E. destruct(warp_exit_taken F uniform before neg p r w E);assumption.
 - intros E Hlaunch Hrange. rewrite Hprog.
   exact(warp_exit_trace F uniform r w before neg p tg rest events taken fall Hclosed Hpair E Hlaunch Hrange).
Qed.
