(* Checked interval-loop certificates. Proposals are untrusted; actual transfers,
   accesses, initialization and every CFG edge are independently checked. *)
From Coq Require Import List ZArith Bool Lia.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning GateSoundRange
 GateSoundOp GateSoundRefine GateSoundAccess GateSoundLocal GateSoundTransfer GateSoundJoin GateThread.
Import ListNotations. Local Open Scope Z_scope.
(* StrictTemplate *)
Record interval_descriptor:={id_low:reg;id_high:reg;id_copy:reg}.
Fixpoint interval_controller(head:nat)(body:list instr):option interval_descriptor:=
 match body with
 |ISetp None CGt 64 true p hi(OReg lo)::IMov None saved source::IBra(Some(pol,q))target::_=>
 if Pos.eqb lo source && Pos.eqb p q && Nat.eqb target head then
 Some{|id_low:=lo;id_high:=hi;id_copy:=saved|}else None
 |_::rest=>interval_controller head rest|[]=>None end.
Definition interval_constraints(F:lfacts)(st:astate)(d:interval_descriptor):list constraint:=
 let lo:=id_low d in let hi:=id_high d in let cp:=id_copy d in
 let lower:=as_form F st lo EX in let upper:=as_form F st hi EX in
 let endpoint r f l h:constraint:=(sorted_terms([((r,EX),1)]++tscale(-1)(ft f)),l,h)in
 [endpoint lo lower(flo lower)(flo lower+2^64-1);
 endpoint hi upper(fhi upper-(2^64-1))(fhi upper);
 endpoint lo upper(fhi upper-(2^64-1))(fhi upper);
 endpoint hi lower(flo lower)(flo lower+2^64-1);
 (sorted_terms[((hi,EX),1);((lo,EX),-1)],1,2^64-1);
 (sorted_terms[((cp,EX),1);((lo,EX),-1)],0,0)].
Definition invariant_template(F:lfacts)(written:list reg)(pre:astate)(d:interval_descriptor):option astate:=
 match reg_range F pre(id_low d),reg_range F pre(id_high d)with
 |Some(l0,h0),Some(l1,h1)=>
 let lo:=Z.min l0 l1 in let hi:=Z.max h0 h1 in
 let remains:=fun r=>negb(existsb(Pos.eqb r)written)in
 let facts:=PM.fold(fun r f m=>if remains r && forallb remains(fact_inputs f)
 then PM.add r f m else m)(amap pre)(PM.empty fact)in
 let facts:=PM.add(id_low d)(FForm({|fm:=64;ft:=[];flo:=lo;fhi:=hi|})0 None)facts in
 let facts:=PM.add(id_high d)(FForm({|fm:=64;ft:=[];flo:=lo;fhi:=hi|})0 None)facts in
 let facts:=PM.add(id_copy d)(FForm(fatom 64(id_low d,EX)1)0 None)facts in
 Some{|amap:=facts;acons:=filter(fun k=>forallb remains(constraint_inputs k))(acons pre)++interval_constraints F pre d|}
 |_,_=>None end.

(* Halving *)
Section Prototype.
Variable F:lfacts.
Definition halving_pair (f:form):bool :=
 let negative:=filter(fun t=>Z.eqb(snd t)(-1)) (ft f) in
 let positive:=filter(fun t=>Z.eqb(snd t)1) (ft f) in
 let endpoint:=fun t=>let '((r,e),_):=t in Z.eqb(wd F r)64 && match e with EX|EZ=>true|_=>false end in
 Nat.eqb(length negative)1 && forallb endpoint negative &&
 Nat.eqb(length(filter endpoint positive))1 &&
 forallb(fun t=>if endpoint t then Z.eqb(Z.abs(snd t))1 else
  let '((r,e),c):=t in match e with EShr k=>Z.eqb k 63 && Z.eqb c 1 && Z.eqb(wd F r)64|_=>false end)(ft f) &&
 Nat.leb(length(ft f))3.
Definition half_range (st:astate)(a:reg)(w:Z)(sg:bool):option range :=
 let fast:=cmp_range F st(OReg a)w sg in
 if match fast with Some(lo,_)=>0<=?lo|None=>false end then fast else
 if sg && Z.eqb w 64 &&
 Z.eqb (flo(cmp_form F st(OReg a)w sg)) (fhi(cmp_form F st(OReg a)w sg)) &&
 halving_pair(cmp_form F st(OReg a)w sg) then
 match check_range F st(cmp_form F st(OReg a)w sg) with
 |Some(lo,hi)=>if (0<=?lo)&&(hi<=?2^63-1) then Some(lo,hi)else None
 |None=>None end else None.
End Prototype.

(* HalfProof *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable E:env. Hypothesis HE:env_ok F E.
Variable t:nat. Variable rs:reg->Z. Hypothesis Hrs:regs_ok F E rs.
Notation tsum := (tsum (wdn F) rs).
Lemma checked_signed64_window_exact : forall st a lo hi,
 state_ok F t st rs -> wd F a=64 ->
 check_range F st(cmp_form F st(OReg a)64 true)=Some(lo,hi) ->
 0<=lo -> hi<=2^63-1 ->
 exists e, flo(cmp_form F st(OReg a)64 true)<=e<=fhi(cmp_form F st(OReg a)64 true) /\
 tsum(ft(cmp_form F st(OReg a)64 true))+e=cval rs(OReg a)64 true /\
 lo<=cval rs(OReg a)64 true<=hi.
Proof.
 intros st a lo hi Hst Hwa Hr Hl Hh.
 assert(Ho:owidth F (OReg a)64) by (intros r H;injection H as <-;exact Hwa).
 destruct(cmp_value F E HE t rs Hrs st(OReg a)64 true Hst ltac:(lia) Ho)as[e[He Hq]].
 pose proof(check_range_sound F E HE t rs Hrs st(cmp_form F st(OReg a)64 true)lo hi e Hst Hr He)as Hrange.
 assert(Hexact:tsum(ft(cmp_form F st(OReg a)64 true))+e=cval rs(OReg a)64 true).
 { apply(cval_exact F E HE t rs Hrs (OReg a)64 true);[lia|unfold lo_ok,hi_ok;lia|exact Hq]. }
 exists e. repeat split;try tauto;lia.
Qed.
End Sound.
Lemma half_range_keeps_fast_nonnegative : forall F st a w sg lo hi,
 cmp_range F st(OReg a)w sg=Some(lo,hi) -> 0<=lo ->
 half_range F st a w sg=Some(lo,hi).
Proof.
 intros F st a w sg lo hi Hr Hlo. unfold half_range. rewrite Hr.
 assert(0<=?lo=true)by(apply Z.leb_le;exact Hlo). rewrite H. reflexivity.
Qed.

(* Alias *)
Section Prototype.
Variable F:lfacts.
Definition alias64 st d u := negb(Pos.eqb d u) && Z.eqb(wd F d)64 && Z.eqb(wd F u)64 &&
 match copy_operand F st(OReg u)with OReg r=>Pos.eqb r d|_=>false end.
End Prototype.

(* AliasSound *)
Section Sound.
Variable F:lfacts. Variable E:env. Hypothesis HE:env_ok F E.
Variable t:nat. Variable rs:reg->Z. Hypothesis Hrs:regs_ok F E rs.
Lemma alias64_value : forall st d u,
 state_ok F t st rs -> alias64 F st d u=true -> rs u=rs d.
Proof.
 intros st d u Hst H. unfold alias64 in H. rewrite !andb_true_iff in H.
 destruct H as [[[Hne Hd]Hu]Hcopy].
 pose proof(copy_operand_value F [] E HE t st rs(OReg u)Hst Hrs)as Hv.
 destruct(copy_operand F st(OReg u))as[r|z]eqn:Hr;[|discriminate].
 apply Pos.eqb_eq in Hcopy. subst r. cbn[opv]in Hv. symmetry;exact Hv.
Qed.
End Sound.

(* ExactAlias *)
Import ListNotations. Local Open Scope Z_scope.
Definition exact_alias_saved F st d u:=
 if alias64 F st d u then
 fold_left(fun out k=>if cons_mentions d k then add_cons out(subst_cons d u k)else out)(acons st)st
 else st.
Theorem exact_alias_saved_sound : forall F E t rs st d u,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 state_ok F t(exact_alias_saved F st d u)rs.
Proof.
 intros F E t rs st d u HE Hrs Hst. unfold exact_alias_saved.
 destruct(alias64 F st d u)eqn:Hal;[|exact Hst].
 pose proof(alias64_value F E HE t rs Hrs st d u Hst Hal)as Hv.
 assert(Hwidth:wdn F u=wdn F d).
 { unfold alias64 in Hal. rewrite !andb_true_iff in Hal.
 destruct Hal as[[[_ Hd]Hu]_]. apply Z.eqb_eq in Hd,Hu. unfold wd in *;lia. }
 assert(Hk:forall k,In k(acons st)->cons_ok F rs(subst_cons d u k)).
 { intros [[ts lo]hi] Hin. unfold subst_cons. cbn[cons_ok].
 rewrite(tsum_sorted_terms F E HE t rs Hrs).
 rewrite(tsum_subst F [] E HE rs rs d u ts Hv Hwidth ltac:(intros;reflexivity)).
 exact(proj1(Forall_forall _ _)(proj2 Hst)_ Hin). }
 assert(Hfold:forall ks out,state_ok F t out rs ->
 (forall k,In k ks->cons_ok F rs(subst_cons d u k))->
 state_ok F t(fold_left(fun out k=>if cons_mentions d k then add_cons out(subst_cons d u k)else out)ks out)rs).
 { induction ks as[|k ks IH];intros out Hout Hks;cbn[fold_left];[exact Hout|].
 apply IH;[|intros j Hj;apply Hks;right;exact Hj].
 destruct(cons_mentions d k);[|exact Hout].
 apply(add_cons_ok F E HE t rs Hrs out _ Hout). apply Hks;left;reflexivity. }
 exact(Hfold(acons st)st Hst Hk).
Qed.

(* TopBit *)
Import ListNotations. Local Open Scope Z_scope.
Definition topbit_zero F st r :=
 if Z.eqb(wd F r)64 && halving_pair F(cmp_form F st(OReg r)64 true) then
 match check_range F st(cmp_form F st(OReg r)64 true)with
 |Some(lo,hi)=>if (0<=?lo)&&(hi<=?2^63-1)then add_cons st([((r,EShr 63),1)],0,0)else st
 |None=>st end else st.
Lemma topbit_zero_sound : forall F E t rs st r,
 env_ok F E -> regs_ok F E rs -> state_ok F t st rs ->
 state_ok F t(topbit_zero F st r)rs.
Proof.
 intros F E t rs st r HE Hrs Hst. unfold topbit_zero.
 destruct(Z.eqb(wd F r)64 && halving_pair F(cmp_form F st(OReg r)64 true))eqn:Htype;[|exact Hst].
 apply andb_true_iff in Htype as[Hw _]. apply Z.eqb_eq in Hw.
 destruct(check_range F st(cmp_form F st(OReg r)64 true))as[[lo hi]|]eqn:Hr;[|exact Hst].
 destruct((0<=?lo)&&(hi<=?2^63-1))eqn:Hb;[|exact Hst].
 apply andb_true_iff in Hb as[Hl Hh]. apply Z.leb_le in Hl,Hh.
 destruct(checked_signed64_window_exact F E HE t rs Hrs st r lo hi Hst Hw Hr Hl Hh)as[e[He[Hex Hrange]]].
 change (lo<=sext 64(rs r)<=hi)in Hrange.
 pose proof(rs_range F E rs Hrs r)as Hraw.
 change(0<=rs r<2^wd F r)in Hraw. rewrite Hw in Hraw.
 rewrite(sext_nonneg 64(rs r))in Hrange by lia.
 apply(add_cons_ok F E HE t rs Hrs st _ Hst).
 cbn[cons_ok tsum atom_val]. rewrite Z.div_small by lia. lia.
Qed.

Module BoundedCore.
Section Helpers.
Variable F:lfacts.

Definition witness_shift_cons (st st1 : astate) (sg : bool) (w : Z) (d a : reg) (b : opnd) : astate :=
  if negb (Z.eqb w 64) then st1 else
  match const_of F st b 32 false with
  | None => st1
  | Some k =>
      if negb (Z.eqb (Z.min (k mod 2 ^ 32) w) 1) then st1 else
      let f := cmp_form F st (OReg a) w sg in
      match half_range F st a w sg with
      | None => st1
      | Some (lo, hi) =>
          let k := Z.min (k mod 2 ^ 32) w in
          if (0 <=? lo) && Z.eqb (flo f) (fhi f) &&
             negb (existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f))
          then let result := add_cons st1 (sorted_terms (ft f ++ [((d, EX), - (2 ^ k))]),
                            - flo f, 2 ^ k - 1 - flo f) in
               result
          else st1
      end
  end.

Definition affine_step_local (st out : astate) (d : reg) (s : form) : astate :=
 if null (ft s) then out else
 if negb (existsb (fun k => let '(ts,_,_) := k in
     cons_mentions d k && Nat.leb 2 (length (filter (fun t => negb (is_fixed F (fst (fst t)))) ts))) (acons st)) then out else
 if match form_range F st s with Some (lo,hi) => Z.eqb lo hi | None => false end then out else
 if negb (Z.eqb (wd F d) 64 && Z.eqb (fm s) 64 && Z.eqb (flo s) (fhi s)) || mentions s d then out else
 let result := {|fm:=64;ft:=sorted_terms ([((d,EX),1)] ++ ft s);flo:=flo s;fhi:=fhi s|} in
 match form_range F st result with
 | Some (lo,hi) => if (0 <=? lo) && (hi <? 2^64) then
   fold_left (fun out k => let '(ts,l,h) := k in
     let ds := filter (on_reg d) ts in
     if null ds || negb (forallb (fun t => match snd (fst t) with EX | EZ => true | _ => false end) ds) then out else
     let q := fold_left (fun acc t => acc+snd t) ds 0 in
     add_cons out (sorted_terms (ts ++ tscale (-q) (ft s)),l+q*flo s,h+q*fhi s)) (acons st) out
   else out
 | None => out end.

Definition selected_jobs st d a b goals := filter(fun j=>existsb(fun goal=>
 let '(ts,lo,hi):=fst j in let '(gs,gl,gh):=goal in
 terms_eqb ts gs && (gl<=?lo)&&(hi<=?gh))goals)(selection_jobs F st 64 d a b).

Definition template_at written r k :=
 let '(ts,lo,hi):=k in match ts with
 |[(x,c);(y,d)]=>
 if Z.eqb c 1 && existsb(Pos.eqb(fst x))written && negb(existsb(Pos.eqb(fst y))written)
 then Some(subst_cons(fst x)r k)else
 if Z.eqb d 1 && existsb(Pos.eqb(fst y))written && negb(existsb(Pos.eqb(fst x))written)
 then Some(subst_cons(fst y)r k)else None
 |_=>None end.

Definition endangered st d :=
 filter(fun pair=>let '(r,x):=pair in negb(Pos.eqb r d)&&Z.eqb(wd F r)64 &&
 match x with FForm f _ _=> Z.eqb(fm f)64 && Z.eqb(flo f)(fhi f) &&
 match ft f with [((r,e),c)]=>Pos.eqb r d && Z.eqb c 1 && (match e with EX|EZ=>true|_=>false end)|_=>false end
 |_=>false end)(PM.elements(amap st)).

End Helpers.
End BoundedCore.

(* BoundedCore *)
Definition declared_atoms (goals:list constraint):=atoms_nodup(flat_map(fun k=>map fst(fst(fst k)))goals)[].

(* StepBaseSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Local Notation stepF := (step prog wdn (e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Lemma unguarded_base_sound : forall t s s' st i d,
 fetch prog(blk s)(idx s)=Some i -> stepF t s=Some s' ->
 dest i=Some d -> guard_of i=None ->
 state_ok F t st(regs s) -> regs_ok F E(regs s) -> wf_instr F i=true ->
 state_ok F(S t)(transfer_dest F st d None(value F st i)false)(regs s') /\
 regs_ok F E(regs s') /\ agree d(regs s)(regs s') /\
 regs s' d=trunc(wdn d)(result(e_mem E)(e_arb E)(e_shm E)t(regs s)i).
Proof.
 intros t s s' st i d Hf Hstep Hd Hg Hst Hrs Hwf.
 destruct(transfer_live_sound F prog E HE t s s' st i Hf Hstep Hst Hrs Hwf)as[_ Hrs'].
 pose proof(step_regs_dest F prog E HE t s s' i d Hf Hd Hstep)as Hnew.
 rewrite Hg in Hnew. cbn[holds]in Hnew.
 assert(Ha:agree d(regs s)(regs s')).
 { rewrite Hnew. apply(upd_agree F prog E HE). }
 assert(Hval:regs s' d=trunc(wdn d)(result(e_mem E)(e_arb E)(e_shm E)t(regs s)i)).
 { rewrite Hnew. unfold upd. rewrite Pos.eqb_refl;reflexivity. }
 split;[|split;[exact Hrs'|split;[exact Ha|exact Hval]]].
 apply(transfer_dest_sound F prog E HE(S t)st(regs s)(regs s')d None _ false).
 - exact(state_ok_later F prog E HE t st(regs s)Hst).
 - exact Hrs.
 - exact Hrs'.
 - exact Ha.
 - intros _ x Hx. rewrite Hval. apply(fact_okv_later F prog E HE).
   exact(value_sound F E HE t st(regs s)Hst Hrs i d x Hwf Hd Hx).
 - discriminate.
 - discriminate.
Qed.
End Sound.

(* HalfRangeSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable E:env. Hypothesis HE:env_ok F E.
Variable t:nat. Variable rs:reg->Z. Hypothesis Hrs:regs_ok F E rs.
Notation tsum := (tsum (wdn F) rs).
Lemma half_range_exact : forall st a w sg lo hi,
 state_ok F t st rs -> 1<=w<=64 -> wd F a=w ->
 half_range F st a w sg=Some(lo,hi) ->
 exists e, flo(cmp_form F st(OReg a)w sg)<=e<=fhi(cmp_form F st(OReg a)w sg) /\
 tsum(ft(cmp_form F st(OReg a)w sg))+e=cval rs(OReg a)w sg /\
 lo<=cval rs(OReg a)w sg<=hi.
Proof.
 intros st a w sg lo hi Hst Hw Hwa. unfold half_range.
 destruct(match cmp_range F st(OReg a)w sg with Some(l,_)=>0<=?l|None=>false end)eqn:Hfast.
 - intro Hr. assert(Ho:owidth F(OReg a)w)by(intros r H;injection H as <-;exact Hwa).
 destruct(cmp_exact F E HE t rs Hrs st(OReg a)w sg lo hi Hst Hw Ho Hr)as[e[He Hex]].
 exists e. split;[exact He|]. split;[exact Hex|].
 exact(cmp_range_sound F E HE t rs Hrs st(OReg a)w sg lo hi Hst Hw Ho Hr).
 - destruct(sg && Z.eqb w 64 && Z.eqb(flo(cmp_form F st(OReg a)w sg))(fhi(cmp_form F st(OReg a)w sg))&&halving_pair F(cmp_form F st(OReg a)w sg))eqn:Htype;[|discriminate].
 rewrite !andb_true_iff in Htype. destruct Htype as[[[Hsg Hw64]_ ]_].
 apply Z.eqb_eq in Hw64. subst w. rewrite Hw64 in *. destruct sg;[|discriminate].
 destruct(check_range F st(cmp_form F st(OReg a)64 true))as[[l h]|]eqn:Hr;[|discriminate].
 destruct((0<=?l)&&(h<=?2^63-1))eqn:Hb;[|discriminate].
 intros H;injection H as <- <-. apply andb_true_iff in Hb as[Hl Hh]. apply Z.leb_le in Hl,Hh.
 exact(checked_signed64_window_exact F E HE t rs Hrs st a l h Hst Hw64 Hr Hl Hh).
Qed.
End Sound.

(* ShiftSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Lemma witness_shift_cons_sound : forall t st st1 (sg : bool) n d a b rs rs',
    state_ok F t st rs -> regs_ok F E rs -> agree d rs rs' ->
    wd F d = Z.of_nat n -> wd F a = Z.of_nat n ->
    rs' d = trunc (wdn d) (binop_eval (if sg then BShrs else BShr) n (rs a) (opv rs b)) ->
    regs_ok F E rs' -> state_ok F (S t) st1 rs' ->
    state_ok F (S t) (BoundedCore.witness_shift_cons F st st1 sg (Z.of_nat n) d a b) rs'.
Proof.
  intros t st st1 sg n d a b rs rs' Hst Hrs Ha Hd Hwa Hv Hrs' Hs.
  unfold BoundedCore.witness_shift_cons.
  destruct (negb (Z.eqb (Z.of_nat n) 64)); [exact Hs |].
  destruct (const_of F st b 32 false) as [k |] eqn:Ek; [| exact Hs].
  destruct (negb (Z.eqb (Z.min (k mod 2 ^ 32) (Z.of_nat n)) 1)); [exact Hs |].
  set (f := cmp_form F st (OReg a) (Z.of_nat n) sg) in *.
  destruct (half_range F st a (Z.of_nat n) sg) as [[lo hi] |] eqn:Er; [| exact Hs].
  destruct ((0 <=? lo) && Z.eqb (flo f) (fhi f) &&
            negb (existsb (fun t => Pos.eqb (fst (fst t)) d) (ft f))) eqn:Hok; [| exact Hs].
  repeat rewrite andb_true_iff in Hok. destruct Hok as [[Hlo Hpoint] Hkeep].
  apply Z.leb_le in Hlo. apply Z.eqb_eq in Hpoint. apply negb_true_iff in Hkeep.
  assert (Hw : 1 <= Z.of_nat n <= 64) by (pose proof (wd_range F E HE d); unfold wd in Hd; lia).
  assert (Hn : wdn d = n) by (unfold wd in Hd; lia).
  assert (Hoa : owidth F (OReg a) (Z.of_nat n)) by (intros r Hr; injection Hr as <-; exact Hwa).
  destruct (half_range_exact F E HE t rs Hrs st a (Z.of_nat n) sg lo hi Hst Hw Hwa Er) as [e [He [Hf Hrange]]].
  fold f in He, Hf.
  assert (Heq : e = flo f) by lia. subst e.
  pose proof (const_of_sound F E HE t st rs Hst Hrs b 32 k ltac:(lia) Ek) as Hk.
  assert (Hsh : shamt n (opv rs b) = Z.min (k mod 2 ^ 32) (Z.of_nat n)).
  { unfold shamt. f_equal. destruct b as [r | z]; cbn [opv]; unfold trunc; cbn [Z.of_nat Pos.of_succ_nat].
    - rewrite Hk. reflexivity.
    - rewrite Hk. rewrite Z.mod_mod by lia. reflexivity. }
  set (power := 2 ^ Z.min (k mod 2 ^ 32) (Z.of_nat n)).
  assert (Hp : 0 < power).
  { unfold power. apply Z.pow_pos_nonneg. lia. pose proof (Z.mod_pos_bound k (2 ^ 32) ltac:(lia)). lia. }
  set (x := cval rs (OReg a) (Z.of_nat n) sg) in *.
  assert (Hx : 0 <= x < 2 ^ Z.of_nat n).
  { pose proof (cval_range F E HE t rs Hrs (OReg a) (Z.of_nat n) sg Hw) as Hb.
    fold x in Hb. unfold hi_ok in Hb. destruct sg.
    - assert (2 ^ (Z.of_nat n - 1) <= 2 ^ Z.of_nat n) by (apply Z.pow_le_mono_r; lia). lia.
    - lia. }
  assert (Hdval : rs' d = x / power).
  { assert (Hval : rs' d = trunc n (x / power)).
    { rewrite Hn in Hv. unfold x, cval, extend. cbn [opv]. rewrite Nat2Z.id.
      destruct sg; cbn [binop_eval] in Hv; rewrite Hsh in Hv; exact Hv. }
    clear Hv. rename Hval into Hv.
    rewrite Hv. apply trunc_small. split; [apply Z.div_pos; lia |].
    apply Z.div_lt_upper_bound; nia. }
  apply (add_cons_ok F E HE (S t) rs' Hrs' st1 _ Hs).
  cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs'), tsum_app.
  cbn [tsum atom_val]. rewrite (tsum_agree wdn d rs rs' (ft f) Ha Hkeep).
  rewrite Hdval. fold power. pose proof (Z.div_mod x power ltac:(lia)) as Hdiv.
  pose proof (Z.mod_pos_bound x power Hp) as Hmod. nia.
Qed.

End Sound.

(* ShiftInterval *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable E:env. Hypothesis HE:env_ok F E.
Lemma signed_half_interval : forall t st rs a b k lo hi v,
 state_ok F t st rs -> regs_ok F E rs -> wd F a=64 ->
 const_of F st b 32 false=Some k -> Z.min(k mod 2^32)64=1 ->
 half_range F st a 64 true=Some(lo,hi) -> 0<=lo ->
 v=trunc 64(binop_eval BShrs 64(rs a)(opv rs b)) ->
 lo/2<=v<=hi/2.
Proof.
 intros t st rs a b k lo hi v Hst Hrs Hwa Hk Hshift Hr Hlo Hv.
 destruct(half_range_exact F E HE t rs Hrs st a 64 true lo hi Hst ltac:(lia) Hwa Hr)as[e[He[Hex Hrange]]].
 pose proof(const_of_sound F E HE t st rs Hst Hrs b 32 k ltac:(lia) Hk)as Hconst.
 assert(Hsh:shamt 64(opv rs b)=1).
 { unfold shamt. rewrite <-Hshift. f_equal. destruct b as[r|z];cbn[opv];unfold trunc;cbn[Z.of_nat Pos.of_succ_nat].
 - rewrite Hconst;reflexivity.
 - rewrite Hconst. rewrite Z.mod_mod by lia;reflexivity. }
 cbn[binop_eval]in Hv. rewrite Hsh in Hv.
 change(lo<=sext 64(rs a)<=hi)in Hrange.
 pose proof(extend_range 64 true(rs a)ltac:(lia))as Hsr.
 change(-2^63<=sext 64(rs a)<=2^63-1)in Hsr.
 assert(Hq:0<=sext 64(rs a)/2<2^64).
 { split;[apply Z.div_pos;lia|apply Z.div_lt_upper_bound;lia]. }
 rewrite Hv,trunc_small by exact Hq.
 split;apply Z.div_le_mono;lia.
Qed.
End Sound.

(* ShiftNarrowSound *)
Import ListNotations. Local Open Scope Z_scope.
Definition checked_half_interval F st out d a b:=
 match const_of F st b 32 false,half_range F st a 64 true with
 |Some k,Some(lo,hi)=>if Z.eqb(Z.min(k mod 2^32)64)1 && (0<=?lo)then
 match tighten_reg F out d EX (lo/2)(hi/2)with Some narrow=>narrow|None=>bottom_state end else out
 |_,_=>out end.
Theorem checked_half_interval_sound : forall F E t st out rs rs' d a b,
 env_ok F E -> state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' ->
 wd F a=64 -> rs' d=trunc 64(binop_eval BShrs 64(rs a)(opv rs b)) ->
 state_ok F(S t)out rs' ->state_ok F(S t)(checked_half_interval F st out d a b)rs'.
Proof.
 intros F E t st out rs rs' d a b HE Hst Hrs Hrs' Hwa Hv Hout.
 unfold checked_half_interval.
 destruct(const_of F st b 32 false)as[k|]eqn:Hk;[|exact Hout].
 destruct(half_range F st a 64 true)as[[lo hi]|]eqn:Hr;[|exact Hout].
 destruct(Z.eqb(Z.min(k mod 2^32)64)1 && (0<=?lo))eqn:Htest;[|exact Hout].
 apply andb_true_iff in Htest as[Hshift Hlo]. apply Z.eqb_eq in Hshift. apply Z.leb_le in Hlo.
 pose proof(signed_half_interval F E HE t st rs a b k lo hi(rs' d)Hst Hrs Hwa Hk Hshift Hr Hlo Hv)as Hb.
 destruct(tighten_reg_sound F E HE(S t)rs' Hrs' out d EX(lo/2)(hi/2)Hout ltac:(exact Hb))as[next[Hnext Hok]].
 rewrite Hnext. exact Hok.
Qed.

(* ShiftArmSound *)
Import ListNotations. Local Open Scope Z_scope.
Definition shift_arm F st d a b:=checked_half_interval F st
 (BoundedCore.witness_shift_cons F st(transfer_dest F st d None(value F st(IBin None BShrs 64 d a b))false)true 64 d a b)d a b.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Local Notation stepF := (step prog wdn (e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Theorem shift_arm_sound : forall t s s' st d a b,
 fetch prog(blk s)(idx s)=Some(IBin None BShrs 64 d a b) ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F(IBin None BShrs 64 d a b)=true ->
 state_ok F(S t)(shift_arm F st d a b)(regs s') /\regs_ok F E(regs s').
Proof.
 intros t s s' st d a b Hf Hstep Hst Hrs Hwf.
 destruct(unguarded_base_sound F prog E HE t s s' st(IBin None BShrs 64 d a b)d Hf Hstep eq_refl eq_refl Hst Hrs Hwf)as[Hbase[Hrs'[Ha Hv]]].
 assert(Hd:wd F d=64). { unfold wf_instr in Hwf. cbn[dest guard_of]in Hwf. rewrite !andb_true_iff in Hwf. apply Z.eqb_eq;tauto. }
 assert(Hwa:wd F a=64). { unfold wf_instr in Hwf. cbn[dest guard_of]in Hwf. rewrite !andb_true_iff in Hwf. apply Z.eqb_eq;tauto. }
 assert(Hdn:wdn d=64%nat)by(unfold wd in Hd;lia).
 cbn[result]in Hv. rewrite Hdn in Hv.
 split;[|exact Hrs']. unfold shift_arm.
 apply(checked_half_interval_sound F E t st _ (regs s)(regs s')d a b HE Hst Hrs Hrs' Hwa Hv).
 apply(witness_shift_cons_sound F E HE t st _ true 64 d a b(regs s)(regs s')Hst Hrs Ha Hd Hwa).
 - rewrite Hdn. exact Hv.
 - exact Hrs'.
 - exact Hbase.
Qed.
End Sound.

(* AffineLocalSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Lemma affine_step_local_sound : forall t st out rs rs' d s,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 eqm 64 (rs' d) (rs d + (tsum wdn rs (ft s) + flo s)) ->
 state_ok F (S t) out rs' -> state_ok F (S t) (BoundedCore.affine_step_local F st out d s) rs'.
Proof.
 intros t st out rs rs' d s Hst Hrs Hrs' Ha Hstep Hout.
 unfold BoundedCore.affine_step_local.
 do 3 (lazymatch goal with |- state_ok _ _ (if ?b then _ else _) _ => destruct b eqn:?; [exact Hout |] end).
 destruct (negb (Z.eqb (wd F d) 64 && Z.eqb (fm s) 64 && Z.eqb (flo s) (fhi s)) || mentions s d) eqn:Etype; [exact Hout |].
 apply orb_false_iff in Etype. destruct Etype as [Etype Hmentions].
 apply negb_false_iff in Etype. rewrite !andb_true_iff in Etype. destruct Etype as [[Hd Hfm] Hconst].
 apply Z.eqb_eq in Hd,Hfm,Hconst.
 set (result := {|fm:=64;ft:=sorted_terms ([((d,EX),1)] ++ ft s);flo:=flo s;fhi:=fhi s|}) in *.
 destruct (form_range F st result) as [[lo hi] |] eqn:Er; [| exact Hout].
 destruct ((0 <=? lo) && (hi <? 2^64)) eqn:Ebounds; [| exact Hout].
 apply andb_true_iff in Ebounds. destruct Ebounds as [Hlo Hhi]. apply Z.leb_le in Hlo. apply Z.ltb_lt in Hhi.
 pose proof (form_range_sound F E HE t st rs Hst Hrs result lo hi (flo s) Er ltac:(unfold result; cbn [flo fhi]; lia)) as Hraw.
 unfold result in Hraw. cbn [ft] in Hraw.
 rewrite (tsum_sorted_terms F E HE t rs Hrs), tsum_app in Hraw. cbn [tsum atom_val] in Hraw.
 assert (Hvalue : rs' d = rs d + (tsum wdn rs (ft s) + flo s)).
 { pose proof (rs_range F E rs' Hrs' d) as Hr. change (0 <= rs' d < 2 ^ wd F d) in Hr. rewrite Hd in Hr.
   apply (eqm_close 64); [lia | exact Hstep | lia]. }
 generalize (proj2 Hst). generalize (acons st). intros ks Hks. revert out Hout.
 induction Hks as [| [[ts l] h] ks Hk Hks IH]; intros out Hout; cbn [fold_left]; [exact Hout |].
 apply IH.
 lazymatch goal with |- state_ok _ _ (if ?b then _ else _) _ => destruct b eqn:Ekeep; [exact Hout |] end.
 repeat match goal with H : (_ || _) = false |- _ => apply orb_false_iff in H; destruct H end.
 match goal with H : negb (forallb _ _) = false |- _ => apply negb_false_iff in H end.
 apply (add_cons_ok F E HE (S t) rs' Hrs' out _ Hout).
 eapply (affine_constraint_sound F prog E HE (S t) rs rs' d s ts l h Hrs' Ha Hmentions Hconst Hvalue); eassumption.
Qed.

End Sound.

(* AliasArmSound *)
Import ListNotations. Local Open Scope Z_scope.
Definition alias_add_arm F st d a u:=
 let i:=IBin None BAdd 64 d a(OReg u)in
 if alias64 F st d u then
 fold_predicate F(BoundedCore.affine_step_local F st(transfer_dest F st d None(value F st i)false)d(fatom 64(a,EX)1))i
 else transfer F st i.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Local Notation stepF := (step prog wdn (e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Theorem alias_add_arm_sound : forall t s s' st d a u,
 fetch prog(blk s)(idx s)=Some(IBin None BAdd 64 d a(OReg u)) ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F(IBin None BAdd 64 d a(OReg u))=true ->
 state_ok F(S t)(alias_add_arm F st d a u)(regs s') /\regs_ok F E(regs s').
Proof.
 intros t s s' st d a u Hf Hstep Hst Hrs Hwf.
 unfold alias_add_arm. destruct(alias64 F st d u)eqn:Hal.
 - destruct(unguarded_base_sound F prog E HE t s s' st(IBin None BAdd 64 d a(OReg u))d Hf Hstep eq_refl eq_refl Hst Hrs Hwf)as[Hbase[Hrs'[Ha Hv]]].
   assert(Hd:wd F d=64). { unfold wf_instr in Hwf. cbn[dest guard_of]in Hwf. rewrite !andb_true_iff in Hwf. apply Z.eqb_eq;tauto. }
   assert(Hdn:wdn d=64%nat)by(unfold wd in Hd;lia).
   cbn[result binop_eval opv]in Hv. rewrite Hdn in Hv.
   pose proof(alias64_value F E HE t(regs s)Hrs st d u Hst Hal)as Heq.
   split;[|exact Hrs']. apply(fold_predicate_sound F prog E HE(S t)_ (regs s')_ Hrs').
   apply(affine_step_local_sound F prog E HE t st _ (regs s)(regs s')d _ Hst Hrs Hrs' Ha);[|exact Hbase].
   change(eqm 64(regs s' d)(regs s d+(1*regs s a+0+0))).
   rewrite Hv,Heq. unfold eqm,trunc. rewrite Z.mod_mod by discriminate. f_equal;ring.
 - exact(transfer_sound F prog E HE t s s' st _ Hf Hstep Hst Hrs Hwf).
Qed.
End Sound.

(* SelectionJobSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Lemma branch_work_job_sound : forall t st w d a b rs rs' target opposite,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' -> agree d rs rs' ->
 (rs' d = trunc(wdn d)(opv rs b) \/ rs' d = trunc(wdn d)(opv rs a)) ->
 In(target,opposite)(branch_work F st w d a b) ->
 cons_ok F rs opposite -> cons_ok F rs' target.
Proof.
 intros t st w d a b rs rs' target opposite Hst Hrs Hrs' Ha Hv Hmember Hopp.
  (* d holds a register's value when it holds that value read at d's width *)
  assert (Hsel : forall rx, wdn d = wdn rx -> rs' d = trunc (wdn d) (rs rx) -> rs' d = rs rx).
  { intros rx Hn Hx. rewrite Hx, Hn. unfold trunc. apply Z.mod_small. exact (rs_range F E rs Hrs rx). }
  (* no atom of a constraint that does not mention d is d's *)
  assert (Hxd : forall ts lo hi r,
    (cons_mentions d (ts, lo, hi) = false \/ r = d) ->
    forall x e c, In ((x, e), c) ts -> x <> r -> x <> d).
  { intros ts lo hi r [Hnd | Hr] x e c Hx Hne; [| congruence].
    intros ->. unfold cons_mentions in Hnd. cbn [fst] in Hnd.
    assert (Hin' : existsb (fun t => Pos.eqb (fst (fst t)) d) ts = true)
      by (apply existsb_exists; exists ((d, e), c); split; [exact Hx | apply Pos.eqb_refl]).
    congruence. }
  (* d is register r: the constraint holds before *)
  assert (Hsame : forall r k, wdn d = wdn r -> rs' d = rs r -> state_ok F t st rs -> In k (acons st) ->
                  (cons_mentions d k = false \/ r = d) ->
                  cons_ok F rs' (subst_cons r d k)).
  { intros r [[ts lo] hi] Hn Hd Hst0 Hin Hnd. unfold subst_cons. cbn [cons_ok].
    rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
    pose proof (proj1 (Forall_forall _ _) (proj2 Hst0) _ Hin) as Hok. cbn [cons_ok] in Hok.
    rewrite (tsum_subst F prog E HE rs' rs r d ts); [exact Hok | exact Hd | congruence |].
    intros x e c Hx Hne. exact (Ha x (Hxd ts lo hi r Hnd x e c Hx Hne)). }
  (* d is register r': the constraint with r' in r's place holds before *)
  assert (Hother : forall r r' k, wdn d = wdn r -> wdn d = wdn r' -> rs' d = rs r' ->
                   (cons_mentions d k = false \/ r = d) ->
                   cons_ok F rs (subst_cons r r' k) -> cons_ok F rs' (subst_cons r d k)).
  { intros r r' [[ts lo] hi] Hn Hn' Hd Hnd Hok.
    unfold subst_cons in Hok |- *. cbn [cons_ok] in Hok |- *.
    rewrite (tsum_sorted_terms F E HE t rs Hrs) in Hok. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
    set (rsx := fun x => if Pos.eqb x r then rs r' else rs x).
    rewrite (tsum_subst F prog E HE rs rsx r r' ts) in Hok;
      [| unfold rsx; rewrite Pos.eqb_refl; reflexivity | congruence
       | intros x e c _ Hne; unfold rsx; apply Pos.eqb_neq in Hne; rewrite Hne; reflexivity].
    rewrite (tsum_subst F prog E HE rs' rsx r d ts); [exact Hok | unfold rsx; rewrite Pos.eqb_refl; exact Hd | congruence |].
    intros x e c Hx Hne. unfold rsx. rewrite (proj2 (Pos.eqb_neq _ _) Hne).
    exact (Ha x (Hxd ts lo hi r Hnd x e c Hx Hne)). }
  (* d is the constant v: the constraint with v in r's place holds before *)
  assert (Hconst : forall r v k, wdn d = wdn r -> rs' d = v ->
                   (cons_mentions d k = false \/ r = d) ->
                   cons_ok F rs (const_cons F r v k) -> cons_ok F rs' (subst_cons r d k)).
  { intros r v [[ts lo] hi] Hn Hd Hnd Hok.
    unfold const_cons in Hok. destruct (constant_terms F r v ts) as [ts' c] eqn:Es. cbn [cons_ok] in Hok.
    unfold subst_cons. cbn [cons_ok]. rewrite (tsum_sorted_terms F E HE (S t) rs' Hrs').
    set (rsx := fun x => if Pos.eqb x r then v else rs x).
    rewrite (tsum_subst F prog E HE rs' rsx r d ts); [| unfold rsx; rewrite Pos.eqb_refl; exact Hd | congruence |].
    - pose proof (constant_terms_tsum F prog E HE rsx r v ts ltac:(unfold rsx; rewrite Pos.eqb_refl; reflexivity)) as Ht.
      rewrite Es in Ht. cbn [fst snd] in Ht.
      pose proof (constant_terms_absent F prog E HE r v ts) as Er. rewrite Es in Er. cbn [fst] in Er.
      assert (Hagree : agree r rs rsx) by (intros x Hne; unfold rsx; apply Pos.eqb_neq in Hne; rewrite Hne; reflexivity).
      rewrite (tsum_agree wdn r rs rsx ts' Hagree Er) in Ht. lia.
    - intros x e c0 Hx Hne. unfold rsx. rewrite (proj2 (Pos.eqb_neq _ _) Hne).
      exact (Ha x (Hxd ts lo hi r Hnd x e c0 Hx Hne)). }

  unfold branch_work in Hmember. cbv zeta in Hmember.
  destruct a as[ra|za],b as[rb|zb];try contradiction.
  - destruct(Z.eqb(wd F rb)w)eqn:Ewb;[|contradiction].
    destruct(Z.eqb(wd F ra)w && Z.eqb(wd F d)w)eqn:Ew;[|contradiction].
    apply andb_true_iff in Ew. destruct Ew as[Ewa Ewd].
    apply Z.eqb_eq in Ewa,Ewb,Ewd.
    assert(Hna:wdn d=wdn ra)by(unfold wd in Ewa,Ewd;lia).
    assert(Hnb:wdn d=wdn rb)by(unfold wd in Ewb,Ewd;lia).
    apply in_flat_map in Hmember. destruct Hmember as[k[Hkin Hjob]].
    destruct(cons_mentions ra k &&(negb(cons_mentions d k)||Pos.eqb ra d))eqn:Hguard;
      [|contradiction].
    cbn in Hjob. destruct Hjob as[Heq|[]]. inversion Heq;subst target opposite.
    apply andb_true_iff in Hguard. destruct Hguard as[_ Hnd].
    rewrite orb_true_iff,negb_true_iff,Pos.eqb_eq in Hnd.
    destruct Hv as[Hv|Hv];cbn[opv]in Hv.
    + exact(Hother ra rb k Hna Hnb(Hsel rb Hnb Hv)Hnd Hopp).
    + exact(Hsame ra k Hna(Hsel ra Hna Hv)Hst Hkin Hnd).
  - destruct(Z.eqb(wd F ra)w && Z.eqb(wd F d)w)eqn:Ew;[|contradiction].
    apply andb_true_iff in Ew. destruct Ew as[Ewa Ewd].
    apply Z.eqb_eq in Ewa,Ewd.
    assert(Hna:wdn d=wdn ra)by(unfold wd in Ewa,Ewd;lia).
    apply in_flat_map in Hmember. destruct Hmember as[k[Hkin Hjob]].
    destruct(cons_mentions ra k &&(negb(cons_mentions d k)||Pos.eqb ra d))eqn:Hguard;
      [|contradiction].
    cbn in Hjob. destruct Hjob as[Heq|[]]. inversion Heq;subst target opposite.
    apply andb_true_iff in Hguard. destruct Hguard as[_ Hnd].
    rewrite orb_true_iff,negb_true_iff,Pos.eqb_eq in Hnd.
    destruct Hv as[Hv|Hv];cbn[opv]in Hv.
    + apply(Hconst ra(zb mod 2^w)k Hna);[|exact Hnd|exact Hopp].
      rewrite Hv. unfold trunc. unfold wd in Ewd. rewrite Ewd. reflexivity.
    + exact(Hsame ra k Hna(Hsel ra Hna Hv)Hst Hkin Hnd).
Qed.

End Sound.

(* ShallowWitness *)
Import ListNotations. Local Open Scope Z_scope.
Definition shallow_singleton F st x:option constraint:=
 match find(fun k=>let '(ts,_,_):=k in terms_eqb ts[(x,1)])(acons st)with
 |Some k=>Some k|None=>match inter(atom_range_with F(rf F 0 st)x)(natural_range F x)with
 |Some(lo,hi)=>Some([(x,1)],lo,hi)|None=>None end end.
Definition shallow_reduce F st keep k:=fold_left(fun current x=>
 if existsb(atom_eqb x)keep then current else match shallow_singleton F st x with
 |Some bound=>eliminate x current bound|None=>current end)(map fst(fst(fst k)))k.
Definition relevant_constraint (keep:list atom) (k:constraint):=existsb(fun x=>existsb(atom_eqb x)keep)(map fst(fst(fst k))).
Definition shallow_witness F keep pivot st:astate:={|amap:=amap st;acons:=fm_step pivot
 (map(shallow_reduce F st keep)(filter(relevant_constraint keep)(acons st)))|}.

(* ShallowSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable E:env. Hypothesis HE:env_ok F E.
Variable t:nat. Variable rs:reg->Z. Hypothesis Hrs:regs_ok F E rs.
Variable st:astate. Hypothesis Hst:state_ok F t st rs.
Lemma shallow_singleton_sound : forall x k,
 shallow_singleton F st x=Some k -> cons_ok F rs k.
Proof.
 intros x k. unfold shallow_singleton.
 destruct(find(fun k=>let '(ts,_,_):=k in terms_eqb ts[(x,1)])(acons st))as[j|]eqn:Hfind.
 - intros H;injection H as <-. apply find_some in Hfind as[Hin _].
 exact(proj1(Forall_forall _ _)(proj2 Hst)_ Hin).
 - destruct(inter(atom_range_with F(rf F 0 st)x)(natural_range F x))as[[lo hi]|]eqn:Hr;[|discriminate].
 intros H;injection H as <-.
 assert(Hrange:lo<=atom_val(wdn F)rs x<=hi).
 { pose proof(inter_ok (atom_range_with F(rf F 0 st)x)(natural_range F x)(atom_val(wdn F)rs x))as H.
   assert(Ha:inter_sound(atom_range_with F(rf F 0 st)x)(atom_val(wdn F)rs x)).
   { intros a b Ha. exact(atom_range_with_sound F E HE rs Hrs _ x a b (rf_sound F E HE t st rs Hst Hrs 0) Ha). }
   assert(Hn:inter_sound(natural_range F x)(atom_val(wdn F)rs x)).
   { intros a b Hn. exact(natural_range_sound F E HE rs Hrs x a b Hn). }
   exact(H Ha Hn lo hi Hr). }
 cbn[cons_ok tsum]. lia.
Qed.
Lemma shallow_reduce_sound : forall keep k,
 cons_ok F rs k -> cons_ok F rs(shallow_reduce F st keep k).
Proof.
 intros keep k Hk. unfold shallow_reduce.
 assert(Hfold:forall xs current,cons_ok F rs current -> cons_ok F rs
 (fold_left(fun current x=>if existsb(atom_eqb x)keep then current else
 match shallow_singleton F st x with Some bound=>eliminate x current bound|None=>current end)xs current)).
 { induction xs as[|x xs IH];intros current Hcurrent;cbn[fold_left];[exact Hcurrent|].
 apply IH. destruct(existsb(atom_eqb x)keep);[exact Hcurrent|].
 destruct(shallow_singleton F st x)as[j|]eqn:Hj;[|exact Hcurrent].
 apply(eliminate_ok F E HE t rs Hrs x current j Hcurrent).
 exact(shallow_singleton_sound x j Hj). }
 exact(Hfold _ _ Hk).
Qed.
Theorem shallow_witness_sound : forall keep pivot,
 state_ok F t(shallow_witness F keep pivot st)rs.
Proof.
 intros keep pivot. split;[exact(proj1 Hst)|].
 apply(fm_step_ok F E HE t rs Hrs).
 apply Forall_map. apply Forall_forall. intros k Hin.
 apply shallow_reduce_sound. apply filter_In in Hin as[Hin _]. exact(proj1(Forall_forall _ _)(proj2 Hst)_ Hin).
Qed.
End Sound.

(* SelValueSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Notation wdn := (wdn F).
Lemma sel_value_fact_sound : forall t st rs (w:nat) a b p x,
 state_ok F t st rs ->regs_ok F E rs ->1<=Z.of_nat w<=64 ->
 sel_value F st a b p(Z.of_nat w)=Some x ->
 fact_okv F t rs x(trunc(Z.to_nat(Z.of_nat w))(if Z.eqb(rs p)0 then opv rs b else opv rs a)).
Proof.
 intros t st rs w a b p x Hst Hrs Hw Hx.
      assert (Hjoin : sel_join F st a b p (Z.of_nat w) = Some x ->
                      fact_okv F t rs x (trunc (Z.to_nat (Z.of_nat w)) (if Z.eqb (rs p) 0 then opv rs b else opv rs a))).
      { intros Hx'. unfold sel_join in Hx'.
        destruct (Z.eqb (rs p) 0) eqn:Ep.
        - (* p fails, so the refinement by its failure succeeds *)
          destruct (refine_sound F E HE t rs Hrs st p false Hst ltac:(rewrite Ep; reflexivity)) as [s1 [Er1 Hs1]].
          rewrite Er1 in Hx'. destruct (refine F st p true) as [s2 |].
          + destruct (sel_fact F s1 b (Z.of_nat w)) as [xb |] eqn:Eb;
              [| destruct (sel_fact F s2 a (Z.of_nat w)); discriminate].
            exact (join_fact_sound_r F E HE t rs Hrs s2 s1 _ xb x _ Hs1
                     (sel_fact_sound F prog E HE t s1 rs b _ xb Hs1 Hrs Hw Eb) Hx').
          + exact (sel_fact_sound F prog E HE t s1 rs b _ x Hs1 Hrs Hw Hx').
        - (* p holds, so the refinement by it succeeds *)
          destruct (refine_sound F E HE t rs Hrs st p true Hst ltac:(rewrite Ep; reflexivity)) as [s1 [Er1 Hs1]].
          rewrite Er1 in Hx'. destruct (refine F st p false) as [s2 |].
          + destruct (sel_fact F s1 a (Z.of_nat w)) as [xa |] eqn:Ea; [| discriminate].
            exact (join_fact_sound_l F E HE t rs Hrs s1 s2 xa _ x _ Hs1
                     (sel_fact_sound F prog E HE t s1 rs a _ xa Hs1 Hrs Hw Ea) Hx').
          + exact (sel_fact_sound F prog E HE t s1 rs a _ x Hs1 Hrs Hw Hx'). }
      unfold sel_value in Hx.
      destruct (refine F st p true) as [s1 |]; [| exact (Hjoin Hx)].
      destruct (refine F st p false) as [s2 |]; [| exact (Hjoin Hx)].
      destruct a as [ra | za]; [exact (Hjoin Hx) |]. destruct b as [rb | zb]; [exact (Hjoin Hx) |].
      destruct (Z.eqb_spec (wd F p) 1) as [Hp1 | Hp1]; [| exact (Hjoin Hx)].
      (* two constants: b + (a - b) * p, with p's value 0 or 1 *)
      injection Hx as <-. cbn [opv].
      pose proof (rs_range F E rs Hrs p) as Hrp. unfold wd in Hp1. rewrite Hp1 in Hrp.
      cbn [fact_okv]. split; [unfold wf; rewrite mk_fm; lia |].
      split; [| split; [apply aligned_nonpos; lia | discriminate]].
      apply (sat_mk wdn rs (Z.of_nat w) _ zb zb _ zb); [lia | lia |]. cbn [tsum atom_val].
      unfold trunc. rewrite Z2Nat.id by lia.
      destruct (Z.eqb_spec (rs p) 0) as [Hz | Hz].
      - rewrite Hz. replace ((za - zb) * 0 + 0 + zb) with zb by ring. apply eqm_mod. lia.
      - assert (rs p = 1) as -> by lia. replace ((za - zb) * 1 + 0 + zb) with za by ring. apply eqm_mod. lia.
Qed.
End Sound.

(* SelectionFoldSound *)
Import ListNotations. Local Open Scope Z_scope.
Lemma selection_jobs_member : forall F st w d a b j,
 In j(selection_jobs F st w d a b) ->
 In j(branch_work F st w d a b) \/ In j(branch_work F st w d b a).
Proof.
 intros F st w d a b j H. destruct a as[ra|za],b as[rb|zb];cbn[selection_jobs]in H.
 - destruct(Pos.eqb ra rb);[left;exact H|]. apply in_app_iff in H. exact H.
 - left;exact H.
 - right;exact H.
 - left;exact H.
Qed.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Theorem selection_work_sound : forall t st w d a b rs rs' holds jobs out,
 state_ok F t st rs -> regs_ok F E rs -> regs_ok F E rs' ->agree d rs rs' ->
 (rs' d=trunc(wdn F d)(opv rs b) \/ rs' d=trunc(wdn F d)(opv rs a)) ->
 (forall j,In j jobs ->In j(selection_jobs F st w d a b)) ->
 (forall k,holds k=true ->cons_ok F rs k) ->
 state_ok F(S t)out rs' -> state_ok F(S t)(work_apply holds jobs out)rs'.
Proof.
 intros t st w d a b rs rs' holds jobs out Hst Hrs Hrs' Ha Hv Hjobs Hholds Hout.
 unfold work_apply. revert out Hout Hjobs. induction jobs as[|[target opposite]jobs IH];intros out Hout Hjobs;cbn[fold_left fst snd].
 - exact Hout.
 - apply IH;[|intros j Hj;apply Hjobs;right;exact Hj].
   destruct(holds opposite)eqn:Ho;[|exact Hout].
   apply(add_cons_ok F E HE(S t)rs' Hrs' out target Hout).
   pose proof(selection_jobs_member F st w d a b(target,opposite)(Hjobs _ (or_introl eq_refl)))as Hmember.
   destruct Hmember as[Hmember|Hmember].
   + exact(branch_work_job_sound F prog E HE t st w d a b rs rs' target opposite Hst Hrs Hrs' Ha Hv Hmember(Hholds _ Ho)).
   + apply(branch_work_job_sound F prog E HE t st w d b a rs rs' target opposite Hst Hrs Hrs' Ha);[tauto|exact Hmember|exact(Hholds _ Ho)].
Qed.
End Sound.

(* SavePlan *)
Definition save_jobs F written goals st d:=flat_map(fun pair=>flat_map(fun k=>
 let direction:=match snd pair with FForm f _ _=>let '(_,lo,hi):=k in
 ((flo f<=?0)&&(hi<=?0))||((0<=?flo f)&&(0<=?lo))|_=>false end in
 if direction then match BoundedCore.template_at written(fst pair)k with Some c=>[c]|None=>[]end else[])goals)(BoundedCore.endangered F st d).

(* CompactSelection *)
Import ListNotations. Local Open Scope Z_scope.
Definition constraint_atoms (ks:list constraint):=flat_map(fun k=>map fst(fst(fst k)))ks.
Definition compact_save F keep pivot written goals st d:=
 let jobs:=save_jobs F written goals st d in
 let keep:=atoms_nodup(keep++constraint_atoms jobs)[]in
 let checked:=shallow_witness F keep pivot st in
 fold_left(fun out k=>if cons_holds F checked k then add_cons out k else out)jobs st.
Definition compact_select F keep pivot written goals st d a b p:=
 let pre:=compact_save F keep pivot written goals st d in
 let out:=transfer_dest F pre d None(sel_value F pre a b p 64)false in
 let jobs:=BoundedCore.selected_jobs F pre d a b goals in
 let keep:=atoms_nodup(keep++constraint_atoms(map fst jobs++map snd jobs))[]in
 let checked:=shallow_witness F keep pivot pre in
 fold_predicate F(work_apply(cons_holds F checked)jobs out)(ISelp None 64 d a b p).

(* CompactSelectionSound *)
Import ListNotations. Local Open Scope Z_scope.
Lemma add_checked_constraints_sound : forall F E t rs checked jobs out,
 env_ok F E ->regs_ok F E rs ->state_ok F t checked rs ->state_ok F t out rs ->
 state_ok F t(fold_left(fun out k=>if cons_holds F checked k then add_cons out k else out)jobs out)rs.
Proof.
 intros F E t rs checked jobs out HE Hrs Hchecked Hout.
 revert out Hout. induction jobs as[|k jobs IH];intros out Hout;cbn[fold_left];[exact Hout|].
 apply IH. destruct(cons_holds F checked k)eqn:Hk;[|exact Hout].
 apply(add_cons_ok F E HE t rs Hrs out k Hout).
 exact(cons_holds_sound F E HE t rs Hrs checked k Hchecked Hk).
Qed.
Lemma compact_save_sound : forall F E t rs st keep pivot written goals d,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->
 state_ok F t(compact_save F keep pivot written goals st d)rs.
Proof.
 intros F E t rs st keep pivot written goals d HE Hrs Hst.
 unfold compact_save. cbv zeta.
 apply(add_checked_constraints_sound F E t rs);try assumption.
 exact(shallow_witness_sound F E HE t rs Hrs st Hst _ _).
Qed.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Local Notation stepF := (step prog(wdn F)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Theorem compact_select_sound : forall t s s' st keep pivot written goals d a b p,
 fetch prog(blk s)(idx s)=Some(ISelp None 64 d a b p) ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F(ISelp None 64 d a b p)=true ->
 state_ok F(S t)(compact_select F keep pivot written goals st d a b p)(regs s') /\regs_ok F E(regs s').
Proof.
 intros t s s' st keep pivot written goals d a b p Hf Hstep Hst Hrs Hwf.
 unfold compact_select.
 set(pre:=compact_save F keep pivot written goals st d).
 assert(Hpre:state_ok F t pre(regs s))by(exact(compact_save_sound F E t(regs s)st keep pivot written goals d HE Hrs Hst)).
 destruct(unguarded_base_sound F prog E HE t s s' pre(ISelp None 64 d a b p)d Hf Hstep eq_refl eq_refl Hpre Hrs Hwf)as[Hbase[Hrs'[Ha Hv]]].
 assert(Hd:wd F d=64).
 { unfold wf_instr in Hwf. cbn[dest guard_of]in Hwf. rewrite !andb_true_iff in Hwf. apply Z.eqb_eq;tauto. }
 assert(Hdn:wdn F d=64%nat)by(unfold wd in Hd;lia).
 assert(Hselected:state_ok F(S t)(transfer_dest F pre d None(sel_value F pre a b p 64)false)(regs s')).
 { apply(transfer_dest_sound F prog E HE(S t)pre(regs s)(regs s')d None _ false).
   - exact(state_ok_later F prog E HE t pre(regs s)Hpre).
   - exact Hrs.
   - exact Hrs'.
   - exact Ha.
   - intros _ x Hx. rewrite Hv. apply(fact_okv_later F prog E HE). cbn[result]. rewrite Hdn.
     exact(sel_value_fact_sound F prog E HE t pre(regs s)64 a b p x Hpre Hrs ltac:(lia) Hx).
   - discriminate.
   - discriminate. }
 clear Hbase.
 split;[|exact Hrs']. apply(fold_predicate_sound F prog E HE(S t)_ (regs s')_ Hrs').
 apply(selection_work_sound F prog E HE t pre 64 d a b(regs s)(regs s'));
 try assumption.
 - cbn[result]in Hv. destruct(Z.eqb(regs s p)0);auto.
 - intros j Hj. unfold BoundedCore.selected_jobs in Hj. apply filter_In in Hj as[Hj _]. exact Hj.
 - intros k Hk. eapply(cons_holds_sound F E HE t(regs s)Hrs);[|exact Hk].
   exact(shallow_witness_sound F E HE t(regs s)Hrs pre Hpre _ _).
Qed.
End Sound.

(* Coherent *)
Import ListNotations. Local Open Scope Z_scope.
Definition compact_goals F keep pivot goals st:=
 let checked:=shallow_witness F keep pivot st in
 fold_left(fun out k=>if cons_holds F checked k then add_cons out k else out)goals st.
Definition checked_core F keep pivot written goals st i:=
 match i with
 |IBin None op w d a b=>if Nat.eqb w 64 then
   match op,b with
   |BShrs,_=>shift_arm F st d a b
   |BAdd,OReg u=>alias_add_arm F(exact_alias_saved F st d u)d a u
   |BShr,OImm k=>if Z.eqb k 63 then transfer F(topbit_zero F st a)i else transfer F st i
   |_,_=>transfer F st i end else transfer F st i
 |ISelp None w d a b p=>if Nat.eqb w 64 then compact_select F keep pivot written goals st d a b p else transfer F st i
 |_=>transfer F st i end.
Definition coherent_transfer F written goals desc st i:=
 let keep:=declared_atoms goals in
 let pivot:=match desc with Some d=>(id_high d,EX)|None=>(1%positive,EX)end in
 let save:=compact_goals F keep pivot goals in
 let st:=match desc,i with
 |Some d,IMov None out input=>if Pos.eqb out(id_copy d)&&Pos.eqb input(id_low d)then save st else st
 |_,_=>st end in
 let out:=checked_core F keep pivot written goals st i in
 match desc,i with Some d,IBin None BAdd 64 dst _ _=>if Pos.eqb dst(id_low d)then save out else out|_,_=>out end.

(* CoreSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Local Notation stepF := (step prog(wdn F)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Theorem checked_core_sound : forall t s s' st keep pivot written goals i,
 fetch prog(blk s)(idx s)=Some i ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F i=true ->
 state_ok F(S t)(checked_core F keep pivot written goals st i)(regs s') /\regs_ok F E(regs s').
Proof.
 intros t s s' st keep pivot written goals i Hf Hstep Hst Hrs Hwf.
 pose proof(transfer_sound F prog E HE t s s' st i Hf Hstep Hst Hrs Hwf)as Hnormal.
 unfold checked_core.
 destruct i as[g d a|g d z|g op w d a b|g w d a b c|g sg d a b c|g to from sg d a
 |g w d a p ln|g c w sg d a b|g d a b|g d a b|g mn sg w d a b|g w d a b p
 |g gen bits d a off|g k gen a off width v fr ds|g ds|g target|g| |g rem sg w d a b
 |g bits d a off|g bits d a off fr|g a off width v fr|vs|g a off width|g d a b];try exact Hnormal.
 - destruct g;[exact Hnormal|]. destruct(Nat.eqb w 64)eqn:Hw;[|exact Hnormal].
   apply Nat.eqb_eq in Hw;subst w. destruct op;try exact Hnormal.
   + destruct b as[u|z];[|exact Hnormal]. apply(alias_add_arm_sound F prog E HE t s s');try assumption.
     exact(exact_alias_saved_sound F E t(regs s)st d u HE Hrs Hst).
   + destruct b as[u|z];[exact Hnormal|]. destruct(Z.eqb z 63);[|exact Hnormal].
     apply(transfer_sound F prog E HE t s s');try assumption.
     exact(topbit_zero_sound F E t(regs s)st a HE Hrs Hst).
   + exact(shift_arm_sound F prog E HE t s s' st d a b Hf Hstep Hst Hrs Hwf).
 - destruct g;[exact Hnormal|]. destruct(Nat.eqb w 64)eqn:Hw;[|exact Hnormal].
   apply Nat.eqb_eq in Hw;subst w.
   exact(compact_select_sound F prog E HE t s s' st keep pivot written goals d a b p Hf Hstep Hst Hrs Hwf).
Qed.
End Sound.

(* CoherentSound *)
Import ListNotations. Local Open Scope Z_scope.
Lemma compact_goals_sound : forall F E t rs st keep pivot goals,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->
 state_ok F t(compact_goals F keep pivot goals st)rs.
Proof.
 intros F E t rs st keep pivot goals HE Hrs Hst. unfold compact_goals.
 apply(add_checked_constraints_sound F E t rs);try assumption.
 exact(shallow_witness_sound F E HE t rs Hrs st Hst keep pivot).
Qed.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env. Hypothesis HE:env_ok F E.
Local Notation stepF := (step prog(wdn F)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Theorem coherent_transfer_sound : forall t s s' st written goals desc i,
 fetch prog(blk s)(idx s)=Some i ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F i=true ->
 state_ok F(S t)(coherent_transfer F written goals desc st i)(regs s') /\regs_ok F E(regs s').
Proof.
 intros t s s' st written goals desc i Hf Hstep Hst Hrs Hwf.
 unfold coherent_transfer. cbv zeta.
 set(keep:=declared_atoms goals).
 set(pivot:=match desc with Some d=>(id_high d,EX)|None=>(1%positive,EX)end).
 set(save:=compact_goals F keep pivot goals).
 assert(Hsave:forall tt rr source,regs_ok F E rr ->state_ok F tt source rr ->state_ok F tt(save source)rr).
 { intros tt rr source Hr Hs. exact(compact_goals_sound F E tt rr source keep pivot goals HE Hr Hs). }
 set(pre:=match desc,i with
 |Some d,IMov None out input=>if Pos.eqb out(id_copy d)&&Pos.eqb input(id_low d)then save st else st
 |_,_=>st end).
 assert(Hpre:state_ok F t pre(regs s)).
 { unfold pre. destruct desc as[d|];[|exact Hst]. destruct i;try exact Hst.
   destruct g;[exact Hst|]. destruct(Pos.eqb d0(id_copy d)&&Pos.eqb s0(id_low d));[apply Hsave;assumption|exact Hst]. }
 destruct(checked_core_sound F prog E HE t s s' pre keep pivot written goals i Hf Hstep Hpre Hrs Hwf)as[Hout Hrout].
 split;[|exact Hrout].
 change(state_ok F(S t)(match desc,i with
 |Some d,IBin None BAdd 64 dst _ _=>if Pos.eqb dst(id_low d)then save(checked_core F keep pivot written goals pre i)else checked_core F keep pivot written goals pre i
 |_,_=>checked_core F keep pivot written goals pre i end)(regs s')).
 destruct desc as[d|];[|exact Hout]. destruct i;try exact Hout.
 destruct g;[exact Hout|]. destruct o;try exact Hout.
 do 64 (destruct w as[|w];[exact Hout|]).
 destruct w as[|w];[|exact Hout].
 all: match goal with |-context[if ?b then _ else _]=>destruct b end;[apply Hsave;assumption|exact Hout].
Qed.
End Sound.

(* CachedChecker *)
Local Close Scope Z_scope.
Import ListNotations.
Record block_cache:={cached_states:list astate;cached_refinements:list bool}.
Definition cached_at c i:=nth i(cached_states c)bottom_state.
Definition cached_cases(inv:list(list block_cache))b:=nth b inv[].
Definition chosen_refinement F (use_refine:bool) out p truth:=if use_refine then refine F out p truth else Some out.
Definition cached_edges F prog b c out:=
 let flag:=fun n=>nth n(cached_refinements c)false in
 match last_instr(block_body prog b)with
 |Some(IBra(Some(neg,p))target)=>[(target,chosen_refinement F(flag 0)out p(negb neg));(S b,chosen_refinement F(flag 1)out p neg)]
 |Some(IBra None target)=>[(target,Some out)]
 |Some(IExit None)=>[]
 |Some(IExit(Some(neg,p)))=>[(S b,chosen_refinement F(flag 0)out p neg)]
 |_=>[(S b,Some out)]end.
Definition cached_node F (T:astate->instr->astate)c body i:=match nth_error body i with
 |Some ins=>wf_instr F ins &&access_of F(cached_at c i)ins &&state_le F(T(cached_at c i)ins)(cached_at c(S i))
 |None=>false end.
Definition cached_block F T prog inv b c:=
 let body:=block_body prog b in
 Nat.eqb(length(cached_states c))(S(length body)) &&
 forallb(cached_node F T c body)(seq 0(length body)) &&
 forallb(fun edge=>let '(v,s):=edge in match s with
 |None=>true|Some out=>if Nat.ltb v(length prog)then existsb(fun target=>state_le F out(cached_at target 0))(cached_cases inv v)else true end)
 (cached_edges F prog b c(cached_at c(length body))).
Definition cached_check F T prog inv:=
 lf_ok F &&structured prog &&existsb(fun c=>entry_case F(cached_at c 0))(cached_cases inv 0) &&
 forallb(fun b=>forallb(cached_block F T prog inv b)(cached_cases inv b))(seq 0(length prog)).

(* CachedSound *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env.
Variable T:astate->instr->astate.
Local Notation stepF := (step prog(wdn F)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Hypothesis T_sound : env_ok F E ->forall t s s' st i,
 fetch prog(blk s)(idx s)=Some i ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F i=true ->
 state_ok F(S t)(T st i)(regs s') /\regs_ok F E(regs s').
Lemma chosen_refinement_sound : forall use_refine t st rs p truth,
 env_ok F E ->regs_ok F E rs ->state_ok F t st rs ->negb(Z.eqb(rs p)0)=truth ->
 exists out,chosen_refinement F use_refine st p truth=Some out /\state_ok F t out rs.
Proof.
 intros use_refine t st rs p truth HE Hr Hst Htruth.
 unfold chosen_refinement. destruct use_refine.
 - exact(refine_sound F E HE t rs Hr st p truth Hst Htruth).
 - exists st;split;[reflexivity|exact Hst].
Qed.
Definition cached_inv_ok inv t s:=
 regs_ok F E(regs s) /\((blk s<length prog)%nat ->
 (idx s<block_len prog(blk s))%nat /\exists c,In c(cached_cases inv(blk s)) /\
 state_ok F t(cached_at c(idx s))(regs s)).
Lemma cached_run_inv : forall inv,
 cached_check F T prog inv=true ->launch_ok F E ->
 forall t s,run' F prog E t=Some s ->cached_inv_ok inv t s.
Proof.
 intros inv Hcheck Hlaunch.
 unfold cached_check in Hcheck. rewrite !andb_true_iff in Hcheck.
 destruct Hcheck as[[[Hlf Hstr]Hentry]Hblocks].
 pose proof(env_of_launch F prog E Hlf Hlaunch)as HE.
 apply existsb_exists in Hentry. destruct Hentry as[entry[Hinentry Hentry]].
 unfold entry_case in Hentry. apply andb_true_iff in Hentry as[Hempty Hderived].
 rewrite forallb_forall in Hblocks.
 assert(Hblock:forall b c,(b<length prog)%nat ->In c(cached_cases inv b) ->cached_block F T prog inv b c=true).
 { intros b c Hb Hc. specialize(Hblocks b ltac:(apply in_seq;lia)). rewrite forallb_forall in Hblocks. exact(Hblocks c Hc). }
 induction t as[|t IH];intros s Hrun.
 - unfold run' in Hrun. cbn[run]in Hrun. injection Hrun as <-.
   split.
   + split;[exact(lk_init F E Hlaunch)|intros r _;reflexivity].
   + intros Hb. cbn[blk idx]. split;[exact(proj1(structured_block F prog E 0 Hstr Hb))|].
     exists entry;split;[exact Hinentry|]. split.
     * intros r x Hx. unfold get in Hx. apply PM.is_empty_2 in Hempty. exfalso;exact(Hempty r x(PM.find_2 _ _ Hx)).
     * apply Forall_forall. intros k Hk. rewrite forallb_forall in Hderived. specialize(Hderived k Hk).
       apply existsb_exists in Hderived. destruct Hderived as[k0[Hk0 Heq]].
       destruct k as[[ts lo]hi],k0 as[[us l]h]. unfold constraint_eqb in Heq.
       rewrite !andb_true_iff in Heq. destruct Heq as[[Ht Hl]Hh]. apply terms_eqb_eq in Ht. apply Z.eqb_eq in Hl,Hh. subst.
       exact(proj1(Forall_forall _ _)(lk_derived F E Hlaunch)_ Hk0).
 - unfold run' in Hrun,IH. cbn[run]in Hrun.
   destruct(run prog(wdn F)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)t)as[s0|]eqn:E0;[|discriminate].
   destruct(fetch prog(blk s0)(idx s0))as[ins|]eqn:Hfetch.
   2:{unfold step in Hrun;rewrite Hfetch in Hrun;discriminate. }
   destruct(IH s0 eq_refl)as[Hrs0 Hinv0].
   destruct(fetch_block F prog E _ _ _ Hfetch)as[Hb Hi].
   destruct(Hinv0 Hb)as[_[entry0[Hentry0 Hst0]]].
   set(b:=blk s0)in *. set(i:=idx s0)in *. set(body:=block_body prog b)in *.
   pose proof(Hblock b entry0 Hb Hentry0)as Hchecked.
   unfold cached_block in Hchecked. fold body in Hchecked. rewrite !andb_true_iff in Hchecked.
   destruct Hchecked as[[Hlen Hnodes]Hedges].
   assert(Hnth:nth_error body i=Some ins)by exact Hfetch.
   rewrite forallb_forall in Hnodes.
   specialize(Hnodes i ltac:(apply in_seq;unfold block_len in Hi;fold body in Hi;lia)).
   unfold cached_node in Hnodes. rewrite Hnth in Hnodes. rewrite !andb_true_iff in Hnodes.
   destruct Hnodes as[[Hwf Haccess]Hpost].
   destruct(T_sound HE t s0 s _ ins Hfetch Hrun Hst0 Hrs0 Hwf)as[Htransfer Hrs1].
   pose proof(state_le_sound F E HE(S t)(regs s)Hrs1 _ _ Htransfer Hpost)as Hst1.
   split;[exact Hrs1|]. intros Hb'.
   assert(Hedge:forall v out,In(v,Some out)(cached_edges F prog b entry0(cached_at entry0(length body))) ->
     (v<length prog)%nat ->state_ok F(S t)out(regs s) ->
     (0<block_len prog v)%nat /\exists target,In target(cached_cases inv v) /\state_ok F(S t)(cached_at target 0)(regs s)).
   { intros v out Hin Hv Hout. rewrite forallb_forall in Hedges. specialize(Hedges _ Hin). cbv beta iota in Hedges.
     destruct(Nat.ltb_spec v(length prog));[|lia]. apply existsb_exists in Hedges. destruct Hedges as[target[Hintarget Hle]].
     split;[exact(proj1(structured_block F prog E v Hstr Hv))|]. exists target;split;[exact Hintarget|].
     exact(state_le_sound F E HE(S t)(regs s)Hrs1 _ _ Hout Hle). }
   assert(Hend:S i=length body ->state_ok F(S t)(cached_at entry0(length body))(regs s)).
   { intros Hl;rewrite <-Hl;exact Hst1. }
   assert(Hlast:is_jump ins=true ->S i=length body)by(exact(proj2(structured_block F prog E b Hstr Hb)i ins Hnth)).
   assert(Hat:forall v,(blk s,idx s)=(v,0%nat) ->(v<length prog)%nat ->
     (0<block_len prog v)%nat /\(exists target,In target(cached_cases inv v) /\state_ok F(S t)(cached_at target 0)(regs s)) ->
     (idx s<block_len prog(blk s))%nat /\exists target,In target(cached_cases inv(blk s)) /\state_ok F(S t)(cached_at target(idx s))(regs s)).
   {intros v Hpos _ H;injection Hpos as -> ->;exact H. }
   destruct(step_pos F prog E t s0 s ins Hfetch Hrun)as[[Hnext[Hbra Hexit]]|[[g[tg[-> [Hg Hpos]]]]|[g[-> [Hg Hpos]]]]].
   + unfold next_pos in Hnext. fold b i in Hnext.
     destruct(Nat.ltb_spec(S i)(block_len prog b))as[Hlt|Hge].
     * injection Hnext as -> ->. split;[exact Hlt|]. exists entry0;split;[exact Hentry0|exact Hst1].
     * injection Hnext as Hbs His. rewrite Hbs in Hb'.
       assert(Hl:S i=length body)by(unfold block_len in Hge,Hi;fold body in Hge,Hi;lia).
       apply(Hat(S b));[rewrite Hbs,His;reflexivity|exact Hb'|].
       pose proof(last_of_nth F prog E body i ins Hnth Hl)as Hlastins.
       unfold cached_edges in Hedge. fold body in Hedge. rewrite Hlastins in Hedge. cbv zeta in Hedge.
       destruct ins as[| | | | | | | | | | | | | | |g tg|g| | | | | | | |];try(apply(Hedge(S b)_ (or_introl eq_refl)Hb'(Hend Hl))).
       -- specialize(Hbra g tg eq_refl). rewrite(jump_regs F prog E t s0 s _ Hfetch eq_refl Hrun)in *.
          destruct g as[[neg p]|];[|discriminate]. cbn[holds]in Hbra.
          destruct(chosen_refinement_sound(nth 1(cached_refinements entry0)false)(S t)_ (regs s0)p neg HE Hrs0(Hend Hl))as[out[Eref Hout]].
          {destruct neg,(Z.eqb(regs s0 p)0);cbn in Hbra|- *;congruence. }
          apply(Hedge(S b)out);[rewrite Eref;right;left;reflexivity|exact Hb'|exact Hout].
       -- specialize(Hexit g eq_refl). rewrite(jump_regs F prog E t s0 s _ Hfetch eq_refl Hrun)in *.
          destruct g as[[neg p]|];[|discriminate]. cbn[holds]in Hexit.
          destruct(chosen_refinement_sound(nth 0(cached_refinements entry0)false)(S t)_ (regs s0)p neg HE Hrs0(Hend Hl))as[out[Eref Hout]].
          {destruct neg,(Z.eqb(regs s0 p)0);cbn in Hexit|- *;congruence. }
          apply(Hedge(S b)out);[rewrite Eref;left;reflexivity|exact Hb'|exact Hout].
   + pose proof Hpos as Hpos'. injection Hpos' as Hbs _. rewrite Hbs in Hb'.
     apply(Hat tg Hpos Hb'). pose proof(Hlast eq_refl)as Hl.
     pose proof(last_of_nth F prog E body i _ Hnth Hl)as Hlastins.
     rewrite(jump_regs F prog E t s0 s _ Hfetch eq_refl Hrun)in *.
     unfold cached_edges in Hedge. fold body in Hedge. rewrite Hlastins in Hedge. cbv zeta in Hedge.
     destruct g as[[neg p]|].
     * cbn[holds]in Hg.
       destruct(chosen_refinement_sound(nth 0(cached_refinements entry0)false)(S t)_ (regs s0)p(negb neg)HE Hrs0(Hend Hl))as[out[Eref Hout]].
       {destruct neg,(Z.eqb(regs s0 p)0);cbn in Hg|- *;congruence. }
       apply(Hedge tg out);[rewrite Eref;left;reflexivity|exact Hb'|exact Hout].
     * apply(Hedge tg _ (or_introl eq_refl)Hb'(Hend Hl)).
   + injection Hpos as Hbs _. rewrite Hbs in Hb';lia.
Qed.
End Sound.

(* CachedSafety *)
Import ListNotations. Local Open Scope Z_scope.
Section Sound.
Variable F:lfacts. Variable prog:program. Variable E:env.
Variable T:astate->instr->astate.
Local Notation stepF := (step prog(wdn F)(e_mem E)(e_arb E)(e_shm E)(e_init E)(lf_cells F)(lf_locals F)(lf_local_window F)).
Hypothesis T_sound : env_ok F E ->forall t s s' st i,
 fetch prog(blk s)(idx s)=Some i ->stepF t s=Some s' ->
 state_ok F t st(regs s) ->regs_ok F E(regs s) ->wf_instr F i=true ->
 state_ok F(S t)(T st i)(regs s') /\regs_ok F E(regs s').
Theorem cached_check_sound : forall inv,
 cached_check F T prog inv=true ->launch_ok F E ->
 forall t s i,run' F prog E t=Some s ->fetch prog(blk s)(idx s)=Some i ->
 (forall k addr width v,access_at(in_window F E)(regs s)i=Some(k,addr,width,v) ->region F E t k addr width v) /\
 (forall addr width,mem_access(regs s)i=Some(addr,width) ->aligned(need width)addr) /\
 (forall p,private_access(in_window F E)(regs s)i=Some p ->private_in F E p) /\wf_instr F i=true.
Proof.
 intros inv Hcheck Hlaunch t s i Hrun Hfetch.
 pose proof Hcheck as Hcheck'. unfold cached_check in Hcheck'. rewrite !andb_true_iff in Hcheck'.
 destruct Hcheck' as[[[Hlf Hstr]_ ]Hblocks].
 pose proof(env_of_launch F prog E Hlf Hlaunch)as HE.
 destruct(cached_run_inv F prog E T T_sound inv Hcheck Hlaunch t s Hrun)as[Hrs Hinv].
 destruct(fetch_block F prog E _ _ _ Hfetch)as[Hb Hi]. destruct(Hinv Hb)as[_[entry[Hentry Hst]]].
 rewrite forallb_forall in Hblocks. specialize(Hblocks(blk s)ltac:(apply in_seq;lia)).
 rewrite forallb_forall in Hblocks. specialize(Hblocks entry Hentry).
 unfold cached_block in Hblocks. rewrite !andb_true_iff in Hblocks. destruct Hblocks as[[_ Hnodes]_].
 rewrite forallb_forall in Hnodes. specialize(Hnodes(idx s)ltac:(apply in_seq;unfold block_len in Hi;lia)).
 unfold cached_node in Hnodes. change(nth_error(block_body prog(blk s))(idx s)=Some i)in Hfetch.
 rewrite Hfetch in Hnodes. rewrite !andb_true_iff in Hnodes. destruct Hnodes as[[Hwf Haccess]_].
 split;[|split;[|split]].
 - intros k addr width v Hacc. exact(access_of_sound F E HE t(regs s)Hrs _ i k addr width v Hst Hwf Haccess Hacc).
 - intros addr width Hacc. exact(aligned_of_sound F E HE t(regs s)Hrs _ i addr width Hst Haccess Hacc).
 - intros p Hacc. exact(private_of_sound F E HE t(regs s)Hrs _ i p Hst Hwf Haccess Hacc).
 - exact Hwf.
Qed.
End Sound.

(* CoherentCertificate *)
Local Close Scope Z_scope.
Theorem coherent_cached_check_sound : forall F prog E written goals desc inv,
 cached_check F(coherent_transfer F written goals desc)prog inv=true ->launch_ok F E ->
 forall t s i,run' F prog E t=Some s ->fetch prog(blk s)(idx s)=Some i ->
 (forall k addr width v,access_at(in_window F E)(regs s)i=Some(k,addr,width,v) ->region F E t k addr width v) /\
 (forall addr width,mem_access(regs s)i=Some(addr,width) ->aligned(need width)addr) /\
 (forall p,private_access(in_window F E)(regs s)i=Some p ->private_in F E p) /\wf_instr F i=true.
Proof.
 intros F prog E written goals desc inv Hcheck Hlaunch.
 refine(cached_check_sound F prog E(coherent_transfer F written goals desc) _ inv Hcheck Hlaunch).
 intros HE t s s' st i Hf Hstep Hst Hrs Hwf.
 exact(coherent_transfer_sound F prog E HE t s s' st written goals desc i Hf Hstep Hst Hrs Hwf).
Qed.

(* IndexedChecker *)
Local Close Scope Z_scope.
Import ListNotations.
Definition selected_target F (prog:program) inv k edge:=let '(v,s):=edge in match s with
|None=>true|Some out=>if Nat.ltb v(length prog)then
 match nth_error(cached_cases inv v)k with Some target=>state_le F out(cached_at target 0)|None=>false end
 else true end.
Fixpoint indexed_edges F prog inv picks edges:=match picks,edges with
|[],[]=>true|k::ks,e::es=>selected_target F prog inv k e&&indexed_edges F prog inv ks es|_,_=>false end.
Definition indexed_block F T prog inv b c picks:=
 let body:=block_body prog b in
 Nat.eqb(length(cached_states c))(S(length body)) &&
 forallb(cached_node F T c body)(seq 0(length body)) &&
 indexed_edges F prog inv picks(cached_edges F prog b c(cached_at c(length body))).
Fixpoint indexed_cases F T prog inv b picks cases:=match picks,cases with
|[],[]=>true|p::ps,c::cs=>indexed_block F T prog inv b c p&&indexed_cases F T prog inv b ps cs|_,_=>false end.
Definition indexed_check F T prog inv picks:=
 lf_ok F&&structured prog&&existsb(fun c=>entry_case F(cached_at c 0))(cached_cases inv 0)&&
 forallb(fun b=>indexed_cases F T prog inv b(nth b picks[])(cached_cases inv b))(seq 0(length prog)).
Lemma indexed_edges_implies : forall F prog inv picks edges,
 indexed_edges F prog inv picks edges=true ->
 forallb(fun edge=>let '(v,s):=edge in match s with None=>true|Some out=>if Nat.ltb v(length prog)then existsb(fun target=>state_le F out(cached_at target 0))(cached_cases inv v)else true end)edges=true.
Proof.
 intros F prog inv picks. induction picks as[|k ks IH];intros edges H;destruct edges as[|[v out]edges];try discriminate;[reflexivity|].
 cbn[indexed_edges]in H. apply andb_true_iff in H as[Ht Hr]. cbn[forallb]. apply andb_true_iff;split;[|eapply IH;exact Hr].
 unfold selected_target in Ht. destruct out as[out|];[|reflexivity]. destruct(Nat.ltb v(length prog));[|reflexivity].
 destruct(nth_error(cached_cases inv v)k)as[target|]eqn:E;[|discriminate]. apply existsb_exists.
 exists target;split;[eapply nth_error_In;exact E|exact Ht].
Qed.
Lemma indexed_block_implies : forall F T prog inv b c picks,
 indexed_block F T prog inv b c picks=true ->cached_block F T prog inv b c=true.
Proof.
 intros. unfold indexed_block in H. unfold cached_block. apply andb_true_iff in H as[Hnodes Hedge].
 apply andb_true_iff;split;[exact Hnodes|eapply indexed_edges_implies;exact Hedge].
Qed.
Lemma indexed_cases_implies : forall F T prog inv b picks cases,
 indexed_cases F T prog inv b picks cases=true ->forallb(cached_block F T prog inv b)cases=true.
Proof.
 intros F T prog inv b picks. induction picks as[|p ps IH];intros cases H;destruct cases as[|c cs];try discriminate;[reflexivity|].
 cbn[indexed_cases]in H. apply andb_true_iff in H as[Hc Hrest]. cbn[forallb]. apply andb_true_iff;split;[eapply indexed_block_implies;exact Hc|eapply IH;exact Hrest].
Qed.
Theorem indexed_check_implies : forall F T prog inv picks,
 indexed_check F T prog inv picks=true ->cached_check F T prog inv=true.
Proof.
 intros F T prog inv picks H. unfold indexed_check in H. unfold cached_check.
 apply andb_true_iff in H as[Hfront Hblocks]. apply andb_true_iff;split;[exact Hfront|].
 rewrite forallb_forall in *. intros b Hb. eapply indexed_cases_implies. apply Hblocks;exact Hb.
Qed.

(* CachedApi *)
Local Close Scope Z_scope.
Import ListNotations.
Record cached_certificate:={
 cert_written:list reg;
 cert_goals:list constraint;
 cert_descriptor:option interval_descriptor;
 cert_blocks:list(list block_cache);
 cert_targets:list(list(list nat))
}.
Definition check_cached_certificate F prog cert:=
 indexed_check F(coherent_transfer F(cert_written cert)(cert_goals cert)(cert_descriptor cert))prog(cert_blocks cert)(cert_targets cert).
Theorem check_cached_certificate_safe : forall F prog cert,
 check_cached_certificate F prog cert=true ->kernel_safe F prog.
Proof.
 intros F prog cert Hcheck E Hlaunch. unfold check_cached_certificate in Hcheck.
 apply indexed_check_implies in Hcheck.
 exact(coherent_cached_check_sound F prog E(cert_written cert)(cert_goals cert)(cert_descriptor cert)(cert_blocks cert)Hcheck Hlaunch).
Qed.
Theorem check_cached_certificate_lf_ok : forall F prog cert,
 check_cached_certificate F prog cert=true ->lf_ok F=true.
Proof.
 intros F prog cert H. unfold check_cached_certificate,indexed_check in H.
 rewrite !andb_true_iff in H. tauto.
Qed.


(* Source structure chooses this certificate family before any semantic check. *)
Definition interval_loop_candidates (prog:program):list(nat*nat):=
 flat_map(fun b=>flat_map(fun i=>match i with IBra _ h=>
 if Nat.ltb h b then [(h,b)]else[]|_=>[]end)(block_body prog b))(seq 0(length prog)).
Definition interval_loop_body (prog:program) (h t:nat):=
 concat(firstn(S(t-h))(skipn h prog)).
Definition interval_has_half body:=existsb(fun i=>match i with
 |IBin None BShrs 64 _ _ (OImm z)=>Z.eqb z 1|_=>false end)body.
Definition interval_shape_at (prog:program) (pair:nat*nat):option(nat*nat*interval_descriptor):=
 let '(h,t):=pair in
 if Nat.eqb t(S(S h)) && interval_has_half(block_body prog h)then
 match last_instr(block_body prog h),last_instr(block_body prog t),interval_controller h(block_body prog t)with
 |Some(IBra(Some _)skip),Some(IBra(Some(false,_))back),Some d=>
 if Nat.eqb skip t&&Nat.eqb back h then Some(h,t,d)else None
 |_,_,_=>None end else None.
Fixpoint interval_first_shape prog pairs:=match pairs with []=>None|item::pairs=>
 match interval_shape_at prog item with Some shape=>Some shape|None=>interval_first_shape prog pairs end end.
Definition interval_shape (prog:program):=interval_first_shape prog(interval_loop_candidates prog).
(* Capability prerequisites for the compact interval template and forward
   exit expansion. These inspect source structure, never checker outcomes. *)
Definition interval_entry_guard (prog:program) (h t:nat) (d:interval_descriptor):bool :=
 Nat.leb 2 h &&
 match rev(block_body prog(h-2)),block_body prog(h-1) with
 |IBra(Some(neg,p))exit::ISetp None cmp 64 true q a(OReg b)::_,[IMov None saved low]=>
 Pos.eqb p q && Nat.ltb t exit && Pos.eqb saved(id_copy d) && Pos.eqb low(id_low d) &&
 match cmp,neg with
 |CLe,false|CGt,true=>Pos.eqb a(id_high d)&&Pos.eqb b(id_low d)
 |CGe,false|CLt,true=>Pos.eqb a(id_low d)&&Pos.eqb b(id_high d)
 |_,_=>false end
 |_,_=>false end.
Definition interval_forward_remainder (prog:program) (h t:nat):bool :=
 forallb(fun b=>forallb(fun v=>Nat.ltb b v || (Nat.eqb b t&&Nat.eqb v h))
 (live_successors prog b))(seq 0(length prog)).
Definition interval_middle_fallthrough (prog:program) (h t:nat):bool :=
 match live_successors prog(S h)with [v]=>Nat.eqb v t|_=>false end.

Definition interval_edge_to F prog b target out:=
 match find(fun edge=>Nat.eqb(fst edge)target)(edge_states F prog b out)with
 |Some(_,state)=>state|None=>None end.
Fixpoint interval_preheader fuel F prog table incoming b st:=match fuel with
 |0%nat=>Some st
 |S fuel=>match interval_edge_to F prog b(S b)(walk F st(block_body prog b))with
 |Some next=>interval_preheader fuel F prog table incoming(S b)
 (trim_join_live_state F table incoming(S b)next)
 |None=>None end end.
Fixpoint interval_cache_walk (T:astate->instr->astate)(st:astate)(body:list instr):list astate:=st::match body with
 |[]=>[]|i::rest=>interval_cache_walk T(T st i)rest end.
Definition interval_make_cache T st body flags:={|
 cached_states:=interval_cache_walk T st body;cached_refinements:=flags|}.
Definition interval_cache_out c:=last(cached_states c)bottom_state.
Fixpoint interval_prefix fuel F T prog table incoming b st:=match fuel with
 |0%nat=>Some([],st)
 |S fuel=>let c:=interval_make_cache T st(block_body prog b)[true;true]in
 match interval_edge_to F prog b(S b)(interval_cache_out c)with
 |Some out=>let next:=trim_join_live_state F table incoming(S b)out in
 match interval_prefix fuel F T prog table incoming(S b)next with
 |Some(rest,last)=>Some((b,c)::rest,last)|None=>None end
 |None=>None end end.
Definition interval_present (base:list(nat*block_cache)) b:=existsb(fun entry=>Nat.eqb(fst entry)b)base.
Definition interval_frontier_flags F prog base b c out:=
 let raw:={|cached_states:=cached_states c;cached_refinements:=[]|}in
 let targets:=map fst(cached_edges F prog b raw out)in
 {|cached_states:=cached_states c;cached_refinements:=map(fun k=>
 negb(interval_present base(nth k targets 0))&&nth k(cached_refinements c)false)(seq 0(length targets))|}.
Definition interval_frontier F prog base:=flat_map(fun entry=>let '(b,c):=entry in
 let out:=interval_cache_out c in
 let selected:=interval_frontier_flags F prog base b c out in
 flat_map(fun edge=>let '(v,out):=edge in if interval_present base v then[]else
 match out with Some st=>if Nat.ltb v(length prog)then[(v,st)]else[]|None=>[]end)
 (cached_edges F prog b selected out))base.
Definition interval_prepared_entry:=((nat*block_cache)*list(nat*option astate))%type.
Definition interval_remember_entry F prog (entry:nat*block_cache):interval_prepared_entry:=
 let '(b,c):=entry in (entry,cached_edges F prog b c(cached_at c(length(block_body prog b)))).
Definition interval_next_edges (prog:program)(fixed:list nat)b (edges:list(nat*option astate)):=
 if forallb(fun edge=>let '(v,out):=edge in match out with None=>true|Some _=>
 negb(Nat.ltb v(length prog))||existsb(Nat.eqb v)fixed||Nat.ltb b v end)edges then
 Some(flat_map(fun edge=>let '(v,out):=edge in match out with None=>[]|Some out=>
 if Nat.ltb v(length prog)&&negb(existsb(Nat.eqb v)fixed)then[(v,out)]else[]end)edges)
 else None.
Fixpoint interval_expand_prepared fuel F T prog fixed pending(acc:list interval_prepared_entry):option(list interval_prepared_entry):=match pending with
 |[]=>Some acc|(b,st)::rest=>match fuel with 0%nat=>None|S fuel=>
 if Nat.ltb b(length prog)then
 let c:=interval_make_cache T st(block_body prog b)[true;true]in
 let edges:=cached_edges F prog b c(cached_at c(length(block_body prog b)))in
 match interval_next_edges prog fixed b edges with
 |Some next=>interval_expand_prepared fuel F T prog fixed(next++rest)(((b,c),edges)::acc)
 |None=>None end else None end end.
Definition interval_assemble (prog:program)(entries:list(nat*block_cache)):=map(fun b=>
 map snd(filter(fun entry=>Nat.eqb(fst entry)b)entries))(seq 0(length prog)).
Fixpoint interval_mapM {A B}(f:A->option B)(xs:list A):option(list B):=match xs with
 |[]=>Some[]|x::xs=>match f x with None=>None|Some y=>
 match interval_mapM f xs with Some ys=>Some(y::ys)|None=>None end end end.
Fixpoint interval_exact_case out cases:=match cases with
 |[]=>None|c::cs=>if state_eqb out(cached_at c 0)then Some 0%nat
 else option_map S(interval_exact_case out cs)end.
Definition interval_edge_pick (prog:program) inv edge:=let '(v,out):=edge in
 match out with None=>Some 0%nat|Some out=>if Nat.ltb v(length prog)then
 match cached_cases inv v with []=>None|[_]=>Some 0%nat
 |cases=>interval_exact_case out cases end else Some 0%nat end.
Definition interval_prepared_picks (F:lfacts) prog (entries:list interval_prepared_entry):=
 let inv:=interval_assemble prog(map fst entries)in
 interval_mapM(fun b=>interval_mapM(fun e=>interval_mapM(interval_edge_pick prog inv)(snd e))
 (filter(fun e=>Nat.eqb(fst(fst e))b)entries))(seq 0(length prog)).
Definition interval_terminal_cache:={|
 cached_states:=[{|amap:=PM.empty fact;acons:=[]|};{|amap:=PM.empty fact;acons:=[]|}];
 cached_refinements:=[]|}.
Definition interval_terminals prog:=flat_map(fun b=>match block_body prog b with
 |[IExit None]=>[(b,interval_terminal_cache)]|_=>[]end)(seq 0(length prog)).
(* Preparation precedes certificate-mode selection and is reused once.
   Missing/bottom entry or missing endpoint ranges means this template is not
   available; it is not a failed certificate check. *)
Record interval_preparation:={
 ip_head:nat; ip_tail:nat; ip_descriptor:interval_descriptor;
 ip_written:list reg; ip_table:list(PM.t unit); ip_incoming:list(PM.t unit);
 ip_preheader:astate; ip_template:astate
}.
Definition interval_prepare (F:lfacts)(prog:program):option interval_preparation:=
 match interval_shape prog with None=>None|Some(h,t,d)=>
 if interval_entry_guard prog h t d && interval_forward_remainder prog h t &&
    interval_middle_fallthrough prog h t then
 let written:=flat_map(instruction_support_outputs F)(interval_loop_body prog h t)in
 let table:=live_program F prog in let incoming:=live_incoming_controls prog in
 match interval_preheader h F prog table incoming 0(empty_state F)with
 |None=>None|Some pre=>if is_bottom pre then None else
 match invariant_template F written pre d with None=>None|Some candidate=>
 Some{|ip_head:=h;ip_tail:=t;ip_descriptor:=d;ip_written:=written;
       ip_table:=table;ip_incoming:=incoming;ip_preheader:=pre;ip_template:=candidate|}
 end end else None end.
Definition interval_candidate_selected (F:lfacts)(prog:program):bool:=
 match interval_prepare F prog with Some _=>true|None=>false end.
Definition propose_prepared_interval (F:lfacts)(prog:program)(prepared:interval_preparation):option cached_certificate:=
 let h:=ip_head prepared in let t:=ip_tail prepared in let d:=ip_descriptor prepared in
 let written:=ip_written prepared in let table:=ip_table prepared in let incoming:=ip_incoming prepared in let pre:=ip_preheader prepared in let candidate:=ip_template prepared in
 let goals:=interval_constraints F pre d in let T:=coherent_transfer F written goals(Some d)in
 match interval_prefix h F T prog table incoming 0(empty_state F)with None=>None|Some(prefix,_)=>
 let head:=interval_make_cache T candidate(block_body prog h)[false;true]in
 match find(fun edge=>Nat.eqb(fst edge)(S h))(cached_edges F prog h head(interval_cache_out head))with
 |Some(_,Some mem_pre)=>
 let memory:=interval_make_cache T mem_pre(block_body prog(S h))[true;true]in
 let tail_memory:=interval_make_cache T(interval_cache_out memory)(block_body prog t)[true;true]in
 let tail_skip:=interval_make_cache T(interval_cache_out head)(block_body prog t)[true;true]in
 let base:=prefix++[(h,head);(S h,memory);(t,tail_memory);(t,tail_skip)]++interval_terminals prog in
 match interval_expand_prepared(8*length prog)F T prog(map fst base)(interval_frontier F prog base)(map(interval_remember_entry F prog)base)with
 |None=>None|Some entries=>let inv:=interval_assemble prog(map fst entries)in
 match interval_prepared_picks F prog entries with None=>None|Some picks=>
 Some{|cert_written:=written;cert_goals:=goals;cert_descriptor:=Some d;cert_blocks:=inv;cert_targets:=picks|}end end
 |_=>None end end.
Definition propose_cached_certificate (F:lfacts)(prog:program):option cached_certificate:=
 match interval_prepare F prog with
 |Some prepared=>propose_prepared_interval F prog prepared|None=>None end.

Print Assumptions check_cached_certificate_safe.
Print Assumptions check_cached_certificate_lf_ok.
