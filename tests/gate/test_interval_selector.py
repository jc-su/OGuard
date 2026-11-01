"""Interval proposal capability excludes unsupported entry and residual CFG shapes.

The gate also proves that the kernel ends (DESIGN decision 40). With bounds that
the kernel reads from the clock, whose values it does not choose, the loop runs forever when `lo` and `copy` start
at -1 below `hi`, and `bra EXIT` ends a kernel in a loop of its own; the gate
refuses those fixtures for termination. It does not prove that `guarded_entry`'s loop
ends either: that needs `copy = lo` at the head, which no rank expresses.
"""

import subprocess

import pytest

from opgate.binding import Binding
from opgate.gate import CHECK_QUERY, export_terms, formal_dir, zeroed
from opgate.signature import Affine

BINDING = Binding("kernel", (), (), (1, 1, 1), (1, 1, 1), Affine(0))
NOT_PROVED_TO_END = {
    "unknown_entry",
    "exit_cycle",
    "guarded_entry",
    "guarded_unknown",
    "guarded_exit_cycle",
}

FIXTURES = {
    "equal_entry": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,0;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
ret;
}
""",
    "unknown_entry": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,%clock64;
mov.b64 %hi,%clock64;
mov.b64 %copy,%clock64;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
ret;
}
""",
    "supported_entry": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,1;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
ret;
}
""",
    "exit_cycle": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,1;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
EXIT:
bra EXIT;
}
""",
    "no_controller": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,1;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.lt.s64 %back,%lo,%hi;
mov.b64 %copy,%lo;
@%back bra HEAD;
ret;
}
""",
    "guarded_entry": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,1;
setp.le.s64 %skip,%hi,%lo;
@%skip bra END;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
END:
ret;
}
""",
    "guarded_unknown": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,%clock64;
mov.b64 %hi,%clock64;
setp.le.s64 %skip,%hi,%lo;
@%skip bra END;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
END:
ret;
}
""",
    "guarded_equal": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,0;
setp.le.s64 %skip,%hi,%lo;
@%skip bra END;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
END:
ret;
}
""",
    "guarded_exit_cycle": r""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {
.reg .b64 %lo,%hi,%copy,%half;
.reg .pred %skip,%back;
mov.b64 %lo,0;
mov.b64 %hi,1;
setp.le.s64 %skip,%hi,%lo;
@%skip bra END;
mov.b64 %copy,%lo;
HEAD:
shr.s64 %half,%copy,1;
setp.eq.s64 %skip,%copy,-1;
@%skip bra TAIL;
mov.b64 %lo,%hi;
TAIL:
setp.gt.s64 %back,%hi,%lo;
mov.b64 %copy,%lo;
@%back bra HEAD;
END:
bra END;
}
""",
}
METADATA = r"""
From OpGate Require Import GateInterval.
Definition parts_o:=Eval vm_compute in match parse_text text size with
|Ok ks=>match ks_entries ks,mfs with
|(_,k)::_,(_,mf)::_=>match front k mf with Ok(prog,F)=>Some(prog,F)|_=>None end
|_,_=>None end|_=>None end.
Definition parts:=ltac:(let x:=eval vm_compute in parts_o in lazymatch x with Some ?p=>exact p|_=>fail "front refused" end).
Definition prog:=fst parts. Definition F:=snd parts.
"""
PRESERVATION = r"""From Coq Require Import List Bool ZArith Lia.
From OpGate Require Import GateIsa GateForm GateDomain GateInterval GateCertificate.
Import ListNotations. Local Close Scope Z_scope.
Definition interval_next F prog fixed b c:option(list(nat*astate)):=
 let edges:=cached_edges F prog b c(interval_cache_out c)in
 if forallb(fun edge=>let '(v,out):=edge in match out with None=>true|Some _=>
 negb(Nat.ltb v(length prog))||existsb(Nat.eqb v)fixed||Nat.ltb b v end)edges then
 Some(flat_map(fun edge=>let '(v,out):=edge in match out with None=>[]|Some out=>
 if Nat.ltb v(length prog)&&negb(existsb(Nat.eqb v)fixed)then[(v,out)]else[]end)edges)
 else None.
Fixpoint interval_expand fuel F T prog fixed pending acc:=match pending with
 |[]=>Some acc|(b,st)::rest=>match fuel with 0%nat=>None|S fuel=>
 if Nat.ltb b(length prog)then
 let c:=interval_make_cache T st(block_body prog b)[true;true]in
 match interval_next F prog fixed b c with
 |Some next=>interval_expand fuel F T prog fixed(next++rest)((b,c)::acc)
 |None=>None end else None end end.
Definition interval_picks F prog inv:=interval_mapM(fun b=>interval_mapM(fun c=>
 interval_mapM(interval_edge_pick prog inv)(cached_edges F prog b c(cached_at c(length(block_body prog b)))))
 (cached_cases inv b))(seq 0(length prog)).
Definition literal_prepared_interval (F:lfacts)(prog:program)(prepared:interval_preparation)
 :option cached_certificate:=
 let h:=ip_head prepared in let t:=ip_tail prepared in let d:=ip_descriptor prepared in
 let written:=ip_written prepared in let table:=ip_table prepared in
 let incoming:=ip_incoming prepared in let pre:=ip_preheader prepared in
 let candidate:=ip_template prepared in
 let goals:=interval_constraints F pre d in
 let T:=coherent_transfer F written goals(Some d)in
 match interval_prefix h F T prog table incoming 0(empty_state F)with None=>None|Some(prefix,_)=>
 let head:=interval_make_cache T candidate(block_body prog h)[false;true]in
 match find(fun edge=>Nat.eqb(fst edge)(S h))(cached_edges F prog h head(interval_cache_out head))with
 |Some(_,Some mem_pre)=>
 let memory:=interval_make_cache T mem_pre(block_body prog(S h))[true;true]in
 let tail_memory:=interval_make_cache T(interval_cache_out memory)(block_body prog t)[true;true]in
 let tail_skip:=interval_make_cache T(interval_cache_out head)(block_body prog t)[true;true]in
 let base:=prefix++[(h,head);(S h,memory);(t,tail_memory);(t,tail_skip)]++interval_terminals prog in
 match interval_expand(8*length prog)F T prog(map fst base)(interval_frontier F prog base)base with
 |None=>None|Some entries=>let inv:=interval_assemble prog entries in
 match interval_picks F prog inv with None=>None|Some picks=>
 Some{|cert_written:=written;cert_goals:=goals;cert_descriptor:=Some d;
 cert_blocks:=inv;cert_targets:=picks|}end end
 |_=>None end end.
Definition entry_prepared F prog (entry:interval_prepared_entry):Prop:=
 snd entry=cached_edges F prog(fst(fst entry))(snd(fst entry))(cached_at(snd(fst entry))(length(block_body prog(fst(fst entry))))).
Lemma remember_entry_prepared : forall F prog x,entry_prepared F prog(interval_remember_entry F prog x).
Proof. intros F prog [b c];reflexivity. Qed.
Lemma interval_make_cache_out : forall T st body flags,
 interval_cache_out(interval_make_cache T st body flags)=cached_at(interval_make_cache T st body flags)(length body).
Proof.
 intros T st body;revert st. induction body as[|i body IH];intros st flags.
 - reflexivity.
 - cbn[interval_cache_out interval_make_cache cached_at cached_states interval_cache_walk length nth last].
   pose proof(IH(T st i)flags)as H. unfold interval_cache_out,interval_make_cache,cached_at in H;cbn[cached_states]in H.
   destruct(interval_cache_walk T(T st i)body)eqn:E;[destruct body;discriminate|exact H].
Qed.
Lemma interval_expand_prepared_erases : forall fuel F T prog fixed pending acc,
 option_map(map fst)(interval_expand_prepared fuel F T prog fixed pending acc)=
 interval_expand fuel F T prog fixed pending(map fst acc).
Proof.
 induction fuel as[|fuel IH];intros F T prog fixed pending acc;destruct pending as[|[b st]rest];cbn[interval_expand_prepared interval_expand];try reflexivity.
 destruct(Nat.ltb b(length prog));[|reflexivity]. unfold interval_next. rewrite interval_make_cache_out.
 fold(interval_next_edges prog fixed b(cached_edges F prog b(interval_make_cache T st(block_body prog b)[true;true])(cached_at(interval_make_cache T st(block_body prog b)[true;true])(length(block_body prog b))))).
 destruct(interval_next_edges _ _ _ _)as[next|];[|reflexivity]. apply IH.
Qed.
Lemma interval_expand_prepared_correct : forall fuel F T prog fixed pending acc out,
 Forall(entry_prepared F prog)acc ->interval_expand_prepared fuel F T prog fixed pending acc=Some out ->Forall(entry_prepared F prog)out.
Proof.
 induction fuel as[|fuel IH];intros F T prog fixed pending acc out Ha H;destruct pending as[|[b st]rest];cbn[interval_expand_prepared]in H;try discriminate;try(injection H as <-;exact Ha).
 destruct(Nat.ltb b(length prog));[|discriminate]. destruct(interval_next_edges _ _ _ _)as[next|];[|discriminate].
 eapply IH;[|exact H]. constructor;[reflexivity|exact Ha].
Qed.
Lemma interval_mapM_map : forall A B C(f:A->B)(g:B->option C)xs,
 interval_mapM g(map f xs)=interval_mapM(fun x=>g(f x))xs.
Proof. intros A B C f g xs;induction xs as[|x xs IH];cbn[map interval_mapM];[reflexivity|]. destruct(g(f x));[rewrite IH|];reflexivity. Qed.
Lemma interval_mapM_ext : forall A B(f g:A->option B)xs,
 (forall x,In x xs->f x=g x)->interval_mapM f xs=interval_mapM g xs.
Proof. intros A B f g xs;induction xs as[|x xs IH];intro H;cbn[interval_mapM];[reflexivity|].
 rewrite(H x(or_introl eq_refl)). destruct(g x);[rewrite IH|];try reflexivity. intros y Hy;apply H;right;exact Hy. Qed.
Lemma prepared_filter : forall entries b,
 map snd(filter(fun e=>Nat.eqb(fst e)b)(map fst entries))=
 map(fun e:interval_prepared_entry=>snd(fst e))(filter(fun e=>Nat.eqb(fst(fst e))b)entries).
Proof. induction entries as[|[[b c]edges]entries IH];intros k;cbn;[reflexivity|].
 destruct(Nat.eqb b k);cbn;rewrite IH;reflexivity. Qed.
Lemma interval_prepared_picks_same : forall F prog entries,
 Forall(entry_prepared F prog)entries ->
 interval_prepared_picks F prog entries=interval_picks F prog(interval_assemble prog(map fst entries)).
Proof.
 intros F prog entries Hcorrect. unfold interval_prepared_picks,interval_picks.
 apply interval_mapM_ext. intros b Hb. apply in_seq in Hb.
 assert(Hcases:cached_cases(interval_assemble prog(map fst entries))b=
   map(fun e:interval_prepared_entry=>snd(fst e))(filter(fun e=>Nat.eqb(fst(fst e))b)entries)).
 { unfold cached_cases,interval_assemble.
   set(select:=fun k=>map snd(filter(fun e=>Nat.eqb(fst e)k)(map fst entries))).
   change(nth b(map select(seq 0(length prog)))[]= map(fun e:interval_prepared_entry=>snd(fst e))(filter(fun e=>Nat.eqb(fst(fst e))b)entries)).
   rewrite(@nth_indep (list block_cache)(map select(seq 0(length prog))) b [] (select 0))by(rewrite map_length,seq_length;lia).
   rewrite map_nth,seq_nth by lia. cbn[Nat.add]. unfold select. apply prepared_filter. }
 rewrite Hcases.
 rewrite interval_mapM_map. apply interval_mapM_ext. intros e He.
 apply filter_In in He as[He Hblock]. apply Nat.eqb_eq in Hblock.
 rewrite Forall_forall in Hcorrect. specialize(Hcorrect e He).
 unfold entry_prepared in Hcorrect. rewrite Hcorrect,Hblock. reflexivity.
Qed.
Lemma remember_entries_erase : forall F prog entries,
 map fst(map(interval_remember_entry F prog)entries)=entries.
Proof. intros F prog entries;induction entries as[|[b c]entries IH];cbn;[reflexivity|]. now rewrite IH. Qed.
Lemma remember_entries_correct : forall F prog entries,
 Forall(entry_prepared F prog)(map(interval_remember_entry F prog)entries).
Proof. intros F prog entries;apply Forall_map;apply Forall_forall;intros x _;apply remember_entry_prepared. Qed.
Theorem propose_prepared_edges_same : forall F prog prepared,
 propose_prepared_interval F prog prepared=literal_prepared_interval F prog prepared.
Proof.
 intros F prog prepared. unfold propose_prepared_interval,literal_prepared_interval.
 destruct(interval_prefix _ _ _ _ _ _ _ _)as[[prefix final]|]eqn:Ep;[|reflexivity].
 destruct(find _ _)as[[target [mem_pre|]]|]eqn:Em;try reflexivity.
 match goal with |-context[interval_expand_prepared ?fuel ?f ?t ?p ?fixed ?pending ?acc]=>
 pose proof(interval_expand_prepared_erases fuel f t p fixed pending acc)as Heq end.
 rewrite remember_entries_erase in Heq.
 destruct(interval_expand_prepared _ _ _ _ _ _ _)as[entries|]eqn:Ex.
 - cbn[option_map]in Heq. rewrite <-Heq.
   assert(Hcorrect:Forall(entry_prepared F prog)entries).
   { eapply interval_expand_prepared_correct;[apply remember_entries_correct|exact Ex]. }
   rewrite(interval_prepared_picks_same F prog entries Hcorrect). reflexivity.
 - cbn[option_map]in Heq. rewrite <-Heq. reflexivity.
Qed.
Definition literal_old_proposer (F:lfacts)(prog:program):option cached_certificate:=
 match interval_shape prog with None=>None|Some(h,t,d)=>
 let written:=flat_map(instruction_support_outputs F)(interval_loop_body prog h t)in
 let table:=live_program F prog in let incoming:=live_incoming_controls prog in
 match interval_preheader h F prog table incoming 0(empty_state F)with None=>None|Some pre=>
 match invariant_template F written pre d with None=>None|Some candidate=>
 let goals:=interval_constraints F pre d in
 let T:=coherent_transfer F written goals(Some d)in
 match interval_prefix h F T prog table incoming 0(empty_state F)with None=>None|Some(prefix,_)=>
 let head:=interval_make_cache T candidate(block_body prog h)[false;true]in
 match find(fun edge=>Nat.eqb(fst edge)(S h))(cached_edges F prog h head(interval_cache_out head))with
 |Some(_,Some mem_pre)=>
 let memory:=interval_make_cache T mem_pre(block_body prog(S h))[true;true]in
 let tail_memory:=interval_make_cache T(interval_cache_out memory)(block_body prog t)[true;true]in
 let tail_skip:=interval_make_cache T(interval_cache_out head)(block_body prog t)[true;true]in
 let base:=prefix++[(h,head);(S h,memory);(t,tail_memory);(t,tail_skip)]++interval_terminals prog in
 match interval_expand(8*length prog)F T prog(map fst base)(interval_frontier F prog base)base with
 |None=>None|Some entries=>let inv:=interval_assemble prog entries in
 match interval_picks F prog inv with None=>None|Some picks=>
 Some{|cert_written:=written;cert_goals:=goals;cert_descriptor:=Some d;
 cert_blocks:=inv;cert_targets:=picks|}end end
 |_=>None end end end end end.

Lemma interval_preparation_keeps_proposal : forall F prog prepared,
 interval_prepare F prog=Some prepared ->
 propose_prepared_interval F prog prepared=literal_old_proposer F prog.
Proof.
 intros F prog prepared H. rewrite propose_prepared_edges_same. unfold interval_prepare in H.
 destruct(interval_shape prog)as[[[h t]d]|]eqn:HS;[|discriminate].
 destruct(interval_entry_guard prog h t d && interval_forward_remainder prog h t &&
 interval_middle_fallthrough prog h t)eqn:HC;[|discriminate].
 destruct(interval_preheader h F prog(live_program F prog)(live_incoming_controls prog)
 0(empty_state F))as[pre|]eqn:HP;[|discriminate].
 destruct(is_bottom pre)eqn:HB;[discriminate|].
 destruct(invariant_template F
 (flat_map(instruction_support_outputs F)(interval_loop_body prog h t))pre d)as[candidate|]eqn:HT;[|discriminate].
 injection H as <-. unfold literal_prepared_interval,literal_old_proposer.
 rewrite HS,HP,HT. reflexivity.
Qed.
Lemma selected_refines_original_shape : forall F prog,
 interval_candidate_selected F prog=true -> exists h t d,interval_shape prog=Some(h,t,d).
Proof.
 intros F prog H. unfold interval_candidate_selected,interval_prepare in H.
 destruct(interval_shape prog)as[[[h t]d]|]eqn:HS;[eauto|discriminate].
Qed.
Lemma unavailable_selects_ordinary_before_check : forall F prog,
 interval_prepare F prog=None -> propose_scalar_certificate F prog=ScalarBlocks(propose_block_certificate F prog).
Proof. intros F prog H;unfold propose_scalar_certificate;now rewrite H. Qed.
Print Assumptions interval_preparation_keeps_proposal.
Print Assumptions selected_refines_original_shape.
Print Assumptions unavailable_selects_ordinary_before_check.

Print Assumptions propose_prepared_edges_same.
Print Assumptions interval_expand_prepared_erases.
Print Assumptions interval_prepared_picks_same.
"""


def run_source(tmp_path, source):
    query = tmp_path / "Selector.v"
    query.write_text(source)
    return subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=15,
    )


def compile_source(tmp_path, source):
    result = run_source(tmp_path, source)
    assert result.returncode == 0, (result.stdout, result.stderr)


@pytest.mark.parametrize("name", FIXTURES)
def test_interval_selector_keeps_supported_query(tmp_path, name):
    expected = "true" if name == "guarded_entry" else "false"
    source = export_terms(zeroed(FIXTURES[name], BINDING), [BINDING]) + METADATA
    source += (
        "Example chosen_route:interval_candidate_selected F prog="
        + expected
        + ".\nProof. vm_compute;reflexivity. Qed.\n"
    )
    if name in NOT_PROVED_TO_END:
        # The route example precedes the query, so a refusal printed by the
        # query means that the route was checked and the safety proof holds.
        result = run_source(tmp_path, source + CHECK_QUERY)
        assert result.returncode != 0
        assert "OPGATE_REFUSAL" in result.stdout, (result.stdout, result.stderr)
        assert "the kernel is not proved to end" in result.stdout, result.stdout
    else:
        compile_source(tmp_path, source + CHECK_QUERY)


def test_prepared_proposal_matches_literal_old_body(tmp_path):
    compile_source(tmp_path, PRESERVATION)
