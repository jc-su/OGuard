"""Checked compound backedges preserve prior schedules and arithmetic checks."""

import subprocess

import pytest

from opgate.binding import Binding
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine


def compile_query(tmp_path, query):
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("start", [-2, 0])
def test_prior_value_compound_loop(tmp_path, width, start):
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p<6>;
.reg .b32 %r<3>;
.reg .b{width} %counter, %old, %stride;
mov.u32 %r1,%tid.x;
mov.s{width} %counter,{start};
mov.pred %p1,0;
LOOP:
mov.b{width} %old,%counter;
setp.lt.s{width} %p2,%old,4;
not.pred %p3,%p1;
and.pred %p4,%p2,%p3;
mov.s{width} %stride,1;
add.s{width} %counter,%old,%stride;
setp.lt.s{width} %p1,%counter,%old;
@%p4 bra LOOP;
bar.sync 0;
setp.ge.u32 %p5,%r1,32;
@%p5 bra DONE;
shfl.sync.idx.b32 %r2,%r1,0,31,-1;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (64, 1, 1), (1, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


META = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.
Import ListNotations. Local Open Scope Z_scope.
Definition body (w:nat):list instr:=[
 IMov None 2%positive 1%positive;
 ISetp None CLt w true 6%positive 2%positive(OImm 4);
 IAndp None 3%positive 6%positive 7%positive;
 IImm None 5%positive 1;
 IBin None BAdd w 1%positive 2%positive(OReg 5%positive);
 IBra(Some(false,3%positive))0%nat].
Definition P w:program:=[body w;[INop;IExit None]].
Definition found prog events:=match compound_event_free prog events(0%nat,0%nat)with Some _=>true|None=>false end.
Definition F w:lfacts:={|
 lf_widths:=PM.add 1%positive w(PM.add 2%positive w(PM.add 3%positive 1%nat
  (PM.add 5%positive w(PM.add 6%positive 1%nat(PM.add 7%positive 1%nat(PM.empty nat))))));
 lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition entry w lo hi:astate:={|amap:=PM.add 1%positive(FForm(range_form(Z.of_nat w)lo hi)0 None)(PM.empty fact);acons:=[]|}.
Definition summary_ok w lo hi:=match compound_occurrence_summary(F w)(P w)[(1%nat,0%nat)](0%nat,0%nat)(entry w lo hi)with
 Checked_loop_summary _ _=>true|_=>false end.
Definition witness_step(w a:nat):signed_loop_step:={|sl_width:=w;sl_alias_width:=a;sl_counter:=1%positive;sl_source:=1%positive;sl_operand:=OImm 1;sl_down:=false|}.
Definition witness_ok w a:=signed_loop_step_ok(F w)[[signed_loop_instr(witness_step w a)]](0%nat,0%nat)(entry w 0 0)(witness_step w a).
Definition direct:program:=[
 [IBin None BAdd 64 1%positive 1%positive(OImm 1);
  ISetp None CLt 64 true 3%positive 1%positive(OImm 4);
  IBra(Some(false,3%positive))0%nat];[INop;IExit None]].
Definition forward:program:=[
 [IBin None BAdd 64 1%positive 1%positive(OImm 1);
  ISetp None CLt 64 true 3%positive 1%positive(OImm 4);
  IBra(Some(false,3%positive))1%nat];[INop;IExit None]].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "witness_ok 32 16=false",
        "witness_ok 32 65=false",
        "witness_ok 32 64=true",
        "witness_ok 16 16=true",
        "witness_ok 32 32=true",
        "witness_ok 64 64=true",
        "found(P 16)[(1%nat,0%nat)]=true",
        "found(P 32)[(1%nat,0%nat)]=true",
        "found(P 64)[(1%nat,0%nat)]=true",
        "found([tl(body 64);[INop]])[]=false",
        "found([[IMov None 2%positive 1%positive; ISetp None CLt 32 true 6%positive 2%positive(OImm 4)]++skipn 2(body 64);[INop]])[]=false",
        "found([firstn 4(body 64)++[IBin None BAdd 64 8%positive 2%positive(OImm 1)]++skipn 4(body 64);[INop]])[]=false",
        "found([firstn 2(body 64)++[IOrp None 3%positive 6%positive 7%positive]++skipn 3(body 64);[INop]])[]=false",
        "found([firstn 2(body 64)++[IImm None 6%positive 0]++skipn 2(body 64);[INop]])[]=false",
        "found(P 64)[(0%nat,0%nat)]=false",
        "found(P 64)[(0%nat,4%nat)]=false",
        "found(P 64)[(0%nat,5%nat)]=false",
        "summary_ok 16 0 0=true",
        "summary_ok 32 (-2)(-2)=true",
        "summary_ok 64 (-2)(-2)=true",
        "summary_ok 16 (2^15-1)(2^15-1)=false",
        "summary_ok 32 (2^31-1)(2^31-1)=false",
        "summary_ok 64 (2^63-1)(2^63-1)=false",
        "match compound_occurrence_summary(F 64)([firstn 4(body 64)++[IImm None 2%positive 99]++skipn 4(body 64);[INop]])[(1%nat,0%nat)](0%nat,0%nat)(entry 64 0 0)with Checked_loop_summary _ _=>false|_=>true end=true",
        "match signed_loop_at forward(1%nat,0%nat)with Some _=>true|None=>false end=true",
        "compound_descriptor_at forward[](1%nat,0%nat)=None",
        "match compound_descriptor_at direct[(1%nat,0%nat)](0%nat,0%nat)with Some(Compound_signed _)=>true|_=>false end=true",
    ],
)
def test_compound_metadata_obligations(tmp_path, obligation):
    compile_query(
        tmp_path,
        META + f"Example checked : {obligation}.\nProof. vm_compute. reflexivity. Qed.\n",
    )


def test_compound_proofs_closed(tmp_path):
    names = [
        "signed_loop_source_sound",
        "signed_loop_step_sound",
        "signed_refine_all_sound",
        "signed_loop_refine_mode_sound",
        "signed_loop_refresh_sound",
        "signed_summary_check_sound",
        "compound_occurrence_summary_sound",
        "scoped_summary_select_sound",
        "scoped_complete_word_preserves",
        "scoped_signed_complete_word_preserves",
        "scoped_complete_word_sound",
        "accepted_front_linked_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


MATRIX = r"""
Definition matrix_program(w:nat)(sg mix:bool)(c:cmp):program:=[
 [IMov None 2%positive 1%positive;
 ISetp None c w sg 6%positive 2%positive(OImm 4)]++
 (if mix then[ISetp None CGe w (negb sg)7%positive 2%positive(OImm 0)]else[])++
 [IAndp None 3%positive 6%positive 7%positive;IImm None 5%positive 1;
 IBin None BAdd w 1%positive 2%positive(OReg 5%positive);
 IBra(Some(false,3%positive))0%nat];[INop;IExit None]].
Definition matrix_ok w sg mix c start:=match compound_occurrence_summary(F w)(matrix_program w sg mix c)[(1%nat,0%nat)](0%nat,0%nat)(entry w start start)with Checked_loop_summary _ _=>true|_=>false end.
"""


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("start", [-2, 0])
@pytest.mark.parametrize("signed", [True, False])
@pytest.mark.parametrize("mixed", [True, False])
@pytest.mark.parametrize("comparison", ["CLt", "CLe"])
def test_compound_width_polarity_matrix(tmp_path, width, start, signed, mixed, comparison):
    obligation = (
        f"matrix_ok {width} {str(signed).lower()} {str(mixed).lower()} {comparison} ({start})=true"
    )
    compile_query(
        tmp_path,
        META + MATRIX + f"Example checked:{obligation}.\nProof. vm_compute. reflexivity. Qed.\n",
    )


def test_false_conjunction_and_overwrite_controls(tmp_path):
    compile_query(tmp_path, META + NEGATIVE_CONTROLS)


NEGATIVE_CONTROLS = r"""
Definition mixedcs:= [{|cc:=CLt;cw:=32;csg:=true;ca:=1%positive;cb:=OImm 4|};{|cc:=CGe;cw:=32;csg:=false;ca:=2%positive;cb:=OImm 1|}].
Definition mixedstate:=put(put(entry 32 (-2)5)2%positive(FForm(range_form 32 0 2)0 None))3%positive(FPred mixedcs true).
Definition mixed_range truth:=match signed_loop_refine_mode true(F 32)mixedstate 1%positive 3%positive truth with Some out=>cmp_range(F 32)out(OReg 1%positive)32 true|None=>None end.
Example true_conjunction: mixed_range true=Some(-2,3). Proof. vm_compute. reflexivity. Qed.
Example false_conjunction: mixed_range false=Some(-2,5). Proof. vm_compute. reflexivity. Qed.
Example false_clauses_unchanged: forall F st p,
 signed_loop_refine_clauses true F st p false=signed_loop_refine F st p false.
Proof. reflexivity. Qed.
Example negated_source_not_selected:
 found([firstn 5(body 32)++[IBra(Some(true,3%positive))0%nat];[INop]])[]=false.
Proof. vm_compute. reflexivity. Qed.
Definition step:signed_loop_step:={|sl_width:=32;sl_alias_width:=32;sl_counter:=1%positive;sl_source:=2%positive;sl_operand:=OReg 5%positive;sl_down:=false|}.
Definition aliasstep:signed_loop_step:={|sl_width:=32;sl_alias_width:=32;sl_counter:=1%positive;sl_source:=1%positive;sl_operand:=OImm 1;sl_down:=false|}.
Definition overwrittenstep:signed_loop_step:={|sl_width:=32;sl_alias_width:=32;sl_counter:=1%positive;sl_source:=2%positive;sl_operand:=OReg 1%positive;sl_down:=false|}.
Example no_old_source: signed_loop_step_relation(entry 32 0 4)aliasstep=entry 32 0 4.
Proof. vm_compute. reflexivity. Qed.
Example no_old_stride: signed_loop_step_relation(entry 32 0 4)overwrittenstep=entry 32 0 4.
Proof. vm_compute. reflexivity. Qed.
Example overwritten_source_discards_relation:
 acons(transfer(F 32)(signed_loop_step_relation(entry 32 0 4)step)(IImm None 2%positive 99))=[].
Proof. vm_compute. reflexivity. Qed.
Example overwritten_stride_discards_relation:
 acons(transfer(F 32)(signed_loop_step_relation(entry 32 0 4)step)(IImm None 5%positive 99))=[].
Proof. vm_compute. reflexivity. Qed.
"""
