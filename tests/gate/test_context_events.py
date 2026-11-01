"""Compact CTA context schedules: CPU-only metadata and ordinary proof controls."""

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


META = r"""
From Coq Require Import List ZArith String.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop GateContext GateFront.
Import ListNotations. Local Open Scope Z_scope. Local Open Scope string_scope.
Definition facts (sl sh lh:Z):lfacts:={|
 lf_widths:=PM.add 4%positive 1%nat(PM.empty nat);
 lf_fixed:=[5%positive;6%positive;7%positive;9%positive;100%positive;101%positive];
 lf_bounds:=[(5%positive,(0,3));(6%positive,(sl,sh));(7%positive,(0,lh))];
 lf_aligns:=[];lf_roles:=[];lf_shared_window:=100%positive;lf_local_window:=101%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];
 lf_names:=[("%ctaid.x",5%positive);("%nctaid.x",6%positive);("size:0",7%positive);
  ("%tid.x",9%positive);("window:shared",100%positive)]|}.
Definition F:=facts 1 2 4.
Definition M:binding:={|b_sizes:=[(0,4)];b_derived:=[];b_params:=[];
 b_block:=(64,1,1);b_grid:=(4,1,1);b_shared:={|af_const:=0;af_terms:=[]|}|}.
Definition P:program:=[
 [IMov None 1%positive 5%positive;IMov None 3%positive 7%positive;
  ISetp None CGe 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))4%nat];
 [IMov None 2%positive 6%positive;IBra None 3%nat];
 [INop;IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);
  ISetp None CLt 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(true,4%positive))4%nat];
 [INop;IBra None 2%nat];[IExit None]].
Definition events:list pc:=[(2%nat,0%nat);(3%nat,0%nat)].
Definition checked F uniform:=match context_shape_select P events with
 |Some shape=>context_check F uniform P events shape|None=>false end.
Definition cf:form:={|fm:=64;ft:=[((5%positive,EX),1)];flo:=0;fhi:=0|}.
Definition tf:form:={|fm:=64;ft:=[((9%positive,EX),1)];flo:=0;fhi:=0|}.
Definition guarded:program:=[[IBra(Some(false,4%positive))2%nat];[IBra None 2%nat];[INop]].
Definition twin:program:=[
 [IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);
  ISetp None CLt 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(true,4%positive))3%nat];
 [IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);
  ISetp None CLt 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(true,4%positive))3%nat];
 [INop];[IExit None]].
Definition fixed_program (limit:Z):program:=[
 [IImm None 1%positive 0;IImm None 3%positive limit;
 ISetp None CGe 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))4%nat];
 [IImm None 2%positive 1;IBra None 3%nat];
 [INop;IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);
 ISetp None CLt 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(true,4%positive))4%nat];
 [INop;IBra None 2%nat];[IExit None]].
Definition fixed_checked n:=match context_shape_select(fixed_program n)events with
 |Some shape=>context_check F(cta_uniform_key F M)(fixed_program n)events shape|None=>false end.

"""


@pytest.mark.parametrize(
    "obligation",
    [
        "checked F(cta_uniform_key F M)=true",
        "fixed_checked 0=true",
        r"event_progress_general F(fixed_program 1)events=true /\ fixed_checked 1=true",
        r"event_progress_general F(fixed_program 2)events=true /\ fixed_checked 2=true",
        r"event_progress_general F(fixed_program 4)events=true /\ fixed_checked 4=true",
        "checked(facts 1 1 0)(cta_uniform_key F M)=true",
        "checked(facts 0 2 4)(cta_uniform_key F M)=false",
        "checked(facts 1 2 (2^63-1))(cta_uniform_key F M)=false",
        "checked F(fun _=>false)=false",
        "lf_ok F=true",
        "is_fixed F 9%positive=true",
        "cta_uniform_key F M 5%positive=true",
        "cta_uniform_key F M 6%positive=true",
        "cta_uniform_key F M 7%positive=true",
        "cta_uniform_key F M 9%positive=false",
        "cta_uniform_key F M 100%positive=false",
        "cta_uniform_key F M 55%positive=false",
        "context_form_ok F(cta_uniform_key F M)cf=true",
        "context_form_ok F(cta_uniform_key F M)tf=false",
        "context_shape_select P[]=None",
        "context_shape_select twin[(2%nat,0%nat)]=None",
        "context_shape_at P events(2%nat,1%nat)(IBin None BAdd 32 1%positive 1%positive(OReg 2%positive))=None",
        "context_shape_at P events(2%nat,1%nat)(IBin None BAdd 64 1%positive 3%positive(OReg 2%positive))=None",
        "context_not_exit(IExit None)=false",
        "context_not_exit(IExit(Some(false,4%positive)))=false",
        "context_region F[[INop;INop]][](0%nat,1%nat)[][(0%nat,0%nat)]=true",
        "context_region F[[IExit None;INop]][](0%nat,1%nat)[][(0%nat,0%nat)]=false",
        "context_region F[[IExit(Some(false,4%positive));INop]][](0%nat,1%nat)[][(0%nat,0%nat)]=false",
        "context_region F[[INop;INop]][(0%nat,0%nat)](0%nat,1%nat)[][(0%nat,0%nat)]=false",
        "context_region F[[IImm None 1%positive 0;INop]][](0%nat,1%nat)[1%positive][(0%nat,0%nat)]=false",
        "context_region F guarded[](2%nat,0%nat)[][(0%nat,0%nat);(1%nat,0%nat)]=true",
        "context_region F guarded[](2%nat,0%nat)[][(0%nat,0%nat)]=false",
        "context_region F guarded[](2%nat,0%nat)[][(1%nat,0%nat);(0%nat,0%nat)]=false",
        "context_region F[[IBra None 0%nat]][](1%nat,0%nat)[][(0%nat,0%nat)]=false",
        "context_region_propose[[IExit None]][](0%nat,0%nat)=None",
        "context_segment F P events(2%nat,1%nat)[1%positive;2%positive;3%positive](2%nat,1%nat)[]=true",
        "context_test false F P events[1%positive;2%positive;3%positive](0%nat,2%nat)(0%nat,3%nat)[]1%positive 3%positive 4%positive CGe false 4=true",
        "context_test false F P events[1%positive;2%positive;3%positive](0%nat,2%nat)(0%nat,3%nat)[]1%positive 3%positive 4%positive CGe true 4=false",
        "context_test false F P events[1%positive;2%positive;3%positive](0%nat,2%nat)(0%nat,3%nat)[]1%positive 3%positive 4%positive CLt false 4=false",
        "context_test false F P events[1%positive;2%positive;3%positive](0%nat,2%nat)(0%nat,3%nat)[]1%positive 3%positive 4%positive CGe false 3=false",
        "context_test false F P[(0%nat,2%nat)][1%positive;2%positive;3%positive](0%nat,2%nat)(0%nat,3%nat)[]1%positive 3%positive 4%positive CGe false 4=false",
    ],
)
def test_context_metadata(tmp_path, obligation):
    compile_query(
        tmp_path,
        META
        + f"Example checked_obligation:{obligation}. Proof. vm_compute. repeat split; reflexivity. Qed.\n",
    )


def test_context_public_theorems_closed(tmp_path):
    names = [
        "context_region_sound",
        "context_events_sound",
        "context_prefix_sound",
        "context_test_sound",
        "context_increment_sound",
        "controller_pair_sound",
        "context_entry_pair_sound",
        "context_check_sound",
        "front_fixed_route_preserved",
        "front_event_check_sound",
        "linked_context_schedule_sound",
        "accepted_front_linked_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateContext OpGate.GateFront OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_context_constant_controller(tmp_path, count):
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p;
.reg .b64 %counter, %limit, %step;
mov.u64 %counter, 0;
mov.u64 %limit, {count};
setp.ge.s64 %p, %counter, %limit;
@%p bra DONE;
mov.u64 %step, 1;
bra BODY;
TAIL:
bar.sync 0;
add.s64 %counter, %counter, %step;
setp.lt.s64 %p, %counter, %limit;
@!%p bra DONE;
BODY:
bar.sync 0;
bra TAIL;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (32, 1, 1), (1, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


UNIFORM = (
    META
    + r"""
Definition U:=cta_uniform_key F M.
Definition marks:=uniform_initial_marks F U.
Definition pure:instr:=IBin None BShr 64 1%positive 5%positive(OImm 1).
Definition overwritten:=IMov None 1%positive 9%positive.
Definition reference:reg->Z:=fun _=>0.
"""
)


@pytest.mark.parametrize(
    "obligation",
    [
        "uniform_pure pure=true",
        "uniform_pure(IMov None 1%positive 5%positive)=true",
        "uniform_pure(IImm None 1%positive 3)=true",
        "uniform_pure(IMad None 64 1%positive 5%positive(OImm 2)(OImm 1))=true",
        "uniform_pure(IWide None false 1%positive 5%positive(OImm 2)None)=true",
        "uniform_pure(ICvt None 64 32 false 1%positive 5%positive)=true",
        "uniform_pure(IMov(Some(false,4%positive))1%positive 5%positive)=false",
        "uniform_pure(ILdS None 32 1%positive 5%positive 0)=false",
        "uniform_pure(IDef None[1%positive])=false",
        "uniform_pure(IDiv None false false 64 1%positive 5%positive(OImm 1))=false",
        "uniform_marks_step F pure marks 1%positive=true",
        "uniform_marks_step F overwritten marks 1%positive=false",
        r"is_fixed F 9%positive=true /\ marks 9%positive=false",
        "uniform_marks_run F[pure;IDef None[1%positive]]marks 1%positive=false",
        "uniform_marks_run F[pure;ILdS None 32 1%positive 5%positive 0]marks 1%positive=false",
        "uniform_marks_run F[pure;IMov None 2%positive 1%positive]marks 2%positive=true",
        "uniform_marks_run F[pure;IMov None 2%positive 9%positive]marks 2%positive=false",
        "uniform_marks_step F(IImm None 1%positive 3)(fun _=>false)1%positive=true",
        "uniform_value_step F(IImm None 1%positive(2^64+3))reference 1%positive=3",
        "uniform_value_step F(IImm None 4%positive 3)reference 4%positive=1",
        "uniform_prefix_collect 1 F[[IExit None]][](1%nat,0%nat)(0%nat,0%nat)=None",
        "uniform_prefix_collect 1 F[[IBra(Some(false,4%positive))1%nat]][](1%nat,0%nat)(0%nat,0%nat)=None",
        "uniform_prefix_collect 1 F[[INop;INop]][(0%nat,0%nat)](0%nat,1%nat)(0%nat,0%nat)=None",
        "uniform_prefix_collect 0 F[[INop;INop]][](0%nat,1%nat)(0%nat,0%nat)=None",
        "uniform_prefix_check 1 F[[pure;INop]][](0%nat,1%nat)(0%nat,0%nat)[pure]=true",
        "uniform_prefix_check 1 F[[pure;INop]][](0%nat,1%nat)(0%nat,0%nat)[overwritten]=false",
        "context_test true F P events[1%positive;2%positive;3%positive](0%nat,2%nat)(0%nat,3%nat)[]1%positive 3%positive 4%positive CGe false 4=false",
    ],
)
def test_uniform_metadata(tmp_path, obligation):
    compile_query(
        tmp_path,
        UNIFORM
        + f"Example uniform_obligation:{obligation}. Proof. vm_compute. repeat split; reflexivity. Qed.\n",
    )


def test_uniform_closed_assumptions(tmp_path):
    names = [
        "uniform_result_ext",
        "uniform_step_agree",
        "uniform_prefix_full_sound",
        "uniform_entry_sound",
        "uniform_entry_agree",
        "context_exact_entry_preserved",
        "context_check_sound",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateContext.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


@pytest.mark.parametrize("reversed_initial", [False, True])
@pytest.mark.parametrize("increment_gap", [False, True])
def test_uniform_launch_expression(tmp_path, reversed_initial, increment_gap):
    comparison = (
        "setp.le.s64 %p, %limit, %counter;"
        if reversed_initial
        else "setp.ge.s64 %p, %counter, %limit;"
    )
    gap = "add.u64 %unused, %unused, 1;" if increment_gap else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p;
.reg .b32 %grid;
.reg .b64 %counter, %limit, %step, %unused;
mov.u64 %counter, 0;
mov.u32 %grid, %nctaid.x;
cvt.u64.u32 %limit, %grid;
add.u64 %limit, %limit, 3;
shr.u64 %limit, %limit, 2;
{comparison}
@%p bra DONE;
mov.u64 %step, 1;
mov.u64 %unused, 0;
bra BODY;
TAIL:
bar.sync 0;
add.s64 %counter, %counter, %step;
{gap}
setp.lt.s64 %p, %counter, %limit;
@!%p bra DONE;
BODY:
bar.sync 0;
bra TAIL;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (32, 1, 1), (8, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


RETENTION = (
    META
    + r"""
Local Open Scope list_scope.
Definition cross_block:program:=[
 [IMov None 1%positive 5%positive;IMov None 3%positive 7%positive;
  ISetp None CGe 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))5%nat];
 [IMov None 2%positive 6%positive;IBra None 4%nat];
 [INop;IBin None BAdd 64 1%positive 1%positive(OReg 2%positive)];
 [ISetp None CLt 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(true,4%positive))5%nat];
 [INop;IBra None 2%nat];[IExit None]].
Definition cross_events:list pc:=[(2%nat,0%nat);(4%nat,0%nat)].
Definition ordered_priority:program:=[
 [IMov None 1%positive 5%positive;IMov None 3%positive 7%positive;
  ISetp None CLe 64 true 4%positive 3%positive(OReg 1%positive);
  ISetp None CGe 64 true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))4%nat]
 ]%list ++ tl P.
Definition checked_program prog ev:=match context_shape_select prog ev with
 |Some d=>context_check F(cta_uniform_key F M)prog ev d|None=>false end.
"""
)


@pytest.mark.parametrize(
    "obligation",
    [
        "checked_program cross_block cross_events=true",
        "match context_shape_select cross_block cross_events with Some d=>cs_test d=(3%nat,0%nat) /\\ cs_increment_gap d=[]|None=>False end",
        "checked_program ordered_priority events=true",
        "match context_shape_select ordered_priority events with Some d=>match context_initial_select ordered_priority d with Some e=>ces_test e=(0%nat,3%nat) /\\ ces_reversed e=false|None=>False end|None=>False end",
    ],
)
def test_context_discovery_retention(tmp_path, obligation):
    compile_query(
        tmp_path,
        RETENTION
        + f"Example retention:{obligation}. Proof. vm_compute. repeat split; reflexivity. Qed.\n",
    )


@pytest.mark.parametrize("cross_block", [False, True])
@pytest.mark.parametrize("extra_reversed", [False, True])
def test_context_old_discovery_whole(tmp_path, cross_block, extra_reversed):
    extra = "setp.le.s64 %p, %limit, %counter;" if extra_reversed else ""
    comparison_label = "COMPARE:" if cross_block else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p;
.reg .b64 %counter, %limit, %step;
mov.u64 %counter, 0;
mov.u64 %limit, 4;
{extra}
setp.ge.s64 %p, %counter, %limit;
@%p bra DONE;
mov.u64 %step, 1;
bra BODY;
TAIL:
bar.sync 0;
add.s64 %counter, %counter, %step;
{comparison_label}
setp.lt.s64 %p, %counter, %limit;
@!%p bra DONE;
BODY:
bar.sync 0;
bra TAIL;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (32, 1, 1), (1, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)
