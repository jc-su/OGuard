"""Signed, event-free counted loops preserve the complete checked schedule."""

import subprocess

import pytest

from opgate.binding import Binding
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine


def compile_query(tmp_path, query):
    p = tmp_path / "Query.v"
    p.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(p)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "width,start,step,limit",
    [
        (32, 0, 1, 100000),
        (32, 99999, -1, -1),
        (64, -7, 1, 1),
        (64, 7, -1, -1),
        (32, -8, 8, 24),
        (64, 0, 4, 16),
    ],
)
@pytest.mark.parametrize("branched", [False, True])
def test_signed_event_free_loop(tmp_path, width, start, step, limit, branched):
    body = "setp.lt.u32 %p2,%r1,32; @%p2 bra JOIN; add.u32 %r4,%r1,1; JOIN:" if branched else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p<4>;
.reg .b32 %r<5>;
.reg .b{width} %counter;
mov.u32 %r1,%tid.x;
mov.s{width} %counter,{start};
LOOP:
{body}
add.s{width} %counter,%counter,{step};
setp.ne.s{width} %p3,%counter,{limit};
@%p3 bra LOOP;
bar.sync 0;
setp.ge.u32 %p1,%r1,32;
@%p1 bra DONE;
shfl.sync.idx.b32 %r2,%r1,0,31,-1;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (64, 1, 1), (1, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


def test_signed_loop_theorems_closed(tmp_path):
    names = [
        "signed_loop_source_sound",
        "signed_loop_step_sound",
        "signed_loop_refine_sound",
        "signed_loop_branch_sound",
        "signed_loop_iteration_sound",
        "signed_loop_validate_progress",
        "signed_occurrence_summary_sound",
        "occurrence_loop_summary_sound",
        "occurrence_signed_loop_summary_sound",
        "scoped_summary_select_sound",
        "scoped_complete_word_preserves",
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


METADATA = r"""
From Coq Require Import ZArith List Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.
Import ListNotations. Local Open Scope Z_scope.
Definition F (w:nat):lfacts:={|
 lf_widths:=PM.add 1%positive w(PM.add 3%positive 1%nat(PM.add 7%positive 1%nat(PM.empty nat)));
 lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;
 lf_derived:=[];lf_names:=[]|}.
Definition bounds (w:nat)(lo hi:Z):astate:={|
 amap:=PM.add 1%positive(FForm(range_form(Z.of_nat w)lo hi)0 None)(PM.empty fact);acons:=[]|}.
Definition descriptor(w:nat)(stride:Z)(down:bool):signed_loop_step:={|
 sl_width:=w;sl_alias_width:=64;sl_counter:=1%positive;sl_source:=1%positive;sl_operand:=OImm stride;sl_down:=down|}.
Definition step_check w lo hi stride down:=
 let d:=descriptor w stride down in
 signed_loop_step_ok(F w)[[signed_loop_instr d]](0%nat,0%nat)(bounds w lo hi)d.
Definition d:=descriptor 32 (-1)true.
Definition P:program:=[
 [IBra(Some(false,7%positive))2%nat];
 [IImm None 8%positive 42;IBra None 2%nat];
 [signed_loop_instr d;ISetp None CNe 32 true 3%positive 1%positive(OImm(-1));
 IBra(Some(false,3%positive))0%nat];
 [INop;IExit None]].
Definition inv:=bounds 32 0 99999.
Definition cert:signed_loop_certificate:={|
 slc_step:=d;slc_conjunction:=false;slc_predicate:=3%positive;slc_negated:=false;
 slc_increment:=(2%nat,0%nat);slc_branch:=(2%nat,2%nat);slc_before:=inv;
 slc_after:=transfer(F 32)(transfer(F 32)inv(signed_loop_instr d))
  (ISetp None CNe 32 true 3%positive 1%positive(OImm(-1)))|}.
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "step_check 32 0 0 (-1)true=true",
        "step_check 32 (-1)(-1)1 false=true",
        "step_check 64 0 0 (-1)true=true",
        "step_check 64 (-1)(-1)1 false=true",
        "step_check 32 (2^31-1)(2^31-1)1 false=false",
        "step_check 32 (-2^31)(-2^31)(-1)true=false",
        "step_check 64 (2^63-1)(2^63-1)1 false=false",
        "step_check 64 (-2^63)(-2^63)(-1)true=false",
        "step_check 32 0 10 0 false=false",
        "step_check 32 0 10 1 true=false",
        "step_check 32 0 10 (-1)false=false",
        "match signed_loop_validate 32(F 32)P[(3%nat,0%nat)](0%nat,0%nat)cert inv with Some _=>true|None=>false end=true",
        "signed_loop_validate 32(F 32)P[(2%nat,0%nat)](0%nat,0%nat)cert inv=None",
        "signed_loop_validate 32(F 32)P[(2%nat,2%nat)](0%nat,0%nat)cert inv=None",
        "signed_loop_validate 32(F 32)P[(0%nat,0%nat)](0%nat,0%nat)cert inv=None",
        "signed_loop_validate 0(F 32)P[(3%nat,0%nat)](0%nat,0%nat)cert inv=None",
        "signed_loop_validate 32(F 32)([IImm None 1%positive 3]::tl P)[] (0%nat,0%nat)cert inv=None",
        "signed_loop_validate 32(F 32)([IExit None]::tl P)[] (0%nat,0%nat)cert inv=None",
        "signed_loop_validate 32(F 32)([IBra None 0%nat]::tl P)[] (0%nat,0%nat)cert inv=None",
    ],
)
def test_signed_loop_static_obligations(tmp_path, obligation):
    # These are checker records only; no malformed kernel is executed.
    compile_query(
        tmp_path,
        METADATA + f"Example checked : {obligation}.\nProof. vm_compute. reflexivity. Qed.\n",
    )


def test_alias_cells_and_endpoint_obligations(tmp_path):
    aliases = (
        METADATA.split("Definition d:=")[0]
        .replace("Definition F (w:nat)", "Definition AliasFacts (w:nat)")
        .replace("PM.add 1%positive w(", "PM.add 2%positive w(PM.add 1%positive w(")
        .replace("1%nat(PM.empty nat)));", "1%nat(PM.empty nat))));")
        .split("Definition bounds")[0]
    )
    cells = (
        METADATA.split("Definition bounds")[0]
        .replace("Definition F (w:nat)", "Definition CellFacts (cell:reg)(w:nat)")
        .replace("lf_cells:=[];", "lf_cells:=[(cell,11%positive,0)];")
    )
    compile_query(
        tmp_path,
        METADATA
        + aliases
        + cells
        + r"""
Definition alias_step:signed_loop_step:={|sl_width:=32;sl_alias_width:=64;sl_counter:=1%positive;
 sl_source:=2%positive;sl_operand:=OImm 1;sl_down:=false|}.
Definition alias_state f:=put(bounds 32 0 10)2%positive f.
Example exact_alias:
 signed_loop_step_ok(AliasFacts 32)[[signed_loop_instr alias_step]](0%nat,0%nat)
 (alias_state(signed_loop_alias alias_step))alias_step=true.
Proof. vm_compute. reflexivity. Qed.
Example unrelated_alias:
 signed_loop_step_ok(AliasFacts 32)[[signed_loop_instr alias_step]](0%nat,0%nat)
 (alias_state(FForm(range_form 32 0 10)0 None))alias_step=false.
Proof. vm_compute. reflexivity. Qed.
Example protected_counter_cell:
 signed_loop_validate 32(CellFacts 1%positive 32)P[(3%nat,0%nat)](0%nat,0%nat)cert inv=None.
Proof. vm_compute. reflexivity. Qed.
Example unrelated_cell:
 match signed_loop_validate 32(CellFacts 9%positive 32)P[(3%nat,0%nat)](0%nat,0%nat)cert inv with
 |Some _=>true|None=>false end=true.
Proof. vm_compute. reflexivity. Qed.
Example backedge_not_contained:
 signed_loop_validate 32(F 32)P[(3%nat,0%nat)](0%nat,0%nat)cert(bounds 32 0 0)=None.
Proof. vm_compute. reflexivity. Qed.
Definition bad_endpoint: signed_loop_certificate:={|slc_step:=d;slc_conjunction:=false;slc_predicate:=3%positive;
 slc_negated:=false;slc_increment:=(2%nat,0%nat);slc_branch:=(2%nat,2%nat);
 slc_before:=inv;slc_after:=bounds 32 5 5|}.
Example endpoint_not_contained:
 signed_loop_validate 32(F 32)P[(3%nat,0%nat)](0%nat,0%nat)bad_endpoint inv=None.
Proof. vm_compute. reflexivity. Qed.
Example event_free_selected:
 match signed_loop_at_event_free P[(3%nat,0%nat)](0%nat,0%nat)with Some _=>true|None=>false end=true.
Proof. vm_compute. reflexivity. Qed.
Example prefix_event_not_selected:
 signed_loop_at_event_free P[(0%nat,0%nat)](0%nat,0%nat)=None.
Proof. vm_compute. reflexivity. Qed.
Example increment_event_not_selected:
 signed_loop_at_event_free P[(2%nat,0%nat)](0%nat,0%nat)=None.
Proof. vm_compute. reflexivity. Qed.
Example branch_event_not_selected:
 signed_loop_at_event_free P[(2%nat,2%nat)](0%nat,0%nat)=None.
Proof. vm_compute. reflexivity. Qed.
""",
    )


@pytest.mark.parametrize("start,step,limit", [(0, 1, 3), (2147483647, 1, -2147483648)])
def test_small_event_bearing_loops_keep_original_route(tmp_path, start, step, limit):
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p<3>;
.reg .b32 %r<4>;
mov.u32 %r1,%tid.x;
mov.s32 %r2,{start};
LOOP:
bar.sync 0;
add.s32 %r2,%r2,{step};
setp.ne.s32 %p1,%r2,{limit};
@%p1 bra LOOP;
setp.ge.u32 %p2,%r1,32;
@%p2 bra DONE;
shfl.sync.idx.b32 %r3,%r1,0,31,-1;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (64, 1, 1), (1, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)
