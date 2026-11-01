"""Checked forward-branch loop summaries; all controls are CPU-only."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol

BINDING = Binding(
    "kernel",
    (SizeSymbol("n", 0, 2147483647),),
    (("n", Size(Affine.symbol(0))),),
    (256, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def compile_query(tmp_path, query):
    path = tmp_path / "Query.v"
    path.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("two_branches", [False, True])
@pytest.mark.parametrize("before_event", [False, True])
def test_guarded_dag_loop_reaches_ordered_events(tmp_path, two_branches, before_event):
    # The size contract requires an unbounded-by-proof-fuel iteration count.
    # Both forward paths preserve the cursor. Stride is initialized inside
    # every iteration, immediately before the checked positive increment.
    extra = "setp.eq.u32 %p4,%r1,1;\n@%p4 bra JOIN;\nadd.u32 %r3,%r3,1;" if two_branches else ""
    first = "bar.sync 0;" if before_event else ""
    last = "bar.sync 0;" if not before_event else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<5>;
.reg .b32 %r<4>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd2,%r1;
{first}
setp.ge.s64 %p1,%rd2,%rd1;
@%p1 bra DONE;
LOOP:
setp.eq.u32 %p3,%r1,0;
@%p3 bra JOIN;
add.u32 %r3,%r1,1;
{extra}
JOIN:
cvt.u64.u32 %rd3,%r2;
add.s64 %rd2,%rd2,%rd3;
setp.lt.s64 %p2,%rd2,%rd1;
@%p2 bra LOOP;
DONE:
bar.sync 0;
{last}
ret;
}}
"""
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(BINDING))
    compile_query(tmp_path, export_terms(text, [BINDING]) + CHECK_QUERY)


def test_loop_dag_theorems_closed(tmp_path):
    names = [
        "loop_dag_successors_complete",
        "loop_dag_all_sound",
        "loop_dag_check_sound",
        "loop_dag_iteration_sound",
        "loop_dag_validate_progress",
        "loop_summary_straight_preserved",
        "loop_summary_legacy_validator",
        "loop_summary_validate_progress",
        "ranked_summary_all_progress",
        "loop_summaries_check_sound",
        "event_progress_general_sound",
        "event_progress_general_legacy",
        "accepted_front_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


METADATA = "From Coq Require Import List ZArith.\nFrom OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.\nImport ListNotations. Local Open Scope Z_scope.\nDefinition F:lfacts:={|\n lf_widths:=PM.add 3%positive 1%nat(PM.add 7%positive 1%nat(PM.empty nat));\n lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];lf_shared_window:=10%positive;\n lf_local_window:=11%positive;lf_shared:=[];lf_locals:=[];lf_cells:=[];\n lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.\nDefinition bounded (hi:Z):astate:={|\n amap:=PM.add 1%positive (FForm(range_form 64 0 hi)0 None)(PM.empty fact);acons:=[]|}.\nDefinition d:loop_info:={|li_prefix:=0;li_suffix:=0;li_counter:=1%positive;\n li_source:=1%positive;li_stride:=2%positive;li_predicate:=3%positive;li_negated:=false|}.\nDefinition P:program:=[\n [IBra (Some(false,7%positive)) 2%nat];\n [IImm None 8%positive 42;IBra None 2%nat];\n [IImm None 2%positive 1;loop_counter_step d;ISetp None CLt 64 false 3%positive 1%positive(OImm 4);loop_branch d(0%nat,0%nat)];\n [INop;IExit None]].\nDefinition events:list pc:=[(3%nat,0%nat)].\nDefinition before:=transfer F (bounded 4)(IImm None 2%positive 1).\nDefinition after:=transfer F(transfer F before(loop_counter_step d))\n (ISetp None CLt 64 false 3%positive 1%positive(OImm 4)).\nDefinition cert:loop_dag_certificate:={|ldc_arithmetic:=d;ldc_increment:=(2%nat,1%nat);\n ldc_branch:=(2%nat,3%nat);ldc_before:=before;ldc_after:=after|}.\n"


@pytest.mark.parametrize(
    "obligation",
    [
        "loop_dag_check 32 F P [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=true",
        "loop_dag_check 0 F P [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=false",
        "loop_dag_check 32 F P [1%positive] [(0%nat,0%nat)] (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=false",
        "loop_dag_check 32 F P [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)(bounded 0)=false",
        "loop_dag_check 32 F P [1%positive] events (8%nat,0%nat)(2%nat,1%nat)(bounded 4)before=false",
        "loop_dag_forward P (0%nat,0%nat) (IExit None)=false",
        "loop_dag_forward P (0%nat,0%nat) (IBra None 0%nat)=false",
        "loop_dag_forward P (1%nat,0%nat) (IBra (Some(false,7%positive)) 0%nat)=false",
        "loop_dag_check 32 F ([IImm None 1%positive 0]::tl P) [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=false",
        "loop_dag_check 32 F ([IDef None [8%positive;1%positive]]::tl P) [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=false",
        "loop_dag_check 32 F ([IExit (Some(false,7%positive))]::tl P) [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=false",
        "loop_dag_validate 32 F P [(2%nat,1%nat)] (0%nat,0%nat)cert(bounded 4)=None",
        "loop_dag_validate 32 F P [(2%nat,3%nat)] (0%nat,0%nat)cert(bounded 4)=None",
        "loop_increment_ok F P (2%nat,1%nat)(transfer F (bounded 4)(IImm None 2%positive 0))d=false",
        "loop_increment_ok F P (2%nat,1%nat)(transfer F (bounded 4)(IImm None 2%positive (-1)))d=false",
        "loop_increment_ok F P (2%nat,1%nat)(transfer F (bounded 18446744073709551615)(IImm None 2%positive 1))d=false",
        "loop_dag_validate 32 F P events (0%nat,0%nat)cert(bounded 0)=None",
        "loop_dag_collect 0 F P (2%nat,1%nat)[((0%nat,0%nat),Some(bounded 4))]=None",
        "loop_dag_collect 32 F P (2%nat,1%nat)[((8%nat,0%nat),Some(bounded 4))]=None",
        "loop_summary_at P (0%nat,0%nat)=option_map Loop_dag_descriptor(loop_dag_at P (0%nat,0%nat))",
    ],
)
def test_dag_checker_metadata(tmp_path, obligation):
    # These are checker data only. No malformed program executes on a GPU.
    compile_query(
        tmp_path,
        METADATA + f"Example checked : {obligation}.\nProof. vm_compute. reflexivity. Qed.\n",
    )


def test_explicit_descriptor_and_entry_boundaries(tmp_path):
    # The old straight descriptor keeps priority. The new outer checker must
    # independently reject an entry outside its proposed invariant.
    straight = "[[IImm None 2%positive 1;loop_counter_step d;ISetp None CLt 64 false 3%positive 1%positive(OImm 4);loop_branch d(0%nat,0%nat)];[INop;IExit None]]"
    with_cells = (
        METADATA.replace("lf_cells:=[];", "lf_cells:=[(9%positive,10%positive,0)];")
        .replace("Definition F:lfacts", "Definition Cells:lfacts")
        .split("Definition bounded")[0]
    )
    compile_query(
        tmp_path,
        METADATA
        + with_cells
        + f"""
Definition straight:program:={straight}.
Example straight_descriptor_priority:
 match loop_at straight(0%nat,0%nat),loop_summary_at straight(0%nat,0%nat) with
 | Some a,Some(Loop_straight_descriptor b)=>true | _,_=>false end=true.
Proof. vm_compute. reflexivity. Qed.
Example unrelated_local_cells_preserved:
 loop_dag_check 32 Cells P [1%positive] events (0%nat,0%nat)(2%nat,1%nat)(bounded 4)before=true.
Proof. vm_compute. reflexivity. Qed.
Example outer_entry_containment:
 ranked_summary_all events(PM.empty unit)32 F P(phase_event(3%nat,0%nat))
 [((0%nat,0%nat),Some(bounded 5))]=false.
Proof. vm_compute. reflexivity. Qed.
""",
    )
