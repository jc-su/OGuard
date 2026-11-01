"""Event-free loop summaries within complete occurrence words; static only."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol


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


@pytest.mark.parametrize("block", [2, 16, 256])
@pytest.mark.parametrize("second_branch", [False, True])
def test_ranked_branched_loop_before_repeated_events(tmp_path, block, second_branch):
    extra = "setp.eq.u32 %p4,%r1,1;\n@%p4 bra JOIN;\nadd.u32 %r3,%r3,1;" if second_branch else ""
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<6>;
.reg .b32 %r<5>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd2,%r1;
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
mov.u32 %r4,%ntid.x;
bar.sync 0;
EVENT_LOOP:
shr.u32 %r4,%r4,1;
bar.sync 0;
setp.gt.u32 %p5,%r4,1;
@%p5 bra EVENT_LOOP;
ret;
}}
"""
    binding = Binding(
        "kernel",
        (SizeSymbol("n", 0, 2147483647),),
        (("n", Size(Affine.symbol(0))),),
        (block, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


def test_occurrence_summary_theorems_closed(tmp_path):
    names = [
        "loop_event_free_continuation",
        "occurrence_collect_with_continuation",
        "occurrence_word_with_sound",
        "occurrence_checked_word_sound",
        "occurrence_loop_summary_sound",
        "occurrence_program_word_general_sound",
        "occurrence_program_word_preserves",
        "event_progress_general_preserves",
        "event_progress_general_sound",
        "accepted_front_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


METADATA = "From Coq Require Import List ZArith.\nFrom OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.\nImport ListNotations. Local Open Scope Z_scope.\nDefinition F:lfacts:={|\n lf_widths:=PM.add 3%positive 1%nat(PM.add 7%positive 1%nat(PM.empty nat));\n lf_fixed:=[10%positive;11%positive];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];\n lf_shared_window:=10%positive;lf_local_window:=11%positive;\n lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.\nDefinition P:program:=[\n [IImm None 1%positive 0;IBra None 1%nat];\n [IBra(Some(false,7%positive))3%nat];\n [IImm None 8%positive 42;IBra None 3%nat];\n [IImm None 2%positive 1;IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);\n  ISetp None CLt 64 false 3%positive 1%positive(OImm 100000);IBra(Some(false,3%positive))1%nat];\n [IImm None 4%positive 4;INop;IBra None 5%nat];\n [INop;IBin None BShr 64 4%positive 4%positive(OImm 1);\n  ISetp None CGt 64 false 3%positive 4%positive(OImm 0);IBra(Some(false,3%positive))5%nat];\n [IExit None]].\nDefinition alphabet:list pc:=[(4%nat,1%nat);(5%nat,0%nat)].\n"


@pytest.mark.parametrize(
    "obligation",
    [
        "occurrence_segments_acyclic P alphabet(occurrence_cut P alphabet)=false",
        "occurrence_program_word F P alphabet=None",
        "occurrence_program_word_general F P alphabet=Some[(4%nat,1%nat);(5%nat,0%nat);(5%nat,0%nat);(5%nat,0%nat)]",
        "event_progress_general F P alphabet=true",
        "occurrence_collect_with(fun _ _=>Rejected_loop_summary)10(PM.empty unit)F P alphabet(occurrence_cut P alphabet)[((0%nat,0%nat),Some(empty_state F))]=None",
        "occurrence_collect_with(occurrence_loop_summary F P alphabet)0(PM.empty unit)F P alphabet(occurrence_cut P alphabet)[((0%nat,0%nat),Some(empty_state F))]=None",
        "occurrence_loop_summary F P [(1%nat,0%nat)](1%nat,0%nat)(transfer F(empty_state F)(IImm None 1%positive 0))=Rejected_loop_summary",
        "occurrence_loop_summary F P [(2%nat,0%nat)](1%nat,0%nat)(transfer F(empty_state F)(IImm None 1%positive 0))=Rejected_loop_summary",
        "occurrence_loop_summary F P [(3%nat,1%nat)](1%nat,0%nat)(transfer F(empty_state F)(IImm None 1%positive 0))=Rejected_loop_summary",
        "occurrence_loop_summary F P [(3%nat,3%nat)](1%nat,0%nat)(transfer F(empty_state F)(IImm None 1%positive 0))=Rejected_loop_summary",
        "occurrence_loop_summary F P alphabet(1%nat,0%nat)(transfer F(empty_state F)(IImm None 1%positive 100001))=Rejected_loop_summary",
        "occurrence_program_word_general F [[IBra None 0%nat];[INop;IExit None]] [(1%nat,0%nat)]=None",
    ],
)
def test_complete_alphabet_and_fail_fast_metadata(tmp_path, obligation):
    # Metadata only: no malformed kernel or control executes on a device.
    compile_query(
        tmp_path,
        METADATA + f"Example checked : {obligation}.\nProof. vm_compute. reflexivity. Qed.\n",
    )
