"""Checked complete event words per physical warp; static CPU checks only."""

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


@pytest.mark.parametrize("block,cut", [(64, 32), (128, 64), (256, 96), (1024, 32)])
@pytest.mark.parametrize("repeated", [False, True])
def test_warp_uniform_optional_collective(tmp_path, block, cut, repeated):
    prefix = (
        "mov.u32 %r4,4;\nREPEAT:\nbar.sync 0;\nshr.u32 %r4,%r4,1;\nsetp.gt.u32 %p2,%r4,0;\n@%p2 bra REPEAT;"
        if repeated
        else "bar.sync 0;"
    )
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel() {{
.reg .pred %p<3>;
.reg .b32 %r<5>;
mov.u32 %r1,%tid.x;
{prefix}
setp.ge.u32 %p1,%r1,{cut};
@%p1 bra DONE;
shfl.sync.idx.b32 %r2,%r1,0,31,-1;
DONE:
ret;
}}
"""
    binding = Binding("kernel", (), (), (block, 1, 1), (1, 1, 1), Affine(0))
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


META = """From Coq Require Import List ZArith String.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop GateFront.
Import ListNotations. Local Open Scope Z_scope. Local Open Scope string_scope.
Definition F (bits:nat) (names:list(string*reg)) : lfacts := {|
 lf_widths:=PM.add 1%positive bits(PM.add 3%positive 1%nat(PM.empty nat));
 lf_fixed:=[1%positive;10%positive;11%positive];lf_bounds:=[(1%positive,(0,63))];
 lf_aligns:=[];lf_roles:=[];lf_shared_window:=10%positive;lf_local_window:=11%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];
 lf_names:=names|}.
Definition named:lfacts:=F 32 [("%tid.x",1%positive)].
Definition M (x y z:Z):binding:={| b_block:=(x,y,z);b_grid:=(1,1,1);b_shared:={|af_terms:=[];af_const:=0|};
 b_sizes:=[];b_params:=[];b_derived:=[]|}.
Definition P (cut:Z):program:=[
 [INop;ISetp None CGe 32 false 3%positive 1%positive(OImm cut);IBra(Some(false,3%positive))2%nat];
 [IDef None[4%positive];IBra None 2%nat];[INop;IExit None]].
Definition events:list pc:=[(0%nat,0%nat);(1%nat,0%nat);(2%nat,0%nat)].
Definition cta:list pc:=[(0%nat,0%nat);(2%nat,0%nat)].
Definition repeated:program:=[
 [ISetp None CGe 32 false 3%positive 1%positive(OImm 32);IBra(Some(false,3%positive))2%nat];
 [IImm None 5%positive 2;IBra None 3%nat];
 [IImm None 5%positive 1;IBra None 3%nat];
 [INop;IBin None BSub 64 5%positive 5%positive(OImm 1);
  ISetp None CGt 64 false 3%positive 5%positive(OImm 0);IBra(Some(false,3%positive))3%nat];
 [IExit None]].
Definition reordered:program:=[
 [ISetp None CGe 32 false 3%positive 1%positive(OImm 32);IBra(Some(false,3%positive))3%nat];
 [INop;IBra(Some(false,3%positive))4%nat];
 [IBra None 3%nat];
 [INop;IBra(Some(false,3%positive))1%nat];
 [IExit None]].
Definition order_events:list pc:=[(1%nat,0%nat);(3%nat,0%nat)].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "warp_launch_shape named(M 64 1 1)=Some(1%positive,2%nat)",
        "warp_launch_shape named(M 63 1 1)=None",
        "warp_launch_shape named(M 64 2 1)=None",
        "warp_launch_shape named(M 64 1 2)=None",
        "warp_launch_shape named(M 0 1 1)=None",
        "warp_launch_shape named(M 1056 1 1)=None",
        'warp_launch_shape(F 5[("%tid.x",1%positive)])(M 64 1 1)=None',
        "warp_launch_shape(F 32[])(M 64 1 1)=None",
        'warp_launch_shape(F 32[("%laneid",1%positive);("%tid.x",1%positive)])(M 64 1 1)=None',
        "warp_scoped_check named(P 32)events cta 1%positive 2=Some[(0%nat,events);(1%nat,cta)]",
        "warp_scoped_check named(P 1)events cta 1%positive 2=None",
        "warp_scoped_check named(P 32)events events 1%positive 2=None",
        "warp_scoped_check named repeated[(3%nat,0%nat)][(3%nat,0%nat)]1%positive 2=None",
        "scoped_complete_word(warp_facts named 1%positive 0)reordered order_events=Some order_events",
        "scoped_complete_word(warp_facts named 1%positive 1)reordered order_events=Some(rev order_events)",
        "warp_scoped_check named reordered order_events order_events 1%positive 2=None",
        "event_projection cta[(0%nat,0%nat);(1%nat,0%nat);(0%nat,0%nat);(2%nat,0%nat)]=[(0%nat,0%nat);(0%nat,0%nat);(2%nat,0%nat)]",
        "event_word_eqb[(0%nat,0%nat);(2%nat,0%nat)][(2%nat,0%nat);(0%nat,0%nat)]=false",
        "event_word_eqb[(0%nat,0%nat);(0%nat,0%nat)][(0%nat,0%nat)]=false",
        "scoped_words_check(fun _ _ _=>None)named(P 32)events cta 1%positive 2=None",
    ],
)
def test_scoped_metadata_obligations(tmp_path, obligation):
    # Direct IR/certificate obligations only; these controls do not launch PTX.
    compile_query(
        tmp_path, META + f"Example checked:{obligation}. Proof. vm_compute. reflexivity. Qed.\n"
    )


def test_scoped_theorems_closed(tmp_path):
    names = [
        "warp_launch",
        "occurrence_trace_warp_same",
        "scoped_words_check_sound",
        "scoped_complete_word_sound",
        "warp_scoped_check_sound",
        "warp_launch_shape_sound",
        "front_event_check_sound",
        "accepted_front_events",
        "physical_warp_partition",
        "launch_value_tid",
        "linked_scoped_schedule_sound",
        "accepted_front_linked_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.GateFront OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
