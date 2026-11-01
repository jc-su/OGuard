"""Checked progress with local cells disjoint from the ranking registers."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir, zeroed
from opgate.signature import Affine, SizeSymbol

CELL_CHECK = "\nRequire Import OpGate.GateBarrier OpGate.GateLoop.\nDefinition actual_parts := Eval vm_compute in\n match parse_text text size with\n | Ok ks=>match ks_entries ks,mfs with\n | [(name,K)],[(bound,Mf)]=>\n  match inline K with\n  | Ok(K',org)=>match(p<-translate_placed K' Mf org;;F<-facts K' Mf;;ret(p,F))init with\n    | Ok((pp,raw),ts)=>Some(K',expanded_places org,pp,payload_facts raw(map(map fst)pp),ts_ids ts,Mf)\n    | _=>None end\n  | _=>None end\n | _,_=>None end\n | _=>None end.\nDefinition args:=ltac:(let x:=eval vm_compute in actual_parts in lazymatch x with Some ?p=>exact p|_=>fail \"metadata unavailable\" end).\nDefinition local_facts:=let '(K,org,pp,F,ids,Mf):=args in F.\nExample exercises_local_cells:negb(Nat.eqb(List.length(lf_cells local_facts))0)=true.\nProof. vm_compute. reflexivity. Qed.\n"

BINDING = Binding(
    "kernel",
    (SizeSymbol("n", 0, 2147483647),),
    (("n", Size(Affine.symbol(0))),),
    (256, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def local_loop(branching, generic):
    branch = "setp.eq.u32 %p3,%r1,0; @%p3 bra JOIN; add.u32 %r3,%r1,1; JOIN:" if branching else ""
    store = "st.u64 [%SP],%rd2;" if generic else "st.local.u64 [%SPL],%rd2;"
    return f""" .version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.local .align 8 .b8 scratch[8];
.reg .b64 %SPL, %SP;
.reg .pred %p<4>;
.reg .b32 %r<4>;
.reg .b64 %rd<5>;
mov.u64 %SPL,scratch;
cvta.local.u64 %SP,%SPL;
ld.param.u64 %rd1,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd2,%r1;
cvt.u64.u32 %rd3,%r2;
bar.sync 0;
setp.ge.s64 %p1,%rd2,%rd1;
@%p1 bra DONE;
LOOP:
{branch}
{store}
ld.local.u64 %rd4,[%SPL];
add.s64 %rd2,%rd2,%rd3;
setp.lt.s64 %p2,%rd2,%rd1;
@%p2 bra LOOP;
DONE:
bar.sync 0;
ret;
}}
"""


def compile_query(tmp_path, query):
    path = tmp_path / "Query.v"
    path.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout").write_text(result.stdout)
    (tmp_path / "stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("branching", [False, True])
@pytest.mark.parametrize("generic", [False, True])
def test_progress_through_unrelated_local_cells(tmp_path, branching, generic):
    text = local_loop(branching, generic)
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(BINDING))
    compile_query(
        tmp_path, export_terms(zeroed(text, BINDING), [BINDING]) + CELL_CHECK + CHECK_QUERY
    )


def test_local_loop_preservation_has_no_axioms(tmp_path):
    names = [
        "loop_step_preserves",
        "loop_linear_sound",
        "loop_iteration_sound",
        "loop_dag_all_sound",
        "loop_dag_check_sound",
        "accepted_front_linked_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.GateFront OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


def test_implicit_cell_writes_cannot_change_protected_registers(tmp_path):
    metadata = r"""
From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateForm GateDomain GateLoop.
Import ListNotations. Open Scope Z_scope.
Definition facts (cells:list(reg*reg*Z)):lfacts:={|
 lf_widths:=PM.add 3%positive 1%nat(PM.empty nat);
 lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=cells;
 lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition empty:astate:={|amap:=PM.empty fact;acons:=[]|}.
Definition d:loop_info:={|li_prefix:=0;li_suffix:=0;li_counter:=1%positive;
 li_source:=1%positive;li_stride:=2%positive;li_predicate:=3%positive;li_negated:=false|}.
Definition P:program:=[[IStL None 10%positive 0 8 (Some(OImm 0)) [];INop]].
Example counter_cell_refused:
 loop_iteration (facts [(1%positive,10%positive,0)]) P (0%nat,0%nat) d empty=None.
Proof. vm_compute. reflexivity. Qed.
Example stride_cell_refused:
 loop_iteration (facts [(2%positive,10%positive,0)]) P (0%nat,0%nat) d empty=None.
Proof. vm_compute. reflexivity. Qed.
Example protected_cell_refused:
 loop_dag_check 8 (facts [(1%positive,10%positive,0)]) P [1%positive] []
 (0%nat,0%nat)(0%nat,1%nat)empty empty=false.
Proof. vm_compute. reflexivity. Qed.
"""
    compile_query(tmp_path, metadata)
