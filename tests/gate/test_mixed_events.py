"""Benign CPU-only progress checks and direct metadata controls; no GPU execution."""

import subprocess

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, Direction


def compile_query(tmp_path, query, timeout=45):
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("mask_mode", ["literal", "register", "snapshot"])
@pytest.mark.parametrize("split_prefix", [False, True])
@pytest.mark.parametrize(
    "collective,destinations",
    [
        ("shfl.sync.idx.b32 %r3, %r1, 0, 31, -1;", 1),
        ("shfl.sync.bfly.b32 %r3|%p2, %r1, 1, 31, 0xffffffff;", 2),
        ("vote.sync.ballot.b32 %r3, %p1, -1;", 1),
        ("vote.sync.all.pred %p2, %p1, -1;", 1),
        ("match.any.sync.b32 %r3, %r1, -1;", 1),
        ("match.all.sync.b32 %r3|%p2, %r1, -1;", 2),
        ("redux.sync.add.u32 %r3, %r1, -1;", 1),
    ],
)
def test_common_order_preserves_arbitrary_results_and_access_checks(
    tmp_path, collective, destinations, split_prefix, mask_mode
):
    if mask_mode != "literal":
        collective = collective.replace("-1;", "%r5;").replace("0xffffffff;", "%r5;")
        if mask_mode == "snapshot":
            collective += "\nmov.u32 %r5,0;"
    text = (
        """.version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {
.reg .pred %p<4>;
.reg .b32 %r<6>;
.reg .b64 %rd<3>;
ld.param.u64 %rd1,[output];
mov.u32 %r1,%tid.x;
mov.u32 %r5,-1;
setp.eq.u32 %p1,%r1,0;
@%p1 bra JOIN;
mov.u32 %r2,7;
JOIN:
bar.sync 0;
"""
        + collective
        + """
bar.sync 1;
mul.wide.u32 %rd2,%r1,4;
add.u64 %rd2,%rd1,%rd2;
st.global.u32 [%rd2],%r1;
ret;
}
"""
    )
    if not split_prefix:
        text = text.replace("@%p1 bra JOIN;\nmov.u32 %r2,7;\nJOIN:\n", "")
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(128), 4, None, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    query = export_terms(text, [binding]) + CHECK_QUERY
    query += """
Definition front_result :=
 match parse_text text size,mfs with
 | Ok ks,[(_,b)]=>match ks_entries ks with [(_,k)]=>front k b|_=>Err "entries" None end
 | _,_=>Err "parse" None end.
Example result_registers_retained :
 (match front_result with Ok (p,_)=>fold_left Nat.add
 (map (fun i=>match i with IDef None ds=>List.length ds|_=>0%nat end) (List.concat p)) 0%nat
 | _=>0%nat end)=COUNT%nat.
Proof. vm_compute. reflexivity. Qed.
""".replace("COUNT", str(destinations))
    compile_query(tmp_path, query)


def test_unresolved_event_metadata_refuses(tmp_path):
    compile_query(
        tmp_path,
        r"""
From Coq Require Import List String NArith ZArith.
Require Import OpGate.GateIsa OpGate.GateBarrier OpGate.GateFront OpGate.GateLoop.
Import ListNotations. Open Scope string_scope. Open Scope Z_scope.
Example marker_keeps_write : phase_marker (IDef None [1%positive;2%positive])=true.
Proof. vm_compute. reflexivity. Qed.
Example guarded_marker_refuses : phase_marker (IDef (Some (false,1%positive)) [2%positive])=false.
Proof. vm_compute. reflexivity. Qed.
Example jump_marker_refuses : phase_marker (IBra None 1%nat)=false.
Proof. vm_compute. reflexivity. Qed.
Example exit_marker_refuses : phase_marker (IExit None)=false.
Proof. vm_compute. reflexivity. Qed.
Example exact_result_origin : placed_event ELiteral [[(IDef None [1%positive],7%N)]] 7%N=Some (0%nat,0%nat).
Proof. vm_compute. reflexivity. Qed.
Example ambiguous_origin_refuses : placed_event ELiteral [[(INop,7%N);(IDef None [1%positive],7%N)]] 7%N=None.
Proof. vm_compute. reflexivity. Qed.
Example register_membermask_requires_checked_span : checked_sync_event_source (PI None "shfl" [] [PReg "%r1"])=true.
Proof. vm_compute. reflexivity. Qed.
Example predicated_event_refuses : checked_sync_event_source (PI (Some (false,"%p1")) "vote" [] [PImm (-1)])=false.
Proof. vm_compute. reflexivity. Qed.
Example partial_membermask_refuses : checked_sync_event_source (PI None "redux" [] [PImm 1])=false.
Proof. vm_compute. reflexivity. Qed.
Example repeat_marker_refuses : phase_closed [[IDef None [1%positive];IBra None 0%nat]] (0%nat,0%nat)=false.
Proof. vm_compute. reflexivity. Qed.
Example reversed_marker_order_refuses : loop_events_ordered [(0%nat,1%nat);(0%nat,0%nat)]=false.
Proof. vm_compute. reflexivity. Qed.
""",
    )


def test_mixed_event_proofs_are_closed(tmp_path):
    names = [
        "phase_event_at_fetch",
        "phase_marker_not_control",
        "phase_event_step",
        "phase_never_returns",
        "single_phase_check_sound",
        "ranked_phase_check_sound",
        "loop_events_check_sound",
        "loop_events_in_order",
        "accepted_front_phase",
        "accepted_front_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateBarrier OpGate.GateLoop OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names), result.stdout


def test_register_event_requires_the_complete_exact_prefix(tmp_path):
    compile_query(
        tmp_path,
        r"""
From Coq Require Import List String NArith ZArith.
Require Import OpGate.GateIsa OpGate.GateDomain OpGate.GateFront.
Import ListNotations. Open Scope string_scope. Open Scope Z_scope.
Definition requirement := ISetp None CNe 32 false 2%positive 1%positive (OImm 4294967295).
Definition witness := IImm None 3%positive 0.
Definition obligation := IMem (Some (false,2%positive)) KStore false 3%positive 0 1 None [] [].
Definition outputs := IDef None [4%positive;6%positive].
Definition placed (xs:list instr) := [map (fun i=>(i,7%N)) xs].
Definition check (xs:list instr) := placed_event (ERegister 1%positive []) (placed xs) 7%N.
Example complete_span : check [requirement;witness;obligation;outputs]=Some (0%nat,3%nat).
Proof. vm_compute. reflexivity. Qed.
Example missing_obligation : check [requirement;witness;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_comparison : check [ISetp None CEq 32 false 2%positive 1%positive (OImm 4294967295);witness;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_source_mask : check [ISetp None CNe 32 false 2%positive 5%positive (OImm 4294967295);witness;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_full_value : check [ISetp None CNe 32 false 2%positive 1%positive (OImm 0);witness;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_comparison_width : check [ISetp None CNe 64 false 2%positive 1%positive (OImm 4294967295);witness;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_witness : check [requirement;IImm None 3%positive 1;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_guard : check [requirement;witness;IMem (Some (false,5%positive)) KStore false 3%positive 0 1 None [] [];outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example wrong_frame : placed_event (ERegister 1%positive [1%N]) (placed [requirement;witness;obligation;outputs]) 7%N=None.
Proof. vm_compute. reflexivity. Qed.
Example noncontiguous_span : placed_event (ERegister 1%positive [])
 [[(requirement,7%N);(INop,8%N);(witness,7%N);(obligation,7%N);(outputs,7%N)]] 7%N=None.
Proof. vm_compute. reflexivity. Qed.
Example guarded_result : check [requirement;witness;obligation;IDef (Some (false,5%positive)) [4%positive;6%positive]]=None.
Proof. vm_compute. reflexivity. Qed.
Example extra_same_origin : check [requirement;witness;obligation;INop;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Definition saved_requirement := ISetp None CNe 32 false 2%positive 5%positive (OImm 4294967295).
Example complete_snapshot : check [IMov None 5%positive 1%positive;saved_requirement;witness;obligation;outputs]=Some (0%nat,4%nat).
Proof. vm_compute. reflexivity. Qed.
Example mismatched_copy_source : check [IMov None 5%positive 4%positive;saved_requirement;witness;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Example snapshot_missing_comparison : check [IMov None 5%positive 1%positive;witness;obligation;outputs]=None.
Proof. vm_compute. reflexivity. Qed.
Definition kernel_with (body:list pinstr) : pkernel :=
 {| k_address_size:=64; k_target:=Some 80; k_ptx_version:=Some (8,0);
    k_params:=[]; k_regs:=[]; k_shared:=[]; k_locals:=[]; k_reqntid:=None;
    k_cluster:=false; k_labels:=[]; k_branch_targets:=[]; k_funcs:=[]; k_insts:=body |}.
Definition warp := PI None "shfl" ["sync";"idx";"b32"] [PReg "%r1";PReg "%r2";PImm 0;PImm 31;PReg "%mask"].
Definition barrier := PI None "bar" ["sync"] [PImm 0].
Definition finish := PI None "ret" [] [].
Definition ids := PM.add (key "%mask") 1%positive (PM.empty reg).
Example actual_mask_map : source_event_request (kernel_with [warp]) [(7%N,[])] ids 7%N=Some (ERegister 1%positive []).
Proof. vm_compute. reflexivity. Qed.
Example missing_actual_mask_map : source_event_request (kernel_with [warp]) [(7%N,[])] (PM.empty reg) 7%N=None.
Proof. vm_compute. reflexivity. Qed.
Example duplicate_expanded_origin : source_event_request (kernel_with [warp;warp]) [(7%N,[]);(7%N,[])] ids 7%N=None.
Proof. vm_compute. reflexivity. Qed.
Example mixed_linear_body_uses_progress : barrier_linear_case (kernel_with [warp;barrier;finish])=false.
Proof. vm_compute. reflexivity. Qed.
Example barrier_only_linear_body : barrier_linear_case (kernel_with [barrier;finish])=true.
Proof. vm_compute. reflexivity. Qed.
""",
    )
