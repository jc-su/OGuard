"""Exact common-source factoring, checked with pure parser metadata."""

import subprocess

from opgate.gate import formal_dir


def compile_metadata(tmp_path, body):
    source = tmp_path / "Query.v"
    source.write_text(
        "From Coq Require Import List ZArith String.\n"
        "Require Import OpGate.GateIsa OpGate.GateDomain OpGate.GateFront.\n"
        "Import ListNotations. Local Open Scope Z_scope.\n"
        "Local Open Scope string_scope.\n" + body
    )
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_common_source_conditions_preserve_empty_and_invalid_metadata(tmp_path):
    compile_metadata(
        tmp_path,
        """
Definition kernel (body:list pinstr) : pkernel :=
 {| k_address_size:=64; k_target:=Some 80; k_ptx_version:=Some (8,0);
    k_params:=[]; k_regs:=[]; k_shared:=[]; k_locals:=[];
    k_reqntid:=None; k_cluster:=false; k_labels:=[];
    k_branch_targets:=[]; k_funcs:=[]; k_insts:=body |}.
Definition ordinary := PI None "mov" [] [].
Definition blocked := PI None "mbarrier" [] [].
Definition unknown_collective := PI None "shfl" [] [].
Definition guarded_uniform := PI (Some(true,"p")) "bra" ["uni"] [].
Definition sites : list place := [(1%N,[]);(2%N,[])].
Example empty_sites_ignore_bad_common_shape :
 phase_sources_ok (kernel [blocked]) [] [] = true.
Proof. vm_compute. reflexivity. Qed.
Example valid_unique_sites :
 phase_sources_ok (kernel [ordinary;ordinary]) sites [1%N;2%N] = true.
Proof. vm_compute. reflexivity. Qed.
Example repeated_requested_origin_preserves_old_result :
 phase_sources_ok (kernel [ordinary;ordinary]) sites [1%N;1%N] = true.
Proof. vm_compute. reflexivity. Qed.
Example length_mismatch :
 phase_sources_ok (kernel [ordinary]) sites [1%N;2%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example missing_first_origin :
 phase_sources_ok (kernel [ordinary;ordinary]) sites [0%N;2%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example missing_later_origin :
 phase_sources_ok (kernel [ordinary;ordinary]) sites [1%N;3%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example ambiguous_origin :
 phase_sources_ok (kernel [ordinary;ordinary])
 [(1%N,[]);(1%N,[])] [1%N;2%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example blocked_source_is_still_checked :
 phase_sources_ok (kernel [ordinary;blocked]) sites [1%N;2%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example collective_source_shape_is_still_checked :
 phase_sources_ok (kernel [ordinary;unknown_collective]) sites [1%N;2%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example uniform_control_is_still_checked :
 phase_sources_ok (kernel [ordinary;guarded_uniform]) sites [1%N;2%N] = false.
Proof. vm_compute. reflexivity. Qed.
Example singleton_uses_complete_old_check : forall k org origin,
 phase_sources_ok k org [origin] = phase_source_ok k org origin.
Proof. intros. cbn [phase_sources_ok short_all].
 destruct (phase_source_ok k org origin); reflexivity. Qed.
""",
    )


def test_common_source_exact_equivalence_is_closed(tmp_path):
    output = compile_metadata(
        tmp_path,
        "Print Assumptions phase_source_unique.\nPrint Assumptions phase_sources_exact.\n",
    )
    assert output.count("Closed under the global context") == 2
