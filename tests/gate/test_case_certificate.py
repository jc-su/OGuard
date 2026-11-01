"""Invariant certificates cover the entry and every original loop edge."""

import subprocess

from opgate.gate import formal_dir


def test_case_certificates_require_complete_reachable_coverage(tmp_path):
    source = r"""
From Coq Require Import ZArith Lists.List Bool.Bool.
From OpGate Require Import GateIsa GateForm GateDomain.
Import ListNotations. Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 1%positive 32%nat (PM.add 2%positive 1%nat (PM.empty nat));
 lf_fixed := [10%positive;11%positive]; lf_bounds := [];
 lf_aligns := []; lf_roles := []; lf_shared_window := 10%positive;
 lf_local_window := 11%positive; lf_shared := []; lf_locals := [];
 lf_cells := []; lf_payloads := PM.empty unit; lf_derived := []; lf_names := [] |}.
Definition code : program := [
 [IImm None 1%positive 0];
 [IBin None BAdd 32 1%positive 1%positive (OImm 1);
  ISetp None CNe 32 false 2%positive 1%positive (OImm 2);
  IBra (Some (false,2%positive)) 1];
 [IExit None]].
Definition certificate := Eval vm_compute in propose_cases facts code [(1%nat,1%nat,3%nat)].
Definition cases := match certificate with Some inv => inv | None => [] end.
Example valid : check_cases facts code cases = true.
Proof. vm_compute. reflexivity. Qed.
Example missing_entry : check_cases facts code (set_nth cases 0 []) = false.
Proof. vm_compute. reflexivity. Qed.
Example missing_iteration :
 check_cases facts code (set_nth cases 1 (firstn 1 (cases_at cases 1))) = false.
Proof. vm_compute. reflexivity. Qed.
Example invented_unreachability : check_cases facts code (set_nth cases 1 [bottom_state]) = false.
Proof. vm_compute. reflexivity. Qed.
Example missing_exit : check_cases facts code (set_nth cases 2 []) = false.
Proof. vm_compute. reflexivity. Qed.
"""
    path = tmp_path / "CaseCoverage.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
