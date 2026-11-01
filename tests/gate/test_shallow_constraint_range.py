"""Shallow classification preserves both terminal answers and pending overlap."""

import subprocess

from opgate.gate import formal_dir


def test_shallow_constraint_range(tmp_path):
    source = r"""
From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain GateSoundJoin.
Import ListNotations.
Open Scope Z_scope.
Example unknown_range_keeps_pending : range_constraint 0 8 None = None.
Proof. reflexivity. Qed.
Example exact_endpoints_are_contained : range_constraint 0 8 (Some (0,8)) = Some true.
Proof. reflexivity. Qed.
Example strict_interior_is_contained : range_constraint 0 8 (Some (1,7)) = Some true.
Proof. reflexivity. Qed.
Example above_is_disjoint : range_constraint 0 8 (Some (9,10)) = Some false.
Proof. reflexivity. Qed.
Example below_is_disjoint : range_constraint 0 8 (Some (-2,-1)) = Some false.
Proof. reflexivity. Qed.
Example overlap_left_is_pending : range_constraint 0 8 (Some (-1,1)) = None.
Proof. reflexivity. Qed.
Example overlap_right_is_pending : range_constraint 0 8 (Some (7,9)) = None.
Proof. reflexivity. Qed.
Example boundary_touch_is_pending : range_constraint 0 8 (Some (8,9)) = None.
Proof. reflexivity. Qed.
Example encompassing_range_is_pending : range_constraint 0 8 (Some (-1,9)) = None.
Proof. reflexivity. Qed.
Example unbounded_integer_range_is_pending :
 range_constraint 0 (2^64-1) (Some (-2^80,2^80)) = None.
Proof. vm_compute. reflexivity. Qed.
Example near_constraint_keeps_api : forall F a ts lo hi,
 near_constraint F a ts lo hi = range_constraint lo hi (form_range_near F a (form_of_terms ts)).
Proof. reflexivity. Qed.
Print Assumptions near_constraint_sound.
Print Assumptions near_constraint_refutes.
Print Assumptions cons_holds_exact.
"""
    path = tmp_path / "ShallowRange.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 3, result.stdout
