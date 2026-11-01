"""Shallow constraint classification respects wraparound and contradictory states."""

import subprocess

from opgate.gate import formal_dir


def test_shallow_constraint_classification_and_structural_precedence(tmp_path):
    source = r"""
From Coq Require Import ZArith Lists.List Bool.Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateSoundJoin.
Import ListNotations. Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 1%positive 64%nat (PM.add 2%positive 64%nat (PM.empty nat));
 lf_fixed := []; lf_bounds := []; lf_aligns := []; lf_roles := [];
 lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit; lf_derived := []; lf_names := [] |}.
Definition span_state (lo hi:Z) : astate :=
 {|amap:=PM.add 1%positive (FForm (range_form 64 lo hi) 0 None) (PM.empty fact);acons:=[]|}.
Definition one := [((1%positive,EX),1)].
Example contained : near_constraint facts (span_state 10 20) one 10 20 = Some true.
Proof. vm_compute. reflexivity. Qed.
Example disjoint_above : near_constraint facts (span_state 10 20) one 0 9 = Some false.
Proof. vm_compute. reflexivity. Qed.
Example disjoint_below : near_constraint facts (span_state 10 20) one 21 30 = Some false.
Proof. vm_compute. reflexivity. Qed.
Example overlap : near_constraint facts (span_state 10 20) one 15 25 = None.
Proof. vm_compute. reflexivity. Qed.
Example touching_is_inconclusive : near_constraint facts (span_state 10 20) one 20 25 = None.
Proof. vm_compute. reflexivity. Qed.
Definition wrapped : astate := {|
 amap:=PM.add 1%positive (FForm (fconst 64 (2^64-6)) 0 None)
       (PM.add 2%positive (FForm (fconst 64 (2^64-7)) 0 None) (PM.empty fact));acons:=[]|}.
Definition sum := [((1%positive,EX),1);((2%positive,EX),1)].
Example integer_sum_is_not_its_modular_representative :
 near_constraint facts wrapped sum (-2^65) (-13) = Some false.
Proof. vm_compute. reflexivity. Qed.
Definition contradictory : astate := {|amap:=amap(span_state 10 20);acons:=[(one,0,0)]|}.
Example exact_structural_fact_retained : cons_holds facts contradictory (one,0,0) = true.
Proof. vm_compute. reflexivity. Qed.
Example contradictory_interval_is_empty :
 form_range_near facts contradictory (form_of_terms one) = Some (10,0).
Proof. vm_compute. reflexivity. Qed.
Example vacuous_containment_retained : near_constraint facts contradictory one 0 0 = Some true.
Proof. vm_compute. reflexivity. Qed.
Print Assumptions near_constraint_sound.
Print Assumptions near_constraint_refutes.
Print Assumptions cons_holds_sound.
"""
    query = tmp_path / "ConstraintTriage.v"
    query.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 3, result.stdout
