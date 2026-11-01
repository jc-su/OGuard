"""Unsigned machine bounds refine the existing modular equality window."""

import subprocess

from opgate.gate import formal_dir


def test_natural_window_preserves_modular_boundaries(tmp_path):
    source = r"""From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateSoundAccess.
Import ListNotations. Local Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths:=PM.empty nat;lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;lf_shared:=[];
 lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition modulus:=2^64.
Definition relation(c:Z):form := {|fm:=64;ft:=[((2%positive,EX),1)];flo:=c;fhi:=c|}.
Definition state(c:Z):astate := {|amap:=PM.add 1%positive(FForm(relation c)0 None)(PM.empty fact);acons:=[]|}.
Definition reads(lo hi:Z):rfuns:=
 {|f_rr:=fun r=>if Pos.eqb r 1%positive then Some(lo,hi)else Some(0,0);
   f_sr:=fun _=>None|}.
Definition natural_only:rfuns:={|f_rr:=fun r=>if Pos.eqb r 1%positive then None else Some(0,0);f_sr:=fun _=>None|}.
Definition equation(c:Z):list constraint := [([((1%positive,EX),1);((2%positive,EX),-1)],c,c)].
Example precise_window_retained:form_eq_with facts(reads 0 0)(state 0)(1%positive,EX)=equation 0.
Proof. vm_compute. reflexivity. Qed.
Example broad_inferred_lower_clipped:form_eq_with facts(reads(-modulus)3)(state 0)(1%positive,EX)=equation 0.
Proof. vm_compute. reflexivity. Qed.
Example broad_inferred_upper_clipped:form_eq_with facts(reads 0 modulus)(state 0)(1%positive,EX)=equation 0.
Proof. vm_compute. reflexivity. Qed.
Example natural_range_when_inference_absent:form_eq_with facts natural_only(state 0)(1%positive,EX)=equation 0.
Proof. vm_compute. reflexivity. Qed.
Example positive_full_modulus_is_not_integer_equality:form_eq_with facts(reads 0 0)(state modulus)(1%positive,EX)=[].
Proof. vm_compute. reflexivity. Qed.
Example negative_full_modulus_is_not_integer_equality:form_eq_with facts(reads 0 0)(state(-modulus))(1%positive,EX)=[].
Proof. vm_compute. reflexivity. Qed.
Example signed_atom_unchanged:form_eq_with facts(reads 0 0)(state 0)(1%positive,ES)=[].
Proof. vm_compute. reflexivity. Qed.
Example shifted_atom_unchanged:form_eq_with facts(reads 0 0)(state 0)(1%positive,EShr 1)=[].
Proof. vm_compute. reflexivity. Qed.
Example low_bits_atom_unchanged:form_eq_with facts(reads 0 0)(state 0)(1%positive,EBf 0 5)=[].
Proof. vm_compute. reflexivity. Qed.
Example zero_extended_atom_clipped:
 form_eq_with facts(reads(-modulus)3)(state 0)(1%positive,EZ)=
 [([((1%positive,EZ),1);((2%positive,EX),-1)],0,0)].
Proof. vm_compute. reflexivity. Qed.
Definition narrow_facts : lfacts := {|
 lf_widths:=PM.add 1%positive 32%nat(PM.empty nat);lf_fixed:=[];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;lf_shared:=[];
 lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Example narrow_register_is_not_lifted:
 form_eq_with narrow_facts(reads 0 0)(state 0)(1%positive,EX)=[].
Proof. vm_compute. reflexivity. Qed.
Definition narrow_state : astate :=
 {|amap:=PM.add 1%positive(FForm(fatom 32(2%positive,EX)1)0 None)(PM.empty fact);acons:=[]|}.
Example narrow_form_is_not_lifted:
 form_eq_with facts(reads 0 0)narrow_state(1%positive,EX)=[].
Proof. vm_compute. reflexivity. Qed.
Print Assumptions project_range_sound.
Print Assumptions check_range_sound.
Print Assumptions form_eq_local.
"""
    path = tmp_path / "NaturalWindow.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 3, result.stdout
