"""Projection keeps complete dynamic components across fixed launch boundaries."""

import subprocess

from opgate.gate import formal_dir

SOURCE = """From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain GateSoundAccess PipelineIsolation.
Import ListNotations. Open Scope Z_scope.
Definition F : lfacts := {|lf_widths:=PM.empty nat;lf_fixed:=[9%positive];
 lf_bounds:=[(9%positive,(1,100))];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=10%positive;lf_local_window:=11%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;
 lf_derived:=[];lf_names:=[]|}.
Definition x:atom:=(1%positive,EX).
Definition y:atom:=(2%positive,EX).
Definition z:atom:=(3%positive,EX).
Definition fixed:atom:=(9%positive,EX).
Definition kxy:constraint:=([(x,1);(y,-1)],-100,0).
Definition kyf:constraint:=([(y,1);(fixed,-1)],-100,-1).
Definition kzf:constraint:=([(z,1);(fixed,-1)],-100,0).
Definition kz:constraint:=([(z,1)],50,100).
Definition kf:constraint:=([(fixed,1)],1,100).
Definition kx:constraint:=([(x,1)],0,100).
Definition all:=[kxy;kyf;kzf;kz;kf;kx].
Example dynamic_chain_retained:projection_slice F [(x,1)] all=[kxy;kyf;kf;kx].
Proof. vm_compute. reflexivity. Qed.
Example fixed_only_kept:projection_slice F [(fixed,1)] all=[kf].
Proof. vm_compute. reflexivity. Qed.
Example target_component_combination:projection_slice F [(x,1);(z,1)] all=all.
Proof. vm_compute. reflexivity. Qed.
Example fixed_separator_does_not_remove_intermediate:
 projection_slice F [(x,1)] [kzf;kyf;kxy;kf;kx]=[kyf;kxy;kf;kx].
Proof. vm_compute. reflexivity. Qed.
Definition contradiction:constraint:=([],1,0).
Example constant_contradiction_retained:
 projection_slice F [(x,1)] (contradiction::all)=contradiction::[kxy;kyf;kf;kx].
Proof. vm_compute. reflexivity. Qed.
Definition conflict:constraint:=([(z,1)],101,200).
Example unrelated_conflict_is_only_weakened:
 projection_slice F [(x,1)] (conflict::all)=[kxy;kyf;kf;kx].
Proof. vm_compute. reflexivity. Qed.
Definition state : astate := {|amap:=PM.empty fact;acons:=all|}.
Definition target:form:=mk 64 [(x,1)] 0 0.
Goal True. idtac "query_chain_bound". exact I. Qed.
Example complete_chain_bound:check_range F state target=Some(0,99).
Proof. vm_compute. reflexivity. Qed.
Goal True. idtac "fixed_target_scope". exact I. Qed.
Example explicit_fixed_precision_scope:check_range F state (mk 64 [(fixed,1)] 0 0)=Some(1,100).
Proof. vm_compute. reflexivity. Qed.
Print Assumptions project_range_sound.
Print Assumptions tenant_isolation.
"""


def test_projection_component_constraints_and_bounds(tmp_path):
    source = tmp_path / "ProjectionComponents.v"
    source.write_text(SOURCE)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout").write_text(result.stdout)
    (tmp_path / "stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 2
