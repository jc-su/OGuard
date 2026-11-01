"""Static eligibility for complete occurrence segments; no device execution."""

import subprocess

import pytest

from opgate.gate import formal_dir

METADATA = "From Coq Require Import List ZArith.\nFrom OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.\nImport ListNotations. Local Open Scope Z_scope.\nDefinition F:lfacts:={|\n lf_widths:=PM.add 3%positive 1%nat(PM.add 7%positive 1%nat(PM.empty nat));\n lf_fixed:=[10%positive;11%positive];lf_bounds:=[];lf_aligns:=[];lf_roles:=[];\n lf_shared_window:=10%positive;lf_local_window:=11%positive;\n lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.\nDefinition P:program:=[\n [IImm None 1%positive 0;IBra None 1%nat];\n [IBra(Some(false,7%positive))3%nat];\n [IImm None 8%positive 42;IBra None 3%nat];\n [IImm None 2%positive 1;IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);\n  ISetp None CLt 64 false 3%positive 1%positive(OImm 100000);IBra(Some(false,3%positive))1%nat];\n [IImm None 4%positive 4;INop;IBra None 5%nat];\n [INop;IBin None BShr 64 4%positive 4%positive(OImm 1);\n  ISetp None CGt 64 false 3%positive 4%positive(OImm 0);IBra(Some(false,3%positive))5%nat];\n [IExit None]].\nDefinition alphabet:list pc:=[(4%nat,1%nat);(5%nat,0%nat)].\n"

METADATA += """
Definition acyclic_priority:program:=[
 [IBra None 2%nat];
 [IBin None BAdd 64 1%positive 1%positive(OReg 2%positive);
  ISetp None CLt 64 false 3%positive 1%positive(OImm 100);IBra(Some(false,3%positive))2%nat];
 [IBra None 3%nat];[INop;IExit None]].
Definition priority_events:list pc:=[(3%nat,0%nat)].
"""


@pytest.mark.parametrize(
    "obligation",
    [
        "occurrence_segments_acyclic acyclic_priority priority_events(scoped_cut acyclic_priority priority_events)=true",
        "occurrence_segments_supported acyclic_priority priority_events(scoped_cut acyclic_priority priority_events)=false",
        "scoped_complete_word F acyclic_priority priority_events=Some priority_events",
        "occurrence_segments_supported P alphabet(scoped_cut P alphabet)=true",
        "scoped_complete_word F P alphabet=Some[(4%nat,1%nat);(5%nat,0%nat);(5%nat,0%nat);(5%nat,0%nat)]",
        "occurrence_segments_supported [[IBra None 0%nat];[INop;IExit None]] [(1%nat,0%nat)](1%nat,0%nat)=false",
        "scoped_complete_word F [[IBra None 0%nat];[INop;IExit None]] [(1%nat,0%nat)]=None",
        "occurrence_segments_supported [[INop;IBra None 0%nat]] [(0%nat,0%nat)](0%nat,1%nat)=true",
        "scoped_complete_word F P ((2%nat,0%nat)::alphabet)=None",
    ],
)
def test_supported_segment_preflight(tmp_path, obligation):
    source = tmp_path / "Query.v"
    source.write_text(
        METADATA + f"Example checked:{obligation}. Proof. vm_compute. reflexivity. Qed.\n"
    )
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
