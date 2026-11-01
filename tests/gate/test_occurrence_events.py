"""Pure event metadata and benign positive static tests; no device execution."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol

HEADER = """
From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.
Import ListNotations. Local Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 2%positive 1%nat (PM.add 3%positive 1%nat (PM.empty nat));
 lf_fixed := [10%positive;11%positive]; lf_bounds := []; lf_aligns := []; lf_roles := [];
 lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit;
 lf_derived := []; lf_names := [] |}.
Definition loop_body := [INop;IBin None BShr 64 1%positive 1%positive (OImm 1);
 ISetp None CGt 64 false 2%positive 1%positive (OImm 0);
 IBra (Some (false,2%positive)) 1%nat].
Definition simple := [[IImm None 1%positive 8;INop;IBra None 1%nat];loop_body;[IExit None]].
Definition alphabet := [(0%nat,1%nat);(1%nat,0%nat)].
"""


def compile_query(tmp_path, source):
    p = tmp_path / "Query.v"
    p.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(p)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "body",
    [
        "occurrence_segments_acyclic simple alphabet (occurrence_cut simple alphabet)=true",
        "occurrence_segments_acyclic [[IBra None 0%nat];[INop;IExit None]] [(1%nat,0%nat)] (1%nat,1%nat)=false",
        "occurrence_segments_acyclic [[INop;IBra None 1%nat];[INop;IBra None 2%nat];[IBra None 2%nat]] [(0%nat,0%nat);(1%nat,0%nat)] (1%nat,1%nat)=true",
        "occurrence_segments_acyclic [[IBra (Some (false,3%positive)) 2%nat];[IBra None 3%nat];[IBra None 3%nat];[INop;IExit None]] [(3%nat,0%nat)] (3%nat,1%nat)=true",
        "occurrence_segments_acyclic [[IImm None 3%positive 0;IBra (Some (false,3%positive)) 0%nat];[INop;IExit None]] [(1%nat,0%nat)] (1%nat,1%nat)=false",
        "lf_ok facts=true",
        "occurrence_program_word facts simple alphabet=Some [(0%nat,1%nat);(1%nat,0%nat);(1%nat,0%nat);(1%nat,0%nat);(1%nat,0%nat)]",
        "occurrence_case simple alphabet=true",
        "event_progress_check facts simple alphabet=true",
        "occurrence_collect 1 (PM.empty unit) facts simple alphabet (1%nat,3%nat) [((0%nat,0%nat),Some (empty_state facts))]=None",
        "occurrence_collect 12 (PM.empty unit) facts [[IBra None 0%nat];[INop;IExit None]] [(1%nat,0%nat)] (1%nat,1%nat) [((0%nat,0%nat),Some (empty_state facts))]=None",
        "occurrence_program_word facts [[IBra (Some (false,3%positive)) 2%nat];[INop;IExit None];[IExit None]] [(1%nat,0%nat)]=None",
        "occurrence_program_word facts [[IBra (Some (false,3%positive)) 2%nat];[INop;IBra None 3%nat];[INop;IBra None 3%nat];[IExit None]] [(1%nat,0%nat);(2%nat,0%nat)]=None",
        "occurrence_program_word facts [[IImm None 1%positive 8;INop;IBra None 1%nat];loop_body;[IBra None 2%nat]] alphabet=Some [(0%nat,1%nat);(1%nat,0%nat);(1%nat,0%nat);(1%nat,0%nat);(1%nat,0%nat)]",
        "occurrence_program_word facts [[IImm None 1%positive 8;INop;IBra None 1%nat];loop_body;[INop;IExit None]] [(0%nat,1%nat);(1%nat,0%nat);(2%nat,0%nat)]=Some [(0%nat,1%nat);(1%nat,0%nat);(1%nat,0%nat);(1%nat,0%nat);(1%nat,0%nat);(2%nat,0%nat)]",
        "occurrence_case [[INop;IExit None]] [(0%nat,0%nat)]=false",
        "occurrence_case [[INop;IBra None 1%nat];[INop;IExit None]] [(0%nat,0%nat);(1%nat,0%nat)]=false",
        "occurrence_program_word facts [[INop;IExit None]] [(0%nat,2%nat)]=None",
    ],
)
def test_occurrence_metadata(tmp_path, body):
    compile_query(
        tmp_path, HEADER + f"Example checked : {body}.\nProof. vm_compute. reflexivity. Qed.\n"
    )


@pytest.mark.parametrize("block", [2, 4, 16, 256])
@pytest.mark.parametrize("divergent_payload", [False, True])
def test_benign_countdown_with_reconvergent_payload(tmp_path, block, divergent_payload):
    payload = (
        """setp.ge.u32 %p2,%r2,%r1;
@%p2 bra JOIN;
add.u32 %r3,%r2,1;
JOIN:
"""
        if divergent_payload
        else ""
    )
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<3>;
.reg .b32 %r<4>;
mov.u32 %r1,%ntid.x;
mov.u32 %r2,%tid.x;
bar.sync 0;
LOOP:
shr.u32 %r1,%r1,1;
{payload}bar.sync 0;
setp.gt.u32 %p1,%r1,1;
@%p1 bra LOOP;
ret;
}}
"""
    binding = Binding(
        "kernel",
        (SizeSymbol("n", 0, 10),),
        (("n", Size(Affine.symbol(0))),),
        (block, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


def test_occurrence_theorems_closed(tmp_path):
    names = [
        "occurrence_collect_continuation",
        "occurrence_word_sound",
        "occurrence_trace_realizes",
        "occurrence_program_word_sound",
        "event_progress_check_sound",
        "event_progress_preserves",
        "accepted_front_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {n}.\n" for n in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
