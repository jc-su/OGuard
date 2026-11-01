"""Shared ranked prefixes and concrete straight event tails; CPU-only checks."""

import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol


def compile_query(tmp_path, source):
    path = tmp_path / "Query.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        text=True,
        capture_output=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "body",
    [
        "event_tail_shape [[INop;INop;IExit None]] [(0%nat,0%nat);(0%nat,1%nat)]=true",
        "event_tail_case [[INop;INop;IExit None]] [(0%nat,0%nat);(0%nat,1%nat)]=false",
        "event_tail_shape [[INop;IExit None]] []=false",
        "event_tail_shape [[INop;IExit None]] [(0%nat,0%nat);(0%nat,2%nat)]=false",
        "event_tail_shape [[INop];[INop;IExit None]] [(0%nat,0%nat);(1%nat,0%nat)]=false",
        "event_tail_shape [[INop;IBra None 0%nat;INop;IExit None]] [(0%nat,0%nat);(0%nat,2%nat)]=false",
        "event_tail_shape [[INop;IBra (Some (false,1%positive)) 0%nat;INop;IExit None]] [(0%nat,0%nat);(0%nat,2%nat)]=false",
        "event_tail_shape [[INop;IExit (Some (false,1%positive));INop;IExit None]] [(0%nat,0%nat);(0%nat,2%nat)]=false",
        "event_tail_shape [[INop;INop;IExit None]] [(0%nat,1%nat);(0%nat,0%nat)]=false",
        # The first marker is stepped; the final marker is reached, not stepped.
        "event_tail_shape [[IBra None 0%nat;INop;IExit None]] [(0%nat,0%nat);(0%nat,1%nat)]=false",
        "event_tail_shape [[INop;INop;IExit None]] [(0%nat,0%nat);(0%nat,2%nat)]=true",
    ],
)
def test_event_tail_metadata(tmp_path, body):
    compile_query(
        tmp_path,
        """
From Coq Require Import List ZArith.
Require Import OpGate.GateIsa OpGate.GateLoop.
Import ListNotations.
"""
        + f"Example checked : {body}.\nProof. vm_compute. reflexivity. Qed.\n",
    )


@pytest.mark.parametrize("events", [2, 16, 128])
def test_ranked_prefix_then_straight_event_tail(tmp_path, events):
    markers = "\n".join("bar.sync 0;" for _ in range(events))
    text = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 n) {{
.reg .pred %p<3>;
.reg .b32 %r<3>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1,[n];
mov.u32 %r1,%tid.x;
mov.u32 %r2,%ntid.x;
cvt.u64.u32 %rd2,%r1;
cvt.u64.u32 %rd3,%r2;
setp.ge.s64 %p1,%rd2,%rd1;
@%p1 bra DONE;
LOOP:
add.s64 %rd2,%rd2,%rd3;
setp.lt.s64 %p2,%rd2,%rd1;
@%p2 bra LOOP;
DONE:
{markers}
ret;
}}
"""
    binding = Binding(
        "kernel",
        (SizeSymbol("n", 0, 2147483647),),
        (("n", Size(Affine.symbol(0))),),
        (256, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    compile_query(tmp_path, export_terms(text, [binding]) + CHECK_QUERY)


def test_event_tail_theorems_closed(tmp_path):
    names = [
        "event_tail_shape_sound",
        "event_tail_step",
        "event_tail_reaches",
        "event_tail_safe",
        "event_tail_check_sound",
        "event_tail_preserves",
        "event_progress_check_sound",
        "event_progress_preserves",
        "accepted_front_events",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateLoop OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
