"""Pure IR progress metadata checks; no device code is executed."""

import subprocess

import pytest

from opgate.gate import formal_dir


@pytest.mark.parametrize(
    "program, expected",
    [
        ("[[IExit None]]", True),
        ("[[INop; IExit None]]", True),
        ("[[IDef None [1%positive]; INop; IExit None]]", True),
        ("[[IImm (Some (false, 1%positive)) 2%positive 7; INop; IExit None]]", True),
        (
            "[[IMem (Some (false, 1%positive)) KStore false 2%positive 0 4 (Some (OImm 0)) [] [] ; INop; IExit None]]",
            True,
        ),
        ("[]", False),
        ("[[]]", False),
        ("[[INop]]", False),
        ("[[IExit (Some (false, 1%positive))]]", False),
        ("[[IExit None; INop; IExit None]]", False),
        ("[[IBra None 0%nat; IExit None]]", False),
        ("[[IBra (Some (false, 1%positive)) 0%nat; IExit None]]", False),
        ("[[INop]; [IExit None]]", False),
        ("[[]; [IExit None]]", False),
    ],
)
def test_actual_ir_structural_case(tmp_path, program, expected):
    source = tmp_path / "Query.v"
    source.write_text(
        "From Coq Require Import List ZArith.\n"
        "Require Import OpGate.GateIsa OpGate.GateLoop.\n"
        "Import ListNotations. Local Open Scope Z_scope.\n"
        f"Example actual_shape : straight_event_program {program} = {str(expected).lower()}.\n"
        "Proof. vm_compute. reflexivity. Qed.\n"
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


def test_actual_ir_event_progress_theorems_are_closed(tmp_path):
    names = [
        "straight_event_body_shape",
        "straight_event_no_branch",
        "straight_event_step",
        "straight_event_run",
        "straight_event_safe",
        "loop_events_check_sound",
        "loop_events_in_order",
        "accepted_front_events",
    ]
    source = tmp_path / "Query.v"
    source.write_text(
        "Require Import OpGate.GateLoop OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names)
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
    assert result.stdout.count("Closed under the global context") == len(names)
