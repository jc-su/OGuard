"""Checked structural closure boundaries; no PTX changes or device execution."""

import subprocess

import pytest

from opgate.gate import formal_dir

HEADER = """
From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.
Import ListNotations.
"""


@pytest.mark.parametrize(
    "program,event,expected",
    [
        ("[]", "(0,0)", True),
        ("[[];[]]", "(0,0)", True),
        ("[[INop;IExit None]]", "(0,0)", True),
        ("[[IBra None 0]]", "(0,0)", True),
        ("[[IBra None 0;INop]]", "(0,1)", True),
        ("[[INop;IBra None 0]]", "(0,0)", False),
        ("[[INop;IBra (Some(false,1%positive)) 0]]", "(0,0)", False),
        ("[[INop;IBra None 1];[IExit None]]", "(0,0)", True),
        ("[[INop];[];[IBra None 0]]", "(0,0)", False),
        ("[[INop];[];[IBra None 1]]", "(0,0)", True),
        ("[[INop];[];[IBra None 1]]", "(1,0)", False),
        ("[[INop;IBra None 0]]", "(7,999)", True),
        ("[[INop];[IBra None 0]]", "(0,999)", False),
        ("[[INop;IBra None 999]]", "(0,0)", True),
    ],
)
def test_phase_closed_boundaries(tmp_path, program, event, expected):
    query = tmp_path / "Query.v"
    query.write_text(
        HEADER
        + f"Example checked : phase_closed {program} {event} = {str(expected).lower()}.\n"
        + "Proof. vm_compute. reflexivity. Qed.\n"
    )
    run = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(run.stdout)
    (tmp_path / "stderr.txt").write_text(run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr


def test_phase_closed_exactness_has_no_assumptions(tmp_path):
    query = tmp_path / "Query.v"
    query.write_text(
        HEADER
        + "Print Assumptions phase_closed_exact.\n"
        + "Print Assumptions phase_closed_branch.\n"
        + "Print Assumptions straight_event_no_branch.\n"
    )
    run = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(run.stdout)
    (tmp_path / "stderr.txt").write_text(run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("Closed under the global context") == 3
    assert "Axioms:" not in run.stdout
