"""Exact lexical control prefixes, including non-existent event positions."""

import subprocess

import pytest

from opgate.gate import formal_dir

HEADER = """
From Coq Require Import List ZArith.
From OpGate Require Import GateIsa GateForm GateDomain GateBarrier.
Import ListNotations.
"""


@pytest.mark.parametrize(
    "program,event,expected",
    [
        ("[]", "(0,0)", "[]"),
        ("[]", "(7,99)", "[]"),
        ("[[];[]]", "(0,5)", "[]"),
        ("[[];[]]", "(3,5)", "[]"),
        ("[[INop;IExit None]]", "(0,0)", "[]"),
        ("[[INop;IExit None]]", "(0,1)", "[INop]"),
        ("[[INop;IExit None]]", "(0,2)", "[INop;IExit None]"),
        ("[[INop;IExit None]]", "(0,99)", "[INop;IExit None]"),
        ("[[INop;IExit None]]", "(1,0)", "[INop;IExit None]"),
        ("[[INop;IExit None]]", "(9,0)", "[INop;IExit None]"),
        ("[[INop];[];[IExit None]]", "(1,0)", "[INop]"),
        ("[[INop];[];[IExit None]]", "(1,9)", "[INop]"),
        ("[[INop];[];[IExit None]]", "(2,0)", "[INop]"),
        ("[[INop];[];[IExit None]]", "(2,1)", "[INop;IExit None]"),
        ("[[INop];[];[IExit None]]", "(3,0)", "[INop;IExit None]"),
        ("[[INop];[];[IExit None]]", "(8,8)", "[INop;IExit None]"),
        (
            "[[INop;INop];[IDef None [1%positive];INop]]",
            "(1,1)",
            "[INop;INop;IDef None [1%positive]]",
        ),
        (
            "[[IBra(Some(false,1%positive)) 0;IExit None];[INop]]",
            "(1,0)",
            "[IBra(Some(false,1%positive)) 0;IExit None]",
        ),
    ],
)
def test_complete_prefix(tmp_path, program, event, expected):
    query = tmp_path / "Query.v"
    query.write_text(
        HEADER
        + f"Example checked : phase_prefix {program} {event} = {expected}.\n"
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


def test_exact_prefix_and_control_map_have_no_assumptions(tmp_path):
    query = tmp_path / "Query.v"
    query.write_text(
        HEADER
        + "Print Assumptions phase_prefix_exact.\n"
        + "Print Assumptions phase_controls_exact.\n"
        + "Print Assumptions single_phase_check_sound.\n"
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
