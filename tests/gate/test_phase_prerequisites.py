"""Prerequisite scheduling preserves every checked Boolean result."""

import subprocess

from opgate.gate import formal_dir


def test_phase_prerequisite_equalities_are_closed(tmp_path):
    source = """
From Coq Require Import List Bool.
Require Import OpGate.GateDomain OpGate.GateBarrier OpGate.GateLoop.
Import ListNotations.
Print Assumptions single_phase_check_exact.
Print Assumptions ranked_phase_check_exact.
Print Assumptions short_all_prerequisites.
Print Assumptions loop_events_check_exact.
Example empty_schedule_classification : forall F prog,
 loop_events_check F prog [] =
 if straight_event_program prog then lf_ok F else true.
Proof.
 intros F prog. rewrite loop_events_check_exact.
 cbn [loop_events_ordered short_all].
 destruct (straight_event_program prog), (lf_ok F); reflexivity.
Qed.
Print Assumptions empty_schedule_classification.
"""
    query = tmp_path / "Prerequisites.v"
    query.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 5
    assert "Axioms:" not in result.stdout
