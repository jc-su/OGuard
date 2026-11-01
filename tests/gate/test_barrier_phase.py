"""Pure checker controls and theorem assumptions; no PTX execution."""

import subprocess

from opgate.gate import formal_dir


def test_progress_and_no_return_theorems_have_no_axioms(tmp_path):
    query = tmp_path / "Assumptions.v"
    query.write_text("""
Require Import OpGate.GateBarrier OpGate.GateFront.
Print Assumptions phase_trim_sound.
Print Assumptions phase_successors_complete.
Print Assumptions phase_check_progress.
Print Assumptions phase_never_returns.
Print Assumptions single_phase_check_sound.
Print Assumptions accepted_front_phase.
""")
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 6, result.stdout


def test_phase_metadata_requires_one_exact_event(tmp_path):
    query = tmp_path / "Metadata.v"
    query.write_text("""
From Coq Require Import List NArith.
Require Import OpGate.GateIsa OpGate.GateBarrier OpGate.GateFront.
Import ListNotations.
Example singleton_event : placed_event EBarrier [[(INop,7%N)]] 7%N = Some (0%nat,0%nat).
Proof. vm_compute. reflexivity. Qed.
Example missing_event : placed_event EBarrier [[(INop,7%N)]] 8%N = None.
Proof. vm_compute. reflexivity. Qed.
Example ambiguous_event : placed_event EBarrier [[(INop,7%N);(INop,7%N)]] 7%N = None.
Proof. vm_compute. reflexivity. Qed.
Example post_event_fallthrough : phase_closed [[INop;IExit None]] (0%nat,0%nat) = true.
Proof. vm_compute. reflexivity. Qed.
Example event_position_is_exact : phase_event (0%nat,1%nat) (0%nat,0%nat) = false.
Proof. vm_compute. reflexivity. Qed.
""")
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
