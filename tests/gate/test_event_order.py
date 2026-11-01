"""Exact adjacent event ordering, using pure program-counter metadata."""

import subprocess

from opgate.gate import formal_dir


def compile_metadata(tmp_path, body):
    source = tmp_path / "Query.v"
    source.write_text(
        "From Coq Require Import List Arith.\n"
        "Require Import OpGate.GateIsa OpGate.GateBarrier OpGate.GateLoop.\n"
        "Import ListNotations.\n" + body
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
    return result.stdout


def test_exact_strict_order_preserves_empty_cross_block_and_duplicate_cases(tmp_path):
    compile_metadata(
        tmp_path,
        """
Example empty_order : loop_events_adjacent [] = true.
Proof. vm_compute. reflexivity. Qed.
Example singleton_order : loop_events_adjacent [(7,500)] = true.
Proof. vm_compute. reflexivity. Qed.
Example increasing_within_block :
 loop_events_adjacent [(0,1);(0,7);(0,500)] = true.
Proof. vm_compute. reflexivity. Qed.
Example increasing_across_blocks :
 loop_events_adjacent [(0,500);(1,0);(2,1)] = true.
Proof. vm_compute. reflexivity. Qed.
Example same_position : loop_events_adjacent [(0,1);(0,1)] = false.
Proof. vm_compute. reflexivity. Qed.
Example reversed_block : loop_events_adjacent [(1,0);(0,500)] = false.
Proof. vm_compute. reflexivity. Qed.
Example reversed_index : loop_events_adjacent [(0,500);(0,1)] = false.
Proof. vm_compute. reflexivity. Qed.
Example later_duplicate :
 loop_events_adjacent [(0,1);(0,2);(0,1)] = false.
Proof. vm_compute. reflexivity. Qed.
Example late_reversal :
 loop_events_adjacent [(0,1);(1,2);(1,1)] = false.
Proof. vm_compute. reflexivity. Qed.
""",
    )


def test_adjacent_event_order_equivalence_is_closed(tmp_path):
    output = compile_metadata(
        tmp_path,
        "Print Assumptions loop_phase_after_transitive.\n"
        "Print Assumptions loop_events_after_exact.\n"
        "Print Assumptions loop_events_adjacent_exact.\n",
    )
    assert output.count("Closed under the global context") == 3
