"""Exact source-origin enumeration, using pure IR metadata only."""

import subprocess

from opgate.gate import formal_dir


def compile_metadata(tmp_path, body):
    source = tmp_path / "Query.v"
    source.write_text(
        "From Coq Require Import List ZArith.\n"
        "Require Import OpGate.GateIsa OpGate.GateFront.\n"
        "Import ListNotations. Local Open Scope Z_scope.\n" + body
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


def test_exact_origin_order_and_positions_across_empty_blocks(tmp_path):
    compile_metadata(
        tmp_path,
        """
Definition pp : list (list placed) :=
 [[];[(INop,7%N);(IDef None [1%positive],9%N);
       (IImm None 2%positive 3,7%N)];[];[(INop,7%N)]].
Example all_original_positions : origin_hits pp 7%N =
 [((1%nat,0%nat),INop);((1%nat,2%nat),IImm None 2%positive 3);
  ((3%nat,0%nat),INop)].
Proof. vm_compute. reflexivity. Qed.
Example separate_origin : origin_hits pp 9%N =
 [((1%nat,1%nat),IDef None [1%positive])].
Proof. vm_compute. reflexivity. Qed.
Example absent_origin : origin_hits pp 8%N = [].
Proof. vm_compute. reflexivity. Qed.
Example empty_program : origin_hits [] 7%N = [].
Proof. vm_compute. reflexivity. Qed.
Example only_empty_blocks : origin_hits [[];[]] 7%N = [].
Proof. vm_compute. reflexivity. Qed.
Example duplicated_origins_still_refuse : placed_event EBarrier pp 7%N = None.
Proof. vm_compute. reflexivity. Qed.
""",
    )


def test_origin_enumeration_exact_theorems_are_closed(tmp_path):
    output = compile_metadata(
        tmp_path,
        "Print Assumptions origin_enumeration_exact.\nPrint Assumptions origin_hits_exact.\n",
    )
    assert output.count("Closed under the global context") == 2
