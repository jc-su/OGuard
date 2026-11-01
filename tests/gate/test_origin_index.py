"""Bulk origin lookup preserves exact lists, including duplicates and order."""

import subprocess

from opgate.gate import formal_dir


def test_origin_index_equalities_and_duplicate_lists(tmp_path):
    source = """
From Coq Require Import NArith List.
Require Import OpGate.GateFront.
Import ListNotations.
Print Assumptions origin_nkey_injective.
Print Assumptions origin_index_exact.
Print Assumptions origin_index_keyed.
Print Assumptions origin_hits_indexed_exact.
Print Assumptions placed_events_exact.
Example complete_duplicate_bucket :
 origin_lookup (origin_index
  [(0%N,1%nat);(1%N,2%nat);(0%N,3%nat);(0%N,1%nat)]) 0%N =
 [1%nat;3%nat;1%nat].
Proof. vm_compute. reflexivity. Qed.
Example absent_bucket :
 origin_lookup (origin_index [(0%N,1%nat);(1%N,2%nat)]) 2%N = [].
Proof. vm_compute. reflexivity. Qed.
Example empty_origin_requests : forall K org ids pp,
 placed_events K org ids pp [] = Some [].
Proof. reflexivity. Qed.
Print Assumptions complete_duplicate_bucket.
Print Assumptions absent_bucket.
Print Assumptions empty_origin_requests.
"""
    query = tmp_path / "Origins.v"
    query.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == 8
    assert "Axioms:" not in result.stdout
