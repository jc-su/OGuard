"""Coupled search cursors and integer (not modular) update bounds."""

import subprocess
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify


@pytest.mark.parametrize("commute_add", [False, True])
@pytest.mark.parametrize("past_member_end", [False, True])
@pytest.mark.slow
def test_coupled_binary_search_cursors(commute_add, past_member_end):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29j.json"
    text = read_artifact(archive, "opus/sorted_contains/nvcc/0", "ptx")
    if commute_add:
        before = "add.s64 \t%rd49, %rd36, %rd49;"
        assert text.count(before) == 1
        text = text.replace(before, "add.s64 \t%rd49, %rd49, %rd36;")
    if past_member_end:
        # Static boundary control only; do not launch this modified kernel.
        before = "add.s64 \t%rd34, %rd17, %rd33;"
        assert text.count(before) == 1
        text = text.replace(before, before + "\n\tadd.s64 \t%rd34, %rd34, 8;")
    binding = binding_of("opus", "sorted_contains", "nvcc", text)
    decision = verify(text, [binding], timeout=90)
    if past_member_end:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
    else:
        assert decision == Accepted()


@pytest.mark.parametrize(
    "coefficient,lo,hi", [(1, "2^64-2", "2^64-1"), (-1, "0", "1"), ("2^64", "0", "1")]
)
def test_affine_constraints_require_integer_nonwrapping_updates(tmp_path, coefficient, lo, hi):
    source = f"""
From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain.
Import ListNotations.
Open Scope Z_scope.
Definition facts : lfacts := {{|
 lf_widths := PM.empty nat; lf_fixed := []; lf_bounds := []; lf_aligns := [];
 lf_roles := []; lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit; lf_derived := []; lf_names := []
|}}.
Definition before : astate := {{|
 amap := PM.add 1%positive (FForm {{|fm:=64;ft:=[];flo:={lo};fhi:={hi}|}} 0 (Some ({lo},{hi})))
          (PM.add 3%positive (FForm {{|fm:=64;ft:=[];flo:=1;fhi:=2|}} 0 (Some (1,2)))
            (PM.empty fact));
 acons := [([((1%positive,EX),1);((2%positive,EX),1)],0,2^65)]
|}}.
Definition delta : form := {{|fm:=64;ft:=[((3%positive,EX),{coefficient})];flo:=0;fhi:=0|}}.
Lemma does_not_propagate : affine_step_cons facts before (empty_state facts) 1%positive delta = empty_state facts.
Proof. vm_compute. reflexivity. Qed.
"""
    path = tmp_path / "Wrap.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
