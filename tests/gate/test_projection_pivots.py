"""Projected cursor bounds survive register ordering and a search's exit join."""

import itertools
import subprocess
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify


@pytest.mark.parametrize("commute_add", [False, True])
@pytest.mark.parametrize("past_member_end", [False, True])
@pytest.mark.slow
def test_numba_search_keeps_the_sum_at_its_exit(commute_add, past_member_end):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29k.json"
    text = read_artifact(archive, "opus/sorted_contains/numba/0", "ptx")
    if commute_add:
        before = "add.s64 \t%rd45, %rd34, %rd45;"
        assert text.count(before) == 1
        text = text.replace(before, "add.s64 \t%rd45, %rd45, %rd34;")
    if past_member_end:
        # Static negative control; never execute this modified PTX.
        before = "ld.u64 \t%rd37, [%rd36];"
        assert text.count(before) == 1
        text = text.replace(before, "ld.u64 \t%rd37, [%rd36+8];")
    binding = binding_of("opus", "sorted_contains", "numba", text)
    decision = verify(text, [binding], timeout=90)
    if past_member_end:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
    else:
        assert decision == Accepted()


@pytest.mark.parametrize("x,y,z", list(itertools.permutations((1, 2, 3))))
def test_projection_finds_the_residual_bound_in_any_register_order(tmp_path, x, y, z):
    # 2*x+y<=z and x>=1 imply z-y-x>=1. Choosing x's ratio alone misses
    # this bound; choosing y's or z's leaves the useful residual x.
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
Definition ranges (a : atom) : option range :=
 if Pos.eqb (fst a) {x}%positive then Some (1,10) else Some (0,100).
Definition target : form := {{|fm:=64;
 ft:=sorted_terms [(({x}%positive,EX),-1);(({y}%positive,EX),-1);(({z}%positive,EX),1)];
 flo:=0;fhi:=0|}}.
Definition relation : constraint :=
 (sorted_terms [(({x}%positive,EX),2);(({y}%positive,EX),1);(({z}%positive,EX),-1)],-2^64,0).
Example lower_bound :
 (match projected_cons_range facts ranges target relation with Some (lo,_) => 1 <=? lo | None => false end) = true.
Proof. vm_compute. reflexivity. Qed.
"""
    path = tmp_path / "Pivot.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
