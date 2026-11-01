"""Coupled merge counters retain checked bounds at loop entry, updates and exit."""

import subprocess
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify


@pytest.mark.parametrize(
    "reverse_comparisons,past_end",
    [
        (False, None),
        (True, None),
        (False, "loop_a"),
        (False, "loop_b"),
        (False, "final_a"),
        (False, "final_b"),
    ],
)
@pytest.mark.slow
def test_merge_checks_both_inputs_during_and_after_the_loop(reverse_comparisons, past_end):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29n.json"
    text = read_artifact(archive, "haiku/merge_sorted/nvcc/0", "ptx")
    if reverse_comparisons:
        for predicate, old, new in (("%p8", "le", "ge"), ("%p9", "gt", "lt")):
            before = f"setp.{old}.s64 \t{predicate}, %rd36, %rd35;"
            assert text.count(before) == 1
            text = text.replace(before, f"setp.{new}.s64 {predicate}, %rd35, %rd36;")
    if past_end is not None:
        # Static boundary controls; never execute the modified PTX.
        loaded, address = {
            "loop_a": ("%rd36", "%rd32"),
            "loop_b": ("%rd35", "%rd34"),
            "final_a": ("%rd18", "%rd43"),
            "final_b": ("%rd44", "%rd17"),
        }[past_end]
        before = f"ld.global.u64 \t{loaded}, [{address}];"
        assert text.count(before) == 1
        text = text.replace(before, f"ld.global.u64 {loaded}, [{address}+8];")
    decision = verify(text, [binding_of("haiku", "merge_sorted", "nvcc", text)], timeout=90)
    if past_end is None:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision


def _check(tmp_path, source):
    path = tmp_path / "Merge.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


FACTS = """
From Coq Require Import ZArith Lists.List.
From OpGate Require Import GateIsa GateForm GateDomain.
Import ListNotations.
Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 4%positive 1%nat (PM.empty nat);
 lf_fixed := []; lf_bounds := []; lf_aligns := []; lf_roles := [];
 lf_shared_window := 10%positive; lf_local_window := 11%positive;
 lf_shared := []; lf_locals := []; lf_cells := []; lf_payloads := PM.empty unit; lf_derived := []; lf_names := []
|}.
Definition interval (lo hi : Z) : fact := FForm (range_form 64 lo hi) 0 None.
"""


@pytest.mark.parametrize("upper", [9, 10])
def test_register_bounds_require_the_whole_counter_sum(tmp_path, upper):
    # Natural x,y>=0 and x+y=z<=10 prove x<=10, but permit x=10,y=0.
    _check(
        tmp_path,
        FACTS
        + f"""
Definition before : astate := {{|
 amap := PM.add 3%positive (interval 0 10) (PM.empty fact);
 acons := [(sorted_terms [((1%positive,EX),1);((2%positive,EX),1);((3%positive,EX),-1)],0,0)]
|}}.
Example bound : register_fact_le facts before 1%positive (interval 0 {upper}) = {str(upper == 10).lower()}.
Proof. vm_compute. reflexivity. Qed.
""",
    )


@pytest.mark.parametrize("comparison", ["CGe", "CGt"])
def test_equivalent_predicates_keep_the_equality_boundary(tmp_path, comparison):
    # With i=0 and m>=0, m<1 is equivalent to i>=m; i>m is not equivalent.
    _check(
        tmp_path,
        FACTS
        + f"""
Definition initial : cmpf := {{|cc:=CLt;cw:=64;csg:=true;ca:=2%positive;cb:=OImm 1|}}.
Definition target : cmpf := {{|cc:={comparison};cw:=64;csg:=true;ca:=1%positive;cb:=OReg 2%positive|}}.
Definition before : astate := {{|
 amap := PM.add 1%positive (interval 0 0)
   (PM.add 2%positive (interval 0 100)
   (PM.add 4%positive (FPred [initial] true) (PM.empty fact)));
 acons := []
|}}.
Example predicate : register_fact_le facts before 4%positive (FPred [target] true) = {str(comparison == "CGe").lower()}.
Proof. vm_compute. reflexivity. Qed.
""",
    )
