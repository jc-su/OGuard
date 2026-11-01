"""Predicate negation and loop limits preserve checked merge input bounds."""

import subprocess
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, Convention, ElementType, derive_binding
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify
from opgate.signature import Affine, Direction


@pytest.mark.parametrize("past_end", [None, "loop_a", "loop_b", "exit_a", "exit_b", "compare_b"])
@pytest.mark.slow
def test_numba_merge_checks_each_input_access(past_end):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29p.json"
    text = read_artifact(archive, "haiku/merge_sorted/numba/0", "ptx")
    if past_end is not None:
        # Static controls only: the modified modules are never executed.
        loaded, address = {
            "loop_a": ("%rd51", "%rd44"),
            "loop_b": ("%rd50", "%rd49"),
            "exit_a": ("%rd16", "%rd59"),
            "exit_b": ("%rd79", "%rd78"),
            "compare_b": ("%rd17", "%rd69"),
        }[past_end]
        before = f"ld.u64 \t{loaded}, [{address}];"
        assert text.count(before) == 1
        text = text.replace(before, f"ld.u64 {loaded}, [{address}+8];")
    decision = verify(text, [binding_of("haiku", "merge_sorted", "numba", text)], timeout=180)
    if past_end is None:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision


SIGNATURE = sig.Signature(
    sizes=(sig.SizeSymbol("n", 8),),
    arguments=(
        sig.Span("out", "out", Direction.WRITE, Affine.symbol(0)),
        sig.Scalar("n", sig.SizeOf(Affine.symbol(0))),
    ),
)
BINDING = derive_binding(
    SIGNATURE,
    {"out": ElementType(4, True)},
    Convention.C,
    "kernel",
    ("kernel_param_0", "kernel_param_1"),
    (32, 1, 1),
    GRID_LIMIT,
    Affine(0),
)


def guarded_kernel(upper=7, identity="and"):
    bit = 1 if identity == "and" else 0
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 kernel_param_0, .param .u64 kernel_param_1) {{
 .reg .pred %p<3>;
 .reg .b32 %r<2>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [kernel_param_0];
 mov.u32 %r1, %tid.x;
 setp.gt.u32 %p1, %r1, {upper};
 mov.pred %p2, {bit};
 {identity}.pred %p1, %p1, %p2;
 not.pred %p2, %p1;
 @!%p2 bra END;
 mul.wide.u32 %rd2, %r1, 4;
 add.u64 %rd3, %rd1, %rd2;
 st.global.u32 [%rd3], %r1;
END:
 ret;
}}
"""


@pytest.mark.parametrize("upper", [7, 8])
@pytest.mark.parametrize("identity", ["and", "or"])
def test_negated_boolean_guard_preserves_the_array_boundary(upper, identity):
    decision = verify(guarded_kernel(upper, identity), [BINDING], timeout=60)
    if upper == 7:
        assert decision == Accepted()
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision


def test_unknown_predicate_cannot_be_claimed_constant(tmp_path):
    source = """
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
Definition cmp : cmpf := {|cc:=CGe;cw:=64;csg:=false;ca:=1%positive;cb:=OImm 1|}.
Definition before (lo hi : Z) : astate := {|
 amap := PM.add 1%positive (FForm (range_form 64 lo hi) 0 None)
   (PM.add 4%positive (FPred [cmp] true) (PM.empty fact));
 acons := []
|}.
Definition bit_range (lo hi : Z) := FForm (range_form 1 lo hi) 0 None.
Example zero : register_fact_le facts (before 0 0) 4%positive (bit_range 0 0) = true.
Proof. vm_compute. reflexivity. Qed.
Example one : register_fact_le facts (before 1 3) 4%positive (bit_range 1 1) = true.
Proof. vm_compute. reflexivity. Qed.
Example unknown_zero : register_fact_le facts (before 0 3) 4%positive (bit_range 0 0) = false.
Proof. vm_compute. reflexivity. Qed.
Example unknown_one : register_fact_le facts (before 0 3) 4%positive (bit_range 1 1) = false.
Proof. vm_compute. reflexivity. Qed.
Example unknown_bit : register_fact_le facts (before 0 3) 4%positive (bit_range 0 1) = true.
Proof. vm_compute. reflexivity. Qed.
"""
    path = tmp_path / "PredicateBounds.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
