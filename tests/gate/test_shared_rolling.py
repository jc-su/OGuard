"""Shared rolling windows retain loop ranges and every memory boundary."""

import subprocess
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify, zeroed


@pytest.mark.parametrize(
    "change,accepted",
    [
        (None, True),
        ("minimum_tile", True),
        ("short_tile", False),
        ("short_warp", False),
        ("input_end", False),
        ("output_end", False),
        ("tile_store_end", False),
        ("negative_tail", False),
        ("tail_end", False),
    ],
)
@pytest.mark.slow
def test_shared_rolling_checks_each_memory_boundary(change, accepted):
    archive = Path(__file__).with_name("fixtures") / "refusal_audit_2026-09-29r.json"
    text = read_artifact(archive, "fable/rolling_sum/nvcc/0", "ptx")
    if change is not None:
        # Static controls only. Modified kernels are never launched.
        before, after = {
            "minimum_tile": ("_ZZ6kernelE1s[10240]", "_ZZ6kernelE1s[10232]"),
            "short_tile": ("_ZZ6kernelE1s[10240]", "_ZZ6kernelE1s[10224]"),
            "short_warp": ("_ZZ6kernelE8warp_sum[64]", "_ZZ6kernelE8warp_sum[56]"),
            "input_end": (
                "ld.global.nc.f64 \t%fd15, [%rd19];",
                "ld.global.nc.f64 %fd15, [%rd19+8];",
            ),
            "output_end": ("st.global.f64 \t[%rd25], %fd31;", "st.global.f64 [%rd25+8], %fd31;"),
            "tile_store_end": ("st.shared.f64 \t[%r21], %fd15;", "st.shared.f64 [%r21+16], %fd15;"),
            "negative_tail": ("@%p15 bra \t$L__BB5_23;", "@!%p15 bra $L__BB5_23;"),
            "tail_end": ("ld.shared.f64 \t%fd24, [%r44];", "ld.shared.f64 %fd24, [%r44+10240];"),
        }[change]
        assert text.count(before) == 1
        text = text.replace(before, after)
    binding = binding_of("fable", "rolling_sum", "nvcc", text)
    decision = verify(zeroed(text, binding), [binding], timeout=180)
    if accepted:
        assert decision == Accepted(), decision
    else:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision


def test_affine_widening_and_range_cache_keep_checked_bounds(tmp_path):
    source = """
From Coq Require Import ZArith Lists.List Bool.Bool.
From OpGate Require Import GateIsa GateForm GateDomain.
Import ListNotations. Open Scope Z_scope.
Definition facts : lfacts := {|
 lf_widths := PM.add 1%positive 32%nat (PM.add 2%positive 32%nat (PM.empty nat));
 lf_fixed := [1%positive]; lf_bounds := [(1%positive,(0,255))];
 lf_aligns := []; lf_roles := []; lf_shared_window := 10%positive;
 lf_local_window := 11%positive; lf_shared := []; lf_locals := [];
 lf_cells := []; lf_payloads := PM.empty unit; lf_derived := []; lf_names := []
|}.
Definition state (lo hi : Z) : astate := {|
 amap := PM.add 2%positive
   (FForm {|fm:=32;ft:=[((1%positive,EZ),1)];flo:=lo;fhi:=hi|} 0 None)
   (PM.empty fact); acons := []
|}.
Definition joined := join_states facts (state 0 2048) (state 256 2304) true true.
Example nonnegative_cursor : match reg_range facts joined 2%positive with
 | Some (lo,hi) => (0 <=? lo) && (hi <? 2^31) | None => false end = true.
Proof. vm_compute. reflexivity. Qed.
Example entry_checked : state_le facts (state 0 2048) joined = true.
Proof. vm_compute. reflexivity. Qed.
Example back_edge_checked : state_le facts (state 256 2304) joined = true.
Proof. vm_compute. reflexivity. Qed.
Example unsigned_cache : f_rr (rf_with_cache facts (cons_index (acons (state 256 2304))) true 8 (state 256 2304)) 2%positive =
 f_rr (rf_with_cache facts (cons_index (acons (state 256 2304))) false 8 (state 256 2304)) 2%positive.
Proof. vm_compute. reflexivity. Qed.
Example signed_cache : f_sr (rf_with_cache facts (cons_index (acons (state 256 2304))) true 8 (state 256 2304)) 2%positive =
 f_sr (rf_with_cache facts (cons_index (acons (state 256 2304))) false 8 (state 256 2304)) 2%positive.
Proof. vm_compute. reflexivity. Qed.
"""
    path = tmp_path / "SharedRanges.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
