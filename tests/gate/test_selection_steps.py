"""Conditional cursor updates retain checked relations across loop steps."""

import subprocess

import pytest

from eval.corpus.adapter import binding_of, directory_of
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify


@pytest.mark.parametrize("reverse_selection", [False, True])
@pytest.mark.parametrize("past_member_end", [False, True])
def test_triton_search_keeps_both_selection_orders(reverse_selection, past_member_end):
    text = (directory_of("opus", "sorted_contains") / "triton.ptx").read_bytes().decode("ascii")
    if reverse_selection:
        before = "selp.b64 \t%rd22, %rd22, %rd14, %p4;"
        assert text.count(before) == 1
        text = text.replace(before, "not.pred %p4, %p4;\n\tselp.b64 %rd22, %rd14, %rd22, %p4;")
    if past_member_end:
        # Static boundary control; never execute this modified PTX.
        before = "ld.global.b64 { %rd16 }, [ %rd17 + 0 ];"
        assert text.count(before) == 1
        text = text.replace(before, "ld.global.b64 { %rd16 }, [ %rd17 + 8 ];")
    binding = binding_of("opus", "sorted_contains", "triton", text)
    decision = verify(text, [binding], timeout=90)
    if past_member_end:
        assert isinstance(decision, Refused), decision
        assert not decision.reason.startswith(TIMEOUT_REASON), decision
    else:
        assert decision == Accepted()


@pytest.mark.parametrize(
    "span,size,base,half,mid", [(1, 2, 3, 4, 5), (5, 4, 3, 2, 1), (2, 4, 1, 5, 3)]
)
@pytest.mark.parametrize("cancel_half", [False, True])
def test_projection_includes_target_equations_and_their_dependencies(
    tmp_path, span, size, base, half, mid, cancel_half
):
    # span+base<=size and mid=base+floor(span/2) imply
    # span+mid-size-half<=0, but do not imply span+mid-size<=0.
    extra = f";(({half}%positive,EX),-1)" if cancel_half else ""
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
Definition interval (lo hi : Z) : fact := FForm {{|fm:=64;ft:=[];flo:=lo;fhi:=hi|}} 0 None.
Definition before : astate := {{|
 amap := PM.add {span}%positive (interval 1 100)
   (PM.add {size}%positive (interval 1 200)
   (PM.add {base}%positive (interval 0 100)
   (PM.add {half}%positive (FForm {{|fm:=64;ft:=[(({span}%positive,EShr 1),1)];flo:=0;fhi:=0|}} 0 None)
   (PM.add {mid}%positive (FForm {{|fm:=64;ft:=sorted_terms [(({span}%positive,EShr 1),1);(({base}%positive,EX),1)];flo:=0;fhi:=0|}} 0 None)
    (PM.empty fact)))));
 acons := [(sorted_terms [(({span}%positive,EX),1);(({base}%positive,EX),1);(({size}%positive,EX),-1)],-2^64,0)]
|}}.
Definition target := form_of_terms (sorted_terms [(({span}%positive,EX),1);(({mid}%positive,EX),1);(({size}%positive,EX),-1){extra}]).
Example expected_bound :
 (match check_range facts before target with Some (_,hi) => hi <=? 0 | None => false end) = {str(cancel_half).lower()}.
Proof. vm_compute. reflexivity. Qed.
"""
    path = tmp_path / "Target.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
