"""Old comparison operands retain their value after later register writes."""

import subprocess
from pathlib import Path

import pytest

from eval.checkpoint import read_artifact
from eval.corpus.adapter import binding_of
from opgate.binding import Base, Binding, Span
from opgate.gate import TIMEOUT_REASON, Accepted, Refused, formal_dir, verify
from opgate.signature import Affine, Direction


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("operand", ["left", "right", "both"])
@pytest.mark.parametrize("upper,accepted", [(8, True), (9, False)])
def test_comparison_uses_values_before_overwrite(width, operand, upper, accepted):
    # Static tests only. The saved address index is checked by a comparison
    # whose operands are subsequently overwritten before its branch.
    left = f"add.u{width} %a, %a, 1;" if operand != "right" else ""
    right = f"add.u{width} %b, %b, 1;" if operand != "left" else ""
    ptx = f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {{
.reg .pred %p;
.reg .b{width} %a, %b;
.reg .b32 %tid;
.reg .b64 %base, %index, %address;
ld.param.u64 %base, [output];
mov.u32 %tid, %tid.x;
cvt.u{width}.u32 %a, %tid;
cvt.u64.u{width} %index, %a;
mov.u{width} %b, {upper};
setp.lt.u{width} %p, %a, %b;
{left}
{right}
@!%p bra END;
add.u64 %address, %base, %index;
st.global.u8 [%address], 1;
END:
ret;
}}
"""
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(8), 8, None, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    result = verify(ptx, [binding], timeout=60)
    if accepted:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result


def test_snapshot_allocation_is_fresh_width_checked_and_origin_preserving(tmp_path):
    source = r"""
From Coq Require Import ZArith Lists.List Strings.String.
From OpGate Require Import GateIsa GateDomain GateFront.
Import ListNotations.
Open Scope Z_scope.
Definition initial : tstate := {|
 ts_ids := PM.empty reg; ts_names := [];
 ts_widths := PM.add 1%positive 32%nat (PM.add 2%positive 32%nat (PM.empty nat));
 ts_next := 4%positive; ts_at := None; ts_fr := [] |}.
Definition body : list placed :=
 [(ISetp None CLt 32 false 3%positive 1%positive (OReg 2%positive), 17%N);
  (IDef None [1%positive;2%positive], 18%N)].
Example fresh_copies :
 match snapshot_block body initial with
 | Ok (p,s) => p = [(IMov None 4%positive 1%positive,17%N);
                    (IMov None 5%positive 2%positive,17%N);
                    (ISetp None CLt 32 false 3%positive 4%positive (OReg 5%positive),17%N);
                    (IDef None [1%positive;2%positive],18%N)] /\
                PM.find 4%positive (ts_widths s) = Some 32%nat /\
                PM.find 5%positive (ts_widths s) = Some 32%nat /\
                ts_next s = 6%positive
 | _ => False end.
Proof. vm_compute. repeat split; reflexivity. Qed.
Example unknown_width_fails :
 match snapshot_operand (PM.add 3%positive tt (PM.empty unit)) 17%N 3%positive initial with
 | Err _ _ => True | _ => False end.
Proof. vm_compute. exact I. Qed.
Example earlier_write_needs_no_copy :
 match snapshot_block (rev body) initial with Ok(p,s) => p = rev body /\ ts_next s = 4%positive | _ => False end.
Proof. vm_compute. split; reflexivity. Qed.
"""
    path = tmp_path / "SnapshotAllocation.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
@pytest.mark.parametrize(
    "access",
    [
        None,
        ("%rd49", "%rd48"),
        ("%rd50", "%rd43"),
        ("%rd76", "%rd75"),
        ("%rd16", "%rd56"),
        ("%rd17", "%rd66"),
    ],
)
def test_current_numba_merge_original_and_each_input_boundary(access):
    archive = Path(__file__).with_name("fixtures") / "registration_final_2026-09-30.json"
    text = read_artifact(archive, "haiku/merge_sorted/numba", "ptx")
    binding = binding_of("haiku", "merge_sorted", "numba", text)
    if access is not None:
        loaded, address = access
        before = f"ld.u64 \t{loaded}, [{address}];"
        assert text.count(before) == 1
        text = text.replace(before, f"ld.u64 {loaded}, [{address}+8];")
    result = verify(text, [binding], timeout=180)
    if access is None:
        assert result == Accepted(), result
    else:
        assert isinstance(result, Refused), result
        assert not result.reason.startswith(TIMEOUT_REASON), result
