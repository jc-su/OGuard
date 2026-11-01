"""Untrusted certificates cannot omit entry, case, edge, text or binding obligations."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import Accepted, Refused, export_query, export_terms, formal_dir, verify
from opgate.signature import Affine, Direction

HEADER = ".version 8.0\n.target sm_80\n.address_size 64\n"
ENTRY = """
.visible .entry kernel(.param .u64 output) {
.reg .pred %p;
.reg .u32 %tid;
.reg .u64 %base, %index, %address;
ld.param.u64 %base, [output];
mov.u32 %tid, %tid.x;
cvt.u64.u32 %index, %tid;
setp.lt.u64 %p, %index, 4;
@!%p bra END;
add.u64 %address, %base, %index;
st.global.u8 [%address], 1;
END:
ret;
}
"""
TEXT = HEADER + ENTRY
BINDING = Binding(
    "kernel",
    (),
    (("output", Base(Span(Direction.WRITE, Affine(4), 4, None, False))),),
    (8, 1, 1),
    (1, 1, 1),
    Affine(0),
)


def coqc(tmp_path: Path, source: str):
    query = tmp_path / "Certificate.v"
    query.write_text(source)
    return subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
    )


PREPARE = """
Definition original_certificate := Eval vm_compute in propose_text_certificate text size mfs.
Definition rewrite_cases (f : case_certificate -> case_certificate) (c : scalar_certificate)
  : scalar_certificate :=
  match c with
  | ScalarBlocks (CertSingle (Some inv)) => ScalarBlocks (CertSingle (Some (f inv)))
  | _ => ScalarMissing
  end.
Definition impossible : astate := {| amap := PM.empty fact; acons := [([],1,0)] |}.
"""


@pytest.mark.parametrize(
    "mutation",
    [
        "[]",
        '[("wrong", snd (hd ("",ScalarMissing) original_certificate))]',
        "List.app original_certificate original_certificate",
        "map (fun nc => (fst nc, ScalarMissing)) original_certificate",
        "map (fun nc => (fst nc, rewrite_cases (fun _ => []) (snd nc))) original_certificate",
        "map (fun nc => (fst nc, rewrite_cases (fun inv => [] :: tl inv) (snd nc))) original_certificate",
        "map (fun nc => (fst nc, rewrite_cases (fun inv => [impossible] :: tl inv) (snd nc))) original_certificate",
        "map (fun nc => (fst nc, rewrite_cases (fun inv => hd [] inv :: [] :: skipn 2 inv) (snd nc))) original_certificate",
        "map (fun nc => (fst nc, rewrite_cases (fun inv => hd [] inv :: [impossible] :: skipn 2 inv) (snd nc))) original_certificate",
        "map (fun nc => (fst nc, match snd nc with ScalarBlocks (CertSingle inv) => ScalarBlocks (CertSplit 1%positive 0 inv None) | _ => ScalarMissing end)) original_certificate",
        "map (fun nc => (fst nc, match snd nc with ScalarBlocks (CertSingle inv) => ScalarBlocks (CertSplit 1%positive 0 None inv) | _ => ScalarMissing end)) original_certificate",
        "map (fun nc => (fst nc, ScalarBlocks (CertSplit 1%positive 0 None None))) original_certificate",
    ],
)
def test_invalid_certificate_cannot_bypass_checker(tmp_path, mutation):
    source = (
        export_terms(TEXT, [BINDING])
        + PREPARE
        + f"""
Example original_accepted : check_text_certificate_ok text size mfs original_certificate = true.
Proof. vm_compute. reflexivity. Qed.
Definition altered : text_certificate := {mutation}.
Example altered_refused : check_text_certificate_ok text size mfs altered = false.
Proof. vm_compute. reflexivity. Qed.
"""
    )
    result = coqc(tmp_path, source)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "change", ["text", "missing-binding", "changed-binding", "extra-binding", "missing-entry"]
)
def test_checker_reconstructs_actual_text_and_binding(tmp_path, change):
    source = export_terms(TEXT, [BINDING]) + PREPARE
    actual_text = TEXT
    actual_bindings = [BINDING]
    if change == "text":
        actual_text = TEXT.replace("[%address], 1", "[%address+4], 1")
    elif change == "missing-binding":
        actual_bindings = []
    elif change == "changed-binding":
        actual_bindings = [
            replace(
                BINDING,
                parameters=(("output", Base(Span(Direction.WRITE, Affine(3), 4, None, False))),),
            )
        ]
    elif change == "extra-binding":
        actual_bindings.append(replace(BINDING, kernel="absent"))
    elif change == "missing-entry":
        actual_text += ENTRY.replace("kernel(", "second(")
        actual_bindings.append(replace(BINDING, kernel="second"))
    changed = export_terms(actual_text, actual_bindings)
    # Separate modules keep all identifiers and actual packed text unambiguous.
    source = "Module Original.\n" + source + "\nEnd Original.\n" + changed
    source += """
Example original_accepted : check_text_certificate_ok Original.text Original.size Original.mfs Original.original_certificate = true.
Proof. vm_compute. reflexivity. Qed.
Example changed_refused : check_text_certificate_ok text size mfs Original.original_certificate = false.
Proof. vm_compute. reflexivity. Qed.
"""
    result = coqc(tmp_path, source)
    assert result.returncode == 0, result.stderr


def test_public_query_checks_proof_and_reports_refused_instruction(tmp_path):
    assert verify(TEXT, [BINDING], timeout=45) == Accepted()
    accepted = coqc(tmp_path, export_query(TEXT, [BINDING]))
    assert accepted.returncode == 0, accepted.stderr
    assert "OPGATE_REFUSAL" not in accepted.stdout
    text = TEXT.replace("[%address], 1", "[%address+4], 1")
    result = verify(text, [BINDING], timeout=45)
    assert isinstance(result, Refused), result
    assert "st.global.u8" in result.reason
    raw = coqc(tmp_path, export_query(text, [BINDING]))
    assert raw.returncode != 0
    assert '"kernel:' in raw.stdout and "OPGATE_REFUSAL" in raw.stdout
    assert "Error:" in raw.stderr


def test_certificate_can_be_checked_without_running_proposal(tmp_path):
    # A ret-only body has a directly authored certificate. The checker never
    # requires this certificate to equal the analysis's proposal.
    text = HEADER + ".visible .entry kernel() { ret; }\n"
    binding = replace(BINDING, parameters=())
    source = (
        export_terms(text, [binding])
        + """
Definition certificate : text_certificate := [("kernel",ScalarBlocks (CertSingle (Some [[{| amap := PM.empty fact; acons := [] |}]])))].
Lemma accepted : check_text_certificate_ok text size mfs certificate = true.
Proof. vm_cast_no_check (eq_refl true). Qed.
Print Assumptions accepted.
"""
    )
    result = coqc(tmp_path, source)
    assert result.returncode == 0, result.stderr


def test_certificate_order_and_complete_module_coverage(tmp_path):
    text = TEXT + ENTRY.replace("kernel(", "second(")
    bindings = [BINDING, replace(BINDING, kernel="second")]
    source = (
        export_terms(text, bindings)
        + PREPARE
        + """
Example both_entries_checked : check_text_certificate_ok text size mfs original_certificate = true.
Proof. vm_compute. reflexivity. Qed.
Example reordered_entries_refused : check_text_certificate_ok text size mfs (rev original_certificate) = false.
Proof. vm_compute. reflexivity. Qed.
"""
    )
    result = coqc(tmp_path, source)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "states, blocks, targets",
    [
        ("[]", "[[node]]", "[[[]]]"),
        ("[empty]", "[[node]]", "[[[]]]"),
        ("[empty; empty; empty]", "[[node]]", "[[[]]]"),
        ("[bottom_state; bottom_state]", "[[node]]", "[[[]]]"),
        ("[empty; bottom_state]", "[[node]]", "[[[]]]"),
        ("[empty; empty]", "[]", "[]"),
        ("[empty; empty]", "[[]]", "[[]]"),
        ("[empty; empty]", "[[node]]", "[]"),
        ("[empty; empty]", "[[node]]", "[[[0%nat]]]"),
    ],
)
def test_cached_certificate_checks_every_node_and_target(tmp_path, states, blocks, targets):
    text = HEADER + ".visible .entry kernel() { ret; }\n"
    binding = replace(BINDING, parameters=())
    source = (
        export_terms(text, [binding])
        + f"""
From OpGate Require Import GateInterval GateCertificate.
Definition empty : astate := {{| amap := PM.empty fact; acons := [] |}}.
Definition good_node : block_cache := {{| cached_states := [empty; empty]; cached_refinements := [] |}}.
Definition good : scalar_certificate := ScalarCached
 {{| cert_written := []; cert_goals := []; cert_descriptor := None;
    cert_blocks := [[good_node]]; cert_targets := [[[]]] |}}.
Example authored_cached_certificate_accepted :
 check_text_certificate_ok text size mfs [("kernel",good)] = true.
Proof. vm_compute; reflexivity. Qed.
Definition node : block_cache := {{| cached_states := {states}; cached_refinements := [] |}}.
Definition bad : scalar_certificate := ScalarCached
 {{| cert_written := []; cert_goals := []; cert_descriptor := None;
    cert_blocks := {blocks}; cert_targets := {targets} |}}.
Example incomplete_cached_certificate_refused :
 check_text_certificate_ok text size mfs [("kernel",bad)] = false.
Proof. vm_compute; reflexivity. Qed.
"""
    )
    result = coqc(tmp_path, source)
    assert result.returncode == 0, result.stderr
