"""Static lowering rejects unknown effects; no query in this file runs PTX."""

import subprocess

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import binding_term, export_terms, formal_dir, zeroed
from opgate.signature import Affine, Direction


def _kernel(body: str) -> str:
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u64 output) {{
.reg .pred %p<3>;
.reg .b32 %r<5>;
.reg .b64 %rd<3>;
.reg .f32 %f<4>;
.shared .align 16 .b8 tile[64];
.local .align 16 .b8 scratch[64];
ld.param.u64 %rd1, [output];
mov.u32 %r1, 1;
mov.f32 %f1, 0f3f800000;
{body}
ret;
}}
"""


def _lower(tmp_path, body: str, expected: bool, *, memory: bool = False):
    text = _kernel(body)
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(64), 16, None, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    suffix = r"""
Definition result : res (program * lfacts) :=
 match parse_text text size, mfs with
 | Ok ks, [(_, b)] =>
   match ks_entries ks with
   | [(_, k)] => front k b
   | _ => Err "expected one entry" None
   end
 | _, _ => Err "parse or binding failure" None
 end.
Definition has_memory (i : instr) : bool :=
 match i with
 | IMem _ _ _ _ _ _ _ _ _ | ILdS _ _ _ _ _ | ILdL _ _ _ _ _ _
 | IStL _ _ _ _ _ _ | IShA _ _ _ _ _ _ | ILoad _ _ _ _ _ _ => true
 | _ => false
 end.
Eval vm_compute in match result with Ok _ => "lowered" | Err why _ => why end.
Example expected_boundary :
 (match result with Ok _ => true | Err _ _ => false end) = EXPECTED.
Proof. vm_compute. reflexivity. Qed.
""".replace("EXPECTED", str(expected).lower())
    if memory:
        suffix += r"""
Example retains_memory_obligation :
 (match result with
 | Ok (p, _) => existsb (existsb has_memory) p
 | Err _ _ => false
 end) = true.
Proof. vm_compute. reflexivity. Qed.
"""
    query = tmp_path / "Boundary.v"
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    query.write_text(export_terms(zeroed(text, binding), [binding]) + suffix)
    command = ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "body",
    [
        "unclassified;",
        "unclassified.u32 %r2, %r1;",
        "unclassified.v2.u32 {%r2, %r3}, %r1;",
        "unclassified.f32 %f2, %f1;",
        "add.f32 0, %f1, %f1;",
        "add.f32;",
        "fence.unclassified;",
        "membar.unclassified;",
    ],
)
def test_unknown_or_resultless_instruction_refuses(tmp_path, body):
    _lower(tmp_path, body, False)


@pytest.mark.parametrize(
    "body",
    [
        "add.f32 %f2, %f1, %f1;",
        "fma.rn.f32 %f2, %f1, %f1, %f1;",
        "sqrt.rn.f32 %f2, %f1;",
        "cvt.rzi.s32.f32 %r2, %f1;",
        "popc.b32 %r2, %r1;",
        "brev.b32 %r2, %r1;",
        "activemask.b32 %r2;",
        "add.cc.u32 %r2, %r1, 1; addc.u32 %r3, %r1, 0;",
        "membar.cta;",
        "fence.acq_rel.gpu;",
    ],
)
def test_documented_register_results_and_ordering_lower(tmp_path, body):
    _lower(tmp_path, body, True)


@pytest.mark.parametrize(
    "body",
    [
        "st.global.u32 [%rd1], %r1;",
        "st.shared.u32 [tile], %r1; ld.shared.u32 %r2, [tile];",
        "st.shared.v2.u32 [tile], {%r1, %r1}; ld.shared.v2.u32 {%r2, %r3}, [tile];",
        "atom.shared.add.u32 %r2, [tile], 1;",
        "st.local.u32 [scratch], %r1; ld.local.u32 %r2, [scratch];",
    ],
)
def test_supported_memory_lowering_retains_access_obligation(tmp_path, body):
    _lower(tmp_path, body, True, memory=True)


def test_effect_families_do_not_inherit_register_only_classification(tmp_path):
    # Only mnemonic names are classified here, not executable instructions.
    query = tmp_path / "EffectNames.v"
    query.write_text(r"""
From Coq Require Import List Strings.String.
Require Import OpGate.GateFront.
Import ListNotations. Open Scope string_scope.
Example effect_classes_are_not_pure :
 forallb (fun mn => negb (pure_register_op mn))
 ["unclassified"; "cp"; "tensormap"; "mbarrier"; "wmma";
  "wgmma"; "tcgen05"; "ldmatrix"; "stmatrix"; "shfl";
  "bar"; "barrier"; "vote"; "match"; "redux"; "elect";
  "createpolicy"; "txq"; "suq"; "ld"; "st"; "atom"; "red"; "call"] = true.
Proof. vm_compute. reflexivity. Qed.
""")
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
