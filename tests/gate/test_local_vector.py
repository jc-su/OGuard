"""Local vector loads retain their complete footprint; all checks are static."""

import json
import subprocess
import time

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import (
    Accepted,
    binding_term,
    export_query,
    export_terms,
    formal_dir,
    verify,
    zeroed,
)
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


def _lower(
    tmp_path, body: str, expected: bool, *, memory: bool = False, load_bits: int | None = None
):
    text = _kernel(body)
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(64), 4, None, False))),),
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
    if load_bits is not None:
        suffix += r"""
Example complete_vector_load_footprint :
 (match result with
 | Ok (p, _) => flat_map (flat_map (fun i =>
     match i with ILdL _ bits _ _ _ _ => [bits] | _ => [] end)) p
 | Err _ _ => []
 end) = [LOAD_BITS%nat].
Proof. vm_compute. reflexivity. Qed.
""".replace("LOAD_BITS", str(load_bits))
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
    "body,load_bits",
    [
        ("st.local.v2.u32 [scratch], {%r1, %r1}; ld.local.v2.u32 {%r2, %r3}, [scratch];", 64),
        (
            "st.local.v4.u32 [scratch], {%r1, %r1, %r1, %r1}; ld.local.v4.u32 {%r1, %r2, %r3, %r4}, [scratch];",
            128,
        ),
        ("st.local.v2.u64 [scratch], {%rd1, %rd1}; ld.local.v2.u64 {%rd1, %rd2}, [scratch];", 128),
    ],
)
def test_complete_local_vector_footprint_lowers(tmp_path, body, load_bits):
    _lower(tmp_path, body, True, memory=True, load_bits=load_bits)
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(64), 4, None, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    assert verify(zeroed(_kernel(body), binding), [binding]) == Accepted()


@pytest.mark.parametrize(
    "body",
    [
        "ld.local.v2.u32 {%r1}, [scratch];",
        "ld.local.v2.u32 {%r1, %r2, %r3}, [scratch];",
        "ld.local.v2.u64 {%r1, %r2}, [scratch];",
        "ld.local.v2.u32 %r1, [scratch];",
    ],
)
def test_malformed_local_vector_results_refuse(tmp_path, body):
    _lower(tmp_path, body, False)


CASES = [
    (
        "guard_true",
        "st.local.v2.u32 [scratch], {%r1, %r1}; setp.eq.u32 %p1, %r1, 1; @%p1 ld.local.v2.u32 {%r2, %r3}, [scratch];",
        64,
        0,
        2,
        1,
        False,
        32,
    ),
    (
        "guard_false",
        "mov.u32 %r2, 2; mov.u32 %r3, 3; st.local.v2.u32 [scratch], {%r1, %r1}; setp.ne.u32 %p1, %r1, 1; @%p1 ld.local.v2.u32 {%r2, %r3}, [scratch];",
        64,
        0,
        2,
        1,
        False,
        32,
    ),
    (
        "negated_guard",
        "st.local.v4.u32 [scratch], {%r1, %r1, %r1, %r1}; setp.ne.u32 %p1, %r1, 1; @!%p1 ld.local.v4.u32 {%r1, %r2, %r3, %r4}, [scratch];",
        128,
        0,
        4,
        2,
        False,
        32,
    ),
    (
        "thread_guard",
        "st.local.v2.u32 [scratch], {%r1, %r1}; mov.u32 %r4, %tid.x; setp.lt.u32 %p1, %r4, 16; @%p1 ld.local.v2.u32 {%r2, %r3}, [scratch];",
        64,
        0,
        2,
        1,
        False,
        32,
    ),
    (
        "address_is_second_result",
        "mov.u64 %rd2, scratch; st.local.v2.u64 [%rd2], {%rd1, %rd1}; ld.local.v2.u64 {%rd1, %rd2}, [%rd2];",
        128,
        0,
        2,
        0,
        True,
        64,
    ),
    (
        "address_is_first_result",
        "mov.u64 %rd2, scratch; st.local.v2.u64 [%rd2], {%rd1, %rd1}; ld.local.v2.u64 {%rd2, %rd1}, [%rd2];",
        128,
        0,
        2,
        0,
        True,
        64,
    ),
    (
        "v2_u32_exact_end",
        "st.local.v2.u32 [scratch+56], {%r1, %r1}; ld.local.v2.u32 {%r2, %r3}, [scratch+56];",
        64,
        56,
        2,
        0,
        False,
        32,
    ),
    (
        "v4_u32_exact_end",
        "st.local.v4.u32 [scratch+48], {%r1, %r1, %r1, %r1}; ld.local.v4.u32 {%r1, %r2, %r3, %r4}, [scratch+48];",
        128,
        48,
        4,
        0,
        False,
        32,
    ),
    (
        "v2_u64_exact_end",
        "st.local.v2.u64 [scratch+48], {%rd1, %rd1}; ld.local.v2.u64 {%rd1, %rd2}, [scratch+48];",
        128,
        48,
        2,
        0,
        False,
        64,
    ),
    (
        "v4_u32_aligned_interior",
        "st.local.v4.u32 [scratch+16], {%r1, %r1, %r1, %r1}; ld.local.v4.u32 {%r1, %r2, %r3, %r4}, [scratch+16];",
        128,
        16,
        4,
        0,
        False,
        32,
    ),
    (
        "v2_u32_aligned_interior",
        "st.local.v2.u32 [scratch+8], {%r1, %r1}; ld.local.v2.u32 {%r2, %r3}, [scratch+8];",
        64,
        8,
        2,
        0,
        False,
        32,
    ),
    (
        "guarded_alias_exact_end",
        "mov.u64 %rd2, scratch; st.local.v2.u64 [%rd2+48], {%rd1, %rd1}; setp.eq.u32 %p1, %r1, 1; @%p1 ld.local.v2.u64 {%rd1, %rd2}, [%rd2+48];",
        128,
        48,
        2,
        1,
        True,
        64,
    ),
]


def _obligations(bits, offset, lanes, tag, aliases, lane_width):
    alignment = 3 if bits == 64 else 4
    suffix = r"""
Open Scope list_scope.
Definition front_result : res (program * lfacts) :=
 match parse_text text size, mfs with
 | Ok ks, [(_, b)] => match ks_entries ks with
   | [(_, k)] => front k b | _ => Err "one entry required" None end
 | _,_ => Err "parse or binding failure" None end.
Definition same_guard (a b : GateIsa.guard) : bool :=
 match a,b with
 | None,None => true
 | Some(na,ra),Some(nb,rb) => Bool.eqb na nb && Pos.eqb ra rb
 | _,_ => false end.
Definition guard_tag (g : GateIsa.guard) : nat :=
 match g with None=>0 | Some(false,_)=>1 | Some(true,_)=>2 end.
Fixpoint vector_pairs (F : lfacts) (body : list instr) :=
 match body with
 | ILdL g bits d a off fr :: ((IDef h ds :: _) as rest) =>
   [(bits, wd F d, off, need (Z.of_nat bits / 8), List.length ds,
     guard_tag g, same_guard g h, negb (existsb (Pos.eqb d) ds),
     negb (Pos.eqb d a), existsb (Pos.eqb a) ds,
     map (wd F) ds, List.length fr)] ++ vector_pairs F rest
 | _ :: rest => vector_pairs F rest
 | [] => [] end.
Fixpoint local_checks (F : lfacts) (st : astate) (body : list instr) :=
 match body with
 | [] => []
 | i :: rest =>
   (match i with ILdL _ _ _ _ _ _ => [(region_of F st i, aligned_ok F st i)] | _=>[] end)
   ++ local_checks F (transfer F st i) rest end.
Example one_block_and_complete_vector :
 (match front_result with
  | Ok(p,F) => (List.length p, flat_map (vector_pairs F) p)
  | _ => (0%nat,[]) end) =
 (1%nat,[(BITS%nat,64%Z,OFFSET%Z,ALIGN%Z,LANES%nat,TAG%nat,
           true,true,true,ALIASES,WIDTHS,0%nat)]).
Proof. vm_compute. reflexivity. Qed.
Example full_local_extent_and_alignment_hold :
 (match front_result with
  | Ok(p,F) => flat_map (local_checks F (empty_state F)) p
  | _ => [] end) = [(true,true)].
Proof. vm_compute. reflexivity. Qed.
"""
    for key, value in {
        "BITS": str(bits),
        "OFFSET": str(offset),
        "ALIGN": str(alignment),
        "LANES": str(lanes),
        "TAG": str(tag),
        "ALIASES": str(aliases).lower(),
        "WIDTHS": "[" + ";".join([f"{lane_width}%Z"] * lanes) + "]",
    }.items():
        suffix = suffix.replace(key, value)
    return suffix


@pytest.mark.parametrize(
    "name,body,bits,offset,lanes,tag,aliases,lane_width", CASES, ids=[c[0] for c in CASES]
)
def test_benign_local_vector_obligations(
    tmp_path, name, body, bits, offset, lanes, tag, aliases, lane_width
):
    library = formal_dir().resolve()
    text = _kernel(body)
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(64), 4, None, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    production = export_query(zeroed(text, binding), [binding])
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    (tmp_path / "ProductionQuery.v").write_text(production)
    query = tmp_path / "Query.v"
    query.write_text(production + _obligations(bits, offset, lanes, tag, aliases, lane_width))
    command = ["coqc", "-q", "-Q", str(library), "OpGate", str(query)]
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "id": name,
                "command": command,
                "returncode": result.returncode,
                "seconds": time.monotonic() - start,
                "library": str(library),
                "bits": bits,
                "offset": offset,
                "lanes": lanes,
                "guard_tag": tag,
                "address_alias": aliases,
                "required_alignment_bytes": bits // 8,
            },
            indent=2,
        )
    )
    assert result.returncode == 0, result.stdout + result.stderr
