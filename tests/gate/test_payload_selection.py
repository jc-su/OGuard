"""Benign static precision controls; every original access remains checked."""

import json
import subprocess

import pytest

from opgate.binding import Base, Binding, ElementType, IndexRange, Span
from opgate.gate import binding_term, export_query, formal_dir, zeroed
from opgate.signature import Affine, Direction

CASES = [
    ("ordinary_payload", "selp.u64 %rd3,1,2,%p1; st.global.u64 [%rd2],%rd3;", False, 1, ""),
    ("indexed_output", "selp.u64 %rd3,1,2,%p1; st.global.u64 [%rd2],%rd3;", True, 0, ""),
    (
        "address_use",
        "selp.u64 %rd3,0,8,%p1; add.u64 %rd4,%rd1,%rd3; st.global.u64 [%rd4],1;",
        False,
        0,
        "",
    ),
    (
        "guard_use",
        "selp.u64 %rd3,0,1,%p1; setp.ne.u64 %p2,%rd3,0; @%p2 st.global.u64 [%rd2],1;",
        False,
        0,
        "",
    ),
    (
        "local_value",
        "selp.u64 %rd3,1,2,%p1; mov.u64 %rd4,slot; st.local.u64 [%rd4],%rd3; ld.local.u64 %rd5,[%rd4]; st.global.u64 [%rd2],%rd5;",
        False,
        0,
        ".local .align 8 .b8 slot[8];",
    ),
    (
        "repeated_definition",
        "selp.u64 %rd3,1,2,%p1; mov.u64 %rd3,3; st.global.u64 [%rd2],%rd3;",
        False,
        0,
        "",
    ),
    (
        "arithmetic_use",
        "selp.u64 %rd3,1,2,%p1; add.u64 %rd4,%rd3,1; st.global.u64 [%rd2],%rd4;",
        False,
        1,
        "",
    ),
    ("unused_result", "selp.u64 %rd3,1,2,%p1; st.global.u64 [%rd2],1;", False, 1, ""),
    (
        "address_payload_alias",
        "selp.u64 %rd3,%rd1,%rd1,%p1; st.global.u64 [%rd3],%rd3;",
        False,
        0,
        "",
    ),
    (
        "guarded_payload",
        "mov.u64 %rd3,0; setp.lt.u32 %p2,%r1,32; @%p2 selp.u64 %rd3,1,2,%p1; st.global.u64 [%rd2],%rd3;",
        False,
        0,
        "",
    ),
    (
        "accumulator_overwrite",
        "selp.u64 %rd3,1,2,%p1; add.u64 %rd4,%rd3,1; add.u64 %rd4,%rd4,1; st.global.u64 [%rd2],%rd4;",
        False,
        1,
        "",
    ),
    (
        "arithmetic_copy",
        "selp.u64 %rd3,1,2,%p1; add.u64 %rd4,%rd3,1; mov.u64 %rd5,%rd4; st.global.u64 [%rd2],%rd5;",
        False,
        1,
        "",
    ),
    (
        "selection_chain",
        "selp.u64 %rd3,1,2,%p1; add.u64 %rd4,%rd3,1; selp.u64 %rd5,%rd4,0,%p1; st.global.u64 [%rd2],%rd5;",
        False,
        2,
        "",
    ),
    (
        "arithmetic_address",
        "selp.u64 %rd3,0,8,%p1; add.u64 %rd4,%rd3,8; add.u64 %rd5,%rd1,%rd4; st.global.u64 [%rd5],1;",
        False,
        0,
        "",
    ),
    (
        "arithmetic_guard",
        "selp.u64 %rd3,0,1,%p1; add.u64 %rd4,%rd3,1; setp.le.u64 %p2,%rd4,2; @%p2 st.global.u64 [%rd2],1;",
        False,
        0,
        "",
    ),
    (
        "arithmetic_local",
        "selp.u64 %rd3,1,2,%p1; add.u64 %rd5,%rd3,1; mov.u64 %rd4,slot; st.local.u64 [%rd4],%rd5; ld.local.u64 %rd5,[%rd4]; st.global.u64 [%rd2],%rd5;",
        False,
        0,
        ".local .align 8 .b8 slot[8];",
    ),
    (
        "arithmetic_indexed",
        "selp.u64 %rd3,1,2,%p1; add.u64 %rd4,%rd3,0; st.global.u64 [%rd2],%rd4;",
        True,
        0,
        "",
    ),
    (
        "overwritten_required_value",
        "selp.u64 %rd3,0,8,%p1; add.u64 %rd4,%rd3,8; add.u64 %rd4,%rd4,8; add.u64 %rd5,%rd1,%rd4; st.global.u64 [%rd5],1;",
        False,
        0,
        "",
    ),
    (
        "shared_payload",
        "selp.u64 %rd3,1,2,%p1; mov.u32 %r1,slot; st.shared.u64 [%r1],%rd3;",
        False,
        1,
        ".shared .align 8 .b8 slot[8];",
    ),
    (
        "division_guard",
        "selp.u64 %rd3,0,1,%p1; div.u64 %rd4,%rd3,1; setp.le.u64 %p2,%rd4,1; @%p2 st.global.u64 [%rd2],1;",
        False,
        0,
        "",
    ),
    (
        "remainder_guard",
        "selp.u64 %rd3,0,1,%p1; rem.u64 %rd4,%rd3,2; setp.le.u64 %p2,%rd4,1; @%p2 st.global.u64 [%rd2],1;",
        False,
        0,
        "",
    ),
]


def fixture(body, indexed, declarations):
    text = f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 output)
.reqntid 32
{{
.reg .b64 %rd<6>;
.reg .b32 %r<2>;
.reg .pred %p<3>;
{declarations}
ld.param.u64 %rd1,[output];
mov.u32 %r1,%tid.x;
setp.lt.u32 %p1,%r1,16;
mad.wide.u32 %rd2,%r1,8,%rd1;
{body}
ret;
}}"""
    ix = IndexRange(ElementType(8, False), Affine(1), Affine(2)) if indexed else None
    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(256), 8, ix, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    return text, binding


def assert_hint_count(text, binding, hints, name, indexed, tmp_path):
    query = export_query(zeroed(text, binding), [binding])
    obligations = f"""
From OpGate Require Import GateDomain GateFront.
Definition selected_hint_count := Eval vm_compute in
 match parse_text text size with Ok ks=>match ks_entries ks,mfs with
 | (_,k)::_,(_,mf)::_=>match front k mf with Ok(_,F)=>PM.cardinal(lf_payloads F)|_=>999%nat end
 | _,_=>999%nat end|_=>999%nat end.
Example expected_payload_precision_hints: selected_hint_count={hints}%nat.
Proof. vm_compute. reflexivity. Qed.
"""
    for filename, data in [
        ("Input.ptx", text),
        ("Binding.txt", binding_term(binding)),
        ("ProductionQuery.v", query),
        ("Query.v", query + obligations),
    ]:
        (tmp_path / filename).write_text(data)
    cmd = [
        "coqc",
        "-q",
        "-Q",
        str(formal_dir()),
        "OpGate",
        str(tmp_path / "Query.v"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    (tmp_path / "stdout").write_text(r.stdout)
    (tmp_path / "stderr").write_text(r.stderr)
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "id": name,
                "command": cmd,
                "returncode": r.returncode,
                "expected_hint_count": hints,
                "indexed_output": indexed,
            },
            indent=2,
        )
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("name,body,indexed,hints,declarations", CASES, ids=[x[0] for x in CASES])
def test_payload_classification_preserves_safety_uses(
    name, body, indexed, hints, declarations, tmp_path
):
    text, binding = fixture(body, indexed, declarations)
    assert_hint_count(text, binding, hints, name, indexed, tmp_path)


READ_INDEX_CASES = [
    (
        "read_index_payload",
        Direction.READ,
        None,
        "selp.u64 %rd6,%rd5,0,%p1; st.global.u64 [%rd4],%rd6;",
        1,
        "",
    ),
    (
        "read_index_arithmetic_payload",
        Direction.READ,
        None,
        "selp.u64 %rd6,%rd5,0,%p1; add.u64 %rd7,%rd6,1; st.global.u64 [%rd4],%rd7;",
        1,
        "",
    ),
    (
        "read_index_address",
        Direction.READ,
        None,
        "selp.u64 %rd6,%rd5,0,%p1; mul.lo.u64 %rd7,%rd6,8; add.u64 %rd8,%rd2,%rd7; st.global.u64 [%rd8],1;",
        0,
        "",
    ),
    (
        "read_index_guard",
        Direction.READ,
        None,
        "selp.u64 %rd6,%rd5,0,%p1; setp.lt.u64 %p2,%rd6,32; @%p2 st.global.u64 [%rd4],1;",
        0,
        "",
    ),
    (
        "read_index_local",
        Direction.READ,
        None,
        "selp.u64 %rd6,%rd5,0,%p1; mov.u64 %rd8,slot; st.local.u64 [%rd8],%rd6; ld.local.u64 %rd7,[%rd8]; st.global.u64 [%rd4],%rd7;",
        0,
        ".local .align 8 .b8 slot[8];",
    ),
    (
        "read_write_index_input",
        Direction.READ_WRITE,
        None,
        "selp.u64 %rd6,%rd5,0,%p1; st.global.u64 [%rd4],%rd6;",
        0,
        "",
    ),
    (
        "read_index_write_index_output",
        Direction.READ,
        Direction.WRITE,
        "selp.u64 %rd6,1,2,%p1; st.global.u64 [%rd4],%rd6;",
        0,
        "",
    ),
    (
        "read_index_read_write_index_output",
        Direction.READ,
        Direction.READ_WRITE,
        "selp.u64 %rd6,1,2,%p1; st.global.u64 [%rd4],%rd6;",
        0,
        "",
    ),
]


@pytest.mark.parametrize(
    "name,input_direction,output_index_direction,body,hints,declarations",
    READ_INDEX_CASES,
    ids=[x[0] for x in READ_INDEX_CASES],
)
def test_read_index_inputs_preserve_required_selection_precision(
    name, input_direction, output_index_direction, body, hints, declarations, tmp_path
):
    text = f""".version 8.0
.target sm_86
.address_size 64
.visible .entry kernel(.param .u64 input, .param .u64 output)
.reqntid 32
{{
.reg .b64 %rd<10>;
.reg .b32 %r<2>;
.reg .pred %p<3>;
{declarations}
ld.param.u64 %rd1,[input];
ld.param.u64 %rd2,[output];
mov.u32 %r1,%tid.x;
setp.lt.u32 %p1,%r1,16;
mad.wide.u32 %rd3,%r1,8,%rd1;
mad.wide.u32 %rd4,%r1,8,%rd2;
ld.global.u64 %rd5,[%rd3];
{body}
ret;
}}"""
    input_ix = IndexRange(ElementType(8, False), Affine(0), Affine(31))
    output_ix = (
        IndexRange(ElementType(8, False), Affine(1), Affine(2))
        if output_index_direction is not None
        else None
    )
    output_direction = (
        output_index_direction if output_index_direction is not None else Direction.WRITE
    )
    binding = Binding(
        "kernel",
        (),
        (
            ("input", Base(Span(input_direction, Affine(256), 8, input_ix, False))),
            ("output", Base(Span(output_direction, Affine(256), 8, output_ix, False))),
        ),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    assert_hint_count(text, binding, hints, name, output_ix is not None, tmp_path)
