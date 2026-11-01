"""Static literal/effect checks. These PTX fragments never execute on a GPU."""

import subprocess
from decimal import Decimal, localcontext

import pytest

from opgate.binding import Base, Binding, ElementType, IndexRange, Span
from opgate.gate import (
    Accepted,
    Refused,
    binding_term,
    export_query,
    export_terms,
    formal_dir,
    verify,
)
from opgate.signature import Affine, Direction

HEADER = ".version 8.0\n.target sm_80\n.address_size 64\n"


def exact_decimal(n, k):
    with localcontext() as ctx:
        ctx.prec = 220
        return format(Decimal(n) / (Decimal(2) ** k), "f")


FRACTIONS = [
    ("1.0", True),
    ("0.5", True),
    ("0.25", True),
    ("0.75", True),
    ("+000.5000e+000", True),
    ("1.000e+0", True),
    ("100000000000e-11", True),
    ("5E-1", True),
    (exact_decimal(1, 149), True),
    (exact_decimal(1, 126), True),
    (exact_decimal(2**24 - 1, 24), True),
    (exact_decimal(2**24 - 1, 149), True),
    ("0f00000001", True),
    ("0f007fffff", True),
    ("0f00800000", True),
    ("0f3f800000", True),
    ("0F3F000000", True),
    ("0.0", False),
    ("-0.5", False),
    ("1.25", False),
    ("0.1", False),
    (exact_decimal(1, 150), False),
    (exact_decimal(2**25 - 1, 25), False),
    (exact_decimal(2**23 + 1, 23), False),
    ("0f00000000", False),
    ("0f80000000", False),
    ("0fbf000000", False),
    ("0f3f800001", False),
    ("0f7f800000", False),
    ("0f7fc00000", False),
    ("0f3f80000", False),
    ("0f03f800000", False),
    ("0f3f80000g", False),
    ("0d3ff0000000000000", False),
    ("1e-1000000000000000", False),
    ("1e1000000000000000", False),
    ("1e", False),
    ("1..0", False),
    ("nan", False),
    ("inf", False),
    ("", False),
]


def binding(indexed=False):
    ix = IndexRange(ElementType(8, False), Affine(0), Affine(10)) if indexed else None
    return Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(64), 4, ix, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )


def kernel(body, header=HEADER):
    return (
        header
        + """.visible .entry kernel(.param .u64 output) {
.reg .pred %p<4>;
.reg .b32 %r<4>;
.reg .b64 %rd<5>;
.reg .f32 %f<4>;
ld.param.u64 %rd1,[output];
mov.u32 %r1,1;
setp.eq.u32 %p1,%r1,1;
"""
        + body
        + "\nret;\n}\n"
    )


def compile_query(tmp_path, query, text=None):
    p = tmp_path / "Check.v"
    p.write_text(query)
    if text is not None:
        (tmp_path / "Input.ptx").write_text(text)
    r = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(p)],
        capture_output=True,
        text=True,
        timeout=40,
    )
    (tmp_path / "stdout.txt").write_text(r.stdout)
    (tmp_path / "stderr.txt").write_text(r.stderr)
    assert r.returncode == 0, r.stdout + r.stderr


def lower(tmp_path, body, expected, header=HEADER):
    text = kernel(body, header)
    b = binding()
    (tmp_path / "Binding.txt").write_text(binding_term(b))
    suffix = r"""
Definition lowered := match parse_text text size,mfs with
| Ok ks,[(_,b)] => match ks_entries ks with [(_,k)] => front k b | _=>Err "entry" None end
| _,_=>Err "parse" None end.
Eval vm_compute in match lowered with Ok _=>"lowered" | Err why _=>why end.
Example checked_shape : (match lowered with Ok _=>true|_=>false end) = EXPECTED.
Proof. vm_compute. reflexivity. Qed.
""".replace("EXPECTED", str(expected).lower())
    compile_query(tmp_path, export_terms(text, [b]) + suffix, text)


@pytest.mark.parametrize("literal,expected", FRACTIONS)
def test_exact_fraction_recognition(tmp_path, literal, expected):
    # Raw string syntax has no escapes in these numeric cases.
    suffix = f'\nExample exact_fraction : fraction_literal "{literal}" = {str(expected).lower()}.\nProof. vm_compute. reflexivity. Qed.\n'
    compile_query(tmp_path, export_terms(kernel(""), [binding()]) + suffix)


@pytest.mark.parametrize("literal,expected", FRACTIONS[:30])
def test_literal_policy_lowering(tmp_path, literal, expected):
    lower(tmp_path, f"createpolicy.fractional.L2::evict_last.b64 %rd2,{literal};", expected)


@pytest.mark.parametrize(
    "primary", ["evict_last", "evict_normal", "evict_first", "evict_unchanged"]
)
@pytest.mark.parametrize("secondary", ["", ".L2::evict_first", ".L2::evict_unchanged"])
def test_valid_priority_combinations_and_default_fraction(tmp_path, primary, secondary):
    lower(tmp_path, f"createpolicy.fractional.L2::{primary}{secondary}.b64 %rd2;", True)


@pytest.mark.parametrize(
    "body",
    [
        "createpolicy.fractional.L2::evict_last.b32 %r2,1.0;",
        "createpolicy.fractional.L2::evict_last.b64 %r2,1.0;",
        "createpolicy.fractional.L2::evict_last.b64 %missing,1.0;",
        "createpolicy.fractional.L2::evict_last.b64 {%rd2,%rd3},1.0;",
        "createpolicy.fractional.L2::evict_last.b64 %rd2,1.0,0.5;",
        "createpolicy.fractional.L2::evict_last.b64 %rd2,%f1;",
        "createpolicy.fractional.L2::evict_last.b64 %rd2,1;",
        "createpolicy.fractional.L1::evict_last.b64 %rd2,1.0;",
        "createpolicy.fractional.L2::evict_last.L2::evict_normal.b64 %rd2,1.0;",
        "createpolicy.fractional.L2::evict_last.L2::evict_last.b64 %rd2,1.0;",
        "createpolicy.fractional.global.L2::evict_last.b64 %rd2,1.0;",
        "createpolicy.range.L2::evict_last.b64 %rd2,[%rd1],16,32;",
        "createpolicy.cvt.L2.b64 %rd2,%rd3;",
        "@%r1 createpolicy.fractional.L2::evict_last.b64 %rd2,1.0;",
    ],
)
def test_unsupported_shapes_fail_closed(tmp_path, body):
    lower(tmp_path, body, False)


@pytest.mark.parametrize(
    "header,expected",
    [
        (HEADER, True),
        (HEADER.replace("8.0", "7.4"), True),
        (HEADER.replace("8.0", "9.4"), True),
        (HEADER.replace("8.0", "7.3"), False),
        (HEADER.replace("8.0", "9.5"), False),
        (HEADER.replace("sm_80", "sm_75"), False),
        (HEADER.replace("sm_80", "sm_90a"), True),
        (".address_size 64\n", False),
    ],
)
def test_reviewed_target_and_version(tmp_path, header, expected):
    lower(tmp_path, "createpolicy.fractional.L2::evict_last.b64 %rd2,1.0;", expected, header)


@pytest.mark.parametrize("guard", ["@%p1", "@!%p1"])
def test_policy_result_guard_is_preserved(tmp_path, guard):
    text = kernel(f"{guard} createpolicy.fractional.L2::evict_last.b64 %rd2,1.0;")
    suffix = r"""
Definition guarded_result := match parse_text text size,mfs with
| Ok ks,[(_,b)]=>match ks_entries ks with [(_,k)]=>front k b|_=>Err "entry" None end
| _,_=>Err "parse" None end.
Example exactly_one_guarded_opaque_result :
(match guarded_result with Ok(p,_)=>
 List.length(filter(fun i=>match i with IDef(Some _)[_]=>true|_=>false end)(List.concat p))
|_=>0%nat end)=1%nat.
Proof. vm_compute. reflexivity. Qed.
"""
    compile_query(tmp_path, export_terms(text, [binding()]) + suffix, text)


@pytest.mark.parametrize("indexed", [False, True])
def test_complete_checker_keeps_indexed_output_obligation(tmp_path, indexed):
    text = kernel("createpolicy.fractional.L2::evict_last.b64 %rd2,1.0; st.global.u64 [%rd1],%rd2;")
    declared = binding(indexed)
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(declared))
    (tmp_path / "FullCheck.v").write_text(export_query(text, [declared]))
    result = verify(text, [declared], timeout=30)
    (tmp_path / "Result.txt").write_text(repr(result) + "\n")
    assert isinstance(result, Refused) if indexed else result == Accepted()


def test_ordinary_float_spelling_survives_scope_and_rendering(tmp_path):
    text = kernel("{ .reg .f32 %v; mov.f32 %v,0.1; add.f32 %v,%v,0F3F000000; }")
    suffix = r"""
Definition floats := match parse_text text size with Ok ks=>
 match ks_entries ks with [(_,k)]=>
 flat_map(fun i=>match i with PI _ _ _ os=>flat_map(fun o=>match o with PFloat s=>[s]|_=>[]end)os end)(k_insts k)
 | _=>[]end|_=>[]end.
Example exact_spelling : floats=["0.1";"0F3F000000"].
Proof. vm_compute. reflexivity. Qed.
Example rendered_spelling : show_opnd(PFloat "0.1")="0.1".
Proof. reflexivity. Qed.
"""
    compile_query(tmp_path, export_terms(text, [binding()]) + suffix, text)
    lower(tmp_path, "{ .reg .f32 %v; mov.f32 %v,0.1; add.f32 %v,%v,0F3F000000; }", True)


def test_float_arguments_survive_inlining(tmp_path):
    function = """ .func use_float(.param .f32 arg) {
.reg .f32 %v;
ld.param.f32 %v,[arg];
add.f32 %v,%v,0.25;
ret;
}
"""
    text = kernel("""{ .param .f32 param0;
st.param.f32 [param0],0F3F000000;
call.uni use_float,(param0);
}""").replace(".visible .entry", function + ".visible .entry", 1)
    suffix = r"""
Definition inlined_floats := match parse_text text size with Ok ks=>
 match ks_entries ks with [(_,k)]=>
 match inline k with Ok (k',_)=>
 flat_map(fun i=>match i with PI _ _ _ os=>flat_map(fun o=>match o with PFloat s=>[s]|_=>[]end)os end)(k_insts k')
 | _=>[]end|_=>[]end|_=>[]end.
Eval vm_compute in inlined_floats.
Example both_values_preserved :
 andb (existsb(String.eqb "0F3F000000")inlined_floats) (existsb(String.eqb "0.25")inlined_floats)=true.
Proof. vm_compute. reflexivity. Qed.
"""
    compile_query(tmp_path, export_terms(text, [binding()]) + suffix, text)
    (tmp_path / "FullCheck.v").write_text(export_query(text, [binding()]))
    result = verify(text, [binding()], timeout=30)
    (tmp_path / "Result.txt").write_text(repr(result) + "\n")
    assert result == Accepted()


def test_exact_division_lemma_has_no_new_assumptions(tmp_path):
    compile_query(
        tmp_path,
        export_terms(kernel(""), [binding()]) + "\nPrint Assumptions divide_fives_exact.\n",
    )
    output = (tmp_path / "stdout.txt").read_text()
    assert "Closed under the global context" in output
