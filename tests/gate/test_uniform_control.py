"""Static uniform-control boundary checks; benign inputs never execute."""

import subprocess

import pytest

from opgate.binding import Binding
from opgate.gate import CHECK_QUERY, export_terms, formal_dir
from opgate.signature import Affine

HEADER = ".version 8.0\n.target sm_80\n.address_size 64\n"
DECLS = ".reg .pred %p; .reg .u32 %r;\nmov.pred %p, 1; mov.u32 %r, 0;\n"
BINDING = Binding("kernel", (), (), (32, 1, 1), (1, 1, 1), Affine(0))


def _run(tmp_path, query, name="Query"):
    source = tmp_path / f"{name}.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / f"{name}.stdout.txt").write_text(result.stdout)
    (tmp_path / f"{name}.stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mn,suffix,pred,expected",
    [
        ("bra", '["uni"]', "None", True),
        ("call", '["uni"]', "None", True),
        ("bra", '["uni"]', 'Some (false,"%p")', False),
        ("call", '["uni"]', 'Some (true,"%p")', False),
        ("brx", '["idx";"uni"]', "None", False),
        ("brx", '["idx";"uni"]', 'Some (false,"%p")', False),
        ("ret", '["uni"]', "None", False),
        ("ret", '["uni"]', 'Some (false,"%p")', False),
        ("bra", "[]", 'Some (false,"%p")', True),
        ("call", "[]", 'Some (false,"%p")', True),
        ("brx", '["idx"]', 'Some (false,"%p")', True),
        ("ret", "[]", 'Some (false,"%p")', True),
        ("vote", '["sync";"uni";"pred"]', 'Some (false,"%p")', True),
    ],
)
def test_source_shape_classification(tmp_path, mn, suffix, pred, expected):
    _run(
        tmp_path,
        """
From Coq Require Import List String.
Require Import OpGate.GateFront.
Import ListNotations. Open Scope string_scope.
"""
        + f'Example boundary : source_control_ok (PI ({pred}) "{mn}" {suffix} []) = '
        + str(expected).lower()
        + ".\nProof. vm_compute. reflexivity. Qed.\n",
    )


CASES = [
    ("direct-branch", "", "bra.uni finish; finish: ret;", True),
    ("predicated-branch-constant", "", "@%p bra.uni finish; finish: ret;", False),
    ("direct-call", ".func helper() { ret; }\n", "call.uni helper; ret;", True),
    ("predicated-call-constant", ".func helper() { ret; }\n", "@%p call.uni helper; ret;", False),
    ("ordinary-predicated-call", ".func helper() { ret; }\n", "@%p call helper; ret;", True),
    ("top-return-qualifier", "", "ret.uni;", False),
    ("called-return-qualifier", ".func helper() { ret.uni; }\n", "call.uni helper; ret;", False),
    ("uncalled-return-qualifier", ".func helper() { ret.uni; }\n", "ret;", True),
    (
        "called-predicated-branch",
        ".func helper() { " + DECLS + "@%p bra.uni finish; finish: ret; }\n",
        "call.uni helper; ret;",
        False,
    ),
    (
        "uncalled-predicated-branch",
        ".func helper() { " + DECLS + "@%p bra.uni finish; finish: ret; }\n",
        "ret;",
        True,
    ),
    (
        "nested-return-qualifier",
        ".func leaf() { ret.uni; }\n.func helper() { call.uni leaf; ret; }\n",
        "call.uni helper; ret;",
        False,
    ),
    (
        "nested-predicated-call",
        ".func leaf() { ret; }\n.func helper() { " + DECLS + "@%p call.uni leaf; ret; }\n",
        "call.uni helper; ret;",
        False,
    ),
    (
        "ordinary-callee-return",
        ".func helper() { " + DECLS + "@%p ret; ret; }\n",
        "call.uni helper; ret;",
        True,
    ),
    (
        "fixed-index-uniform-still-needs-proof",
        "",
        "targets: .branchtargets finish; brx.idx.uni %r, targets; finish: ret;",
        False,
    ),
    (
        "fixed-index-ordinary",
        "",
        "targets: .branchtargets finish; brx.idx %r, targets; finish: ret;",
        True,
    ),
    ("direct-branch-before-barrier", "", "bra.uni common; common: bar.sync 0; ret;", True),
]


@pytest.mark.parametrize("name,functions,body,expected", CASES, ids=[c[0] for c in CASES])
def test_expansion_checks_only_visited_function_bodies(tmp_path, name, functions, body, expected):
    text = HEADER + functions + ".visible .entry kernel() {\n" + DECLS + body + "\n}\n"
    (tmp_path / "Input.ptx").write_text(text)
    terms = export_terms(text, [BINDING])
    query = (
        terms
        + """
Definition boundary :=
 match parse_text text size, mfs with
 | Ok ks, [(_,b)] =>
   match ks_entries ks with [(_,k)] => front k b | _ => Err "entry" None end
 | _,_ => Err "parse" None
 end.
Eval vm_compute in match boundary with Ok _ => "lowered" | Err e _ => e end.
Example expected : (match boundary with Ok _=>true|Err _ _=>false end) = EXPECTED.
Proof. vm_compute. reflexivity. Qed.
""".replace("EXPECTED", str(expected).lower())
    )
    if not expected:
        query += """
Example refusal_is_uniform_control :
 (match boundary with Err e _=>String.eqb e "a uniform control qualifier requires a checked warp-uniformity proof"|_=>false end) = true.
Proof. vm_compute. reflexivity. Qed.
"""
    _run(tmp_path, query)
    if expected:
        _run(tmp_path, terms + CHECK_QUERY, "Checked")


def test_uniform_control_lemmas_have_no_assumptions(tmp_path):
    query = """
Require Import OpGate.GateFront.
Print Assumptions source_control_indirect.
Print Assumptions source_control_return.
Print Assumptions source_control_predicated_direct.
"""
    _run(tmp_path, query)
    assert (tmp_path / "Query.stdout.txt").read_text().count("Closed under the global context") == 3
