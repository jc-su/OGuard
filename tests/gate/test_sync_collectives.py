"""CPU-only lowering controls; none of these instruction fragments execute."""

import subprocess

import pytest

from opgate.binding import Base, Binding, Span
from opgate.gate import binding_term, export_terms, formal_dir
from opgate.signature import Affine, Direction

HEADER = ".version 8.0\n.target sm_80\n.address_size 64\n"
SHFL = "shfl.sync.idx.b32 %r2, %r1, 0, 31, -1;"
VOTE = "vote.sync.ballot.b32 %r2, %p1, -1;"


def _kernel(body, header=HEADER):
    return (
        header
        + """.visible .entry kernel(.param .u64 output) {
.reg .pred %p<5>;
.reg .b16 %h<3>;
.reg .b32 %r<8>;
.reg .b64 %rd<3>;
ld.param.u64 %rd1, [output];
mov.u32 %r1, 1;
setp.eq.u32 %p1, %r1, 1;
"""
        + body
        + "\nret;\n}\n"
    )


def _lower(tmp_path, body, expected, header=HEADER, destinations=0):
    text = _kernel(body, header)
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
Eval vm_compute in match result with Ok _ => "lowered" | Err why _ => why end.
Example expected_boundary :
 (match result with Ok _ => true | Err _ _ => false end) = EXPECTED.
Proof. vm_compute. reflexivity. Qed.
""".replace("EXPECTED", str(expected).lower())
    if destinations:
        suffix += r"""
Definition arbitrary_destinations (i : instr) : nat :=
 match i with IDef _ ds => List.length ds | _ => 0%nat end.
Example every_collective_result_is_arbitrary :
 (match result with
 | Ok (p, _) => fold_left Nat.add (map arbitrary_destinations (List.concat p)) 0%nat
 | Err _ _ => 0%nat end) = COUNT%nat.
Proof. vm_compute. reflexivity. Qed.
""".replace("COUNT", str(destinations))
    query = tmp_path / "Boundary.v"
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    query.write_text(export_terms(text, [binding]) + suffix)
    command = ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["up", "down", "bfly", "idx"])
@pytest.mark.parametrize("mask", ["-1", "4294967295", "0xffffffff"])
def test_shuffle_all_modes_and_full_mask_spellings(tmp_path, mode, mask):
    _lower(tmp_path, f"shfl.sync.{mode}.b32 %r2, %r1, 0, 31, {mask};", True, destinations=1)


@pytest.mark.parametrize(
    "body,count",
    [
        ("shfl.sync.idx.b32 %r2|%p2, %r1, %r1, 31, -1;", 2),
        (VOTE, 1),
        ("vote.sync.all.pred %p2, %p1, -1;", 1),
        ("vote.sync.any.pred %p2, !%p1, 4294967295;", 1),
        ("vote.sync.uni.pred %p2, %p1, -1;", 1),
    ],
)
def test_results_are_preserved(tmp_path, body, count):
    _lower(tmp_path, body, True, destinations=count)


@pytest.mark.parametrize(
    "body",
    [
        "@%p1 shfl.sync.idx.b32 %r2, %r1, 0, %r1, -1;",
        "@!%p1 shfl.sync.idx.b32 %r2|%p2, %r1, 0, 31, -1;",
        "@%p1 vote.sync.ballot.b32 %r2, !%p1, -1;",
        "@%p1 match.all.sync.b32 %r2|%p2, %r1, -1;",
        "@!%p1 redux.sync.min.s32 %r2, %r1, -1;",
    ],
)
def test_a_predicated_collective_has_no_participation_proof(tmp_path, body):
    # Every lane that the member mask names must reach the collective; the
    # phase checker proves this only for an unpredicated source instruction.
    _lower(tmp_path, body, False)


@pytest.mark.parametrize(
    "target", ["sm_70", "sm_72", "sm_75", "sm_80", "sm_86", "sm_90a", "sm_100f", "compute_80"]
)
def test_explicit_modern_target(tmp_path, target):
    _lower(tmp_path, SHFL, True, HEADER.replace("sm_80", target), 1)


@pytest.mark.parametrize(
    "header",
    [
        "",
        ".version 8.0\n",
        ".target sm_80\n",
        HEADER.replace("sm_80", "sm_60"),
        HEADER.replace("8.0", "5.0"),
        HEADER.replace("sm_80", "sm_80garbage"),
        HEADER.replace("sm_80", "sm_080"),
        HEADER.replace("sm_80", "sm_80a"),
        HEADER.replace("sm_80", "sm_999"),
        HEADER.replace("sm_80", "sm_80, unsupported"),
        HEADER.replace("sm_80", "sm_80, debug, debug"),
        HEADER.replace("sm_80", "sm_80, texmode_unified, texmode_independent"),
        HEADER.replace("sm_80", "sm_80;"),
        HEADER.replace("8.0", "8.0extra"),
        HEADER.replace("8.0", "8..0"),
        HEADER.replace("8.0", "8.0 7.0"),
        HEADER + ".target sm_60\n",
        HEADER + ".target sm_80\n",
        HEADER + ".version 8.0\n",
        '.file 1 "ignored"\n' + HEADER,
        "// .target sm_80\n" + HEADER.replace("sm_80", "sm_60"),
        HEADER.replace("sm_80", "sm_60") + '.file 1 ".target sm_80"\n',
    ],
)
def test_architecture_is_not_inferred_or_partially_parsed(tmp_path, header):
    _lower(tmp_path, SHFL, False, header)


@pytest.mark.parametrize(
    "body",
    [
        SHFL.replace("-1", "0"),
        SHFL.replace("-1", "2147483647"),
        SHFL.replace("-1", "%rd1"),
        SHFL.replace("-1", "8589934591"),
        SHFL.replace("sync.idx", "idx"),
        SHFL.replace("sync.idx", "sync.unclassified"),
        SHFL.replace(".b32", ".b64"),
        SHFL.replace(".b32", ".b32.extra"),
        SHFL.replace("%r2,", "%rd2,"),
        SHFL.replace("%r1,", "%rd2,"),
        SHFL.replace("0, 31", "%p2, 31"),
        SHFL.replace("0, 31", "0, %rd2"),
        SHFL.replace("0, 31", "0, 4294967296"),
        SHFL.replace("%r2,", "{%r2, %r3},"),
        SHFL.replace("%r2,", "%r2|%r3,"),
        SHFL.replace("%r2,", "%r2|%p2|%p3,"),
        SHFL.replace("%r2,", "%missing,"),
        SHFL.replace(", -1", ""),
        "@%r1 " + SHFL,
        VOTE.replace("%p1", "%r1"),
        VOTE.replace("%r2", "%p2"),
        VOTE.replace(".b32", ".pred"),
        VOTE.replace("ballot", "unclassified"),
        VOTE.replace("sync.", ""),
        VOTE.replace("-1", "%rd1"),
        VOTE.replace(", -1", ""),
        VOTE.replace("%r2", "[%rd1]"),
        "@%p1 bar.sync 0;",
        "barrier.sync 0;",
    ],
)
def test_unsupported_shapes_and_effects_fail_closed(tmp_path, body):
    _lower(tmp_path, body, False)


def test_comments_and_strings_cannot_override_real_header(tmp_path):
    header = "// .target sm_60\n/* .version 1.0 */\n" + HEADER + '.file 1 ".target sm_60"\n'
    _lower(tmp_path, SHFL, True, header, 1)


def test_no_header_preserves_noncollective_module(tmp_path):
    _lower(tmp_path, "st.global.u32 [%rd1], %r1;", True, ".address_size 64\n")


def _checked(tmp_path, body, expected):
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
Definition certificate : text_certificate :=
 Eval vm_compute in propose_text_certificate text size mfs.
Eval vm_compute in text_diagnosis text size mfs certificate.
Example expected_checked_decision :
 check_text_certificate_ok text size mfs certificate = EXPECTED.
Proof. vm_compute. reflexivity. Qed.
""".replace("EXPECTED", str(expected).lower())
    query = tmp_path / "Checked.v"
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    query.write_text(export_terms(text, [binding]) + suffix)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "body,expected",
    [
        ("mov.b32 %r7, -1; " + SHFL.replace("-1", "%r7"), True),
        ("mov.b32 %r7, 4294967295; " + VOTE.replace("-1", "%r7"), True),
        # a predicated collective has no participation proof
        ("mov.b32 %r7, -1; @%p1 " + SHFL.replace("-1", "%r7"), False),
        ("mov.b32 %r7, -1; mov.b32 %r7, 1; " + SHFL.replace("-1", "%r7"), False),
        ("mov.b32 %r7, 1; " + VOTE.replace("-1", "%r7"), False),
        ("activemask.b32 %r7; " + VOTE.replace("-1", "%r7"), False),
        ("mov.b32 %r7, 1; setp.eq.u32 %p1, %r1, 0; @%p1 " + SHFL.replace("-1", "%r7"), False),
    ],
)
def test_register_mask_requires_an_independent_checked_equality(tmp_path, body, expected):
    _checked(tmp_path, body, expected)


@pytest.mark.parametrize(
    "body",
    [
        "bar.sync 0;",
        "bar.sync 15;",
        "bar.sync 0; bar.sync 0; bar.sync 15;",
        "@%p1 add.u32 %r2, %r1, 1; bar.sync 0;",
        SHFL + " bar.sync 0; " + VOTE,
    ],
)
def test_straight_line_unconditional_barrier_is_checked(tmp_path, body):
    _checked(tmp_path, body, True)


@pytest.mark.parametrize(
    "body",
    [
        "@%p1 bar.sync 0;",
        "bar.sync %r1;",
        "bar.sync -1;",
        "bar.sync 16;",
        "bar.sync 0, 32;",
        "bar.cta.sync 0;",
        "barrier.sync 0;",
        "@%p1 bra next; bar.sync 0; next: mov.u32 %r2, 1;",
        "loop: @%p1 bra loop; bar.sync 0;",
        "@%p1 ret; bar.sync 0;",
        "exit; bar.sync 0;",
        "trap; bar.sync 0;",
        "brkpt; bar.sync 0;",
    ],
)
def test_barrier_without_a_checked_participation_proof_refuses(tmp_path, body):
    _checked(tmp_path, body, False)


@pytest.mark.parametrize(
    "version,expected", [("6.0", True), ("9.4", True), ("9.5", False), ("10.0", False)]
)
def test_collectives_require_reviewed_isa_version(tmp_path, version, expected):
    _lower(tmp_path, SHFL, expected, HEADER.replace("8.0", version), 1 if expected else 0)


@pytest.mark.parametrize(
    "mode,ty",
    [(m, t) for m in ["add", "min", "max"] for t in ["s32", "u32"]]
    + [(m, "b32") for m in ["and", "or", "xor"]],
)
@pytest.mark.parametrize("mask", ["-1", "4294967295", "0xffffffff"])
def test_integer_reduction_modes(tmp_path, mode, ty, mask):
    _lower(tmp_path, f"redux.sync.{mode}.{ty} %r2, %r1, {mask};", True, destinations=1)


@pytest.mark.parametrize(
    "body,count",
    [
        ("match.any.sync.b32 %r2, %r1, -1;", 1),
        ("match.any.sync.b64 %r2, %rd1, -1;", 1),
        ("match.all.sync.b32 %r2, %r1, -1;", 1),
        ("match.all.sync.b64 %r2|%p2, %rd1, -1;", 2),
    ],
)
def test_match_destinations(tmp_path, body, count):
    _lower(tmp_path, body, True, destinations=count)


@pytest.mark.parametrize(
    "body,count",
    [
        ("mov.b32 %r3, -1; redux.sync.max.s32 %r2, %r1, %r3;", 1),
        ("mov.b32 %r3, -1; match.all.sync.b64 %r2|%p2, %rd1, %r3;", 2),
    ],
)
def test_register_mask_checks_complete(tmp_path, body, count):
    from opgate.gate import Accepted, verify

    binding = Binding(
        "kernel",
        (),
        (("output", Base(Span(Direction.WRITE, Affine(64), 4, None, False))),),
        (32, 1, 1),
        (1, 1, 1),
        Affine(0),
    )
    text = _kernel(body + "\nst.global.u32 [%rd1], %r2;")
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Input.v").write_text(export_terms(text, [binding]))
    assert isinstance(verify(text, [binding]), Accepted)
    _lower(tmp_path, body, True, destinations=count)


@pytest.mark.parametrize(
    "body",
    [
        "bra next; next: bar.sync 0;",
        "@%p1 bra next; add.u32 %r2, %r1, 1; next: bar.sync 0;",
        "bar.sync 0; @%p1 ret;",
        "mov.u32 %r2, %tid.x; loop: add.u32 %r2, %r2, 32; "
        "setp.lt.u32 %p2, %r2, 45; @%p2 bra loop; bar.sync 0;",
    ],
)
def test_finite_paths_reach_one_barrier(tmp_path, body):
    # Threads may take different loop counts; every path reaches the same event.
    _checked(tmp_path, body, True)
