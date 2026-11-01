"""General mixed-event schedule metadata; ordinary checked Coq controls."""

import subprocess

import pytest

from opgate.gate import formal_dir


def compile_query(tmp_path, query):
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


META = r"""
From Coq Require Import List ZArith Bool.
From OpGate Require Import GateIsa GateForm GateDomain GateMeaning GateBarrier GateLoop GateContext.
Import ListNotations. Local Open Scope Z_scope.
Definition mfacts w hi:lfacts:={|
 lf_widths:=PM.add 1%positive w(PM.add 3%positive w(PM.add 4%positive 1%nat(PM.add 7%positive w(PM.add 9%positive w(PM.empty nat)))));
 lf_fixed:=[7%positive;9%positive;100%positive;101%positive];
 lf_bounds:=[(7%positive,(0,hi));(9%positive,(0,31))];lf_aligns:=[];lf_roles:=[];
 lf_shared_window:=100%positive;lf_local_window:=101%positive;
 lf_shared:=[];lf_locals:=[];lf_cells:=[];lf_payloads:=PM.empty unit;lf_derived:=[];lf_names:=[]|}.
Definition muniform r:=Pos.eqb r 7%positive.
Definition mprogram w source stride marker:program:=[
 [IImm None 1%positive 0;IMov None 3%positive source;
  ISetp None CGe w true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))3%nat];
 [INop;IImm None 1%positive 0;IBra None 2%nat];
 [marker;INop;IBin None BAdd w 1%positive 1%positive(OImm stride);
  ISetp None CLt w true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))2%nat];
 [IExit None]].
Definition mevents:list pc:=[(1%nat,0%nat);(2%nat,0%nat);(2%nat,1%nat)].
Definition mchecked F uniform prog events:=match mixed_context_shape_select prog events with
 |Some shape=>mixed_context_check F uniform prog events shape|None=>false end.
Definition marker:=IDef None[6%positive].
Definition literal_program w n:=
 [IImm None 1%positive 0;IImm None 3%positive n;
  ISetp None CGe w true 4%positive 1%positive(OReg 3%positive);IBra(Some(false,4%positive))3%nat]
 ::tl(mprogram w 7%positive 1 marker).
"""


@pytest.mark.parametrize(
    "name, obligation",
    [
        (
            "positive32",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2 marker)mevents=true",
        ),
        (
            "positive64",
            "mchecked(mfacts 64 9)muniform(mprogram 64 7%positive 2 marker)mevents=true",
        ),
        (
            "nonuniform_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 9%positive 2 marker)mevents=false",
        ),
        (
            "zero_stride_shape_absent",
            "mixed_context_shape_select(mprogram 32 7%positive 0 marker)mevents=None",
        ),
        (
            "descending_shape_absent",
            "mixed_context_shape_select(mprogram 32 7%positive (-1) marker)mevents=None",
        ),
        (
            "overflow32_refused",
            "mchecked(mfacts 32(2^31-1))muniform(mprogram 32 7%positive 2 marker)mevents=false",
        ),
        (
            "overflow64_refused",
            "mchecked(mfacts 64(2^63-1))muniform(mprogram 64 7%positive 2 marker)mevents=false",
        ),
        (
            "event_counter_write_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2(IDef None[1%positive]))mevents=false",
        ),
        (
            "event_limit_write_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2(IDef None[3%positive]))mevents=false",
        ),
        (
            "increment_as_event_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2 marker)((2%nat,2%nat)::mevents)=false",
        ),
        (
            "suffix_event_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2 marker)(mevents++[(3%nat,0%nat)])=false",
        ),
        (
            "literal_shapes_retained",
            "\n map(fun n=>mixed_context_shape_select(literal_program 32 n)mevents)[0;1;4]=[None;None;None] /\\\n map(fun n=>mixed_context_shape_select(literal_program 64 n)mevents)[0;1;4]=[None;None;None]",
        ),
        (
            "copied_constant_unmarked",
            "\n mixed_input_run[IImm None 3%positive 0;IMov None 8%positive 3%positive](fun _=>false)8%positive=false",
        ),
        (
            "converted_constant_unmarked",
            "\n mixed_input_run[IImm None 3%positive 4;ICvt None 32 64 false 8%positive 3%positive](fun _=>false)8%positive=false",
        ),
        (
            "arithmetic_constant_unmarked",
            "\n mixed_input_run[IImm None 3%positive 4;IBin None BAdd 64 8%positive 3%positive(OImm 5)](fun _=>false)8%positive=false",
        ),
        (
            "exit_span_cannot_preserve_written_counter",
            "\n total_span_check(mfacts 32 9)[[IImm None 1%positive 4;IExit None]](1%nat,0%nat)[1%positive][(0%nat,0%nat);(0%nat,1%nat)]=false",
        ),
        (
            "dead_sink_is_preservation_only",
            "\n total_span_check(mfacts 32 9)[[IExit None]](0%nat,1%nat)[1%positive][(0%nat,0%nat);(1%nat,0%nat)]=true /\\\n occurrence_segment_with no_loop_summary 10 10(PM.empty unit)(mfacts 32 9)[[IExit None]][]\n (0%nat,1%nat)(0%nat,0%nat)(empty_state(mfacts 32 9))=None",
        ),
        (
            "width_mismatch_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 64 7%positive 2 marker)mevents=false",
        ),
        (
            "guarded_marker_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2(IDef(Some(false,4%positive))[6%positive]))mevents=false",
        ),
        (
            "secondary_counter_write_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2(IDef None[6%positive;1%positive]))mevents=false",
        ),
        (
            "early_exit_refused",
            "mchecked(mfacts 32 9)muniform(mprogram 32 7%positive 2(IExit None))mevents=false",
        ),
    ],
)
def test_mixed_metadata(tmp_path, name, obligation):
    compile_query(
        tmp_path,
        META
        + f"Example {name}: {obligation}. Proof. vm_compute. repeat split; reflexivity. Qed.\n",
    )


def test_mixed_public_proofs_closed(tmp_path):
    names = [
        "occurrence_segment_with_sound",
        "total_span_preserves",
        "counted_add_sound",
        "counted_test_sound",
        "counted_bounds_sound",
        "mixed_outer_pair",
        "mixed_entry_check_sound",
        "mixed_context_check_sound",
    ]
    result = compile_query(
        tmp_path,
        "Require Import OpGate.GateContext.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout
