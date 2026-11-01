"""Whole PTX controls for launch-dependent mixed synchronization loops."""

import re
import subprocess

import pytest

from opgate.binding import Binding, Size
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine, SizeSymbol

ROUTE = r"""
Require Import OpGate.GateContext.
Definition selected_mixed_route:bool:=
 match parse_text text size,mfs with
 |Ok ks,[(bound,Mf)]=>match ks_entries ks with
 |[(name,K)]=>match inline K with
 |Ok(K',org)=>match(p<-translate_placed K' Mf org;; F<-facts K' Mf;;ret(p,F))init with
 |Ok((pp,raw),ts)=>let placed:=expanded_places org in
   let F:=payload_facts raw(map(map fst)pp)in
   match placed_events K' placed(ts_ids ts)pp(source_events K' placed)with
   |Some events=>match front_context_route F(cta_uniform_key F Mf)K' placed(ts_ids ts)pp events with
     |Some(inr _)=>true|_=>false end
   |None=>false end
 |_=>false end
 |_=>false end
 |_=>false end
 |_,_=>false end.
"""


def mixed_program(width, *, nested=False, variant="ordinary", literal=None):
    load = f"ld.param.u{width} %limit,[count];"
    if literal is not None:
        load = f"mov.u{width} %limit,{literal};"
    initial_test = f"setp.ge.s{width} %skip,%counter,%limit;"
    if variant == "nonuniform_skip":
        initial_test = "setp.eq.u32 %skip,%tid,0;"
    ready = ""
    if variant == "nonuniform_limit":
        ready = f"cvt.u{width}.u32 %limit,%tid;\nadd.u{width} %limit,%limit,1;"
    mask = "0xfffffffe" if variant == "partial_mask" else "0xffffffff"
    collective = f"shfl.sync.bfly.b32 %value,%tid,1,31,{mask};"
    if nested:
        collective = f"""mov.u32 %inner,16;
INNER:
{collective}
shr.u32 %inner,%inner,1;
setp.gt.u32 %again,%inner,0;
@%again bra INNER;"""
    overwrite = ""
    if variant == "overwritten_limit":
        overwrite = f"cvt.u{width}.u32 %limit,%value;\nadd.u{width} %limit,%limit,1;"
    stride = 2 ** (width - 1) if variant == "signed_wrap" else 1
    return f""".version 8.0
.target sm_80
.address_size 64
.visible .entry kernel(.param .u{width} count) {{
.reg .pred %skip,%more,%again;
.reg .b32 %tid,%value,%inner;
.reg .b{width} %counter,%limit;
{load}
mov.u32 %tid,%tid.x;
mov.u{width} %counter,0;
{initial_test}
@%skip bra DONE;
{ready}
bar.sync 0;
BODY:
{collective}
bar.sync 1;
{overwrite}
add.s{width} %counter,%counter,{stride};
setp.lt.s{width} %more,%counter,%limit;
@%more bra BODY;
DONE:
ret;
}}
"""


def compile_program(tmp_path, text, block, *, accepted, selected=None, count_bounds=(1, 9)):
    binding = Binding(
        "kernel",
        (SizeSymbol("count", *count_bounds),),
        (("count", Size(Affine(0, ((0, 1),)))),),
        (block, 1, 1),
        (4, 1, 1),
        Affine(0),
    )
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(binding))
    terms = export_terms(text, [binding])
    query = terms + CHECK_QUERY
    if selected is not None:
        route_source = tmp_path / "Selected.v"
        route_source.write_text(
            terms
            + ROUTE
            + f"Example expected_route:selected_mixed_route={'true' if selected else 'false'}.\n"
            + "Proof. vm_compute. reflexivity. Qed.\n"
        )
        route = subprocess.run(
            ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(route_source)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        (tmp_path / "selected-stdout.txt").write_text(route.stdout)
        (tmp_path / "selected-stderr.txt").write_text(route.stderr)
        assert route.returncode == 0, route.stdout + route.stderr
    source = tmp_path / "Query.v"
    source.write_text(query)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    if accepted:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "the unproved variant was accepted"
        assert "OPGATE_REFUSAL" in result.stdout, result.stdout + result.stderr
        # the query's refusal of the front end's result or of the certificate
        assert re.search(
            r"Error: Tactic failure: (front end|certificate) refused\.", result.stderr
        ), result.stderr


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("block", [32, 256])
@pytest.mark.parametrize("nested", [False, True])
def test_uniform_count_mixed_loop(tmp_path, width, block, nested):
    compile_program(
        tmp_path,
        mixed_program(width, nested=nested),
        block,
        accepted=True,
        selected=True,
    )


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize(
    "variant",
    ["nonuniform_skip", "nonuniform_limit", "partial_mask", "overwritten_limit", "signed_wrap"],
)
def test_unproved_participation_or_counter_is_refused(tmp_path, width, variant):
    compile_program(
        tmp_path,
        mixed_program(width, variant=variant),
        32,
        accepted=False,
    )


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_literal_count_preserves_existing_route(tmp_path, width, count):
    compile_program(
        tmp_path,
        mixed_program(width, literal=count),
        32,
        accepted=True,
        selected=False,
    )


def test_mixed_context_theorems_are_closed(tmp_path):
    names = [
        "mixed_entry_check_sound",
        "mixed_context_check_sound",
        "front_event_check_sound",
        "front_without_mixed_capability_preserved",
        "front_mixed_ineligible_preserved",
        "front_all_cta_route_preserved",
        "front_no_cta_route_preserved",
        "front_mixed_refusal_final",
        "accepted_front_linked_events",
    ]
    source = tmp_path / "Closed.v"
    source.write_text(
        "Require Import OpGate.GateContext OpGate.GateFront OpGate.StoreLink.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names)
    )
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


def test_lane_dependent_body_event_skip_is_refused(tmp_path):
    text = mixed_program(32)
    assert text.count("BODY:\n") == 1
    assert text.count("add.s32 %counter,%counter,1;") == 1
    text = text.replace(
        "BODY:\n",
        "BODY:\nsetp.eq.u32 %again,%tid,0;\n@%again bra AFTER_EVENTS;\n",
    ).replace("add.s32 %counter,%counter,1;", "AFTER_EVENTS:\nadd.s32 %counter,%counter,1;")
    compile_program(tmp_path, text, 32, accepted=False, selected=True)


@pytest.mark.parametrize("width", [32, 64])
def test_warp_varying_finite_words_keep_scoped_route(tmp_path, width):
    text = mixed_program(width).replace("bar.sync 1;", "")
    text = text.replace(
        "mov.u32 %tid,%tid.x;",
        f"mov.u32 %tid,%tid.x;\nshr.u32 %inner,%tid,5;\ncvt.u{width}.u32 %limit,%inner;\nadd.u{width} %limit,%limit,1;",
    )
    compile_program(tmp_path, text, 256, accepted=True, selected=False)


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("variant", ["ordinary", "initial_one", "suffix_cta"])
def test_singleton_limit_keeps_finite_route(tmp_path, width, variant):
    text = mixed_program(width)
    if variant == "initial_one":
        text = text.replace(f"mov.u{width} %counter,0;", f"mov.u{width} %counter,1;")
    elif variant == "suffix_cta":
        text = text.replace("DONE:\nret;", "bar.sync 2;\nDONE:\nret;")
    compile_program(tmp_path, text, 32, accepted=True, selected=False, count_bounds=(3, 3))


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("initial", [0, 1])
def test_known_one_iteration_preserves_initialization(tmp_path, width, initial):
    text = mixed_program(width).replace(
        f"mov.u{width} %counter,0;", f"mov.u{width} %counter,{initial};"
    )
    text = text.replace(
        f"add.s{width} %counter,%counter,1;", f"add.s{width} %counter,%counter,256;"
    )
    compile_program(tmp_path, text, 32, accepted=True, selected=initial == 0, count_bounds=(2, 3))


@pytest.mark.parametrize("width", [32, 64])
def test_guarded_zero_write_is_not_initialization_authority(tmp_path, width):
    text = mixed_program(width).replace(
        f"mov.u{width} %counter,0;", f"setp.eq.u32 %again,%tid,0;\n@%again mov.u{width} %counter,0;"
    )
    compile_program(tmp_path, text, 32, accepted=False, selected=False)


@pytest.mark.parametrize("width", [32, 64])
def test_nonsingleton_suffix_event_preserves_finite_route(tmp_path, width):
    text = (
        mixed_program(width)
        .replace(f"add.s{width} %counter,%counter,1;", f"add.s{width} %counter,%counter,256;")
        .replace("DONE:\nret;", "bar.sync 2;\nDONE:\nret;")
    )
    compile_program(tmp_path, text, 32, accepted=True, selected=False, count_bounds=(2, 3))
