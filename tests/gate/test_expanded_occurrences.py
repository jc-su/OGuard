"""Benign static calls plus pure metadata rejection controls; no GPU work."""

import subprocess

import pytest

from opgate.binding import Binding
from opgate.gate import CHECK_QUERY, binding_term, export_terms, formal_dir
from opgate.signature import Affine

BINDING = Binding("kernel", (), (), (32, 1, 1), (1, 1, 1), Affine(0))


def run(tmp_path, source, name="Query"):
    path = tmp_path / f"{name}.v"
    path.write_text(source)
    result = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    (tmp_path / f"{name}.stdout").write_text(result.stdout)
    (tmp_path / f"{name}.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def helper_program(mode, nested):
    mask = "-1" if mode == "literal" else "%r3"
    suffix = "mov.u32 %r3,0;" if mode == "snapshot" else ""
    functions = f"""
.func leaf() {{
.reg .b32 %r<5>;
mov.u32 %r1,1;
mov.u32 %r3,-1;
shfl.sync.idx.b32 %r2,%r1,0,31,{mask};
shfl.sync.bfly.b32 %r4,%r2,1,31,{mask};
{suffix}
ret;
}}
"""
    if nested:
        functions += ".func helper() { call.uni leaf; call.uni leaf; ret; }\n"
    callee = "helper" if nested else "leaf"
    return (
        ".version 8.0\n.target sm_80\n.address_size 64\n"
        + functions
        + f".visible .entry kernel() {{ call.uni {callee}; bar.sync 0; "
        + f"call.uni {callee}; bar.sync 1; ret; }}\n"
    )


@pytest.mark.parametrize("mode", ["literal", "register", "snapshot"])
@pytest.mark.parametrize("nested", [False, True])
def test_every_expanded_helper_collective_is_checked(tmp_path, mode, nested):
    text = helper_program(mode, nested)
    (tmp_path / "Input.ptx").write_text(text)
    (tmp_path / "Binding.txt").write_text(binding_term(BINDING))
    terms = export_terms(text, [BINDING])
    run(tmp_path, terms + CHECK_QUERY, "Checked")
    count = 10 if nested else 6
    query = (
        terms
        + r"""
Require Import OpGate.GateLoop.
Definition metadata := Eval vm_compute in
 match parse_text text size,mfs with
 | Ok ks,[(_,b)] => match ks_entries ks with
 | [(_,k)] => match inline k with
 | Ok (expanded,raw) => match
   (p <- translate_placed expanded b raw ;; F <- facts expanded b ;; ret (p,F)) init with
 | Ok ((pp,F),ts) =>
   let tags := expanded_places raw in
   match placed_events expanded tags (ts_ids ts) pp (source_events expanded tags) with
   | Some events => Some (raw,tags,pp,events)
   | None => None end
 | _ => None end
 | _ => None end
 | _ => None end
 | _,_ => None end.
Example complete_collective_count :
 (match metadata with Some (_,_,_,events) => List.length events | None => 0%nat end)=COUNT%nat.
Proof. vm_compute. reflexivity. Qed.
Example event_positions_are_distinct :
 (match metadata with Some (_,_,_,events) => loop_events_ordered events | None => false end)=true.
Proof. vm_compute. reflexivity. Qed.
Example raw_caller_origins_can_repeat :
 (match metadata with Some (raw,_,_,_) =>
   negb (Nat.eqb (List.length (nodup N.eq_dec (map fst raw))) (List.length raw))
 | None=>false end)=true.
Proof. vm_compute. reflexivity. Qed.
Example diagnostic_restore_keeps_all_instructions :
 match metadata with Some (raw,_,pp,_) =>
   exists restored, diagnostic_program raw pp=Some restored /\ map (map fst) restored=map (map fst) pp
 | None => False end.
Proof. vm_compute. eexists. split;reflexivity. Qed.
""".replace("COUNT", str(count))
    )
    run(tmp_path, query, "Metadata")


def test_expanded_identity_and_diagnostic_proofs_are_closed(tmp_path):
    names = [
        "expanded_places_length",
        "expanded_places_nth",
        "expanded_places_in_bounds",
        "expanded_places_unique",
        "diagnostic_block_instructions",
        "diagnostic_program_instructions",
        "accepted_front_phase",
        "accepted_front_events",
    ]
    result = run(
        tmp_path,
        "Require Import OpGate.GateFront.\n"
        + "".join(f"Print Assumptions {name}.\n" for name in names),
    )
    assert result.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in result.stdout


def test_metadata_does_not_merge_repeat_calls_or_incomplete_spans(tmp_path):
    run(
        tmp_path,
        r"""
From Coq Require Import List String ZArith NArith.
Require Import OpGate.GateIsa OpGate.GateDomain OpGate.GateFront.
Import ListNotations. Open Scope string_scope.
Lemma actual_mark_retains_each_expanded_source : forall tp n after items,
 map (fun '(occ,(_,fr),i,_) => (occ,fr,i)) (mark tp n after items) =
 map (fun '(p,i) => (fst p,snd p,i))
   (combine (expanded_places_from n (map fst items)) (map snd items)).
Proof.
 intros tp n after items; revert n after.
 induction items as [|[[raw fr] i] rest IH]; intros n after; cbn; [reflexivity|].
 f_equal. apply IH.
Qed.
Print Assumptions actual_mark_retains_each_expanded_source.
Definition raw : list place := [(60%N,[0%N]);(60%N,[0%N]);(87%N,[1%N])].
Example distinct_occurrences : expanded_places raw=[(0%N,[0%N]);(1%N,[0%N]);(2%N,[1%N])].
Proof. vm_compute. reflexivity. Qed.
Example complete_raw_diagnostics : diagnostic_program raw
 [[(IDef None [1%positive],0%N);(IDef None [2%positive],1%N)];[(INop,2%N)]] =
 Some [[(IDef None [1%positive],60%N);(IDef None [2%positive],60%N)];[(INop,87%N)]].
Proof. vm_compute. reflexivity. Qed.
Example unknown_occurrence_fails : diagnostic_program raw [[(INop,3%N)]]=None.
Proof. vm_compute. reflexivity. Qed.
Example same_occurrence_with_extra_marker_refuses : placed_event ELiteral
 [[(IDef None [1%positive],0%N);(IDef None [2%positive],0%N)]] 0%N=None.
Proof. vm_compute. reflexivity. Qed.
Example missing_actual_marker_refuses : placed_event ELiteral [[(IDef None [1%positive],0%N)]] 1%N=None.
Proof. vm_compute. reflexivity. Qed.
Example unrelated_occurrence_is_not_in_span : placed_event ELiteral
 [[(IDef None [1%positive],0%N);(IDef None [2%positive],1%N)]] 0%N=Some (0%nat,0%nat).
Proof. vm_compute. reflexivity. Qed.
""",
    )
