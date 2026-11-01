"""Exact DFS representation controls; no PTX or device execution."""

import subprocess

import pytest

from opgate.gate import formal_dir

HEADER = "From Coq Require Import List ZArith Bool.\nFrom OpGate Require Import GateIsa GateForm GateDomain GateBarrier GateLoop.\nImport ListNotations.\nLocal Open Scope nat_scope.\n\nDefinition root : pc := (0, 0).\nDefinition left : pc := (0, 1).\nDefinition right : pc := (1, 0).\nDefinition leaf : pc := (1, 1).\nDefinition cut : pc := (10, 10).\nDefinition diamond (pos : pc) : list pc :=\n  if phase_event root pos then [left; right] else\n  if phase_event left pos || phase_event right pos then [leaf] else [].\nDefinition cycle (pos : pc) : list pc := [pos].\nDefinition none (_ : pc) : list pc := [].\nDefinition walk next fuel events pending active done :=\n  occurrence_graph_walk_with next fuel events cut pending active done.\n\n\nDefinition active_after_pop (nodes : list pc) (query : pc) : bool :=\n  match occurrence_active_pop (occurrence_active_of_list nodes) with\n  | Some (_, parents) => occurrence_active_member parents query\n  | None => false\n  end.\n"

CASES = [
    ("empty_zero_fuel", "walk cycle 0 [] [] [] [] = true"),
    ("pending_zero_fuel", "walk none 0 [] [(true, root)] [] [] = false"),
    ("shared_child_visited_once", "walk diamond 9 [] [(true, root)] [] [] = true"),
    ("exact_fuel_preserved", "walk diamond 8 [] [(true, root)] [] [] = false"),
    ("cycle_detected", "walk cycle 3 [] [(true, root)] [] [] = false"),
    ("event_stops_cycle", "walk cycle 1 [root] [(true, root)] [] [] = true"),
    ("duplicate_events_unchanged", "walk cycle 1 [root; root] [(true, root)] [] [] = true"),
    ("arbitrary_done_stops_cycle", "walk cycle 1 [] [(true, root)] [] [root] = true"),
    ("duplicate_initial_done", "walk cycle 1 [] [(true, root)] [] [root; root] = true"),
    ("initial_active_is_checked", "walk none 1 [] [(true, root)] [root] [] = false"),
    ("done_precedes_active", "walk none 1 [] [(true, root)] [root] [root] = true"),
    ("event_precedes_active", "walk none 1 [root] [(true, root)] [root] [] = true"),
    ("exit_matches_top", "walk none 1 [] [(false, root)] [root] [] = true"),
    ("exit_mismatch_refuses", "walk none 1 [] [(false, left)] [root] [] = false"),
    ("empty_exit_stack_refuses", "walk none 1 [] [(false, root)] [] [] = false"),
    (
        "duplicate_initial_active_preserved",
        "walk none 2 [] [(false, root); (false, root)] [root; root] [] = true",
    ),
    (
        "initial_active_need_not_be_empty_when_pending_finishes",
        "walk none 0 [] [] [root] [] = true",
    ),
    (
        "block_and_offset_are_distinct_keys",
        "occurrence_pc_member (occurrence_pc_of_list [left]) right = false",
    ),
    (
        "zero_coordinates_are_represented",
        "occurrence_pc_member (occurrence_pc_of_list [root]) root = true",
    ),
    (
        "beyond_cut_stops_before_cycle",
        "occurrence_graph_walk_with cycle 1 [] root [(true, left)] [] [] = true",
    ),
    (
        "completed_shared_child_with_repeated_roots",
        "walk diamond 10 [] [(true, root); (true, root)] [] [] = true",
    ),
    (
        "done_precedes_remaining_duplicate",
        "walk none 2 [] [(false, root); (true, root)] [root; root] [] = true",
    ),
    (
        "done_precedes_deeper_duplicate",
        "walk none 3 [] [(false, root); (false, left); (true, root)] [root; left; root] [] = true",
    ),
    (
        "completed_node_skips_on_reentry",
        "walk none 4 [] [(false, left); (true, left)] [left; root] [] = true",
    ),
    ("duplicate_membership_survives_parent_restore", "active_after_pop [root; root] root = true"),
    ("deeper_duplicate_membership_survives", "active_after_pop [root; left; root] root = true"),
    ("unique_membership_is_removed_on_restore", "active_after_pop [left; root] left = false"),
    ("other_parent_membership_is_retained", "active_after_pop [left; root] root = true"),
]


def compile_query(tmp_path, suffix):
    query = tmp_path / "Query.v"
    query.write_text(HEADER + suffix)
    run = subprocess.run(
        ["coqc", "-q", "-Q", str(formal_dir()), "OpGate", str(query)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "stdout.txt").write_text(run.stdout)
    (tmp_path / "stderr.txt").write_text(run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    return run


@pytest.mark.parametrize("name,proposition", CASES, ids=[name for name, _ in CASES])
def test_graph_representation_boundaries(tmp_path, name, proposition):
    compile_query(
        tmp_path, f"Example {name}: {proposition}.\nProof. vm_compute. reflexivity. Qed.\n"
    )


def test_graph_representation_equivalence_is_closed(tmp_path):
    names = (
        "occurrence_graph_walk_member_exact",
        "occurrence_graph_walk_state_ext",
        "occurrence_graph_walk_with_exact",
        "occurrence_active_list_exact",
        "occurrence_pc_list_exact",
    )
    run = compile_query(tmp_path, "".join(f"Print Assumptions {name}.\n" for name in names))
    assert run.stdout.count("Closed under the global context") == len(names)
    assert "Axioms:" not in run.stdout
