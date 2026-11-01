"""The theorem chain assumes nothing beyond Rocq's primitive integers and arrays.

`verify_certificate_sound`, the gate's guarantee, depends only on the axioms that
declare the primitive 63-bit integers and the primitive arrays, which the
checker's evaluation uses; `tenant_isolation`, the end-to-end theorem, with
`own_decision_same` and `tenant_isolation_same_steps`, and `term_check_sound`,
that every thread of an accepted kernel ends, are closed under the global
context (design section 7 and decisions 40 and 52).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from opgate.gate import formal_dir

PRIMITIVE = re.compile(r"^(PrimInt63|PArray)\.\w+\b")


def _assumptions(theorem: str, library: str) -> str:
    coqc = shutil.which("coqc")
    assert coqc is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Assumptions.v"
        path.write_text(f"Require Import OpGate.{library}.\nPrint Assumptions {theorem}.\n")
        result = subprocess.run(
            [coqc, "-q", "-Q", str(formal_dir()), "OpGate", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
        )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_the_gate_assumes_only_the_primitive_integers_and_arrays() -> None:
    printed = _assumptions("verify_certificate_sound", "GateParse")
    lines = printed.split("Axioms:", 1)[1].splitlines()
    axioms = [line for line in lines if line and not line[0].isspace()]
    assert axioms, printed
    assert all(PRIMITIVE.match(axiom) for axiom in axioms), axioms


def test_the_verified_premise_follows_from_acceptance_on_the_primitives_only() -> None:
    # decision 50: the gate's acceptance of each operator's text gives the
    # model's `verified` premise
    printed = _assumptions("verified_of_accepted", "StoreModel")
    lines = printed.split("Axioms:", 1)[1].splitlines()
    axioms = [line for line in lines if line and not line[0].isspace()]
    assert axioms, printed
    assert all(PRIMITIVE.match(axiom) for axiom in axioms), axioms


@pytest.mark.parametrize(
    "theorem", ["tenant_isolation", "own_decision_same", "tenant_isolation_same_steps"]
)
def test_tenant_isolation_assumes_nothing(theorem: str) -> None:
    # decision 52: runs whose other tenants differ, a step of A decided alike
    # on two catalogs with the same own part of A, and runs of the same steps
    assert "Closed under the global context" in _assumptions(theorem, "PipelineIsolation")


def test_participation_assumes_only_the_primitive_integers_and_arrays() -> None:
    # the check the gate's query proves gives the participation of every
    # entry kernel with what verify_certificate_sound gives
    printed = _assumptions("verify_certificate_participation", "GateParticipation")
    lines = printed.split("Axioms:", 1)[1].splitlines()
    axioms = [line for line in lines if line and not line[0].isspace()]
    assert axioms, printed
    assert all(PRIMITIVE.match(axiom) for axiom in axioms), axioms


@pytest.mark.parametrize(
    "theorem",
    [
        "tenant_declassification",
        "observe_steps_run",
        "run_observe_steps",
        "check_step_grants_nodup",
    ],
)
def test_tenant_isolation_with_grants_assumes_nothing(theorem: str) -> None:
    # runs with the same events of A, its steps and the grants to it
    assert "Closed under the global context" in _assumptions(theorem, "PipelineGrants")


def test_termination_assumes_nothing() -> None:
    assert "Closed under the global context" in _assumptions("term_check_sound", "GateTerm")


@pytest.mark.parametrize("theorem", ["accepted_zeroing", "zero_element_thread", "zero_plan_count"])
def test_the_effect_of_the_loops_that_clear_dynamic_shared_memory_assumes_nothing(
    theorem: str,
) -> None:
    # decision 49: the threads of a block store zero at every byte of the
    # launch's dynamic shared memory before the barrier
    assert "Closed under the global context" in _assumptions(theorem, "GateZero")


@pytest.mark.parametrize(
    "theorem,library",
    [("prologue_run", "GatePrologue"), ("launch_ni", "StoreLaunch"), ("launch_ni0", "StoreLaunch")],
)
def test_the_zeroing_prologue_in_the_launch_model_assumes_nothing(
    theorem: str, library: str
) -> None:
    # decision 50: each thread of a block stores zero over its share of the
    # shared variables before the barrier, and two runs of a launch from any
    # two shared memories agree on the bytes the launch may read
    assert "Closed under the global context" in _assumptions(theorem, library)


@pytest.mark.parametrize(
    "theorem,library",
    [
        ("type_default_sound", "GateSoundTransfer"),
        ("default_load_sound", "GateSoundTransfer"),
        ("refine_origins_sound", "GateSoundAccess"),
        ("access_guard_sound", "GateSoundAccess"),
        ("access_state_ok_sound", "GateSoundAccess"),
        ("access_ok_sound", "GateSoundAccess"),
    ],
)
def test_conditional_load_helpers_assume_nothing(theorem, library):
    assert "Closed under the global context" in _assumptions(theorem, library)


@pytest.mark.parametrize("theorem", ["constant_terms_tsum", "constant_terms_absent"])
def test_constant_substitution_helpers_assume_nothing(theorem):
    assert "Closed under the global context" in _assumptions(theorem, "GateSoundTransfer")


def test_range_memoization_preserves_results_without_axioms() -> None:
    for theorem in (
        "reg_range_exact",
        "sreg_range_exact",
        "atom_range_exact",
        "form_range_exact",
        "form_range_nat_exact",
        "rf_memo_equivalent",
    ):
        assert "Closed under the global context" in _assumptions(theorem, "GateSoundRange")


def test_projection_memoization_preserves_results_without_axioms():
    assert "Closed under the global context" in _assumptions(
        "project_range_exact", "GateSoundAccess"
    )


def test_short_circuit_containment_preserves_results_without_axioms() -> None:
    assert "Closed under the global context" in _assumptions("fact_le_same", "GateDomain")


def test_short_circuit_bound_checks_preserve_results_without_axioms() -> None:
    assert "Closed under the global context" in _assumptions("short_all_same", "GateDomain")


def test_selection_origin_refinement_assumes_nothing() -> None:
    assert "Closed under the global context" in _assumptions(
        "refine_selection_sound", "GateSoundTransfer"
    )


@pytest.mark.parametrize("theorem", ["clip_constraint_sound", "clip_join_state_sound"])
def test_widening_clipping_assumes_nothing(theorem) -> None:
    assert "Closed under the global context" in _assumptions(theorem, "GateSoundJoin")


def test_constant_selection_constraint_transport_is_identical() -> None:
    assert "Closed under the global context" in _assumptions("sel_refined_cons_same", "GateDomain")


def test_phase_slice_fixed_point_preserves_all_full_pass_results() -> None:
    assert "Closed under the global context" in _assumptions("phase_slice_exact", "GateBarrier")


def test_shared_value_participation_assumes_nothing() -> None:
    # The three facts of the PTX semantics are premises of the theorems
    # (`cta_family`, `linked_shared_schedule`), never axioms (decision 41).
    assert "Closed under the global context" in _assumptions("shared_check_sound", "GateShared")
    assert "Closed under the global context" in _assumptions(
        "linked_shared_schedule_sound", "StoreLink"
    )
    assert "Closed under the global context" in _assumptions(
        "accepted_front_linked_events", "StoreLink"
    )
