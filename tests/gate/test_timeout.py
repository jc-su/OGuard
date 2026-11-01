"""A check that does not finish in its time is a refusal.

`verify` gives each module `timeout` seconds; a check that runs longer proves
nothing, so the module is refused with a reason that says so, and the
executor raises `RefusedError` for it as for any other refusal.
"""

import subprocess
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import pytest

import opgate.gate as gate
from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from opgate.binding import GRID_LIMIT, Binding, Constant
from opgate.gate import TIMEOUT_REASON, Refused, verify
from opgate.signature import Affine


def test_a_check_past_its_time_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(gate, "_decisions", {})
    model, task, toolchain = "sonnet", "gather", "numba"
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    decision = verify(text, [binding_of(model, task, toolchain, text)], timeout=0.01)
    assert isinstance(decision, Refused)
    assert decision.reason.startswith(TIMEOUT_REASON)


@pytest.fixture
def processes(monkeypatch, tmp_path):
    (tmp_path / "GateParse.vo").write_bytes(b"test library")
    monkeypatch.setattr(gate, "formal_dir", lambda: tmp_path)
    monkeypatch.setattr(gate, "_decisions", {})
    proof = Mock(returncode=1)
    proof.communicate.return_value = ("", "not accepted")
    proof.communicate.return_value = (
        'OPGATE_REFUSAL (Some "not proved")\nOPGATE_UNPROVED []',
        "not accepted",
    )

    @contextmanager
    def coqc(source, directory):
        assert directory == tmp_path
        assert source.endswith(gate.CHECK_QUERY)
        yield proof

    monkeypatch.setattr(gate, "_coqc", coqc)
    return proof, tmp_path


def test_proposal_diagnosis_and_proof_share_one_process_and_deadline(monkeypatch, processes):
    proof, _ = processes
    clock = iter((100.0, 102.0))
    monkeypatch.setattr(gate.time, "monotonic", lambda: next(clock))
    assert gate.verify("", [], timeout=10) == Refused("not proved", ())
    proof.communicate.assert_called_once_with(timeout=8.0)


def test_a_refusal_lists_every_access_that_the_check_refuses(processes):
    proof, _ = processes
    proof.communicate.return_value = (
        'OPGATE_REFUSAL (Some "k: r: ld [%rd1]")\n'
        'OPGATE_UNPROVED ["r: ld [%rd1]";\n "s: st ""x"" [%rd2+4]"]',
        "not accepted",
    )
    assert gate.verify("", []) == Refused("k: r: ld [%rd1]", ("r: ld [%rd1]", 's: st "x" [%rd2+4]'))


def test_a_refusal_without_its_list_is_an_error(processes):
    proof, _ = processes
    proof.communicate.return_value = ('OPGATE_REFUSAL (Some "not proved")', "not accepted")
    with pytest.raises(RuntimeError, match="without a reason"):
        gate.verify("", [])


def test_verification_timeout_is_not_cached(processes):
    proof, _ = processes
    proof.communicate.side_effect = subprocess.TimeoutExpired("coqc", 1)
    timed_out = gate.verify("", [], timeout=1)
    assert isinstance(timed_out, Refused)
    assert timed_out.reason.startswith(TIMEOUT_REASON)
    proof.communicate.side_effect = None
    assert gate.verify("", [], timeout=1) == Refused("not proved", ())
    assert proof.communicate.call_count == 2


def test_cache_is_invalidated_by_rebuilt_library(processes):
    proof, directory = processes
    assert gate.verify("", []) == Refused("not proved", ())
    proof.returncode = 0
    assert gate.verify("", []) == Refused("not proved", ())
    (directory / "GateParse.vo").write_bytes(b"rebuilt library")
    assert gate.verify("", []) == gate.Accepted()
    assert proof.communicate.call_count == 2


def test_cached_decision_does_not_repack_unchanged_ptx(processes, monkeypatch):
    proof, _ = processes
    proof.returncode = 0
    export = Mock(wraps=gate.export_terms)
    monkeypatch.setattr(gate, "export_terms", export)
    assert gate.verify("exact PTX bytes", []) == gate.Accepted()
    assert gate.verify("exact PTX bytes", []) == gate.Accepted()
    export.assert_called_once()
    assert gate.verify("exact PTX bytes\n", []) == gate.Accepted()
    assert export.call_count == 2
    assert proof.communicate.call_count == 2


def test_cache_compares_kernel_names_and_printed_binding_values(processes):
    proof, _ = processes
    proof.returncode = 0
    binding = Binding("first", (), (("p", Constant(1)),), (1, 1, 1), GRID_LIMIT, Affine(0))
    assert gate.verify("", [binding]) == gate.Accepted()
    assert gate.verify("", [replace(binding)]) == gate.Accepted()
    assert proof.communicate.call_count == 1
    for changed in (
        replace(binding, kernel="second"),
        replace(binding, parameters=(("p", Constant(2)),)),
    ):
        gate.verify("", [changed])
    assert proof.communicate.call_count == 3


def test_a_library_change_during_verification_fails_closed(processes):
    proof, directory = processes

    def rebuild(**kwargs):
        (directory / "GateParse.vo").write_bytes(b"rebuilt while checking")
        return "", ""

    proof.returncode = 0
    proof.communicate.side_effect = rebuild
    with pytest.raises(RuntimeError, match="changed during verification"):
        gate.verify("", [])
    assert not gate._decisions


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_fails_before_starting_a_process(timeout, processes):
    proof, _ = processes
    with pytest.raises(ValueError, match="finite and positive"):
        gate.verify("", [], timeout=timeout)
    proof.communicate.assert_not_called()


def test_cancellation_is_not_recorded_as_a_decision(processes):
    from threading import Event

    proof, _ = processes
    cancel = Event()

    def wait(**kwargs):
        cancel.set()
        raise subprocess.TimeoutExpired("coqc", 0.1)

    proof.communicate.side_effect = wait
    with pytest.raises(InterruptedError, match="cancelled"):
        gate.verify("", [], timeout=10, cancel=cancel)
    assert not gate._decisions


def test_a_printed_acceptance_without_a_checked_theorem_is_an_error(processes):
    proof, _ = processes
    proof.communicate.return_value = ("= None : option string\n= true : bool", "Qed failed")
    with pytest.raises(RuntimeError, match="without a reason"):
        gate.verify("", [])
    assert not gate._decisions


def test_export_query_is_the_actual_complete_query(processes, monkeypatch):
    proof, directory = processes
    proof.returncode = 0
    queries = []

    @contextmanager
    def capture(source, selected):
        assert selected == directory
        queries.append(source)
        yield proof

    monkeypatch.setattr(gate, "_coqc", capture)
    assert gate.verify("", []) == gate.Accepted()
    assert queries == [gate.export_query("", [])]
    # one front end outside the kernel, checked once by the kernel; one proposal from it
    assert queries[0].count("text_front front_program text size mfs") == 1
    assert "Lemma lowered_front : text_front front text size mfs = lowered." in queries[0]
    assert queries[0].count("Eval vm_compute in propose_front lowered.") == 1
    assert "check_text_certificate_ok text size mfs certificate = true" in queries[0]
    assert queries[0].rstrip().endswith("Qed.")


@pytest.mark.parametrize("value", [True, 1.0, "1", b"1", None])
@pytest.mark.parametrize("cached", [False, True])
def test_malformed_numeric_binding_fails_before_query_or_cache(
    value, cached, processes, monkeypatch
):
    proof, _ = processes
    proof.returncode = 0
    binding = Binding("kernel", (), (("p", Constant(1)),), (1, 1, 1), GRID_LIMIT, Affine(0))
    if cached:
        assert gate.verify("", [binding]) == gate.Accepted()
    decisions = dict(gate._decisions)
    compiler = Mock(side_effect=AssertionError("a malformed binding reached Rocq"))
    export = Mock(side_effect=AssertionError("a malformed binding reached query export"))
    monkeypatch.setattr(gate, "_coqc", compiler)
    monkeypatch.setattr(gate, "export_query", export)
    with pytest.raises(TypeError, match="Rocq integer fields must be integers"):
        gate.verify("", [replace(binding, parameters=(("p", Constant(value)),))])
    compiler.assert_not_called()
    export.assert_not_called()
    assert gate._decisions == decisions


def test_transformation_rounds_share_one_budget_and_each_check_takes_what_remains(monkeypatch):
    clock = iter((100.0, 105.0, 560.0, 701.0))
    monkeypatch.setattr(gate.time, "monotonic", lambda: next(clock))
    checker = Mock(return_value=gate.Accepted())
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    budget = gate.VerificationBudget(600, checker=checker)
    assert budget("first", binding) == gate.Accepted()
    assert budget("second", binding) == gate.Accepted()
    assert budget("third", binding) == Refused(f"{TIMEOUT_REASON} 600 seconds", ())
    # no limit of its own: a check may run past `TIMEOUT` within the budget
    assert [call.kwargs["timeout"] for call in checker.call_args_list] == [595, 140]


def test_a_check_that_ends_with_the_budget_is_refused_for_the_budget():
    checker = Mock(return_value=Refused(f"{TIMEOUT_REASON} 599.999 seconds", ()))
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    budget = gate.VerificationBudget(600, checker=checker)
    decision = budget("first", binding)
    assert decision == Refused(f"{TIMEOUT_REASON} 600 seconds", ())
    assert gate.timed_out(decision)


def test_a_conclusive_refusal_keeps_its_reason():
    checker = Mock(return_value=Refused("kernel: not proved", ()))
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    budget = gate.VerificationBudget(600, checker=checker)
    assert budget("first", binding) == Refused("kernel: not proved", ())
    assert not gate.timed_out(Refused("kernel: not proved", ()))


def test_cancellation_stops_later_transformation_rounds():
    from threading import Event

    cancel = Event()
    checker = Mock(return_value=Refused("unproved", ()))
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    budget = gate.VerificationBudget(cancel=cancel, checker=checker)
    assert budget("first", binding) == Refused("unproved", ())
    cancel.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        budget("second", binding)
    assert checker.call_count == 1
