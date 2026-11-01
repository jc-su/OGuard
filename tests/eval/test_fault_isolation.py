"""The fault-isolation experiment's pure parts: the wild module, the outcome
measures, and the Numba array ABI it launches the wild kernel with."""

import json

from eval.rq4_cost import fault_isolation as experiment
from eval.rq4_cost.fault_isolation import Completion


def test_the_wild_module_is_the_archived_wild_address_kernel(
    tmp_path, monkeypatch, raw_record
) -> None:
    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps(raw_record("e8_service_2026-09-29.json")))
    monkeypatch.setattr(experiment, "ARCHIVE", archive)
    text, name = experiment._wild_module()
    assert ".entry" in text
    assert name and name in text
    # the kernel reads a far index, which the gate refuses as written and an unchecked
    # launch faults on
    assert experiment.WILD_INDEX == 1 << 32


def test_correct_compares_every_completion_with_the_reference() -> None:
    expected = 1234.5
    assert experiment._correct([Completion(0.1, expected), Completion(0.2, expected)], expected)
    assert not experiment._correct(
        [Completion(0.1, expected), Completion(0.2, expected + 1.0)], expected
    )
    # a tenant that completed nothing is vacuously correct
    assert experiment._correct([], expected)


def test_correct_allows_a_relative_tolerance() -> None:
    expected = 1e9
    near = expected * (1 + 1e-10)
    far = expected * (1 + 1e-3)
    assert experiment._correct([Completion(0.0, near)], expected)
    assert not experiment._correct([Completion(0.0, far)], expected)


def test_max_gap_spans_the_start_the_completions_and_the_end() -> None:
    end = 6.0
    # a tenant that stops at 3.0 has a 3.0-second gap to the end
    stalled = [Completion(1.0, 1.0), Completion(2.0, 1.0), Completion(3.0, 1.0)]
    assert experiment._max_gap(stalled, end) == 3.0
    # a tenant that completes nothing has one gap of the whole window
    assert experiment._max_gap([], end) == end
    # evenly spaced completions have a small gap
    even = [Completion(t, 1.0) for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    assert experiment._max_gap(even, end) == 1.0


def test_after_counts_completions_from_the_fault_on() -> None:
    completions = [
        Completion(1.0, 1.0),
        Completion(2.9, 1.0),
        Completion(3.0, 1.0),
        Completion(5.0, 1.0),
    ]
    assert experiment._after(completions, 3.0) == 2
    assert experiment._after([], 3.0) == 0


def test_numba_array_abi_is_seven_values_with_the_pointer_fifth() -> None:
    abi = experiment._numba_array(0xDEAD0000, count=1, itemsize=8)
    assert abi == [0, 0, 1, 8, 0xDEAD0000, 1, 8]
