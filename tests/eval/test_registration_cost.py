"""Cold registration records distinguish actual checks, reuse, and timeouts."""

from unittest.mock import Mock

import pytest

from eval.rq4_cost import registration_cost as cost
from eval.rq4_cost.registration import Input
from opgate.binding import GRID_LIMIT, Binding
from opgate.gate import TIMEOUT_REASON, Accepted, Refused
from opgate.signature import Affine


@pytest.fixture(autouse=True)
def restore_library_environment(monkeypatch):
    monkeypatch.setenv("OPGATE_FORMAL", "/proof/library")


@pytest.fixture
def item():
    binding = Binding("kernel", (), (), (1, 1, 1), GRID_LIMIT, Affine(0))
    # without shared or local memory, the registered text is the text as written
    text = ".visible .entry kernel()\n{\nret;\n}\n"
    return Input(("model", "task", "cuda"), text, binding, "safe", None)


@pytest.mark.parametrize("decision", [Accepted(), Refused("not proved", ())])
def test_conclusive_samples_check_every_reuse_result(monkeypatch, item, decision):
    verify = Mock(return_value=decision)
    monkeypatch.setattr(cost, "verify", verify)
    row = cost._sample(item, "/proof/library", 7.0, 3)
    assert row["decision"] == ("accepted" if isinstance(decision, Accepted) else "refused")
    assert len(row["reuse"]) == 3
    assert all(sample["decision"] == row["decision"] for sample in row["reuse"])
    assert verify.call_count == 4
    verify.assert_called_with(item.ptx, [item.binding], timeout=7.0)


def test_timeouts_are_not_measured_as_cache_hits(monkeypatch, item):
    verify = Mock(return_value=Refused(f"{TIMEOUT_REASON} 7 seconds", ()))
    monkeypatch.setattr(cost, "verify", verify)
    row = cost._sample(item, "/proof/library", 7.0, 10)
    assert row["decision"] == "timeout"
    assert row["reuse"] == []
    verify.assert_called_once()


def test_changed_reuse_decision_fails_the_measurement(monkeypatch, item):
    monkeypatch.setattr(cost, "verify", Mock(side_effect=[Accepted(), Refused("changed", ())]))
    with pytest.raises(RuntimeError, match="decision changed"):
        cost._sample(item, "/proof/library", 7.0, 1)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"repeats": 0}, "positive"),
        ({"warm_repeats": 0}, "positive"),
        ({"timeout": 0}, "finite and positive"),
        ({"timeout": float("nan")}, "finite and positive"),
        ({"timeout": float("inf")}, "finite and positive"),
        ({"keys": []}, "at least one"),
        ({"keys": ["a", "a"]}, "duplicate"),
    ],
)
def test_invalid_settings_fail_before_creating_record(tmp_path, options, message):
    output = tmp_path / "record.json"
    with pytest.raises(ValueError, match=message):
        cost.measure(
            output,
            options.get("keys", ["model/task/cuda"]),
            repeats=options.get("repeats", 3),
            warm_repeats=options.get("warm_repeats", 10),
            timeout=options.get("timeout", 180),
        )
    assert not output.exists()


def test_unknown_kernel_fails_before_creating_record(tmp_path, monkeypatch, item):
    monkeypatch.setattr(cost.os, "sched_getaffinity", lambda _: {2, 3})
    monkeypatch.setattr(cost, "inputs", lambda: [item])
    output = tmp_path / "record.json"
    with pytest.raises(ValueError, match="unknown corpus kernels"):
        cost.measure(output, ["missing/task/cuda"])
    assert not output.exists()


def test_unpinned_measurement_fails_before_creating_record(tmp_path, monkeypatch):
    monkeypatch.setattr(cost.os, "sched_getaffinity", lambda _: {2, 3, 4})
    output = tmp_path / "record.json"
    with pytest.raises(ValueError, match="exactly two CPU cores"):
        cost.measure(output, ["model/task/cuda"])
    assert not output.exists()
