"""CPU references use task semantics, including cases the GPU may compute incorrectly."""

import numpy as np
import pytest

from eval.corpus import heldout_workloads as heldout
from eval.corpus.adapter import SYNTHESIZED, Role, read_raw, read_roles
from eval.corpus.run import TASK_SETS
from eval.corpus.workloads import workload


def test_workloads_cover_both_published_task_lists():
    assert heldout.TASKS == TASK_SETS["held-out-4"] | TASK_SETS["held-out-5"]


@pytest.mark.parametrize("task", sorted(heldout.TASKS))
@pytest.mark.parametrize("n", [0, 1, 33, 257])
def test_inputs_supply_each_reviewed_contract(task, n):
    w = workload(task, n, np.random.default_rng(7))
    for model in SYNTHESIZED.models:
        path = SYNTHESIZED.root / model / task / "signature.json"
        if not path.exists():
            continue  # this model did not contribute this task to the corpus
        roles = read_roles(model, task)
        assert set(roles.symbols) <= w.sizes.keys()
        for raw in read_raw(path):
            role = roles.arguments[raw.name]
            if role is Role.SIZE:
                assert raw.name in w.sizes
            elif role is Role.SCALAR:
                assert raw.name in w.scalars
            else:
                assert raw.name in w.arrays
                assert w.arrays[raw.name].dtype == np.dtype(raw.dtype)


def test_range_window_includes_later_peers_and_wraps_int64():
    arrays = {"ts": np.array([1, 2, 2, 5], np.int64), "values": np.array([3, 4, 7, -1], np.int64)}
    observed = heldout.reference("range_window_sum", arrays, {"range": 1}, {"n": 4})
    assert observed["out"].tolist() == [3, 14, 14, -1]
    arrays = {"ts": np.array([0, 0], np.int64), "values": np.array([2**63 - 1, 1], np.int64)}
    assert heldout.reference("range_window_sum", arrays, {"range": 0}, {"n": 2})[
        "out"
    ].tolist() == [-(2**63), -(2**63)]


def test_decimal_rounds_ties_away_from_zero_and_detects_overflow():
    a = {"values": np.array([-15, -5, 5, 15, 14, -14], np.int64)}
    assert heldout.reference("decimal_rescale", a, {"shift": -1}, {"n": 6})["out"].tolist() == [
        -2,
        -1,
        1,
        2,
        1,
        -1,
    ]
    a = {"values": np.array([2**63 - 1, -(2**63), 1], np.int64)}
    observed = heldout.reference("decimal_rescale", a, {"shift": 1}, {"n": 3})
    assert observed["out"].tolist() == [0, 0, 10]
    assert observed["overflow"].tolist() == [1, 1, 0]


def test_bitmap_slice_crosses_bytes_and_clears_padding():
    a = {"source": np.array([0b10110000, 0b11111101], np.uint8)}
    assert heldout.reference("bitmap_slice", a, {"offset": 5}, {"n": 6})["out"].tolist() == [
        0b101101
    ]


def test_bit_unpack_spans_words():
    a = {"packed": np.array([0x89ABCDEF, 0x01234567], np.uint32)}
    assert heldout.reference("bit_unpack", a, {"bits": 20}, {"n": 3})["out"].tolist() == [
        0xBCDEF,
        0x6789A,
        0x12345,
    ]


def test_month_arithmetic_clamps_leap_day():
    import datetime

    epoch = datetime.date(1970, 1, 1)
    a = {
        "days": np.array([(datetime.date(2020, 1, 31) - epoch).days], np.int32),
        "months": np.array([1], np.int32),
    }
    actual = heldout.reference("add_months", a, {}, {"n": 1})["out"][0]
    assert int(actual) == (datetime.date(2020, 2, 29) - epoch).days


def test_join_materialization_returns_matching_build_rows():
    arrays = {
        "build": np.array([2, 7, 7, 9], np.int64),
        "probe": np.array([7, 2, 5], np.int64),
        "offsets": np.array([0, 2, 3, 3], np.int32),
    }
    sizes = {"n": 3, "b": 4, "p": 3}
    actual = heldout.reference("join_materialize", arrays, {}, sizes)
    assert actual["out_build"].tolist() == [1, 2, 0]
    assert actual["out_probe"].tolist() == [0, 0, 1]
    wrong = actual | {"out_build": np.array([0, 1, 2], np.int32)}
    assert heldout.check("join_materialize", arrays, {}, sizes, wrong) is not None


def test_timestamp_floor_rounds_negative_nonmultiples_once():
    arrays = {"ts": np.array([-2_000_000, -1_000_001, -1, 0, 1, 1_000_001, 2_000_000], np.int64)}
    actual = heldout.reference("timestamp_floor", arrays, {"unit": 1_000_000}, {"n": 7})
    assert actual["out"].tolist() == [
        -2_000_000,
        -2_000_000,
        -1_000_000,
        0,
        0,
        1_000_000,
        2_000_000,
    ]


def test_string_integer_reference_handles_int64_edges_and_long_leading_zeros():
    texts = [
        b"-9223372036854775808",
        b"9223372036854775808",
        b"+",
        b" 1",
        b"-" + b"0" * 5000 + b"5",
    ]
    offsets = np.asarray([0, *np.cumsum([len(t) for t in texts])], np.int32)
    a = {"offsets": offsets, "chars": np.frombuffer(b"".join(texts), np.uint8)}
    actual = heldout.reference("string_to_int", a, {}, {"n": len(texts)})
    assert actual["out"].tolist() == [-(2**63), 0, 0, 0, -5]
    assert actual["valid"].tolist() == [0b10001]


def test_compaction_checks_duplicates_and_run_encoding_keeps_unused_output():
    a: dict[str, np.ndarray] = {
        "values": np.array([2, 2, 3], np.int64),
        "keep": np.ones(3, np.int8),
    }
    assert (
        heldout.check(
            "stream_compact",
            a,
            {},
            {"n": 3},
            {"count": np.array([3], np.int32), "out": np.array([3, 2, 2], np.int64)},
        )
        is None
    )
    assert (
        heldout.check(
            "stream_compact",
            a,
            {},
            {"n": 3},
            {"count": np.array([3], np.int32), "out": np.array([3, 3, 2], np.int64)},
        )
        is not None
    )
    a |= {"run_ends": np.full(3, -7, np.int32), "run_values": np.full(3, -11, np.int64)}
    actual = heldout.reference("run_end_encode", a, {}, {"n": 3})
    assert actual["run_ends"].tolist() == [2, 3, -7]
    assert actual["run_values"].tolist() == [2, 3, -11]
