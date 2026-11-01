"""Corpus references cover exact string and Bloom outputs and empty allocations."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from eval.corpus import workloads
from eval.corpus.workloads import Workload, check, workload
from opgate.catalog import BufferId, TenantId
from opgate.executor import Executor


@pytest.mark.parametrize("rows", [0, 1, 4])
def test_string_equal_covers_empty_and_mismatched_strings(rows):
    w = workload("string_equal", rows, np.random.default_rng(1))
    expected = np.array([1, 0, 0, 1][:rows], dtype=np.int8)
    assert check("string_equal", w, {"match": expected}) is None
    for prefix in ("a", "b"):
        offsets, chars = w.arrays[f"{prefix}_offsets"], w.arrays[f"{prefix}_chars"]
        assert len(offsets) == rows + 1
        assert offsets[0] == 0 and offsets[-1] == len(chars)
        assert np.all(offsets[1:] >= offsets[:-1])
    if rows:
        wrong = expected.copy()
        wrong[0] = 0
        assert check("string_equal", w, {"match": wrong}) is not None
        wrong[0] = 2
        assert check("string_equal", w, {"match": wrong}) is not None


def _bloom_positions(key, words):
    # Independent scalar arithmetic: Python integers do not overflow.
    return [
        (((int(key) * multiplier) & ((1 << 64) - 1)) >> 32) % (32 * words)
        for multiplier in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F)
    ]


def _bloom_bits(keys, words):
    result = [0] * words
    for key in keys:
        for position in _bloom_positions(key, words):
            result[position // 32] |= 1 << (position % 32)
    return np.array(result, dtype=np.uint32)


@pytest.mark.parametrize("words", [1, 2, 64])
@pytest.mark.parametrize("rows", [0, 1, 9])
def test_bloom_reference_matches_wrapping_scalar_arithmetic(rows, words):
    keys = np.array([0, -1, -(2**63), 2**63 - 1, 1, -2, 0, -1, 123], dtype=np.int64)[:rows]
    sizes = {"n": rows, "words": words}
    build = Workload({"keys": keys, "bits": np.zeros(words, dtype=np.uint32)}, {}, sizes)
    expected = _bloom_bits(keys, words)
    assert check("bloom_build", build, {"bits": expected}) is None
    wrong = expected.copy()
    wrong[0] ^= np.uint32(1)
    assert check("bloom_build", build, {"bits": wrong}) is not None

    bits = _bloom_bits(keys[::2], words)
    probe = Workload({"keys": keys, "bits": bits}, {}, sizes)
    match = np.array(
        [
            all(int(bits[p // 32]) & (1 << (p % 32)) for p in _bloom_positions(key, words))
            for key in keys
        ],
        dtype=np.int8,
    )
    assert check("bloom_probe", probe, {"match": match}) is None
    if rows:
        wrong = match.copy()
        wrong[0] = 2
        assert check("bloom_probe", probe, {"match": wrong}) is not None
        wrong[0] = -1
        assert check("bloom_probe", probe, {"match": wrong}) is not None
        wrong[0] = 1 - match[0]
        assert check("bloom_probe", probe, {"match": wrong}) is not None


@pytest.mark.parametrize("rows", [0, 1, 32])
def test_bloom_workload_initialization_and_cpu_probe_table(rows):
    build = workload("bloom_build", rows, np.random.default_rng(13))
    probe = workload("bloom_probe", rows, np.random.default_rng(13))
    words = build.sizes["words"]
    assert words >= 1 and words <= 2**26 and words & (words - 1) == 0
    assert build.arrays["keys"].dtype == np.dtype("int64")
    assert build.arrays["keys"].shape == (rows,)
    assert build.arrays["bits"].dtype == np.dtype("uint32")
    assert build.arrays["bits"].shape == (words,)
    assert not np.any(build.arrays["bits"])
    assert np.array_equal(probe.arrays["keys"], build.arrays["keys"])
    assert np.array_equal(probe.arrays["bits"], _bloom_bits(probe.arrays["keys"][::2], words))
    assert probe.arrays["match"].dtype == np.dtype("int8")
    assert probe.arrays["match"].shape == (rows,)
    if rows >= 4:
        assert set(build.arrays["keys"][:4]) == {0, -1, -(2**63), 2**63 - 1}


def test_bloom_probe_requires_both_bits():
    keys = np.array([1], dtype=np.int64)
    first, second = _bloom_positions(1, 64)
    assert first != second
    bits = np.zeros(64, dtype=np.uint32)
    bits[first // 32] |= np.uint32(1 << (first % 32))
    probe = Workload({"keys": keys, "bits": bits}, {}, {"n": 1, "words": 64})
    assert check("bloom_probe", probe, {"match": np.array([0], dtype=np.int8)}) is None
    assert check("bloom_probe", probe, {"match": np.array([1], dtype=np.int8)}) is not None


def test_bloom_hashes_support_the_maximum_word_bound_without_allocating_the_filter():
    keys = np.array([-(2**63), -1, 0, 1, 2**63 - 1], dtype=np.int64)
    positions = workloads._bloom_positions(keys, 2**26)
    expected = np.array([_bloom_positions(key, 2**26) for key in keys], dtype=np.uint64)
    assert np.array_equal(np.column_stack(positions), expected)


@pytest.mark.parametrize("words", [-1, 0, 3, 2**27])
def test_bloom_reference_rejects_invalid_word_counts(words):
    with pytest.raises(ValueError, match="power of two"):
        workloads._bloom_bits(np.array([], dtype=np.int64), words)


@pytest.mark.parametrize("count", [0, 1])
def test_output_decoding_respects_logical_count_and_releases_buffers(monkeypatch, count):
    executor = Mock(spec=Executor)
    executor.tenant.return_value = nullcontext()
    executor.read.return_value = b"\x00"  # minimum physical allocation for an empty array
    buffer = BufferId(TenantId(1), 1)
    prepared = SimpleNamespace(
        launch=Mock(), buffers=(buffer,), outputs={"out": (buffer, np.dtype("float64"), count)}
    )
    monkeypatch.setattr(workloads, "prepare", lambda *args: prepared)
    if count == 0:
        result = workloads.run(executor, TenantId(1), Mock(), Mock())
        assert result["out"].shape == (0,) and result["out"].dtype == np.dtype("float64")
    else:
        with pytest.raises(ValueError):
            workloads.run(executor, TenantId(1), Mock(), Mock())
    executor.free.assert_called_once_with(buffer)
