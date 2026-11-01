"""The cold-query table's pure parts and the generation experiment's compile, without a GPU."""

import os
from dataclasses import replace

import pytest

from eval.corpus.adapter import directory_of
from eval.export.corpus import Kernel, Key
from eval.export.rq4 import SIZE_GROUPS, _percentile, _seconds, plans, schedule
from eval.rq4_cost.generation import COMPILED, compile_seconds, corpus_texts


def _kernel(key: Key, instructions: int = 10) -> Kernel:
    return Kernel(
        key=key,
        tasks="datakernelbench-core" if "_core_" in key[1] else "datakernelbench",
        cls="safe",
        baseline=None,
        check="",
        repaired=None,
        final="accepted",
        seconds=1.0,
        bytes=100,
        instructions=instructions,
        features=(),
        source=key[1],
    )


def test_plans_group_the_kernels_of_one_plan_file() -> None:
    aggregate = ("gpt-5.5", "q01_core_cuda_q1_aggregate_kernel", "nvcc")
    finalize = ("gpt-5.5", "q01_core_cuda_q1_finalize_kernel", "nvcc")
    full = ("gpt-5.5", "q01_cuda_q1_kernel6", "nvcc")
    same = (directory_of(*aggregate[:2]) / "plan.py").read_bytes()
    assert (directory_of(*finalize[:2]) / "plan.py").read_bytes() == same
    assert (directory_of(*full[:2]) / "plan.py").read_bytes() != same
    synthesized = replace(_kernel(aggregate), key=("opus", "q6_sum", "numba"), tasks="development")
    found = plans([_kernel(aggregate), synthesized, _kernel(full), _kernel(finalize)])
    assert [[k.key for k in p] for p in found] == [[aggregate, finalize], [full]]


def test_size_groups_cover_every_size_once() -> None:
    for size in (0, 99, 100, 999, 1_000, 4_999, 5_000, 1 << 20):
        matching = [g for g in SIZE_GROUPS if g.low <= size and (g.high is None or size < g.high)]
        assert len(matching) == 1


def test_percentiles_and_seconds() -> None:
    values = [float(v) for v in range(1, 102)]  # 1 to 101
    assert _percentile(values, 90) == 91.0
    assert _percentile(values, 99) == 100.0
    assert _seconds(4.94) == "4.9"
    assert _seconds(1186.6) == "1,187"


def test_compile_seconds_gives_the_corpus_text_with_its_own_triton_cache() -> None:
    key = ("sonnet", "q6_sum", "triton")
    before = os.environ.get("TRITON_CACHE_DIR")
    text, seconds = compile_seconds(key)
    cache = os.environ["TRITON_CACHE_DIR"]
    assert cache != before and not os.path.exists(cache)
    assert seconds > 0
    corpus = corpus_texts()[key].decode("ascii")
    # the same text up to the debug sections, which name the source's path
    assert "\t.file" in corpus
    assert text.split("\t.file")[0] == corpus.split("\t.file")[0]


def test_corpus_texts_hold_the_toolchains_that_compile_without_a_gpu() -> None:
    texts = corpus_texts()
    assert {toolchain for _, _, toolchain in texts} == set(COMPILED)
    assert ("sonnet", "q6_sum", "nvcc") in texts and ("sonnet", "q6_sum", "numba") not in texts


def test_schedule_takes_the_checks_in_order_on_the_first_free_thread() -> None:
    assert schedule([3.0, 1.0, 2.0], 1) == 6.0  # one thread: the sum
    assert schedule([3.0, 1.0, 2.0], 3) == 3.0  # a thread for each check: the longest
    # the third check starts at 1 on the second thread, the fourth at 3
    assert schedule([3.0, 1.0, 2.0, 2.0], 2) == 5.0
    with pytest.raises(ValueError):
        schedule([1.0], 0)
