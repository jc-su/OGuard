"""Query-plan aggregation never treats configuration entries as whole queries."""

from dataclasses import replace

import pytest

from eval.export.corpus import Kernel
from eval.rq2_coverage.plan_support import summarize, totals


def entry(task: str, source: str, final="accepted", cls="safe") -> Kernel:
    return Kernel(
        ("deepseek-v4-flash", task, "triton"),
        "datakernelbench",
        cls,
        None,
        "",
        None,
        final,
        1.0,
        100,
        10,
        (),
        source,
    )


def test_plan_kernel_and_configuration_denominators_are_separate():
    kernels = [
        entry("q03_triton_filter", "q03_triton_filter"),
        entry("q03_triton_filter_2", "q03_triton_filter"),
        entry("q03_triton_sum", "q03_triton_sum", "timeout"),
        entry("q03_core_triton_sum", "q03_core_triton_sum"),
        entry("q06_triton_reduce", "q06_triton_reduce"),
    ]
    plans = summarize(kernels)
    full = [p for p in plans if p["level"] == "full"]
    q3 = next(p for p in full if p["query"] == "q03")
    assert q3["configuration_entries"] == 3
    assert q3["unique_extracted_kernels"] == 2
    assert not q3["all_configurations_accepted"]
    assert q3["timeout_configurations"] == 1
    result = totals(plans)
    assert result["full"]["represented_query_plans"] == 2
    assert result["full"]["all_extracted_configurations_accepted"] == 1
    assert result["full"]["configuration_entries"] == 4
    assert result["core"]["represented_query_plans"] == 1
    assert result["core"]["configuration_entries"] == 1


def test_reviewed_defect_is_distinct_from_incompleteness_of_all_safe_plan():
    plans = summarize(
        [
            entry("q03_triton_filter", "q03_triton_filter", "refused", "C"),
            entry("q06_triton_reduce", "q06_triton_reduce", "refused"),
        ]
    )
    result = totals(plans)["full"]
    assert result["represented_query_plans"] == 2
    assert result["all_entries_reviewed_safe"] == 1
    assert result["all_extracted_configurations_accepted"] == 0
    assert result["all_safe_plans_with_all_configurations_accepted"] == 0


def test_non_query_corpus_does_not_expand_query_denominator():
    kernel = entry("q03_triton_filter", "q03_triton_filter")
    assert summarize([replace(kernel, key=("sonnet", "projection", "numba"))]) == []


@pytest.mark.parametrize(
    "task,compiler", [("unrecognized", "triton"), ("q03_cuda_filter", "triton")]
)
def test_unknown_plan_or_compiler_fails(task, compiler):
    kernel = entry(task, task)
    with pytest.raises(ValueError):
        summarize([replace(kernel, key=(kernel.key[0], task, compiler))])


def test_stale_registration_is_rejected_before_creating_output(tmp_path, monkeypatch):
    from eval.rq2_coverage import plan_support

    def stale(_):
        raise ValueError("experiment sources input changed")

    monkeypatch.setattr(plan_support, "corpus", stale)
    output = tmp_path / "plans.json"
    with pytest.raises(ValueError, match="input changed"):
        plan_support.report(tmp_path / "registration.json", output)
    assert not output.exists()
