"""Replay historical decision-table acceptances through the executor.

Each kernel that the decision table records as accepted, of a task in
`WORKLOADS`, runs as one step of one tenant, on inputs of 1, 1000, and
100003 rows that meet its task's contract, and its outputs agree with a
NumPy reference (`eval/corpus/workloads.py`). That module builds no inputs
for the other tasks, which include the TPC-H plans of DataKernelBench, so
their accepted kernels do not run here. The gate's participation and
termination checks became stricter after the table recorded its
acceptances (DESIGN decision 40) and refuse some of them; such a kernel
does not run, and a refusal for any other reason fails the test. The
catalog checks every allocation and the launch, and the executor builds
each parameter from the checked record. The gate verifies the kernels on
`WORKERS` threads first; registration then reads the recorded decisions.
This selection is not all acceptances of the latest checker on the
expanded corpus; dedicated records retain those scoped replays.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS, WORKERS, read_table
from eval.corpus.workloads import check, corpus_kernel, run, workload_of
from opgate.catalog import TenantId
from opgate.executor import Executor
from opgate.gate import Refused, verify, zeroed

TENANT = TenantId(1)
ROWS = (1, 1000, 100003)
# the tasks whose inputs and reference `eval/corpus/workloads.py` builds
WORKLOADS = frozenset(
    {
        "bloom_build",
        "bloom_probe",
        "days_to_year",
        "dictionary_decode",
        "filter_select",
        "gather",
        "groupby_sum",
        "hash_build",
        "hash_probe",
        "histogram",
        "merge_sorted",
        "min_max",
        "projection",
        "q6_sum",
        "radix_scatter",
        "rolling_sum",
        "run_end_decode",
        "scatter_by_offsets",
        "segmented_sum",
        "sorted_contains",
        "starts_with",
        "string_contains",
        "string_equal",
        "string_length",
        "valid_rows",
    }
)

# the reasons of the rules that refuse some of the table's acceptances
STRICTER = (
    "CTA barrier participation is not proved",
    "warp collective participation is not proved",
    "the kernel is not proved to end",
)


@pytest.mark.slow
@pytest.mark.gpu
def test_every_accepted_kernel_gives_its_tasks_result() -> None:
    recorded = sorted(
        key
        for key, row in read_table().items()
        if row.expected == "accepted" and row.task in WORKLOADS
    )
    texts = [(directory_of(m, t) / TOOLCHAINS[c]).read_text() for m, t, c in recorded]
    bindings = [binding_of(m, t, c, text) for (m, t, c), text in zip(recorded, texts, strict=True)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        decisions = list(
            pool.map(
                lambda text, binding: verify(zeroed(text, binding), [binding]), texts, bindings
            )
        )
    # the stricter participation and termination rules refuse some recorded acceptances
    refused = {
        key: decision.reason
        for key, decision in zip(recorded, decisions, strict=True)
        if isinstance(decision, Refused)
    }
    assert all(any(r in reason for r in STRICTER) for reason in refused.values()), refused
    accepted = [key for key in recorded if key not in refused]
    rng = np.random.default_rng(1)
    wrong = []
    with Executor(0, 1 << 28) as executor:
        with executor.tenant(TENANT):
            executor.reserve(1 << 28)
        for model, task, toolchain in accepted:
            kernel = corpus_kernel(executor, model, task, toolchain)
            for rows in ROWS:
                w = workload_of(executor, TENANT, model, task, rows, rng)
                problem = check(task, w, run(executor, TENANT, kernel, w))
                if problem is not None:
                    wrong.append(f"{model}/{task} {toolchain} at {rows} rows: {problem}")
    assert wrong == []
