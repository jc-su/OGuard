"""The guard pass gives a loop that the gate does not prove ends an iteration limit.

A thread retries a compare-and-swap until it succeeds, which it does
because another thread's swap succeeded; the gate's model gives an atomic
an arbitrary result, so it refuses the loop as not proved to end. The pass
counts the loop's iterations from its head and leaves the loop, storing 1
to the status buffer, when the count reaches the sum of the buffers' extents
and the grid's thread count (DESIGN decision 43). The gate accepts the
limited texts of hash insertions, a group-by, atomic minimum and maximum of
doubles, and a Bloom filter's loop over its seeds; run on their tasks'
workloads, they give the tasks' results and report no failed check.
Sonnet's CUDA C++ `min_max` reduces a block in shared memory under a branch
that the barrier rewrite replaces by guards, which also guard the loads the
branch skips, so that a load past the shared array does not run and fail
its check. A hash insertion into a table without a free slot, which as
written never ends, reports a failed check instead. The gate refuses a
limited text whose counter does not move, is set to 0 inside the loop, or
whose limit does not leave the loop.
"""

import re
from functools import cache

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import Loaded, check, prepare, run, workload
from opgate.binding import Base, Size
from opgate.catalog import Step, TenantId
from opgate.executor import Executor
from opgate.frontends.guard import ENDS, LIMIT, STATUS, Guarded, applies, guard, reports
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.hosts import CheckFailedError, execute

TENANT = TenantId(1)
ROWS = (1, 1000, 100003)
LIMITED = (
    # a probe whose compare-and-swap claims a slot, inlined from nvcc's atomicCAS
    ("sonnet", "hash_build", "nvcc"),
    # a probe that leaves on each thread's own predicates after the barrier rewrite
    ("sonnet", "groupby_sum", "triton"),
    # atomic minimum and maximum of doubles by compare-and-swap, in Numba
    ("opus", "min_max", "numba"),
    # the same in CUDA C++, after a block reduction that the barrier rewrite guards
    ("sonnet", "min_max", "nvcc"),
    # a probe that a thread enters past the loop's head
    ("opus", "hash_build", "numba"),
    # a loop over a constant tuple of seeds
    ("opus", "bloom_build", "numba"),
)


@cache
def _limited(model: str, task: str, toolchain: str) -> Guarded:
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    decision = verify(zeroed(text, binding), (binding,))
    assert isinstance(decision, Refused) and ENDS in decision.reason, (model, task, decision)
    return guard(text, binding, lambda t, b: verify(t, (b,)))


def test_the_pass_applies_to_a_loop_that_the_gate_does_not_prove_ends() -> None:
    edge = "the check does not prove that the loop of this back edge ends: @%p2 bra $L__BB1_2"
    assert applies(f"kernel: {ENDS}: {edge}")
    assert not applies(f"kernel: {ENDS}: its loops are not nested or start in the first block")


@pytest.mark.slow
def test_the_limited_kernels_are_accepted() -> None:
    for key in LIMITED:
        guarded = _limited(*key)
        assert isinstance(guarded.decision, Accepted), (key, guarded.decision)
        assert guarded.limited, key
        assert reports(guarded.binding), key
        # the limit's parameter holds the sum of the extents of the kernel's buffers
        extents = [
            p.span.extent
            for n, p in guarded.binding.parameters
            if isinstance(p, Base) and n != STATUS
        ]
        limit = dict(guarded.binding.parameters)[LIMIT]
        assert isinstance(limit, Size), key
        assert limit.value.constant == sum(e.constant for e in extents), key
        terms: dict[int, int] = {}
        for extent in extents:
            for symbol, coefficient in extent.terms:
                terms[symbol] = terms.get(symbol, 0) + coefficient
        assert dict(limit.value.terms) == terms, key
        # each limited loop increments its counter at its head and leaves at the limit
        counters = re.findall(r"add\.u64 \t(%og64_\d+), \1, 1;", guarded.text)
        assert len(counters) == len(guarded.limited), key


def _mutated(text: str, mutation: str) -> str:
    """`text` with its one limited loop changed so that the limit does not end it."""
    (counter,) = re.findall(r"add\.u64 \t(%og64_\d+), \1, 1;", text)
    reset = f"\tmov.u64 \t{counter}, 0;\n"
    head = re.search(rf"{re.escape(reset)}([$\w]+):\n", text)
    leave = re.search(r"\t@(%ogp\d+) bra \t(\$og_exit_\w+);\n", text)
    assert head is not None and leave is not None
    match mutation:
        case "the counter does not move":
            return text.replace(f"{counter}, {counter}, 1;", f"{counter}, {counter}, 0;")
        case "the counter is set to 0 at the head":
            label = f"{head.group(1)}:\n"
            return text.replace(reset + label, label + reset)
        case "the limit returns to the head":
            return text.replace(leave.group(0), f"\t@{leave.group(1)} bra \t{head.group(1)};\n")
        case "the limit is not tested":
            return text.replace(leave.group(0), "")
        case _:
            raise ValueError(mutation)


@pytest.mark.slow
def test_a_limit_that_does_not_end_the_loop_is_refused() -> None:
    guarded = _limited("sonnet", "hash_build", "nvcc")
    assert isinstance(guarded.decision, Accepted)
    for mutation in (
        "the counter does not move",
        "the counter is set to 0 at the head",
        "the limit returns to the head",
        "the limit is not tested",
    ):
        text = _mutated(guarded.text, mutation)
        assert text != guarded.text, mutation
        decision = verify(zeroed(text, guarded.binding), (guarded.binding,))
        assert isinstance(decision, Refused) and ENDS in decision.reason, (mutation, decision)


@pytest.mark.slow
@pytest.mark.gpu
def test_limited_kernels_give_their_tasks_results() -> None:
    # `run` raises CheckFailedError if a limit or a check of a safe kernel fails
    rng = np.random.default_rng(7)
    wrong = []
    with Executor(0, 1 << 28) as executor:
        with executor.tenant(TENANT):
            executor.reserve(1 << 28)
        for model, task, toolchain in LIMITED:
            guarded = _limited(model, task, toolchain)
            module = executor.register(guarded.text, guarded.binding)
            kernel = Loaded(model, task, toolchain, module)
            for rows in ROWS:
                w = workload(task, rows, rng)
                problem = check(task, w, run(executor, TENANT, kernel, w))
                if problem is not None:
                    wrong.append(f"{model}/{task} {toolchain} at {rows} rows: {problem}")
    assert wrong == []


@pytest.mark.slow
@pytest.mark.gpu
def test_a_probe_of_a_full_table_reports_a_failed_check() -> None:
    n = 64
    guarded = _limited("sonnet", "hash_build", "nvcc")
    w = workload("hash_build", n, np.random.default_rng(4))
    # no slot is empty (-1), so the compare-and-swap never succeeds: as written,
    # every thread probes forever
    w.arrays["table_keys"][:] = 0
    with Executor(0, 1 << 24) as executor, executor.tenant(TENANT):
        executor.reserve(1 << 24)
        module = executor.register(guarded.text, guarded.binding)
        kernel = Loaded("sonnet", "hash_build", "nvcc", module)
        prepared = prepare(executor, TENANT, kernel, w)
        with pytest.raises(CheckFailedError, match=re.escape(guarded.binding.kernel)):
            execute(executor, executor, Step(TENANT, (prepared.launch,)))
        table, _ = (prepared.outputs[name][0] for name in ("table_keys", "table_rows"))
        assert not np.frombuffer(executor.read(TENANT, table), dtype=np.int64).any()
        for buffer in prepared.buffers:
            executor.free(buffer)
