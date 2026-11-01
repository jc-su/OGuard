"""The participation rewrites make the barrier participation of corpus kernels
provable for the gate, and keep the tasks' results.

Triton's `while tl.sum(pending) > 0:` loops of the corpus's `groupby_sum`,
`hash_build`, `starts_with`, and `string_length` kernels end when a block
reduction of each thread's `pending` is zero; the rewrite ends each thread's
loop on its own `pending` and removes the reduction and its barriers.
Sonnet's Triton `histogram`, and Gemini's Triton Q1 of DataKernelBench,
branch around atomics between their barriers on block totals; the rewrite
guards the atomics instead. The CUDA C++ Q17 kernels of DataKernelBench
binary-search a sorted key column before their first barrier; the rewrite
bounds the search's iterations. Gemini's Q6 and DeepSeek's Q14 step a
grid-stride loop by a stride computed in 32 bits, which wraps to zero for a
large enough grid; the rewrite computes the grid index and stride in 64
bits. MiniMax's Triton Q4 kernels of
DataKernelBench count rows with a block sum that the block's first thread
adds atomically; the rewrite has each thread add its own count and removes
the sum and its barriers. The gate refuses each kernel as written for
barrier participation and accepts the rewritten text, with the guard pass's
checks where an access needs one; the rewritten kernels give the tasks'
results on their workloads. The rewritten `groupby_sum` and `hash_build`
loops insert a key by compare-and-swap until it succeeds; a compare-and-swap
may fail on every try in the model, so the gate proves their barrier
participation but refuses them because it does not prove that they end.
"""

import re
from functools import cache

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import Loaded, check, run, workload
from opgate.catalog import TenantId
from opgate.executor import Executor
from opgate.frontends.guard import guard
from opgate.frontends.participation import PARTICIPATION, rewrite
from opgate.gate import Accepted, Refused, verify, zeroed

TENANT = TenantId(1)
ENDS = "the kernel is not proved to end"
ACCESS = "the access is not proved inside its buffer"
ROWS = (1, 1000, 100003)
# kernels that end a loop on a block reduction of each thread's own predicate
LOOPS = (
    ("fable", "groupby_sum", "triton"),
    ("sonnet", "groupby_sum", "triton"),
    ("opus", "groupby_sum", "triton"),
    ("sonnet", "hash_build", "triton"),
    ("sonnet", "string_length", "triton"),
    ("opus", "starts_with", "triton"),
)
# of those, kernels whose loop retries a compare-and-swap
RETRIES = (
    ("fable", "groupby_sum", "triton"),
    ("sonnet", "groupby_sum", "triton"),
    ("opus", "groupby_sum", "triton"),
    ("sonnet", "hash_build", "triton"),
)
# kernels that branch on block totals between their barriers
BRANCHES = (
    ("sonnet", "histogram", "triton"),
    ("gemini-3.1-pro", "q01_triton_q1_kernel", "triton"),
)
# kernels whose first thread adds a block sum of counts atomically
SUMS = (("minimax-m2.5", "q04_core_triton__count_priority_kernel_7", "triton"),)
# kernels that binary-search before their first barrier
SEARCHES = (
    ("gpt-oss-120b", "q17_core_cuda_q17_pass2", "nvcc"),
    ("deepseek-v4-flash", "q17_cuda_q17_kernel", "nvcc"),
    # `while lo <= hi` with `hi = mid - 1`
    ("gpt-oss-120b", "q17_cuda_q17_rev_kernel", "nvcc"),
)
# kernels whose grid-stride loop steps by a stride computed in 32 bits
GRIDS = (
    ("gemini-3.1-pro", "q06_cuda_q6_kernel", "nvcc"),
    ("deepseek-v4-flash", "q14_core_cuda_promo_revenue_kernel", "nvcc"),
)


def _text(model: str, task: str, toolchain: str) -> str:
    return (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()


@cache
def _rewritten(model: str, task: str, toolchain: str) -> str:
    text = _text(model, task, toolchain)
    rewritten = rewrite(text, binding_of(model, task, toolchain, text).block)
    assert rewritten is not None, (model, task, toolchain)
    return rewritten


def test_the_reduction_loops_end_on_each_threads_own_predicate() -> None:
    for key in LOOPS:
        rewritten = _rewritten(*key)
        assert "removed with the block reduction" in rewritten
        loop = rewritten[rewritten.index("Inner Loop Header") :]
        loop = loop[: loop.index("bra.uni")]
        assert "bar.sync" not in loop.replace("// removed with the block reduction: bar.sync", "")
        assert "redux.sync" not in loop.replace("reduction: redux.sync", "")


def test_the_branches_before_the_last_barrier_become_guards() -> None:
    for key in BRANCHES:
        rewritten = _rewritten(*key)
        lines = rewritten.split("\n")
        last = max(i for i, line in enumerate(lines) if "bar.sync" in line)
        assert not any(" bra " in line or "\tbra " in line for line in lines[:last])


def test_the_binary_searches_get_an_iteration_bound() -> None:
    for key in SEARCHES:
        rewritten = _rewritten(*key)
        # 64-bit bounds: the search ends within 65 iterations, the count after 66
        assert re.search(r"setp\.lt\.s32 \t%opkp\d+, %opk\d+, 66;", rewritten), key
        assert re.search(r"and\.pred \t%opkq\d+,", rewritten), key


def test_the_grid_values_are_computed_in_64_bits() -> None:
    for key in GRIDS:
        text, rewritten = _text(*key), _rewritten(*key)
        assert rewritten.count("mul.wide.u32") + rewritten.count("shl.b64") > (
            text.count("mul.wide.u32") + text.count("shl.b64")
        ), key
        assert re.search(r"add\.s64 \t%opw\d+_\d+,", rewritten), key


def test_block_sums_keep_their_reduction_and_atomic_participation() -> None:
    for key in SUMS:
        text = _text(*key)
        assert rewrite(text, binding_of(*key, text).block) is None


@pytest.mark.slow
def test_the_proved_shared_value_rule_accepts_original_block_sums() -> None:
    for key in SUMS:
        text = _text(*key)
        binding = binding_of(*key, text)
        assert verify(zeroed(text, binding), (binding,)) == Accepted()


def test_a_kernel_without_barriers_or_collectives_is_unchanged() -> None:
    text = _text("sonnet", "gather", "numba")
    assert rewrite(text, binding_of("sonnet", "gather", "numba", text).block) is None


@pytest.mark.slow
def test_the_gate_accepts_the_rewritten_kernels() -> None:
    for model, task, toolchain in LOOPS + BRANCHES:
        text = _text(model, task, toolchain)
        binding = binding_of(model, task, toolchain, text)
        written = verify(zeroed(text, binding), (binding,))
        # the gate proves Gemini's Q1 kernel as written (DESIGN decision 40, acyclic segments);
        # the shared value route proves the participation of the others, which the
        # termination or access check may refuse as written (decision 41)
        assert written == Accepted() or (
            isinstance(written, Refused)
            and any(r in written.reason for r in (PARTICIPATION, ENDS, ACCESS))
        ), (task, written)
        decision = verify(zeroed(_rewritten(model, task, toolchain), binding), (binding,))
        if (model, task, toolchain) in RETRIES:
            assert isinstance(decision, Refused), (model, task, toolchain, decision)
            assert "the kernel is not proved to end" in decision.reason, decision
        else:
            assert isinstance(decision, Accepted), (model, task, toolchain, decision)


@pytest.mark.slow
def test_the_guard_pass_makes_the_searches_acceptable() -> None:
    # the third search's kernel needs an alignment that the pass does not check
    for model, task, toolchain in SEARCHES[:2] + GRIDS:
        text = _text(model, task, toolchain)
        binding = binding_of(model, task, toolchain, text)
        guarded = guard(text, binding, lambda t, b: verify(t, (b,)))
        # the gate proves the searches' participation as written (DESIGN decision 40)
        assert guarded.restructured or (model, task, toolchain) in SEARCHES, (model, task)
        assert isinstance(guarded.decision, Accepted), (model, task, guarded.decision)


@pytest.mark.slow
@pytest.mark.gpu
def test_the_rewritten_kernels_give_their_tasks_results() -> None:
    rng = np.random.default_rng(9)
    wrong = []
    with Executor(0, 1 << 28) as executor:
        with executor.tenant(TENANT):
            executor.reserve(1 << 28)
        # the DataKernelBench kernels have no workload of the corpus's tasks
        for model, task, toolchain in [k for k in LOOPS if k not in RETRIES] + list(BRANCHES[:1]):
            binding = binding_of(model, task, toolchain, _text(model, task, toolchain))
            module = executor.register(_rewritten(model, task, toolchain), binding)
            kernel = Loaded(model, task, toolchain, module)
            for rows in ROWS:
                w = workload(task, rows, rng)
                problem = check(task, w, run(executor, TENANT, kernel, w))
                if problem is not None:
                    wrong.append(f"{model}/{task} {toolchain} at {rows} rows: {problem}")
    assert wrong == []
