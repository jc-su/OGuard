"""The guard pass checks the accesses that the gate does not prove when they run.

Kernels that the gate refuses for an access it does not prove are accepted
with the pass's checks of global and shared accesses, with an extent
parameter appended for each guarded buffer and then the status buffer, to
which each check stores 1 when it fails: also when the access is in a
device function that the kernel calls, which the pass inlines, when it is
one of many that a round checks at once, which the gate lists in one
refusal, when its shared address is
swizzled, and when Numba declares its shared array `.visible`; a weakened
check of such an array is refused. The driver compiles each guarded text. Run on their tasks'
workloads, the guarded kernels give the tasks' results and report no failed
check, since a safe kernel's accesses pass their checks at the addresses
they name; so do kernels with every access checked (`guard_every`), which
the gate accepts. Sonnet's CUDA C++ `filter_select`, a defect whose
position counter a tenant can start at `n`, stores past its `sel` buffer
into the next buffer of the pool, `count`; guarded, the store does not run,
`count` holds the counter's value, and the launch reports the failed check,
while a launch whose counter starts at zero reports none. The gate also
proves that a kernel ends (DESIGN decision 40). It refuses three guarded
texts for that reason alone, after the checks make every access provable:
the Triton loops of Haiku's `dictionary_decode` and `rolling_sum` reach their
head from two back edges, a shape that the termination check does not take,
and the evaluator of the search that the pass inlines into Qwen's Q15 bounds
its 32-bit midpoint by the bounds' ranges, which do not keep it inside the
width.
"""

import re
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from eval.corpus.adapter import binding_of, directory_of
from eval.corpus.run import TOOLCHAINS
from eval.corpus.workloads import Loaded, check, prepare, run, workload
from opgate.binding import GRID_LIMIT, Binding, Constant, Size
from opgate.catalog import Step, TenantId
from opgate.executor import Executor
from opgate.frontends.guard import STATUS, UNPROVED, Guarded, guard, guard_every, reports
from opgate.gate import Accepted, Refused, verify, zeroed
from opgate.hosts import CheckFailedError, execute
from opgate.signature import Affine

TENANT = TenantId(1)
ROWS = (1, 1000, 100003)
# safe kernels that the gate refuses for an access that it does not prove; the
# tasks of the first three have workloads. (Haiku's Triton `dictionary_decode`
# left the list when the gate proved its accesses as written, decision 42.)
# (Haiku's Numba `bloom_build`, which
# the pass also makes acceptable, sets bits by addition and gives a wrong
# filter with or without its checks.)
SAFE = (
    ("haiku", "bloom_probe", "numba"),
    ("haiku", "rolling_sum", "triton"),
    ("sonnet", "rolling_sum", "triton"),
    ("opus", "tile_reverse", "nvcc"),
    ("haiku", "string_reverse", "nvcc"),
    # shared memory: a pointer that a loop advances through a shared array
    ("deepseek-v4-flash", "q05_core_cuda_q5_kernel", "nvcc"),
    # calls of device functions, which the pass inlines: an atomic addition, and
    # a binary search that loads from a buffer and returns from inside its loop
    ("haiku", "bucket_histogram", "nvcc"),
    ("qwen3.5-397b", "q15_cuda_q15_kernel", "nvcc"),
    # more refused accesses than single rounds reach
    ("minimax-m2.5", "q14_core_triton__promo_revenue_kernel_3", "triton"),
    # a shared store at an address that Triton's layout swizzles with `xor`
    ("fable", "block_sort", "triton"),
    # nvcc's atomicAdd of a double in shared memory, through a generic address
    # that `cvta.shared` makes in a scope of its own, one scope per array
    ("gpt-5.5", "q01_cuda_q1_kernel_generic", "nvcc"),
    # a store to a Numba shared array, which the module declares `.visible`
    ("sonnet", "masked_sum", "numba"),
)
# the guarded text of Qwen's Q15, which the gate refuses because it does not prove
# that the inlined binary search ends, gets the search's iteration bound
# (DESIGN decision 39) and is accepted
RUN = SAFE[:3]
# kernels with a workload whose every access the pass checks
EVERY = (
    ("sonnet", "gather", "numba"),
    ("fable", "q6_sum", "triton"),
    ("haiku", "rolling_sum", "triton"),
)


# kernels whose inlined callees take a double as a 64-bit integer, an immediate
# as a double, or a 32-bit parameter into a 64-bit register
MOVED = (
    ("fable", "min_max", "nvcc"),
    ("opus", "min_max", "nvcc"),
    ("fable", "groupby_sum", "numba"),
    ("opus", "segmented_sum", "numba"),
)


def _failure_stores(text: str) -> int:
    """The stores of 1 to the status buffer, one for each check."""
    return len(re.findall(r"@%ogp\d+ st\.global\.u32 \t\[%og64_\d+\], 1;", text))


def test_the_checks_store_to_the_status_buffer_when_they_fail() -> None:
    for model, task, toolchain in MOVED + EVERY:
        text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
        every, binding = guard_every(text, binding_of(model, task, toolchain, text))
        assert reports(binding), (model, task)
        assert binding.parameters[-1][0] == STATUS
        entry = re.search(rf"\.entry\s+{re.escape(binding.kernel)}\s*\((.*?)\)", every, re.S)
        assert entry is not None, (model, task)
        assert re.findall(r"\.param\s+\.\w+\s+(\w+)", entry.group(1))[-1] == STATUS
        # each check runs its access or, when it fails, stores to the status buffer
        stores = _failure_stores(every)
        assert stores > 0 and 2 * stores == every.count("@%ogp"), (model, task)


def test_a_checked_load_leaves_zero_in_its_registers_when_its_check_fails() -> None:
    # each load runs into a register of the pass that holds zero before it, which
    # the destination copies after it: the kernel stores the destination, which
    # would otherwise hold what the hardware left there (DESIGN decision 46)
    model, task, toolchain = EVERY[0]
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    every, binding = guard_every(text, binding_of(model, task, toolchain, text))
    loads = re.findall(r"@%ogp\d+ ld\.[\w.]+ \t(%og(\d+)_\d+), ", every)
    assert loads
    for register, bits in loads:
        assert every.count(f"mov.b{bits} \t{register}, 0;") == 1, register
        assert re.search(rf"mov\.b{bits} \t%\w+, {register};", every), register
    assert isinstance(verify(zeroed(every, binding), (binding,)), Accepted)


def test_ptxas_assembles_the_texts_with_every_access_checked() -> None:
    ptxas = shutil.which("ptxas")
    assert ptxas is not None
    with tempfile.TemporaryDirectory() as directory:
        for model, task, toolchain in MOVED:
            text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
            every, _ = guard_every(text, binding_of(model, task, toolchain, text))
            path = Path(directory) / "every.ptx"
            path.write_text(every)
            result = subprocess.run(
                [ptxas, "-arch=sm_86", "-o", "/dev/null", str(path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert result.returncode == 0, (model, task, toolchain, result.stderr[-500:])


def test_the_gate_lists_every_access_that_its_walk_refuses() -> None:
    # minimax's Triton Q14 refuses more accesses than single rounds reach; the
    # refusal names the first, and lists it with the others
    model, task, toolchain = SAFE[8]
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    decision = verify(zeroed(text, binding), (binding,))
    assert isinstance(decision, Refused)
    listed = [item for item in decision.unproved if any(p in item for p in UNPROVED)]
    assert len(set(listed)) > 1, decision
    assert any(decision.reason.endswith(item) for item in listed), decision


UNWRITTEN = """.version 8.0
.target sm_80
.address_size 64

.visible .entry k(
	.param .u64 k_param_0
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<5>;

	mov.u32 	%r1, %tid.x;
	setp.eq.s32 	%p1, %r1, 0;
	@%p1 bra 	$L__BB0_2;
	mov.u32 	%r2, 1;
$L__BB0_2:
	add.s32 	%r3, %r2, %r1;
	@%p1 mov.u32 	%r4, 2;
	add.s32 	%r3, %r3, %r4;
	ret;
}
"""


def test_one_round_zeroes_every_register_that_a_path_reads_before_it_writes() -> None:
    # the path that takes the branch reads %r2 unwritten, and every path reads
    # %r4, which only an instruction under a predicate writes
    binding = Binding("k", (), (("k_param_0", Constant(1)),), (1, 1, 1), GRID_LIMIT, Affine(0))
    named = "k: an instruction may read the register %r2 before an instruction writes it"
    decisions: list[Accepted | Refused] = [Refused(named, ()), Accepted()]
    texts: list[str] = []

    def check(text: str, bound: Binding) -> Accepted | Refused:
        texts.append(text)
        return decisions[len(texts) - 1]

    guarded = guard(UNWRITTEN, binding, check)
    assert guarded.decision == Accepted() and len(texts) == 2
    assert "mov.b32 \t%r2, 0;" in guarded.text and "mov.b32 \t%r4, 0;" in guarded.text
    assert re.findall(r"mov\.b\d+ \t(%\w+), 0;", guarded.text) == ["%r2", "%r4"]


@cache
def _guarded(model: str, task: str, toolchain: str) -> Guarded:
    text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
    binding = binding_of(model, task, toolchain, text)
    assert not isinstance(verify(zeroed(text, binding), (binding,)), Accepted)
    return guard(text, binding, lambda t, b: verify(t, (b,)))


@pytest.mark.slow
def test_the_guarded_kernels_are_accepted() -> None:
    for key in SAFE:
        guarded = _guarded(*key)
        assert isinstance(guarded.decision, Accepted), (key, guarded.decision)
        assert guarded.guarded
        appended = [p for n, p in guarded.binding.parameters if n.startswith("opgate_extent_")]
        assert all(isinstance(p, Size) for p in appended)
        assert all(f"opgate_extent_{i}" in guarded.text for i in range(len(appended)))
        assert reports(guarded.binding), key
        stores = _failure_stores(guarded.text)
        assert stores > 0 and 2 * stores == guarded.text.count("@%ogp"), key


@pytest.mark.slow
def test_a_weakened_check_of_a_numba_shared_array_is_refused() -> None:
    # the check keeps the store's offset in [0, 248] of the 256-byte array;
    # one that lets the offset reach 256 does not keep the store inside it
    guarded = _guarded("sonnet", "masked_sum", "numba")
    assert isinstance(guarded.decision, Accepted), guarded.decision
    bound = "setp.le.s64 \t%ogp2, %og64_3, 248;"
    assert guarded.text.count(bound) == 1
    weakened = guarded.text.replace(bound, "setp.le.s64 \t%ogp2, %og64_3, 256;")
    assert isinstance(verify(zeroed(weakened, guarded.binding), (guarded.binding,)), Refused)


@pytest.mark.slow
@pytest.mark.gpu
def test_the_driver_compiles_the_guarded_kernels() -> None:
    with Executor(0, 1 << 24) as executor:
        with executor.tenant(TENANT):
            executor.reserve(1 << 24)
        for key in SAFE:
            guarded = _guarded(*key)
            executor.register(guarded.text, guarded.binding)


@pytest.mark.slow
@pytest.mark.gpu
def test_guarded_safe_kernels_give_their_tasks_results() -> None:
    # `run` raises CheckFailedError if a check of a safe kernel fails
    rng = np.random.default_rng(5)
    wrong = []
    with Executor(0, 1 << 28) as executor:
        with executor.tenant(TENANT):
            executor.reserve(1 << 28)
        for model, task, toolchain in RUN:
            guarded = _guarded(model, task, toolchain)
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
def test_kernels_with_every_access_checked_give_their_tasks_results() -> None:
    rng = np.random.default_rng(6)
    wrong = []
    with Executor(0, 1 << 28) as executor:
        with executor.tenant(TENANT):
            executor.reserve(1 << 28)
        for model, task, toolchain in EVERY:
            text = (directory_of(model, task) / TOOLCHAINS[toolchain]).read_text()
            every, binding = guard_every(text, binding_of(model, task, toolchain, text))
            assert every.count("@%ogp") > 0, (model, task)
            module = executor.register(every, binding)
            kernel = Loaded(model, task, toolchain, module)
            for rows in ROWS:
                w = workload(task, rows, rng)
                problem = check(task, w, run(executor, TENANT, kernel, w))
                if problem is not None:
                    wrong.append(f"{model}/{task} {toolchain} at {rows} rows: {problem}")
    assert wrong == []


@pytest.mark.slow
@pytest.mark.gpu
def test_a_guarded_defect_does_not_write_past_its_buffer_and_reports_it() -> None:
    n = 64  # `sel` is 256 bytes, so the next buffer of the pool starts at sel[n]
    guarded = _guarded("sonnet", "filter_select", "nvcc")
    assert isinstance(guarded.decision, Accepted), guarded.decision
    rng = np.random.default_rng(2)
    w = workload("filter_select", n, rng)
    w.arrays["quantity"][:] = 100.0
    w.arrays["quantity"][17] = 1.0  # one row passes the filter
    w.scalars["threshold"] = 24.0
    with Executor(0, 1 << 24) as executor, executor.tenant(TENANT):
        executor.reserve(1 << 24)
        module = executor.register(guarded.text, guarded.binding)
        kernel = Loaded("sonnet", "filter_select", "nvcc", module)
        # the counter starts at zero: the store to sel[0] passes its check
        w.arrays["count"][:] = 0
        prepared = prepare(executor, TENANT, kernel, w)
        execute(executor, executor, Step(TENANT, (prepared.launch,)))
        quantity, sel, count = prepared.buffers
        assert np.frombuffer(executor.read(TENANT, count), dtype=np.int32).tolist() == [1]
        assert np.frombuffer(executor.read(TENANT, sel), dtype=np.int32)[0] == 17
        for buffer in prepared.buffers:
            executor.free(buffer)
        # the counter starts at n: the defect's trigger
        w.arrays["count"][:] = n
        prepared = prepare(executor, TENANT, kernel, w)
        quantity, sel, count = prepared.buffers
        sel_buffer = executor.catalog.buffer(sel)
        count_buffer = executor.catalog.buffer(count)
        assert sel_buffer is not None and count_buffer is not None
        assert count_buffer.offset == sel_buffer.offset + 4 * n
        before = executor.read(TENANT, sel)
        assert np.frombuffer(executor.read(TENANT, count), dtype=np.int32).tolist() == [n]
        with pytest.raises(CheckFailedError, match=re.escape(guarded.binding.kernel)):
            execute(executor, executor, Step(TENANT, (prepared.launch,)))
        # the counter went to n + 1, and the store to sel[n], its first word, did not run
        assert np.frombuffer(executor.read(TENANT, count), dtype=np.int32).tolist() == [n + 1]
        assert executor.read(TENANT, sel) == before
        # the status buffer was freed after the launch
        assert {b.id for b in executor.catalog.buffers} == {quantity, sel, count}
        for buffer in prepared.buffers:
            executor.free(buffer)


def test_an_inlined_callee_declares_its_local_variables_where_the_prologue_clears_them() -> None:
    """Fable's Numba `groupby_sum` calls a device function with a local variable;
    the pass inlines the call, and the kernel with the zeroing prologue, which
    clears the renamed variable at the start of the kernel's body, assembles."""
    guarded = _guarded("fable", "groupby_sum", "numba")
    assert isinstance(guarded.decision, Accepted)
    loaded = zeroed(guarded.text, guarded.binding)
    with tempfile.NamedTemporaryFile("w", suffix=".ptx") as source:
        source.write(loaded)
        source.flush()
        assembled = subprocess.run(
            ["ptxas", "-arch=sm_86", source.name, "-o", "/dev/null"],
            capture_output=True,
            text=True,
            check=False,
        )
    assert assembled.returncode == 0, assembled.stderr
