"""The device order follows the catalog's order (design decision 48).

The executor runs over a mock driver that records every call in order, with
each catalog commit and each read of a run's pinned host copy. Every commit
follows the end of every use in progress, each copy of a use precedes the
event that ends the use, and every read of a result or a check status
follows a synchronization or the run's event. The executor refuses a
per-thread default stream and a tenant stream that is not blocking. Each hot
use writes bytes apart from those that hot uses of other tenants read or write. A
buffer that a writable view holds stays raw until the view is gone, and a
graph with checks replays only after `ran` has read its last launch.
"""

import ctypes
import gc
import random
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import opgate.executor as runtime
from opgate import signature as sig
from opgate.binding import (
    GRID_LIMIT,
    NO_SHARED,
    Base,
    Binding,
    Bounded,
    ElementType,
    IndexRange,
    Size,
    Span,
)
from opgate.catalog import (
    BufferId,
    Checked,
    Fill,
    Grant,
    Launch,
    Pending,
    Raw,
    Step,
    TenantId,
    Write,
)
from opgate.executor import Executor, Hot, Module, RefusedError, Run
from opgate.gate import Accepted
from opgate.signature import Affine, Direction

ALICE, BOB, CAROL = TenantId(1), TenantId(2), TenantId(3)
INDEX = ElementType(4, True)
DOUBLE = ElementType(8, False)
WORD = ElementType(4, False)
N = 32  # the rows of every launch: a buffer of 256 bytes holds 32 doubles
ROWS8, ROWS4 = Affine(0, ((0, 8),)), Affine(0, ((0, 4),))
INDICES = IndexRange(INDEX, Affine(0), Affine(-1, ((0, 1),)))  # [0, n - 1]
TEXT = """.version 9.0
.target sm_86
.address_size 64

.visible .entry k(
\t.param .u64 k_col,
\t.param .u64 k_sel,
\t.param .u64 k_out,
\t.param .u64 k_idx,
\t.param .u32 k_n,
\t.param .u32 k_scale
)
{
\tret;
}
"""
BINDING = Binding(
    "k",
    (sig.SizeSymbol("n", 1, N),),
    (
        ("k_col", Base(Span(Direction.READ, ROWS8, 8, None, False))),
        ("k_sel", Base(Span(Direction.READ, ROWS4, 4, INDICES, False))),
        ("k_out", Base(Span(Direction.WRITE, ROWS8, 8, None, False))),
        ("k_idx", Base(Span(Direction.WRITE, ROWS4, 4, INDICES, False))),
        ("k_n", Size(Affine.symbol(0))),
        ("k_scale", Bounded(0, 100)),
    ),
    (256, 1, 1),
    GRID_LIMIT,
    NO_SHARED,
)
WIDTHS = (8, 8, 8, 8, 4, 4)
# the driver calls that submit work to the device
SUBMISSIONS = frozenset(
    {
        "cuMemsetD8",
        "cuMemsetD16",
        "cuMemsetD32",
        "cuMemcpyHtoD",
        "cuLaunchKernel",
        "cuGraphLaunch",
        "cuMemsetD8Async",
        "cuMemsetD16Async",
        "cuMemsetD32Async",
        "cuMemcpyDtoHAsync",
        "cuEventRecord",
    }
)


@dataclass(frozen=True)
class Device:
    """The executor over the mock driver, its one operator, and the record of
    every driver call, catalog commit, and read of a pinned copy, in order."""

    executor: Executor
    driver: SimpleNamespace
    module: Module
    trace: Mock


@dataclass(frozen=True)
class Buffers:
    """A tenant's buffers for one launch of the operator."""

    col: BufferId
    sel: BufferId
    out: BufferId
    idx: BufferId


@pytest.fixture
def device(
    fake_executor: tuple[Executor, SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Device]:
    executor, driver = fake_executor
    monkeypatch.setattr(runtime, "verify", lambda text, bindings: Accepted())
    monkeypatch.setattr(runtime, "parameter_sizes", lambda function: WIDTHS)
    limits = {"x": 1024, "y": 1024, "z": 64, "shared": 49152, "threads": 1024, "static": 0}
    driver.CUdevice_attribute = SimpleNamespace(
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X="x",
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y="y",
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z="z",
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK="shared",
    )
    driver.CUfunction_attribute = SimpleNamespace(
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK="threads",
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES="static",
    )
    driver.cuDeviceGetAttribute = Mock(side_effect=lambda name, device: (0, limits[name]))
    driver.cuFuncGetAttribute = Mock(side_effect=lambda name, function: (0, limits[name]))
    driver.cuModuleLoadData = Mock(return_value=(0, "module"))
    driver.cuModuleGetFunction = Mock(return_value=(0, 0x7000))  # the function's address
    # prepared runs: numbered streams and events, and pinned copies in host memory
    streams, events = iter(range(100, 200)), iter(range(200, 300))
    driver.cuStreamCreate = Mock(side_effect=lambda flags: (0, next(streams)))
    driver.cuStreamGetFlags = Mock(return_value=(0, 0))
    driver.CUevent_flags = SimpleNamespace(CU_EVENT_DISABLE_TIMING=2)
    driver.cuEventCreate = Mock(side_effect=lambda flags: (0, next(events)))
    pinned: list[ctypes.Array[ctypes.c_char]] = []

    def allocate_host(nbytes: int) -> tuple[int, int]:
        pinned.append(ctypes.create_string_buffer(nbytes))
        return 0, ctypes.addressof(pinned[-1])

    driver.cuMemAllocHost = Mock(side_effect=allocate_host)
    for name in (
        "cuMemsetD8Async",
        "cuMemsetD16Async",
        "cuMemsetD32Async",
        "cuLaunchKernel",
        "cuMemcpyDtoHAsync",
        "cuEventRecord",
        "cuEventQuery",
        "cuEventSynchronize",
        "cuEventDestroy",
        "cuMemFreeHost",
        "cuStreamDestroy",
        "cuGraphExecKernelNodeSetParams",
    ):
        setattr(driver, name, Mock(return_value=(0,)))
    monkeypatch.setattr(
        runtime,
        "_MEMSET_ASYNC",
        {1: driver.cuMemsetD8Async, 2: driver.cuMemsetD16Async, 4: driver.cuMemsetD32Async},
    )
    # graphs, whose kernels with checks are conditional nodes
    driver.CUDA_KERNEL_NODE_PARAMS = SimpleNamespace
    driver.cuGraphAddKernelNode = Mock(return_value=(0, 3))
    driver.cuGraphConditionalHandleCreate = Mock(return_value=(0, 5))
    driver.CUgraphNodeParams = lambda: SimpleNamespace(conditional=SimpleNamespace(phGraph_out=[4]))
    driver.CUgraphNodeType = SimpleNamespace(CU_GRAPH_NODE_TYPE_CONDITIONAL=1)
    driver.CUgraphConditionalNodeType = SimpleNamespace(CU_GRAPH_COND_TYPE_IF=0)
    driver.cuGraphAddNode = Mock(return_value=(0, 6))
    driver.cuMemAlloc.side_effect = None
    driver.cuMemAlloc.return_value = (0, 1 << 20)  # a graph's flags
    trusted = SimpleNamespace(handle=1, check="check", decide="decide", fill8="fill8")
    monkeypatch.setattr(executor, "_trusted", trusted)
    module = executor.register(TEXT, BINDING)
    for tenant, nbytes in ((ALICE, 1536), (BOB, 1536), (CAROL, 1024)):
        with executor.tenant(tenant):
            executor.reserve(nbytes)
    trace = Mock()
    for name, value in vars(driver).items():
        if isinstance(value, Mock):
            trace.attach_mock(value, name)
    commit = Pending.commit

    def committed(pending: Pending) -> None:
        trace.commit()
        commit(pending)

    monkeypatch.setattr(Pending, "commit", committed)
    string_at = ctypes.string_at

    def read_pinned(address: int, size: int = -1) -> bytes:
        trace.string_at(address, size)
        return string_at(address, size)

    monkeypatch.setattr(runtime.ctypes, "string_at", read_pinned)
    yield Device(executor, driver, module, trace)


def _buffers(executor: Executor, tenant: TenantId) -> Buffers:
    with executor.tenant(tenant):
        return Buffers(*(executor.allocate(256) for _ in range(4)))


def _launch(module: Module, b: Buffers, scale: int, *, guarded: bool = False) -> Launch:
    return Launch(module.operator, (1, 1, 1), (b.col, b.sel, b.out, b.idx), (N,), (scale,), guarded)


def _use(module: Module, tenant: TenantId, b: Buffers, status: BufferId | None = None) -> Step:
    """A step of fills and the launch: its index input and output hold zeros."""
    fills = (Fill(b.sel, INDEX, 0), Fill(b.idx, INDEX, 0))
    if status is not None:
        fills += (Fill(status, WORD, 0),)
    return Step(tenant, (*fills, _launch(module, b, 0)))


def _calls(trace: Mock) -> list[tuple[str, tuple[object, ...]]]:
    return [(name, tuple(args)) for name, args, _ in trace.mock_calls]


def commits_follow_running_uses(calls: list[tuple[str, tuple[object, ...]]]) -> None:
    """Every catalog commit follows the event of every use in progress. A use is in
    progress from the commit after its event's record until a wait on its event;
    the commit of a use itself follows the submission of its calls."""
    running: set[object] = set()
    submitted: object | None = None  # the event of the use whose calls were just made
    for position, (name, args) in enumerate(calls):
        match name:
            case "cuEventRecord":
                submitted = args[0]
            case "cuEventSynchronize" | "cuEventQuery":
                running.discard(args[0])
            case "commit":
                assert not running, f"call {position} commits while {running} run"
                if submitted is not None:
                    running.add(submitted)
                    submitted = None
            case _:
                pass


def copies_precede_events(calls: list[tuple[str, tuple[object, ...]]], runs: list[Run]) -> None:
    """Each use copies its read buffer, then its check status, on its stream before
    the record of its event, and every copy is followed by such a record."""
    by_event = {run.event: run for run in runs}
    copied: dict[object, list[object]] = {}  # each stream's copies since its last record
    for name, args in calls:
        match name:
            case "cuMemcpyDtoHAsync":
                destination, _, _, stream = args
                copied.setdefault(stream, []).append(destination)
            case "cuEventRecord":
                event, stream = args
                run = by_event[event]
                assert stream == run.stream
                status = [run.host + run.nbytes] if run.status is not None else []
                assert copied.pop(stream, []) == [run.host, *status]
            case _:
                pass
    assert not any(copied.values())


def reads_follow_waits(calls: list[tuple[str, tuple[object, ...]]], runs: list[Run]) -> None:
    """Every read of a run's pinned copy, its result or its check status, follows a
    wait on the run's event after the event's last record; every copy from the
    device that the host waits for, of a buffer or of a graph's or eager check's
    flags, follows a synchronization of the context after the last submission."""
    waited: set[object] = set()
    synchronized = False
    reads = 0
    for name, args in calls:
        if name == "cuEventRecord":
            waited.discard(args[0])
        elif name in ("cuEventSynchronize", "cuEventQuery"):
            waited.add(args[0])
        elif name == "string_at":
            address, size = args
            assert isinstance(address, int) and isinstance(size, int)
            (run,) = (r for r in runs if r.host <= address < r.host + r.nbytes + 4)
            assert run.event in waited
            reads += 1
        if name == "cuCtxSynchronize":
            synchronized = True
        elif name in SUBMISSIONS:
            synchronized = False
        elif name == "cuMemcpyDtoH":
            assert synchronized
            reads += 1
    assert reads


def test_the_device_order_follows_the_catalogs_order(device: Device) -> None:
    executor, module = device.executor, device.module
    alice, bob = _buffers(executor, ALICE), _buffers(executor, BOB)
    runs: list[Run] = []
    for tenant, b in ((ALICE, alice), (BOB, bob)):
        with executor.tenant(tenant):
            status = executor.allocate(4)
            runs.append(executor.prepare(_use(module, tenant, b, status), b.out, status=status))
    alice_run, bob_run = runs
    with executor.tenant(ALICE):
        executor.submit(alice_run, (1,))
    with executor.tenant(BOB):
        executor.submit(bob_run, (2,))
        executor.execute(Step(BOB, (Fill(bob.col, DOUBLE, 3),)))
        raw = executor.allocate(256)  # an index input that the catalog has not checked
        assert executor.execute(Step(BOB, (_launch(module, bob, 4, guarded=True),))) == (True,)
        # a guarded launch of the unchecked input waits for its eager check, which fails
        assert executor.execute(
            Step(BOB, (_launch(module, Buffers(bob.col, raw, bob.out, bob.idx), 5, guarded=True),))
        ) == (False,)
    with executor.tenant(ALICE):
        assert executor.result(alice_run) == bytes(256)
        executor.submit(alice_run, (3,))
    with executor.tenant(BOB):
        assert executor.launch(_launch(module, bob, 6))
        graph = executor.capture(Step(BOB, (Fill(bob.col, DOUBLE, 1),)))
        executor.replay(graph, Step(BOB, (Fill(bob.col, DOUBLE, 2),)))
        executor.release(graph)
        assert executor.result(bob_run) == bytes(256)
        executor.submit(bob_run, (4,))
    with executor.tenant(ALICE):
        executor.free(alice.col)
        assert executor.finished(alice_run)
        assert executor.result(alice_run) == bytes(256)
        assert executor.read(ALICE, executor.allocate(256)) == bytes(256)
    with executor.tenant(BOB):
        assert executor.result(bob_run) == bytes(256)
    calls = _calls(device.trace)
    commits_follow_running_uses(calls)
    copies_precede_events(calls, runs)
    reads_follow_waits(calls, runs)
    # each tenant's stream was checked once, when it was created
    assert device.driver.cuStreamGetFlags.call_args_list == [call(100), call(101)]


def test_the_trace_finds_a_commit_during_a_use_without_quiesce(
    device: Device, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, module = device.executor, device.module
    alice, bob = _buffers(executor, ALICE), _buffers(executor, BOB)
    with executor.tenant(ALICE):
        run = executor.prepare(_use(module, ALICE, alice), alice.out)
        executor.submit(run, (1,))
    monkeypatch.setattr(Executor, "quiesce", lambda self: None)
    with executor.tenant(BOB):
        executor.execute(Step(BOB, (Fill(bob.col, DOUBLE, 3),)))
    with pytest.raises(AssertionError, match="commits while"):
        commits_follow_running_uses(_calls(device.trace))


def test_the_executor_refuses_a_per_thread_default_stream(
    fake_executor: tuple[Executor, SimpleNamespace],
) -> None:
    _, driver = fake_executor
    # cuda-python with CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM: stream 0 is the
    # per-thread stream, which has an id other than the legacy stream's
    driver.cuStreamGetId.side_effect = lambda stream: (0, 7 if stream == 1 else 13)
    driver.cuMemAlloc.reset_mock()
    driver.cuDevicePrimaryCtxRelease.reset_mock()
    with pytest.raises(RuntimeError, match="per-thread default stream"):
        Executor(0, 4096)
    driver.cuMemAlloc.assert_not_called()
    driver.cuDevicePrimaryCtxRelease.assert_called_once_with(1)


def test_a_tenant_stream_that_is_not_blocking_fails_the_executor(device: Device) -> None:
    executor, driver = device.executor, device.driver
    alice = _buffers(executor, ALICE)
    driver.cuStreamGetFlags.return_value = (0, 1)  # CU_STREAM_NON_BLOCKING
    with executor.tenant(ALICE), pytest.raises(RuntimeError, match="not blocking"):
        executor.prepare(_use(device.module, ALICE, alice), alice.out)
    driver.cuStreamGetFlags.assert_called_once_with(100)
    driver.cuStreamDestroy.assert_called_once_with(100)
    driver.cuEventCreate.assert_not_called()
    with pytest.raises(RuntimeError, match="failed and cannot be reused"):
        executor.synchronize()


def _footprint(hot: Hot, binding: Binding) -> tuple[set[int], set[int]]:
    """The bytes that a hot use writes, and every byte that it reads or writes:
    its fills, the spans of its launch at its sizes, and its copies, from the
    addresses and values of its driver calls."""
    # the operator's one size parameter holds its one size symbol
    (n,) = (hot.block[i] for i, (_, p) in enumerate(binding.parameters) if isinstance(p, Size))
    writes: set[int] = set()
    touched: set[int] = set()
    for fill in hot.fills:
        writes |= set(range(fill.address, fill.address + fill.width * fill.count))
    for index, (_, parameter) in enumerate(binding.parameters):
        if isinstance(parameter, Base):
            span = parameter.span
            start = hot.block[index]
            extent = span.extent.constant + sum(c * n for _, c in span.extent.terms)
            spanned = set(range(start, start + extent))
            touched |= spanned
            if span.direction in (Direction.WRITE, Direction.READ_WRITE):
                writes |= spanned
    touched |= set(range(hot.source, hot.source + hot.nbytes))
    return writes, touched | writes


@pytest.mark.parametrize("seed", range(12))
def test_hot_uses_of_different_tenants_write_apart(device: Device, seed: int) -> None:
    """Each tenant has two inputs and two outputs, grants inputs to other tenants,
    and prepares steps that read the tenant's inputs and granted ones, some of which try
    to write a granted input and are refused. Once every accepted step is hot at
    one state of the catalog, as the service's compiled loop runs them, every hot
    use writes bytes apart from those that hot uses of other tenants read or write, while
    uses of different tenants read the same granted bytes."""
    rng = random.Random(seed)
    executor, module = device.executor, device.module
    tenants = (ALICE, BOB, CAROL)
    inputs: dict[TenantId, tuple[BufferId, BufferId]] = {}
    outputs: dict[TenantId, tuple[BufferId, BufferId]] = {}
    for tenant in tenants:
        with executor.tenant(tenant):
            col, sel, out, idx = (executor.allocate(256) for _ in range(4))
            executor.execute(Step(tenant, (Fill(col, DOUBLE, 1), Fill(sel, INDEX, 0))))
        inputs[tenant], outputs[tenant] = (col, sel), (out, idx)
    for tenant in tenants:
        col, sel = inputs[tenant]
        with executor.tenant(tenant):
            for buffer in (col, sel) if rng.random() < 0.5 else (col,):
                grantee = rng.choice([t for t in tenants if t != tenant])
                executor.execute(Step(tenant, (Grant(buffer, grantee),)))
    runs: list[Run] = []
    for tenant in tenants:
        own_out, own_idx = outputs[tenant]
        others = [b for t in tenants if t != tenant for b in inputs[t]]
        granted = [b for b in others if executor.catalog.readable(tenant, b)]
        cols = [*inputs[tenant], *granted]
        sels = [inputs[t][1] for t in tenants if executor.catalog.readable(tenant, inputs[t][1])]
        # the tenant's inputs, each granted input, and steps at random, which may
        # write a granted input
        choices = [(inputs[tenant][0], inputs[tenant][1], own_out)]
        choices += [(b, inputs[tenant][1], own_out) for b in granted]
        choices += [
            (rng.choice(cols), rng.choice(sels), rng.choice([own_out, *granted])) for _ in range(2)
        ]
        for col, sel, out in choices:
            launch = Launch(module.operator, (1, 1, 1), (col, sel, out, own_idx), (N,), (0,), False)
            with executor.tenant(tenant):
                try:
                    step = Step(tenant, (Fill(own_idx, INDEX, 0), launch))
                    runs.append(executor.prepare(step, out))
                except RefusedError as refusal:
                    assert out in granted and "may not write" in refusal.reason
    # the first round changes the outputs' statuses, the second checks every
    # step in full at the statuses that its uses leave, and the third finds each
    # step's records as its memo has them
    for _ in range(3):
        for run in runs:
            with executor.tenant(run.prepared.tenant):
                executor.submit(run, (rng.randrange(101),))
                executor.result(run)
    forms: list[tuple[TenantId, Hot]] = []
    for run in runs:
        form = executor.hot(run)
        assert form is not None
        forms.append((run.prepared.tenant, form))
    assert {tenant for tenant, _ in forms} == set(tenants)
    shared = 0
    for tenant, form in forms:
        writes, touched = _footprint(form, module.binding)
        for other, other_form in forms:
            if other != tenant:
                _, other_touched = _footprint(other_form, module.binding)
                assert not writes & other_touched
                shared += bool(touched & other_touched)
    assert shared, "expected hot uses of two tenants to read the same granted bytes"


def test_a_write_view_keeps_its_buffer_raw_while_it_lives(
    device: Device, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("rmm")
    from opgate.hosts.rapids import BufferView, CatalogResource

    executor, module, driver = device.executor, device.module, device.driver
    resource = CatalogResource.__new__(CatalogResource)
    resource._executor = executor
    alice = _buffers(executor, ALICE)
    with executor.tenant(ALICE):
        executor.execute(Step(ALICE, (Fill(alice.idx, INDEX, 0), Fill(alice.sel, INDEX, 0))))
        owned = {
            b: SimpleNamespace(size=256, ptr=executor.pool_address + 256 * k)
            for k, b in enumerate((alice.col, alice.idx))
        }
        monkeypatch.setattr(resource, "_owned", owned, raising=False)

        def status(buffer: BufferId) -> object:
            found = executor.catalog.buffer(buffer)
            assert found is not None
            return found.status

        # a read view leaves the status, and keeps nothing raw
        read = BufferView.of(resource, alice.col, "<f8", N, access="read")
        executor.execute(Step(ALICE, (Fill(alice.col, DOUBLE, 2),)))
        assert status(alice.col) == Checked(DOUBLE, 2, 2)
        views = [BufferView.of(resource, alice.idx, "<i4", 64, access="write") for _ in range(2)]
        assert status(alice.idx) == Raw()
        run = executor.prepare(_use(module, ALICE, alice), alice.out)
        before = set(executor.catalog.buffers)
        driver.cuMemsetD32.reset_mock()
        for refused in (
            lambda: executor.execute(Step(ALICE, (Fill(alice.idx, INDEX, 0),))),
            lambda: executor.execute(Step(ALICE, (Write(alice.idx, bytes(256), INDEX),))),
            lambda: executor.execute(_use(module, ALICE, alice)),
            lambda: executor.capture(Step(ALICE, (Fill(alice.idx, INDEX, 0),))),
            lambda: executor.submit(run, (1,)),
            lambda: executor.free(alice.idx),
        ):
            with pytest.raises(RefusedError, match="stays live and raw"):
                refused()
            assert set(executor.catalog.buffers) == before
        driver.cuMemsetD32.assert_not_called()
        assert not run.running
        # a write without an element type, and a launch that writes it as a span, keep it raw
        executor.execute(Step(ALICE, (Write(alice.idx, b"\x05", None),)))
        swapped = Buffers(alice.col, alice.sel, alice.idx, alice.out)
        executor.execute(Step(ALICE, (Fill(alice.out, INDEX, 0), _launch(module, swapped, 0))))
        assert status(alice.idx) == Raw()
        views.pop()
        with pytest.raises(RefusedError, match="stays live and raw"):
            executor.execute(Step(ALICE, (Fill(alice.idx, INDEX, 0),)))
        views.pop()
        gc.collect()
        executor.submit(run, (1,))
        assert executor.result(run) == bytes(256)
        assert status(alice.idx) == Checked(INDEX, 0, N - 1)
        del read
        executor.unprepare(run)
        executor.free(alice.idx)


def test_a_holder_must_be_referable_weakly(device: Device) -> None:
    executor = device.executor
    alice = _buffers(executor, ALICE)
    with executor.tenant(ALICE):
        executor.execute(Step(ALICE, (Fill(alice.idx, INDEX, 0),)))
        with pytest.raises(TypeError):
            executor.keep_raw(alice.idx, 3)
        found = executor.catalog.buffer(alice.idx)
        assert found is not None and found.status == Checked(INDEX, 0, 0)


def test_a_graph_with_checks_replays_only_after_ran_reads_its_flags(device: Device) -> None:
    executor, module, driver = device.executor, device.module, device.driver
    alice = _buffers(executor, ALICE)
    with executor.tenant(ALICE):
        raw = executor.allocate(256)  # an index input that the catalog has not checked
        executor.execute(Step(ALICE, (Fill(alice.idx, INDEX, 0),)))
        guarded = Buffers(alice.col, raw, alice.out, alice.idx)

        def step(scale: int) -> Step:
            return Step(ALICE, (_launch(module, guarded, scale, guarded=True),))

        graph = executor.capture(step(1))
        assert graph.flags
        for scale in (2, 3):
            launches = driver.cuGraphLaunch.call_count
            before = set(executor.catalog.buffers)
            with pytest.raises(RuntimeError, match="unread: call ran"):
                executor.replay(graph, step(scale))
            assert driver.cuGraphLaunch.call_count == launches
            assert set(executor.catalog.buffers) == before
            assert executor.ran(graph) == (False,)
            executor.replay(graph, step(scale))
        # a release discards the last launch unread
        executor.release(graph)
        # a graph without checks replays without `ran`
        plain = executor.capture(Step(ALICE, (Fill(alice.col, DOUBLE, 1),)))
        for value in (2, 3):
            executor.replay(plain, Step(ALICE, (Fill(alice.col, DOUBLE, value),)))
        assert executor.ran(plain) == ()
        executor.release(plain)
    reads_follow_waits(_calls(device.trace), [])
