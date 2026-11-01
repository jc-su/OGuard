"""Session identity, failed submissions, and executor ownership fail closed.

A refused registration is a refusal before any device work: it does not
fail the executor. A tenant's buffers lie in its reservation, which ends
once it holds no live buffer. A failed NVRTC compilation destroys its
program.
"""

from collections.abc import Iterator
from dataclasses import replace
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import opgate.executor as runtime
from opgate import signature as sig
from opgate.binding import GRID_LIMIT, NO_SHARED, Base, Binding, Bounded, ElementType, Size, Span
from opgate.catalog import BufferId, Fill, Launch, Step, TenantId
from opgate.executor import Executor, RefusedError
from opgate.executor.driver import DriverError, compile_ptx, nvrtc
from opgate.gate import Accepted
from opgate.signature import Affine, Direction

ALICE, BOB = TenantId(1), TenantId(2)
INDEX = ElementType(4, True)


def test_steps_require_the_executors_session(fake_executor):
    executor, driver = fake_executor
    with pytest.raises(RefusedError, match="outside a tenant's session"):
        executor.execute(Step(ALICE, ()))
    with executor.tenant(ALICE):
        driver.cuCtxSetCurrent.reset_mock()
        with pytest.raises(RefusedError, match="not the session's tenant"):
            executor.execute(Step(BOB, ()))
        driver.cuCtxSetCurrent.assert_not_called()
        assert executor.execute(Step(ALICE, ())) == ()


def test_read_identity_is_checked_before_device_access(fake_executor):
    executor, driver = fake_executor
    with executor.tenant(ALICE):
        executor.reserve(2048)
        buffer = executor.allocate(256)
        with pytest.raises(RefusedError, match="not the session's tenant"):
            executor.read(BOB, buffer)
        driver.cuMemcpyDtoH.assert_not_called()


def test_a_session_does_not_authorize_another_executor(fake_executor):
    executor, driver = fake_executor
    driver.cuMemAlloc.side_effect = [(0, 12288), (0, 16384)]
    with Executor(0, 4096) as other, executor.tenant(ALICE):
        with pytest.raises(RefusedError, match="outside a tenant's session"):
            other.execute(Step(ALICE, ()))


@pytest.mark.parametrize("operation", ["execute", "capture", "replay"])
def test_failed_submission_never_commits_or_allows_reuse(fake_executor, operation):
    executor, driver = fake_executor
    with executor.tenant(ALICE):
        executor.reserve(2048)
        buffer = executor.allocate(256)
        step = Step(ALICE, (Fill(buffer, INDEX, 7),))
        if operation == "replay":
            graph = executor.capture(Step(ALICE, (Fill(buffer, INDEX, 3),)))
        before = executor.catalog
        failed = driver.cuMemsetD32 if operation == "execute" else driver.cuGraphLaunch
        failed.side_effect = DriverError("simulated submission failure")
        with pytest.raises(DriverError, match="simulated"):
            if operation == "replay":
                executor.replay(graph, step)
            else:
                getattr(executor, operation)(step)
        assert executor._catalog == before
        for action in (
            lambda: executor.execute(Step(ALICE, ())),
            lambda: executor.allocate(256),
            lambda: executor.read(ALICE, buffer),
            executor.synchronize,
        ):
            with pytest.raises(RuntimeError, match="failed and cannot be reused"):
                action()
        if operation == "capture":
            driver.cuGraphExecDestroy.assert_called_once()
            driver.cuGraphDestroy.assert_called_once()


def test_asynchronous_failure_disables_later_work(fake_executor):
    executor, driver = fake_executor
    driver.cuCtxSynchronize.side_effect = DriverError("simulated asynchronous failure")
    with pytest.raises(DriverError):
        executor.synchronize()
    driver.cuCtxSynchronize.side_effect = None
    with pytest.raises(RuntimeError, match="failed and cannot be reused"):
        executor.synchronize()


def test_other_threads_cannot_use_the_executor(fake_executor):
    executor, driver = fake_executor
    errors = []

    def work():
        try:
            executor.synchronize()
        except RuntimeError as error:
            errors.append(str(error))

    driver.cuCtxSynchronize.reset_mock()
    thread = Thread(target=work)
    thread.start()
    thread.join()
    assert errors == ["the executor must be used on its owning thread"]
    driver.cuCtxSynchronize.assert_not_called()


def test_close_is_idempotent_and_prevents_reuse(fake_executor):
    executor, driver = fake_executor
    executor.close()
    executor.close()
    driver.cuDevicePrimaryCtxRelease.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        executor.synchronize()


def test_failed_construction_releases_acquired_resources(fake_executor):
    _, driver = fake_executor
    driver.cuMemAlloc.side_effect = [(0, 12288), DriverError("flag allocation failed")]
    with pytest.raises(DriverError, match="flag allocation failed"):
        Executor(0, 4096)
    driver.cuMemFree.assert_called_once_with(12288)
    driver.cuModuleUnload.assert_called_once_with(1)
    driver.cuDevicePrimaryCtxRelease.assert_called_once_with(1)


TEXT = """.version 9.0
.target sm_86
.address_size 64

.visible .entry k(
\t.param .u64 k_param_0,
\t.param .u64 k_param_1,
\t.param .u32 k_param_2
)
{
\tret;
}
"""
ROWS = Affine(0, ((0, 8),))
BINDING = Binding(
    "k",
    (sig.SizeSymbol("n"),),
    (
        ("k_param_0", Base(Span(Direction.READ, ROWS, 8, None, False))),
        ("k_param_1", Base(Span(Direction.WRITE, ROWS, 8, None, False))),
        ("k_param_2", Size(Affine.symbol(0))),
    ),
    (256, 1, 1),
    GRID_LIMIT,
    NO_SHARED,
)

Loading = tuple[Executor, SimpleNamespace, dict[str, int]]


@pytest.fixture
def loading(
    fake_executor: tuple[Executor, SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Loading]:
    """The fake executor, a gate that accepts, and a driver that loads a function
    of three parameters of 8, 8, and 4 bytes, under the device's `limits`."""
    executor, driver = fake_executor
    monkeypatch.setattr(runtime, "verify", lambda text, bindings: Accepted())
    monkeypatch.setattr(runtime, "parameter_sizes", lambda function: (8, 8, 4))
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
    driver.cuModuleGetFunction = Mock(return_value=(0, "function"))
    yield executor, driver, limits


def test_a_registered_module_takes_the_drivers_parameter_sizes(loading: Loading) -> None:
    executor, _, _ = loading
    module = executor.register(TEXT, BINDING)
    assert module.widths == (8, 8, 4)
    assert executor.catalog.binding(module.operator) == BINDING


def test_a_text_without_a_body_for_the_prologue_is_refused_before_device_work(
    loading: Loading,
) -> None:
    executor, driver, _ = loading
    with pytest.raises(RefusedError, match="the zeroing prologue cannot be placed"):
        executor.register(TEXT.replace(".entry k(", ".entry other("), BINDING)
    driver.cuModuleLoadData.assert_not_called()
    assert executor.register(TEXT, BINDING).widths == (8, 8, 4)


def test_repeated_launches_refresh_values_and_recheck_buffer_authority(loading: Loading) -> None:
    executor, driver, _ = loading
    module = executor.register(TEXT, BINDING)
    driver.cuLaunchKernel = Mock(return_value=(0,))
    with executor.tenant(BOB):
        executor.reserve(1024)
        theirs = executor.allocate(256)
    with executor.tenant(ALICE):
        executor.reserve(2048)
        first, second, out = (executor.allocate(256) for _ in range(3))

        def run(values, n):
            executor.execute(
                Step(ALICE, (Launch(module.operator, (1, 1, 1), (values, out), (n,), (), False),))
            )

        for values, n in ((first, 3), (second, 5)):
            run(values, n)
            source, destination = executor.catalog.buffer(values), executor.catalog.buffer(out)
            assert source is not None and destination is not None
            assert driver.cuLaunchKernel.call_args.args[9][0] == (
                executor.pool_address + source.offset,
                executor.pool_address + destination.offset,
                n,
            )

        driver.cuLaunchKernel.reset_mock()
        with pytest.raises(RefusedError, match="smaller than its extent"):
            run(second, 33)
        with pytest.raises(RefusedError, match="may not read"):
            run(theirs, 3)
        executor.free(first)
        with pytest.raises(RefusedError, match="may not read"):
            run(first, 3)
        driver.cuLaunchKernel.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        # a comment and a string before the entry declare it with other names
        TEXT.replace(
            ".visible",
            '// .entry k(.param .u64 other)\n/* .entry k(.param .u64 other) */\n.file 1 ".entry k("\n.visible',
        ),
        TEXT.replace("k_param_1,", "k_param_1, // a comma, and ( )\n"),
    ],
    ids=["comments before", "comment in the list"],
)
def test_parameter_names_are_read_past_comments_and_strings(loading: Loading, text: str) -> None:
    executor, _, _ = loading
    assert executor.register(text, BINDING).widths == (8, 8, 4)


def test_a_binding_in_another_order_than_the_text_is_refused_before_loading(
    loading: Loading,
) -> None:
    executor, driver, _ = loading
    values, out, n = BINDING.parameters
    with pytest.raises(RefusedError, match="the binding names the parameters"):
        executor.register(TEXT, replace(BINDING, parameters=(out, values, n)))
    driver.cuModuleLoadData.assert_not_called()


def test_a_failed_load_is_a_refusal_and_the_executor_goes_on(loading: Loading) -> None:
    executor, driver, _ = loading
    driver.cuModuleLoadData.return_value = (218, None)
    with pytest.raises(RefusedError, match="the driver does not load it"):
        executor.register(TEXT, BINDING)
    driver.cuModuleUnload.assert_not_called()
    assert not executor.catalog.operators
    executor.synchronize()
    driver.cuModuleLoadData.return_value = (0, "module")
    assert executor.register(TEXT, BINDING).widths == (8, 8, 4)


@pytest.mark.parametrize(
    ("limit", "value", "reason"),
    [
        ("threads", 128, "a block of 256 threads, and the loaded function runs at most 128"),
        ("x", 128, r"the block \(256, 1, 1\) exceeds the device's \(128, 1024, 64\)"),
        ("static", 49153, "49153 bytes of static and up to 0 of dynamic shared memory"),
    ],
)
def test_a_launch_configuration_the_function_cannot_take_is_refused_and_unloaded(
    loading: Loading, limit: str, value: int, reason: str
) -> None:
    executor, driver, limits = loading
    limits[limit] = value
    with pytest.raises(RefusedError, match=reason):
        executor.register(TEXT, BINDING)
    driver.cuModuleUnload.assert_called_once_with("module")
    assert not executor.catalog.operators
    executor.synchronize()


@pytest.mark.parametrize(
    ("sizes", "parameter", "reason"),
    [
        ((4, 8, 4), None, "parameter k_param_0 of 4 bytes cannot hold Base"),
        ((8, 8), None, "the loaded function has 2 parameters, the binding 3"),
        ((8, 8, 16), None, "parameter k_param_2 of 16 bytes cannot hold Size"),
        ((8, 8, 4), Bounded(0, 2**32), "parameter k_param_2 of 4 bytes cannot hold Bounded"),
    ],
)
def test_parameter_sizes_that_cannot_hold_the_binding_are_refused(
    loading: Loading,
    monkeypatch: pytest.MonkeyPatch,
    sizes: tuple[int, ...],
    parameter: Bounded | None,
    reason: str,
) -> None:
    executor, driver, _ = loading
    monkeypatch.setattr(runtime, "parameter_sizes", lambda function: sizes)
    binding = BINDING
    if parameter is not None:
        values, out, _ = BINDING.parameters
        binding = replace(BINDING, parameters=(values, out, ("k_param_2", parameter)))
    with pytest.raises(RefusedError, match=reason):
        executor.register(TEXT, binding)
    driver.cuModuleUnload.assert_called_once_with("module")
    executor.synchronize()


def test_a_tenants_buffers_lie_in_its_reservation(
    fake_executor: tuple[Executor, SimpleNamespace],
) -> None:
    executor, _ = fake_executor
    with executor.tenant(BOB):
        executor.reserve(1024)
    with executor.tenant(ALICE):
        with pytest.raises(RefusedError, match="holds no reservation"):
            executor.allocate(256)
        executor.reserve(2048)
        with pytest.raises(RefusedError, match="holds a reservation"):
            executor.reserve(256)
        first = executor.allocate(256)
        reservation = executor.catalog.reservation(ALICE)
        found = executor.catalog.buffer(first)
        assert reservation is not None and reservation.offset == 1024
        assert found is not None and found.offset == 1024 and first == BufferId(ALICE, 1)
        with pytest.raises(MemoryError):
            executor.allocate(2048)
        with pytest.raises(RefusedError, match="holds a live buffer"):
            executor.unreserve()
        executor.synchronize()
        executor.free(first)
        executor.unreserve()
        assert executor.catalog.reservation(ALICE) is None
        # the suballocator over the pool proposes the ended reservation's region again
        executor.reserve(2048)
        again = executor.allocate(256)
        found = executor.catalog.buffer(again)
        assert again == BufferId(ALICE, 2) and found is not None and found.offset == 1024


def test_nvrtc_destroys_a_program_that_does_not_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    destroy = Mock(wraps=nvrtc.nvrtcDestroyProgram)
    monkeypatch.setattr(nvrtc, "nvrtcDestroyProgram", destroy)
    with pytest.raises(DriverError, match="broken.cu does not compile"):
        compile_ptx("this is not CUDA C", "broken.cu", "compute_86")
    destroy.assert_called_once()
