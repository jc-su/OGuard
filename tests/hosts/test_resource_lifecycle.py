"""The allocation wrapper serves a tenant's engine allocations from the tenant's
arena, whose chunks are the catalog's only steps, returns an arena in one step
once it holds no allocation, and frees a buffer of its own at once."""

from unittest.mock import Mock

import pytest

rmm = pytest.importorskip("rmm")

from opgate.catalog import Step, TenantId, View, Write  # noqa: E402
from opgate.executor import Executor  # noqa: E402
from opgate.hosts.rapids import CatalogResource  # noqa: E402

ALICE, BOB = TenantId(1), TenantId(2)
CHUNK = 1 << 16


def counted(executor: Executor, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The executor's steps, counted as the catalog begins each."""
    begin = Mock(wraps=executor._begin)
    monkeypatch.setattr(executor, "_begin", begin)
    return begin


@pytest.mark.gpu
def test_engine_allocations_take_a_step_only_when_the_arena_grows(monkeypatch):
    with (
        Executor(0, 1 << 22) as executor,
        CatalogResource(executor, chunk_bytes=CHUNK) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 21)
        steps = counted(executor, monkeypatch)
        with executor.tenant(ALICE):
            small = [rmm.DeviceBuffer(size=n, mr=resource.resource) for n in (8, 300, 1000, 4096)]
            assert steps.call_count == 1  # the first chunk
            (chunk,) = resource.chunks(ALICE)
            found = executor.catalog.buffer(chunk)
            assert found is not None and found.owner == ALICE and found.nbytes == CHUNK
            for allocation in small:
                view = resource.argument_at(allocation.ptr)
                assert isinstance(view, View) and view.buffer == chunk
                assert view.offset % 256 == 0
                assert allocation.ptr == executor.pool_address + found.offset + view.offset
            del small, allocation
            # freed bytes serve any later allocation of the tenant
            again = [rmm.DeviceBuffer(size=n, mr=resource.resource) for n in (2000, 3000)]
            assert steps.call_count == 1
            large = rmm.DeviceBuffer(size=2 * CHUNK, mr=resource.resource)
            assert steps.call_count == 2 and len(resource.chunks(ALICE)) == 2
            with pytest.raises(ValueError, match="no buffer of its own"):
                resource.buffer_at(large.ptr)
            del again, large
        assert resource.release(ALICE) == 2
        assert steps.call_count == 3
        assert not executor.catalog.buffers
        assert all(region.owner == ALICE for region in executor.catalog.freed)


@pytest.mark.gpu
def test_a_released_arena_returns_when_its_last_allocation_is_freed(monkeypatch):
    with (
        Executor(0, 1 << 22) as executor,
        CatalogResource(executor, chunk_bytes=CHUNK) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 21)
            first = rmm.DeviceBuffer(size=64, mr=resource.resource)
            second = rmm.DeviceBuffer(size=2 * CHUNK, mr=resource.resource)
        assert resource.release(ALICE) == 0
        assert len(resource.chunks(ALICE)) == 2
        steps = counted(executor, monkeypatch)
        # an allocation is freed in another tenant's session, as RMM may
        with executor.tenant(BOB):
            del first
            assert steps.call_count == 0
            del second
        assert steps.call_count == 1
        assert resource.chunks(ALICE) == () and not executor.catalog.buffers
        assert all(region.owner == ALICE for region in executor.catalog.freed)


@pytest.mark.gpu
def test_a_chunk_reaches_another_tenant_cleared():
    with (
        Executor(0, 1 << 20) as executor,
        CatalogResource(executor, chunk_bytes=CHUNK) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 19)
            secret = rmm.DeviceBuffer(size=4096, mr=resource.resource)
            secret.copy_from_host(b"\x07" * 4096)
            del secret
        resource.release(ALICE)
        with executor.tenant(ALICE):
            executor.unreserve()
        with executor.tenant(BOB):
            # BOB's reservation is the region that ALICE's held
            executor.reserve(1 << 19)
            theirs = rmm.DeviceBuffer(size=CHUNK, mr=resource.resource)
            assert theirs.tobytes() == bytes(CHUNK)
            del theirs
        resource.release(BOB)


@pytest.mark.gpu
def test_a_buffer_of_its_own_is_freed_in_one_step(monkeypatch):
    with (
        Executor(0, 1 << 20) as executor,
        CatalogResource(executor, chunk_bytes=CHUNK) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 19)
        steps = counted(executor, monkeypatch)
        with executor.tenant(ALICE):
            buffer = resource.allocate(256)
            with resource.dedicated():
                engine = rmm.DeviceBuffer(size=512, mr=resource.resource)
            own = resource.buffer_at(engine.ptr)
            assert resource.argument_at(engine.ptr) == own
            assert resource.chunks(ALICE) == () and steps.call_count == 2
            executor.execute(Step(ALICE, (Write(own, bytes(512), None),)))
            del engine
            assert executor.catalog.buffer(own) is None
            resource.free(buffer)
            assert executor.catalog.buffer(buffer) is None
        assert steps.call_count == 5 and not executor.catalog.buffers


@pytest.mark.gpu
@pytest.mark.parametrize("limit", ["bytes", "buffers"])
def test_chunks_and_buffers_share_the_tenant_budget(limit):
    """The tenant's reservation bounds its bytes, and the resource its buffers."""
    with (
        Executor(0, 1 << 22) as executor,
        CatalogResource(
            executor, tenant_buffers=2 if limit == "buffers" else None, chunk_bytes=CHUNK
        ) as resource,
    ):
        refusal = "no free region" if limit == "bytes" else "tenant"
        with executor.tenant(ALICE):
            executor.reserve(2 * CHUNK if limit == "bytes" else 1 << 21)
            engine = rmm.DeviceBuffer(size=CHUNK, mr=resource.resource)  # fills the first chunk
            buffer = resource.allocate(CHUNK)
            with pytest.raises(MemoryError, match=refusal):
                rmm.DeviceBuffer(size=256, mr=resource.resource)
            with pytest.raises(MemoryError, match=refusal):
                resource.allocate(256)
            resource.free(buffer)
            more = rmm.DeviceBuffer(size=256, mr=resource.resource)  # the arena grows
            del engine, more
        with executor.tenant(BOB):
            executor.reserve(CHUNK)
            theirs = resource.allocate(CHUNK)
            resource.free(theirs)
        assert resource.release(ALICE) == 2
        assert not executor.catalog.buffers


@pytest.mark.gpu
def test_a_failed_return_stops_the_executor(monkeypatch):
    with (
        Executor(0, 1 << 20) as executor,
        CatalogResource(executor, chunk_bytes=CHUNK) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 19)
            allocation = rmm.DeviceBuffer(size=64, mr=resource.resource)
            del allocation
        before = executor.catalog

        def fail(offset: int) -> None:
            raise RuntimeError("simulated allocator failure")

        # the free's step fails as its device work releases the chunk's region
        monkeypatch.setattr(executor._suballocators[ALICE], "release", fail)
        with pytest.raises(RuntimeError, match="allocator failure"):
            resource.release(ALICE)
        assert executor._catalog == before
        with pytest.raises(RuntimeError, match="failed and cannot be reused"):
            executor.synchronize()


@pytest.mark.gpu
def test_the_session_tenants_arena_is_rmms_current_resource():
    """Inside `current`, an allocation without a resource goes to the session
    tenant's arena without the resource's dispatch, and outside a session the
    dispatch refuses it."""
    with (
        Executor(0, 1 << 22) as executor,
        CatalogResource(executor, chunk_bytes=CHUNK) as resource,
        resource.current(),
    ):
        assert rmm.mr.get_current_device_resource() is resource.resource
        with pytest.raises(RuntimeError, match="outside a tenant's session"):
            rmm.DeviceBuffer(size=64)
        for tenant in (BOB, ALICE):
            with executor.tenant(tenant):
                executor.reserve(1 << 21)
        with executor.tenant(ALICE):
            first = rmm.DeviceBuffer(size=64)  # opens the arena through the dispatch
            assert rmm.mr.get_current_device_resource() is not resource.resource
            dispatched = dict(resource._engine)
            second = rmm.DeviceBuffer(size=64)
            assert resource._engine == dispatched
            view = resource.argument_at(second.ptr)
            assert isinstance(view, View) and view.buffer in resource.chunks(ALICE)
            with resource.dedicated():
                own = rmm.DeviceBuffer(size=64)
            assert resource.argument_at(own.ptr) == resource.buffer_at(own.ptr)
        assert rmm.mr.get_current_device_resource() is resource.resource
        with executor.tenant(BOB):
            theirs = rmm.DeviceBuffer(size=64)
            assert resource.chunks(BOB) and theirs.ptr not in dispatched
        with executor.tenant(ALICE):
            assert resource.release(ALICE) == 0  # the arena still holds allocations
            # a released arena serves no allocation
            assert rmm.mr.get_current_device_resource() is resource.resource
            del first, second, own
        assert resource.release(ALICE) == 1
        del theirs
        assert resource.release(BOB) == 1
        assert not executor.catalog.buffers
