"""Every RMM allocation of a cuDF query goes through the catalog.

With `CatalogResource` as RMM's current resource, a cuDF query of one
tenant allocates only inside chunks that the catalog records as that
tenant's buffers, in the tenant's reservation, and the release of the
tenant's arena frees them back to the catalog. A region that one tenant
frees reaches another tenant, through the end of the first tenant's
reservation, cleared. An allocation outside a tenant's session fails.

A `BufferView` shows a buffer only to a tenant that may read it. It is
read-only unless the tenant may write the buffer, and a writable view makes
the buffer raw, which it stays while the view lives. The view holds the
allocation, so the buffer stays live until the engine drops the view.
"""

import gc
import struct

import numpy as np
import pytest

cudf = pytest.importorskip("cudf")
rmm = pytest.importorskip("rmm")

from opgate.binding import ElementType  # noqa: E402
from opgate.catalog import Checked, Grant, Raw, Step, Write  # noqa: E402
from opgate.executor import Executor, RefusedError, TenantId  # noqa: E402
from opgate.frontends import Target  # noqa: E402
from opgate.hosts.cudf import install_cudf  # noqa: E402
from opgate.hosts.rapids import BufferView, CatalogResource  # noqa: E402

ALICE, BOB = TenantId(1), TenantId(2)
INDEX = ElementType(8, True)


@pytest.mark.gpu
def test_rmm_allocations_enforce_the_budget_until_explicit_release() -> None:
    with (
        Executor(0, 1 << 24) as executor,
        CatalogResource(executor, tenant_buffers=1, chunk_bytes=256) as resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(256)
            allocation = rmm.DeviceBuffer(size=256, mr=resource.resource)
            with pytest.raises(MemoryError, match="tenant"):
                rmm.DeviceBuffer(size=128, mr=resource.resource)
            # the arena's freed bytes serve any size without more budget
            del allocation
            allocation = rmm.DeviceBuffer(size=128, mr=resource.resource)
            with pytest.raises(MemoryError, match="tenant"):
                resource.allocate(128)
            del allocation
        assert resource.release(ALICE) == 1
        with executor.tenant(ALICE):
            buffer = resource.allocate(128)
            resource.free(buffer)
        assert not executor.catalog.buffers


@pytest.mark.gpu
def test_a_cudf_query_allocates_only_through_the_catalog() -> None:
    with (
        Executor(0, 1 << 26) as executor,
        CatalogResource(executor, chunk_bytes=1 << 22) as catalog_resource,
    ):
        with executor.tenant(ALICE):
            executor.reserve(1 << 25)
        with catalog_resource.current():
            with executor.tenant(ALICE), install_cudf(executor, catalog_resource, Target(8, 6)):
                values = cudf.Series(np.arange(100_000, dtype=np.int64))
                doubled = values * 2
                total = int(doubled.sum())
                # every allocation of the query lies in a chunk of the tenant
                assert {b.id for b in executor.catalog.buffers} == set(
                    catalog_resource.chunks(ALICE)
                )
                assert all(b.owner == ALICE for b in executor.catalog.buffers)
                secret = rmm.DeviceBuffer.to_device(b"\x07" * 4096)
                del values, doubled, secret
                gc.collect()
            assert catalog_resource.release(ALICE) >= 1
            assert total == 2 * sum(range(100_000))
            assert not executor.catalog.buffers
            assert executor.catalog.freed and all(r.owner == ALICE for r in executor.catalog.freed)
            with executor.tenant(ALICE):
                executor.unreserve()
            assert not executor.catalog.freed
            with executor.tenant(BOB):
                # BOB's reservation is the region that ALICE's held
                executor.reserve(1 << 25)
                theirs = rmm.DeviceBuffer(size=4096)
                assert theirs.tobytes() == bytes(4096)
                del theirs
            with pytest.raises(RuntimeError, match="an operation outside a tenant's session"):
                rmm.DeviceBuffer(size=256)


@pytest.mark.gpu
def test_a_buffer_view_follows_the_catalog_and_holds_its_allocation() -> None:
    rows = struct.pack("<4q", 0, 1, 2, 3)
    with (
        Executor(0, 1 << 24) as executor,
        CatalogResource(executor, chunk_bytes=1 << 20) as resource,
    ):
        with executor.tenant(BOB):
            executor.reserve(1 << 23)
        with executor.tenant(ALICE):
            executor.reserve(1 << 23)
            shared, own = resource.allocate(32), resource.allocate(32)
            writes = (Write(shared, rows, INDEX), Write(own, rows, INDEX), Grant(shared, BOB))
            executor.execute(Step(ALICE, writes))
        with executor.tenant(BOB):
            with pytest.raises(RefusedError, match="may not read"):
                BufferView.of(resource, own, "<i8", 4, access="read")
            # a grantee reads a buffer, and its view is read-only
            view = BufferView.of(resource, shared, "<i8", 4, access="read")
            assert view.readonly and view.__cuda_array_interface__["data"] == (
                view.allocation.ptr,
                True,
            )
            assert cudf.Series(view).to_numpy().tolist() == [0, 1, 2, 3]
        with executor.tenant(ALICE):
            # the owner may not write a granted buffer either
            with pytest.raises(RefusedError, match="may not write"):
                BufferView.of(resource, shared, "<i8", 4, access="write")
            # a read view of its own buffer keeps the buffer's checked range
            assert BufferView.of(resource, own, "<i8", 4, access="read").readonly
            found = executor.catalog.buffer(own)
            assert found is not None and found.status == Checked(INDEX, 0, 3)
            # a write view of it is writable, and the buffer becomes raw
            view = BufferView.of(resource, own, "<i8", 4, access="write")
            assert not view.readonly
            found = executor.catalog.buffer(own)
            assert found is not None and found.status == Raw()
            # and stays raw while the engine holds the view
            with pytest.raises(RefusedError, match="stays live and raw"):
                executor.execute(Step(ALICE, (Write(own, rows, INDEX),)))
            with pytest.raises(RefusedError, match="fewer than 5 elements"):
                BufferView.of(resource, own, "<i8", 5, access="read")
            engine = rmm.DeviceBuffer(size=64, mr=resource.resource)
            with pytest.raises(ValueError, match="no buffer of its own"):
                resource.buffer_at(engine.ptr)
            # the view holds the allocation after the buffer's free
            resource.free(own)
            assert executor.catalog.buffer(own) is not None
            del view, engine
            gc.collect()
            assert executor.catalog.buffer(own) is None
            assert resource.release(ALICE) == 1
            # a granted buffer leaves the catalog at its free
            resource.free(shared)
            gc.collect()
            assert executor.catalog.buffer(shared) is None
        assert not executor.catalog.buffers
