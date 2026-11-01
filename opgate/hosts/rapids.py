"""The allocation wrapper of the RAPIDS engines: one RMM memory resource.

cuDF, Sirius, the RAPIDS accelerator for Spark, and Velox's cuDF backend
allocate device memory through RMM's current memory resource.
`CatalogResource` is that resource. Each region is a buffer of the current
tenant that the executor allocates in the tenant's reservation
(`Executor.allocate`), so the regions of a tenant lie where its own
allocations and frees put them; the catalog clears a region unless the
tenant last owned it (design section 6.2). An allocation outside a tenant's
session, or of a tenant without a reservation, fails. A zero-byte allocation
is no region, and gets the null pointer.

The engine's allocations of a tenant come from the tenant's arena: an RMM
pool resource whose upstream regions, its chunks, are buffers of the tenant.
The engine allocates and frees inside the chunks without a catalog step,
since only the tenant can reach a chunk; the catalog takes a step when the
arena grows by a chunk, and when `release` returns the chunks of an arena
that holds no allocation, all in one step. Inside `current`, RMM's current
resource is the session tenant's arena, so that an allocation takes no
Python call, and outside a session it is the resource's own dispatch, which
refuses; the executor tells the resource when a session begins and ends
(`Executor.observe_sessions`). A launch takes an engine allocation as a view
of its chunk (`argument_at`). A buffer that a session names, grants, or
passes with an index range is a buffer of its own: `allocate` gives one,
and so does every allocation of the engine inside `dedicated`; its free is
a catalog step. Chunks and buffers both count towards the tenant's buffer
limit, and lie in its reservation. Allocation fails immediately when the
limit or the reservation is exhausted.

Trusted host code runs the engine's operators on the pool: `BufferView`
shows a buffer that `allocate` gave to the engine through the CUDA array
interface. An output of a trusted operator is raw, like any allocation: an
untrusted consumer that takes it as an index input is a guarded launch. A
view requires that the session's tenant may read the buffer, and holds the
buffer's allocation, so the region stays allocated while the engine holds
the view. The view is read-only unless the tenant may write the buffer; a
writable view first makes the buffer raw, since the engine may write any
value through it, and the buffer stays raw while the view lives: a step that
would record a checked range for it, as a fill or a write with an element
type does, is refused until the engine drops the view (`Executor.keep_raw`).

Imports RMM, `opgate.catalog`, and `opgate.executor`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Self, assert_never

import rmm
from rmm.pylibrmm.stream import Stream

from opgate.catalog import View
from opgate.executor import BufferId, Executor, RefusedError, TenantId

__all__ = ["CHUNK", "BufferView", "CatalogResource"]

CHUNK = 64 << 20
"""The first chunk of an arena; RMM's pool grows by the larger of the
allocation and the arena's size, so the chunks of a tenant are few."""


@dataclass
class _Arena:
    """An arena of `tenant`: its pool, the statistics adaptor that counts the
    allocations in it, which the engine allocates through, and its chunks by
    start, each the buffer and its bytes. A released arena serves no new
    allocation and returns its chunks once it holds no allocation."""

    tenant: TenantId
    pool: rmm.mr.PoolMemoryResource
    counted: rmm.mr.StatisticsResourceAdaptor
    chunks: dict[int, tuple[BufferId, int]]
    released: bool = False

    def live(self) -> int:
        return self.counted.allocation_counts.current_count


class CatalogResource:
    """An RMM memory resource whose every region the catalog checks."""

    def __init__(
        self,
        executor: Executor,
        *,
        tenant_buffers: int | None = None,
        chunk_bytes: int = CHUNK,
    ) -> None:
        self._executor = executor
        if tenant_buffers is not None and tenant_buffers < 1:
            raise ValueError("tenant_buffers must be positive")
        if chunk_bytes < 1 or chunk_bytes % 256 or chunk_bytes > executor.catalog.pool:
            raise ValueError("chunk_bytes must be a positive multiple of 256 within the pool")
        self._tenant_buffers = tenant_buffers
        self._chunk_bytes = chunk_bytes
        self._usage: dict[TenantId, int] = {}  # the number of each tenant's buffers and chunks
        self._arenas: list[_Arena] = []
        # each allocation in an arena through the dispatch: its arena, for its free
        self._engine: dict[int, _Arena] = {}
        # each buffer of its own, by start: the buffer, its owner, its bytes; the
        # owner survives its session and is authoritative for the free
        self._buffers: dict[int, tuple[BufferId, TenantId, int]] = {}
        self._owned: dict[BufferId, rmm.DeviceBuffer] = {}  # what `allocate` gave
        self._returned: list[int] | None = None  # the start of each chunk of an arena at its end
        self._dedicated = 0
        self._current = False
        self._session: TenantId | None = None  # the session's tenant, inside `current`
        self._closed = False
        self.resource = rmm.mr.CallbackMemoryResource(self._allocate, self._deallocate)

    def close(self) -> None:
        """End the resource with its executor, before the executor frees the pool.

        Engine objects can hold allocations past their owner thread, after a
        failure too, and RMM returns them later from any thread. After `close`
        the pool and its catalog are gone, so such a return has nothing to give
        back and changes nothing; a new allocation is refused. The arenas' pools
        end here, while their callbacks are intact: RMM calls them as a pool
        ends, and the garbage collector, which may clear a callback first, would
        otherwise end them with the cycle the resource is part of."""
        self._closed = True
        self._owned.clear()
        self._engine.clear()
        self._arenas.clear()
        self._buffers.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def current(self) -> Iterator[None]:
        """Make the resource RMM's current resource for the block: the session
        tenant's arena inside a session, and the dispatch outside."""
        previous = rmm.mr.get_current_device_resource()
        self._executor.observe_sessions(self._observe)
        self._current = True
        try:
            self._observe(self._executor.session())
            yield
        finally:
            self._current = False
            self._executor.observe_sessions(None)
            rmm.mr.set_current_device_resource(previous)

    @contextmanager
    def dedicated(self) -> Iterator[None]:
        """Every allocation of the engine inside is a buffer of its own, which a
        launch can take with an index range."""
        self._dedicated += 1
        try:
            self._apply()
            yield
        finally:
            self._dedicated -= 1
            self._apply()

    def allocate(self, nbytes: int) -> BufferId:
        """A buffer of its own of `nbytes` for the current tenant, allocated through
        RMM, as the engine allocates; it lives until `free`."""
        if nbytes < 1:
            raise ValueError("a buffer has at least one byte")
        with self.dedicated():
            allocation = rmm.DeviceBuffer(size=nbytes, mr=self.resource)
        buffer = self._buffers[allocation.ptr][0]
        self._owned[buffer] = allocation
        return buffer

    def free(self, buffer: BufferId) -> None:
        """Free a buffer that `allocate` gave."""
        found = self._executor.catalog.buffer(buffer)
        if found is None or found.owner != self._executor.session_tenant:
            raise ValueError("the session does not own this allocation")
        del self._owned[buffer]

    def buffer_at(self, pointer: int) -> BufferId:
        """The buffer of its own that starts at `pointer`."""
        found = self._buffers.get(pointer)
        if found is None:
            raise ValueError(f"no buffer of its own starts at {pointer:#x}")
        return found[0]

    def argument_at(self, pointer: int) -> BufferId | View:
        """The launch argument of the allocation that starts at `pointer`: a buffer
        of its own, or the view of the arena's chunk that holds it from there."""
        found = self._buffers.get(pointer)
        if found is not None:
            return found[0]
        for arena in self._arenas:
            for start, (buffer, nbytes) in arena.chunks.items():
                if start <= pointer < start + nbytes:
                    return View(buffer, pointer - start)
        raise ValueError(f"no allocation of the resource starts at {pointer:#x}")

    def live(self) -> tuple[frozenset[int], int]:
        """The start of every buffer of its own, and the number of allocations in
        the arenas."""
        return frozenset(self._buffers), sum(arena.live() for arena in self._arenas)

    def chunks(self, tenant: TenantId) -> tuple[BufferId, ...]:
        """The chunks of `tenant`'s arenas."""
        return tuple(
            buffer
            for arena in self._arenas
            if arena.tenant == tenant
            for buffer, _ in arena.chunks.values()
        )

    def every_chunk(self) -> frozenset[BufferId]:
        """The chunks of every tenant's arenas."""
        return frozenset(buffer for arena in self._arenas for buffer, _ in arena.chunks.values())

    def release_every(self) -> None:
        """Release every tenant's arena."""
        for tenant in {arena.tenant for arena in self._arenas}:
            self.release(tenant)

    def release(self, tenant: TenantId) -> int:
        """Release `tenant`'s arena: it serves no new allocation, and its chunks
        return to the pool in one step once it holds no allocation, now or at a
        later release or free. The number of chunks returned now, of any
        released arena of the tenant."""
        for arena in self._arenas:
            if arena.tenant == tenant and not arena.released:
                arena.released = True
        self._apply()
        return sum(self._return(arena) for arena in self._drained() if arena.tenant == tenant)

    def _drained(self) -> list[_Arena]:
        return [arena for arena in self._arenas if arena.released and not arena.live()]

    def _arena(self, tenant: TenantId) -> _Arena | None:
        """The arena that serves `tenant`'s allocations."""
        for arena in self._arenas:
            if arena.tenant == tenant and not arena.released:
                return arena
        return None

    def _observe(self, tenant: TenantId | None) -> None:
        """A session of `tenant` begins or ends (`Executor.observe_sessions`)."""
        self._session = tenant
        self._apply()

    def _apply(self) -> None:
        """Make RMM's current resource the arena that serves the session's tenant,
        or the dispatch outside a session, inside `dedicated`, and for a tenant
        with no arena."""
        if not self._current:
            return
        tenant = self._session
        arena = None if tenant is None or self._dedicated else self._arena(tenant)
        rmm.mr.set_current_device_resource(self.resource if arena is None else arena.counted)

    def _region(self, nbytes: int) -> tuple[int, BufferId]:
        """A buffer of `nbytes` in the current tenant's reservation, and its start."""
        tenant = self._executor.session_tenant
        count = self._usage.get(tenant, 0)
        if self._tenant_buffers is not None and count >= self._tenant_buffers:
            raise MemoryError("the tenant's buffers and chunks reach its buffer limit")
        buffer = self._executor.allocate(nbytes)
        found = self._executor.catalog.buffer(buffer)
        if found is None:
            raise RuntimeError("the executor's allocation is not a live buffer")
        self._usage[tenant] = count + 1
        return self._executor.pool_address + found.offset, buffer

    def _unregister(self, owner: TenantId, buffers: tuple[BufferId, ...]) -> None:
        """`owner`'s buffers leave the catalog in one step."""
        with self._executor.tenant(owner):
            self._executor.free(*buffers)
        self._usage[owner] -= len(buffers)

    def _open(self, tenant: TenantId) -> _Arena:
        """A new arena of the tenant, with its first chunk."""
        chunks: dict[int, tuple[BufferId, int]] = {}

        def grow(nbytes: int, stream: Stream) -> int:
            if self._executor.session_tenant != tenant:
                raise RuntimeError("an arena grows only in its tenant's session")
            pointer, buffer = self._region(nbytes)
            chunks[pointer] = buffer, nbytes
            return pointer

        def shrink(pointer: int, nbytes: int, stream: Stream) -> None:
            if self._closed:
                return  # the pool is gone with the executor (`close`)
            if self._returned is None:
                raise RuntimeError("an arena ended outside its return")
            self._returned.append(pointer)

        upstream = rmm.mr.CallbackMemoryResource(grow, shrink)
        pool = rmm.mr.PoolMemoryResource(upstream, initial_pool_size=self._chunk_bytes)
        arena = _Arena(tenant, pool, rmm.mr.StatisticsResourceAdaptor(pool), chunks)
        self._arenas.append(arena)
        return arena

    def _return(self, arena: _Arena) -> int:
        """End a released arena that holds no allocation: RMM's pool gives back each
        chunk, and the chunks leave the catalog in one step."""
        self._arenas.remove(arena)
        tenant, chunks = arena.tenant, arena.chunks
        self._returned = []
        try:
            # the last references to the adaptor and its pool: the pool ends here
            del arena.counted, arena.pool
            returned = self._returned
        finally:
            self._returned = None
        if sorted(returned) != sorted(chunks):
            raise RuntimeError("an arena's pool did not give back exactly its chunks")
        self._unregister(tenant, tuple(chunks[p][0] for p in returned))
        return len(returned)

    def _allocate(self, nbytes: int, stream: Stream) -> int:
        """The dispatch: a buffer of its own inside `dedicated`, and otherwise an
        allocation in the session tenant's arena, which opens it if it has none
        and makes it RMM's current resource."""
        if self._closed:
            raise MemoryError("the resource is closed with its executor")
        tenant = self._executor.session_tenant
        if nbytes == 0:
            return 0
        if self._dedicated:
            pointer, buffer = self._region(nbytes)
            self._buffers[pointer] = (buffer, tenant, nbytes)
            return pointer
        for drained in self._drained():
            self._return(drained)
        arena = self._arena(tenant)
        if arena is None:
            arena = self._open(tenant)
            self._apply()
        pointer = arena.counted.allocate(nbytes, stream)
        self._engine[pointer] = arena
        return pointer

    def _deallocate(self, pointer: int, nbytes: int, stream: Stream) -> None:
        if self._closed:
            return  # the pool is gone with the executor (`close`)
        if nbytes == 0:
            return
        arena = self._engine.pop(pointer, None)
        if arena is None:
            buffer, owner, _ = self._buffers.pop(pointer)
            self._unregister(owner, (buffer,))
            return
        arena.counted.deallocate(pointer, nbytes, stream)
        if arena.released and not arena.live():
            self._return(arena)


@dataclass(frozen=True)
class BufferView:
    """The first `count` elements of a catalog buffer, of NumPy type string
    `typestr` such as `<i8`, as the CUDA array interface describes them. The
    view holds the buffer's RMM allocation, and the engine may write through
    it only if it is not `readonly`."""

    allocation: rmm.DeviceBuffer
    typestr: str
    count: int
    readonly: bool

    @staticmethod
    def of(
        resource: CatalogResource,
        buffer: BufferId,
        typestr: str,
        count: int,
        *,
        access: Literal["read", "write"],
    ) -> "BufferView":
        """A view of a buffer that `resource.allocate` gave. A read view needs the
        session's tenant to be allowed to read the buffer and leaves the buffer's
        status as it is; a write view needs it to be allowed to write the buffer,
        which becomes raw before the engine sees it and stays raw while the view
        lives: the executor refuses every step that would record a checked range
        for it (`Executor.keep_raw`)."""
        executor = resource._executor
        tenant = executor.session_tenant
        allocation = resource._owned.get(buffer)
        if allocation is None:
            raise ValueError(f"buffer {buffer} is not an allocation that the resource gave")
        if count < 0:
            raise RefusedError("a buffer view must have a nonnegative element count")
        if count * int(typestr[2:]) > allocation.size:
            raise RefusedError(f"buffer {buffer} holds fewer than {count} elements of {typestr}")
        match access:
            case "read":
                if not executor.catalog.readable(tenant, buffer):
                    raise RefusedError("the tenant may not read the buffer")
                return BufferView(allocation, typestr, count, True)
            case "write":
                view = BufferView(allocation, typestr, count, False)
                # the buffer is raw from here, and stays raw while the view lives
                executor.keep_raw(buffer, view)
                return view
            case _ as never:
                assert_never(never)

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        return {
            "shape": (self.count,),
            "strides": None,
            "typestr": self.typestr,
            "data": (self.allocation.ptr, self.readonly),
            "version": 3,
        }
